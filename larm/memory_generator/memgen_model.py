import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    PreTrainedTokenizerBase,
    GenerationConfig,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
    
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import PeftConfig, LoraConfig

import random
from typing import Tuple, Optional, List, Union
import logging
import functools

from larm.task.base_model import BaseModel
from larm.common.registry import registry

from .weaver import MemGenWeaver
from .trigger import MemGenTrigger, NanoTrigger
from .utils import (
    CONVERSATION_TEMPLATE, 
    load_state_dict_from_safetensor, 
    fix_model_parameters,
    log_trainable_params,
)

# Decorator to log function calls in blue
def log_function_call(func):
    """Decorator to log function calls with blue color."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        # logging.info(f"\033[94m[CALL] {func_name}\033[0m") # blue color
        return func(*args, **kwargs)
    return wrapper

# @log_function_call
def get_next_token(next_token_logits: torch.Tensor, do_sample: bool, temperature: float) -> torch.Tensor:
    """
    Selects the next token from model logits.

    Two modes are supported:
    1. Sampling mode (do_sample=True and temperature>0):  
       Apply temperature scaling to the logits, compute a probability distribution with softmax, 
       and randomly sample one token.
    2. Greedy mode (do_sample=False or temperature==0):  
       Select the token with the highest probability (argmax).

    Args:
        next_token_logits (torch.Tensor): 
            Logits for the next tokens, shape [batch_size, vocab_size].
        do_sample (bool): 
            Whether to perform stochastic sampling. If False, greedy decoding is used.
        temperature (float): 
            Sampling temperature. Higher values make the distribution flatter (more randomness), 
            while lower values make it sharper (more deterministic). 
            When set to 0, greedy decoding is enforced.

    Returns:
        torch.Tensor: 
            The selected next token indices, shape [batch_size, 1].
    """
    if len(next_token_logits.shape) != 2:
        raise ValueError("Input logits must be a 2D tensor [batch_size, vocab_size]")
    if do_sample and temperature != 0:
        # Apply temperature scaling and sample from the resulting probability distribution
        probs = F.softmax(next_token_logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    else:
        # Greedy decoding: pick the token with the highest probability
        return torch.argmax(next_token_logits, dim=-1, keepdim=True)


# @log_function_call
def generate_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Generate position ID tensor based on the given attention mask.

    The position IDs are computed as the cumulative count of non-padding tokens 
    within each sequence. Padding tokens are always assigned position ID 0.

    Args:
        attention_mask (torch.Tensor): 
            A tensor of shape (batch_size, sequence_length). 
            Typically, 1 indicates a valid (non-padding) token and 0 indicates a padding token.

    Returns:
        torch.Tensor: 
            A tensor of shape (batch_size, sequence_length) containing position IDs.
            - For non-padding tokens: position IDs start at 0 and increase consecutively.
            - For padding tokens: position ID is always 0.
    """
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    position_ids.masked_fill_(attention_mask == 0, 0)
    return position_ids

@log_function_call
def is_conversation(input_ids: torch.Tensor, tokenizer) -> bool:
    """
    Check whether the given input IDs represent a conversation format.  
    Only the first sample in the batch is inspected.

    The function verifies whether the sequence contains at least one pair 
    of special tokens: <|im_start|> and <|im_end|>.

    Args:
        input_ids (torch.Tensor): 
            Tensor of shape (batch_size, seq_len) containing token IDs.
        tokenizer: 
            A HuggingFace tokenizer used to obtain the special token IDs.

    Returns:
        bool: 
            True if the sequence contains both <|im_start|> and <|im_end|>, 
            False otherwise.
    """
    if len(input_ids.shape) != 2:
        raise ValueError("input_ids must be a 2D tensor of shape (batch_size, seq_len)")
    
    seq = input_ids[0].tolist()

    # Encode the special tokens to obtain their ID sequences
    im_start_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    im_end_ids   = tokenizer.encode("<|im_end|>",   add_special_tokens=False)

    # Check if the sequence contains at least one <|im_start|> and one <|im_end|>
    has_start = any(seq[i:i+len(im_start_ids)] == im_start_ids for i in range(len(seq) - len(im_start_ids) + 1))
    has_end   = any(seq[i:i+len(im_end_ids)]   == im_end_ids   for i in range(len(seq) - len(im_end_ids) + 1))
    
    logging.info(f"has_start: {has_start}, has_end: {has_end}")
    # logging.info(f"seq: {seq}")
    logging.info(f"im_start_ids: {im_start_ids}")
    logging.info(f"im_end_ids: {im_end_ids}")

    return has_start and has_end


@log_function_call
def postprocess_assistant_labels(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer
) -> torch.Tensor:
    """
    Mask out labels corresponding to the `<|im_start|>assistant` marker.  

    This ensures that the special tokens used to indicate the start of the assistant's
    response do not contribute to the loss during training.

    Args:
        input_ids (torch.Tensor): 
            Tensor of shape (batch_size, seq_len) containing the conversation token IDs.
        labels (torch.Tensor): 
            Tensor of shape (batch_size, seq_len) containing training labels. 
            A value of -100 indicates positions that should be ignored in loss computation.
        tokenizer: 
            A HuggingFace tokenizer used to encode the `<|im_start|>assistant\n` marker.

    Returns:
        torch.Tensor: 
            The modified labels tensor with positions corresponding to 
            `<|im_start|>assistant\n` masked as -100.
    """
    # if tokenizer.chat_template != CONVERSATION_TEMPLATE:
    #     raise ValueError(
    #         "Invalid tokenizer.chat_template detected.\n"
    #         f"Expected:\n{CONVERSATION_TEMPLATE}\n\n"
    #         f"Got:\n{tokenizer.chat_template}\n\n"
    #         "Please ensure that you are using the correct conversation template."
    #     )
    
    # Encode the token sequence for "<|im_start|>assistant\n"
    pattern_ids: List[int] = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    batch_size, seq_len = input_ids.shape
    new_labels = labels.clone()

    for b in range(batch_size):
        seq = input_ids[b].tolist()
        for i in range(len(seq) - len(pattern_ids) + 1):
            # Mask positions matching the pattern
            if seq[i : i + len(pattern_ids)] == pattern_ids:
                new_labels[b, i : i + len(pattern_ids)] = -100

    return new_labels

# @log_function_call
def check_ends_with_delimiter(
    input_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase, delimiters: List[str]
) -> torch.Tensor:
    """
    Check whether each sequence in the batch ends with any of the specified delimiter strings.

    Args:
        input_ids (torch.Tensor): 
            Tensor of shape (batch_size, seq_len) containing token IDs for each sequence.
        tokenizer (PreTrainedTokenizerBase): 
            HuggingFace tokenizer used to decode input_ids back to text.
        delimiters (List[str]): 
            A list of delimiter strings to check against. 
            If a sequence ends with any of these delimiters, it is marked as True.

    Returns:
        torch.Tensor: 
            A boolean tensor of shape (batch_size, 1), where each entry indicates 
            whether the corresponding sequence ends with one of the delimiters.
    """
    batch_size = input_ids.size(0)

    # Initialize result tensor: False by default
    augmentation_decisions = torch.zeros(batch_size, 1, dtype=torch.bool, device=input_ids.device)

    # Fast path: compare suffix in token ID space if delimiter id tensors precomputed
    delimiter_id_tensors = getattr(getattr(tokenizer, "memgen_model_ref", None), "_delimiter_id_tensors", None)
    if delimiter_id_tensors is None:
        try:
            delimiter_id_tensors = [
                torch.tensor(tokenizer.encode(d, add_special_tokens=False), dtype=torch.long, device=input_ids.device)
                for d in delimiters
            ]
        except Exception:
            delimiter_id_tensors = []

    for i in range(batch_size):
        seq = input_ids[i]
        matched = False
        for did in delimiter_id_tensors:
            L = did.numel()
            if L == 0 or L > seq.numel():
                continue
            if torch.equal(seq[-L:], did):
                matched = True
                break
        if not matched:
            # Fallback to decode once when necessary
            decoded = tokenizer.decode(seq.tolist())
            for aug_str in delimiters:
                if decoded.endswith(aug_str):
                    matched = True
                    break
        augmentation_decisions[i] = matched
    
    return augmentation_decisions


@registry.register_model("latmem")
class LatentMemoryModel(BaseModel):

    @log_function_call
    def __init__(
        self, 
        reasoner_model_name: str, 
        weaver_model_name: str,
        prompt_latents_len: int,
        inference_latents_len: int,
        weaver_peft_config: Optional[PeftConfig] = None,
        trigger_model_name: str = None,   
        trigger_peft_config: Optional[PeftConfig] = None,
        max_prompt_aug_num: int = 1,     
        max_inference_aug_num: int = 5,
    ):   
        super().__init__()

        # # build reasoner LLM
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     reasoner_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        # )
        
        # todo: add vlm: qwen2.5-vl
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            reasoner_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        
        self.processor = AutoProcessor.from_pretrained(reasoner_model_name)
        self.tokenizer = self.processor.tokenizer
        self.config = self.model.config
        
        # build weaver LLM
        self.weaver = MemGenWeaver(
            weaver_model_name, prompt_latents_len, inference_latents_len, weaver_peft_config
        )
        
        # build trigger LLM
        self.trigger = NanoTrigger()   # always return true
        if trigger_model_name is not None:
            self.trigger = MemGenTrigger(
                trigger_model_name, trigger_peft_config
            )
            logging.info(f"Use Trigger: {trigger_model_name}")
        
        # projection layers for mapping embeddings between reasoner and weaver
        # map reasoner input embeddings to weaver input embeddings
        self.reasoner_to_weaver = nn.Linear(
            self.model.config.hidden_size, self.weaver.config.hidden_size, dtype=torch.bfloat16
        ) # NOTE: 2048 -> 1536
        # Map weaver hidden states to reasoner input embeddings
        self.weaver_to_reasoner = nn.Linear(
            self.weaver.config.hidden_size, self.model.config.hidden_size, dtype=torch.bfloat16
        ) # NOTE: 1536 -> 2048
        
        # todo; delimiters
        self.delimiters: List[str] = [",", ".", "\n"]  # delimiters for detecting augmentation points
        # self.delimiters: List[str] = ["."]  # delimiters for detecting augmentation points
        self.max_prompt_aug_num = max_prompt_aug_num  # insert latents after input prompt
        self.max_inference_aug_num = max_inference_aug_num  # insert latents after specified delimiters

        # Precompute delimiter token ID sequences for fast suffix checks (avoid repeated decode)
        try:
            self._delimiter_id_tensors = [
                torch.tensor(self.tokenizer.encode(d, add_special_tokens=False), dtype=torch.long)
                for d in self.delimiters
            ]
        except Exception:
            # Fallback: empty list; code path will behave as if no delimiter found
            self._delimiter_id_tensors = []

        # postprocess
        self._postprocess_models()
        
        self.warnings_issued = {}
        self.model_tags = None
        log_trainable_params(self)

    @log_function_call
    def add_model_tags(self, tags: Union[list[str], str]) -> None:
        r"""
        Add custom tags into the model that gets pushed to the Hugging Face Hub. Will
        not overwrite existing tags in the model.

        Args:
            tags (`Union[list[str], str]`):
                The desired tags to inject in the model

        Examples:

        ```python
        from transformers import AutoModel

        model = AutoModel.from_pretrained("google-bert/bert-base-cased")

        model.add_model_tags(["custom", "custom-bert"])

        # Push the model to your namespace with the name "my-custom-bert".
        model.push_to_hub("my-custom-bert")
        ```
        """
        if isinstance(tags, str):
            tags = [tags]

        if self.model_tags is None:
            self.model_tags = []

        for tag in tags:
            if tag not in self.model_tags:
                self.model_tags.append(tag)
    
    @log_function_call
    def _postprocess_models(self):
        """
        Postprocess the components of the latent memory model: reasoner, weaver, trigger, and tokenizer.

        Steps:
            1. Freeze all parameters of the reasoner (no gradient updates).
            2. Cast all models to bfloat16 for memory and compute efficiency.
            3. Ensure the tokenizer has a valid pad token:
            - If pad token is missing, use the EOS token as the pad token.
            - Set `padding_side` to "left" for compatibility with generation tasks.
            4. Standardize the tokenizer's chat template to `CONVERSATION_TEMPLATE`.
        """
        # Freeze all parameters of the reasoner by default
        fix_model_parameters(self.model)

        # Convert all sub-models to bfloat16
        self.model = self.model.bfloat16()
        self.weaver = self.weaver.bfloat16()
        self.trigger = self.trigger.bfloat16()

        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"
            logging.info(
                f"Tokenizer has no pad token. Using EOS token ({self.tokenizer.eos_token}) as pad token."
            )
        
        # Synchronize model config with tokenizer's special tokens to avoid warnings
        # Update model config
        if hasattr(self.model, 'config'):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.config.bos_token_id = self.tokenizer.bos_token_id
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
        
        # Update generation config
        if hasattr(self.model, 'generation_config'):
            if self.model.generation_config is not None:
                self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
                self.model.generation_config.bos_token_id = self.tokenizer.bos_token_id
                self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        
        # Also update processor's tokenizer if it exists
        if hasattr(self, 'processor') and hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.pad_token_id = self.tokenizer.pad_token_id
            self.processor.tokenizer.bos_token_id = self.tokenizer.bos_token_id
            self.processor.tokenizer.eos_token_id = self.tokenizer.eos_token_id
            # IMPORTANT: Do NOT override VLM chat templates; they encode image placeholders.
            is_vlm = isinstance(self.model, Qwen2_5_VLForConditionalGeneration) or hasattr(self.processor, 'image_processor')
            if not is_vlm:
                self.processor.tokenizer.chat_template = CONVERSATION_TEMPLATE

        # Normalize the tokenizer's chat template (skip for VLMs)
        is_vlm = isinstance(self.model, Qwen2_5_VLForConditionalGeneration) or hasattr(getattr(self, 'processor', None), 'image_processor')
        if not is_vlm:
            self.tokenizer.chat_template = CONVERSATION_TEMPLATE
        
        logging.info(
            f"Synchronized special tokens - pad_token_id: {self.tokenizer.pad_token_id}, "
            f"bos_token_id: {self.tokenizer.bos_token_id}, eos_token_id: {self.tokenizer.eos_token_id}"
        )

    @property
    def device(self):
        assert self.model.device == self.weaver.device == self.trigger.device
        return self.model.device
    
    
    @log_function_call
    def _forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[Tuple[int, int, int]] = None,
        **kwargs
    ) -> torch.Tensor:
        # preprocess inputs
        assert input_ids.shape == attention_mask.shape == labels.shape
        
        tokenizer = self.tokenizer
        reasoner = self.model
        weaver = self.weaver
        delimiters = self.delimiters
        max_augment_num = self.max_inference_aug_num  # Limit the number of inference augmentation points to avoid excessive augmentation
        device = self.device
        embeds_dtype = reasoner.get_input_embeddings().weight.dtype
        B, _ = input_ids.shape
        hidden_size = reasoner.config.hidden_size

        # select augment idx
        augmentation_indices = self._select_augment_points_after_delimiter(
            input_ids, labels, delimiters, tokenizer, max_augment_num
        )
        logging.info(f"augmentation_indices: {augmentation_indices}")
        
        # origin inputs embeds (use fused embeddings when image inputs are provided)
        if pixel_values is not None:
            with torch.no_grad():
                seed_outputs = reasoner(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=True,
                )
            inputs_embeds = seed_outputs.hidden_states[0]
        else:
            inputs_embeds = reasoner.get_input_embeddings()(input_ids)
        
        # Initialize the start index and empty tensors for accumulating processed segments
        current_start_idx = 0
        current_inputs_embeds = torch.empty((B, 0, hidden_size), device=device, dtype=embeds_dtype)
        current_attention_mask = torch.empty((B, 0), device=device, dtype=attention_mask.dtype)
        current_latents_mask = torch.empty((B, 0), device=device, dtype=torch.bool)

        # Prepare incremental weaver-side embeds to avoid re-projecting the whole history
        current_weaver_inputs_embeds = torch.empty(
            (B, 0, self.weaver.config.hidden_size), device=device, dtype=self.weaver_to_reasoner.weight.dtype
        )

        # Iterate over the selected augmentation points
        for aug_point_idx in augmentation_indices:
            # Slice the current segment of original embeddings and attention mask
            segment_inputs_embeds = inputs_embeds[:, current_start_idx:aug_point_idx]
            segment_attention_mask = attention_mask[:, current_start_idx:aug_point_idx]
            segment_latents_mask = torch.zeros((B, segment_inputs_embeds.size(1)), device=device, dtype=torch.bool)

            # Concatenate the current segment to the accumulated embeddings and masks
            current_inputs_embeds = torch.cat([current_inputs_embeds, segment_inputs_embeds], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, segment_attention_mask], dim=1)
            current_position_ids = generate_position_ids(current_attention_mask)
            current_latents_mask = torch.cat([current_latents_mask, segment_latents_mask], dim=1)

            # Project only the newly added segment to weaver space and append (incremental)
            segment_weaver_inputs = self.reasoner_to_weaver(segment_inputs_embeds)
            current_weaver_inputs_embeds = torch.cat([current_weaver_inputs_embeds, segment_weaver_inputs], dim=1)

            # Determine whether this point is the end of the prompt (prompt augmentation)
            is_prompt_end_aug = (labels[:, aug_point_idx] != -100).all() and (labels[:, aug_point_idx-1] == -100).all().item()
            # Depending on type, use weaver to augment prompt or inference
            if is_prompt_end_aug:
                logging.info(f"[augment] idx={aug_point_idx} TYPE=PROMPT")
                weaver_hidden_states, attn_mask, pos_ids = weaver.augment_prompt(
                    current_weaver_inputs_embeds, current_attention_mask, current_position_ids
                )
            else:
                logging.info(f"[augment] idx={aug_point_idx} TYPE=INFERENCE")
                weaver_hidden_states, attn_mask, pos_ids = weaver.augment_inference(
                    current_weaver_inputs_embeds, current_attention_mask, current_position_ids
                ) 

            # Map weaver hidden states back to reasoner embeddings
            latent_inputs_embeds = self.weaver_to_reasoner(weaver_hidden_states) # NOTE

            # Update accumulated embeddings and masks with the newly augmented segment
            current_inputs_embeds = torch.cat([current_inputs_embeds, latent_inputs_embeds], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, attn_mask], dim=1)
            current_start_idx = aug_point_idx
            
            # Update latent mask for the newly added latent embeddings
            latent_mask = torch.ones((B, latent_inputs_embeds.size(1)), device=device, dtype=torch.bool)
            current_latents_mask = torch.cat([current_latents_mask, latent_mask], dim=1)

            # Keep weaver-side embeds in sync by appending the latent hidden states directly
            current_weaver_inputs_embeds = torch.cat([current_weaver_inputs_embeds, weaver_hidden_states], dim=1)
            
        # Process the remaining segment after the last augmentation point
        remaining_inputs_embeds = inputs_embeds[:, current_start_idx:]
        remaining_attention_mask = attention_mask[:, current_start_idx:]
        latent_mask = torch.zeros((B, remaining_attention_mask.size(1)), device=device, dtype=torch.bool)
        
        current_inputs_embeds = torch.cat([current_inputs_embeds, remaining_inputs_embeds], dim=1)
        current_attention_mask = torch.cat([current_attention_mask, remaining_attention_mask], dim=1)
        current_position_ids = generate_position_ids(current_attention_mask)
        current_latents_mask = torch.cat([current_latents_mask, latent_mask], dim=1)

        reasoner_outputs = reasoner(
            inputs_embeds=current_inputs_embeds,
            attention_mask=current_attention_mask,
            position_ids=current_position_ids
        )
        logits = reasoner_outputs.logits
        
        # Identify valid positions in logits (positions that should contribute to loss)
        shifted = torch.zeros_like(current_latents_mask)
        shifted[:, :-1] = current_latents_mask[:, 1:]
        valid_mask = ~shifted
        
        valid_logits = logits[valid_mask].view(logits.size(0), -1, logits.size(2))  
        # assert shifted.sum() == current_latents_mask.sum()
        # assert valid_logits.shape[:2] == input_ids.shape
        return valid_logits
    
    # @log_function_call
    def _instructional_forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[Tuple[int, int, int]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for single-turn instructional data (no multi-turn conversation required).

        This method is used for instruction-following tasks (SFT), where the input
        consists of a single instruction and the corresponding labels. It directly
        delegates to the single-turn forward method `_forward`.

        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) containing input token IDs.
            attention_mask (torch.Tensor): Tensor indicating padding positions.
            labels (torch.Tensor): Tensor containing the target labels for supervised fine-tuning.
            **kwargs: Additional keyword arguments passed to `_forward`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - logits: The output logits from the model for each input token.
                - labels: The same as input labels, used for loss computation.
        """
        logits = self._forward(
            input_ids,
            attention_mask,
            labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )
        # For Instruction SFT, labels remain the same as input
        return logits, labels

    @log_function_call
    def _conversational_forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[Tuple[int, int, int]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for conversational (multi-turn) data.

        Multi-turn forward is constructed by sequentially calling the single-turn forward
        for each conversation turn. Latents inserted in turn i-1 are not visible to turn i.

        Args:
            input_ids (torch.Tensor): Input token IDs, shape (1, seq_len). Batch size must be 1.
            attention_mask (torch.Tensor): Attention mask for input tokens.
            labels (torch.Tensor): Target labels for supervised fine-tuning (-100 for ignore positions).
            **kwargs: Additional arguments passed to `_forward`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - all_logits: Logits for the entire sequence, with zeros for unsupervised positions.
                - all_labels: Labels for the entire sequence, with -100 for unsupervised positions.
        """
        assert input_ids.shape[0] == 1, "Conversational SFT currently only supports batch_size = 1"
        seq_len = input_ids.shape[1]
        vocab_size = self.config.vocab_size
        device = input_ids.device

        # Identify single-turn segments within the conversation based on labels
        label_row = labels[0]
        should_supervise = label_row != -100
        if not should_supervise.any():
            raise ValueError("At least one completion segment is required")

        # Compute the start and end indices of valid supervised segments
        valid_mask = should_supervise.int()
        diff = torch.diff(torch.cat([torch.tensor([0], device=device), valid_mask]))
        valid_starts = (diff == 1).nonzero(as_tuple=True)[0].tolist()  # Transition 0 -> 1
        ends = (diff == -1).nonzero(as_tuple=True)[0].tolist()          # Transition 1 -> 0
        if len(ends) < len(valid_starts):
            ends.append(seq_len)
        assert len(valid_starts) == len(ends)
        
        # Build triplets (start of previous segment, start of supervised segment, end of supervised segment)
        triplets = []
        start = 0
        for s, e in zip(valid_starts, ends):
            triplets.append((start, s, e))
            start = e
        
        # If there are more segments than allowed, randomly select self.max_prompt_aug_num segments
        # IMPORTANT: During GRPO training (weaver training), we MUST supervise all turns consistently
        # to ensure that supervision masks match across multiple forward passes.
        # Only apply random sampling when training the reasoner (not the weaver).
        if self.training and hasattr(self, 'weaver') and self.weaver is not None and self.weaver.training:
            # Training weaver (GRPO): supervise ALL turns for consistency
            select_turns = [1] * len(triplets)
            # logging.info(f"[SUPERVISE DEBUG] Weaver training mode: supervising ALL {len(triplets)} turns")
        elif len(triplets) <= self.max_prompt_aug_num:
            select_turns = [1] * len(triplets)
        else:
            triplets_num = len(triplets)
            selected_indices = set(random.sample(range(triplets_num), self.max_prompt_aug_num))
            select_turns = [1 if i in selected_indices else 0 for i in range(triplets_num)]
        
        # logging.info(f"[SUPERVISE DEBUG] len(triplets): {len(triplets)}, max_prompt_aug_num: {self.max_prompt_aug_num}, select_turns: {select_turns}, training: {self.training}")

        # Initialize tensors to store logits and labels for the entire sequence
        all_logits = torch.zeros(1, seq_len, vocab_size, device=device)
        all_labels = torch.full((1, seq_len), -100, device=device)

        # Loop over each conversation turn and perform single-turn forward if supervised
        for triplet, should_supervise in zip(triplets, select_turns):
            start, valid_start, end = triplet
            if should_supervise:
                cur_input_ids = input_ids[0, :end].unsqueeze(0)
                cur_attention = attention_mask[0, :end].unsqueeze(0)
                # cur_labels only used for _forward, does not represent the true supervision range
                cur_labels = labels[0, :end].clone().unsqueeze(0)
                cur_labels[0, :valid_start] = -100  # Mask tokens before supervision start

                # Single-turn forward for the current conversation segment
                logits = self._forward(
                    cur_input_ids,
                    cur_attention,
                    cur_labels,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    **kwargs,
                )
                
                # Update overall logits and labels with the results of this segment
                all_logits[0, start:end, :] = logits[0, start:end, :]
                all_labels[0, start:end] = labels[0, start:end]

        # Return logits and labels:
        # - supervised positions retain computed logits and original labels
        # - unsupervised positions have logits = 0 and labels = -100
        # logging.info(f"[SUPERVISE DEBUG] all_labels non-(-100) count: {(all_labels != -100).sum().item()}, total: {all_labels.numel()}")
        return all_logits, all_labels

    @log_function_call
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[Tuple[int, int, int]] = None,
        **kwargs
    ):  
        tokenizer = self.tokenizer

        # Ensure labels are provided, required for training the reasoning processor
        assert labels is not None, "Reasoning Processor requires input labels for training"
        
        # Determine whether the input is single-turn (instruction) or multi-turn (conversation)
        forward_func = self._instructional_forward
        if is_conversation(input_ids, tokenizer):
            # For conversational data, mask assistant tokens in labels
            logging.info("is_conversation: True")
            labels = postprocess_assistant_labels(input_ids, labels, tokenizer)
            forward_func = self._conversational_forward
        
        batch_size = 1  # Currently process one sequence per batch
        iter_num = input_ids.size(0) // batch_size

        # Forward pass per batch
        logits, supervised_labels = [], []
        for i in range(iter_num):
            batch_input_ids = input_ids[i * batch_size: (i + 1) * batch_size]
            batch_attention_mask = attention_mask[i * batch_size: (i + 1) * batch_size]
            batch_labels = labels[i * batch_size: (i + 1) * batch_size]

            # Call the appropriate forward function (instruction or conversation)
            batch_logits, batch_supervised_labels = forward_func( # forward_func --> _forward
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                **kwargs
            )
            logits.append(batch_logits)
            supervised_labels.append(batch_supervised_labels)
        
        # Concatenate results from all batches
        all_logits = torch.concat(logits, dim=0)
        all_labels = torch.concat(supervised_labels, dim=0)

        # Compute causal language modeling loss (shifted by one)
        shift_logits = all_logits[..., :-1, :].contiguous()
        shift_labels = all_labels[..., 1:].contiguous()
        # assert shift_logits.shape[:-1] == shift_labels.shape
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Return model outputs
        outputs = CausalLMOutputWithPast(loss=loss, logits=all_logits)
        outputs.supervised_labels = all_labels  # Positions in input_ids that are supervised
        return outputs

    
    @log_function_call
    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        generation_config: GenerationConfig = None,
        return_augmentation_mask: bool = False,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[Tuple[int, int, int]] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
        
        tokenizer = self.tokenizer
        reasoner = self.model
        weaver = self.weaver
        trigger = self.trigger
        delimiters = self.delimiters
        max_augment_num = self.max_inference_aug_num
        invalid_token_id = -100

        # preproecess inputs
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        max_new_tokens = generation_config.max_new_tokens
        do_sample = generation_config.do_sample
        temperature = generation_config.temperature    # control reasoner generate and trigger generate
        pad_token_id = tokenizer.pad_token_id
        # Prefer chat end token (<|im_end|>) if available; fall back to tokenizer.eos
        try:
            im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
            if isinstance(im_end_ids, list) and len(im_end_ids) == 1:
                eos_token_id = im_end_ids[0]
            else:
                eos_token_id = tokenizer.eos_token_id
        except Exception:
            eos_token_id = tokenizer.eos_token_id
        prompt_len = input_ids.size(1)
        generation_config = GenerationConfig(
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            # use_cache=True
        )
        
        with torch.no_grad():
            outputs = reasoner(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
        
        # Use embedding layer output (hidden_states[0]) as inputs_embeds to ensure compatibility with model's embedding space.
        fused_embeds = outputs.hidden_states[0]  # Embedding output after token & vision embedding projection
        # logging.info(f"Seed embeds shape (from hidden_states[0]): {fused_embeds.shape}")

        inputs_embeds = fused_embeds

        B, _, hidden_size = inputs_embeds.shape
        device = inputs_embeds.device

        try:
            current_inputs_embeds = inputs_embeds
            current_attention_mask = attention_mask
            current_position_ids = generate_position_ids(current_attention_mask)
            current_input_ids = input_ids

            weaver_inputs_embeds = self.reasoner_to_weaver(current_inputs_embeds)
            weaver_hidden_states, attn_mask, pos_ids = weaver.augment_prompt(
                weaver_inputs_embeds, current_attention_mask, current_position_ids
            )
            latent_inputs_embeds = self.weaver_to_reasoner(weaver_hidden_states)

            # Concatenate initial augmented prompt
            current_inputs_embeds = torch.cat([current_inputs_embeds, latent_inputs_embeds], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, attn_mask], dim=1)
            current_position_ids = torch.cat([current_position_ids, pos_ids], dim=1)

            # Generation Loop Initialization
            sentence_augment_count = torch.zeros(B, dtype=torch.int, device=device)
            augmentation_pos = torch.full((B, max_new_tokens), fill_value=invalid_token_id, device=device)
            inserted_embeds: List[List[torch.Tensor]] = [[] for _ in range(B)]
            for i in range(max_new_tokens):
                
                # If all sequences in the batch have already generated an EOS token, stop early
                if (current_input_ids[:, -1] == eos_token_id).all():
                    break   
                
                # Check if all sequences have reached the maximum number of augmentations
                if (sentence_augment_count >= max_augment_num).all():
                    # Adjust the remaining generation length
                    generation_config.max_new_tokens = max_new_tokens - i

                    # Perform generation for the remaining tokens using the reasoner
                    generated = reasoner.generate(
                        inputs_embeds=current_inputs_embeds,
                        attention_mask=current_attention_mask,
                        generation_config=generation_config,
                    )
                    current_input_ids = torch.cat([current_input_ids, generated], dim=1)
                    break

                outputs = reasoner(
                    inputs_embeds=current_inputs_embeds,
                    attention_mask=current_attention_mask,
                    position_ids=current_position_ids,
                    output_hidden_states=False,
                )
                current_inputs_embeds, current_attention_mask, current_position_ids, current_input_ids = self._append_one_step(
                    outputs, current_inputs_embeds, current_attention_mask, current_position_ids, current_input_ids, do_sample, temperature
                )
 
                if i == max_new_tokens - 1:  
                    break 

                # Determine which sentences in the batch should be augmented
                augment_decision = self._should_augment(
                    current_input_ids, current_attention_mask, sentence_augment_count=sentence_augment_count, 
                    do_sample=do_sample, temperature=temperature  
                )
                augmentation_pos[:, i + 1] = augment_decision
                augment_indices = torch.where(augment_decision == 1)[0]

                # If there are sentences to augment, apply augmentation; others remain with left padding
                if len(augment_indices) > 0:
                    # Increment the augmentation count for sentences that are being augmented
                    sentence_augment_count[augment_indices] += 1

                    # Select embeddings, attention masks, and position IDs for sentences to be augmented
                    candidate_inputs_embeds = current_inputs_embeds[augment_indices]
                    candidate_attention_mask = current_attention_mask[augment_indices]
                    candidate_position_ids = current_position_ids[augment_indices]
                    
                    # Perform inference augmentation using the weaver
                    weaver_inputs_embeds = self.reasoner_to_weaver(candidate_inputs_embeds)
                    weaver_hidden_states, attn_mask, _ = weaver.augment_inference(
                        weaver_inputs_embeds, candidate_attention_mask, candidate_position_ids
                    )
                    latent_inputs_embeds = self.weaver_to_reasoner(weaver_hidden_states)
                    
                    candidate_inputs_embeds = torch.cat([candidate_inputs_embeds, latent_inputs_embeds], dim=1)
                    candidate_attention_mask = torch.cat([candidate_attention_mask, attn_mask], dim=1)
                    
                    # Create a single merged tensor for all sequences
                    new_len = candidate_inputs_embeds.size(1)
                    merged_inputs_embeds = torch.zeros((B, new_len, hidden_size), device=device, dtype=current_inputs_embeds.dtype)
                    merged_attention_mask = torch.zeros((B, new_len), device=device, dtype=current_attention_mask.dtype)
                    
                    # Directly place augmented and non-augmented sequences
                    merged_inputs_embeds[augment_indices] = candidate_inputs_embeds
                    merged_attention_mask[augment_indices] = candidate_attention_mask
                    
                    # Non-augmented sequences now include both -100 and 0
                    non_augment_indices = torch.where(augment_decision != 1)[0]
                    if len(non_augment_indices) > 0:
                        non_aug_inputs_embeds = current_inputs_embeds[non_augment_indices]
                        non_aug_attention_mask = current_attention_mask[non_augment_indices]
                        non_aug_inputs_embeds, non_aug_attention_mask, _ = self._left_pad(
                            non_aug_inputs_embeds, non_aug_attention_mask, None, weaver.inference_latents_num
                        )
                        merged_inputs_embeds[non_augment_indices] = non_aug_inputs_embeds
                        merged_attention_mask[non_augment_indices] = non_aug_attention_mask
                    
                    current_inputs_embeds = merged_inputs_embeds
                    current_attention_mask = merged_attention_mask
                    current_position_ids = generate_position_ids(current_attention_mask)
                    
                    # Record inserted embeds for post-processing
                    for idx, embed in zip(augment_indices, latent_inputs_embeds):
                        inserted_embeds[idx].append(embed.clone().detach().cpu())
        except Exception:
            logging.exception(
                f"[generate] Exception. do_sample={do_sample}, temperature={temperature}, pad_id={pad_token_id}, eos_id={eos_token_id}, "
                f"pixel_present={pixel_values is not None}, max_new_tokens={max_new_tokens}"
            )
            raise
        
        # postprocess
        new_generated_len = current_input_ids.size(1) - prompt_len
        augmentation_pos = augmentation_pos[:, :new_generated_len]
         
        if not return_augmentation_mask:
            return current_input_ids
        else:
            return current_input_ids, augmentation_pos

    
    @classmethod
    @log_function_call
    def from_config(cls, config):
        # reasoner configs
        reasoner_model_name = config.get("reasoner_model_name", None)
        max_prompt_aug_num = config.get("max_prompt_aug_num", None)
        max_inference_aug_num = config.get("max_inference_aug_num", None)

        # processor configs
        weaver_configs = config.get("weaver")
        weaver_model_name = weaver_configs.get("weaver_model_name", None)
        prompt_latents_len = weaver_configs.get("prompt_latents_len", 8)
        inference_latents_len = weaver_configs.get("inference_latents_len", 2)
        weaver_use_peft = weaver_configs.get("use_peft", True)
        weaver_peft_config = weaver_configs.get("peft_config", None) if weaver_use_peft else None

        if weaver_peft_config is not None:  
            weaver_peft_config = LoraConfig(**weaver_peft_config)
        
        # trigger configs
        trigger_configs = config.get("trigger")
        trigger_model_name = trigger_configs.get("trigger_model_name", None)
        trigger_use_peft = trigger_configs.get("use_peft", True)
        trigger_peft_config = trigger_configs.get("peft_config", None) if trigger_use_peft else None

        if trigger_peft_config is not None:
            trigger_peft_config = LoraConfig(**trigger_peft_config)

        model = cls(
            reasoner_model_name,
            # weaver configs
            weaver_model_name=weaver_model_name,
            prompt_latents_len=prompt_latents_len,
            inference_latents_len=inference_latents_len,
            weaver_peft_config=weaver_peft_config,
            # trigger configs
            trigger_model_name=trigger_model_name,
            trigger_peft_config=trigger_peft_config,
            # augmentations
            max_prompt_aug_num=max_prompt_aug_num,
            max_inference_aug_num=max_inference_aug_num
        )

        # load model state dict
        load_model_path = config.get("load_model_path", None)
        if load_model_path is not None:
            model_state_dict = load_state_dict_from_safetensor(load_model_path)
            model.load_state_dict(model_state_dict, strict=False)
            logging.info(f"Load model state dict from: {load_model_path}")

        return model
    

    # @log_function_call
    @torch.no_grad()
    def _append_one_step(
        self,
        reasoner_outputs, 
        current_inputs_embeds: torch.Tensor,
        current_attention_mask: torch.Tensor,
        current_position_ids: torch.Tensor,
        current_input_ids: torch.Tensor,
        do_sample: bool,
        temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reasoner = self.model
        B = current_inputs_embeds.size(0)
        
        # Append next token
        next_token_logits = reasoner_outputs.logits[:, -1]
        next_token_ids = get_next_token(next_token_logits, do_sample, temperature)
        current_input_ids = torch.cat([current_input_ids, next_token_ids], dim=1)
        
        # Append next token embeds
        next_token_embeds = reasoner.get_input_embeddings()(next_token_ids)
        current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embeds], dim=1)
        
        # Append attention mask
        attn_mask = torch.ones((B, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)
        current_attention_mask = torch.cat([current_attention_mask, attn_mask], dim=1)
        
        # Append position ids
        next_position_id = current_position_ids[:, -1:] + 1
        current_position_ids = torch.cat([current_position_ids, next_position_id], dim=1)

        return current_inputs_embeds, current_attention_mask, current_position_ids, current_input_ids
    
    # @log_function_call
    def _select_augment_points_after_delimiter(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        delimiters: List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_num: int = 10,
    ) -> List[int]:
        """
        Select positions in a sequence suitable for data augmentation based on labels and delimiters.

        This function identifies two types of augmentation points:
        1. **Prompt augmentation points**: positions where the label transitions from -100 to a valid token.
        - Typically corresponds to the start of the assistant's response.
        - There must be exactly one such point for single-turn forward processing.
        2. **Inference augmentation points**: positions inside valid label regions that follow a delimiter.
        - Only the first `max_num` points are kept.

        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) containing token IDs.
            labels (torch.Tensor): Tensor of shape (batch_size, seq_len), where -100 indicates
                                positions that should not contribute to loss.
            delimiters (List[str]): List of string delimiters used to determine inference augmentation points.
            tokenizer (PreTrainedTokenizerBase): Tokenizer used to decode input_ids for delimiter matching.
            max_num (int, optional): Maximum number of inference augmentation points to select. Default is 10.

        Returns:
            List[int]: Sorted list of selected augmentation positions in the sequence.

        Raises:
            ValueError: If the prompt augmentation point count is not exactly one.
            RuntimeError: If no valid augmentation points are found.
        """
        assert input_ids.shape == labels.shape
        B, seq_len = input_ids.size(0), input_ids.size(1)

        prompt_augment_idx = []
        inference_augment_idx = []

        for i in range(1, seq_len):  # Skip the first token and last token for augmentation
            # Detect the boundary between prompt and label for prompt augmentation
            if (labels[:, i] != -100).all() and (labels[:, i - 1] == -100).all():
                prompt_augment_idx.append(i)

            # Detect valid label regions for inference augmentation
            elif (labels[:, i] != -100).all() and (labels[:, i - 1] != -100).all():
                batch_tokens_before_i = input_ids[:, :i]
                # Assume check_ends_with_delimiter is defined
                if any(check_ends_with_delimiter(batch_tokens_before_i, tokenizer, delimiters)):
                    inference_augment_idx.append(i)
        
        # Ensure exactly one prompt augmentation point exists for single-turn processing
        if len(prompt_augment_idx) != 1:
            raise ValueError("Single-turn forward must have exactly one prompt augment index")

        final_points = prompt_augment_idx[:1]

        # Limit the number of inference augmentation points to max_num
        if len(inference_augment_idx) > max_num:
            inference_augment_idx = inference_augment_idx[:max_num]

        final_points.extend(inference_augment_idx)
        
        if len(final_points) == 0:
            raise RuntimeError("No valid augmentation points found")
        
        final_points.sort()
        return final_points

    # @log_function_call
    @torch.no_grad()
    def _should_augment(
        self, 
        input_ids, 
        attention_mask, 
        sentence_augment_count: torch.Tensor, 
        do_sample: bool, 
        temperature: float = 0.0
    ) -> torch.Tensor:
        """
        Determine whether each sentence in the batch should be augmented based on
        the model's strategy and trigger predictions.

        aug_mask values:
            - -100: No sampling or augmentation decision not applicable
            - 0   : Sampled but trigger indicates no augmentation
            - 1   : Sampled and trigger indicates augmentation

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len)
            attention_mask (torch.Tensor): Attention mask for input_ids
            sentence_augment_count (torch.Tensor): Tracks how many times each sentence has been augmented
            do_sample (bool): Whether to sample from the trigger output
            temperature (float): Sampling temperature

        Returns:
            torch.Tensor: Augmentation mask for each sentence in the batch
        """
        tokenizer = self.tokenizer
        delimiters = self.delimiters
        trigger = self.trigger
        max_augment_num = self.max_inference_aug_num

        batch_size = input_ids.size(0)

        # Initialize aug_mask with -100, meaning no augmentation by default
        aug_mask = torch.full((batch_size,), -100, dtype=torch.long, device=input_ids.device)

        # Mark sentences ending with delimiters as candidates for augmentation (set to 0)
        ends_with_delimiters = check_ends_with_delimiter(input_ids, tokenizer, delimiters).squeeze(1)
        aug_mask[ends_with_delimiters] = 0

        # If a sentence has already reached the max augmentation count, reset to -100
        over_limit = (sentence_augment_count >= max_augment_num)
        aug_mask[over_limit] = -100

        # Apply trigger model only on sentences that are candidates (aug_mask != -100)
        trigger_indices = (aug_mask != -100).nonzero(as_tuple=True)[0]
        if trigger_indices.numel() > 0:
            trigger_logits = trigger(
                input_ids=input_ids[trigger_indices],
                attention_mask=attention_mask[trigger_indices],
            )
            last_token_logits = trigger_logits[:, -1]  # [num_trigger_samples, 2]

            # Sample the next token to decide whether to augment
            next_tokens = get_next_token(last_token_logits, do_sample, temperature).view(-1)

            # Update aug_mask: 0 = no augmentation, 1 = augment
            aug_mask[trigger_indices] = next_tokens

        return aug_mask

    
    # @log_function_call
    @torch.no_grad()
    def _left_pad(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        pad_num: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if input_embeds is not None:
            B, L, D = input_embeds.shape
            pad_embeds = torch.zeros((B, pad_num, D), dtype=input_embeds.dtype, device=input_embeds.device)
            input_embeds = torch.cat([pad_embeds, input_embeds], dim=1)  # [B, pad_num + L, D]
        
        if attention_mask is not None:
            B = attention_mask.size(0)
            pad_mask = torch.zeros((B, pad_num), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([pad_mask, attention_mask], dim=1)  # [B, pad_num + L]
        
        if position_ids is not None:
            B = position_ids.size(0)
            pad_pos = torch.zeros((B, pad_num), dtype=position_ids.dtype, device=position_ids.device)
            position_ids = torch.cat([pad_pos, position_ids], dim=1)  # [B, pad_num + L]

        return input_embeds, attention_mask, position_ids
    
    @log_function_call
    @torch.no_grad()
    def _left_clip_pad_tokens(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Left-trim padding tokens based on the attention mask.

        This function identifies the leftmost padding (attention_mask=0) in each
        sequence. If all sequences in the batch have at least some padding on the
        left, it trims the batch by the minimal left-padding length to remove
        unnecessary computation.

        Args:
            inputs_embeds (torch.Tensor): Input embeddings of shape (B, L, D)
            attention_mask (torch.Tensor): Attention mask of shape (B, L)
            position_ids (torch.Tensor): Position IDs of shape (B, L)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                Trimmed inputs_embeds, attention_mask, and position_ids
        """
        B, L, D = inputs_embeds.shape
        device = inputs_embeds.device

        # Find the index of the first non-padding token in each sequence
        first_nonpad_idx = []
        for b in range(B):
            nonzero = (attention_mask[b] != 0).nonzero(as_tuple=True)[0]
            if len(nonzero) == 0:
                # Entire row is padding; can potentially trim the whole sequence
                first_nonpad_idx.append(L)
            else:
                first_nonpad_idx.append(nonzero[0].item())
        
        # Determine the minimum number of left-padding tokens across the batch
        min_pad = min(first_nonpad_idx)

        # If no padding on the left, return original tensors
        if min_pad == 0:
            return inputs_embeds, attention_mask, position_ids

        # Trim the left-padding from all sequences in the batch
        inputs_embeds = inputs_embeds[:, min_pad:, :]
        attention_mask = attention_mask[:, min_pad:]
        position_ids = position_ids[:, min_pad:]

        return inputs_embeds, attention_mask, position_ids
    
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Enable gradient checkpointing for all sub-modules that support it.
        """
        kwargs = gradient_checkpointing_kwargs or {}
        for module in [self.model, getattr(self, "weaver", None),
                        getattr(self, "trigger", None)]:
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        """
        Disable gradient checkpointing for all sub-modules that support it.
        """
        for module in [self.model, getattr(self, "weaver", None),
                        getattr(self, "trigger", None)]:
            if hasattr(module, "gradient_checkpointing_disable"):
                module.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self):
        """
        Check if gradient checkpointing is enabled for any sub-module.
        """
        for module in [self.model, getattr(self, "weaver", None),
                        getattr(self, "trigger", None)]:
            if hasattr(module, "is_gradient_checkpointing"):
                if module.is_gradient_checkpointing:
                    return True
        return False

