import copy
import gc
import json
import logging
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union

import torch
from PIL import Image
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_flash_attn_2_available, is_peft_available, is_rich_available
from trl import GRPOTrainer, GRPOConfig
from trl.trainer.utils import selective_log_softmax
from trl.data_utils import maybe_apply_chat_template, is_conversational 
from trl.models import create_reference_model, prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
if is_peft_available():
    from peft import PeftConfig, get_peft_model
if is_wandb_available():
    import wandb

from larm.data.interactions.base_interaction import (
    InteractionManager, InteractionDataProto
)
from larm.data.envs.base_env import StaticEnv, DynamicEnv

from .utils import (
    nanstd, nanmax, nanmin,
    init_grpo_log_files, persist_grpo_logs,
    log_prompt_truncation, extract_answer
)
from ..memgen_model import LatentMemoryModel
from .verifier import verify_solution_equivalence

import functools

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

# Decorator to log function calls in blue
def log_function_call(func):
    """Decorator to log function calls with blue color."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logging.info(f"\033[94m[CALL] {func_name}\033[0m") # blue color
        return func(*args, **kwargs)
    return wrapper

class WeaverGRPOTrainer(GRPOTrainer):

    def __init__(
        self,
        model: LatentMemoryModel,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        env_class = None,   # env main class
        env_main_config = None,  # configs to initialize an env object
        generation_manager: InteractionManager = None  # manage the interaction between agent and env
    ):
        super().__init__(
            model, 
            reward_funcs,
            args,
            train_dataset,
            eval_dataset,
            processing_class,
            reward_processing_classes,
            callbacks,
            optimizers,
            peft_config
        )
        
        self.env_class = env_class
        self.env_main_config = env_main_config
        self.generation_manager = generation_manager
        
        # Enforce single-turn mode only
        if not issubclass(self.env_class, StaticEnv):
            raise ValueError("WeaverGRPOTrainer is in single-turn mode; please provide a StaticEnv.")
        # Disable vLLM path in simplified single-turn trainer
        if getattr(self, "use_vLLM", None) is True or getattr(self, "use_vllm", None) is True:
            raise ValueError("vLLM path is disabled in single-turn trainer. Please set use_vllm=False.")

        assert self.max_prompt_length == generation_manager.config.max_start_length
        assert self.max_completion_length == generation_manager.config.max_response_length
        assert self.temperature == generation_manager.config.temperature
        
        # Initialize GRPO logging files via utility
        self.grpo_log_file, self.grpo_jsonl_file = init_grpo_log_files(args.output_dir)
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step on a batch of inputs, with memory cleanup after each step.
        """
        # Call the parent class's training_step
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # Clear memory after each training step
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return loss
    
    @log_function_call
    def _get_per_token_logps(
        self, model, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        labels: torch.Tensor, 
        logits_to_keep: int,
        batch_size: int = None,
        pixel_values: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None
    ) -> torch.Tensor:
        logging.info(f"[GRAD DEBUG _get_per_token_logps] Called with pixel_values: {pixel_values is not None}, image_grid_thw: {image_grid_thw is not None}")
        if pixel_values is not None:
            logging.info(f"[GRAD DEBUG _get_per_token_logps] pixel_values shape: {pixel_values.shape}, requires_grad: {pixel_values.requires_grad}")
        logging.info(f"[GRAD DEBUG _get_per_token_logps] model.training: {model.training}")
        
        # Check if model parameters have gradients
        trainable_params = sum(p.requires_grad for p in model.parameters())
        total_params = sum(1 for _ in model.parameters())
        logging.info(f"[GRAD DEBUG _get_per_token_logps] Model params: {trainable_params}/{total_params} trainable")
        
        # Check weaver training status
        if hasattr(model, 'weaver') and model.weaver is not None:
            weaver_trainable = sum(p.requires_grad for p in model.weaver.parameters())
            weaver_total = sum(1 for _ in model.weaver.parameters())
            # logging.info(f"[GRAD DEBUG _get_per_token_logps] Weaver params: {weaver_trainable}/{weaver_total} trainable, weaver.training: {model.weaver.training}")
        
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        supervise_masks = []   
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            labels_slice = labels[start : start + batch_size]
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch, "labels": labels_slice}

            # Add vision inputs if present (for weaver/VLM models)
            if pixel_values is not None:
                pixel_values_batch = pixel_values[start : start + batch_size]
                model_inputs["pixel_values"] = pixel_values_batch
            if image_grid_thw is not None:
                image_grid_thw_batch = image_grid_thw[start : start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw_batch

            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            outputs = model(**model_inputs)
            logits = outputs.logits
            supervised_labels = outputs.supervised_labels
            
            # Debug: Check gradient status
            logging.info(f"[GRAD DEBUG] logits.requires_grad: {logits.requires_grad}, has grad_fn: {logits.grad_fn is not None}")
            if hasattr(outputs, 'pixel_values') or 'pixel_values' in model_inputs:
                logging.info(f"[GRAD DEBUG] Vision inputs present in forward pass")
            
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            logging.info(f"[GRAD DEBUG] logps.requires_grad: {logps.requires_grad}, has grad_fn: {logps.grad_fn is not None}")
            all_logps.append(logps)
            
            supervised_labels = supervised_labels[:, -logits_to_keep:]
            mask = (supervised_labels != -100).long()  
            supervise_masks.append(mask)

        logps = torch.cat(all_logps, dim=0)
        masks = torch.cat(supervise_masks, dim=0)
        return logps, masks

    @log_function_call
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]  # batch_size * num_generations
    ) -> dict[str, Union[torch.Tensor, Any]]:

        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # build no-tensor part
        batch_gen_keys = []
        if "prompt" in inputs[0]:  # text-based raw prompt
            batch_gen_keys.append("prompt")
        if "tools_kwargs" in inputs[0]:  # tool-integrated     
            batch_gen_keys.append("tools_kwargs")
        if "interaction_kwargs" in inputs[0]:  # interaction args
            batch_gen_keys.append("interaction_kwargs")
        if "agent_name" in inputs[0]:  # agent name
            batch_gen_keys.append("agent_name")    

        gen_batch = InteractionDataProto()
        for key in batch_gen_keys:  
            gen_batch.no_tensor_batch[key] = [x[key] for x in inputs]
        
        if not issubclass(self.env_class, StaticEnv):
            raise ValueError("Single-turn mode only: DynamicEnv is not supported.")

        from larm.memory_generator.utils import THINK_SYS_PROMPT
        prompts_text = []
        messages_list = []
        images_list = []
        any_image = False
        for i, example in enumerate(inputs):
            try:
                prompt_text = maybe_apply_chat_template(example, self.processing_class)["prompt"]
            except Exception as e:
                logging.warning(
                    f"Bad example at index {i}: {e}. Example type={type(example)}, value preview={repr(example)[:200]}"
                )
                prompt_text = ""
            prompts_text.append(prompt_text)

            # Try to load image if present
            img = None
            try:
                image_path = example.get("image_path") if isinstance(example, dict) else None
                if image_path is not None:
                    img = Image.open(image_path).convert("RGB")
            except Exception as e:
                logging.warning(f"Failed to load image for sample {i}: {e}")

            if img is not None:
                any_image = True
                messages_list.append([
                    {
                        "role": "system",
                        "content": THINK_SYS_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ])
                images_list.append(img)
            else:
                messages_list.append([
                    {
                        "role": "system",
                        "content": THINK_SYS_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt_text,
                    }
                ])
                images_list.append(None)

        # Prefer model.processor when available for VLMs
        processor = getattr(self.model, "processor", None)
        proc = processor if processor is not None else self.processing_class

        # If mixing image/no-image in one batch is detected, raise error (do not fallback)
        if any_image and any(x is None for x in images_list):
            logging.error("\033[91m[ERROR] Mixed image and text-only samples detected in the same batch. "
                          "Please group samples by modality (all-with-image or all-text).\033[0m")
            raise ValueError("Mixed image and text-only samples in one batch are not supported. Group by modality.")

        # Build encodings
        if hasattr(proc, "apply_chat_template"):
            # Each element in messages_list is already a list of messages (system + user)
            texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
        else:
            # Fallback: use the plain prompt texts
            texts = prompts_text

        if any_image:
            enc = proc(
                text=texts,
                images=images_list,
                return_tensors="pt",
                padding=True,
            )
        else:
            enc = proc(
                text=texts,
                return_tensors="pt",
                padding=True,
            )

        prompt_inputs = enc
            
        prompts_before_truncation = prompt_inputs["input_ids"].to(device)
        prompt_mask_before_truncation = prompt_inputs["attention_mask"].to(device)
        
        # Log original prompt info
        logging.info(f"[PROMPT INFO] Original prompt shape: {prompts_before_truncation.shape}, max_prompt_length: {self.max_prompt_length}")
        
        prompts, prompt_mask = prompts_before_truncation, prompt_mask_before_truncation
        # Do NOT truncate when vision inputs are present; truncation may break image-token alignment.
        has_pixels = isinstance(prompt_inputs, dict) and ("pixel_values" in prompt_inputs)
        if self.max_prompt_length is not None and not has_pixels:
            # prompts = prompts[:, -self.max_prompt_length :]
            # prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            
            logging.info(f"[PROMPT INFO] After truncation shape: {prompts.shape}")
            
            # Log prompt truncation for the first sample in the batch
            # Check if actual content was truncated (not just padding)
            if prompts_before_truncation.size(1) > self.max_prompt_length:
                logging.info(f"[PROMPT INFO] Truncation detected: {prompts_before_truncation.size(1)} -> {prompts.size(1)}")
                if self.accelerator.is_main_process:
                    log_prompt_truncation(
                        prompts_before=prompts_before_truncation,
                        prompts_after=prompts,
                        prompt_mask_before=prompt_mask_before_truncation,
                        prompt_mask_after=prompt_mask,
                        processing_class=self.processing_class,
                        max_prompt_length=self.max_prompt_length,
                        sample_idx=0
                    )
            else:
                logging.info(f"[PROMPT INFO] No truncation needed: length {prompts_before_truncation.size(1)} <= max {self.max_prompt_length}")
        elif has_pixels:
            logging.info("[PROMPT INFO] Vision inputs detected; skipping prompt truncation to preserve image tokens.")

        gen_batch.batch["input_ids"] = prompts 
        gen_batch.batch["attention_mask"] = prompt_mask
        # Attach image tensors if present from processor encoding
        vision_pixel_values = None
        vision_image_grid_thw = None
        if "pixel_values" in prompt_inputs:
            try:
                vision_pixel_values = prompt_inputs["pixel_values"].to(device).to(torch.bfloat16)
            except Exception:
                vision_pixel_values = prompt_inputs["pixel_values"].to(device)
            gen_batch.batch["pixel_values"] = vision_pixel_values
            # logging.info(f"[GRAD DEBUG _generate] vision_pixel_values shape: {vision_pixel_values.shape}, requires_grad: {vision_pixel_values.requires_grad}")
        if "image_grid_thw" in prompt_inputs:
            vision_image_grid_thw = prompt_inputs["image_grid_thw"].to(device)
            gen_batch.batch["image_grid_thw"] = vision_image_grid_thw
            # logging.info(f"[GRAD DEBUG _generate] vision_image_grid_thw shape: {vision_image_grid_thw.shape}")
        
        # logging.info(f"[GRAD DEBUG _generate] Saving vision tensors - pixel_values: {vision_pixel_values is not None}, image_grid_thw: {vision_image_grid_thw is not None}")
        
        # Regular generation path only (vLLM disabled in single-turn trainer)
        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            with (
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext()
            ):
                # Use GenerationManager to coordinate the interaction between the agent and the environment
                self.generation_manager.actor_rollout_wg = unwrapped_model
                # Enable augmentation mask return for training
                self.generation_manager._return_augmentation_mask = True
                final_gen_batch_output = self.generation_manager.run_agent_loop(gen_batch=gen_batch)
        
        # Clear memory after generation (inference only, no gradients needed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # parse outputs
        prompts = final_gen_batch_output.batch["prompts"].to(device)  # prompt ids
        completion_ids = final_gen_batch_output.batch["responses"].to(device)  # completion ids
        prompt_completion_ids = final_gen_batch_output.batch["input_ids"].to(device)  # prompt and completion ids
        attention_mask = final_gen_batch_output.batch["attention_mask"].to(device)  # attention_mask on prompt and response
        # Get augmentation positions if available
        augmentation_pos = final_gen_batch_output.batch.get("augmentation_pos", None)
        prompt_mask = attention_mask[:, :prompts.size(1)]  
        completion_mask = final_gen_batch_output.batch["info_mask"][:, prompts.size(1):].to(device) 
        # Prefer chat EOS (<|im_end|>) if tokenizer supports it
        _tok = getattr(self.processing_class, "tokenizer", self.processing_class)
        try:
            im_end_ids = _tok.encode("<|im_end|>", add_special_tokens=False)
            if isinstance(im_end_ids, list) and len(im_end_ids) == 1:
                eos_token_id = im_end_ids[0]
            else:
                eos_token_id = _tok.eos_token_id
        except Exception:
            eos_token_id = getattr(_tok, "eos_token_id", None)
            if eos_token_id is None:
                # Fallback: if eos token id is missing, mark nothing as eos to avoid crashes
                eos_token_id = -999999
        is_eos = completion_ids == eos_token_id
        assert completion_ids.shape == completion_mask.shape

        # Construct labels: Supervise only the agent response portion.
        prompt_labels = torch.full(prompt_mask.shape, -100, device=device)
        completion_labels = torch.where(completion_mask == 1, completion_ids, -100)
        labels = torch.cat([prompt_labels, completion_labels], dim=1)
        
        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)

        logits_to_keep = completion_mask.size(1)

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        with torch.no_grad():
            # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
            # old_per_token_logps == per_token_logps, so we can skip it's computation here, and use
            # per_token_logps.detach() instead.
            # NOTE: Do NOT pass vision inputs when computing logps on prompt+completion sequences,
            # as the vision grid info is only valid for the original prompt length.
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps, old_supervise_mask = self._get_per_token_logps( 
                    self.model, prompt_completion_ids, attention_mask, labels, logits_to_keep,
                    pixel_values=None, image_grid_thw=None
                )
            else:
                old_per_token_logps, old_supervise_mask = None, None
            
            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, ref_supervise_mask = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask, labels, logits_to_keep,
                        pixel_values=None, image_grid_thw=None
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, ref_supervise_mask = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, labels, logits_to_keep,
                            pixel_values=None, image_grid_thw=None
                        )
            else: 
                ref_per_token_logps, ref_supervise_mask = None, None
        
        # Clear memory after all inference computations (generation + old/ref logps)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Decode prompts and generated completions
        prompt_texts = self.processing_class.batch_decode(prompts, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions = completions_text
        
        # compute rewards
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        
        # fix: add scale_rewards
        # if self.scale_rewards:
        advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())
        
        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        # self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        # Persist per-sample solution (completion) and rewards via utility
        try:
            all_prompt_texts = gather_object(prompt_texts)
            all_completion_texts = gather_object(completions_text)
            all_rewards = gather_object(rewards.tolist())
            all_rewards_by_func = {
                name: gather_object(rewards_per_func[:, i].tolist())
                for i, name in enumerate(self.reward_func_names)
            }
            all_token_counts = gather_object(completion_lengths.tolist())
            # Gather ground truths from inputs (dataset label)
            all_ground_truths = gather_object([ex.get("solution", "") for ex in inputs])
            # Gather image paths from inputs
            all_image_paths = gather_object([ex.get("image_path", "") for ex in inputs])
            # Build stop reasons using EOS detection (True => eos, False => max_tokens)
            try:
                eos_flags = agg_terminated_with_eos.detach().cpu().tolist()
                all_stop_reasons = ["eos" if bool(f) else "max_tokens" for f in eos_flags]
            except Exception:
                all_stop_reasons = ["unknown"] * len(all_completion_texts)
            
            # Extract augmentation positions and insert <AUG> markers in completions
            all_augmentation_positions = None
            all_completion_texts_with_markers = None
            
            if augmentation_pos is not None:
                aug_pos_list = []
                completions_with_markers = []
                
                for i in range(completion_ids.size(0)):
                    response_len = completion_lengths[i].item()
                    aug_mask = augmentation_pos[i]  # augmentation mask for this sample
                    
                    # Find positions where augmentation occurred (mask value > 0)
                    aug_positions = []
                    for pos in range(min(len(aug_mask), response_len)):
                        if aug_mask[pos] > 0:
                            aug_positions.append(pos)
                    
                    # Format as "pos/total_len" for each augmentation
                    if len(aug_positions) > 0:
                        aug_info = ", ".join([f"{pos}/{response_len}" for pos in aug_positions])
                    else:
                        aug_info = "no_augmentation"
                    
                    aug_pos_list.append(aug_info)
                    
                    # Insert <AUG> markers in completion text
                    if len(aug_positions) > 0:
                        # Decode completion tokens
                        completion_tokens = completion_ids[i][:response_len].tolist()
                        
                        # Insert <AUG> marker after each augmentation position
                        # Work backwards to preserve indices
                        marked_tokens = []
                        last_pos = 0
                        aug_token_marker = " <AUG> "
                        
                        for pos in sorted(aug_positions):
                            # Decode tokens up to this position
                            if pos < len(completion_tokens):
                                segment_tokens = completion_tokens[last_pos:pos+1]
                                segment_text = self.processing_class.decode(segment_tokens, skip_special_tokens=True)
                                marked_tokens.append(segment_text)
                                marked_tokens.append(aug_token_marker)
                                last_pos = pos + 1
                        
                        # Add remaining tokens
                        if last_pos < len(completion_tokens):
                            remaining_tokens = completion_tokens[last_pos:]
                            remaining_text = self.processing_class.decode(remaining_tokens, skip_special_tokens=True)
                            marked_tokens.append(remaining_text)
                        
                        completion_with_markers = "".join(marked_tokens)
                    else:
                        # No augmentation, use original text
                        completion_with_markers = completions_text[i]
                    
                    completions_with_markers.append(completion_with_markers)
                
                # Gather augmentation positions and marked completions from all processes
                all_augmentation_positions = gather_object(aug_pos_list)
                all_completion_texts_with_markers = gather_object(completions_with_markers)
            else:
                # No augmentation info, use original completions
                all_completion_texts_with_markers = all_completion_texts

            if self.accelerator.is_main_process:
                # Compute extracted solutions and verify flags on main process
                all_solutions_extracted = [extract_answer(t) for t in all_completion_texts]
                # all_solutions_extracted = all_completion_texts
                # all_verifies = []
                # for sol, gt in zip(all_solutions_extracted, all_ground_truths):
                #     try:
                #         verdict = verify_solution_equivalence(sol, gt)
                #     except Exception:
                #         verdict = False
                #     all_verifies.append(verdict)
                persist_grpo_logs(
                    log_file=self.grpo_log_file,
                    jsonl_file=self.grpo_jsonl_file,
                    step=int(self.state.global_step),
                    mode=mode,
                    prompt_texts=all_prompt_texts,
                    completion_texts=all_completion_texts_with_markers,  # Use marked completions
                    rewards=all_rewards,
                    rewards_by_func=all_rewards_by_func,
                    token_counts=all_token_counts,
                    ground_truths=all_ground_truths,
                    solutions_extracted=all_solutions_extracted,
                    # verifies=all_verifies,
                    stop_reasons=all_stop_reasons,
                    reward_func_names=self.reward_func_names,
                    image_paths=all_image_paths,
                    augmentation_positions=all_augmentation_positions,
                )
        except Exception as e:
            logging.warning(f"Failed to persist GRPO logs: {e}")

        return {
            "prompt_ids": prompts,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "old_supervise_mask": old_supervise_mask,   
            "ref_per_token_logps": ref_per_token_logps,
            "ref_supervise_mask": ref_supervise_mask,
            "pixel_values": vision_pixel_values,
            "image_grid_thw": vision_image_grid_thw
        }

    @log_function_call
    def _compute_loss(self, model, inputs):
        device = self.accelerator.device

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        old_supervise_mask, ref_supervise_mask = inputs["old_supervise_mask"], inputs["ref_supervise_mask"]
        # Get vision inputs if present (for weaver/VLM models)
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)
        
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        prompt_labels = torch.full(prompt_mask.shape, -100, device=device)
        completion_labels = torch.where(completion_mask == 1, completion_ids, -100)
        labels = torch.cat([prompt_labels, completion_labels], dim=1)
        logits_to_keep = completion_labels.size(1)
        
        assert prompt_ids.shape == prompt_mask.shape
        assert completion_ids.shape == completion_mask.shape
        assert input_ids.shape == attention_mask.shape == labels.shape
        
        # Debug: Check vision inputs
        logging.info(f"[GRAD DEBUG _compute_loss] pixel_values: {pixel_values is not None}, image_grid_thw: {image_grid_thw is not None}")
        if pixel_values is not None:
            logging.info(f"[GRAD DEBUG _compute_loss] pixel_values shape: {pixel_values.shape}, requires_grad: {pixel_values.requires_grad}")
        
        # IMPORTANT: Do NOT pass vision inputs to _compute_loss forward pass
        # Vision inputs (pixel_values, image_grid_thw) are based on the original prompt,
        # but input_ids here contains prompt+completion which has a different length.
        # This mismatch causes index out of bounds errors in the vision encoder.
        # Vision processing is only needed during generation, not during loss computation.
        per_token_logps, supervise_mask = self._get_per_token_logps(
            model, input_ids, attention_mask, labels, logits_to_keep,
            pixel_values=None, image_grid_thw=None
        )
        logging.info(f"[GRAD DEBUG _compute_loss] Called _get_per_token_logps WITHOUT vision inputs to avoid index mismatch")
        
        logging.info(f"[GRAD DEBUG _compute_loss] per_token_logps.requires_grad: {per_token_logps.requires_grad}, has grad_fn: {per_token_logps.grad_fn is not None}")
        
        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        logging.info(f"[GRAD DEBUG _compute_loss] advantages.requires_grad: {advantages.requires_grad}, shape: {advantages.shape}")
        
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )
        logging.info(f"[GRAD DEBUG _compute_loss] old_per_token_logps.requires_grad: {old_per_token_logps.requires_grad}")
        
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        logging.info(f"[GRAD DEBUG _compute_loss] coef_1.requires_grad: {coef_1.requires_grad}, has grad_fn: {coef_1.grad_fn is not None}")
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        
        if old_supervise_mask is None:
            old_supervise_mask = supervise_mask
        if ref_supervise_mask is None:
            ref_supervise_mask = supervise_mask
        # Consistency check: The positions that are supervised must be a subset of the completion mask.
        assert (
            torch.all(supervise_mask <= completion_mask) and
            torch.all(old_supervise_mask <= completion_mask) and
            torch.all(ref_supervise_mask <= completion_mask)
        )
        supervised_mask = completion_mask * supervise_mask * old_supervise_mask * ref_supervise_mask  
        
        logging.info(f"[GRAD DEBUG _compute_loss] per_token_loss.requires_grad: {per_token_loss.requires_grad}, has grad_fn: {per_token_loss.grad_fn is not None}")
        logging.info(f"[GRAD DEBUG _compute_loss] supervised_mask sum: {supervised_mask.sum().item()}, completion_mask sum: {completion_mask.sum().item()}")

        if self.loss_type == "grpo":
            loss = ((per_token_loss * supervised_mask).sum(-1) / supervised_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * supervised_mask).sum() / supervised_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * supervised_mask).sum() / (supervised_mask.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        logging.info(f"[GRAD DEBUG _compute_loss] FINAL loss.requires_grad: {loss.requires_grad}, has grad_fn: {loss.grad_fn is not None}, loss value: {loss.item()}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * supervised_mask).sum() / supervised_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * supervised_mask).sum() / supervised_mask.sum()
        high_clip = (is_high_clipped * supervised_mask).sum() / supervised_mask.sum()
        clip_ratio = (is_region_clipped * supervised_mask).sum() / supervised_mask.sum()

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss