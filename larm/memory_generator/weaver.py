import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import PeftConfig, get_peft_model

from typing import Optional, Tuple

# Decorator to log function calls in blue
import functools
import logging

def log_function_call(func):
    """Decorator to log function calls with blue color."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        # Blue color ANSI code
        logging.info(f"\033[94m[Weaver] {func_name}\033[0m")
        return func(*args, **kwargs)
    return wrapper


class MemGenWeaver(torch.nn.Module):
    """
    Weaver module for the MemGen Model.
    - Input: the weaver receives `inputs_embeds` from the reasoner model's current decoding sequence.
    - Output: the weaver produces a sequence of hidden states with length K, 
      which are concatenated to the original `inputs_embeds` to alter the reasoner's decoding path.
    """
    def __init__(
        self, 
        pretrained_model_name_or_path: str, 
        prompt_latents_num: int,
        inference_latents_num: int,
        peft_config: Optional[PeftConfig] = None
    ):
        super().__init__()
        
        # base model
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        if peft_config is not None:
            self.model = get_peft_model(self.model, peft_config)
        
        self.config = self.model.config
        
        # prompt augmentation
        self.prompt_query_latents = nn.Parameter(
            torch.randn(prompt_latents_num, self.config.hidden_size), 
            requires_grad=True
        )

        # inference augmentation
        self.inference_query_latents = nn.Parameter(
            torch.randn(inference_latents_num, self.config.hidden_size), 
            requires_grad=True
        )
    
    @property
    def prompt_latents_num(self) -> int:
        return self.prompt_query_latents.size(0)

    @property
    def inference_latents_num(self) -> int:
        return self.inference_query_latents.size(0)

    @property
    def device(self):
        return self.model.device

    @log_function_call
    def _augment(
        self, 
        latents: torch.Tensor,
        inputs_embeds: torch.Tensor, 
        attention_mask: torch.Tensor, 
        position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size = attention_mask.shape[0]
        latents_num = latents.size(0)
        latents = latents.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # inputs_embeds
        inputs_embeds = torch.cat([inputs_embeds, latents], dim=1)

        # attention_mask: (B, L_total)
        latents_mask = torch.ones(latents.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, latents_mask], dim=1)
        
        # get position ids
        last_position_ids = position_ids.max(dim=1)[0]
        latents_relative_positions = torch.arange(latents_num, device=attention_mask.device)
        latents_position_ids = last_position_ids.unsqueeze(1) + latents_relative_positions + 1
        position_ids = torch.cat([position_ids.long(), latents_position_ids.long()], dim=1) 

        # the processor only outputs the hidden states
        assert inputs_embeds.shape[:2] == attention_mask.shape == position_ids.shape

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,  
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        latents_hidden_states = hidden_states[:, -latents_num:, :]

        return latents_hidden_states, latents_mask, latents_position_ids

    @log_function_call
    def augment_prompt(
        self, 
        inputs_embeds: torch.Tensor, 
        attention_mask: torch.Tensor, 
        position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._augment(
            latents=self.prompt_query_latents,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

    @log_function_call
    def augment_inference(
        self, 
        inputs_embeds: torch.Tensor, 
        attention_mask: torch.Tensor, 
        position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._augment(
            latents=self.inference_query_latents,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids
        )