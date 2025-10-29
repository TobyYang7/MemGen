import torch
import logging
from typing import Dict, List
from transformers import GenerationConfig

from larm.data.interactions.base_interaction import (
    InteractionConfig, 
    InteractionManager,
    InteractionDataProto
)


class SingleTurnInteractionManager(InteractionManager):
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: InteractionConfig,
        is_validation: bool = False,
    ):
        super().__init__(
            tokenizer, actor_rollout_wg, config, is_validation
        )
        # generation configs for agent
        # Prefer chat end token (<|im_end|>) if available for EOS
        try:
            im_end_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
            if isinstance(im_end_ids, list) and len(im_end_ids) == 1:
                eos_id = im_end_ids[0]
            else:
                eos_id = self.tokenizer.eos_token_id
        except Exception:
            eos_id = self.tokenizer.eos_token_id

        self.generation_config = GenerationConfig(
            do_sample=self.config.do_sample,
            max_new_tokens=self.config.max_response_length,
            temperature=self.config.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=eos_id
        )
    
    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(  
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']    

    def _info_masked_concatenate_with_padding(self, 
        prompt: torch.Tensor, 
        prompt_with_mask: torch.Tensor, 
        response: torch.Tensor, 
        info: torch.Tensor = None,
        pad_to_left: bool = True
    ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info
    
    def _update_right_side(
        self, right_side: Dict, 
        cur_responses: torch.Tensor,
        next_obs_ids: torch.Tensor = None
    ) -> Dict:
        """Update right side state."""
        if next_obs_ids != None: 
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False   
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}
    
    def _log_generation_input(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, sample_idx: int = 0):
        """Log the actual input tokens passed to model.generate()."""
        try:
            # Check for vision/image tokens - use known IDs directly
            # Qwen2.5-VL vision token IDs:
            # 151652: <|vision_start|>
            # 151653: <|vision_end|>  
            # 151654: <|video_pad|>
            # 151655: <|image_pad|>
            vision_token_ids = [151652, 151653, 151654, 151655]
            
            # Also try to get them from tokenizer
            vision_token_names = ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>", "<|vision_pad|>"]
            for token_name in vision_token_names:
                try:
                    token_id = self.tokenizer.encode(token_name, add_special_tokens=False)
                    if isinstance(token_id, list) and len(token_id) > 0:
                        if token_id[0] not in vision_token_ids:
                            vision_token_ids.append(token_id[0])
                except Exception:
                    pass
            
            logging.info(f"[DEBUG] Vision token IDs to check: {vision_token_ids}")
            
            # Extract single sample
            sample_ids = input_ids[sample_idx]
            sample_mask = attention_mask[sample_idx]
            
            # Filter out padding tokens
            valid_tokens = sample_ids[sample_mask.bool()].tolist()
            
            # Debug: show unique token IDs in the input
            unique_tokens = set(valid_tokens)
            logging.info(f"[DEBUG] Total unique token IDs in input: {len(unique_tokens)}")
            logging.info(f"[DEBUG] Token ID range: {min(valid_tokens)} to {max(valid_tokens)}")
            
            # Check for vision tokens
            vision_tokens_present = set(valid_tokens) & set(vision_token_ids)
            has_vision = len(vision_tokens_present) > 0
            
            # Debug: check if any tokens are in the vision range
            vision_range_tokens = [t for t in valid_tokens if 151650 <= t <= 151660]
            if vision_range_tokens:
                logging.info(f"[DEBUG] Found tokens in vision range (151650-151660): {set(vision_range_tokens)}")
            
            # Log
            logging.info("=" * 80)
            logging.info(f"[MODEL.GENERATE INPUT] Sample {sample_idx}")
            logging.info(f"Input length: {len(valid_tokens)} tokens")
            logging.info(f"Batch shape: {input_ids.shape}")
            
            if has_vision:
                logging.info(f"✓ Contains vision tokens: {vision_tokens_present}")
            else:
                logging.info("ℹ️  No vision tokens (text-only)")
            
            # logging.info("-" * 80)
            # logging.info("[TOKENS TO MODEL]")
            # logging.info(f"Decoded: {self.tokenizer.decode(valid_tokens, skip_special_tokens=False)}")
            # logging.info("=" * 80)
        except Exception as e:
            logging.warning(f"Failed to log generation input: {e}")
    
    def run_agent_loop(self, gen_batch: InteractionDataProto) -> InteractionDataProto:
        
        initial_input_ids = gen_batch.batch["input_ids"]
        has_pixels = "pixel_values" in gen_batch.batch
        # Do NOT truncate prompts when vision inputs are present to preserve image tokens
        if has_pixels:
            original_left_side = {'input_ids': initial_input_ids}
        else:
            original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}

        # postprocess model inputs
        rollings = gen_batch
        if has_pixels:
            # Keep full sequence to maintain alignment between image features and tokens
            rollings_active = {k: v for k, v in rollings.batch.items()}
        else:
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask']    
            )
            rollings_active = {k: v for k, v in rollings.batch.items()}  

        # Log tokens before actual model generation
        self._log_generation_input(rollings_active["input_ids"], rollings_active["attention_mask"])

        # Log image paths if present
        if "image_paths" in gen_batch.no_tensor_batch:
            image_paths = gen_batch.no_tensor_batch["image_paths"]
            if any(path is not None for path in image_paths):
                logging.info(f"[Image Paths in Batch] {image_paths}")

        # model generation (pass image tensors if available)
        gen_kwargs = {
            "input_ids": rollings_active["input_ids"],
            "attention_mask": rollings_active["attention_mask"],
            "generation_config": self.generation_config,
        }
        if "pixel_values" in rollings_active:
            gen_kwargs["pixel_values"] = rollings_active["pixel_values"]
        if "image_grid_thw" in rollings_active:
            gen_kwargs["image_grid_thw"] = rollings_active["image_grid_thw"]

        # Log shapes right before generate
        logging.info("[Final Shapes before model.generate]")
        logging.info(f"  input_ids: {gen_kwargs['input_ids'].shape}")
        logging.info(f"  attention_mask: {gen_kwargs['attention_mask'].shape}")
        if "pixel_values" in gen_kwargs:
            logging.info(f"  pixel_values: {gen_kwargs['pixel_values'].shape}")
        if "image_grid_thw" in gen_kwargs:
            logging.info(f"  image_grid_thw: {gen_kwargs['image_grid_thw'].shape}")

        # Check if we should return augmentation mask (for training)
        should_return_aug_mask = hasattr(self.actor_rollout_wg, 'weaver') and hasattr(self, '_return_augmentation_mask') and self._return_augmentation_mask
        
        if should_return_aug_mask:
            gen_kwargs["return_augmentation_mask"] = True
            gen_output, augmentation_pos = self.actor_rollout_wg.generate(**gen_kwargs)
        else:
            gen_output = self.actor_rollout_wg.generate(**gen_kwargs)
            augmentation_pos = None
        
        responses_ids = gen_output[:, rollings_active["input_ids"].size(1):]
        # Prefer chat_eos_token_id (<|im_end|>) if set, otherwise fallback to tokenizer.eos_token_id
        eos_id = getattr(self, "chat_eos_token_id", self.tokenizer.eos_token_id)
        responses_ids = self.tensor_fn.erase_after_first_eos(responses_ids, eos_id)
        
        # update right side
        original_right_side = self._update_right_side(original_right_side, responses_ids, next_obs_ids=None)
        
        # Store augmentation positions if available
        if augmentation_pos is not None:
            original_right_side['augmentation_pos'] = augmentation_pos
        
        # construct final output
        return self._compose_final_output(original_left_side, original_right_side)
    
    def _compose_final_output(
        self, left_side: Dict,
        right_side: Dict,
    ) -> InteractionDataProto:
        """Compose final generation output."""
        
        final_output_batch = right_side.copy()
        final_output_batch['prompts'] = left_side['input_ids']
        final_output_batch["responses"] = right_side['responses']
        
        # Combine input IDs: input_ids + responses
        final_output_batch['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)

        # Create attention mask
        final_output_batch['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output_batch['responses'])
        ], dim=1)
        
        final_output_batch['info_mask'] = torch.cat([  
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output_batch['responses_with_info_mask'])
        ], dim=1)
        
        final_output = InteractionDataProto(batch=final_output_batch)

        return final_output
        