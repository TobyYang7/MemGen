import re
import torch
import os
import json
import logging
from typing import List, Dict

# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)

def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])

def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def init_grpo_log_files(output_dir: str) -> tuple[str, str]:
    """
    Initialize GRPO log files (human-readable txt and machine-readable jsonl).

    Returns the tuple of (txt_log_path, jsonl_log_path).
    """
    grpo_log_file = os.path.join(output_dir, "../logs/grpo_logs.txt")
    grpo_jsonl_file = os.path.join(output_dir, "../logs/grpo_samples.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(grpo_log_file), exist_ok=True)

    # Create/clear the log file
    with open(grpo_log_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("GRPO Training Logs - WeaverGRPOTrainer\n")
        f.write("=" * 80 + "\n\n")

    # Create/clear the JSONL file
    with open(grpo_jsonl_file, "w", encoding="utf-8"):
        pass

    return grpo_log_file, grpo_jsonl_file


def log_prompt_truncation(
    prompts_before: torch.Tensor,
    prompts_after: torch.Tensor,
    prompt_mask_before: torch.Tensor,
    prompt_mask_after: torch.Tensor,
    processing_class,
    max_prompt_length: int,
    sample_idx: int = 0
) -> None:
    """
    Log prompt before and after truncation in token format.
    Also checks if image/vision tokens were truncated.
    
    Args:
        prompts_before: Prompt token IDs before truncation [batch_size, seq_len_before]
        prompts_after: Prompt token IDs after truncation [batch_size, seq_len_after]
        prompt_mask_before: Attention mask before truncation
        prompt_mask_after: Attention mask after truncation
        processing_class: Tokenizer or processor for decoding
        max_prompt_length: Maximum prompt length configured
        sample_idx: Index of sample to log (default: 0, first sample in batch)
    """
    # Get tokenizer
    _tok = getattr(processing_class, "tokenizer", processing_class)
    
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
            token_id = _tok.encode(token_name, add_special_tokens=False)
            if isinstance(token_id, list) and len(token_id) > 0:
                if token_id[0] not in vision_token_ids:
                    vision_token_ids.append(token_id[0])
        except Exception:
            pass
    
    # Extract single sample
    prompt_before = prompts_before[sample_idx]
    prompt_after = prompts_after[sample_idx]
    mask_before = prompt_mask_before[sample_idx]
    mask_after = prompt_mask_after[sample_idx]
    
    # Filter out padding tokens (where mask == 0)
    valid_tokens_before = prompt_before[mask_before.bool()].tolist()
    valid_tokens_after = prompt_after[mask_after.bool()].tolist()
    
    # Check if vision tokens were truncated
    vision_tokens_before = set(valid_tokens_before) & set(vision_token_ids)
    vision_tokens_after = set(valid_tokens_after) & set(vision_token_ids)
    vision_tokens_lost = vision_tokens_before - vision_tokens_after
    has_vision_loss = len(vision_tokens_lost) > 0
    
    # Convert token IDs to readable format with special tokens
    def tokens_to_readable(token_ids):
        """Convert token IDs to readable string with special tokens visible."""
        # ANSI escape codes for colors
        GREEN = "\033[92m"
        RESET = "\033[0m"
        
        tokens = []
        prev_tid = None
        consecutive_count = 0
        
        for tid in token_ids:
            try:
                # Decode single token
                token_str = _tok.decode([tid], skip_special_tokens=False)
                
                # Check if this is image_pad (151655) or other vision pad tokens
                is_image_pad = tid == 151655 or (tid in vision_token_ids and 'pad' in token_str.lower())
                
                # If consecutive image_pad tokens, just count them
                if is_image_pad and prev_tid == tid:
                    consecutive_count += 1
                    continue
                else:
                    # Output the previous consecutive tokens if any
                    if consecutive_count > 0 and prev_tid is not None:
                        prev_str = _tok.decode([prev_tid], skip_special_tokens=False)
                        tokens.append(f"{GREEN}[IMG]{prev_str.strip()}[/IMG]{RESET}×{consecutive_count + 1}")
                        consecutive_count = 0
                    
                    # Highlight vision tokens
                    if tid in vision_token_ids:
                        if is_image_pad:
                            prev_tid = tid
                            consecutive_count = 0
                            continue  # Will be added in next iteration or at the end
                        else:
                            tokens.append(f"{GREEN}[IMG]{token_str.strip()}[/IMG]{RESET}")
                    # Show special tokens
                    elif tid == _tok.pad_token_id:
                        tokens.append(f"<|pad|>")
                    elif tid == _tok.eos_token_id:
                        tokens.append(f"<|eos|>")
                    elif tid == _tok.bos_token_id:
                        tokens.append(f"<|bos|>")
                    elif token_str.strip() in ["<|im_start|>", "<|im_end|>", "<|im_sep|>"]:
                        tokens.append(token_str.strip())
                    else:
                        tokens.append(f"[{tid}:{repr(token_str)}]")
                    
                    prev_tid = tid
            except Exception:
                tokens.append(f"[{tid}:?]")
                prev_tid = tid
        
        # Handle any remaining consecutive tokens at the end
        if consecutive_count > 0 and prev_tid is not None:
            try:
                prev_str = _tok.decode([prev_tid], skip_special_tokens=False)
                tokens.append(f"{GREEN}[IMG]{prev_str.strip()}[/IMG]{RESET}×{consecutive_count + 1}")
            except Exception:
                pass
        
        return " ".join(tokens)
    
    # Log information
    logging.info("=" * 80)
    logging.info(f"[PROMPT TRUNCATION] Sample {sample_idx}")
    logging.info(f"Length before truncation: {len(valid_tokens_before)}")
    logging.info(f"Length after truncation: {len(valid_tokens_after)}")
    logging.info(f"Max prompt length: {max_prompt_length}")
    logging.info(f"Tokens truncated: {len(valid_tokens_before) - len(valid_tokens_after)}")
    
    # Warn if vision tokens were lost
    if has_vision_loss:
        logging.warning("⚠️  WARNING: IMAGE/VISION TOKENS WERE TRUNCATED!")
        logging.warning(f"⚠️  Lost vision token IDs: {vision_tokens_lost}")
        logging.warning(f"⚠️  Vision tokens before: {vision_tokens_before}")
        logging.warning(f"⚠️  Vision tokens after: {vision_tokens_after}")
        logging.warning("⚠️  The model will NOT see the image information!")
    elif len(vision_tokens_before) > 0:
        logging.info(f"✓ Vision tokens preserved: {vision_tokens_before}")
    
    logging.info("-" * 80)
    
    # Log tokens before truncation
    logging.info("[BEFORE TRUNCATION]")
    tokens_before_str = tokens_to_readable(valid_tokens_before)
    logging.info(f"Tokens: {tokens_before_str}")
    # logging.info(f"Decoded text: {_tok.decode(valid_tokens_before, skip_special_tokens=False)}")
    logging.info("-" * 80)
    
    # Log tokens after truncation
    logging.info("[AFTER TRUNCATION]")
    tokens_after_str = tokens_to_readable(valid_tokens_after)
    logging.info(f"Tokens: {tokens_after_str}")
    # logging.info(f"Decoded text: {_tok.decode(valid_tokens_after, skip_special_tokens=False)}")
    logging.info("=" * 80)


def log_rollout_input(
    prompts: torch.Tensor,
    prompt_mask: torch.Tensor,
    processing_class,
    sample_idx: int = 0
) -> None:
    """
    Log the input tokens before model generation (rollout).
    
    Args:
        prompts: Prompt token IDs [batch_size, seq_len]
        prompt_mask: Attention mask [batch_size, seq_len]
        processing_class: Tokenizer or processor for decoding
        sample_idx: Index of sample to log (default: 0, first sample in batch)
    """
    # Get tokenizer
    _tok = getattr(processing_class, "tokenizer", processing_class)
    
    # Check for vision/image tokens
    vision_token_names = ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>", "<|vision_pad|>"]
    vision_token_ids = []
    for token_name in vision_token_names:
        try:
            token_id = _tok.encode(token_name, add_special_tokens=False)
            if isinstance(token_id, list) and len(token_id) > 0:
                vision_token_ids.append(token_id[0])
        except Exception:
            pass
    
    # Extract single sample
    prompt = prompts[sample_idx]
    mask = prompt_mask[sample_idx]
    
    # Filter out padding tokens
    valid_tokens = prompt[mask.bool()].tolist()
    
    # Check for vision tokens
    vision_tokens_present = set(valid_tokens) & set(vision_token_ids)
    has_vision = len(vision_tokens_present) > 0
    
    # Convert token IDs to readable format
    def tokens_to_readable(token_ids):
        """Convert token IDs to readable string with special tokens visible."""
        # ANSI escape codes for colors
        GREEN = "\033[92m"
        RESET = "\033[0m"
        
        tokens = []
        for tid in token_ids:
            try:
                token_str = _tok.decode([tid], skip_special_tokens=False)
                # Highlight vision tokens
                if tid in vision_token_ids:
                    tokens.append(f"{GREEN}[IMG]{token_str.strip()}[/IMG]{RESET}")
                # Show special tokens
                elif tid == _tok.pad_token_id:
                    tokens.append(f"<|pad|>")
                elif tid == _tok.eos_token_id:
                    tokens.append(f"<|eos|>")
                elif tid == _tok.bos_token_id:
                    tokens.append(f"<|bos|>")
                elif token_str.strip() in ["<|im_start|>", "<|im_end|>", "<|im_sep|>"]:
                    tokens.append(token_str.strip())
                else:
                    tokens.append(f"[{tid}:{repr(token_str)}]")
            except Exception:
                tokens.append(f"[{tid}:?]")
        return " ".join(tokens)
    
    # Log information
    logging.info("=" * 80)
    logging.info(f"[ROLLOUT INPUT] Sample {sample_idx}")
    logging.info(f"Prompt length: {len(valid_tokens)} tokens")
    logging.info(f"Batch shape: {prompts.shape}")
    
    if has_vision:
        logging.info(f"✓ Contains vision tokens: {vision_tokens_present}")
    else:
        logging.info("ℹ️  No vision tokens detected (text-only prompt)")
    
    logging.info("-" * 80)
    
    # Log tokens
    logging.info("[INPUT TOKENS]")
    tokens_str = tokens_to_readable(valid_tokens)
    logging.info(f"Tokens: {tokens_str}")
    logging.info(f"Decoded text: {_tok.decode(valid_tokens, skip_special_tokens=False)}")
    logging.info("=" * 80)
    

def extract_answer(text: str) -> str:
    """
    Extract the FINAL CHOICE from a solution.
    Priority:
      1) Last occurrence inside \boxed{...} (supports nested braces)
      2) <answer>...</answer> fallback
      3) Raw text (fallback)
    Cleans common LaTeX wrappers like \text{...}, \displaystyle, and surrounding $ ... $.
    """
    try:
        s = text

        # -------- 1) Gather all contents inside \boxed{...} with a small brace parser --------
        boxed_contents = []

        # find all occurrences of "\boxed" followed by optional spaces then "{"
        for m in re.finditer(r'\\boxed\s*\{', s):
            # position of the opening brace "{"
            open_brace_pos = s.find('{', m.end() - 1)
            if open_brace_pos == -1:
                continue

            # brace matching to find the corresponding closing brace
            depth = 0
            i = open_brace_pos
            while i < len(s):
                ch = s[i]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        # extract inside { ... }
                        boxed = s[open_brace_pos + 1:i]

                        # light cleaning: strip spaces and surrounding $ ... $
                        boxed = boxed.strip()
                        boxed = boxed.strip('$')

                        # remove \displaystyle and similar display-style macros at the front
                        boxed = re.sub(r'\\displaystyle\s*', '', boxed)

                        # unwrap \text{...} if the whole thing is a single \text{...}
                        def unwrap_text_env(x: str) -> str:
                            x_strip = x.strip()
                            if x_strip.startswith(r'\text{') and x_strip.endswith('}'):
                                # naive unwrap of a single-level \text{...}
                                inner = x_strip[len(r'\text{'):-1].strip()
                                return inner
                            return x
                        boxed = unwrap_text_env(boxed)

                        # collapse whitespace
                        boxed = re.sub(r'\s+', ' ', boxed).strip()

                        if boxed:
                            boxed_contents.append(boxed)
                        break
                i += 1

        # If we found any \boxed content, return the last one (FINAL CHOICE)
        if boxed_contents:
            return boxed_contents[-1]

        # -------- 2) Fallback: <answer>...</answer> --------
        low = s.lower()
        start = low.find("<answer>")
        end = low.find("</answer>")
        if start != -1 and end != -1 and end > start:
            ans = s[start + len("<answer>"):end].strip()
            ans = ans.strip('$')
            ans = re.sub(r'\\displaystyle\s*', '', ans)
            ans = re.sub(r'\s+', ' ', ans).strip()
            return ans if ans else s

    except Exception:
        # fall through to raw text fallback
        pass

    # -------- 3) Fallback: return original text ----------
    return text


def persist_grpo_logs(
    log_file: str,
    jsonl_file: str,
    step: int,
    mode: str,
    prompt_texts: list[str],
    completion_texts: list[str],
    rewards: list[float],
    rewards_by_func: dict[str, list[float]],
    token_counts: list[int],
    ground_truths: list[str] | None,
    solutions_extracted: list[str] | None,
    # verifies: list[bool] | None,
    reward_func_names: list[str],
    stop_reasons: list[str] | None = None,
    image_paths: list[str] | None = None,
) -> None:
    """
    Append per-sample human-readable and JSONL logs for GRPO.
    """
    try:
        # Flatten possibly nested lists (from distributed gather)
        def _flatten(lst):
            if isinstance(lst, list) and len(lst) > 0 and isinstance(lst[0], list):
                return [item for sub in lst for item in sub]
            return lst

        prompt_texts = _flatten(prompt_texts)
        completion_texts = _flatten(completion_texts)
        rewards = _flatten(rewards)
        token_counts = _flatten(token_counts)
        rewards_by_func = {k: _flatten(v) for k, v in rewards_by_func.items()}
        stop_reasons = _flatten(stop_reasons) if stop_reasons is not None else None
        ground_truths = _flatten(ground_truths) if ground_truths is not None else None
        solutions_extracted = _flatten(solutions_extracted) if solutions_extracted is not None else None
        # verifies = _flatten(verifies) if verifies is not None else None
        image_paths = _flatten(image_paths) if image_paths is not None else None

        # Guard against length mismatches
        n = min(
            len(prompt_texts),
            len(completion_texts),
            len(rewards),
            len(token_counts),
            *[len(rewards_by_func[name]) for name in reward_func_names],
            *( [len(ground_truths)] if ground_truths is not None else [] ),
            *( [len(solutions_extracted)] if solutions_extracted is not None else [] ),
            # *( [len(verifies)] if verifies is not None else [] ),
            *( [len(stop_reasons)] if stop_reasons is not None else [] ),
            *( [len(image_paths)] if image_paths is not None else [] ),
        )
        if n == 0:
            return

        with open(log_file, "a", encoding="utf-8") as f_txt:
            f_txt.write(f"\n{'='*80}\n")
            f_txt.write(f"Step: {step} | Mode: {mode}\n")
            f_txt.write(f"{'='*80}\n")
            for idx in range(n):
                p_txt = prompt_texts[idx]
                c_txt = completion_texts[idx]
                r_total = rewards[idx]
                f_txt.write(f"\n[Sample {idx}]\n")
                if image_paths is not None:
                    f_txt.write(f"Image path: {image_paths[idx]}\n")
                f_txt.write(f"Prompt: {p_txt}\n")
                comp_str = ", ".join([f"{name}: {float(rewards_by_func[name][idx]):.6f}" for name in reward_func_names])
                f_txt.write(f"Reward: {float(r_total):.6f} | Components: {comp_str}\n")
                if ground_truths is not None:
                    f_txt.write(f"Ground truth: {ground_truths[idx]}\n")
                if solutions_extracted is not None:
                    f_txt.write(f"Solution: {solutions_extracted[idx]}\n")
                # if verifies is not None:
                #     f_txt.write(f"Verify: {bool(verifies[idx])}\n")
                s_reason = (
                    stop_reasons[idx]
                    if stop_reasons is not None and idx < len(stop_reasons)
                    else "unknown"
                )
                f_txt.write(f"Stop reason: {s_reason}\n")
                # Always place completion last in the per-sample block
                f_txt.write(f"Completion: {c_txt}\n")
                f_txt.write(f"{'-'*80}\n")

        with open(jsonl_file, "a", encoding="utf-8") as f_jsonl:
            for idx in range(n):
                s_reason = (
                    stop_reasons[idx]
                    if stop_reasons is not None and idx < len(stop_reasons)
                    else "unknown"
                )
                record = {
                    "reward": float(rewards[idx]),
                    "token_count": int(token_counts[idx]),
                    # "step": int(step),
                    # "mode": mode,
                    # "sample_index": int(idx),
                    "stop_reason": s_reason,
                }
                if ground_truths is not None:
                    record["ground_truth"] = ground_truths[idx]
                if solutions_extracted is not None:
                    record["solution"] = solutions_extracted[idx]
                # if verifies is not None:
                #     record["verify"] = bool(verifies[idx])
                # Add image_path before completion
                if image_paths is not None:
                    record["image_path"] = image_paths[idx]
                # Add completion and prompt at the end
                record["completion"] = completion_texts[idx]
                record["prompt"] = prompt_texts[idx]
                f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.warning(f"Failed to persist GRPO logs: {e}")