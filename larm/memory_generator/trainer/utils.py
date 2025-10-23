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
    reward_func_names: list[str],
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

        # Guard against length mismatches
        n = min(
            len(prompt_texts),
            len(completion_texts),
            len(rewards),
            len(token_counts),
            *[len(rewards_by_func[name]) for name in reward_func_names]
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
                f_txt.write(f"Prompt: {p_txt}\n")
                f_txt.write(f"Completion: {c_txt}\n")
                comp_str = ", ".join([f"{name}: {float(rewards_by_func[name][idx]):.6f}" for name in reward_func_names])
                f_txt.write(f"Reward: {float(r_total):.6f} | Components: {comp_str}\n")
                f_txt.write(f"{'-'*80}\n")

        with open(jsonl_file, "a", encoding="utf-8") as f_jsonl:
            for idx in range(n):
                record = {
                    "reward": float(rewards[idx]),
                    "token_count": int(token_counts[idx]),
                    "step": int(step),
                    "mode": mode,
                    "sample_index": int(idx),
                    "completion": completion_texts[idx],
                }
                f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.warning(f"Failed to persist GRPO logs: {e}")