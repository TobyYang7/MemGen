import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import GenerationConfig

from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List
import os
import logging
import json

# ===== chat template =====

# THINK_SYS_PROMPT = 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>'

THINK_SYS_PROMPT = 'You are a helpful assistant.'

# Qwen2.5-VL chat template with vision support
CONVERSATION_TEMPLATE = r"""{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif %}<|im_start|>{{ message['role'] }}
{% if message['content'] is string %}{{ message['content'] }}<|im_end|>
{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""

# ===== torch part =====

def get_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Calculate the entropy of token probability distributions.
    
    Entropy measures the uncertainty in the model's predictions:
    - High entropy: model is uncertain (probability is spread across many tokens)
    - Low entropy: model is confident (probability is concentrated on few tokens)
    
    Args:
        logits (torch.Tensor): 
            Logits tensor of shape (batch_size, seq_len, vocab_size) or (batch_size, vocab_size).
            Raw model outputs before softmax.
    
    Returns:
        torch.Tensor: 
            Entropy values of shape (batch_size, seq_len) or (batch_size,).
            Higher values indicate more uncertainty.
    
    Example:
        >>> logits = torch.randn(2, 10, 50000)  # batch=2, seq=10, vocab=50000
        >>> entropy = get_entropy(logits)  # shape: (2, 10)
        >>> # High entropy (e.g., 5.0) means uncertain, low entropy (e.g., 0.1) means confident
    """
    with torch.no_grad():
        # Compute probabilities using softmax
        probs = F.softmax(logits, dim=-1)
        
        # Compute entropy: -sum(p * log(p))
        # Handle numerical stability: log(0) = nan, so we mask these values
        log_probs = torch.log(probs)
        entropy_values = probs * log_probs
        
        # Replace NaN values (from log(0)) with zeros
        entropy_values = torch.where(
            ~torch.isnan(entropy_values),
            entropy_values,
            torch.zeros_like(entropy_values)
        )
        
        # Sum over vocabulary dimension
        entropy = -torch.sum(entropy_values, dim=-1)
        
    return entropy


def load_state_dict_from_safetensor(model_path) -> Dict:
    """Load a safetensor file from the given path and return a state_dict.

    Args:
        model_path (str): Path to the safetensor file.

    Returns:
        Dict[str, torch.Tensor]: A dictionary of model parameters, 
        where keys are parameter names and values are corresponding tensors.
    """
    model_state_dict = {}
    with safe_open(model_path, framework="pt") as f:
        for key in f.keys():
            model_state_dict[key] = f.get_tensor(key)
    return model_state_dict

def fix_model_parameters(model: nn.Module):
    """Freeze all parameters of the given model.

    Args:
        model (nn.Module): The PyTorch model whose parameters will be frozen.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

def open_model_parameters(model: nn.Module):
    """Unfreeze all parameters of the given model.

    Args:
        model (nn.Module): The PyTorch model whose parameters will be unfrozen.
    """
    for parameter in model.parameters():
        parameter.requires_grad = True

def log_trainable_params(model: nn.Module):
    """Log all trainable parameters of the given model.

    Args:
        model (nn.Module): The PyTorch model to inspect.
    """
    logging.info("Trainable parameters in the model:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.info(f"  {name}: {param.numel()} params, shape={param.shape}")



# ===== Eval Part =====
@dataclass
class EvalConfig:
    output_dir: str = None
    batch_size: int = 1
    generation_config: GenerationConfig = None

@dataclass
class StaticEvalRecorder:
    compute_metrics: List[Callable[[str, str, str], float]] = field(default_factory=list)
    log_file: Optional[str] = None
    writer: Optional[object] = None

    # Internal storage
    metric_sums: Dict[str, float] = field(init=False)
    metric_counts: Dict[str, int] = field(init=False)

    def __post_init__(self):
        self.metric_sums = {metric.__name__: 0.0 for metric in self.compute_metrics}
        self.metric_counts = {metric.__name__: 0 for metric in self.compute_metrics}
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, 'w') as f:
                f.write('')  # Clear file

    def record_batch(self, completions: List[str], examples: List[Dict]):
        """Record results for a batch of model outputs.

        Args:
            completions (List[str]): The model's answers (outputs).
            examples (List[Dict]): Each completion's corresponding question and related attributes.
                Each example is expected to contain the keys: "prompt" and "solution".
        """
        # Extract all keys from the first example
        keys = [key for key in examples[0]]
        # Build kwargs for metrics computation (one list per field)
        reward_kwargs = {key: [example[key] for example in examples] for key in keys}
        reward_kwargs['completions'] = completions
        
        # Compute all metrics in batch
        batched_results = {}
        for metric in self.compute_metrics:  # iterate over each metric function
            metric_name = metric.__name__   # use function name as metric name
            batched_scores = metric(**reward_kwargs)  # compute scores for the entire batch
            batched_results[metric_name] = batched_scores
        
        # Record experiment results for each example
        for i, (completion, example) in enumerate(zip(completions, examples)):
            # Collect the metric results for this specific example
            metrics_result = {
                metric_name: batched_results[metric_name][i]
                for metric_name in batched_results
            }

            # Update running totals for metrics
            for metric_name, score in metrics_result.items():
                self.metric_sums[metric_name] += score
                self.metric_counts[metric_name] += 1
            
            # Create a log record with prompt, solution, completion, and metrics
            prompt = example.get("prompt", "")
            solution = example.get("solution", "")
            record = {
                'prompt': prompt,
                'solution': solution,
                'completion': completion,
                'metrics': metrics_result
            }

            # Write the record into a log file (if available)
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            # Update TensorBoard metrics (if writer is available)
            if self.writer:
                mean_metrics = self.get_mean_metrics()  # get average metrics across all data so far
                for name, value in mean_metrics.items():
                    self.writer.add_scalar(name, value, global_step=self.metric_counts[name])


    def get_mean_metrics(self) -> Dict[str, float]:
        return {
            name: (self.metric_sums[name] / self.metric_counts[name]) if self.metric_counts[name] > 0 else 0.0
            for name in self.metric_sums
        }

    def finalize(self):
        mean_metrics = self.get_mean_metrics()
        final_record = {
            'summary_metrics': mean_metrics
        }

        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(final_record, ensure_ascii=False) + '\n')

        if self.writer:
            mean_metrics = self.get_mean_metrics()
            for name, value in mean_metrics.items():
                self.writer.add_scalar(name + "_final", value, global_step=self.metric_counts[name])


@dataclass
class DynamicEvalRecorder:
    log_file: Optional[str] = None  # path to the txt log file
    writer: object = field(default=None)  # TensorBoard SummaryWriter

    def __post_init__(self):
        if self.log_file is None:
            raise ValueError("log_file path must be provided")

        # Ensure the directory for the log file exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.logger = logging.getLogger("DynamicEvalRecorder")

        # Internal counters
        self._total_reward = 0.0
        self._count = 0

        # Initialize the file (clear previous content if any)
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("DynamicEvalRecorder Log\n\n")

    def record_batch(self, conversations: List[str], rewards: List[float]):
        """Record a batch of conversations and their associated rewards.

        Args:
            conversations (List[str]): List of conversation texts.
            rewards (List[float]): List of reward values corresponding to conversations.
        """
        if len(conversations) != len(rewards):
            raise ValueError("conversations and rewards must have the same length")

        # Append batch results to the log file
        with open(self.log_file, "a", encoding="utf-8") as f:
            for conv, rew in zip(conversations, rewards):
                f.write(f"Conversation:\n{conv}\n")
                f.write(f"Reward: {rew:.4f}\n")
                f.write("-" * 40 + "\n")

                # Update statistics
                self._total_reward += rew
                self._count += 1

        # Compute running average reward
        avg_reward = self._total_reward / self._count if self._count > 0 else 0.0

        # Write running average to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar("reward/avg", avg_reward, self._count)

        # Log summary info
        self.logger.info(f"Recorded {len(conversations)} items, avg_reward={avg_reward:.4f}")

    def finalize(self):
        """Finalize evaluation: write final average reward to both log file and TensorBoard."""
        # Compute final average reward
        avg_reward = self._total_reward / self._count if self._count > 0 else 0.0

        # Append final result to log file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\nFinal Results\n")
            f.write("=" * 40 + "\n")
            f.write(f"Average Reward: {avg_reward:.4f}\n")

        # Write final result to TensorBoard
        if self.writer:
            self.writer.add_scalar("ave_reward_final", avg_reward, global_step=self._count)

