"""
Thinking-GRPO Dataset Preprocessing Script

This script loads and preprocesses the VLAA Thinking-GRPO dataset.
The preprocessed data will be saved to the data/thinking_grpo directory with
fields: prompt, completion, solution, image_path

Usage:
    # Download dataset
    hf download UCSC-VLAA/VLAA-Thinking --repo-type dataset --local-dir dataset/thinking_grpo

    # Preprocess dataset
    uv run data/thinking_grpo_process.py --output_dir data/thinking_grpo --dataset_root dataset/thinking_grpo --images_dir dataset/thinking_grpo/images --json_path dataset/thinking_grpo/VLAA-Thinking-GRPO-25K.json
"""

import os
import re
import json
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple

from datasets import DatasetDict, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_thinking_grpo_data(json_path: str) -> List[Dict[str, Any]]:
    """Load Thinking-GRPO data from JSON file.

    The source JSON may be minified and extremely long on a single line.

    Args:
        json_path: Path to VLAA-Thinking-GRPO-25K.json

    Returns:
        List of raw data samples
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    logging.info(f"Loading Thinking-GRPO data from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            # Try standard JSON (array)
            data = json.load(f)
        except json.JSONDecodeError as e:
            # Fallback to JSONL parsing (one JSON object per line)
            logging.info(f"Standard JSON parse failed ({e}). Trying JSONL format...")
            f.seek(0)
            data = []
            for idx, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                    # Some lines may contain an array; extend in that case
                    if isinstance(obj, list):
                        data.extend(obj)
                    else:
                        data.append(obj)
                except json.JSONDecodeError as e2:
                    logging.warning(f"Skipping invalid JSONL line {idx}: {e2}")

    if not isinstance(data, list):
        raise ValueError("Expected a top-level list in the JSON file")

    logging.info(f"Loaded {len(data)} raw samples from Thinking-GRPO JSON")
    return data


def split_dataset(data: List[Dict[str, Any]], train_ratio: float = 0.98, test_ratio: float = 0.01, val_ratio: float = 0.01) -> DatasetDict:
    """Split dataset into train/valid/test.

    Args:
        data: List of raw examples
        train_ratio: Training set ratio
        test_ratio: Test set ratio
        val_ratio: Validation set ratio

    Returns:
        DatasetDict with train/valid/test splits
    """
    if abs(train_ratio + test_ratio + val_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + test_ratio + val_ratio}")

    import random
    rnd = list(range(len(data)))
    random = __import__('random')
    random.Random(42).shuffle(rnd)

    n_total = len(data)
    n_test = max(1, int(n_total * test_ratio)) if test_ratio > 0 else 0
    n_val = max(1, int(n_total * val_ratio)) if val_ratio > 0 else 0
    n_train = n_total - n_test - n_val

    idx_test = rnd[:n_test] if n_test > 0 else []
    idx_val = rnd[n_test:n_test + n_val] if n_val > 0 else []
    idx_train = rnd[n_test + n_val:]

    dataset_dict = DatasetDict()
    if n_train > 0:
        dataset_dict["train"] = Dataset.from_dict({"idx": idx_train})
    if n_val > 0:
        dataset_dict["valid"] = Dataset.from_dict({"idx": idx_val})
    if n_test > 0:
        dataset_dict["test"] = Dataset.from_dict({"idx": idx_test})

    logging.info(f"Split sizes - train: {len(dataset_dict.get('train', []))}, valid: {len(dataset_dict.get('valid', []))}, test: {len(dataset_dict.get('test', []))}")
    return dataset_dict


def count_verifier_types(samples: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count occurrences of verifier_type values in given samples."""
    counts: Dict[str, int] = {}
    for s in samples:
        vt = s.get('verifier_type', 'unknown')
        if not isinstance(vt, str):
            vt = str(vt)
        counts[vt] = counts.get(vt, 0) + 1
    return counts


def _extract_between_tags(text: str, tag: str) -> str:
    """Extract content between <tag>...</tag> from text, return first match or empty string."""
    if not text:
        return ""
    pattern = rf"<{tag}>([\s\S]*?)</{tag}>"
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _extract_last_boxed(text: str) -> str:
    """Extract the last \\boxed{...} content with balanced braces; return the content without braces."""
    if not text:
        return ""
    pattern = r"\\\\?boxed\{"  # matches \boxed{ or \\boxed{
    matches = list(re.finditer(pattern, text))
    if not matches:
        return ""

    def _extract_balanced_braces(s: str, start_pos: int) -> Optional[str]:
        if start_pos >= len(s) or s[start_pos] != '{':
            return None
        depth = 0
        i = start_pos
        while i < len(s):
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                depth -= 1
                if depth == 0:
                    return s[start_pos + 1:i]
            i += 1
        return None

    for m in reversed(matches):
        pos = m.end() - 1
        content = _extract_balanced_braces(text, pos)
        if content is not None:
            return content.strip()
    return ""


def _format_solution(answer_text: str) -> str:
    ans = (answer_text or "").strip()
    if not ans:
        return ""
    if ans.startswith("\\boxed{") and ans.endswith("}"):
        return ans
    return "\\boxed{" + ans + "}"


def _guess_prompt(sample: Dict[str, Any]) -> str:
    """Try to derive the prompt from common keys."""
    candidates = [
        'prompt', 'question', 'instruction', 'query', 'input', 'task', 'text'
    ]
    for key in candidates:
        if key in sample and isinstance(sample[key], str) and len(sample[key].strip()) > 0:
            return sample[key].strip()

    # Some datasets use chat-style messages
    # Try to reconstruct a prompt from messages if present
    msgs = sample.get('messages') or sample.get('conversations')
    if isinstance(msgs, list) and len(msgs) > 0:
        try:
            parts: List[str] = []
            for msg in msgs:
                role = (msg.get('role') or '').strip()
                content = msg.get('content')
                if isinstance(content, str):
                    parts.append(content.strip())
                elif isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and 'text' in c and isinstance(c['text'], str):
                            parts.append(c['text'].strip())
            if parts:
                return "\n".join(parts)
        except Exception:
            pass
    return ""


def _guess_completion_and_solution(sample: Dict[str, Any]) -> Tuple[str, str]:
    """Derive completion (full response) and solution (short answer) from sample.

    Strategy:
    - Prefer explicit fields if present (e.g., 'completion', 'response', 'output').
    - Attempt to extract <answer>...</answer> from completion.
    - Fallback to explicit 'answer'/'solution' keys.
    - Fallback to last \boxed{...} extraction from completion or long solution.
    """
    # Prefer explicit completion-like fields
    completion_candidates = ['completion', 'response', 'output', 'assistant_answer', 'assistant', 'generated']
    completion_text = ""
    for key in completion_candidates:
        v = sample.get(key)
        if isinstance(v, str) and len(v.strip()) > 0:
            completion_text = v.strip()
            break

    # If chat messages include assistant content, try to reconstruct
    if not completion_text:
        msgs = sample.get('messages') or sample.get('conversations')
        if isinstance(msgs, list):
            for msg in reversed(msgs):
                if isinstance(msg, dict) and (msg.get('role') or '').strip().lower() in ("assistant", "assistant_answer"):
                    content = msg.get('content')
                    if isinstance(content, str) and content.strip():
                        completion_text = content.strip()
                        break
                    if isinstance(content, list):
                        texts = []
                        for c in content:
                            if isinstance(c, dict) and 'text' in c and isinstance(c['text'], str):
                                texts.append(c['text'].strip())
                        if texts:
                            completion_text = "\n".join(texts)
                            break

    # Prefer explicit ground-truth short answers when available
    short_answer = ""
    for k in ['gt', 'answer', 'short_answer', 'label']:
        v = sample.get(k)
        if isinstance(v, str) and v.strip():
            short_answer = v.strip()
            break

    # If still empty, try extract short answer from tags in completion
    if not short_answer:
        short_answer = _extract_between_tags(completion_text, 'answer')
    if not short_answer:
        # Try explicit short answer keys
        for k in ['answer', 'short_answer', 'label', 'gt', 'solution']:
            v = sample.get(k)
            if isinstance(v, str) and v.strip():
                short_answer = v.strip()
                break

    # If still not found, try boxed extraction from completion or long solution
    if not short_answer:
        long_solution_candidates = ['solution', 'rationale', 'explanation']
        long_solution_text = completion_text
        if not long_solution_text:
            for k in long_solution_candidates:
                v = sample.get(k)
                if isinstance(v, str) and v.strip():
                    long_solution_text = v.strip()
                    break
        short_answer = _extract_last_boxed(long_solution_text)

    solution_text = _format_solution(short_answer)
    return completion_text, solution_text


def _guess_image_path(sample: Dict[str, Any], images_dir: str) -> Optional[str]:
    """Construct absolute image path from common image keys.

    It tries keys: image_path, image, img, file_name, filename, image_id, image_url (basename).
    """
    image_key_candidates = [
        'image_path', 'image', 'img', 'file_name', 'filename', 'image_id', 'image_url', 'id'
    ]
    image_value: Optional[str] = None
    for key in image_key_candidates:
        v = sample.get(key)
        if isinstance(v, str) and v.strip():
            image_value = v.strip()
            break

    if image_value is None:
        # Sometimes image is nested in conversations content as a dict
        msgs = sample.get('messages') or sample.get('conversations')
        if isinstance(msgs, list):
            for msg in msgs:
                content = msg.get('content') if isinstance(msg, dict) else None
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict):
                            for k in ('image', 'image_url', 'image_path', 'img'):
                                if k in c and isinstance(c[k], str) and c[k].strip():
                                    image_value = os.path.basename(c[k].strip())
                                    break
                        if image_value:
                            break
                if image_value:
                    break

    if image_value is None:
        return None

    # Try full relative path under images_dir first
    candidate1 = os.path.join(images_dir, image_value)
    if os.path.exists(candidate1):
        return candidate1

    # Fallback to basename under images_dir
    image_name = os.path.basename(image_value)
    candidate2 = os.path.join(images_dir, image_name)
    return candidate2


def preprocess_samples(samples: List[Dict[str, Any]], images_dir: str) -> Dict[str, List[Any]]:
    """Preprocess a list of raw samples to required fields.

    Returns: dict with lists: prompt, completion, solution, image_path
    """
    prompts: List[str] = []
    completions: List[str] = []
    solutions: List[str] = []
    image_paths: List[Optional[str]] = []

    prefix = r"""Solve the problem based on the image. Provide reasoning and put the FINAL ANSWER inside \boxed{}. \n """

    for sample in samples:
        prompt_text = _guess_prompt(sample)
        completion_text, solution_text = _guess_completion_and_solution(sample)
        image_path = _guess_image_path(sample, images_dir)

        final_prompt = prefix + (f"Question: {prompt_text}\n" if prompt_text else "")

        prompts.append(final_prompt)
        completions.append(completion_text or "")
        solutions.append(solution_text or "")
        image_paths.append(image_path)

    return {
        "prompt": prompts,
        "completion": completions,
        "solution": solutions,
        "image_path": image_paths,
    }


def preprocess_dataset(dataset_dict: DatasetDict, raw_data: List[Dict[str, Any]], images_dir: str, batch_size: int = 512) -> DatasetDict:
    """Preprocess all splits to have fields: prompt, completion, solution, image_path

    Expects each split to contain an 'idx' column referencing entries in raw_data.
    """

    def _map(split: str) -> Dataset:
        logging.info(f"Preprocessing {split} split with batch_size={batch_size}")
        ds = dataset_dict[split].map(
            lambda batch: preprocess_samples([raw_data[i] for i in batch["idx"]], images_dir),
            batched=True,
            batch_size=batch_size,
            num_proc=None,
            remove_columns=["idx"],
            desc=f"Thinking-GRPO preprocess ({split})",
        )

        # Filter out samples with empty solution
        def has_valid_solution(example: Dict[str, Any]) -> bool:
            solution = example.get("solution", "") or ""
            return len(solution.strip()) > 0

        before = len(ds)
        ds = ds.filter(has_valid_solution, num_proc=None, desc=f"Filter empty solutions ({split})")
        after = len(ds)
        if before != after:
            logging.warning(f"{split}: Filtered out {before - after} samples with empty solutions")

        # Filter out samples with missing image_path
        def has_valid_image_path(example: Dict[str, Any]) -> bool:
            p = example.get("image_path", None)
            return p is not None and len(str(p).strip()) > 0

        before2 = len(ds)
        ds = ds.filter(has_valid_image_path, num_proc=None, desc=f"Filter missing image_path ({split})")
        after2 = len(ds)
        if before2 != after2:
            logging.warning(f"{split}: Filtered out {before2 - after2} samples with missing image_path")

        # Further filter: image file must exist on disk
        def has_existing_image(example: Dict[str, Any]) -> bool:
            p = example.get("image_path", None)
            try:
                return p is not None and os.path.exists(str(p))
            except Exception:
                return False

        before3 = len(ds)
        ds = ds.filter(has_existing_image, num_proc=None, desc=f"Filter non-existent image files ({split})")
        after3 = len(ds)
        if before3 != after3:
            logging.warning(f"{split}: Filtered out {before3 - after3} samples with non-existent image files")

        logging.info(f"Preprocessing done for {split}: {len(ds)} samples")
        return ds

    processed = DatasetDict()
    for split_name in dataset_dict.keys():
        processed[split_name] = _map(split_name)
    return processed


def save_processed_data(dataset_dict: DatasetDict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for split_name, dataset in dataset_dict.items():
        output_path = os.path.join(output_dir, f"{split_name}.json")
        logging.info(f"Saving {split_name} split to {output_path} ({len(dataset)} samples)")
        data = [dict(item) for item in dataset]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    logging.info(f"All splits saved to {output_dir}")


def verify_processed_data(dataset_dict: DatasetDict):
    example: Optional[Dict[str, Any]] = None
    total = 0
    for split_name in dataset_dict.keys():
        ds = dataset_dict[split_name]
        total += len(ds)
        if len(ds) > 0 and example is None:
            example = ds[0]
    if example is not None:
        logging.info(f"Example after preprocessing: {example}")
    logging.info(f"Total samples across all splits: {total}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess VLAA Thinking-GRPO dataset")

    parser.add_argument("--output_dir", type=str, default="data/thinking_grpo",
                        help="Output directory for preprocessed data")
    parser.add_argument("--dataset_root", type=str, default="dataset/thinking_grpo",
                        help="Root directory for dataset")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Directory where images are located (default: <dataset_root>/images)")
    parser.add_argument("--json_path", type=str, default=None,
                        help="Path to VLAA-Thinking-GRPO-25K.json (default: <dataset_root>/VLAA-Thinking-GRPO-25K.json)")
    parser.add_argument("--train_ratio", type=float, default=0.98,
                        help="Training set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.01,
                        help="Test set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.01,
                        help="Validation set ratio")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for preprocessing")
    parser.add_argument("--allowed_verifier_types", type=str, default="digit,math,mcq",
                        help="Comma-separated verifier types to keep (default: digit,math,mcq)")

    args = parser.parse_args()

    output_dir = args.output_dir
    dataset_root = args.dataset_root
    images_dir = args.images_dir or os.path.join(dataset_root, "images")
    json_path = args.json_path or os.path.join(dataset_root, "VLAA-Thinking-GRPO-25K.json")

    logging.info("=" * 80)
    logging.info("Thinking-GRPO Dataset Preprocessing")
    logging.info("=" * 80)
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Dataset root: {dataset_root}")
    logging.info(f"Images dir: {images_dir}")
    logging.info(f"JSON path: {json_path}")
    logging.info(f"Train ratio: {args.train_ratio}, Test ratio: {args.test_ratio}, Val ratio: {args.val_ratio}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info("=" * 80)

    # Step 1: Load data
    data = load_thinking_grpo_data(json_path)

    # Log overall verifier_type distribution
    overall_counts = count_verifier_types(data)
    logging.info("Verifier type distribution (overall):")
    for k in sorted(overall_counts.keys()):
        logging.info(f"  {k}: {overall_counts[k]}")

    # Step 1.5: Filter by verifier_type
    allowed = set([t.strip() for t in (args.allowed_verifier_types or "").split(',') if t.strip()])
    if allowed:
        before_n = len(data)
        filtered_data = [s for s in data if str(s.get('verifier_type', '')).strip() in allowed]
        after_n = len(filtered_data)
        logging.info(f"Applying verifier_type filter: {sorted(allowed)} | kept {after_n}/{before_n} samples")
        filtered_counts = count_verifier_types(filtered_data)
        logging.info("Verifier type distribution (filtered):")
        for k in sorted(filtered_counts.keys()):
            logging.info(f"  {k}: {filtered_counts[k]}")
    else:
        filtered_data = data

    # Step 2: Split dataset
    dataset_dict = split_dataset(filtered_data, args.train_ratio, args.test_ratio, args.val_ratio)

    # Log verifier_type distribution per split (based on raw indices)
    for split_name in dataset_dict.keys():
        try:
            idxs = list(dataset_dict[split_name]["idx"])  # type: ignore[index]
            split_samples = [filtered_data[i] for i in idxs]
            split_counts = count_verifier_types(split_samples)
            logging.info(f"Verifier type distribution ({split_name}):")
            for k in sorted(split_counts.keys()):
                logging.info(f"  {k}: {split_counts[k]}")
        except Exception as e:
            logging.warning(f"Failed to compute verifier_type counts for split {split_name}: {e}")

    # Step 3: Preprocess dataset
    processed_dict = preprocess_dataset(dataset_dict, filtered_data, images_dir, args.batch_size)

    # Step 4: Verify data quality
    verify_processed_data(processed_dict)

    # Step 5: Save processed data
    save_processed_data(processed_dict, output_dir)

    logging.info("=" * 80)
    logging.info("Preprocessing complete!")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()


