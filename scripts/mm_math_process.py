"""
MM Math Dataset Preprocessing Script

This script downloads, splits, and preprocesses the MM Math dataset.
The preprocessed data will be saved to the data/mm_math directory with
fields: prompt, completion, solution, image_path

Usage:
    # Using config file
    uv run scripts/mm_math_process.py --config configs/latent_memory/mm_math.yaml
    
    # Manual parameters
    uv run scripts/mm_math_process.py --output_dir data/mm_math --val_ratio 0.05 --test_ratio 0.05
"""

import os
import re
import json
import logging
import argparse
from typing import Dict, List, Optional
import requests
import zipfile
import io
import yaml
from datasets import load_dataset, DatasetDict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def download_data(cache_path: str) -> str:
    """Download MM Math JSONL data.
    
    Args:
        cache_path: Directory to save downloaded data
        
    Returns:
        Path to downloaded JSONL file
    """
    os.makedirs(cache_path, exist_ok=True)
    jsonl_url = "https://huggingface.co/datasets/THU-KEG/MM_Math/resolve/main/MM_Math/MM_Math.jsonl"
    jsonl_path = os.path.join(cache_path, "mm_math.jsonl")

    if os.path.exists(jsonl_path):
        logging.info(f"JSONL file already exists at {jsonl_path}")
    else:
        logging.info(f"Downloading MM_Math dataset to {jsonl_path}")
        resp = requests.get(jsonl_url, timeout=120)
        resp.raise_for_status()
        with open(jsonl_path, "wb") as f:
            f.write(resp.content)
        logging.info("Download complete")

    return jsonl_path


def download_images(image_root: str):
    """Download MM Math images.
    
    Args:
        image_root: Directory to extract images to
    """
    if os.path.isdir(image_root) and len(os.listdir(image_root)) > 0:
        logging.info(f"Images already exist at {image_root}")
        return

    os.makedirs(image_root, exist_ok=True)
    logging.info(f"Downloading MM_Math images to {image_root}")
    zip_url = "https://huggingface.co/datasets/THU-KEG/MM_Math/resolve/main/MM_Math/MM_Math.zip"
    resp = requests.get(zip_url, timeout=300)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        z.extractall(image_root)
    logging.info("Image download complete")


def split_dataset(jsonl_path: str, val_ratio: float = 0.05, test_ratio: float = 0.05) -> DatasetDict:
    """Split dataset into train/valid/test.
    
    Args:
        jsonl_path: Path to the JSONL file
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        
    Returns:
        DatasetDict with train/valid/test splits
    """
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("Invalid val_ratio/test_ratio; ensure 0 <= ratios and sum < 1.0")

    logging.info(f"Loading dataset from {jsonl_path}")
    raw_ds = load_dataset("json", data_files={"train": jsonl_path})
    base_train = raw_ds["train"]

    logging.info(f"Splitting dataset: val_ratio={val_ratio}, test_ratio={test_ratio}")
    holdout_ratio = val_ratio + test_ratio
    first_split = base_train.train_test_split(test_size=holdout_ratio, seed=42, shuffle=True)
    split_train = first_split["train"]
    holdout = first_split["test"]

    if holdout_ratio > 0:
        test_fraction = test_ratio / holdout_ratio
        second_split = holdout.train_test_split(test_size=test_fraction, seed=42, shuffle=True)
        split_valid = second_split["train"]
        split_test = second_split["test"]
    else:
        split_valid = split_train.select(range(0))
        split_test = split_train.select(range(0))

    dataset_dict = DatasetDict({
        "train": split_train,
        "valid": split_valid,
        "test": split_test,
    })

    logging.info(f"Split sizes - train: {len(split_train)}, valid: {len(split_valid)}, test: {len(split_test)}")
    return dataset_dict


def preprocess_batch(batch: Dict, image_root: str) -> Dict:
    """Preprocess a batch of examples.
    
    Args:
        batch: Batch of raw examples
        image_root: Root directory for images
        
    Returns:
        Preprocessed batch with fields:
        - prompt: formatted question prompt
        - completion: original full solution text
        - solution: extracted boxed answer (for reward computation)
        - image_path: path to image file
    """
    def _format_answer(answer: str) -> str:
        answer = (answer or "").strip()
        if answer.startswith("\\boxed{") and answer.endswith("}"):
            return answer
        return "\\boxed{" + answer + "}"

    def _extract_answer_from_solution(solution_text: str) -> str:
        """Extract the LAST \\boxed{...} occurrence from solution text."""
        if not solution_text:
            return ""

        # Try both single and double backslash patterns
        matches = list(re.finditer(r"\\\\boxed\{([^}]+)\}", solution_text, flags=re.DOTALL))
        if not matches:
            matches = list(re.finditer(r"\\boxed\{([^}]+)\}", solution_text, flags=re.DOTALL))

        if matches:
            return matches[-1].group(1).strip()

        # Handle incomplete patterns
        incomplete_pattern = r"\\\\?boxed\{([^}]*?)(?:\}|$|\n)"
        incomplete_matches = list(re.finditer(incomplete_pattern, solution_text, flags=re.DOTALL))
        if incomplete_matches:
            content = incomplete_matches[-1].group(1).strip()
            if content:
                return content

        return ""

    format_template = r"""Solve the problem and output the answer in the format of <answer>Your answer here</answer>. \n """
    prompt_template = "Question: {prompt}\n"

    questions: List[str] = batch.get("question") or [""] * len(batch.get("solution", []))
    answers_src: List[str] = batch.get("answer") or [""] * len(questions)
    solutions_src: List[str] = batch.get("solution") or [""] * len(questions)
    file_names_src = batch.get("file_name", [None] * len(questions))
    image_paths_src = batch.get("image_path", [None] * len(questions))
    file_names: List[str] = [fn if fn is not None else ip for fn, ip in zip(file_names_src, image_paths_src)]

    prompts: List[str] = []
    completions: List[str] = []
    solutions: List[str] = []
    image_paths: List[str] = []

    for q, a_src, sol_src, fn in zip(questions, answers_src, solutions_src, file_names):
        processed_prompt = format_template + prompt_template.format(prompt=(q or "").strip())

        # Prefer explicit short answer; fallback to extracting from long solution
        answer_text = (a_src or "").strip()
        if len(answer_text) == 0:
            answer_text = _extract_answer_from_solution((sol_src or "").strip())

        # completion: original full solution text
        # solution: extracted boxed answer for reward computation
        completion_text = (sol_src or "").strip()
        solution_label = _format_answer(answer_text)

        prompts.append(processed_prompt)
        completions.append(completion_text)
        solutions.append(solution_label)

        if fn is not None:
            image_paths.append(os.path.join(image_root, fn))
        else:
            image_paths.append(None)

    return {
        "prompt": prompts,
        "completion": completions,
        "solution": solutions,
        "image_path": image_paths,
    }


def preprocess_dataset(dataset_dict: DatasetDict, image_root: str, batch_size: int = 512) -> DatasetDict:
    """Preprocess all splits.
    
    Args:
        dataset_dict: Raw dataset dictionary
        image_root: Root directory for images
        batch_size: Batch size for processing
        
    Returns:
        Preprocessed DatasetDict with fields: prompt, completion, solution, image_path
    """
    keep_keys = ["prompt", "completion", "solution", "image_path"]

    def _map(split):
        logging.info(f"Preprocessing {split} split with batch_size={batch_size}")
        ds = dataset_dict[split].map(
            lambda batch: preprocess_batch(batch, image_root),
            batched=True,
            batch_size=batch_size,
            num_proc=None,
            remove_columns=dataset_dict[split].column_names,
            desc=f"MM_Math preprocess ({split})",
        )

        # Filter out samples with empty solution
        def has_valid_solution(example):
            solution = example.get("solution", "")
            return solution is not None and len(solution.strip()) > 0

        original_size = len(ds)
        ds = ds.filter(has_valid_solution, num_proc=None, desc=f"Filter empty solutions ({split})")
        filtered_size = len(ds)
        if original_size != filtered_size:
            logging.warning(f"{split}: Filtered out {original_size - filtered_size} samples with empty solutions")

        logging.info(f"Preprocessing done for {split}: {len(ds)} samples")
        return ds

    processed_dict = DatasetDict()
    for split_name in dataset_dict.keys():
        processed_dict[split_name] = _map(split_name)

    return processed_dict


def save_processed_data(dataset_dict: DatasetDict, output_dir: str):
    """Save preprocessed data to JSON files.
    
    Args:
        dataset_dict: Preprocessed dataset dictionary
        output_dir: Output directory to save JSON files
    """
    os.makedirs(output_dir, exist_ok=True)

    for split_name, dataset in dataset_dict.items():
        output_path = os.path.join(output_dir, f"{split_name}.json")
        logging.info(f"Saving {split_name} split to {output_path} ({len(dataset)} samples)")

        # Convert to list of dicts and save with json module for proper formatting
        data = [dict(item) for item in dataset]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    logging.info(f"All splits saved to {output_dir}")


def verify_processed_data(dataset_dict: DatasetDict):
    """Verify processed data quality.
    
    Args:
        dataset_dict: Processed dataset dictionary
    """
    pattern = re.compile(r"\\boxed\{[^}]+\}")
    found = None
    checked = 0
    nonempty = 0
    total = 0

    for split_name in ("train", "valid", "test"):
        if split_name in dataset_dict:
            ds = dataset_dict[split_name]
            total += len(ds)
            limit = min(100, len(ds))
            for i in range(limit):
                ex = ds[i]
                sol = ex.get("solution", "") or ""
                if pattern.search(sol):
                    nonempty += 1
                    if found is None:
                        found = ex
                checked += 1

    if found is not None:
        logging.info(f"Example after preprocessing (verified boxed): {found}")
    logging.info(f"Boxed solution stats: checked={checked}, nonempty_boxed={nonempty}, total={total}")


def load_config_from_yaml(config_path: str) -> Dict:
    """Load configuration from yaml file.
    
    Args:
        config_path: Path to yaml config file
        
    Returns:
        Dictionary with configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract mm_math dataset config
    dataset_config = config.get('datasets', {}).get('mm_math', {})

    # Try to get from sft config first, then grpo, then fallback to dataset level
    sft_config = dataset_config.get('sft', {})
    grpo_config = dataset_config.get('grpo', {})

    # Use sft as default, they should have same structure anyway
    mode_config = sft_config if sft_config else grpo_config

    return {
        'cache_path': mode_config.get('cache_path') or dataset_config.get('cache_path'),
        'val_ratio': mode_config.get('val_ratio') or dataset_config.get('val_ratio', 0.05),
        'test_ratio': mode_config.get('test_ratio') or dataset_config.get('test_ratio', 0.05),
        'image_root': mode_config.get('image_root') or dataset_config.get('image_root'),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess MM Math dataset")

    # Config file options
    parser.add_argument("--config", type=str, default=None,
                        help="Path to yaml config file (e.g., configs/latent_memory/mm_math.yaml)")

    # Manual override options
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for preprocessed data (overrides config)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for downloaded raw data (overrides config)")
    parser.add_argument("--image_root", type=str, default=None,
                        help="Directory for image files (overrides config)")
    parser.add_argument("--val_ratio", type=float, default=None,
                        help="Validation set ratio (overrides config)")
    parser.add_argument("--test_ratio", type=float, default=None,
                        help="Test set ratio (overrides config)")

    # Other options
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for preprocessing")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip downloading data and images")

    args = parser.parse_args()

    # Load config from yaml if provided
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")

        logging.info(f"Loading configuration from {args.config}")
        yaml_config = load_config_from_yaml(args.config)

        # Use config values, but allow command-line overrides
        # Note: output_dir defaults to data/mm_math (not cache_path from config)
        output_dir = args.output_dir or "data/mm_math"
        cache_dir = args.cache_dir or ".cache/mm_math"
        image_root = args.image_root or yaml_config.get('image_root') or "dataset/mm_math/images/MM_Math"
        val_ratio = args.val_ratio if args.val_ratio is not None else yaml_config.get('val_ratio', 0.05)
        test_ratio = args.test_ratio if args.test_ratio is not None else yaml_config.get('test_ratio', 0.05)
    else:
        # Use command-line arguments or defaults
        output_dir = args.output_dir or "data/mm_math"
        cache_dir = args.cache_dir or ".cache/mm_math"
        image_root = args.image_root or "dataset/mm_math/images/MM_Math"
        val_ratio = args.val_ratio if args.val_ratio is not None else 0.05
        test_ratio = args.test_ratio if args.test_ratio is not None else 0.05

    logging.info("=" * 80)
    logging.info("MM Math Dataset Preprocessing")
    logging.info("=" * 80)
    if args.config:
        logging.info(f"Config file: {args.config}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Cache directory: {cache_dir}")
    logging.info(f"Image root: {image_root}")
    logging.info(f"Val ratio: {val_ratio}, Test ratio: {test_ratio}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info("=" * 80)

    # Step 1: Download data (auto-skip if already exists)
    jsonl_path = os.path.join(cache_dir, "mm_math.jsonl")

    if args.skip_download:
        logging.info("Skipping download as --skip_download flag is set")
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"JSONL file not found at {jsonl_path}. "
                f"Remove --skip_download flag to download."
            )
    else:
        # Auto-download only if needed
        jsonl_path = download_data(cache_dir)
        download_images(image_root)

    # Step 2: Split dataset
    dataset_dict = split_dataset(jsonl_path, val_ratio, test_ratio)

    # Step 3: Preprocess dataset
    processed_dict = preprocess_dataset(dataset_dict, image_root, args.batch_size)

    # Step 4: Verify data quality
    verify_processed_data(processed_dict)

    # Step 5: Save processed data
    save_processed_data(processed_dict, output_dir)

    logging.info("=" * 80)
    logging.info("Preprocessing complete!")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
