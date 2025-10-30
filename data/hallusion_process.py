"""
HallusionBench Dataset Preprocessing Script

This script loads and preprocesses the HallusionBench dataset.
The preprocessed data will be saved to the data/hallusion_bench directory with
fields: prompt, completion, solution, image_path

Usage:
    # Using config file
    uv run data/hallusion_process.py --config configs/latent_memory/hallusion_bench.yaml
    
    # Manual parameters
    uv run data/hallusion_process.py --output_dir data/hallusion_bench --val_ratio 0.01 --test_ratio 0.2
"""

import os
import json
import logging
import argparse
from typing import Dict, List
import yaml
from datasets import DatasetDict, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_hallusion_data(json_path: str, dataset_root: str) -> List[Dict]:
    """Load HallusionBench data from JSON file.
    
    Args:
        json_path: Path to HallusionBench.json
        dataset_root: Root directory where images are located
        
    Returns:
        List of data samples with adjusted image paths
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    logging.info(f"Loading HallusionBench data from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Adjust image paths
    for sample in data:
        if sample.get('filename'):
            # Convert relative path to absolute path based on dataset_root
            # e.g., "./hallusion_bench/VD/illusion/0_0.png" -> "dataset/hallusion_bench/VD/illusion/0_0.png"
            filename = sample['filename']
            if filename.startswith('./hallusion_bench/'):
                # Remove the "./" and "hallusion_bench/" prefix
                relative_path = filename.replace('./hallusion_bench/', '')
                sample['filename'] = os.path.join(dataset_root, relative_path)
            elif filename.startswith('./'):
                # Remove "./" prefix
                relative_path = filename[2:]
                sample['filename'] = os.path.join(dataset_root, relative_path)
            else:
                sample['filename'] = os.path.join(dataset_root, filename)
    
    logging.info(f"Loaded {len(data)} samples from HallusionBench")
    return data


def split_dataset(data: List[Dict], train_ratio: float = 0.79, test_ratio: float = 0.2, val_ratio: float = 0.01) -> DatasetDict:
    """Split dataset into train/valid/test with stratified sampling by category.
    
    This ensures that each category/subcategory combination is evenly distributed
    across train/valid/test splits.
    
    Args:
        data: List of data samples
        train_ratio: Training set ratio
        test_ratio: Test set ratio
        val_ratio: Validation set ratio
        
    Returns:
        DatasetDict with train/valid/test splits
    """
    if abs(train_ratio + test_ratio + val_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + test_ratio + val_ratio}")
    
    # Group data by category and subcategory
    from collections import defaultdict
    import random
    
    category_groups = defaultdict(list)
    for item in data:
        category = item.get('category', 'unknown')
        subcategory = item.get('subcategory', 'unknown')
        key = f"{category}/{subcategory}"
        category_groups[key].append(item)
    
    logging.info(f"Found {len(category_groups)} category groups")
    for key, items in sorted(category_groups.items()):
        logging.info(f"  {key}: {len(items)} samples")
    
    logging.info(f"Splitting dataset with stratified sampling: train={train_ratio}, test={test_ratio}, val={val_ratio}")
    
    # Split each category group proportionally
    train_data = []
    valid_data = []
    test_data = []
    
    random.seed(42)
    
    for category_key, items in category_groups.items():
        # Shuffle items within each category
        items_copy = items.copy()
        random.shuffle(items_copy)
        
        n_total = len(items_copy)
        n_test = max(1, int(n_total * test_ratio)) if test_ratio > 0 else 0
        n_valid = max(1, int(n_total * val_ratio)) if val_ratio > 0 else 0
        n_train = n_total - n_test - n_valid
        
        # Ensure at least some samples in train if possible
        if n_train < 1 and n_total > (n_test + n_valid):
            n_train = 1
            # Reduce test or valid to make room
            if n_valid > 1:
                n_valid -= 1
            elif n_test > 1:
                n_test -= 1
        
        # Split the items
        test_items = items_copy[:n_test] if n_test > 0 else []
        valid_items = items_copy[n_test:n_test + n_valid] if n_valid > 0 else []
        train_items = items_copy[n_test + n_valid:]
        
        train_data.extend(train_items)
        valid_data.extend(valid_items)
        test_data.extend(test_items)
    
    # Shuffle the final splits
    random.shuffle(train_data)
    random.shuffle(valid_data)
    random.shuffle(test_data)
    
    # Convert to HuggingFace Dataset
    dataset_dict = DatasetDict()
    
    if len(train_data) > 0:
        dataset_dict["train"] = Dataset.from_list(train_data)
    
    if len(valid_data) > 0:
        dataset_dict["valid"] = Dataset.from_list(valid_data)
    
    if len(test_data) > 0:
        dataset_dict["test"] = Dataset.from_list(test_data)
    
    logging.info(f"Stratified split sizes - train: {len(train_data)}, valid: {len(valid_data)}, test: {len(test_data)}")
    
    # Verify distribution
    for split_name, split_data in [("train", train_data), ("valid", valid_data), ("test", test_data)]:
        if len(split_data) > 0:
            split_categories = defaultdict(int)
            for item in split_data:
                category = item.get('category', 'unknown')
                subcategory = item.get('subcategory', 'unknown')
                key = f"{category}/{subcategory}"
                split_categories[key] += 1
            logging.info(f"{split_name} split category distribution:")
            for key in sorted(split_categories.keys()):
                logging.info(f"  {key}: {split_categories[key]} samples")
    
    return dataset_dict


def preprocess_batch(batch: Dict) -> Dict:
    """Preprocess a batch of examples.
    
    Args:
        batch: Batch of raw examples with HallusionBench format
        
    Returns:
        Preprocessed batch with fields:
        - prompt: formatted question prompt
        - completion: ground truth answer details
        - solution: binary answer (Yes/No based on gt_answer)
        - image_path: path to image file
        - category: category of the sample (preserved for tracking)
        - subcategory: subcategory of the sample (preserved for tracking)
    """
    questions: List[str] = batch.get("question", [])
    gt_answers: List = batch.get("gt_answer", [])
    gt_answer_details: List[str] = batch.get("gt_answer_details", [])
    filenames: List = batch.get("filename", [])
    categories: List[str] = batch.get("category", [])
    subcategories: List[str] = batch.get("subcategory", [])
    
    format_template = r"""Answer the question based on the image. Provide a clear Yes or No answer with proper reasoning, and make sure to put the FINAL ANSWER (Yes or No) inside \boxed{}. \n """
    prompt_template = "Question: {question}\n"
    
    prompts: List[str] = []
    completions: List[str] = []
    solutions: List[str] = []
    image_paths: List[str] = []
    out_categories: List[str] = []
    out_subcategories: List[str] = []
    
    for question, gt_ans, gt_details, filename, category, subcategory in zip(
        questions, gt_answers, gt_answer_details, filenames, categories, subcategories
    ):
        # Format question
        question_text = str(question).strip() if question else ""
        
        # Create prompt
        processed_prompt = format_template + prompt_template.format(question=question_text)
        
        # Process ground truth answer
        # gt_answer is "0" (No) or "1" (Yes)
        gt_answer_str = str(gt_ans).strip() if gt_ans is not None else "0"
        answer_label = "Yes" if gt_answer_str == "1" else "No"
        
        # Completion: use the detailed answer if available, otherwise use simple Yes/No
        gt_details_str = str(gt_details).strip() if gt_details else ""
        if gt_details_str:
            completion_text = gt_details_str
        else:
            completion_text = answer_label
        
        # Solution: boxed answer (Yes or No)
        solution_text = f"\\boxed{{{answer_label}}}"
        
        prompts.append(processed_prompt)
        completions.append(completion_text)
        solutions.append(solution_text)
        
        # Image path
        image_path = filename if filename else None
        image_paths.append(image_path)
        
        # Preserve category information
        out_categories.append(category if category else "unknown")
        out_subcategories.append(subcategory if subcategory else "unknown")
    
    return {
        "prompt": prompts,
        "completion": completions,
        "solution": solutions,
        "image_path": image_paths,
        "category": out_categories,
        "subcategory": out_subcategories,
    }


def preprocess_dataset(dataset_dict: DatasetDict, batch_size: int = 512) -> DatasetDict:
    """Preprocess all splits.
    
    Args:
        dataset_dict: Raw dataset dictionary
        batch_size: Batch size for processing
        
    Returns:
        Preprocessed DatasetDict with fields: prompt, completion, solution, image_path
    """
    def _map(split):
        logging.info(f"Preprocessing {split} split with batch_size={batch_size}")
        ds = dataset_dict[split].map(
            preprocess_batch,
            batched=True,
            batch_size=batch_size,
            num_proc=None,
            remove_columns=dataset_dict[split].column_names,
            desc=f"HallusionBench preprocess ({split})",
        )
        
        # Filter out samples with empty solution
        def has_valid_solution(example):
            solution = example.get("solution", "")
            return solution is not None and len(str(solution).strip()) > 0
        
        original_size = len(ds)
        ds = ds.filter(has_valid_solution, num_proc=None, desc=f"Filter empty solutions ({split})")
        filtered_size = len(ds)
        if original_size != filtered_size:
            logging.warning(f"{split}: Filtered out {original_size - filtered_size} samples with empty solutions")
        
        # Further filter out samples with missing or empty image_path
        def has_valid_image_path(example):
            image_path = example.get("image_path", None)
            return image_path is not None and len(str(image_path).strip()) > 0
        
        image_before_size = len(ds)
        ds = ds.filter(has_valid_image_path, num_proc=None, desc=f"Filter missing image_path ({split})")
        image_after_size = len(ds)
        if image_before_size != image_after_size:
            logging.warning(f"{split}: Filtered out {image_before_size - image_after_size} samples with missing image_path")
        
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
    found = None
    total = 0
    
    for split_name in dataset_dict.keys():
        ds = dataset_dict[split_name]
        total += len(ds)
        if len(ds) > 0 and found is None:
            found = ds[0]
    
    if found is not None:
        logging.info(f"Example after preprocessing: {found}")
    logging.info(f"Total samples across all splits: {total}")


def load_config_from_yaml(config_path: str) -> Dict:
    """Load configuration from yaml file.
    
    Args:
        config_path: Path to yaml config file
        
    Returns:
        Dictionary with configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract hallusion_bench dataset config
    dataset_config = config.get('datasets', {}).get('hallusion_bench', {})
    
    # Try to get from sft config first, then grpo, then fallback to dataset level
    sft_config = dataset_config.get('sft', {})
    grpo_config = dataset_config.get('grpo', {})
    
    # Use sft as default
    mode_config = sft_config if sft_config else grpo_config
    
    return {
        'cache_path': mode_config.get('cache_path') or dataset_config.get('cache_path'),
        'train_ratio': mode_config.get('train_ratio') or dataset_config.get('train_ratio', 0.79),
        'test_ratio': mode_config.get('test_ratio') or dataset_config.get('test_ratio', 0.2),
        'val_ratio': mode_config.get('val_ratio') or dataset_config.get('val_ratio', 0.01),
        'dataset_root': mode_config.get('dataset_root') or dataset_config.get('dataset_root'),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess HallusionBench dataset")
    
    # Config file options
    parser.add_argument("--config", type=str, default=None,
                        help="Path to yaml config file (e.g., configs/latent_memory/hallusion_bench.yaml)")
    
    # Manual override options
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for preprocessed data (overrides config)")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Root directory for dataset (where HallusionBench.json and images are located)")
    parser.add_argument("--json_path", type=str, default=None,
                        help="Path to HallusionBench.json file")
    parser.add_argument("--train_ratio", type=float, default=None,
                        help="Training set ratio (overrides config)")
    parser.add_argument("--test_ratio", type=float, default=None,
                        help="Test set ratio (overrides config)")
    parser.add_argument("--val_ratio", type=float, default=None,
                        help="Validation set ratio (overrides config)")
    
    # Other options
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for preprocessing")
    
    args = parser.parse_args()
    
    # Load config from yaml if provided
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        
        logging.info(f"Loading configuration from {args.config}")
        yaml_config = load_config_from_yaml(args.config)
        
        # Use config values, but allow command-line overrides
        output_dir = args.output_dir or "data/hallusion_bench"
        dataset_root = args.dataset_root or yaml_config.get('dataset_root') or "dataset/hallusion_bench"
        train_ratio = args.train_ratio if args.train_ratio is not None else yaml_config.get('train_ratio', 0.79)
        test_ratio = args.test_ratio if args.test_ratio is not None else yaml_config.get('test_ratio', 0.2)
        val_ratio = args.val_ratio if args.val_ratio is not None else yaml_config.get('val_ratio', 0.01)
    else:
        # Use command-line arguments or defaults
        output_dir = args.output_dir or "data/hallusion_bench"
        dataset_root = args.dataset_root or "dataset/hallusion_bench"
        train_ratio = args.train_ratio if args.train_ratio is not None else 0.79
        test_ratio = args.test_ratio if args.test_ratio is not None else 0.2
        val_ratio = args.val_ratio if args.val_ratio is not None else 0.01
    
    # Determine JSON path
    json_path = args.json_path or os.path.join(dataset_root, "HallusionBench.json")
    
    logging.info("=" * 80)
    logging.info("HallusionBench Dataset Preprocessing")
    logging.info("=" * 80)
    if args.config:
        logging.info(f"Config file: {args.config}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Dataset root: {dataset_root}")
    logging.info(f"JSON path: {json_path}")
    logging.info(f"Train ratio: {train_ratio}, Test ratio: {test_ratio}, Val ratio: {val_ratio}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info("=" * 80)
    
    # Step 1: Load data
    data = load_hallusion_data(json_path, dataset_root)
    
    # Step 2: Split dataset
    dataset_dict = split_dataset(data, train_ratio, test_ratio, val_ratio)
    
    # Step 3: Preprocess dataset
    processed_dict = preprocess_dataset(dataset_dict, args.batch_size)
    
    # Step 4: Verify data quality
    verify_processed_data(processed_dict)
    
    # Step 5: Save processed data
    save_processed_data(processed_dict, output_dir)
    
    logging.info("=" * 80)
    logging.info("Preprocessing complete!")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()

