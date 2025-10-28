"""
MMVP Dataset Preprocessing Script

This script downloads, splits, and preprocesses the MMVP dataset.
The preprocessed data will be saved to the data/mmvp directory with
fields: prompt, completion, solution, image_path

Usage:
    # Using config file
    uv run data/mmvp_process.py --config configs/latent_memory/mmvp.yaml
    
    # Manual parameters
    uv run data/mmvp_process.py --output_dir data/mmvp --val_ratio 0.01 --test_ratio 0.2 --train_ratio 0.79
"""

import os
import json
import logging
import argparse
from typing import Dict, List
import pandas as pd
import yaml
from datasets import load_dataset, DatasetDict, Dataset
from huggingface_hub import snapshot_download
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def download_data(cache_path: str) -> Dict[str, str]:
    """Download MMVP dataset from HuggingFace.
    
    Args:
        cache_path: Directory to save downloaded data
        
    Returns:
        Dict with paths to questions file and images directory
    """
    os.makedirs(cache_path, exist_ok=True)
    
    # Check if data already exists
    questions_path = os.path.join(cache_path, "Questions.xlsx")
    images_path = os.path.join(cache_path, "MMVP Images")
    
    if os.path.exists(questions_path) and os.path.exists(images_path):
        logging.info(f"MMVP data already exists at {cache_path}")
        return {
            "questions": questions_path,
            "images": images_path
        }
    
    logging.info(f"Downloading MMVP dataset to {cache_path}")
    try:
        # Download entire repository
        snapshot_download(
            repo_id="MMVP/MMVP",
            repo_type="dataset",
            local_dir=cache_path,
            local_dir_use_symlinks=False
        )
        logging.info("Download complete")
    except Exception as e:
        logging.error(f"Failed to download dataset: {e}")
        raise
    
    return {
        "questions": questions_path,
        "images": images_path
    }


def move_images_to_dataset(images_src: str, dataset_root: str) -> str:
    """Move MMVP images to dataset directory.
    
    Args:
        images_src: Source images directory
        dataset_root: Target dataset root directory
        
    Returns:
        Path to moved images directory
    """
    images_dst = os.path.join(dataset_root, "MMVP Images")
    
    if os.path.exists(images_dst):
        logging.info(f"Images already exist at {images_dst}")
        return images_dst
    
    logging.info(f"Moving images from {images_src} to {images_dst}")
    os.makedirs(os.path.dirname(images_dst), exist_ok=True)
    
    if os.path.exists(images_src):
        shutil.copytree(images_src, images_dst)
        logging.info("Images moved successfully")
    else:
        logging.warning(f"Source images directory not found: {images_src}")
    
    return images_dst


def load_questions(questions_path: str) -> pd.DataFrame:
    """Load questions from Excel file.
    
    Args:
        questions_path: Path to Questions.xlsx
        
    Returns:
        DataFrame with columns: index, question, options, correct answer
    """
    logging.info(f"Loading questions from {questions_path}")
    
    # Try different possible column names
    df = pd.read_excel(questions_path)
    
    # Normalize column names (handle variations)
    df.columns = df.columns.str.strip()
    
    # Check for required columns
    required_cols = ['index', 'question', 'options', 'correct answer']
    for col in required_cols:
        if col not in df.columns:
            # Try case-insensitive match
            matched = [c for c in df.columns if c.lower() == col.lower()]
            if matched:
                df.rename(columns={matched[0]: col}, inplace=True)
            else:
                raise ValueError(f"Required column '{col}' not found in Questions.xlsx. Available columns: {df.columns.tolist()}")
    
    logging.info(f"Loaded {len(df)} questions")
    return df


def split_dataset(df: pd.DataFrame, train_ratio: float = 0.8, test_ratio: float = 0.2, val_ratio: float = 0.0) -> DatasetDict:
    """Split dataset into train/valid/test.
    
    Args:
        df: DataFrame with questions
        train_ratio: Training set ratio
        test_ratio: Test set ratio
        val_ratio: Validation set ratio
        
    Returns:
        DatasetDict with train/valid/test splits
    """
    if abs(train_ratio + test_ratio + val_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + test_ratio + val_ratio}")
    
    # Convert DataFrame to HuggingFace Dataset
    base_dataset = Dataset.from_pandas(df, preserve_index=False)
    total_size = len(base_dataset)
    
    logging.info(f"Splitting dataset: train={train_ratio}, test={test_ratio}, val={val_ratio}")
    
    # Calculate actual sizes, ensuring at least 1 sample if ratio > 0
    test_size = max(1, int(total_size * test_ratio)) if test_ratio > 0 else 0
    val_size = max(1, int(total_size * val_ratio)) if val_ratio > 0 else 0
    
    if val_ratio == 0.0 or val_size == 0:
        # Simple train/test split
        if test_size == 0:
            # No test set, all train
            dataset_dict = DatasetDict({
                "train": base_dataset,
            })
            logging.info(f"Split sizes - train: {len(base_dataset)}")
        else:
            split = base_dataset.train_test_split(test_size=test_size, seed=42, shuffle=True)
            dataset_dict = DatasetDict({
                "train": split["train"],
                "test": split["test"],
            })
            logging.info(f"Split sizes - train: {len(split['train'])}, test: {len(split['test'])}")
    else:
        # Three-way split: train/valid/test
        holdout_size = test_size + val_size
        
        # First split: separate training from validation+test
        first_split = base_dataset.train_test_split(test_size=holdout_size, seed=42, shuffle=True)
        split_train = first_split["train"]
        holdout = first_split["test"]
        
        # Second split: separate validation from test
        second_split = holdout.train_test_split(test_size=test_size, seed=42, shuffle=True)
        split_valid = second_split["train"]
        split_test = second_split["test"]
        
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
        batch: Batch of raw examples with keys: index, question, options, correct answer
        image_root: Root directory for images
        
    Returns:
        Preprocessed batch with fields:
        - prompt: formatted question prompt with options
        - completion: correct answer
        - solution: correct answer (same as completion for compatibility)
        - image_path: path to image file
    """
    indices: List = batch.get("index", [])
    questions: List[str] = batch.get("question", [])
    options_list: List = batch.get("options", [])
    correct_answers: List = batch.get("correct answer", [])
    
    format_template = r"""Solve the problem with proper reasoning, and make sure to put the FINAL CHOICE inside \boxed{}. \n """
    prompt_template = "Question: {question}\nOptions: {options}\n"
    
    prompts: List[str] = []
    completions: List[str] = []
    solutions: List[str] = []
    image_paths: List[str] = []
    
    for idx, question, options, correct_answer in zip(indices, questions, options_list, correct_answers):
        # Format question and options
        question_text = str(question).strip() if question else ""
        options_text = str(options).strip() if options else ""
        
        # Create prompt
        processed_prompt = format_template + prompt_template.format(
            question=question_text,
            options=options_text
        )
        
        # Correct answer as completion and solution
        answer_text = str(correct_answer).strip() if correct_answer else ""
        
        prompts.append(processed_prompt)
        completions.append(answer_text)
        solutions.append(answer_text)
        
        # Image path: index corresponds to image filename
        # Assuming images are named like "1.jpg", "2.jpg", etc.
        # Adjust extension as needed (.jpg, .png, etc.)
        image_file = None
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
            potential_path = os.path.join(image_root, f"{idx}{ext}")
            if os.path.exists(potential_path):
                image_file = potential_path
                break
        
        if image_file is None:
            # Try without extension, let the file exist check happen later
            image_file = os.path.join(image_root, str(idx))
        
        image_paths.append(image_file)
    
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
    def _map(split):
        logging.info(f"Preprocessing {split} split with batch_size={batch_size}")
        ds = dataset_dict[split].map(
            lambda batch: preprocess_batch(batch, image_root),
            batched=True,
            batch_size=batch_size,
            num_proc=None,
            remove_columns=dataset_dict[split].column_names,
            desc=f"MMVP preprocess ({split})",
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
    
    # Extract mmvp dataset config
    dataset_config = config.get('datasets', {}).get('mmvp', {})
    
    # Try to get from sft config first, then grpo, then fallback to dataset level
    sft_config = dataset_config.get('sft', {})
    grpo_config = dataset_config.get('grpo', {})
    
    # Use sft as default
    mode_config = sft_config if sft_config else grpo_config
    
    return {
        'cache_path': mode_config.get('cache_path') or dataset_config.get('cache_path'),
        'train_ratio': mode_config.get('train_ratio') or dataset_config.get('train_ratio', 0.8),
        'test_ratio': mode_config.get('test_ratio') or dataset_config.get('test_ratio', 0.2),
        'val_ratio': mode_config.get('val_ratio') or dataset_config.get('val_ratio', 0.0),
        'image_root': mode_config.get('image_root') or dataset_config.get('image_root'),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess MMVP dataset")
    
    # Config file options
    parser.add_argument("--config", type=str, default=None,
                        help="Path to yaml config file (e.g., configs/latent_memory/mmvp.yaml)")
    
    # Manual override options
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for preprocessed data (overrides config)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for downloaded raw data (overrides config)")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Root directory for dataset (where images will be moved)")
    parser.add_argument("--image_root", type=str, default=None,
                        help="Directory for image files (overrides config)")
    parser.add_argument("--train_ratio", type=float, default=None,
                        help="Training set ratio (overrides config)")
    parser.add_argument("--test_ratio", type=float, default=None,
                        help="Test set ratio (overrides config)")
    parser.add_argument("--val_ratio", type=float, default=None,
                        help="Validation set ratio (overrides config)")
    
    # Other options
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for preprocessing")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip downloading data")
    parser.add_argument("--skip_move_images", action="store_true",
                        help="Skip moving images to dataset directory")
    
    args = parser.parse_args()
    
    # Load config from yaml if provided
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        
        logging.info(f"Loading configuration from {args.config}")
        yaml_config = load_config_from_yaml(args.config)
        
        # Use config values, but allow command-line overrides
        output_dir = args.output_dir or "data/mmvp"
        cache_dir = args.cache_dir or ".cache/mmvp"
        dataset_root = args.dataset_root or "dataset/mmvp"
        image_root = args.image_root or yaml_config.get('image_root') or "dataset/mmvp/MMVP Images"
        train_ratio = args.train_ratio if args.train_ratio is not None else yaml_config.get('train_ratio', 0.8)
        test_ratio = args.test_ratio if args.test_ratio is not None else yaml_config.get('test_ratio', 0.2)
        val_ratio = args.val_ratio if args.val_ratio is not None else yaml_config.get('val_ratio', 0.0)
    else:
        # Use command-line arguments or defaults
        output_dir = args.output_dir or "data/mmvp"
        cache_dir = args.cache_dir or ".cache/mmvp"
        dataset_root = args.dataset_root or "dataset/mmvp"
        image_root = args.image_root or "dataset/mmvp/MMVP Images"
        train_ratio = args.train_ratio if args.train_ratio is not None else 0.8
        test_ratio = args.test_ratio if args.test_ratio is not None else 0.2
        val_ratio = args.val_ratio if args.val_ratio is not None else 0.0
    
    logging.info("=" * 80)
    logging.info("MMVP Dataset Preprocessing")
    logging.info("=" * 80)
    if args.config:
        logging.info(f"Config file: {args.config}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Cache directory: {cache_dir}")
    logging.info(f"Dataset root: {dataset_root}")
    logging.info(f"Image root: {image_root}")
    logging.info(f"Train ratio: {train_ratio}, Test ratio: {test_ratio}, Val ratio: {val_ratio}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info("=" * 80)
    
    # Step 1: Download data (auto-skip if already exists)
    if args.skip_download:
        logging.info("Skipping download as --skip_download flag is set")
        questions_path = os.path.join(cache_dir, "Questions.xlsx")
        images_src = os.path.join(cache_dir, "MMVP Images")
        if not os.path.exists(questions_path):
            raise FileNotFoundError(
                f"Questions file not found at {questions_path}. "
                f"Remove --skip_download flag to download."
            )
    else:
        # Auto-download only if needed
        paths = download_data(cache_dir)
        questions_path = paths["questions"]
        images_src = paths["images"]
    
    # Step 2: Move images to dataset directory
    if not args.skip_move_images:
        image_root = move_images_to_dataset(images_src, dataset_root)
    
    # Step 3: Load questions
    df = load_questions(questions_path)
    
    # Step 4: Split dataset
    dataset_dict = split_dataset(df, train_ratio, test_ratio, val_ratio)
    
    # Step 5: Preprocess dataset
    processed_dict = preprocess_dataset(dataset_dict, image_root, args.batch_size)
    
    # Step 6: Verify data quality
    verify_processed_data(processed_dict)
    
    # Step 7: Save processed data
    save_processed_data(processed_dict, output_dir)
    
    logging.info("=" * 80)
    logging.info("Preprocessing complete!")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()

