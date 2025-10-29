"""
Math Vision Dataset Preprocessing Script

This script downloads, extracts, and preprocesses the Math Vision dataset.
The preprocessed data will be saved to the data/math_vision directory with
fields: prompt, completion, solution, image_path

Usage:
    # Using config file
    uv run data/math_vision_process.py --config configs/latent_memory/math_vision.yaml
    
    # Manual parameters
    uv run data/math_vision_process.py --output_dir data/math_vision --dataset_root dataset/math_vision
"""

import os
import json
import logging
import argparse
import tarfile
import urllib.request
from typing import Dict, List
import yaml
from datasets import DatasetDict, Dataset
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dataset URLs
DATASET_TAR_URL = "https://huggingface.co/TobyYang7/math_vision/resolve/main/math_vision.tar.gz"
TRAIN_JSON_URL = "https://huggingface.co/TobyYang7/math_vision/resolve/main/train.json"
VALID_JSON_URL = "https://huggingface.co/TobyYang7/math_vision/resolve/main/valid.json"
TEST_JSON_URL = "https://huggingface.co/TobyYang7/math_vision/resolve/main/test.json"


def download_file(url: str, dest_path: str):
    """Download a file from URL to destination path.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
    """
    if os.path.exists(dest_path):
        logging.info(f"File already exists: {dest_path}")
        return
    
    logging.info(f"Downloading {url} to {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, dest_path)
        logging.info(f"Download complete: {dest_path}")
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")
        raise


def extract_tar_gz(tar_path: str, extract_to: str):
    """Extract a tar.gz file.
    
    Args:
        tar_path: Path to tar.gz file
        extract_to: Directory to extract to
    """
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"Tar file not found: {tar_path}")
    
    logging.info(f"Extracting {tar_path} to {extract_to}")
    os.makedirs(extract_to, exist_ok=True)
    
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_to)
        logging.info(f"Extraction complete")
    except Exception as e:
        logging.error(f"Failed to extract {tar_path}: {e}")
        raise


def download_and_extract_dataset(cache_dir: str, dataset_root: str) -> str:
    """Download and extract the Math Vision dataset.
    
    Args:
        cache_dir: Directory to cache downloaded tar file
        dataset_root: Root directory to extract dataset to
        
    Returns:
        Path to extracted dataset directory
    """
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(dataset_root, exist_ok=True)
    
    # Download tar.gz
    tar_path = os.path.join(cache_dir, "math_vision.tar.gz")
    download_file(DATASET_TAR_URL, tar_path)
    
    # Extract tar.gz
    extract_tar_gz(tar_path, dataset_root)
    
    # Find the extracted directory (might be named math_vision or similar)
    extracted_dir = dataset_root
    if os.path.exists(os.path.join(dataset_root, "math_vision")):
        extracted_dir = os.path.join(dataset_root, "math_vision")
    
    return extracted_dir


def download_json_files(cache_dir: str) -> Dict[str, str]:
    """Download JSON annotation files.
    
    Args:
        cache_dir: Directory to save JSON files
        
    Returns:
        Dictionary mapping split names to JSON file paths
    """
    json_files = {}
    
    for split_name, url in [
        ("train", TRAIN_JSON_URL),
        ("valid", VALID_JSON_URL),
        ("test", TEST_JSON_URL),
    ]:
        dest_path = os.path.join(cache_dir, f"{split_name}.json")
        download_file(url, dest_path)
        json_files[split_name] = dest_path
    
    return json_files


def load_json_data(json_path: str) -> List[Dict]:
    """Load data from JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        List of data samples
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    logging.info(f"Loading data from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logging.info(f"Loaded {len(data)} samples from {json_path}")
    return data


def adjust_image_paths(data: List[Dict], dataset_root: str) -> List[Dict]:
    """Adjust image paths to point to the correct location.
    
    Args:
        data: List of data samples with image_path field
        dataset_root: Root directory where images are located
        
    Returns:
        Data with adjusted image paths
    """
    adjusted_data = []
    
    for sample in data:
        adjusted_sample = sample.copy()
        
        # Get original image path
        original_path = sample.get('image_path', '')
        
        if original_path:
            # Extract just the filename or relative path
            if os.path.isabs(original_path):
                # If absolute path, extract filename
                image_name = os.path.basename(original_path)
            else:
                # If relative path, use as is
                image_name = original_path
            
            # Construct new path relative to dataset_root
            # Try different possible locations
            possible_paths = [
                os.path.join(dataset_root, image_name),
                os.path.join(dataset_root, "math_vision", image_name),
                os.path.join(dataset_root, "images", image_name),
            ]
            
            # Find the first path that exists
            new_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    new_path = path
                    break
            
            # If no existing path found, use the first possibility
            if new_path is None:
                new_path = possible_paths[0]
            
            adjusted_sample['image_path'] = new_path
        
        adjusted_data.append(adjusted_sample)
    
    return adjusted_data


def validate_data_fields(data: List[Dict]) -> List[Dict]:
    """Validate that data has required fields.
    
    Args:
        data: List of data samples
        
    Returns:
        Validated data with required fields
    """
    required_fields = ['solution', 'prompt', 'completion', 'image_path']
    validated_data = []
    
    for i, sample in enumerate(data):
        # Check if all required fields exist
        missing_fields = [field for field in required_fields if field not in sample]
        
        if missing_fields:
            logging.warning(f"Sample {i} missing fields: {missing_fields}")
            # Add missing fields with empty values
            for field in missing_fields:
                sample[field] = ""
        
        # Filter out samples with empty solution
        if sample.get('solution', '').strip():
            validated_data.append(sample)
        else:
            logging.warning(f"Sample {i} has empty solution, skipping")
    
    return validated_data


def process_dataset(json_files: Dict[str, str], dataset_root: str, adjust_paths: bool = True) -> DatasetDict:
    """Process all dataset splits.
    
    Args:
        json_files: Dictionary mapping split names to JSON file paths
        dataset_root: Root directory where images are located
        adjust_paths: Whether to adjust image paths
        
    Returns:
        DatasetDict with processed splits
    """
    dataset_dict = DatasetDict()
    
    for split_name, json_path in json_files.items():
        logging.info(f"Processing {split_name} split")
        
        # Load JSON data
        data = load_json_data(json_path)
        
        # Validate data fields
        data = validate_data_fields(data)
        
        # Adjust image paths if needed
        if adjust_paths:
            data = adjust_image_paths(data, dataset_root)
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(data)
        dataset_dict[split_name] = dataset
        
        logging.info(f"Processed {split_name}: {len(dataset)} samples")
    
    return dataset_dict


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
    total = 0
    
    for split_name in dataset_dict.keys():
        ds = dataset_dict[split_name]
        total += len(ds)
        
        if len(ds) > 0:
            example = ds[0]
            logging.info(f"Example from {split_name} split:")
            logging.info(f"  prompt: {example.get('prompt', '')[:100]}...")
            logging.info(f"  completion: {example.get('completion', '')[:100]}...")
            logging.info(f"  solution: {example.get('solution', '')[:100]}...")
            logging.info(f"  image_path: {example.get('image_path', '')}")
    
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
    
    # Extract math_vision dataset config
    dataset_config = config.get('datasets', {}).get('math_vision', {})
    
    # Try to get from sft config first, then grpo, then fallback to dataset level
    sft_config = dataset_config.get('sft', {})
    grpo_config = dataset_config.get('grpo', {})
    
    # Use sft as default
    mode_config = sft_config if sft_config else grpo_config
    
    return {
        'cache_path': mode_config.get('cache_path') or dataset_config.get('cache_path'),
        'dataset_root': mode_config.get('dataset_root') or dataset_config.get('dataset_root'),
        'image_root': mode_config.get('image_root') or dataset_config.get('image_root'),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess Math Vision dataset")
    
    # Config file options
    parser.add_argument("--config", type=str, default=None,
                        help="Path to yaml config file (e.g., configs/latent_memory/math_vision.yaml)")
    
    # Manual override options
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for preprocessed data (overrides config)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for downloaded raw data (overrides config)")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Root directory for dataset (where images are located)")
    parser.add_argument("--image_root", type=str, default=None,
                        help="Directory for image files (overrides config)")
    
    # Other options
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip downloading data")
    parser.add_argument("--skip_extract", action="store_true",
                        help="Skip extracting tar.gz file")
    parser.add_argument("--no_adjust_paths", action="store_true",
                        help="Don't adjust image paths")
    
    args = parser.parse_args()
    
    # Load config from yaml if provided
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        
        logging.info(f"Loading configuration from {args.config}")
        yaml_config = load_config_from_yaml(args.config)
        
        # Use config values, but allow command-line overrides
        output_dir = args.output_dir or "data/math_vision"
        cache_dir = args.cache_dir or yaml_config.get('cache_path') or ".cache/math_vision"
        dataset_root = args.dataset_root or yaml_config.get('dataset_root') or "dataset/math_vision"
        image_root = args.image_root or yaml_config.get('image_root')
    else:
        # Use command-line arguments or defaults
        output_dir = args.output_dir or "data/math_vision"
        cache_dir = args.cache_dir or ".cache/math_vision"
        dataset_root = args.dataset_root or "dataset/math_vision"
        image_root = args.image_root
    
    # If image_root is not specified, use dataset_root
    if image_root is None:
        image_root = dataset_root
    
    logging.info("=" * 80)
    logging.info("Math Vision Dataset Preprocessing")
    logging.info("=" * 80)
    if args.config:
        logging.info(f"Config file: {args.config}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Cache directory: {cache_dir}")
    logging.info(f"Dataset root: {dataset_root}")
    logging.info(f"Image root: {image_root}")
    logging.info("=" * 80)
    
    # Step 1: Download and extract dataset
    if not args.skip_download and not args.skip_extract:
        extracted_dir = download_and_extract_dataset(cache_dir, dataset_root)
        logging.info(f"Dataset extracted to: {extracted_dir}")
    elif not args.skip_download:
        # Only download tar, don't extract
        tar_path = os.path.join(cache_dir, "math_vision.tar.gz")
        download_file(DATASET_TAR_URL, tar_path)
    else:
        logging.info("Skipping download and extraction")
    
    # Step 2: Download JSON files
    if not args.skip_download:
        json_files = download_json_files(cache_dir)
    else:
        logging.info("Using cached JSON files")
        json_files = {
            "train": os.path.join(cache_dir, "train.json"),
            "valid": os.path.join(cache_dir, "valid.json"),
            "test": os.path.join(cache_dir, "test.json"),
        }
    
    # Step 3: Process dataset
    adjust_paths = not args.no_adjust_paths
    dataset_dict = process_dataset(json_files, image_root, adjust_paths)
    
    # Step 4: Verify data quality
    verify_processed_data(dataset_dict)
    
    # Step 5: Save processed data
    save_processed_data(dataset_dict, output_dir)
    
    logging.info("=" * 80)
    logging.info("Preprocessing complete!")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()

