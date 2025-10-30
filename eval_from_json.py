#!/usr/bin/env python
"""
Standalone evaluation script that loads data directly from a JSON file.
Directly uses LatentMemoryModel.generate() for inference.

Usage:
    python eval_from_json.py \
        --json_path data/mmvp/test.json \
        --model_path test_output/mmvp/weaver/model.safetensors \
        --reasoner_model UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B \
        --weaver_model Qwen/Qwen2.5-1.5B-Instruct \
        --output_dir test_output/mmvp_eval \
        --batch_size 8
"""

import os
import json
import argparse
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
from accelerate import Accelerator
from transformers import GenerationConfig
from peft import LoraConfig
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Import necessary modules from the codebase
from larm.memory_generator.memgen_model import LatentMemoryModel
from larm.memory_generator.utils import load_state_dict_from_safetensor, THINK_SYS_PROMPT
from larm.memory_generator.trainer.utils import extract_answer
from larm.memory_generator.trainer.verifier import verify_solution_equivalence


class JSONDataset(Dataset):
    """Simple dataset that loads data from a JSON file."""
    
    def __init__(self, json_path: str):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        logging.info(f"Loaded {len(self.data)} samples from {json_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model from JSON file")
    
    # Data
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to JSON file containing evaluation data")
    
    # Model paths
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model (.safetensors file)")
    parser.add_argument("--reasoner_model", type=str, required=True,
                        help="Reasoner model name or path")
    parser.add_argument("--weaver_model", type=str, required=True,
                        help="Weaver model name or path")
    parser.add_argument("--trigger_model", type=str, default=None,
                        help="Trigger model name or path (optional)")
    
    # Model configuration
    parser.add_argument("--max_prompt_aug_num", type=int, default=1,
                        help="Maximum prompt augmentation number")
    parser.add_argument("--max_inference_aug_num", type=int, default=3,
                        help="Maximum inference augmentation number")
    parser.add_argument("--prompt_latents_len", type=int, default=8,
                        help="Prompt latents length")
    parser.add_argument("--inference_latents_len", type=int, default=8,
                        help="Inference latents length")
    
    # Entropy filtering configuration
    parser.add_argument("--use_entropy_filter", action="store_true",
                        help="Enable entropy-based augmentation filtering")
    parser.add_argument("--entropy_threshold", type=float, default=1.0,
                        help="Entropy threshold for augmentation filtering")
    
    # Generation configuration
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--do_sample", action="store_true",
                        help="Whether to use sampling for generation")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for generation")
    parser.add_argument("--max_response_length", type=int, default=256,
                        help="Maximum response length")
    
    # Base model evaluation
    parser.add_argument("--base_model", action="store_true",
                        help="Evaluate base model only (without weaver/trigger)")
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--output_filename", type=str, default=None,
                        help="Output JSON filename (default: auto-generated based on model type)")
    
    return parser.parse_args()


def build_model(args):
    """Build the model from configuration."""
    
    logging.info("Building model...")
    
    if args.base_model:
        # Load base model only (for baseline comparison)
        logging.info("Loading BASE MODEL (no weaver/trigger)")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.reasoner_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        processor = AutoProcessor.from_pretrained(args.reasoner_model)
        
        # Add processor and tokenizer as attributes for compatibility
        model.processor = processor
        model.tokenizer = processor.tokenizer
        
        logging.info(f"Base model loaded: {args.reasoner_model}")
        return model
    else:
        # Load full LatentMemoryModel (with weaver/trigger)
        logging.info("Loading LATENT MEMORY MODEL (with weaver/trigger)")
        
        # Weaver PEFT config
        weaver_peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Trigger PEFT config (if using trigger)
        trigger_peft_config = None
        if args.trigger_model is not None:
            trigger_peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
        
        # Build model
        model = LatentMemoryModel(
            reasoner_model_name=args.reasoner_model,
            weaver_model_name=args.weaver_model,
            prompt_latents_len=args.prompt_latents_len,
            inference_latents_len=args.inference_latents_len,
            weaver_peft_config=weaver_peft_config,
            trigger_model_name=args.trigger_model,
            trigger_peft_config=trigger_peft_config,
            max_prompt_aug_num=args.max_prompt_aug_num,
            max_inference_aug_num=args.max_inference_aug_num,
            use_entropy_filter=args.use_entropy_filter,
            entropy_threshold=args.entropy_threshold
        )
        
        # Load trained weights
        if args.model_path and os.path.exists(args.model_path):
            logging.info(f"Loading model weights from: {args.model_path}")
            model_state_dict = load_state_dict_from_safetensor(args.model_path)
            model.load_state_dict(model_state_dict, strict=False)
            logging.info("Model weights loaded successfully")
        else:
            logging.warning(f"Model path not found: {args.model_path}")
        
        return model


def _compute_single_accuracy(pred: str, gt: str, idx: int) -> Dict:
    """
    Compute accuracy for a single prediction (used for multiprocessing).
    Returns: detailed result dict with index
    """
    try:
        # Extract answer from model output (same as training)
        candidate = extract_answer(pred)
        # Verify equivalence using LLM verifier (same as training)
        is_correct = verify_solution_equivalence(candidate, gt)
        return {
            "idx": idx,
            "extracted_answer": candidate,
            "ground_truth": gt,
            "correct": is_correct
        }
    except Exception as e:
        logging.warning(f"Error computing accuracy for sample {idx}: {e}")
        return {
            "idx": idx,
            "extracted_answer": "",
            "ground_truth": gt,
            "correct": False
        }


def compute_accuracy(predictions: List[str], ground_truths: List[str], max_workers: int = 8) -> tuple:
    """
    Compute accuracy using multiprocessing for speed (each call requires OpenAI API).
    Uses the same method as training (extract_answer + verify_solution_equivalence).
    Returns: (accuracy, correct_count, total_count, detailed_results)
    """
    total = len(predictions)
    detailed_results = [None] * total  # Pre-allocate list
    
    # Use multiprocessing to parallelize API calls
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {}
        for idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            future = executor.submit(_compute_single_accuracy, pred, gt, idx)
            futures[future] = idx
        
        # Collect results with progress bar
        with tqdm(total=total, desc="Computing accuracy", unit="sample") as pbar:
            for future in as_completed(futures):
                result = future.result()
                idx = result["idx"]
                detailed_results[idx] = {
                    "extracted_answer": result["extracted_answer"],
                    "ground_truth": result["ground_truth"],
                    "correct": result["correct"]
                }
                pbar.update(1)
    
    # Count correct predictions
    correct = sum(1 for r in detailed_results if r["correct"])
    accuracy = correct / total if total > 0 else 0.0
    
    return accuracy, correct, total, detailed_results


def evaluate(args):
    """Main evaluation function."""
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "eval.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*80)
    logging.info("Starting evaluation from JSON file")
    logging.info(f"JSON path: {args.json_path}")
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info("="*80)
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load dataset
    dataset = JSONDataset(args.json_path)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch  # identity function
    )
    # Don't prepare dataloader to avoid automatic data sharding across processes
    # We want all processes to evaluate all data
    
    # Build model
    model = build_model(args)
    model = accelerator.prepare_model(model=model, evaluation_mode=True)
    model.eval()
    
    # Generation config
    # Prefer chat end token (<|im_end|>) if available
    try:
        im_end_ids = model.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if isinstance(im_end_ids, list) and len(im_end_ids) == 1:
            eos_id = im_end_ids[0]
        else:
            eos_id = model.tokenizer.eos_token_id
    except Exception:
        eos_id = model.tokenizer.eos_token_id
    
    generation_config = GenerationConfig(
        max_new_tokens=args.max_response_length,
        do_sample=args.do_sample,
        temperature=args.temperature,
        pad_token_id=model.tokenizer.pad_token_id,
        eos_token_id=eos_id
    )
    
    # Log augmentation configuration
    if args.base_model:
        augment_info = {
            'model_type': 'base_model',
            'reasoner_model': args.reasoner_model,
        }
    else:
        augment_info = {
            'model_type': 'latent_memory_model',
            'max_prompt_aug_num': args.max_prompt_aug_num,
            'max_inference_aug_num': args.max_inference_aug_num,
            'prompt_latents_len': args.prompt_latents_len,
            'inference_latents_len': args.inference_latents_len,
        }
    logging.info(f"Model configuration: {augment_info}")
    
    # Store all results (only on main process)
    all_results = []
    all_predictions = []
    all_ground_truths = []
    
    # Evaluation loop
    logging.info("Starting evaluation loop...")
    total_samples = len(dataset)
    
    # Create progress bar for samples (not batches)
    pbar = tqdm(total=total_samples, desc="Evaluating", unit="sample", disable=not accelerator.is_main_process)
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                # Prepare messages with images
                messages_list = []
                images_list = []
                
                for example in batch:
                    prompt_text = example["prompt"]
                    
                    # Load image if present
                    img = None
                    try:
                        image_path = example.get("image_path")
                        if image_path and os.path.exists(image_path):
                            img = Image.open(image_path).convert("RGB")
                    except Exception as e:
                        logging.warning(f"Failed to load image: {e}")
                    
                    # Construct message
                    if img is not None:
                        messages_list.append([
                            {"role": "system", "content": THINK_SYS_PROMPT},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": img},
                                    {"type": "text", "text": prompt_text},
                                ],
                            }
                        ])
                        images_list.append(img)
                    else:
                        messages_list.append([
                            {"role": "system", "content": THINK_SYS_PROMPT},
                            {"role": "user", "content": prompt_text},
                        ])
                        images_list.append(None)
                
                # Use processor to tokenize
                processor = model.processor
                has_images = any(img is not None for img in images_list)
                
                # Set padding side to left for decoder-only models
                original_padding_side = processor.tokenizer.padding_side
                processor.tokenizer.padding_side = "left"
                
                # Apply chat template
                texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) 
                         for m in messages_list]
                
                # Tokenize
                if has_images:
                    inputs = processor(
                        text=texts,
                        images=images_list,
                        return_tensors="pt",
                        padding=True,
                    )
                else:
                    inputs = processor(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=True
                    )
                
                # Restore original padding side
                processor.tokenizer.padding_side = original_padding_side
                
                # Move to device
                input_ids = inputs["input_ids"].to(accelerator.device)
                attention_mask = inputs["attention_mask"].to(accelerator.device)
                
                # Prepare generation kwargs
                gen_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "generation_config": generation_config,
                }
                
                # Add vision inputs if present
                if "pixel_values" in inputs:
                    gen_kwargs["pixel_values"] = inputs["pixel_values"].to(accelerator.device).to(torch.bfloat16)
                if "image_grid_thw" in inputs:
                    gen_kwargs["image_grid_thw"] = inputs["image_grid_thw"].to(accelerator.device)
                
                # Generate
                augmentation_info_list = []  # Store augmentation positions for each sample
                
                if args.base_model:
                    # Base model: standard generate
                    gen_output = model.generate(**gen_kwargs)
                    # Extract responses (remove prompt part)
                    prompt_len = input_ids.size(1)
                    responses_ids = gen_output[:, prompt_len:]
                    # No augmentation info for base model
                    augmentation_info_list = [None] * len(responses_ids)
                else:
                    # LatentMemoryModel: custom generate with weaver, get augmentation mask
                    gen_kwargs["return_augmentation_mask"] = True
                    gen_output, augmentation_pos = model.generate(**gen_kwargs)
                    
                    # Extract responses (remove prompt part)
                    prompt_len = input_ids.size(1)
                    responses_ids = gen_output[:, prompt_len:]
                    
                    # Extract augmentation positions for each sample in batch
                    for i in range(len(responses_ids)):
                        response_len = responses_ids[i].size(0)
                        aug_mask = augmentation_pos[i]  # augmentation mask for this sample
                        
                        # Find positions where augmentation occurred (mask value > 0)
                        aug_positions = []
                        for pos in range(min(len(aug_mask), response_len)):
                            if aug_mask[pos] > 0:
                                aug_positions.append(pos)
                        
                        # Format as "pos/total_len" for each augmentation
                        if len(aug_positions) > 0:
                            aug_info = ", ".join([f"{pos}/{response_len}" for pos in aug_positions])
                        else:
                            aug_info = "no_augmentation"
                        
                        augmentation_info_list.append(aug_info)
                
                # Decode
                responses = model.tokenizer.batch_decode(responses_ids, skip_special_tokens=True)
                
                # Record results (only on main process to avoid duplication)
                if accelerator.is_main_process:
                    for example, response, aug_info in zip(batch, responses, augmentation_info_list):
                        prompt = example["prompt"]
                        ground_truth = example.get("solution", example.get("completion", ""))
                        
                        result = {
                            "prompt": prompt,
                            "response": response,
                            "ground_truth": ground_truth,
                        }
                        
                        # Add augmentation info for non-base models
                        if not args.base_model and aug_info is not None:
                            result["augmentation_positions"] = aug_info
                        
                        all_results.append(result)
                        all_predictions.append(response)
                        all_ground_truths.append(ground_truth)
                        
                        logging.debug(f"Prompt: {prompt[:100]}...")
                        logging.debug(f"Response: {response}")
                        logging.debug(f"Ground truth: {ground_truth}")
                        if not args.base_model and aug_info is not None:
                            logging.debug(f"Augmentation positions: {aug_info}")
                        logging.debug("-" * 80)
                    
                    # Update progress bar with number of samples processed
                    pbar.update(len(batch))
                    
            except Exception as e:
                logging.error(f"Error during generation: {e}")
                import traceback
                traceback.print_exc()
                if accelerator.is_main_process:
                    pbar.update(len(batch))
    
    # Close progress bar
    if accelerator.is_main_process:
        pbar.close()
    
    # Only main process computes metrics and saves results
    if accelerator.is_main_process:
        # Compute metrics using the same method as training
        logging.info("Computing accuracy using training method (extract_answer + verify_solution_equivalence)...")
        accuracy, correct_count, total_count, detailed_results = compute_accuracy(all_predictions, all_ground_truths)
        
        # Merge detailed results into all_results
        for i, (result, detail) in enumerate(zip(all_results, detailed_results)):
            result.update(detail)
        
        # Generate output filename
        if args.output_filename:
            # Use user-specified filename
            output_filename = args.output_filename
            # Add .json extension if not present
            if not output_filename.endswith('.json'):
                output_filename += '.json'
        elif args.base_model:
            # For base model, include model name in filename
            # Convert model path to safe filename: "Qwen/Qwen2.5-VL-7B" -> "Qwen_Qwen2.5_VL_7B"
            model_name = args.reasoner_model.replace("/", "_").replace("-", "_").replace(".", "_")
            output_filename = f"answer_{model_name}.json"
        else:
            # For trained model, use default filename
            output_filename = "answer.json"
        
        output_file = os.path.join(args.output_dir, output_filename)
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write each result with indent
            for result in all_results:
                f.write(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
            
            # Write final summary at the end
            summary = {
                "summary": {
                    "total_samples": total_count,
                    "correct": correct_count,
                    "accuracy": accuracy,
                    "augment_config": augment_info
                }
            }
            f.write(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
        
        logging.info("="*80)
        logging.info("Evaluation completed!")
        logging.info(f"Total samples: {total_count}")
        logging.info(f"Correct: {correct_count}")
        logging.info(f"Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
        logging.info(f"Results saved to: {output_file}")
        logging.info("="*80)
        
        return {"accuracy": accuracy}
    else:
        return {}


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
