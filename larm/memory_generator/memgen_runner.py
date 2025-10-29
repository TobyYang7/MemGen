import os
import logging
from torch.utils.data import DataLoader
from datasets import Dataset
from accelerate import Accelerator
from transformers import PreTrainedTokenizerBase
from trl import SFTTrainer, SFTConfig, GRPOConfig
from trl.models import unwrap_model_for_generation

from typing import Tuple, Dict, List
from tqdm import tqdm

from larm.task.base_runner import BaseRunner
from larm.common.registry import registry
from larm.common.config import Config
from larm.data.interactions.base_interaction import (
    InteractionConfig,
    InteractionManager,
    InteractionDataProto
)

from .memgen_model import LatentMemoryModel
from .trainer.weaver_grpo_trainer import WeaverGRPOTrainer
from .trainer.trigger_grpo_trainer import TriggerGRPOTrainer
from .utils import (
    fix_model_parameters,
    open_model_parameters,
    EvalConfig,
    StaticEvalRecorder,
    DynamicEvalRecorder
)
from PIL import Image
import torch


@registry.register_runner("latmem")
class LatentMemoryRunner(BaseRunner):

    def __init__(
        self,
        model: LatentMemoryModel,
        processing_class: PreTrainedTokenizerBase,
        datasets_dict: Dict,
        configs: Config,
        env_and_gens_dict: Dict
    ):
        super().__init__(
            model,
            processing_class,
            datasets_dict,
            configs,
            env_and_gens_dict
        )
        # parse configs
        self._parse_configs(configs.run_cfg)

        # initialize envs and generation managers
        dataset_config = configs.datasets_cfg[self.dataset_name]
        self.env = self.env_cls(dataset_config)

        # partition datasets
        self.weaver_train_dataset, self.trigger_train_dataset = self._parse_train_dataset(self.dataset_dict["train"])
        self.valid_dataset = self.dataset_dict["valid"]
        self.test_dataset = self.dataset_dict["test"]

        self.weaver_train_dataset = self._filter_dataset(self.weaver_train_dataset)
        self.trigger_train_dataset = self._filter_dataset(self.trigger_train_dataset)
        self.valid_dataset = self._filter_dataset(self.valid_dataset)

        # initialize generation manager
        self.generation_manager: InteractionManager = self.gen_cls(
            self.processing_class, self.model, self.interaction_config
        )

        # VIS: build multimodal features if present in dataset
        if self.weaver_train_dataset.column_names and any(k in self.weaver_train_dataset.column_names for k in ("image_path",)):
            self.weaver_train_dataset = self._prepare_mm_features(self.weaver_train_dataset)
        if self.valid_dataset.column_names and any(k in self.valid_dataset.column_names for k in ("image_path",)):
            self.valid_dataset = self._prepare_mm_features(self.valid_dataset)
        if self.test_dataset.column_names and any(k in self.test_dataset.column_names for k in ("image_path",)):
            self.test_dataset = self._prepare_mm_features(self.test_dataset)

    def _parse_train_dataset(self, train_dataset: Dataset) -> Tuple[Dataset, Dataset]:
        trigger_trainset_size = min(500, len(train_dataset))
        return train_dataset, train_dataset.select(range(trigger_trainset_size))

    def _filter_dataset(self, dataset: Dataset) -> Dataset:
        """Filter the dataset based on maximum sequence length.

        The maximum length depends on the training mode and method:
        - For Weaver SFT training: use `weaver_training_args.max_length`.
        - For Weaver GRPO training: use `weaver_training_args.max_prompt_length`.
        - For Trigger GRPO training: use `trigger_training_args.max_prompt_length`.

        Any sample exceeding the maximum length is filtered out.

        Args:
            dataset (Dataset): The input dataset to be filtered.

        Returns:
            Dataset: A filtered dataset containing only samples within the max length.
        """
        tokenizer = self.processing_class

        # Determine max length based on training mode
        max_len = 1024
        if self.train_weaver and self.train_weaver_method == "sft":
            max_len = self.weaver_training_args.max_length
        elif self.train_weaver and self.train_weaver_method == "grpo":
            max_len = self.weaver_training_args.max_prompt_length
        elif self.train_trigger and self.train_trigger_method == "grpo":
            max_len = self.trigger_training_args.max_prompt_length
        else:
            raise ValueError("Wrong training mode.")

        original_size = len(dataset)
        logging.info(f"[Filter] Starting filter with max_len={max_len}, dataset_size={original_size}")

        # Pre-extract tokenizer outside filter_func for better performance
        plain_tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)

        # Function to filter out samples exceeding max length
        def filter_func(sample):
            # Check if sample is None or not a dict
            if sample is None or not isinstance(sample, dict):
                return False

            if "prompt" in sample and sample["prompt"] is not None:
                # Use pre-extracted tokenizer to avoid repeated getattr calls
                encoded = plain_tokenizer(sample["prompt"], add_special_tokens=True)
                return len(encoded["input_ids"]) < max_len
            elif "messages" in sample and sample["messages"] is not None:
                conversation = tokenizer.apply_chat_template(sample["messages"][:2], tokenize=True)
                return len(conversation) < max_len
            return True

        # Apply filtering with optimized settings
        logging.info(f"[Filter] Starting to apply filter function...")
        dataset = dataset.filter(
            filter_func,
            num_proc=None,  # 使用单进程（None 表示主进程，避免子进程序列化问题）
            load_from_cache_file=False,  # 禁用缓存避免卡住
            desc="Filter"
        )
        logging.info(f"[Filter] Filter function completed")

        filtered_size = len(dataset)
        logging.info(f"[Filter] Completed: {original_size} -> {filtered_size} samples (filtered out {original_size - filtered_size})")

        return dataset

    def _prepare_mm_features(self, dataset: Dataset) -> Dataset:
        processor = self.processing_class  # AutoProcessor for VL models
        tokenizer = getattr(processor, "tokenizer", processor)

        # Cache vision token IDs to avoid repeated encoding
        # For Qwen2.5-VL, check what vision tokens are actually used
        vision_start_ids = tokenizer.encode("<|vision_start|>", add_special_tokens=False)
        vision_end_ids = tokenizer.encode("<|vision_end|>", add_special_tokens=False)

        # Also try alternative vision token names for Qwen2.5-VL
        if len(vision_start_ids) == 0:
            # Try common alternatives
            for token in ["<|image_pad|>", "<|vision_pad|>", "<image>", "<img>"]:
                test_ids = tokenizer.encode(token, add_special_tokens=False)
                if len(test_ids) > 0:
                    vision_start_ids = test_ids
                    logging.info(f"[MM_Features] Found vision start token: {token} -> {test_ids}")
                    break

        # logging.info(f"[MM_Features] Vision token IDs: start={vision_start_ids}, end={vision_end_ids}")
        # logging.info(f"[MM_Features] Tokenizer special tokens: {tokenizer.special_tokens_map}")

        def _encode(example: Dict) -> Dict:
            prompt = example.get("prompt")
            completion = example.get("completion")
            image_path = example.get("image_path")

            image = None
            # Debug: log only one image path candidate and its existence
            try:
                _path_logged = getattr(_encode, "_path_logged_count", 0)
            except Exception:
                _path_logged = 0
            if _path_logged < 1:
                exists_str = os.path.exists(image_path) if image_path is not None else "N/A"
                logging.info(f"[MM_Features] example image_path: {image_path}, exists={exists_str}")
                try:
                    _encode._path_logged_count = _path_logged + 1
                except Exception:
                    pass
            if image_path is not None and os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception:
                    image = None

            if image is not None:
                # Use Qwen2.5-VL's recommended message format
                # Build messages with image and text following official API
                from larm.memory_generator.utils import THINK_SYS_PROMPT
                messages = [
                    {
                        "role": "system",
                        "content": THINK_SYS_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                # Apply chat template first to insert vision tokens
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                # Debug: log formatted text (once)
                try:
                    _template_logged = getattr(_encode, "_template_logged_count", 0)
                except Exception:
                    _template_logged = 0
                if _template_logged < 1:
                    logging.info(f"[MM_Features] Formatted text after apply_chat_template: {text[:200]}...")
                    try:
                        _encode._template_logged_count = _template_logged + 1
                    except Exception:
                        pass
                # Then process with image
                enc_prompt = processor(text=[text], images=[image], return_tensors="pt", padding=False)
                # Debug: confirm (log once) that we are encoding with image
                try:
                    _enc_logged = getattr(_encode, "_enc_with_img_logged_count", 0)
                except Exception:
                    _enc_logged = 0
                if _enc_logged < 1:
                    logging.info(f"[MM_Features] encoding with image: path={image_path}, size={getattr(image, 'size', None)}")
                    logging.info(f"[MM_Features] enc_prompt keys: {list(enc_prompt.keys())}")
                    # Check what processor returns
                    for key, val in enc_prompt.items():
                        if isinstance(val, torch.Tensor):
                            logging.info(f"  {key}: Tensor{tuple(val.shape)}")
                        else:
                            logging.info(f"  {key}: {type(val)}")
                    try:
                        _encode._enc_with_img_logged_count = _enc_logged + 1
                    except Exception:
                        pass
                prompt_ids = enc_prompt["input_ids"][0]
                prompt_mask = enc_prompt["attention_mask"][0]
                pixel_values = enc_prompt.get("pixel_values")
                image_grid_thw = enc_prompt.get("image_grid_thw")
                if pixel_values is not None:
                    pixel_values = pixel_values[0]
                if image_grid_thw is not None:
                    image_grid_thw = image_grid_thw[0]
            else:
                enc_prompt = processor(text=[prompt], return_tensors="pt", padding=False)
                prompt_ids = enc_prompt["input_ids"][0]
                prompt_mask = enc_prompt["attention_mask"][0]
                pixel_values = None
                image_grid_thw = None

            tokenizer = getattr(processor, "tokenizer", processor)
            enc_comp = tokenizer(text=[completion], add_special_tokens=False, return_tensors="pt")
            comp_ids = enc_comp["input_ids"][0]

            input_ids = torch.cat([prompt_ids, comp_ids], dim=0)
            attention_mask = torch.cat([prompt_mask, torch.ones_like(comp_ids)], dim=0)
            labels = torch.cat([torch.full_like(prompt_ids, -100), comp_ids.clone()], dim=0)

            # For Qwen2.5-VL: image tokens are NOT explicitly in input_ids
            # Instead, the model uses image_grid_thw to locate image features
            # So image_token_mask may remain all False, which is correct
            image_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            if image is not None:
                ids_list = prompt_ids.tolist()

                # Debug: log first example's token ids
                try:
                    _debug_logged = getattr(_encode, "_debug_token_ids_logged", 0)
                except Exception:
                    _debug_logged = 0
                if _debug_logged < 1:
                    logging.info(f"[MM_Features] Sample prompt_ids (first 50): {ids_list[:50]}")
                    logging.info(f"[MM_Features] Looking for vision_start_ids: {vision_start_ids}, vision_end_ids: {vision_end_ids}")
                    # Decode to show tokens
                    decoded_tokens = [tokenizer.decode([tid]) for tid in ids_list[:50]]
                    logging.info(f"[MM_Features] Decoded tokens (first 50): {decoded_tokens}")
                    try:
                        _encode._debug_token_ids_logged = 1
                    except Exception:
                        pass

                # For Qwen2.5-VL: check for <|vision_pad|> or <|image_pad|> tokens
                vision_pad_id = tokenizer.encode("<|vision_pad|>", add_special_tokens=False)
                image_pad_id = tokenizer.encode("<|image_pad|>", add_special_tokens=False)

                if _debug_logged < 1:
                    logging.info(f"[MM_Features] vision_pad_id: {vision_pad_id}, image_pad_id: {image_pad_id}")

                # Look for vision/image pad tokens in the sequence
                target_ids = []
                if len(vision_pad_id) > 0:
                    target_ids.append(vision_pad_id[0])
                if len(image_pad_id) > 0:
                    target_ids.append(image_pad_id[0])

                if len(target_ids) > 0:
                    # Mark all positions with vision_pad or image_pad tokens
                    for i, token_id in enumerate(ids_list):
                        if token_id in target_ids:
                            image_token_mask[i] = True

                    if _debug_logged < 1:
                        num_marked = image_token_mask[:len(ids_list)].sum().item()
                        logging.info(f"[MM_Features] Marked {num_marked} image pad tokens (ids={target_ids})")

                # Also try vision_start/vision_end markers
                def find_subseq(seq, sub):
                    n, m = len(seq), len(sub)
                    for i in range(0, n - m + 1):
                        if seq[i:i+m] == sub:
                            return i
                    return -1

                s_idx = find_subseq(ids_list, vision_start_ids) if len(vision_start_ids) > 0 else -1
                e_idx = find_subseq(ids_list, vision_end_ids) if len(vision_end_ids) > 0 else -1

                if _debug_logged < 1:
                    logging.info(f"[MM_Features] Found indices: start_idx={s_idx}, end_idx={e_idx}")

                if s_idx != -1 and e_idx != -1 and e_idx >= s_idx:
                    image_token_mask[s_idx:e_idx+len(vision_end_ids)] = True
                    if _debug_logged < 1:
                        num_marked = image_token_mask.sum().item()
                        logging.info(f"[MM_Features] Set image_token_mask[{s_idx}:{e_idx+len(vision_end_ids)}] = True (total marked: {num_marked})")

                # Final check
                if _debug_logged < 1:
                    total_marked = image_token_mask.sum().item()
                    if total_marked == 0:
                        logging.warning(f"[MM_Features] Image present but no vision tokens found in input_ids! This may be expected for Qwen2.5-VL.")

            out = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "image_token_mask": image_token_mask,
            }
            if pixel_values is not None:
                out["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                out["image_grid_thw"] = image_grid_thw
            return out

        # Disable multiprocessing to avoid subprocess crashes with PIL/Processor
        # num_proc=None means no multiprocessing (runs in main process)
        logging.info(f"[MM_Features] Processing {len(dataset)} samples with multimodal features...")
        # Safely get columns to remove, checking if column_names exists
        columns_to_remove = []
        if dataset.column_names:
            columns_to_remove = [c for c in dataset.column_names if c not in ("prompt", "completion", "solution", "image_path")]

        dataset = dataset.map(
            _encode,
            remove_columns=columns_to_remove,
            num_proc=None,  # Disable multiprocessing to avoid crashes
            load_from_cache_file=False,  # Disable cache due to closure serialization issues
            desc="Preparing multimodal features"
        )
        logging.info(f"[MM_Features] Completed processing {len(dataset)} samples.")

        # Log distribution statistics of image_token_mask after processing
        try:
            sample_size = min(1000, len(dataset))
            counted = 0
            num_with_image_tokens = 0
            num_with_pixel_values = 0
            total_true_tokens = 0
            min_true_tokens = None
            max_true_tokens = None

            for i in range(sample_size):
                ex = dataset[i]
                mask = ex.get("image_token_mask", None)
                if mask is None:
                    continue

                # Check if this sample has actual pixel_values (real image data)
                pixel_values = ex.get("pixel_values", None)
                if pixel_values is not None:
                    num_with_pixel_values += 1

                # Support torch.Tensor, list, or numpy-like
                try:
                    if isinstance(mask, torch.Tensor):
                        cnt = int(mask.sum().item())
                    else:
                        cnt = int(sum(mask))
                except Exception:
                    try:
                        cnt = int(mask.sum())
                    except Exception:
                        cnt = 0

                counted += 1
                total_true_tokens += cnt
                if min_true_tokens is None or cnt < min_true_tokens:
                    min_true_tokens = cnt
                if max_true_tokens is None or cnt > max_true_tokens:
                    max_true_tokens = cnt
                if cnt > 0:
                    num_with_image_tokens += 1

            if counted > 0:
                mean_true_tokens = total_true_tokens / counted
                ratio_with_image = num_with_image_tokens / counted
                ratio_with_pixels = num_with_pixel_values / counted
                logging.info(
                    f"[MM_Features] Statistics on {counted} samples:"
                )
                logging.info(
                    f"  - Samples with pixel_values (actual images): {num_with_pixel_values}/{counted} ({ratio_with_pixels:.1%})"
                )
                logging.info(
                    f"  - Samples with vision tokens in input_ids: {num_with_image_tokens}/{counted} ({ratio_with_image:.1%})"
                )
                logging.info(
                    f"  - Vision tokens per sample: min/mean/max = {min_true_tokens}/{mean_true_tokens:.2f}/{max_true_tokens}"
                )
                if num_with_pixel_values > 0 and num_with_image_tokens == 0:
                    logging.warning(
                        f"[MM_Features] WARNING: {num_with_pixel_values} samples have images (pixel_values) "
                        f"but NO vision tokens found in input_ids. This is EXPECTED for Qwen2.5-VL, which uses "
                        f"image_grid_thw instead of explicit vision tokens in input_ids."
                    )
            else:
                logging.info("[MM_Features] image_token_mask: no samples counted (mask missing)")
        except Exception as e:
            logging.warning(f"[MM_Features] Failed to summarize image_token_mask distribution: {e}")

        # Print a complete example after processing
        try:
            if len(dataset) > 0:
                example = dataset[0]
                logging.info("=" * 80)
                logging.info("[MM_Features] COMPLETE EXAMPLE AFTER PROCESSING:")
                logging.info("=" * 80)

                # Print each field
                for key, value in example.items():
                    if key == "pixel_values":
                        if value is not None:
                            if isinstance(value, torch.Tensor):
                                logging.info(f"  {key}: Tensor{tuple(value.shape)} dtype={value.dtype}")
                            else:
                                logging.info(f"  {key}: {type(value)}")
                        else:
                            logging.info(f"  {key}: None")

                    elif key == "image_grid_thw":
                        if value is not None:
                            if isinstance(value, torch.Tensor):
                                logging.info(f"  {key}: Tensor{tuple(value.shape)} = {value.tolist()}")
                            else:
                                logging.info(f"  {key}: {value}")
                        else:
                            logging.info(f"  {key}: None")

                    elif key == "input_ids":
                        if isinstance(value, torch.Tensor):
                            logging.info(f"  {key}: Tensor{tuple(value.shape)}")
                            # Show first 50 token IDs
                            logging.info(f"    Token IDs (first 50): {value.tolist()[:50]}")
                            # Decode each token individually to show special tokens
                            try:
                                tokens_display = []
                                for i in range(min(50, len(value))):
                                    token_id = value[i].item()
                                    token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                                    # Show token with its ID
                                    tokens_display.append(f"{token_str}[{token_id}]")
                                logging.info(f"    Tokens (first 50): {' '.join(tokens_display)}")
                            except Exception as e:
                                logging.warning(f"    Failed to decode tokens: {e}")
                        else:
                            logging.info(f"  {key}: {value[:50]}...")

                    elif key == "labels":
                        if isinstance(value, torch.Tensor):
                            # Count non-ignored labels
                            num_labels = (value != -100).sum().item()
                            logging.info(f"  {key}: Tensor{tuple(value.shape)}, non-ignored labels={num_labels}")
                            # Show where labels start (first non -100 position)
                            non_ignore_indices = (value != -100).nonzero(as_tuple=True)[0]
                            if len(non_ignore_indices) > 0:
                                first_label_idx = non_ignore_indices[0].item()
                                logging.info(f"    Labels start at position {first_label_idx}")
                                # Decode the label tokens (non -100 parts)
                                try:
                                    label_tokens = value[value != -100][:30]  # First 30 label tokens
                                    tokens_display = []
                                    for token_id in label_tokens:
                                        token_str = tokenizer.decode([token_id.item()], skip_special_tokens=False)
                                        tokens_display.append(f"{token_str}[{token_id.item()}]")
                                    logging.info(f"    Label tokens (first 30): {' '.join(tokens_display)}")
                                except Exception as e:
                                    logging.warning(f"    Failed to decode label tokens: {e}")
                        else:
                            logging.info(f"  {key}: {value[:50]}...")

                    elif key == "attention_mask":
                        if isinstance(value, torch.Tensor):
                            num_valid = int(value.sum().item())
                            logging.info(f"  {key}: Tensor{tuple(value.shape)}, valid positions={num_valid}/{len(value)}")
                            logging.info(f"    First 50 values: {value.tolist()[:50]}")
                        else:
                            logging.info(f"  {key}: {value[:50]}...")

                    elif key == "image_token_mask":
                        if isinstance(value, torch.Tensor):
                            num_image_tokens = int(value.sum().item())
                            logging.info(f"  {key}: Tensor{tuple(value.shape)}, image token positions={num_image_tokens}")
                            if num_image_tokens > 0:
                                # Show which positions are image tokens
                                image_positions = value.nonzero(as_tuple=True)[0].tolist()[:20]
                                logging.info(f"    Image token positions (first 20): {image_positions}")
                            else:
                                logging.info(f"    No image tokens found in input_ids (expected for Qwen2.5-VL)")
                        else:
                            logging.info(f"  {key}: {value[:50]}...")

                    else:
                        # Other fields (prompt, completion, solution, image_path)
                        if isinstance(value, str):
                            logging.info(f"  {key}: '{value[:200]}{'...' if len(value) > 200 else ''}'")
                        else:
                            logging.info(f"  {key}: {value}")

                logging.info("=" * 80)
        except Exception as e:
            logging.warning(f"[MM_Features] Failed to print example: {e}")

        return dataset

    # ===== train weaver =====
    def _create_weaver_trainer(self):

        # SFT Trainer
        if self.train_weaver_method == "sft":
            # JSONL log path under save_dir
            weaver_trainer = SFTTrainer(
                model=self.model,
                args=self.weaver_training_args,
                train_dataset=self.weaver_train_dataset,
                eval_dataset=self.valid_dataset,
                processing_class=self.processing_class,
            )

        # GRPO Trainer
        elif self.train_weaver_method == 'grpo':

            reward_funcs = []     # get reward funcs for GRPO Trainer
            for reward_name in self.weaver_reward_names:
                reward_funcs.append(self.env_cls.get_reward_func(reward_name))

            weaver_trainer = WeaverGRPOTrainer(
                model=self.model,
                reward_funcs=reward_funcs,
                args=self.weaver_training_args,
                train_dataset=self.weaver_train_dataset,
                eval_dataset=self.valid_dataset,
                processing_class=self.processing_class,
                env_class=self.env_cls,
                env_main_config=self.configs.datasets_cfg[self.dataset_name],
                generation_manager=self.generation_manager
            )
        else:
            raise ValueError("Unsupported weaver training method.")

        return weaver_trainer

    def _train_weaver(self):

        # fix trigger parameters
        fix_model_parameters(self.model.trigger)

        # train weaver
        weaver_trainer = self._create_weaver_trainer()
        weaver_trainer.train()
        weaver_trainer.save_model()   # save the best model

        # remove checkpoints and save weaver
        output_dir = weaver_trainer.args.output_dir
        self._remove_trainer_ckpts(output_dir)

        # open trigger parameters
        open_model_parameters(self.model.trigger)

    # ===== train trigger =====

    def _create_trigger_trainer(self):

        # get reward funcs for RL Trainer
        reward_funcs = []
        for reward_name in self.trigger_reward_names:
            reward_funcs.append(self.env_cls.get_reward_func(reward_name))

        # build trainer
        trigger_trainer = TriggerGRPOTrainer(
            model=self.model,
            processing_class=self.processing_class,
            train_dataset=self.trigger_train_dataset,
            eval_dataset=self.valid_dataset,
            reward_funcs=reward_funcs,
            args=self.trigger_training_args
        )

        return trigger_trainer

    def _train_trigger(self):

        # fix weaver parameters
        fix_model_parameters(self.model.weaver)

        # train trigger
        trigger_trainer = self._create_trigger_trainer()
        trigger_trainer.train()
        trigger_trainer.save_model()     # save the best model

        # remove checkpoints and save weaver
        output_dir = trigger_trainer.args.output_dir
        self._remove_trainer_ckpts(output_dir)

        # open trigger parameters
        open_model_parameters(self.model.weaver)

    # ===== train weaver/trigger =====
    def train(self):
        # train weaver
        if self.train_weaver:
            self._train_weaver()

        # train trigger
        if self.train_trigger:
            self._train_trigger()

    # ===== evaluate =====
    def _static_evaluate(self):

        accelerator = Accelerator()
        writer = self._create_tensorboard(mode="evaluate")

        batch_size = self.eval_config.batch_size
        output_dir = self.eval_config.output_dir
        generation_config = self.eval_config.generation_config
        _actual_tokenizer = getattr(self.processing_class, 'tokenizer', self.processing_class)
        generation_config.eos_token_id = _actual_tokenizer.eos_token_id

        # prepare dataset and dataloader
        test_dataloader = accelerator.prepare(DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch  # use the identity function
        ))

        # prepare model
        model_wrapped = accelerator.prepare_model(model=self.model, evaluation_mode=True)
        model_wrapped.eval()

        # construct eval recorder
        test_funcs = [self.env_cls.get_reward_func("accuracy")]
        save_file = os.path.join(output_dir, "answer.json")
        recorder = StaticEvalRecorder(compute_metrics=test_funcs, writer=writer, log_file=save_file)
        
        # Prepare augment configuration info for recording
        augment_info = {
            'max_prompt_aug_num': self.model.max_prompt_aug_num,
            'max_inference_aug_num': self.model.max_inference_aug_num,
            'prompt_latents_len': self.model.weaver.prompt_latents_num if self.model.weaver else 0,
            'inference_latents_len': self.model.weaver.inference_latents_num if self.model.weaver else 0,
        }
        logging.info(f"[EVAL] Augment configuration: {augment_info}")

        # batch generation
        for test_batch in tqdm(test_dataloader):
            with unwrap_model_for_generation(
                model_wrapped, accelerator
            ) as unwrapped_model:
                # construct InteractionDataProto object with image support
                prompts = [x["prompt"] for x in test_batch]
                
                # Check if batch contains images and prepare messages
                messages_list = []
                images_list = []
                any_image = False
                
                from larm.memory_generator.utils import THINK_SYS_PROMPT
                for i, example in enumerate(test_batch):
                    prompt_text = example["prompt"]
                    
                    # Try to load image if present
                    img = None
                    try:
                        image_path = example.get("image_path") if isinstance(example, dict) else None
                        if image_path is not None:
                            img = Image.open(image_path).convert("RGB")
                    except Exception as e:
                        logging.warning(f"Failed to load image for sample {i}: {e}")
                    
                    if img is not None:
                        any_image = True
                        messages_list.append([
                            {
                                "role": "system",
                                "content": THINK_SYS_PROMPT
                            },
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
                            {
                                "role": "system",
                                "content": THINK_SYS_PROMPT
                            },
                            {
                                "role": "user",
                                "content": prompt_text,
                            }
                        ])
                        images_list.append(None)
                
                # Prefer model.processor when available for VLMs
                processor = getattr(self.model, "processor", None)
                proc = processor if processor is not None else self.processing_class
                
                # Check for mixed modality batches
                if any_image and any(x is None for x in images_list):
                    logging.error("\033[91m[ERROR] Mixed image and text-only samples detected in the same batch. "
                                  "Please group samples by modality (all-with-image or all-text).\033[0m")
                    raise ValueError("Mixed image and text-only samples in one batch are not supported. Group by modality.")
                
                # Build encodings
                if hasattr(proc, "apply_chat_template"):
                    # Each element in messages_list is already a list of messages (system + user)
                    texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
                else:
                    # Fallback: use the plain prompt texts
                    texts = prompts
                
                if any_image:
                    prompt_inputs = proc(
                        text=texts,
                        images=images_list,
                        return_tensors="pt",
                        padding=True,
                    )
                else:
                    prompt_inputs = proc(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="left",
                        add_special_tokens=True
                    )
                
                prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
                gen_batch = InteractionDataProto()
                gen_batch.batch["input_ids"] = prompt_ids.to(accelerator.device)
                gen_batch.batch["attention_mask"] = prompt_mask.to(accelerator.device)
                gen_batch.no_tensor_batch["initial_prompts"] = prompts
                
                # Attach vision tensors if present
                if "pixel_values" in prompt_inputs:
                    try:
                        vision_pixel_values = prompt_inputs["pixel_values"].to(accelerator.device).to(torch.bfloat16)
                    except Exception:
                        vision_pixel_values = prompt_inputs["pixel_values"].to(accelerator.device)
                    gen_batch.batch["pixel_values"] = vision_pixel_values
                    logging.info(f"[EVAL] Vision inputs added - pixel_values shape: {vision_pixel_values.shape}")
                
                if "image_grid_thw" in prompt_inputs:
                    vision_image_grid_thw = prompt_inputs["image_grid_thw"].to(accelerator.device)
                    gen_batch.batch["image_grid_thw"] = vision_image_grid_thw
                    logging.info(f"[EVAL] Vision inputs added - image_grid_thw shape: {vision_image_grid_thw.shape}")

                # generation manager
                self.generation_manager.actor_rollout_wg = unwrapped_model
                gen_output = self.generation_manager.run_agent_loop(gen_batch)

                # postprocess: 由 generation manager 保证 completion ids 的正确性
                completion_ids = gen_output.batch["responses"]
                completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

            # 转为 seperated examples
            recorder.record_batch(completions, test_batch)
        recorder.finalize()
        writer.close()

    def _dynamic_evaluate(self):

        def _set_batch_envs(batch: List) -> Tuple[List[str], List[str], List]:  # batch set envs
            system_prompts, init_user_prompts, envs = [], [], []
            for task_config in batch:
                env = self.env_cls(self.configs.datasets_cfg[self.dataset_name])
                system_prompt, init_user_prompt = env.set_env(task_config)

                system_prompts.append(system_prompt)
                init_user_prompts.append(init_user_prompt)
                envs.append(env)

            return system_prompts, init_user_prompts, envs

        def _build_data_proto(
            system_prompts: List[str], init_user_prompts: List[str], envs: List
        ) -> InteractionDataProto:
            messages = []
            for system_prmopt, init_user_prompt in zip(system_prompts, init_user_prompts):
                system_message = {"role": "system", "content": system_prmopt}
                user_message = {"role": "user", "content": init_user_prompt}
                init_messages = [system_message, user_message]
                messages.append(init_messages)

            data_proto = InteractionDataProto()
            data_proto.no_tensor_batch["init_prompts"] = messages
            data_proto.no_tensor_batch["envs"] = envs

            return data_proto

        # ===== body =====
        output_dir = self.eval_config.output_dir

        accelerator = Accelerator()
        writer = self._create_tensorboard(mode="evaluate")
        save_file = os.path.join(output_dir, "conversations.txt")
        recorder = DynamicEvalRecorder(writer=writer, log_file=save_file)

        batch_size = self.eval_config.batch_size
        generation_config = self.eval_config.generation_config
        _actual_tokenizer = getattr(self.processing_class, 'tokenizer', self.processing_class)
        generation_config.eos_token_id = _actual_tokenizer.eos_token_id

        # prepare dataset and dataloader
        test_dataloader = accelerator.prepare(DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch  # use the identity function
        ))

        # prepare model
        model_wrapped = accelerator.prepare_model(model=self.model, evaluation_mode=True)
        model_wrapped.eval()

        # batch generate
        for step, test_batch in tqdm(enumerate(test_dataloader)):
            with unwrap_model_for_generation(
                model_wrapped, accelerator
            ) as unwrapped_model:
                system_prompts, init_user_prompts, envs = _set_batch_envs(test_batch)
                input_data_proto = _build_data_proto(system_prompts, init_user_prompts, envs)

                self.generation_manager.actor_rollout_wg = unwrapped_model
                outputs: InteractionDataProto = self.generation_manager.run_agent_loop(input_data_proto)

                inter_histories = outputs.no_tensor_batch["inter_histories"]
                inter_context = self.processing_class.apply_chat_template(inter_histories, tokenize=False)

            # batch record
            rewards = []
            for env in input_data_proto.no_tensor_batch["envs"]:
                reward = env.feedback()
                rewards.append(reward)

            recorder.record_batch(inter_context, rewards)

        recorder.finalize()
        writer.close()

    # ===== runner configs =====

    def _parse_configs(self, configs):
        """build configs
        - weaver training config
        - trigger training config
        - interaction config
        - evaluatoin config
        """
        self.save_dir = configs.get("save_dir")
        use_tensorboard = configs.get("use_wandb")

        # weaver configs
        self.train_weaver = configs.get("train_weaver", True)
        if self.train_weaver:
            self.train_weaver_method = configs.get("train_weaver_method", "sft")
            weaver_save_dir = os.path.join(self.save_dir, "weaver")
            weaver_config = configs.get("weaver", {})

            # train weaver with sft
            if self.train_weaver_method == "sft":
                sft_config = weaver_config.get("sft", {})
                weaver_args_dict = self._parse_common_training_args(sft_config, weaver_save_dir, use_tensorboard)
                self.weaver_training_args = SFTConfig(**weaver_args_dict)

            # train weaver with grpo
            elif self.train_weaver_method == "grpo":
                grpo_config = weaver_config.get("grpo", {})
                weaver_args_dict = self._parse_common_training_args(grpo_config, weaver_save_dir, use_tensorboard, is_grpo=True)
                self.weaver_reward_names = weaver_args_dict.pop("reward_names")

                self.weaver_training_args = GRPOConfig(**weaver_args_dict)

            else:
                raise ValueError("Unsupported weaver training mode")

        # Trigger configs
        self.train_trigger = configs.get("train_trigger", False)
        if self.train_trigger:
            self.train_trigger_method = configs.get("train_trigger_method", "grpo")
            trigger_save_dir = os.path.join(self.save_dir, "trigger")
            trigger_config = configs.get("trigger", {})

            if self.train_trigger_method == "grpo":
                grpo_config = trigger_config.get("grpo", {})
                trigger_args_dict = self._parse_common_training_args(grpo_config, trigger_save_dir, use_tensorboard, is_grpo=True)
                self.trigger_reward_names = trigger_args_dict.pop("reward_names")

                self.trigger_training_args = GRPOConfig(**trigger_args_dict)
            else:
                raise ValueError("Unsupported weaver training mode")

        # Interaction configs
        generation_configs = configs.get("generation", {})
        self.interaction_config = InteractionConfig(
            max_turns=generation_configs.get("max_turns", 30),
            max_start_length=generation_configs.get("max_start_length", 1024),
            max_prompt_length=generation_configs.get("max_prompt_length", 4096),
            max_response_length=generation_configs.get("max_response_length", 512),
            max_obs_length=generation_configs.get("max_obs_length", 512),
            do_sample=generation_configs.get("do_sample", False),
            temperature=generation_configs.get("temperature", 1.0)
        )

        # Evaluation configs
        eval_dir = os.path.join(self.save_dir, "evaluate")
        eval_batch_size = generation_configs.get("eval_batch_size", 32)
        self.eval_config = EvalConfig(
            output_dir=eval_dir, batch_size=eval_batch_size, generation_config=self.interaction_config
        )

        # Align GRPO generation configuration with the interaction configuration:
        # All generation-related settings are controlled by the interaction config.
        if (self.train_weaver and self.train_weaver_method == "grpo"):
            self.weaver_training_args.max_prompt_length = self.interaction_config.max_start_length
            self.weaver_training_args.max_completion_length = self.interaction_config.max_response_length
            self.weaver_training_args.temperature = self.interaction_config.temperature
        elif (self.train_trigger and self.train_trigger_method == "grpo"):
            self.trigger_training_args.max_prompt_length = self.interaction_config.max_start_length
            self.trigger_training_args.max_completion_length = self.interaction_config.max_response_length
            self.trigger_training_args.temperature = self.interaction_config.temperature

    def _parse_common_training_args(self, config_dict, output_dir, use_tensorboard, is_grpo=False):
        batch_size = config_dict.get("batch_size", 4)
        max_epochs = config_dict.get("max_epochs", 2)
        grad_accum_steps = config_dict.get("grad_accum_steps", 1)

        optim = config_dict.get("optim", "adamw_torch")
        lr = config_dict.get("lr", 1e-5)
        scheduler = config_dict.get("schedular", "cosine")
        warmup_ratio = config_dict.get("warmup_ratio", 0.1)

        logging_strategy = config_dict.get("logging_strategy", "steps")
        logging_steps = config_dict.get("logging_steps", 1) if logging_strategy == "steps" else None

        eval_strategy = config_dict.get("eval_strategy", "steps")
        eval_steps = config_dict.get("eval_steps", 200) if eval_strategy == "steps" else None

        save_strategy = config_dict.get("save_strategy", "steps")
        save_steps = config_dict.get("save_steps", 200) if save_strategy == "steps" else None

        # Disable load_best_model_at_end when eval_strategy is "no" to avoid evaluation at the end of training
        load_best_model = False if eval_strategy == "no" else True

        # common args dict
        args_dict = {
            "output_dir": output_dir,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "num_train_epochs": max_epochs,
            "gradient_accumulation_steps": grad_accum_steps,
            "optim": optim,
            "learning_rate": lr,
            "lr_scheduler_type": scheduler,
            "warmup_ratio": warmup_ratio,
            "logging_strategy": logging_strategy,
            "logging_steps": logging_steps,
            "save_strategy": save_strategy,
            "save_steps": save_steps,
            "eval_strategy": eval_strategy,
            "eval_steps": eval_steps,
            "report_to": ["wandb"] if use_tensorboard else [],   # Report metrics to wandb when enabled
            "remove_unused_columns": False,
            "load_best_model_at_end": load_best_model,
            "bf16": True,
        }

        # add grpo specific args
        if is_grpo:
            args_dict.update({
                "num_generations": config_dict.get("num_generations", 16),
                "num_iterations": config_dict.get("num_iterations", 1),
                "beta": config_dict.get("beta", 0.0),
                "loss_type": config_dict.get("loss_type", "grpo"),
                "max_prompt_length": config_dict.get("max_prompt_length", 1024),
                "max_completion_length": config_dict.get("max_completion_length", 512),
            })

            rewards = config_dict.get("reward_funcs", [])
            reward_weights = [r["weight"] for r in rewards]
            reward_names = [r["name"] for r in rewards]

            args_dict.update({
                "reward_weights": reward_weights,
                "reward_names": reward_names
            })
        # add sft specific args
        else:
            args_dict.update({
                "max_length": config_dict.get("max_length", 1024),
                "assistant_only_loss": config_dict.get("assistant_only_loss", True)
            })

        return args_dict
