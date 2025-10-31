import logging
from typing import Optional, Union, Any, List, Dict

import torch
from datasets import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase, ProcessorMixin, TrainerCallback
from trl import SFTTrainer, SFTConfig

from .utils import (
    init_sft_log_files,
    persist_sft_logs,
)


class VisionAwareDataCollatorForMemgen:
    """Pad and batch text tensors and optionally stack vision tensors.

    - Pads `input_ids` with pad_token_id, `attention_mask` with 0, `labels` with -100.
    - If every example has non-None `pixel_values`, stacks to (B, C, H, W);
      otherwise omits the key to avoid mixing modalities in one batch.
    - If every example has non-None `image_grid_thw`, stacks to (B, 3) or (B, ...).
    """

    def __init__(self, pad_token_id: int, processor: Optional[ProcessorMixin] = None) -> None:
        self.pad_token_id = int(pad_token_id)
        self.processor = processor

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Input IDs
        ids_list = [ex["input_ids"] for ex in examples]
        ids_list = [t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.long) for t in ids_list]
        input_ids = torch.nn.utils.rnn.pad_sequence(ids_list, batch_first=True, padding_value=self.pad_token_id)

        # Attention mask (fallback to ones if missing)
        if "attention_mask" in examples[0]:
            mask_list = [ex["attention_mask"] for ex in examples]
            mask_list = [t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.long) for t in mask_list]
        else:
            mask_list = [torch.ones_like(t, dtype=torch.long) for t in ids_list]
        attention_mask = torch.nn.utils.rnn.pad_sequence(mask_list, batch_first=True, padding_value=0)

        # Labels
        labels_list = [ex["labels"] for ex in examples]
        labels_list = [t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.long) for t in labels_list]
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)

        batch: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Vision tensors - prefer recomputing via processor to ensure correct shapes
        try:
            # Case 1: recompute from image_path if processor is available and all examples have images
            if self.processor is not None:
                paths = [ex.get("image_path", None) for ex in examples]
                if all(p is not None for p in paths):
                    from PIL import Image
                    imgs = []
                    for p in paths:
                        try:
                            with Image.open(p) as im:
                                imgs.append(im.convert("RGB"))
                        except Exception:
                            imgs.append(None)
                    if all(img is not None for img in imgs):
                        proc_out = self.processor(
                            images=imgs,
                            text=[""] * len(imgs),  # text is unused; included for processor API compatibility
                            return_tensors="pt",
                            padding=True,
                        )
                        if "pixel_values" in proc_out:
                            pv = proc_out["pixel_values"]
                            # Ensure batch-major shape for features: (B, T, D)
                            if isinstance(pv, torch.Tensor) and pv.ndim == 2:
                                try:
                                    B = len(imgs)
                                    total_T = pv.size(0)
                                    D = pv.size(1)
                                    if B > 0 and total_T % B == 0:
                                        T = total_T // B
                                        pv = pv.view(B, T, D)
                                except Exception:
                                    pass
                            batch["pixel_values"] = pv
                        if "image_grid_thw" in proc_out:
                            batch["image_grid_thw"] = proc_out["image_grid_thw"]
                        return batch

            # Case 2: fall back to stacking precomputed tensors when shapes are consistent and >=3D
            all_have_pixels = all(("pixel_values" in ex and isinstance(ex["pixel_values"], torch.Tensor)) for ex in examples)
            if all_have_pixels:
                pix_list = [ex["pixel_values"] for ex in examples]
                # Case 2a: features already computed per-sample with shape (T, D);
                # stack to (B, T, D) if all T and D match
                if all(getattr(t, "ndim", 0) == 2 for t in pix_list):
                    try:
                        t_lens = [t.size(0) for t in pix_list]
                        d_dims = [t.size(1) for t in pix_list]
                        if len(set(t_lens)) == 1 and len(set(d_dims)) == 1:
                            base_dtype = pix_list[0].dtype
                            norm_list = [t.to(dtype=base_dtype) if isinstance(t, torch.Tensor) else torch.as_tensor(t, dtype=base_dtype) for t in pix_list]
                            batch["pixel_values"] = torch.stack(norm_list, dim=0)
                    except Exception:
                        pass
                else:
                    # Case 2b: raw images as (C,H,W) or (1,C,H,W) tensors; normalize to (B,C,H,W)
                    norm_list = []
                    base_dtype = None
                    for t in pix_list:
                        if not isinstance(t, torch.Tensor):
                            t = torch.as_tensor(t)
                        if t.ndim == 4 and t.size(0) == 1:
                            t = t.squeeze(0)
                        norm_list.append(t)
                        if base_dtype is None:
                            base_dtype = t.dtype
                    # Try aligning dtypes
                    try:
                        norm_list = [tt.to(dtype=base_dtype) for tt in norm_list]
                    except Exception:
                        pass
                    if all(getattr(t, "ndim", 0) == 3 for t in norm_list):
                        try:
                            batch["pixel_values"] = torch.stack(norm_list, dim=0)
                        except Exception:
                            pass

            all_have_grid = all(("image_grid_thw" in ex and ex["image_grid_thw"] is not None) for ex in examples)
            if all_have_grid:
                grid_list = [ex["image_grid_thw"] for ex in examples]
                grid_list = [t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in grid_list]
                try:
                    batch["image_grid_thw"] = torch.stack(grid_list, dim=0)
                except Exception:
                    pass
        except Exception:
            # Silently skip vision fields on failure, keeping text-only training running
            pass

        return batch


class WeaverSFTTrainer(SFTTrainer):

    def __init__(
        self,
        model,
        args: Optional[SFTConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            **kwargs,
        )

        self.processing_class = processing_class
        # Initialize SFT logs
        self.sft_log_file, self.sft_jsonl_file = init_sft_log_files(args.output_dir)

    def _decode_tokens(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        tok = getattr(self.processing_class, "tokenizer", self.processing_class)
        try:
            return tok.decode(token_ids.tolist(), skip_special_tokens=skip_special_tokens)
        except Exception:
            try:
                return tok.decode(token_ids, skip_special_tokens=skip_special_tokens)
            except Exception:
                return ""

    def _log_sft_batch(self, inputs: dict[str, Any], mode: str) -> None:
        try:
            if not self.accelerator.is_main_process:
                return
            
            # todo: log the inputs
            logging.info(f"\033[94m[SFT DEBUG _log_sft_batch] inputs: {inputs.keys()}\033[0m")

            input_ids = inputs.get("input_ids")
            labels = inputs.get("labels")
            if input_ids is None or labels is None:
                return

            # Ensure CPU tensors for decoding
            input_ids_cpu = input_ids.detach().cpu()
            labels_cpu = labels.detach().cpu()

            solutions = []
            completions = []
            model_outputs = []

            # Try to fetch solution/completion fields from dataset batch (if present)
            # inputs may be lists of strings
            if isinstance(inputs.get("solution"), list):
                solutions_raw = inputs.get("solution")
            else:
                solutions_raw = None
            if isinstance(inputs.get("completion"), list):
                completions_raw_field = inputs.get("completion")
            else:
                completions_raw_field = None

            # Compute model outputs once (no grad)
            with torch.no_grad():
                # Build minimal model inputs
                model_kwargs = {k: inputs[k] for k in ["input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"] if k in inputs}
                # Temporarily disable gradient checkpointing to avoid warnings under no_grad
                reenable_gc = False
                try:
                    if hasattr(self.model, "is_gradient_checkpointing") and self.model.is_gradient_checkpointing:
                        reenable_gc = True
                        if hasattr(self.model, "gradient_checkpointing_disable"):
                            self.model.gradient_checkpointing_disable()
                except Exception:
                    pass
                outputs = self.model(**model_kwargs)
                try:
                    if reenable_gc and hasattr(self.model, "gradient_checkpointing_enable"):
                        self.model.gradient_checkpointing_enable()
                except Exception:
                    pass
                logits = outputs.logits.detach().cpu()

            for i in range(input_ids_cpu.size(0)):
                ids = input_ids_cpu[i]
                labs = labels_cpu[i]
                # supervised region mask
                sup_mask = labs != -100
                # completion tokens (ground truth): from labels
                completion_tokens = labs[sup_mask].to(torch.long)
                completion_text = self._decode_tokens(completion_tokens, skip_special_tokens=True)

                # model output tokens: greedy from logits on supervised positions
                pred_ids_full = logits[i].argmax(dim=-1)
                pred_ids = pred_ids_full[sup_mask]
                model_output_text = self._decode_tokens(pred_ids, skip_special_tokens=True)

                # ===== Insert <AUG> markers similar to GRPO logs =====
                try:
                    # Prepare single-sample tensors on model device
                    dev = next(self.model.parameters()).device
                    ids_dev = inputs["input_ids"][i:i+1].to(dev)
                    labs_dev = inputs["labels"][i:i+1].to(dev)
                    # Use model's delimiter-based selector
                    aug_points = self.model._select_augment_points_after_delimiter(
                        input_ids=ids_dev,
                        labels=labs_dev,
                        delimiters=getattr(self.model, "delimiters", [",", ".", "\n"]),
                        tokenizer=getattr(self.model, "tokenizer", None),
                        max_num=getattr(self.model, "max_inference_aug_num", 10),
                    )
                    # Map to completion-relative positions
                    # Find first supervised index (start of completion region)
                    sup_indices = (labs != -100).nonzero(as_tuple=True)[0]
                    if len(sup_indices) > 0:
                        start_idx = int(sup_indices[0].item())
                        end_idx = int(sup_indices[-1].item()) + 1
                        comp_len = end_idx - start_idx
                        aug_rel = []
                        for p in aug_points:
                            if p >= start_idx and p < end_idx:
                                aug_rel.append(int(p - start_idx))

                        # Build marked strings by token slicing to preserve alignment
                        def insert_markers(token_ids: torch.Tensor, positions: list[int]) -> str:
                            positions = sorted([pos for pos in positions if 0 <= pos < token_ids.numel()])
                            if len(positions) == 0:
                                return self._decode_tokens(token_ids, skip_special_tokens=True)
                            marked = []
                            last = 0
                            AUG = " <AUG> "
                            for pos in positions:
                                seg = token_ids[last:pos+1]
                                if seg.numel() > 0:
                                    marked.append(self._decode_tokens(seg, skip_special_tokens=True))
                                    marked.append(AUG)
                                last = pos + 1
                            if last < token_ids.numel():
                                marked.append(self._decode_tokens(token_ids[last:], skip_special_tokens=True))
                            return "".join(marked)

                        # Insert markers ONLY in model output; keep completion (ground truth) untouched
                        model_output_text = insert_markers(pred_ids, aug_rel)
                except Exception:
                    # If anything fails, fall back to unmarked texts
                    pass

                # solution text if present
                sol_text = ""
                if solutions_raw is not None and i < len(solutions_raw):
                    try:
                        sol_text = str(solutions_raw[i])
                    except Exception:
                        sol_text = ""

                completions.append(completion_text)
                model_outputs.append(model_output_text)
                solutions.append(sol_text)

            persist_sft_logs(
                log_file=self.sft_log_file,
                jsonl_file=self.sft_jsonl_file,
                step=int(self.state.global_step),
                mode=mode,
                model_outputs=model_outputs,
                solutions=solutions,
                completions=completions,
            )
        except Exception as e:
            logging.warning(f"Failed to log SFT batch: {e}")

    def training_step(self, model, inputs, num_items_in_batch: int | None = None):
        logging.info(f"\033[94m[SFT DEBUG training_step] inputs: {inputs.keys()}\033[0m")
        logging.info(f"\033[94m[SFT DEBUG training_step] num_items_in_batch: {num_items_in_batch}\033[0m")
        loss = super().training_step(model, inputs, num_items_in_batch)
        self._log_sft_batch(inputs, mode="train")
        return loss

    def evaluation_loop(self, *args, **kwargs):
        # Hook into evaluation batches by intercepting dataloader iteration via callback in superclass is complex;
        # instead, rely on training_step logging for now, and optionally add eval logging if needed later.
        return super().evaluation_loop(*args, **kwargs)


