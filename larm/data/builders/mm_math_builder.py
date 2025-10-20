import os
from typing import Dict, List

from datasets import DatasetDict, load_dataset
import requests
import zipfile
import io
from PIL import Image

from larm.data.builders.base_builder import BaseDatasetBuilder
from larm.data.interactions.singleturn_interaction import SingleTurnInteractionManager
from larm.common.registry import registry
from larm.data.envs.mm_math_env import MMMathEnv


@registry.register_builder("mm_math")
class MMMathBuilder(BaseDatasetBuilder):

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mm_math/default.yaml",
    }
    CACHE_PATH = None

    def _build_datasets(self) -> DatasetDict:
        """Build SFT/RL datasets with image-aware samples.

        Expected directory structure under `cache_path`:
            cache_path/
              train.json   # list of {"question": str, "answer": str, "image_path": str}
              valid.json
              test.json
              images/      # image files referenced by image_path
        """

        cache_path = self.config.get("cache_path", None)
        if cache_path is None:
            raise ValueError("mm_math cache_path must be set in dataset config")

        # Preferred save directory under global cache_root (typically 'data/mm_math')
        cache_root = registry.get_path("cache_root")
        save_dir = os.path.join(cache_root, "mm_math")
        os.makedirs(save_dir, exist_ok=True)

        # Prefer previously saved splits from save_dir
        data_files = {}
        for split_name in ["train", "valid", "test"]:
            json_path = os.path.join(save_dir, f"{split_name}.json")
            if os.path.exists(json_path):
                data_files[split_name] = json_path

        # If not found in save_dir, look under dataset-specific cache_path
        if len(data_files) == 0:
            for split_name in ["train", "valid", "test"]:
                json_path = os.path.join(cache_path, f"{split_name}.json")
                if os.path.exists(json_path):
                    data_files[split_name] = json_path

        # Auto-download if missing
        if len(data_files) == 0:
            os.makedirs(cache_path, exist_ok=True)
            print(f"Downloading MM_Math dataset to {cache_path}")
            jsonl_url = "https://huggingface.co/datasets/THU-KEG/MM_Math/resolve/main/MM_Math/MM_Math.jsonl"
            jsonl_path = os.path.join(cache_path, "mm_math.jsonl")
            if not os.path.exists(jsonl_path):
                resp = requests.get(jsonl_url, timeout=120)
                resp.raise_for_status()
                with open(jsonl_path, "wb") as f:
                    f.write(resp.content)
            # Load as a single split and split locally if needed
            data_files = {"train": jsonl_path}

        raw_ds = load_dataset("json", data_files=data_files)

        # If only a single split exists, split into train/valid/test and save to save_dir
        if ("train" in raw_ds) and ("valid" not in raw_ds or "test" not in raw_ds):
            base_train = raw_ds["train"]
            val_ratio = float(self.config.get("val_ratio", 0.05))
            test_ratio = float(self.config.get("test_ratio", 0.05))
            if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1.0:
                raise ValueError("Invalid val_ratio/test_ratio; ensure 0 <= ratios and sum < 1.0")

            holdout_ratio = val_ratio + test_ratio
            first_split = base_train.train_test_split(test_size=holdout_ratio, seed=42, shuffle=True)
            split_train = first_split["train"]
            holdout = first_split["test"]
            if holdout_ratio > 0:
                # further split the holdout into valid/test by proportion
                test_fraction = test_ratio / holdout_ratio if holdout_ratio > 0 else 1.0
                second_split = holdout.train_test_split(test_size=test_fraction, seed=42, shuffle=True)
                split_valid = second_split["train"]
                split_test = second_split["test"]
            else:
                split_valid = split_train.select(range(0))
                split_test = split_train.select(range(0))

            # Save to global cache_root for future reuse
            train_path = os.path.join(save_dir, "train.json")
            valid_path = os.path.join(save_dir, "valid.json")
            test_path = os.path.join(save_dir, "test.json")
            os.makedirs(save_dir, exist_ok=True)
            split_train.to_json(train_path)
            split_valid.to_json(valid_path)
            split_test.to_json(test_path)

            # Rebuild raw_ds from the fresh splits
            raw_ds = DatasetDict({
                "train": split_train,
                "valid": split_valid,
                "test": split_test,
            })

        # add image_root into each example for downstream joining
        image_root = self.config.get("image_root", None)
        if image_root is None:
            raise ValueError("mm_math image_root must be set in dataset config")

        # Auto-download images zip if folder missing
        if not os.path.isdir(image_root) or len(os.listdir(image_root)) == 0:
            os.makedirs(image_root, exist_ok=True)
            print(f"Downloading MM_Math images to {image_root}")
            zip_url = "https://huggingface.co/datasets/THU-KEG/MM_Math/resolve/main/MM_Math/MM_Math.zip"
            resp = requests.get(zip_url, timeout=300)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                z.extractall(image_root)

        def _inject_root(example: Dict) -> Dict:
            example["image_root"] = image_root
            return example
        for split_name in raw_ds.keys():
            raw_ds[split_name] = raw_ds[split_name].map(_inject_root)

        # preprocess
        # Use safe, batched single-process mapping to avoid multiprocessing hangs
        try:
            num_workers = int(self.config.get("num_workers", 1))
        except Exception:
            num_workers = 1
        try:
            batch_size = int(self.config.get("batch_size", 512))
        except Exception:
            batch_size = 512
        keep_keys = self._keep_keys()

        def _preprocess_batch(batch: Dict) -> Dict:
            def _format_answer(answer: str) -> str:
                answer = (answer or "").strip()
                if answer.startswith("\\boxed{") and answer.endswith("}"):
                    return answer
                return "\\boxed{" + answer + "}"

            format_template = r"""Solve the visual math problem with proper reasoning, and make sure to put the FINAL ANSWER inside \boxed{}."""
            prompt_template = "Question: {prompt}\n"

            questions: List[str] = batch.get("question") or [""] * len(batch.get("answer", []))
            answers: List[str] = batch.get("answer") or [""] * len(questions)
            # Prefer file_name; fallback to existing image_path field if present in source
            file_names_src = batch.get("file_name", [None] * len(questions))
            image_paths_src = batch.get("image_path", [None] * len(questions))
            file_names: List[str] = [fn if fn is not None else ip for fn, ip in zip(file_names_src, image_paths_src)]
            image_roots: List[str] = batch.get("image_root") or [None] * len(questions)

            prompts: List[str] = []
            completions: List[str] = []
            solutions: List[str] = []
            image_paths: List[str] = []
            for q, a, fn, root in zip(questions, answers, file_names, image_roots):
                processed_prompt = format_template + prompt_template.format(prompt=(q or "").strip())
                processed_label = _format_answer(a or "")
                prompts.append(processed_prompt)
                completions.append(processed_label)
                solutions.append(processed_label)
                if fn is not None:
                    image_paths.append(os.path.join(root, fn) if root is not None else fn)
                else:
                    image_paths.append(None)

            return {
                "prompt": prompts,
                "completion": completions,
                "solution": solutions,
                "image_path": image_paths,
            }

        def _map(split):
            print(f"[MM_Math] Preprocess start: split={split}, batch_size={batch_size}")
            ds = raw_ds[split].map(
                _preprocess_batch,
                batched=True,
                batch_size=batch_size,
                num_proc=None,  # force single-process mapping
                load_from_cache_file=False,
                desc=f"MM_Math preprocess ({split})",
            ).select_columns(keep_keys)
            print(f"[MM_Math] Preprocess done: split={split}, num_rows={len(ds)}")
            return ds

        dataset_dict = DatasetDict()
        print(f"Building MM_Math dataset with {num_workers} workers (batched single-process mapping)")
        if "train" in raw_ds:
            dataset_dict["train"] = _map("train")
        if "valid" in raw_ds:
            dataset_dict["valid"] = _map("valid")
        if "test" in raw_ds:
            dataset_dict["test"] = _map("test")

        # Print a final processed example datapoint for quick verification
        try:
            if "train" in dataset_dict and len(dataset_dict["train"]) > 0:
                sample = dataset_dict["train"][0]
                preview = {k: sample.get(k) for k in keep_keys if k in sample}
                print("[MM_Math] Final processed example:", preview)
        except Exception as e:
            print("[MM_Math] Final example print failed:", e)

        return dataset_dict

    def _build_sft_datasets(self) -> DatasetDict:
        return self._build_datasets()

    def _build_rl_datasets(self) -> DatasetDict:
        return self._build_datasets()

    @classmethod
    def _preprocess(cls, example: Dict):
        def _format_answer(answer: str) -> str:
            answer = (answer or "").strip()
            if answer.startswith("\\boxed{") and answer.endswith("}"):
                return answer
            return "\\boxed{" + answer + "}"

        format_template = r"""Solve the visual math problem with proper reasoning, and make sure to put the FINAL ANSWER inside \boxed{}."""
        prompt_template = "Question: {prompt}\n"

        question = (example.get("question") or "").strip()
        answer = (example.get("answer") or "").strip()
        # Build absolute/relative image path using file_name and image_root from config
        file_name = example.get("file_name") or example.get("image_path")
        image_root = example.get("image_root")
        image_path = None
        if file_name is not None:
            image_path = os.path.join(image_root, file_name) if image_root is not None else file_name

        processed_prompt = format_template + prompt_template.format(prompt=question)
        processed_label = _format_answer(answer)

        out: Dict = {
            "prompt": processed_prompt,
            "completion": processed_label,
            "solution": processed_label,
        }

        # Attach image path for downstream processor/collator to build pixel_values.
        if image_path is not None:
            out["image_path"] = image_path

        if not hasattr(cls, "_printed_example"):
            try:
                preview = {k: out.get(k) for k in ("prompt", "completion", "solution", "image_path")}
                print("[MM_Math] Example after _preprocess:", preview)
            except Exception as e:
                print("[MM_Math] Example print failed:", e)
            cls._printed_example = True

        return out

    @classmethod
    def _keep_keys(cls) -> List[str]:
        return ["prompt", "completion", "solution", "image_path"]

    def get_env_cls(self):
        return MMMathEnv

    def get_generation_manager_cls(self):
        return SingleTurnInteractionManager


