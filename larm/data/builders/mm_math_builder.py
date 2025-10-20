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

        data_files = {}
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
        num_workers = 64
        keep_keys = self._keep_keys()

        def _map(split):
            return raw_ds[split].map(self._preprocess, num_proc=num_workers).select_columns(keep_keys)

        dataset_dict = DatasetDict()
        print(f"Building MM_Math dataset with {num_workers} workers")
        if "train" in raw_ds:
            dataset_dict["train"] = _map("train")
        if "valid" in raw_ds:
            dataset_dict["valid"] = _map("valid")
        if "test" in raw_ds:
            dataset_dict["test"] = _map("test")

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


