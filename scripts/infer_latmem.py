import argparse
import logging
import os
from typing import Optional

import torch
from PIL import Image
import requests
from io import BytesIO
import yaml

from transformers import GenerationConfig, AutoProcessor

# Allow running without installation when working inside the repo
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from larm.memory_generator.memgen_model import LatentMemoryModel


def _load_image(image_path_or_url: str) -> Image.Image:
    """Load image from local path or URL as RGB PIL.Image."""
    if image_path_or_url.startswith("http"):
        resp = requests.get(image_path_or_url, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    if not os.path.exists(image_path_or_url):
        raise FileNotFoundError(f"Image not found: {image_path_or_url}")
    return Image.open(image_path_or_url).convert("RGB")


def build_inputs(processor, messages, image: Optional[Image.Image] = None):
    """Build model inputs (input_ids, attention_mask, pixel_values, image_grid_thw)."""
    if image is not None:
        # Apply chat template first so that processor knows where to insert image tokens
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = processor(text=[text], images=[image], return_tensors="pt", padding=False)
    else:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = processor(text=[text], return_tensors="pt", padding=False)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    pixel_values = enc.get("pixel_values")
    image_grid_thw = enc.get("image_grid_thw")
    if pixel_values is not None:
        pixel_values = pixel_values.to(torch.bfloat16)
    return input_ids, attention_mask, pixel_values, image_grid_thw


def load_model_from_cfg(cfg_path: str, device: torch.device):
    """Load LatentMemoryModel from a YAML config (same structure as training)."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"] if "model" in cfg else cfg  # support wrapped config
    model = LatentMemoryModel.from_config(model_cfg).to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="LatentMemoryModel inference script")
    parser.add_argument("--cfg", required=True, help="Path to YAML config used to instantiate the model")
    parser.add_argument("--image", help="Optional image path or URL")
    parser.add_argument("--text", required=True, help="User prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling")
    parser.add_argument("--options", nargs="*", help="Override model config via KEY VALUE pairs, e.g. --options model.max_prompt_aug_num 0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Load model & processor
    # Handle overrides
    overrides = args.options or []
    if len(overrides) % 2 != 0:
        raise ValueError("--options should contain KEY VALUE pairs")

    if overrides:
        import copy, yaml
        with open(args.cfg, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)

        cfg = copy.deepcopy(base_cfg)

        def set_nested(cfg_dict, key_path, value):
            keys = key_path.split('.')
            cur = cfg_dict
            for k in keys[:-1]:
                if k not in cur or not isinstance(cur[k], dict):
                    cur[k] = {}
                cur = cur[k]
            # try to cast value to int/float/bool
            if value.lower() == 'null':
                val_cast = None
            else:
                for cast in (int, float):
                    try:
                        val_cast = cast(value)
                        break
                    except ValueError:
                        val_cast = value
                if value.lower() in ("true", "false"):
                    val_cast = value.lower() == "true"
            cur[keys[-1]] = val_cast

        for k, v in zip(overrides[::2], overrides[1::2]):
            set_nested(cfg, k, v)

        # write to tmp then load
        import tempfile, os
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as tmp:
            yaml.safe_dump(cfg, tmp)
            tmp_path = tmp.name
        model = load_model_from_cfg(tmp_path, device)
        os.remove(tmp_path)
    else:
        model = load_model_from_cfg(args.cfg, device)
    processor = model.processor  # AutoProcessor loaded inside the model

    # 2. Build messages list
    messages = []
    if args.image:
        image = _load_image(args.image)
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.text},
            ],
        })
    else:
        messages.append({
            "role": "user",
            "content": args.text,
        })

    # 3. Tokenize / encode
    input_ids, attention_mask, pixel_values, image_grid_thw = build_inputs(processor, messages, image if args.image else None)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)

    # 4. Build generation config
    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    # 5. Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_cfg,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
    if isinstance(outputs, tuple):  # when return_augmentation_mask=True
        outputs = outputs[0]
    generated_ids = outputs[0].detach().cpu()
    # Filter out placeholder ids (e.g., -100) that may be inserted during generation
    valid_token_ids = [tid for tid in generated_ids.tolist() if tid >= 0]
    text = processor.tokenizer.decode(valid_token_ids, skip_special_tokens=True)

    print("\n===== MODEL OUTPUT =====\n")
    print(text)


if __name__ == "__main__":
    main()
