import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ========= 加载模型 =========
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    # device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# ========= 构造输入 =========
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]

# 拼接 chat 模板
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("\n===== [Chat Template Text] =====")
print(text)

# 视觉输入信息
image_inputs, video_inputs = process_vision_info(messages)

# Processor 构造完整输入
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(model.device)

# ========= 打印输入结构 =========
print("\n===== [Inputs Summary] =====")
for k, v in inputs.items():
    print(f"\033[94m{k:<20}\033[0m : {tuple(v.shape)}  dtype={v.dtype}")

# ========= 打印前 50 个 token =========
prompt_ids = inputs.input_ids[0].tolist()
print("\n===== [MM_Features] Sample prompt_ids (first 50) =====")
print(prompt_ids[:50])

decoded_tokens = processor.tokenizer.batch_decode([prompt_ids[:50]])[0].split()
print("\n===== [MM_Features] Decoded tokens (first 50) =====")
print(decoded_tokens)

# ========= 前向计算 fused embedding =========
with torch.no_grad():
    outputs = model.model(
        input_ids=inputs.input_ids,
        pixel_values=inputs.pixel_values,
        image_grid_thw=inputs.image_grid_thw,
        attention_mask=inputs.attention_mask,
        return_dict=True,
        # output_hidden_states=True,
    )

# print all keys of outputs
print("\n===== [Outputs Keys] =====")
print(outputs.keys())

fused_embeds = outputs.last_hidden_state
print("\n===== [Fused Embedding Summary] =====")
print(f"fused_embeds shape: {tuple(fused_embeds.shape)}  dtype={fused_embeds.dtype}")

# ========= 打印视觉段位置 =========
tokenizer = processor.tokenizer
ids = inputs.input_ids[0].tolist()
vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

vision_start_idx = ids.index(vision_start_id)
vision_end_idx = ids.index(vision_end_id)
print("\n===== [Vision Embedding Span] =====")
print(f"vision_start_idx: {vision_start_idx}, vision_end_idx: {vision_end_idx}")
print(f"vision segment length: {vision_end_idx - vision_start_idx - 1}")
print(f"total fused_embeds length: {fused_embeds.shape[1]}")



# ========= 从 fused_embeds 计算 logits =========
with torch.no_grad():
    logits = model.lm_head(fused_embeds)  # [batch, seq_len, vocab_size]

# 取每个位置上概率最大的 token id
pred_token_ids = torch.argmax(logits, dim=-1)  # [batch, seq_len]

# ========= 根据 token id 解码文本 =========
decoded_from_embeds = processor.tokenizer.batch_decode(
    pred_token_ids,
    skip_special_tokens=True,
    # clean_up_tokenization_spaces=False,
)

print("\n===== [Decoded from fused_embeds via lm_head] =====")
print(decoded_from_embeds[0][:1000])
