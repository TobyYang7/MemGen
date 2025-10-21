### Weaver + VLM 训练与推理设计（代码说明）

本项目的 Reasoner 支持 VLM（如 Qwen-VL via `AutoModelForVision2Seq`）与纯 LLM；Weaver 保持为 LLM。Weaver 的作用是在指定插入点向 Reasoner 的隐空间注入“潜在记忆”向量，从而影响解码轨迹。新版本遵循“仅在图像 token 范围内进行检索与插入”的约束。

---

#### 组件与主要文件
- `larm/memory_generator/memgen_model.py`：核心模型 `LatentMemoryModel`（注册名 `latmem`）。
  - 子组件：`reasoner`（VLM/LLM）、`weaver`（LLM）、`trigger`（二分类/规则）、双向投影层 `reasoner_to_weaver` 与 `weaver_to_reasoner`。
- `larm/memory_generator/memgen_runner.py`：Runner，解析配置，构建数据、环境与交互管理器；选择 SFT 或 GRPO 训练流程；附带 SFT/GRPO 日志。
- `larm/memory_generator/trainer/weaver_grpo_trainer.py`：GRPO Trainer，负责采样、奖励与策略优化，已扩展多模态输入与 JSONL 日志。
- `larm/data/interactions/singleturn_interaction.py`：单轮交互管理器；已将 `pixel_values` 与 `image_token_mask` 透传给模型生成。
- `larm/memory_generator/weaver.py`：Weaver 模块，提供 `augment_prompt` 与 `augment_inference` 两类注入。

---

#### 训练（SFT）前向（`LatentMemoryModel.forward`）
输入
- `input_ids: Tensor[B, L]`：拼接后的 prompt+completion 序列
- `attention_mask: Tensor[B, L]`
- `labels: Tensor[B, L]`：-100 为忽略监督
- （可选）`pixel_values: Tensor`：VLM 图像特征输入
- （可选）`image_token_mask: Bool[B, L]`：标记文本序列中属于“图像 token 区间”的位置

步骤
1) 任务类型判断：单轮（instructional）或多轮（conversational）；多轮拆段逐段调用单轮逻辑。
2) 选择插入点：
   - 1 个“提示插入点”：label 从 -100→有效 的边界
   - 若干“推理插入点”：监督区域内，紧随分隔符（"," "." "\n"）之后
   - 若 `image_token_mask` 提供，则过滤上述插入点，仅保留位于图像区间的索引
3) 逐段累积 + 注入：
   - 将 Reasoner 的 `inputs_embeds` 分段累积，得到 `current_inputs_embeds/mask/pos`；
   - 若提供图像上下文：先通过 VLM 前向获得最后层 hidden，抽取 `image_token_mask` 对应 span 的 `img_ctx/img_mask/img_pos`，再经 `reasoner_to_weaver` 投影为 `weaver_context_inputs/mask/pos`；
   - 提示插入：`weaver.augment_prompt(weaver_context_inputs, ...)`（若无图像上下文则用 `reasoner_to_weaver(current_inputs_embeds)`）；
   - 推理插入：`weaver.augment_inference(...)`（同上）；
   - 将 weaver 的隐状态经 `weaver_to_reasoner` 投回 Reasoner 空间并拼接到当前序列。
4) Reasoner 监督：
   - Vision2Seq 情况下，传入 `pixel_values` 以保证解码条件化图像；
   - 对插入位置的 logits 屏蔽监督（仅“非潜在位置”参与 CE 损失）。

输出
- `CausalLMOutputWithPast`：`loss`（忽略 -100）、`logits`（与 `labels` 对齐）

目的与意义
- Weaver 的潜在向量提供“记忆/技能”片段，注入到 Reasoner 的隐空间，引导其在包含图像线索的关键位置进行更有效的推理；只在图像 token 范围插入确保记忆与视觉信息强对齐。

---

#### 推理（`LatentMemoryModel.generate`）
输入
- `input_ids, attention_mask`（batch 推理）
- （可选）`pixel_values, image_token_mask` 同训练
- `generation_config`（采样温度/最大生成长度等）

步骤
1) 若提供图像：先通过 Reasoner 计算 hidden，抽取 `image_token_mask` 区间形成 weaver 的图像上下文三元组（inputs/mask/pos）。
2) 初始提示插入：优先使用图像上下文执行 `augment_prompt`，将潜在向量投回并拼接。
3) 逐步生成：每步用 Reasoner 产生下一个 token；
4) 动态插入决策：通过 `_should_augment` 决策并限制仅在图像区间内才允许插入；若选择插入，使用 `augment_inference` 执行注入。

输出
- `input_ids` 或 `(input_ids, augmentation_pos)`：包含生成序列（可选返回插入位置掩码）

意义
- 将记忆注入行为与图像区域动态对齐，在解码过程中于视觉相关位置插入，从而提升视觉推理质量与稳定性。

---

#### GRPO 训练（`WeaverGRPOTrainer`）
输入
- 从 `InteractionManager` 构建的 `InteractionDataProto`，包含：
  - `batch`: `input_ids/attention_mask`（如有图像：还包含 `pixel_values` 与 `image_token_mask`）
  - `no_tensor_batch`: 任务元信息

生成与打分
- 使用 `actor_rollout_wg.generate` 进行批量生成（多模态参数透传），内部即调用模型的 `generate`，因此遵循上面的图像上下文与插入逻辑；
- 解析输出构造 `prompt_ids/completion_ids/...`，按配置的奖励函数评估 `rewards_per_func` 并聚合为优势；
- 计算 GRPO 损失并反传，仅在 completion 段落监督。

日志
- SFT：`results/<method>/<time>/logs/sft_train_log.txt`
- GRPO：`results/<method>/<time>/logs/grpo_train_log.txt`
  - 每条记录包含：step、prompt（尽力解码）、pred、ref（若数据包含）、参数（长度/温度等）、是否含图像等。

---

#### 数据与特征（以 MM_Math 为例）
- `mm_math_builder.py` 会构造 `prompt/completion/solution/image_path`；
- `memgen_runner._prepare_mm_features` 用 `AutoProcessor` 读取图像，生成 `pixel_values`，并尝试用 `tokenizer` 的视觉特殊标记推断 `image_token_mask`；
- 训练与生成均会消费这些字段。

---

#### 输入/输出汇总（按阶段）
- 训练前向：
  - 入：`input_ids/attention_mask/labels`，可选 `pixel_values/image_token_mask`
  - 出：`loss, logits`（插入位置被屏蔽监督）
- 推理生成：
  - 入：`input_ids/attention_mask`，可选 `pixel_values/image_token_mask`，`generation_config`
  - 出：`generated_ids`（可选 `augmentation_pos`）
- GRPO：
  - 入：`InteractionDataProto(batch=..., no_tensor_batch=...)`
  - 出：trainer 内部的 `advantages/loss/metrics`，外部仅见日志与模型权重更新

---

#### 设计要点与取舍
- 只在图像 token 范围内插入，保证记忆与视觉线索强绑定，避免无关文本处的扰动。
- 通过双向投影层隔离 Reasoner 与 Weaver 的隐空间差异，便于替换模型与减少梯度耦合。
- Vision2Seq 下训练必须传入 `pixel_values`，确保 loss 真正条件化图像。
- 日志 JSONL 便于训练中排错与案例级诊断（包含 QA、预测与关键参数）。


