## Training

### Weaver 训练机制（基于 `larm/memory_generator/memgen_model.py: LatentMemoryModel.forward`）

- **训练目标**: 在指定插入点将 weaver 生成的潜在向量注入到 reasoner 的隐空间中，增强解码路径；仅对“非潜在位置”做语言模型监督，梯度主要更新 weaver 与映射层。
- **两类前向**:
  - **单轮 SFT**: `_instructional_forward` → 核心 `_forward`，在“提示结束处”与“推理分隔符后”插入潜在向量。
  - **多轮 SFT**: `_conversational_forward` 将对话拆成多个监督段，每段独立走一次 `_forward`，最后拼回整条序列。
  - **GRPO**: 由 `WeaverGRPOTrainer` 负责采样、奖励与策略优化，但插入与映射机制与 SFT 一致。

#### 关键入口
- 函数签名：`LatentMemoryModel.forward(input_ids, attention_mask, labels, **kwargs)`
- 路由：判断是否为对话格式，选择 `_instructional_forward`（单轮）或 `_conversational_forward`（多轮），最终统一计算自回归 LM 损失（忽略 `-100` 标签）。

#### 单轮训练流程（`_instructional_forward` → `_forward`）
1) **选择插入点**
   - 1 个“提示插入点”：label 从 `-100 → 有效` 的边界（单轮必须恰好一个）。
   - 若干“推理插入点”：在监督区间内且位于分隔符 `[",", ".", "\n"]` 之后，数量由 `max_inference_aug_num` 限制。

2) **双向投影与插入**
   - 用 reasoner 的 embedding 得到 `inputs_embeds`，逐段累加到当前序列。
   - 将当前累积的 `inputs_embeds` 投影到 weaver 隐空间（`reasoner_to_weaver`），调用：
     - `weaver.augment_prompt(...)`（提示处），或
     - `weaver.augment_inference(...)`（推理处），
     取 weaver 最后一层在新拼接位置的隐向量作为“潜在向量”。
   - 将潜在向量投影回 reasoner 隐空间（`weaver_to_reasoner`），并与当前序列拼接，同时同步 `attention_mask/position_ids`。

3) **屏蔽潜在位置监督**
   - reasoner 在“插入后序列”上前向得到 `logits`。
   - 通过位移后的潜在掩码剔除“潜在位置”的 `logits`，仅保留“非潜在位置”的监督目标。

4) **语言模型损失（忽略 -100）**
```python
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
loss = CrossEntropyLoss(ignore_index=-100)(
    shift_logits.reshape(-1, shift_logits.size(-1)),
    shift_labels.reshape(-1),
)
```
- **梯度路径**: 非潜在位置的 CE 损失 → `weaver_to_reasoner` → weaver → `reasoner_to_weaver`（reasoner 通常冻结）。

#### 多轮训练流程（`_conversational_forward`）
- 按 `labels` 找到多个监督段（assistant 回复区间），逐段截取“到段末”的子序列分别调用一次 `_forward`（各段的潜在向量互不“泄漏”），把每段的 `logits` 回填到整序列对应区间，拼接后统一计算 LM 损失（忽略 `-100`）。

#### 与训练方式（SFT vs GRPO）的关系
- **SFT**: 使用上述 CE 损失；weaver 查询向量在“提示结束 + 推理分隔符后”插入。
- **GRPO**: `WeaverGRPOTrainer` 负责采样/奖励/优势计算与策略优化；生成阶段仍通过 weaver 插入与双向投影影响 reasoner 的解码。

#### 关键要点小结
- **插入点选择**: 1 个提示插入 + 若干推理插入（分隔符后，限额）。
- **双向投影**: Reasoner 隐空间 ↔ Weaver 隐空间。
- **监督屏蔽**: 仅监督“非潜在位置”，潜在位置不参与 loss。
- **多轮拆段**: 段内独立插入与前向，段间不共享潜在。
- **参数更新**: 主要更新 weaver 与映射层；reasoner 默认冻结（除非显式解冻/PEFT）。


