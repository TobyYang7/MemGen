# ⚠️ 图像Token截断问题说明

## 问题描述

在 GRPO 训练中，当 prompt 长度超过 `max_prompt_length` (默认512) 时，代码会从**右侧**截取 tokens：

```python
prompts = prompts[:, -self.max_prompt_length :]
```

这意味着**左侧的 tokens 会被截断**。

## 风险

对于包含图像的 prompt，图像 tokens 通常位于 prompt 的**开头或中间**位置：

```
<|im_start|>user
🖼️<|vision_start|><|image_pad|><|vision_end|>🖼️
这是一道数学题，请解答...
<|im_end|><|im_start|>assistant
```

如果 prompt 太长被截断，**图像 tokens 会丢失**，导致：
- ❌ 模型无法看到图像信息
- ❌ 只能基于文本部分回答
- ❌ 对于视觉问答任务，性能会严重下降

## 检测机制

我已经在 `log_prompt_truncation` 函数中添加了自动检测：

### 1. 检测的 Vision Token 类型
- `<|vision_start|>`
- `<|vision_end|>`
- `<|image_pad|>`
- `<|video_pad|>`
- `<|vision_pad|>`

### 2. 日志输出

**如果图像 tokens 被截断**，会显示警告：
```
⚠️  WARNING: IMAGE/VISION TOKENS WERE TRUNCATED!
⚠️  Lost vision token IDs: {151652, 151653, 151654}
⚠️  Vision tokens before: {151652, 151653, 151654, 151655}
⚠️  Vision tokens after: {151655}
⚠️  The model will NOT see the image information!
```

**如果图像 tokens 保留完整**：
```
✓ Vision tokens preserved: {151652, 151653, 151654, 151655}
```

### 3. Token 可视化

在 token 列表中，vision tokens 会用 🖼️ 标记：
```
[BEFORE TRUNCATION]
Tokens: <|im_start|> [1587:'user'] 🖼️<|vision_start|>🖼️ 🖼️<|image_pad|>🖼️ 🖼️<|vision_end|>🖼️ ...

[AFTER TRUNCATION]
Tokens: [2768:' following'] [7033:' math'] [3575:' problem'] ...
```

## 解决方案

### 方案 1：增加 max_prompt_length（推荐）

修改配置文件 `configs/latent_memory/mm_math.yaml`：

```yaml
generation:
  max_start_length: 1024  # 从 512 增加到 1024 或更大
  max_prompt_length: 4096
```

### 方案 2：优化 Prompt 长度

- 简化问题描述
- 移除不必要的上下文
- 确保图像相关的核心内容在前 512 tokens 内

### 方案 3：修改截断策略（需要代码修改）

当前是从右侧截取（保留后面的内容）：
```python
prompts = prompts[:, -self.max_prompt_length :]  # 保留右侧
```

可以改为从左侧截取（保留前面的内容，包括图像）：
```python
prompts = prompts[:, :self.max_prompt_length]  # 保留左侧
```

但这样会丢失问题的后半部分，需要权衡。

### 方案 4：智能截断（最佳但复杂）

检测 vision tokens 的位置，确保它们不被截断：
1. 找到 vision tokens 的位置
2. 如果需要截断，从 vision tokens 之后开始截取
3. 保证图像信息始终保留

## 当前状态

✅ **已添加检测和警告机制**
- 自动检测 vision tokens 是否被截断
- 在日志中高亮显示 vision tokens
- 如果截断发生会显著警告

⚠️ **需要手动配置**
- 根据实际数据集调整 `max_start_length`
- 监控日志中的截断警告
- 如果频繁出现截断，增加长度限制

## 日志输出层次

现在有三个层次的日志输出：

### 1. PROMPT INFO - 基本信息
```
[PROMPT INFO] Original prompt shape: torch.Size([8, 650]), max_prompt_length: 512
[PROMPT INFO] After truncation shape: torch.Size([8, 512])
[PROMPT INFO] Truncation detected: 650 -> 512
```

### 2. PROMPT TRUNCATION - 截断详情（如果发生截断）
```
================================================================================
[PROMPT TRUNCATION] Sample 0
Length before truncation: 650
Length after truncation: 512
⚠️  WARNING: IMAGE/VISION TOKENS WERE TRUNCATED!
...
================================================================================
```

### 3. ROLLOUT INPUT - 传给 trainer 的输入
```
================================================================================
[ROLLOUT INPUT] Sample 0
Prompt length: 512 tokens
✓ Contains vision tokens: {151652, 151653}
[INPUT TOKENS]
Tokens: <|im_start|> 🖼️<|vision_start|>🖼️ ...
================================================================================
```

### 4. MODEL.GENERATE INPUT - 实际传给模型的输入
```
================================================================================
[MODEL.GENERATE INPUT] Sample 0
Input length: 512 tokens
✓ Contains vision tokens: {151652, 151653}
[TOKENS TO MODEL]
Tokens: <|im_start|> 🖼️<|vision_start|>🖼️ ...
================================================================================
```

## 查看日志

运行训练后，检查是否有截断警告：

```bash
# 查看所有 prompt 信息
grep "\[PROMPT INFO\]" test_output/mm_math/logs/log.txt

# 查看截断警告
grep "WARNING.*VISION.*TRUNCATED" test_output/mm_math/logs/log.txt

# 查看详细的截断日志
grep -A 30 "\[PROMPT TRUNCATION\]" test_output/mm_math/logs/log.txt

# 查看 rollout 输入
grep -A 15 "\[ROLLOUT INPUT\]" test_output/mm_math/logs/log.txt

# 查看实际传给模型的输入
grep -A 15 "\[MODEL.GENERATE INPUT\]" test_output/mm_math/logs/log.txt
```

## 建议

1. **训练前**：先运行一个 epoch，检查日志中是否有 vision token 截断警告
2. **如果有警告**：立即增加 `max_start_length`，重新开始训练
3. **监控**：定期检查日志，确保图像信息没有丢失
4. **数据统计**：统计数据集中 prompt 长度分布，设置合适的 `max_start_length`

## 示例输出

### 正常情况（无截断）
```
[PROMPT INFO] Original prompt shape: torch.Size([8, 450]), max_prompt_length: 512
[PROMPT INFO] After truncation shape: torch.Size([8, 450])
[PROMPT INFO] No truncation needed: length 450 <= max 512
```

### 有截断但保留图像
```
[PROMPT INFO] Truncation detected: 650 -> 512
================================================================================
[PROMPT TRUNCATION] Sample 0
Length before truncation: 650
Length after truncation: 512
✓ Vision tokens preserved: {151652, 151653, 151654}
[BEFORE TRUNCATION]
Tokens: <|im_start|> 🖼️<|vision_start|>🖼️ 🖼️<|image_pad|>🖼️ ...
[AFTER TRUNCATION]
Tokens: 🖼️<|vision_start|>🖼️ 🖼️<|image_pad|>🖼️ ...
================================================================================
```

### 危险情况（图像被截断）⚠️
```
[PROMPT INFO] Truncation detected: 650 -> 512
================================================================================
⚠️  WARNING: IMAGE/VISION TOKENS WERE TRUNCATED!
⚠️  Lost vision token IDs: {151652, 151653}
⚠️  The model will NOT see the image information!
[BEFORE TRUNCATION]
Tokens: <|im_start|> 🖼️<|vision_start|>🖼️ 🖼️<|image_pad|>🖼️ ...
[AFTER TRUNCATION]
Tokens: [2768:' following'] [7033:' math'] ... (无图像 tokens)
================================================================================
```

**如果看到这种警告，必须立即调整配置！**

