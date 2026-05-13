# MiniMind 逐行代码学习计划（刻意练习方法）

> **总代码量**：核心约 1400 行，全部约 4200 行
> **学习原则**：每次只聚焦一个模块，先理解「为什么这样设计」，再逐行读懂「怎么实现的」，最后能「自己写出来」

---

## 刻意练习方法论

| 原则 | 在 MiniMind 学习中如何体现 |
|------|--------------------------|
| **分解技能** | 按训练管线拆成 7 个模块，每个模块聚焦一个核心概念 |
| **专注练习** | 每次只学一个文件/一个类，不跳来跳去 |
| **即时反馈** | 每学完一段，回答验证问题 + 自己尝试修改/重写 |
| **跳出舒适区** | 从会用的层次→理解原理→能实现→能改进 |
| **心理表征** | 最终能在脑中构建 Transformer 从数据到生成的完整画面 |

---

## 模块总览：MiniMind 训练管线

```
数据预处理 → 数据加载 → 模型架构 → 预训练 → SFT → 对齐(DPO/GRPO)
   (1)         (2)        (3)        (4)      (5)       (6)
```

加上 **Tokenizer 训练 (0)** 作为前置基础。

---

## 阶段 0：环境准备（0.5 天）

**目标**：能在本地/服务器上跑通 MiniMind 的最小推理

- [ ] 确认服务器代码路径和理解项目目录结构
- [ ] 在服务器上加载 MiniMind 模型，跑一次推理
- [ ] 理解 tokenizer 的输入输出：`tokenizer.encode("你好")` 返回什么？

---

## 阶段 1：Tokenizer & 词表（1-2 天）

**核心概念**：语言模型看到的是数字，不是文字。Tokenizer 是文字→数字的桥梁。

### 1.1 Tokenizer 基础（0.5 天）
**文件**：`model/tokenizer.json` + `model/tokenizer_config.json`

- [ ] 理解 tokenizer.json 的结构（词表映射：token → id）
- [ ] 理解特殊 token：bos_token_id=1, eos_token_id=2, pad_token
- [ ] 为什么 vocab_size=6400？（而不是 32000 或 151936）
- [ ] 什么是 BPE？6400 词表如何训练出来的？
- [ ] 理解 chat_template 的作用

> **验证问题**：`"你好世界"` 会被 tokenize 成几个 token？每个 token 的 id 是多少？

### 1.2 Tokenizer 训练（1 天）
**文件**：`trainer/train_tokenizer.py`（168 行）

- [ ] 逐行理解 BPE tokenizer 训练流程
- [ ] 特殊 token 的定义和添加
- [ ] `train_new_from_iterator` 的原理
- [ ] 词表大小对模型的影响（参数量 + 编码效率）

> **动手练习**：用自己的文本训练一个 1000 词表的 mini tokenizer

---

## 阶段 2：数据管线（1-2 天）

**核心概念**：数据质量决定模型上限。理解从原始文本到训练 batch 的完整链路。

### 2.1 数据预处理（1 天）
**文件**：`scripts/preprocess_data.py`（183 行）

- [ ] 原始数据格式：jsonl，每行一个 JSON 对象
- [ ] 数据清洗流程（去重、过滤、格式化）
- [ ] 预训练数据 vs SFT 数据的格式区别
- [ ] chat template 的构造逻辑（system/user/assistant 角色的组织）

> **验证问题**：为什么预训练数据只需要 text 字段，而 SFT 数据需要 conversations 字段？

### 2.2 数据加载（1 天）
**文件**：`dataset/lm_dataset.py`（255 行）

- [ ] `PretrainDataset` 类的 `__getitem__` 方法逐行理解
- [ ] 为什么要在序列前后加 BOS 和 EOS token？
- [ ] `max_length` 截断策略
- [ ] `SFTDataset` 和 `DPODataset` 的区别
- [ ] loss mask 如何构造（为什么 SFT 只计算 assistant 部分的 loss？）

> **动手练习**：写一个 `__getitem__` 方法，从 jsonl 加载数据并 tokenize

---

## 阶段 3：模型架构（3-4 天）

**这是整个学习中最核心的部分。286 行代码，凝聚了 Transformer 的精髓。**

### 3.1 配置类（0.5 天）
**代码**：`MiniMindConfig`（第 10-43 行）

- [ ] 每个配置参数的含义和默认值
- [ ] `intermediate_size` 为什么是 `ceil(hidden_size * π / 64) * 64`？
- [ ] `num_key_value_heads` vs `num_attention_heads`（GQA 原理）
- [ ] `rope_theta` 和 YaRN 外推机制
- [ ] `flash_attn` 标志的作用

> **关键理解**：这个 Config 类本质上是模型的「配方」——给定 hidden_size 和 layers，自动计算出所有其他维度的合理值。

### 3.2 RMSNorm（0.5 天）
**代码**：`RMSNorm` 类（第 49-56 行）

- [ ] RMSNorm vs LayerNorm 的数学区别
- [ ] 为什么大模型都用 RMSNorm？（更快、更稳定）
- [ ] `_norm` 方法的逐行数学含义
- [ ] `weight` 参数的作用（可学习的缩放因子）

> **验证问题**：写出 RMSNorm 的数学公式，解释为什么不需要减去均值。

### 3.3 RoPE 位置编码（1 天）
**代码**：`precompute_freqs_cis` + `apply_rotary_pos_emb`（第 58-75 行）

- [ ] 为什么需要位置编码？（Attention 本身没有位置信息）
- [ ] 绝对位置编码 vs 相对位置编码 vs RoPE
- [ ] RoPE 的数学原理：旋转矩阵
- [ ] `rope_base`（theta）的作用
- [ ] YaRN 扩展的原理（从 2048 外推到 32768）
- [ ] `freqs_cos` 和 `freqs_sin` 的预计算逻辑

> **关键理解**：RoPE 的思想是把位置信息编码为复数旋转，两个 token 的内积只依赖它们的相对位置。
> **验证问题**：如果 rope_base 从 1e6 改成 1e4，会有什么影响？

### 3.4 Attention 层（1 天）
**代码**：`Attention` 类（约第 77-130 行）

- [ ] Q、K、V 投影的维度变换（hidden_size → num_heads × head_dim）
- [ ] GQA（Grouped Query Attention）的实现：`kv_heads` 如何复用
- [ ] `scaled_dot_product_attention` vs 手动实现
- [ ] Flash Attention 的条件启用
- [ ] attention_mask 的作用（causal mask + padding mask）
- [ ] 输出投影 `o_proj` 的作用

> **关键理解**：Self-Attention 本质是每个 token 向所有 token「查询」相关信息，然后加权聚合。
> **验证问题**：如果 num_heads=8, kv_heads=2, head_dim=112，画出 QKV 的维度变换图。

### 3.5 FFN / MoE 层（0.5 天）
**代码**：`MiniMindMLP` + `MiniMindMoE`（约第 132-200 行）

- [ ] 标准 FFN 的三层结构：gate → up → down（SwiGLU）
- [ ] SwiGLU vs ReLU vs GELU 的区别
- [ ] MoE 的核心概念：多个 expert + router
- [ ] Top-K 路由（`num_experts_per_tok`）
- [ ] load balancing loss（router aux loss）
- [ ] Shared expert 的作用

> **关键理解**：MoE 是「用空间换能力」——多个 FFN 专家，每次只用其中几个，总参数量大但计算量小。

### 3.6 完整模型组装（0.5 天）
**代码**：`MiniMindModel` + `MiniMindForCausalLM`（第 200-286 行）

- [ ] Transformer Block 的组装：RMSNorm → Attention → Residual → RMSNorm → FFN → Residual
- [ ] Pre-Norm vs Post-Norm（现在都用 Pre-Norm）
- [ ] `lm_head` 和 `embed_tokens` 的 weight tying
- [ ] `CausalLMOutputWithPast` 的输出结构
- [ ] 模型的 forward 流程完整追踪

> **终极验证**：手动画出 MiniMind 的完整计算图，标注每一步的张量形状变化。

---

## 阶段 4：预训练（1-2 天）

**核心概念**：大规模自回归语言建模——给定前文，预测下一个 token。

### 4.1 训练工具函数（0.5 天）
**文件**：`trainer/trainer_utils.py`（176 行）

- [ ] 分布式训练初始化：`init_distributed_mode()`
- [ ] 学习率调度：cosine schedule with warmup
- [ ] 参数计算：`get_model_params()`（理解 Dense vs MoE 的参数计算）
- [ ] `DistributedSampler` 的原理

> **验证问题**：为什么 DDP 模式下需要 DistributedSampler？如果不用会怎样？

### 4.2 预训练主循环（1 天）
**文件**：`trainer/train_pretrain.py`（169 行）

- [ ] 训练循环的完整结构：`for step in range(total_steps):`
- [ ] `torch.compile()` 的作用和原理
- [ ] 梯度累积（`accumulation_steps`）：为什么需要？如何实现？
- [ ] `torch.cuda.amp` 混合精度训练
- [ ] loss 的计算和反向传播流程
- [ ] checkpoint 保存策略
- [ ] 学习率 warmup + cosine decay 的配合

> **动手练习**：写一个最简单的训练循环（不用分布式），训练一个 1 层的模型做 next-token prediction。

---

## 阶段 5：SFT 监督微调（1 天）

**核心概念**：从「预测下一个 token」到「遵循指令回答问题」。

### 5.1 SFT 训练
**文件**：`trainer/train_full_sft.py`（170 行）

- [ ] SFT 和预训练的核心区别（数据格式 + loss 计算）
- [ ] Chat template 的构造
- [ ] `loss_mask` 机制：为什么只计算 assistant 部分的 loss？
- [ ] 为什么 SFT 不需要 `torch.compile`？（训练步数少）

> **验证问题**：如果 SFT 对 user 部分也计算 loss，会发生什么？

---

## 阶段 6：对齐训练（2-3 天）

**核心概念**：让模型不仅「能回答」，还要「回答得好」——符合人类偏好。

### 6.1 DPO（1 天）
**文件**：`trainer/train_dpo.py`（225 行）

- [ ] DPO vs RLHF 的核心区别
- [ ] DPO 的数学原理：直接优化偏好，不需要 reward model
- [ ] chosen vs rejected 数据结构
- [ ] DPO loss 公式的逐行代码对应
- [ ] reference model 的作用（冻结的 SFT 模型）

> **关键理解**：DPO 让模型增大 chosen 和 rejected 之间的概率差。

### 6.2 GRPO（1 天）
**文件**：`trainer/train_grpo.py`（331 行）

- [ ] GRPO vs PPO 的区别（group relative advantage）
- [ ] reward 的计算方式
- [ ] advantage 的归一化（group normalization）
- [ ] KL 散度约束（防止模型偏离太远）

### 6.3 LoRA 微调（0.5 天）
**文件**：`model/model_lora.py`（65 行）+ `trainer/train_lora.py`（183 行）

- [ ] LoRA 的数学原理：低秩分解
- [ ] `lora_A` 和 `lora_B` 的维度设计
- [ ] 为什么 LoRA 可以大幅减少训练参数？
- [ ] LoRA 适用于哪些层？（q_proj, v_proj）

> **回头对照**：比较 LoRA 训练和全量训练的显存和速度差异。

---

## 阶段 7：进阶主题（按兴趣选学）

### 7.1 知识蒸馏
**文件**：`trainer/train_distillation.py`（245 行）
- 大模型教小模型的核心思想，KL 散度 loss

### 7.2 PPO 强化学习
**文件**：`trainer/train_ppo.py`（443 行）+ `trainer/rollout_engine.py`（212 行）
- 完整的 RLHF 流程，reward model，policy gradient

### 7.3 Agent 训练
**文件**：`trainer/train_agent.py`（487 行）
- Tool calling，function call 的训练

---

## 建议的学习节奏

| 天 | 内容 | 预计时间 |
|----|------|----------|
| 1 | 阶段0（环境）+ 阶段1（Tokenizer） | 3-4h |
| 2 | 阶段2（数据管线） | 3-4h |
| 3 | 阶段3.1-3.3（Config + RMSNorm + RoPE） | 3-4h |
| 4 | 阶段3.4（Attention 层） | 3-4h |
| 5 | 阶段3.5-3.6（FFN/MoE + 模型组装） | 3-4h |
| 6 | 阶段4（预训练） | 3-4h |
| 7 | 阶段5（SFT）+ 阶段6.3（LoRA） | 3-4h |
| 8-9 | 阶段6.1-6.2（DPO + GRPO） | 每天 3-4h |
| 10+ | 阶段7（进阶主题，按需） | 灵活 |

**总计**：约 10 天可以覆盖核心 1400 行代码，理解每一行在做什么。

---

## 刻意练习的核心习惯

1. **每次开始前**：问自己「这一节的核心概念是什么？它解决了什么问题？」
2. **读代码时**：不要走马观花，每一行都问「为什么要这样写？有没有替代方案？」
3. **学完后**：尝试用中文向别人解释（费曼学习法），或者自己写一遍
4. **遇到不理解**：马上停下来问，不要攒到最后
5. **每日结束**：用 5 分钟回顾今天学到的最重要的 3 个点

---

## 现在开始？

准备好了就告诉我，我们从**阶段 0（环境准备）**开始，或者你告诉我你想从哪个阶段切入。
