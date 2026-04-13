# BERT掩码与Transformer掩码的自监督学习数学推导

## 目录
1. [Transformer自注意力机制的数学基础](#1-transformer自注意力机制的数学基础)
2. [掩码的数学定义与作用](#2-掩码的数学定义与作用)
3. [BERT掩码语言模型(MLM)的自监督学习过程](#3-bert掩码语言模型mlm的自监督学习过程)
4. [自监督损失函数的严格推导](#4-自监督损失函数的严格推导)
5. [语义规则的表示与黑盒问题分析](#5-语义规则的表示与黑盒问题分析)

---

## 1. Transformer自注意力机制的数学基础

### 1.1 输入表示

设输入序列为 $\mathbf{x} = [x_1, x_2, \ldots, x_L]$，其中 $L$ 为序列长度。

首先将输入转换为嵌入向量：
$$
\mathbf{E} = [\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_L] \in \mathbb{R}^{L \times d_{model}}
$$

其中 $d_{model}$ 为模型维度。

### 1.2 Query、Key、Value的生成

通过可学习的权重矩阵 $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$ 生成查询、键、值矩阵：

$$
\mathbf{Q} = \mathbf{E}\mathbf{W}^Q \in \mathbb{R}^{L \times d_k}
$$

$$
\mathbf{K} = \mathbf{E}\mathbf{W}^K \in \mathbb{R}^{L \times d_k}
$$

$$
\mathbf{V} = \mathbf{E}\mathbf{W}^V \in \mathbb{R}^{L \times d_v}
$$

其中 $d_k$ 和 $d_v$ 分别为键和值的维度。

### 1.3 自注意力机制的数学定义

**无掩码的自注意力计算：**

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

**展开推导：**

1. **计算注意力分数矩阵：**
$$
\mathbf{S} = \mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{L \times L}
$$

其中每个元素：
$$
S_{ij} = \mathbf{q}_i^\top \mathbf{k}_j = \sum_{m=1}^{d_k} q_{im} k_{jm}
$$

表示位置 $i$ 对位置 $j$ 的原始注意力分数。

2. **缩放操作：**
$$
\tilde{\mathbf{S}} = \frac{\mathbf{S}}{\sqrt{d_k}}
$$

**为什么要缩放？**

假设 $q_i$ 和 $k_j$ 的元素是独立同分布的随机变量，均值为0，方差为1。
则 $S_{ij}$ 的方差为：
$$
\text{Var}(S_{ij}) = \text{Var}\left(\sum_{m=1}^{d_k} q_{im} k_{jm}\right) = d_k
$$

当 $d_k$ 很大时，点积值会很大，导致softmax函数的梯度接近于0（梯度消失问题）。

缩放后：
$$
\text{Var}\left(\frac{S_{ij}}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1
$$

3. **Softmax归一化：**
$$
\alpha_{ij} = \text{softmax}(\tilde{S}_{ij}) = \frac{\exp(\tilde{S}_{ij})}{\sum_{l=1}^{L} \exp(\tilde{S}_{il})}
$$

得到的注意力权重矩阵：
$$
\mathbf{A} = \text{softmax}(\tilde{\mathbf{S}}) \in \mathbb{R}^{L \times L}
$$

其中每行满足：
$$
\sum_{j=1}^{L} \alpha_{ij} = 1, \quad \alpha_{ij} \geq 0
$$

4. **加权聚合：**
$$
\mathbf{O} = \mathbf{A}\mathbf{V} \in \mathbb{R}^{L \times d_v}
$$

每个位置的输出：
$$
\mathbf{o}_i = \sum_{j=1}^{L} \alpha_{ij} \mathbf{v}_j
$$

---

## 2. 掩码的数学定义与作用

### 2.1 掩码矩阵的定义

掩码矩阵 $\mathbf{M} \in \mathbb{R}^{L \times L}$ 定义为：

$$
M_{ij} = \begin{cases}
0 & \text{如果位置 } i \text{可以关注位置 } j \\
-\infty & \text{如果位置 } i \text{不能关注位置 } j
\end{cases}
$$

### 2.2 带掩码的注意力计算

$$
\text{MaskedAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{M}) = \text{softmax}\left(\mathbf{M} + \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

**数学推导：**

当 $M_{ij} = -\infty$ 时：
$$
\alpha_{ij} = \frac{\exp(M_{ij} + \tilde{S}_{ij})}{\sum_{l} \exp(M_{il} + \tilde{S}_{il})} = \frac{\exp(-\infty + \tilde{S}_{ij})}{\sum_{l} \cdots} = \frac{0}{\sum_{l} \cdots} = 0
$$

因为：
$$
\lim_{x \to -\infty} \exp(x) = 0
$$

### 2.3 Transformer中的三种掩码

#### (1) 因果掩码 (Causal Mask) - 用于解码器

$$
M_{ij}^{causal} = \begin{cases}
0 & \text{如果 } j \leq i \\
-\infty & \text{如果 } j > i
\end{cases}
$$

**目的：** 防止当前位置看到未来位置的信息，保证自回归生成的正确性。

**因果掩码矩阵形式（L=4）：**
$$
\mathbf{M}^{causal} = \begin{bmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

#### (2) 填充掩码 (Padding Mask)

$$
M_{ij}^{pad} = \begin{cases}
0 & \text{如果位置 } j \text{是真实token} \\
-\infty & \text{如果位置 } j \text{是padding token}
\end{cases}
$$

#### (3) BERT双向掩码 - 无因果约束

BERT作为编码器，使用**无掩码**的双向注意力：
$$
M_{ij}^{BERT} = 0, \quad \forall i, j
$$

这意味着每个位置都可以看到所有其他位置的信息。

---

## 3. BERT掩码语言模型(MLM)的自监督学习过程

### 3.1 问题定义

给定一个句子 $\mathbf{x} = [x_1, x_2, \ldots, x_L]$，MLM的目标是：

$$
\text{最大化：} \log P(x_i | \mathbf{x}_{\setminus i}; \theta)
$$

其中 $\mathbf{x}_{\setminus i}$ 表示除去位置 $i$ 后的上下文。

### 3.2 掩码策略的数学描述

设掩码位置集合为 $\mathcal{M} \subset \{1, 2, \ldots, L\}$，掩码比例 $|\mathcal{M}|/L = 15\%$。

**掩码操作的三种方式：**

定义掩码函数 $\text{mask}(x_i)$：
$$
\text{mask}(x_i) = \begin{cases}
[\text{MASK}] & \text{概率 } 80\% \\
\text{随机token} & \text{概率 } 10\% \\
x_i & \text{概率 } 10\%
\end{cases}
$$

**为什么要保留10%原token和10%随机token？**

数学分析：
- 如果100%用[MASK]，模型只在预训练时见到[MASK]，微调时没有这个token，会造成**分布不匹配**
- 加入随机token迫使模型学习上下文语义判断，而非简单的"填空模式"

### 3.3 掩码后的输入表示

$$
\tilde{\mathbf{x}} = [\tilde{x}_1, \tilde{x}_2, \ldots, \tilde{x}_L]
$$

其中：
$$
\tilde{x}_i = \begin{cases}
\text{mask}(x_i) & \text{如果 } i \in \mathcal{M} \\
x_i & \text{如果 } i \notin \mathcal{M}
\end{cases}
$$

### 3.4 BERT编码过程

**步骤1：输入嵌入**
$$
\mathbf{E} = \mathbf{W}^{embed}[\tilde{\mathbf{x}}] + \mathbf{W}^{pos} + \mathbf{W}^{seg}
$$

其中：
- $\mathbf{W}^{embed}$：词嵌入矩阵 $\in \mathbb{R}^{V \times d}$，$V$为词汇表大小
- $\mathmathbf{W}^{pos}$：位置嵌入 $\in \mathbb{R}^{L \times d}$
- $\mathmathbf{W}^{seg}$：句子片段嵌入（区分句子A/B）

**步骤2：多层Transformer编码**

设第 $l$ 层的输出为 $\mathbf{H}^l$，初始 $\mathbf{H}^0 = \mathbf{E}$。

每层的计算：
$$
\mathbf{H}^{l+1} = \text{TransformerBlock}(\mathbf{H}^l)
$$

**TransformerBlock的详细计算：**

1. **多头注意力：**
$$
\text{MultiHead}(\mathbf{H}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}^O
$$

其中每个头：
$$
\text{head}_m = \text{Attention}(\mathbf{H}\mathbf{W}_m^Q, \mathbf{H}\mathbf{W}_m^K, \mathbf{H}\mathbf{W}_m^V)
$$

2. **残差连接与层归一化：**
$$
\mathbf{H}' = \text{LayerNorm}(\mathbf{H} + \text{MultiHead}(\mathbf{H}))
$$

3. **前馈网络：**
$$
\text{FFN}(\mathbf{H}') = \max(0, \mathbf{H}'\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2
$$

4. **最终输出：**
$$
\mathbf{H}^{l+1} = \text{LayerNorm}(\mathbf{H}' + \text{FFN}(\mathbf{H}'))
$$

**层归一化(LayerNorm)的数学定义：**
$$
\text{LayerNorm}(\mathbf{h}) = \frac{\mathbf{h} - \mu}{\sigma} \cdot \gamma + \beta
$$

其中：
$$
\mu = \frac{1}{d}\sum_{i=1}^{d} h_i, \quad \sigma = \sqrt{\frac{1}{d}\sum_{i=1}^{d}(h_i - \mu)^2}
$$

---

## 4. 自监督损失函数的严格推导

### 4.1 MLM损失函数

设最终层的隐藏状态为 $\mathbf{H}^L \in \mathbb{R}^{L \times d}$。

对于掩码位置 $i \in \mathcal{M}$，预测概率的计算：

$$
P(x_i | \tilde{\mathbf{x}}; \theta) = \text{softmax}(\mathbf{h}_i^L \mathbf{W}^{vocab})
$$

其中 $\mathbf{W}^{vocab} \in \mathbb{R}^{d \times V}$ 是预测头的权重矩阵。

**展开：**
$$
P(v | \tilde{\mathbf{x}}; \theta) = \frac{\exp(\mathbf{h}_i^L \mathbf{w}_v)}{\sum_{v'=1}^{V} \exp(\mathbf{h}_i^L \mathbf{w}_{v'})}
$$

其中 $\mathbf{w}_v$ 是 $\mathbf{W}^{vocab}$ 的第 $v$ 列，对应词汇表中第 $v$ 个词。

### 4.2 MLM损失函数的数学推导

**单个掩码位置的损失：**

$$
\mathcal{L}_{MLM}(x_i) = -\log P(x_i | \tilde{\mathbf{x}}; \theta)
$$

$$
= -\log \frac{\exp(\mathbf{h}_i^L \mathmathbf{w}_{x_i})}{\sum_{v'=1}^{V} \exp(\mathbf{h}_i^L \mathbf{w}_{v'})}
$$

$$
= -\mathbf{h}_i^L \mathbf{w}_{x_i} + \log \sum_{v'=1}^{V} \exp(\mathbf{h}_i^L \mathbf{w}_{v'})
$$

**总的MLM损失：**

$$
\mathcal{L}_{MLM} = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log P(x_i | \tilde{\mathbf{x}}; \theta)
$$

### 4.3 梯度推导

**对权重 $\mathbf{W}^{vocab}$ 的梯度：**

对于正确类别 $x_i$ 的权重 $\mathbf{w}_{x_i}$：
$$
\frac{\partial \mathcal{L}_{MLM}(x_i)}{\partial \mathbf{w}_{x_i}} = -\mathbf{h}_i^L + P(x_i | \tilde{\mathbf{x}}; \theta) \mathbf{h}_i^L
$$

$$
= -(1 - P(x_i | \tilde{\mathbf{x}}; \theta)) \mathbf{h}_i^L
$$

对于错误类别 $v \neq x_i$ 的权重 $\mathbf{w}_v$：
$$
\frac{\partial \mathcal{L}_{MLM}(x_i)}{\partial \mathbf{w}_v} = P(v | \tilde{\mathbf{x}}; \theta) \mathbf{h}_i^L
$$

**梯度解释：**
- 正确类别：当预测概率接近1时，梯度接近0（模型已正确预测）
- 错误类别：梯度为正，推动权重减少对这些词的预测概率

### 4.4 反向传播的全链推导

设隐藏状态 $\mathbf{h}_i^L$ 是通过多层Transformer计算的。

**链式法则：**
$$
\frac{\partial \mathcal{L}_{MLM}}{\partial \mathbf{W}^Q} = \frac{\partial \mathcal{L}_{MLM}}{\partial \mathbf{h}_i^L} \cdot \frac{\partial \mathmathbf{h}_i^L}{\partial \mathbf{H}^{L-1}} \cdot \ldots \cdot \frac{\partial \mathbf{H}^1}{\partial \mathbf{Q}} \cdot \frac{\partial \mathbf{Q}}{\partial \mathbf{W}^Q}
$$

**Softmax对隐藏状态的梯度：**
$$
\frac{\partial \mathcal{L}_{MLM}(x_i)}{\partial \mathbf{h}_i^L} = -\mathbf{w}_{x_i} + \sum_{v=1}^{V} P(v | \tilde{\mathbf{x}}; \theta) \mathbf{w}_v
$$

$$
= -\mathbf{w}_{x_i} + \mathbf{W}^{vocab} \mathbf{p}_i
$$

其中 $\mathbf{p}_i = [P(1|\tilde{\mathbf{x}}), \ldots, P(V|\tilde{\mathbf{x}})]^\top$ 是概率向量。

### 4.5 自监督信号的形成过程

**关键洞察：**

自监督的核心在于**掩码创造了监督信号**：

$$
\text{监督信号} = \text{原始token} \quad (\text{从数据本身提取，无需人工标注})
$$

$$
\text{模型输入} = \text{掩码后的序列}
$$

$$
\text{目标} = \text{预测原始token}
$$

**自监督学习的数学本质：**

设数据分布为 $P_{data}(\mathbf{x})$，自监督学习的目标是：

$$
\max_\theta \mathbb{E}_{\mathbf{x} \sim P_{data}} \left[ \sum_{i \in \mathcal{M}} \log P(x_i | \tilde{\mathbf{x}}; \theta) \right]
$$

这等价于最大化数据的**条件概率**，从而学习语言的统计规律。

---

## 5. 语义规则的表示与黑盒问题分析

### 5.1 学习到的语义表示

BERT学习到的语义信息编码在以下参数中：

1. **词嵌入矩阵 $\mathbf{W}^{embed}$：**
$$
\mathbf{e}_w = \mathbf{W}^{embed}[w] \in \mathbb{R}^{d}
$$
每个词的嵌入向量编码了该词的语义信息。

2. **注意力权重矩阵 $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$：**
编码了词与词之间的语义关系。

3. **前馈网络权重：**
编码了复杂的语义变换规则。

### 5.2 语义规则的数学解释

**注意力机制的语义解释：**

$$
\alpha_{ij} = \frac{\exp(\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_k})}{\sum_l \exp(\mathbf{q}_i^\top \mathbf{k}_l / \sqrt{d_k})}
$$

可以理解为：
- $\mathbf{q}_i$：位置 $i$ 的"语义查询"——我在找什么类型的语义信息
- $\mathbf{k}_j$：位置 $j$ 的"语义特征"——我有什么语义信息可以提供
- $\alpha_{ij}$：语义相关性权重

**语义相似度的表示：**

设两个词的嵌入为 $\mathbf{e}_w$ 和 $\mathbf{e}_{w'}$，它们的语义相似度可以通过嵌入向量空间中的距离来衡量：

$$
\text{sim}(w, w') = \frac{\mathbf{e}_w \cdot \mathbf{e}_{w'}}{||\mathbf{e}_w|| \cdot ||\mathbf{e}_{w'}||}
$$

### 5.3 黑盒问题的分析

#### 学习到的规则是否可解释？

**问题1：参数的可解释性**

参数数量：BERT-base有约110M参数，BERT-large有约340M参数。

$$
\theta = \{\mathbf{W}^{embed}, \mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V, \mathbf{W}^O, \mathbf{W}_1, \mathbf{W}_2, \ldots\}
$$

**数学上：**
- 单个参数 $\theta_i \in \mathbb{R}$ 没有直接的语义解释
- 参数的组合效应产生语义行为

**问题2：语义规则的隐式编码**

设语义规则为 $R$，模型通过以下方式隐式编码规则：

$$
R(x) \approx f_\theta(x)
$$

其中 $f_\theta$ 是复杂的非线性函数：
$$
f_\theta(x) = \text{softmax}(\mathbf{H}^L \mathbf{W}^{vocab})
$$

**数学本质：**
规则被编码在高维参数空间中，而非显式的符号规则。

#### 黑盒的定量分析

**表示熵：**

设模型的隐藏状态分布为 $P(\mathbf{h})$，其信息熵：
$$
H(\mathbf{h}) = -\sum_{\mathbf{h}} P(\mathbf{h}) \log P(\mathbf{h})
$$

高熵意味着模型学到了丰富的语义信息，但这些信息的结构难以直接解释。

**注意力头的可解释性：**

某些注意力头显示出可解释的模式：

$$
\text{Head}_{syntactic} : \alpha_{ij} \approx \text{句法依赖关系}
$$

$$
\text{Head}_{position} : \alpha_{ij} \approx \text{位置距离}
$$

但这只是少数头，大多数头的语义是混合的。

### 5.4 打破黑盒的方法

#### 探针任务 (Probing Tasks)

通过特定任务测试模型学到的知识：

$$
\mathcal{L}_{probe} = -\sum_{(h, y)} \log P(y | h; \theta_{probe})
$$

其中 $h$ 是模型的隐藏状态，$y$ 是语言学标签（如句法角色）。

如果探针模型能高准确率预测 $y$，说明 $h$ 编码了相关信息。

#### 数学框架：

设隐藏状态 $\mathbf{h}$ 包含信息 $I$：
$$
I(\mathbf{h}, y) = H(y) - H(y|\mathbf{h})
$$

通过探针可以量化这种信息量。

### 5.5 结论：部分黑盒，部分可解释

**数学结论：**

| 层面 | 可解释性 | 数学依据 |
|------|----------|----------|
| 词嵌入 | 较高 | 向量空间中的几何关系可量化 |
| 单个注意力头 | 中等 | 部分头显示清晰的句法模式 |
| 参数组合 | 低 | 高维非线性组合难以分解 |
| 整体行为 | 低 | 340M参数的复杂交互 |

**数学上的根本原因：**

自监督学习优化的是：
$$
\min_\theta \mathcal{L}_{MLM}(\theta)
$$

这个目标函数只要求预测准确，不要求参数可解释。

因此模型学到的是**隐式表示**而非**显式规则**。

---

## 附录：完整的自监督学习流程图

```
原始数据 x = [x_1, x_2, ..., x_L]
    │
    ▼
随机选择掩码位置 M (15%的token)
    │
    ▼
掩码操作：x̃ = mask(x, M)
    │        ├── 80% → [MASK]
    │        ├── 10% → 随机token
    │        └── 10% → 保持原token
    │
    ▼
嵌入层：E = Embed(x̃) + PosEmbed + SegEmbed
    │
    ▼
多层Transformer编码（双向注意力，无因果掩码）
    │
    ▼ H^L ∈ R^(L×d)
    │
    ▼
预测头：P(v|x̃) = softmax(H^L · W^vocab)
    │
    ▼
损失函数：L_MLM = -∑_{i∈M} log P(x_i|x̃)
    │
    ▼
反向传播：∂L/∂θ
    │
    ▼
参数更新：θ ← θ - η·∂L/∂θ
    │
    ▼
循环迭代直到收敛
```

---

## 参考文献

1. Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
2. Vaswani et al. (2017). "Attention is All You Need"
3. Jurafsky & Martin. "Speech and Language Processing" (Chapter 11)
4. Rogers et al. (2020). "A Primer in BERTology"