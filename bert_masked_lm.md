# BERT掩码与Transformer自监督学习数学推导

## 一、BERT掩码语言模型 (MLM)

### 1. 数据准备阶段

给定输入序列 $\mathbf{x} = (x_1, x_2, ..., x_T)$，BERT首先进行掩码操作：

**掩码策略**：
$$
\tilde{x}_i = \begin{cases}
\text{[MASK]} & \text{概率 } p_m = 15\% \times 80\% \\
\text{随机token} & \text{概率 } p_r = 15\% \times 10\% \\
x_i & \text{概率 } p_k = 15\% \times 10\% \text{ (保持原样)} \\
x_i & \text{概率 } 85\% \text{ (不掩码)}
\end{cases}
$$

### 2. Embedding层

将掩码后的序列映射到向量空间：

$$
\mathbf{e}_i = \mathbf{W}_{\text{token}} \tilde{x}_i + \mathbf{W}_{\text{position}} i + \mathbf{W}_{\text{segment}} s_i
$$

其中：
- $\mathbf{W}_{\text{token}} \in \mathbb{R}^{d \times V}$ 是词嵌入矩阵（$V$为词汇表大小，$d$为维度）
- $\mathbf{W}_{\text{position}} \in \mathbb{R}^{d \times T_{\max}}$ 是位置编码
- $\mathbf{W}_{\text{segment}} \in \mathbb{R}^{d \times 2}$ 是句子分割编码

得到输入矩阵 $\mathbf{E} = (\mathbf{e}_1, ..., \mathbf{e}_T)^T \in \mathbb{R}^{T \times d}$

### 3. Transformer Encoder层

#### 多头自注意力机制

对于第 $l$ 层，首先计算单头注意力：

**Step 3.1: 计算Query、Key、Value**
$$
\mathbf{Q} = \mathbf{E}^{(l)} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{E}^{(l)} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{E}^{(l)} \mathbf{W}_V
$$

其中 $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times d_k}$，通常 $d_k = d/h$（$h$为头数）

**Step 3.2: 计算注意力分数**
$$
\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)
$$

展开为：
$$
A_{ij} = \frac{\exp\left(\frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}}\right)}{\sum_{j'=1}^{T} \exp\left(\frac{\mathbf{q}_i \cdot \mathbf{k}_{j'}}{\sqrt{d_k}}\right)}
$$

**Step 3.3: 加权求和**
$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathbf{A} \mathbf{V} = \sum_{j=1}^{T} A_{ij} \mathbf{v}_j
$$

**Step 3.4: 多头注意力**
$$
\text{MultiHead}(\mathbf{E}) = \text{Concat}(\text{head}_1, ..., \text{head}_h)\mathbf{W}_O
$$

其中：
$$
\text{head}_k = \text{Attention}(\mathbf{E}\mathbf{W}_Q^k, \mathbf{E}\mathbf{W}_K^k, \mathbf{E}\mathbf{W}_V^k)
$$

#### 残差连接与层归一化

$$
\mathbf{H}' = \text{LayerNorm}(\mathbf{E} + \text{MultiHead}(\mathbf{E}))
$$

层归一化公式：
$$
\text{LayerNorm}(\mathbf{h}) = \frac{\mathbf{h} - \mu}{\sigma} \cdot \gamma + \beta
$$

其中：
$$
\mu = \frac{1}{d}\sum_{i=1}^{d} h_i, \quad \sigma = \sqrt{\frac{1}{d}\sum_{i=1}^{d}(h_i - \mu)^2}
$$

#### FFN层

$$
\mathbf{H}^{(l+1)} = \text{LayerNorm}\left(\mathbf{H}' + \text{FFN}(\mathbf{H}')\right)
$$

其中：
$$
\text{FFN}(\mathbf{h}) = \max(0, \mathbf{h}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2
$$

### 4. MLM预测层

经过 $L$ 层后，对掩码位置的预测：

**Step 4.1: 输出向量**
$$
\mathbf{h}_i^{(L)} = \text{TransformerEncoder}(\tilde{\mathbf{x}}, L)[i]
$$

**Step 4.2: 预测概率分布**
$$
\mathbf{p}_i = \text{softmax}(\mathbf{h}_i^{(L)} \mathbf{W}_{\text{out}} + \mathbf{b}_{\text{out}})
$$

其中 $\mathbf{W}_{\text{out}} \in \mathbb{R}^{d \times V}$，展开为：
$$
p_i(v) = \frac{\exp(\mathbf{h}_i^{(L)} \cdot \mathbf{w}_v + b_v)}{\sum_{v'=1}^{V} \exp(\mathbf{h}_i^{(L)} \cdot \mathbf{w}_{v'} + b_{v'})}
$$

### 5. 自监督损失函数

**交叉熵损失**：

只对被掩码的位置计算损失：
$$
\mathcal{L}_{\text{MLM}} = -\frac{1}{|M|} \sum_{i \in M} \log p_i(x_i)
$$

其中 $M$ 是被掩码位置的集合，$|M|$ 是掩码数量。

展开为：
$$
\mathcal{L}_{\text{MLM}} = -\frac{1}{|M|} \sum_{i \in M} \log \left[\frac{\exp(\mathbf{h}_i^{(L)} \cdot \mathbf{w}_{x_i})}{\sum_{v=1}^{V} \exp(\mathbf{h}_i^{(L)} \cdot \mathbf{w}_v)}\right]
$$

### 6. 梯度反向传播

对参数 $\theta$（包括所有 $\mathbf{W}$ 和 $\mathbf{b}$）：

$$
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_{\text{MLM}}
$$

---

## 二、Transformer自监督学习完整推导

### 1. 自监督学习的本质

自监督学习的核心思想是将**输入本身**转化为监督信号：

$$
\text{输入}: \mathbf{x} \xrightarrow{\text{掩码}} \tilde{\mathbf{x}} \\
\text{监督信号}: y = x_{\text{masked}} \text{ (原始token)}
$$

### 2. 完整的优化目标

BERT的总损失函数：
$$
\mathcal{L}_{\text{BERT}} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}
$$

**NSP (Next Sentence Prediction) 损失**：

给定句子对 $(A, B)$：
$$
\mathcal{L}_{\text{NSP}} = -\left[y \log p(\text{IsNext}) + (1-y) \log p(\text{NotNext})\right]
$$

其中：
$$
p(\text{IsNext}) = \text{sigmoid}(\mathbf{h}_{\text{[CLS]}}^{(L)} \cdot \mathbf{w}_{\text{cls}})
$$

---

## 三、自监督语义规则的"黑盒"问题

### 1. 为什么看起来是黑盒？

模型学习到的语义规则通过以下方式编码：

$$
\text{语义关系} \approx \mathbf{W}_{\text{token}} \cdot \text{共现统计}
$$

但这些权重是**隐式**的：
- 没有显式的"主语-谓语"规则
- 没有可解释的语法树
- 概率分布是高维空间的非线性映射

### 2. 可解释性分析

**注意力矩阵的可解释性**：

$$
A_{ij} = P(\text{token}_i \text{ 依赖 } \text{token}_j | \text{context})
$$

可以通过可视化注意力矩阵来理解：
- 某些头学习句法依赖
- 某些头学习语义关联

**嵌入空间的结构**：

通过Probing任务检验：
$$
\text{Probe}(\mathbf{h}) = \mathbf{h} \cdot \mathbf{w}_{\text{probe}} \rightarrow \text{语法特征}
$$

如果 probing 分类器能从 $\mathbf{h}$ 预测语法标签，说明语义信息被编码了。

### 3. 数学上的"黑盒"本质

自监督学习的语义规则可以理解为：

$$
\text{学习到的规则} = f_\theta(\mathbf{x}) = \arg\max_{\mathbf{x}'} P(\mathbf{x}'|\tilde{\mathbf{x}}; \theta)
$$

其中 $f_\theta$ 是一个**统计映射**而非显式规则系统：

| 特性 | 显式规则系统 | 自监督模型 |
|------|-------------|-----------|
| 表示形式 | IF-THEN规则 | 权重矩阵 $\mathbf{W}$ |
| 可读性 | 直接可读 | 需要probing分析 |
| 推理过程 | 符号推理 | 概率推理 $P(x_i|\text{context})$ |
| 泛化机制 | 规则匹配 | 分布相似性 $\mathbf{h}_i \approx \mathbf{h}_j$ |

---

## 四、完整流程图

```
┌─────────────────────────────┐
│   原始文本 x                 │
└─────────────────────────────┘
              ↓
┌─────────────────────────────┐
│   掩码 x̃ (15% tokens)       │
└─────────────────────────────┘
              ↓
┌─────────────────────────────┐
│   Embedding E = Wtoken +    │
│   Wpos + Wseg               │
└─────────────────────────────┘
              ↓
┌─────────────────────────────┐
│   Transformer × L 层        │
└─────────────────────────────┘
              ↓
┌─────────────────────────────┐
│   输出 H^(L)                 │
└─────────────────────────────┘
              ↓
┌─────────────────────────────┐
│   Softmax p = softmax(      │
│   H^(L)Wout)                │
└─────────────────────────────┘
              ↓
┌─────────────────────────────┐
│   损失 L = -Σ log p_i(x_i)  │
└─────────────────────────────┘
              ↓
┌─────────────────────────────┐
│   反向传播 θ ← θ - η∇θL    │
└─────────────────────────────┘
```

---

## 五、关键数学总结

自监督学习的本质是：

$$
\max_\theta \sum_{\mathbf{x} \in \mathcal{D}} \log P(\mathbf{x}_{\text{masked}} | \mathbf{x}_{\text{unmasked}}; \theta)
$$

这个目标函数迫使模型：
1. **编码上下文依赖**：通过注意力机制 $\mathbf{A}$
2. **学习token共现模式**：通过 $\mathbf{W}_{\text{token}}$
3. **推断缺失信息**：通过 $\mathbf{H}^{(L)} \rightarrow \mathbf{p}$

语义规则以**统计相关性**的形式存在于参数 $\theta$ 中，而非显式符号规则，这就是其"黑盒"特性的数学根源。