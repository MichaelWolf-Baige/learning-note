---
name: path-memory-technical-design
description: Path Memory 技术方案——结构化路径记忆，保留所有成功+失败路径，通过最短路径检索优化Agent决策效率
metadata: 
  node_type: memory
  type: project
  created: 2026-06-07
  status: 设计阶段
  replaces: "agent-memory-consolidation-project (方向从\"记忆巩固/遗忘\"调整为\"路径优化/最短路径\")"
  originSessionId: adfcf12b-4dc7-4e20-b55e-6b84e69dd623
---

# Path Memory：基于经验路径的 Agent 决策优化

## 零、一句话说清楚

> RAG 告诉 Agent "别人怎么做的"（一段文本描述）。
> Path Memory 告诉 Agent "你这样走，3 步能到；那几条路走不通，别试"（一条可直接执行的路径）。

---

## 一、核心思路

### 1.1 问题定义

Agent 使用工具完成任务时，经常走弯路：

```
目标：找到关于 Transformer 注意力机制的最新论文

第一次探索（从零开始）：
  Step 1: search_web("transformer attention") → 返回噪音
  Step 2: search_arxiv("transformer attention") → 返回20篇，太多
  Step 3: search_arxiv("transformer attention mechanism") + filter 2026 → 5篇高质量
  Step 4: fetch paper 1 → 不是想要的
  Step 5: fetch paper 3 → ✅ 找到
  总步数：5 步，其中 2 步是弯路

如果 Agent 记得这条路径：
  Step 1: search_arxiv("transformer attention mechanism", filter=2026) → 5篇
  Step 2: fetch paper 3 → ✅ 找到
  总步数：2 步，零弯路

更进一步，如果 100 次类似任务后：
  Agent 发现了更短的通用路径：对于"找论文"类任务，直接 arxiv + filter year
  → 平均从 5 步降到 1.5 步
```

### 1.2 核心洞察

```
传统观点：记忆 = 存储信息，需要遗忘来管理容量
我们的观点：记忆 = 存储决策路径，容量不是问题，检索精度才是关键

人类需要遗忘，因为生物记忆容量有限。
Agent 不需要遗忘，因为向量数据库可以存几百万条路径，检索一条路径只需要几毫秒。
Agent 需要的是：在几百万条路径中，精确找到"当前情境下最短的那条成功路径"。
```

### 1.3 与已有方案的本质区别

| | RAG | Auto-Dreamer | ExpeL | **Path Memory（我们）** |
|---|---|---|---|---|
| 存什么 | 文档片段 | 压缩后的抽象记忆 | 经验文本 | **结构化决策路径** |
| 失败信息 | 不存 | 默认遗忘 | 存文本描述 | **存为"此路不通"路标** |
| 检索结果 | 相关文本 | 相关抽象记忆 | 相关经验故事 | **可直接执行的最短路径** |
| Agent 怎么用 | 读了文本自己判断 | 读了抽象记忆自己判断 | 读了故事自己判断 | **路径可以直接执行，Agent 只需验证** |

---

## 二、数据结构设计

### 2.1 Path Memory 核心数据模型

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class StepOutcome(str, Enum):
    SUCCESS = "success"       # 这一步达到了预期效果
    FAILED = "failed"         # 这一步失败了（报错/无结果/结果不对）
    PARTIAL = "partial"       # 部分有用但不够好


@dataclass
class ToolStep:
    """路径中的一个步骤"""
    tool_name: str                      # 工具名，如 "search_arxiv"
    tool_params: dict                   # 参数，如 {"query": "...", "limit": 5}
    outcome: StepOutcome                # 这一步的结果
    error_msg: Optional[str] = None     # 失败原因（如果 failed）
    observation_summary: str = ""       # 这一步观察到的结果摘要
    timestamp: float = 0.0              # 执行时间戳


@dataclass
class TaskPath:
    """一条完整的任务执行路径"""
    path_id: str                        # 唯一标识
    task_description: str               # 任务的自然语言描述
    task_category: str                  # 任务类别（自动分类或手动标记）
    
    # 成功路径（按执行顺序）
    success_steps: list[ToolStep] = field(default_factory=list)
    
    # 失败的尝试（每条都是试过的弯路）
    failed_attempts: list[list[ToolStep]] = field(default_factory=list)
    
    # 元信息
    total_steps: int = 0                # 总尝试步数（含失败）
    success_steps_count: int = 0        # 成功路径的步数
    created_at: float = 0.0
    last_hit_at: float = 0.0            # 最后一次被检索到的时间
    hit_count: int = 0                  # 被检索命中的次数
    success_rate: float = 0.0           # 这条路径的复用成功率


@dataclass  
class StepTransition:
    """路径图中两个步骤之间的转移"""
    from_step: tuple[str, str]          # (tool_name, 参数签名)
    to_step: tuple[str, str]
    outcome: StepOutcome
    count: int = 0                      # 这个转移被走过的次数
```

### 2.2 路径存储

```
两层存储结构：

第一层：Task Index（任务索引） — ChromaDB
  存储: task_description → embedding
  目的: 给定一个新任务，找到历史上最相似的任务

第二层：Path Store（路径存储） — SQLite
  存储: 每个 TaskPath 的完整结构化数据
  目的: 
    - 给定 path_id，返回完整的成功路径 + 失败路径
    - 支持按 task_category 聚合分析
    - 支持统计（哪种路径最短、哪个工具最容易失败）
```

### 2.3 路径编码

每条路径在存入时，生成一个嵌入向量用于相似度匹配：

```python
def encode_path(path: TaskPath) -> list[float]:
    """编码一条路径为向量，用于相似任务匹配"""
    
    # 编码内容 = 任务描述 + 成功路径的步骤摘要
    path_text = f"""
    Task: {path.task_description}
    Steps: {' → '.join(s.tool_name for s in path.success_steps)}
    Failed approaches: {'; '.join(
        ' → '.join(s.tool_name for s in attempt) 
        for attempt in path.failed_attempts
    )}
    """
    
    return embedding_model.encode(path_text)
```

---

## 三、系统架构

### 3.1 运行时流程

```
Agent 收到新任务
    │
    ▼
┌──────────────────────────────────────┐
│ 1. Task Matching（任务匹配）          │
│    将任务描述编码为向量               │
│    → ChromaDB 搜索最相似的历史任务    │
│    → 返回 Top-K 条候选路径            │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ 2. Path Selection（路径选择）         │
│    对 K 条候选路径排序：              │
│    规则 a: 优先选 success_steps 最短的│
│    规则 b: 排除已知的 failed_attempts │
│    规则 c: 优先选 hit_count 最高的    │
│    → 返回最优路径                     │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ 3. Path Execution（路径执行）         │
│    Agent 按 success_steps 顺序执行    │
│    每一步：验证结果是否仍然有效       │
│    如果某步失败 → 跳过，走次优路径    │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ 4. Path Update（路径更新）            │
│    执行完成后：                       │
│    - 如果路径有效 → hit_count++       │
│    - 如果某步失效 → 标记为 failed     │
│    - 如果发现了更短路径 → 新增一条    │
└──────────────────────────────────────┘
```

### 3.2 MCP Tools 设计

```python
# Memory MCP Server 提供的工具

path_add(task_description, success_steps, failed_attempts) 
  → path_id
  # 存一条新的决策路径

path_search(task_description, limit=5)
  → [TaskPath, ...]
  # 找到历史上最相似的任务路径

path_get_shortest(task_description, top_k=5)
  → TaskPath  # 最短的那条成功路径
  # 综合匹配度 + 路径步数 + 复用成功率，返回最优路径

path_avoid(task_description)
  → [list[ToolStep], ...]
  # 返回历史上针对这类任务失败过的尝试，Agent 可以主动避开

path_stats()
  → 路径库统计信息
```

### 3.3 与现有 mem-agent 代码的关系

```
现有 mem-agent 代码：
  - memory/types.py → 新增 TaskPath, ToolStep 类型
  - memory/store.py → 新增 path_search, path_add
  - memory/buffer.py → 保留，作为"当前会话的路径草稿区"
  - memory/retriever.py → 保留，路径检索复用多信号排序
  - consolidation/engine.py → 改名为 path_optimizer.py，逻辑改为路径优化
  - tools.py → 新增 path_* 系列工具
  - server.py → 新增 path_* handler
```

---

## 四、实验设计

### 4.1 核心假设

> **H1（效率假设）：** 使用 Path Memory 的 Agent 完成同等任务的步数显著少于 RAG-only Agent。

> **H2（失败规避假设）：** 保留了失败路径的 Agent 比只保留成功路径的 Agent 更少重复犯同类错误。

> **H3（跨任务泛化假设）：** Path Memory 在相似但不完全相同的任务间仍然有效（相似任务匹配）。

### 4.2 实验设置

**Agent 配置：**
- LLM：Qwen2.5:7B (本地 Ollama，免费)
- 工具集：search_web, search_arxiv, search_news, fetch_content, suggest_keywords
- 任务数量：50 个信息检索任务（学术搜索、新闻搜索、公司调研、技术问答）

**实验组（3 组）：**

| 组 | 记忆方式 | Agent 看到什么 |
|----|---------|---------------|
| **No Memory** | 无 | 只有当前对话上下文 |
| **RAG Memory** | 向量检索相关经验文本 | 5 条最相关的历史经验（自然语言描述） |
| **Path Memory（我们）** | 结构化路径检索 | 1 条最短成功路径 + N 条标记为"避开"的失败路径 |

**实验流程：**

```
Phase 1: 数据收集（Agent 从零探索）
  - 3 组 Agent 各自完成 50 个任务（无记忆辅助）
  - 记录所有路径（成功+失败），构建初始路径库
  - 这 50 个任务的数据作为后续实验的"训练集"

Phase 2: 记忆检索（有记忆辅助）  
  - 同样的 50 个任务（打乱顺序）
  - No Memory 组：重新从零探索
  - RAG Memory 组：每次检索 5 条相关经验文本
  - Path Memory 组：每次检索 1 条最短路径 + 失败标记

Phase 3: 新任务泛化
  - 20 个新任务（与 Phase 1 的任务相似但不相同）
  - 各组使用 Phase 2 中建立的记忆库
  - 评估记忆是否能在新任务上产生正向迁移
```

### 4.3 评估指标

| 指标 | 含义 | 怎么算 |
|------|------|--------|
| **Step Efficiency**（主指标） | 完成任务的平均步数 | total_steps / 50 |
| **Success Rate** | 任务完成率 | completed / 50 |
| **First-Attempt Hit** | 第一次尝试就走对的比例 | correct_first_step / 50 |
| **Failure Avoidance** | 成功避开了已知失败路径的比例 | 走了已知失败路径的次数 |
| **Cross-Task Transfer** | 新任务上效率提升的比例 | Phase 3 步数 / Phase 1 步数 |

### 4.4 论文中预期的结果图

```
图 1: 三组在 50 个任务上的平均步数对比（柱状图）
  预期: No Memory ≈ 8步, RAG ≈ 5步, Path Memory ≈ 2.5步

图 2: 随任务数量增加，各组步数的变化趋势（折线图）
  预期: Path Memory 组随经验积累步数持续下降，RAG 组下降缓慢

图 3: 失败路径的复用分析（热力图）
  展示哪些失败模式被反复触发，Path Memory 如何逐步规避

图 4: Case Study — 两个具体任务的路径对比
  直观展示 Path Memory 如何"走近路"
```

---

## 五、论文大纲

### 标题（草案）
*"Path Memory: Structured Decision-Path Retrieval for Efficient Tool-Augmented Agents"*

### 结构

```
Section 1: Introduction（1.5页）
  - Agent 使用工具时效率低下的问题（重复探索、重复犯错）
  - 现有记忆方案（RAG/Consolidation）的局限：存文本不存路径
  - 本文贡献：Path Memory 的概念、实现、实验验证

Section 2: Related Work（2页）
  - Agent Memory Systems（RAG, Mem0, MemGPT）
  - Memory Consolidation（Auto-Dreamer, Evo-Memory）
  - Experiential Learning for Agents（ExpeL, AWM, Voyager）
  - 差异化：我们不做文本记忆，不做压缩遗忘，我们做路径结构记忆

Section 3: Path Memory（3页）
  - 3.1 形式化定义：TaskPath, ToolStep, StepOutcome
  - 3.2 存储与检索：ChromaDB 任务匹配 + SQLite 路径存储
  - 3.3 路径选择算法：最短路径优先 + 失败路径过滤
  - 3.4 运行时集成：MCP Server 接口

Section 4: Experimental Setup（2页）
  - 50 个信息检索任务的设计
  - 3 组对比实验
  - 评估指标定义

Section 5: Results & Analysis（3页）
  - 主实验结果
  - Case Study × 2
  - Ablation：去掉失败路径只用成功路径 vs 全保留

Section 6: Discussion & Limitations（1页）
  - 什么情况下路径记忆最有效？什么情况下可能有害？
  - 路径库冷启动问题
  - 路径过时失效问题

Section 7: Conclusion（0.5页）
```

### 目标投稿

| 会议 | 截稿 | 适合度 |
|------|------|--------|
| COLING 2027 | ~2026 年底 | ⭐⭐⭐ 中等难度，CCF-B，适合 |
| NAACL 2027 | ~2026.12 | ⭐⭐⭐ 中等难度，CCF-B |
| EMNLP 2027 | ~2027.06 | ⭐⭐ 偏难但可试 |

---

## 六、与已有工作的详细对比

### 6.1 vs ExpeL (2024)

```
ExpeL 做的事：
  "我告诉你，上次我这样做了，然后遇到了这个错误，然后我改成了那样"
  → 存的是"叙事"，Agent 需要阅读理解这个故事

Path Memory 做的事：
  "从任务描述到成功，最短路径是：[arxiv → filter → fetch]"
  "以下路径不要走：[web_search], [arxiv_without_filter]"
  → 存的是"路径"，Agent 可以直接执行

本质区别：
  ExpeL: 经验 → 文本 → Agent 理解 → 决策（有信息损失）
  Ours:   经验 → 结构化路径 → 直接检索 → 执行（信息无损）
```

### 6.2 vs Auto-Dreamer (2026.05)

```
Auto-Dreamer 做的事：
  把 50 条记忆压缩成 8 条抽象记忆
  → 关键设计：压缩 = 遗忘不重要的

Path Memory 做的事：
  保�留所有 50 条路径，但在检索时只返回"最有用的一条"
  → 关键设计：检索精度 > 存储成本
  
为什么我们不需要压缩：
  Auto-Dreamer 的 Agent 需要把所有记忆塞进上下文窗口 → 所以必须压缩
  我们只塞一条路径（20 tokens）→ 不需要压缩
```

### 6.3 vs RAG

```
RAG 做的事：
  User: "帮我找论文"
  RAG: 搜到 5 条相关文本 → "上次用户用 arxiv 搜到了论文"
       → "arXiv API 有时候会限流"
       → "有用户喜欢用 Google Scholar"
       → "论文筛选可以用年份过滤"
  Agent: 读 500 tokens 的文本 → 自己组织策略 → 执行

Path Memory 做的事：
  User: "帮我找论文"  
  Path Memory: 匹配到最相似任务 → 返回：
    成功路径: [search_arxiv(query, filter=2026), fetch_top(3)]
    避开路径: [search_web]  ← 因为学术搜索中 web search 噪音多
  Agent: 直接执行路径，只需要 20 tokens 的指令

核心差异：
  RAG: 搜到的是"素材"，Agent 需要重新加工
  Path Memory: 搜到的是"答案"，Agent 只需要执行
```

---

## 七、代码实现计划

### 7.1 需要新增的文件

```
src/mem_agent/
├── path/
│   ├── __init__.py
│   ├── types.py          # TaskPath, ToolStep, StepOutcome
│   ├── encoder.py         # 路径编码为向量
│   ├── store.py           # 路径存储（SQLite + ChromaDB 索引）
│   ├── retriever.py       # 路径检索 + 最短路径选择
│   └── recorder.py        # 自动记录 Agent 的工具调用路径
```

### 7.2 需要修改的现有文件

```
src/mem_agent/
├── tools.py              # 新增 path_add, path_search, path_get_shortest, path_avoid
├── server.py             # 新增 handler
├── memory/store.py       # 可能不需要改了，路径存储独立
```

### 7.3 实现优先级

```
Phase 1（本周）: path/types.py + path/store.py
  → 能存路径、能搜路径（最基础功能）

Phase 2（下周）: path/retriever.py  
  → 最短路径选择算法

Phase 3（第三周）: path/recorder.py
  → Agent 自动记录工具调用 → 自动生成 TaskPath

Phase 4（第四周起）: 实验
  → 在 50 个任务上跑三组对比实验
```

---

## 八、风险评估

| 风险 | 等级 | 应对 |
|------|------|------|
| 路径匹配不准（新任务匹配到不相关的旧路径） | 中 | 设置相似度阈值，低于阈值则不走路径记忆 |
| 路径过时（旧路径在新环境下失效） | 中 | 每次执行路径时做"新鲜度验证"，失效则更新 |
| 路径库冷启动（前 10 个任务没有路径可参考） | 低 | 这正是 No Memory baseline 的情况，数据本身就有意义 |
| 实验效果不够显著 | 中 | 即使效率提升只有 20-30%，论文也可以论证"路径记忆有正向效果" |

---

## 九、与毕设要求的对应

| 毕设要求 | Path Memory 对应 |
|---------|-----------------|
| 理论研究 | 结构化路径记忆的形式化定义 + 路径选择算法 |
| 系统实现 | Memory MCP Server + Path Store |
| 实验验证 | 50 任务 × 3 组对比 |
| 创新点 | 首次将 Agent 记忆定义为"决策路径结构"而非"文本存储" |

---

## 十、现有传家宝（可直接复用）

| 已有资产 | 复用方式 |
|---------|---------|
| search-mcp | 作为实验平台的标准工具集 |
| mem-agent 骨架代码 | MCP Server 框架直接复用 |
| ChromaDB 经验 | 路径编码的向量索引 |
| MultiSignalRetriever | 路径检索排序 |
| server.py / tools.py | 新增 handler 即可 |
