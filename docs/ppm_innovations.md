# PPM 创新方法
# PPM Innovations for Math Reasoning with MCTS

> COMS 6895 Advanced Big Data and AI — Columbia University  
> Project: rStar-Math inspired MCTS + Process Preference Model

---

## 背景与动机 / Background & Motivation

### 原始 PPM 的四大缺陷 / Four Limitations of the Baseline PPM

| # | 问题 / Problem | 影响 / Impact |
|---|---|---|
| 1 | 步骤孤立评分，不知道题目内容 / Steps scored without problem context | 同一步骤在不同题中质量不同，但 PPM 给出相同分数 |
| 2 | 固定间隔损失 margin=1.0 / Fixed hinge margin | 无论奖励差距大小，学习信号强度相同 |
| 3 | MCTS Rollout 永远取第一个动作 / Rollout always picks `actions[0]` | 丧失 Monte Carlo 随机探索特性，值估计有偏 |
| 4 | 训练数据仅 20 道题 MATH Level 5 / Tiny dataset | PPM 严重过拟合，泛化能力几乎为零 |

---

## 创新总览 / Innovation Overview

```
┌─────────────────────────────────────────────────────────────┐
│  5 Innovations Implemented                                  │
│                                                             │
│  架构层 Architecture                                         │
│    ① Context-Aware Dual Encoder PPM (上下文感知双编码器)      │
│                                                             │
│  损失函数层 Loss Function                                    │
│    ② Reward-Proportional Soft Margin Loss (奖励自适应损失)   │
│    ③ Hard Negative Mining (硬负样本挖掘)                     │
│                                                             │
│  训练层 Training                                             │
│    ④ Multi-Dataset Curriculum Learning (多数据集课程学习)    │
│    ⑤  Dropout + Cosine LR Annealing (正则化+余弦退火)        │
│                                                             │
│  搜索层 Search                                               │
│    ⑥ PPM-Guided Monte Carlo Rollout (PPM引导随机展开)        │
└─────────────────────────────────────────────────────────────┘
```

---

## Innovation ① — 上下文感知双编码器 PPM
## Innovation ① — Context-Aware Dual Encoder PPM

### 问题 / Problem

基准 PPM 将每个推理步骤**孤立**嵌入，完全不知道它是在解什么题。

The baseline PPM embeds each reasoning step **in isolation**, with no knowledge of which problem is being solved.

```
baseline:   step_text  →  embed(384)  →  MLP  →  score
                ✗ no problem context
```

### 创新方案 / Innovation

将题目陈述与步骤文本**拼接后联合编码**，PPM 的打分成为"该步骤对于这道题而言有多好"。

Concatenate the problem statement embedding with the step embedding, so the PPM score becomes *"how good is this step for this specific problem."*

```
innovation:
  problem_emb (D) ──┐
                     ├─ concat → (2D) → StepEncoder(2D→H) → ValueHead → score
  step_emb    (D) ──┘
                ✓ problem-conditioned scoring
```

### 实现 / Implementation

```python
# src/core/ppm.py — ContextAwarePPM
class ContextAwarePPM(ProcessPreferenceModel):
    def evaluate_step(self, step: str, embedder, problem: str = "") -> float:
        step_emb = embedder.embed_text(step)           # dim D
        prob_emb = embedder.embed_text(problem)        # dim D
        combined = prob_emb + step_emb                 # dim 2D (concat)
        return self(torch.FloatTensor(combined)).item()
```

### 预期效果 / Expected Effect

- 对同一步骤"代入 x=2"，在代数题中得高分，在数论题中得低分
- Distinguishes "substitute x=2" as high-quality for an algebra problem, irrelevant for a number theory problem

---

## Innovation ② — 奖励自适应软间隔损失
## Innovation ② — Reward-Proportional Soft Margin Loss

### 问题 / Problem

原损失函数使用**固定间隔** margin=1.0，对奖励差距为 0.01 和 0.9 的两对样本施加完全相同的学习压力。

Original loss used a **fixed margin** of 1.0, applying identical learning pressure regardless of whether the reward gap between a pair is 0.01 or 0.9.

```
Original:   L = relu(v_neg - v_pos + 1.0).mean()
                                    ^^^
                              fixed — ignores reward gap
```

### 创新方案 / Innovation

将间隔设为奖励差的函数：奖励差越大，学习信号越强。

Make the margin proportional to the actual reward gap: larger quality differences push harder.

```
Innovation:  margin = max(0.1, r_preferred - r_non_preferred)
             L = relu(v_neg - v_pos + margin).mean()
```

### 对比示例 / Comparison Example

| 样本对 / Pair | 奖励差 / Reward Gap | 原间隔 / Old Margin | 新间隔 / New Margin |
|---|---|---|---|
| 几乎相同质量 / Nearly equal | 0.05 | 1.0 | 0.10 |
| 明显差距 / Clear gap        | 0.50 | 1.0 | 0.50 |
| 极端差距 / Extreme gap      | 0.90 | 1.0 | 0.90 |

效果：高置信度的偏好对贡献更大梯度，低置信度的对贡献更小，避免噪声标注污染训练。

Effect: High-confidence pairs contribute larger gradients; near-equal pairs contribute less, reducing noise from ambiguous labels.

---

## Innovation ③ — 硬负样本挖掘
## Innovation ③ — Hard Negative Mining

### 问题 / Problem

随机配对的偏好样本中，大多数 preferred/non-preferred 步骤**语义差异很大**，PPM 容易区分，学习效率低。

Randomly paired preferred/non-preferred steps are usually **semantically distant**, making them trivially easy to distinguish — the PPM learns little from them.

### 创新方案 / Innovation

在每个 batch 内，找到与 preferred 步骤**余弦相似度最高**的 non-preferred 步骤作为难样本，替换部分随机负样本。

Within each batch, find the non-preferred steps with the **highest cosine similarity** to the preferred steps and substitute them as hard negatives.

```
Hard Negative Selection:
  sim_matrix = pref_norm @ nonpref_norm.T    # cosine similarities
  hard_idx   = sim_matrix.argmax(dim=1)      # most confusing non-preferred
  nonpref[0:n_hard] = nonpref[hard_idx]      # replace easy negatives
```

### 直觉示意 / Intuition

```
Easy negative (random):
  preferred:      "Factor x²-4 = (x+2)(x-2)"   ← clearly algebraic
  non_preferred:  "Draw a Venn diagram"           ← clearly wrong context
  → PPM trivially learns to separate these

Hard negative (mined):
  preferred:      "Factor x²-4 = (x+2)(x-2)"   ← correct factoring
  non_preferred:  "Factor x²-4 = (x-2)(x+1)"   ← similar-looking, wrong sign
  → PPM must learn fine-grained quality distinction
```

### 实现 / Implementation

```python
# src/core/ppm.py — _apply_hard_negatives()
def _apply_hard_negatives(pref_embs, nonpref_embs, n_hard):
    sim = F.normalize(pref_embs) @ F.normalize(nonpref_embs).T
    hard_idx = sim.argmax(dim=1)
    new_nonpref = nonpref_embs.clone()
    new_nonpref[:n_hard] = nonpref_embs[hard_idx[:n_hard]]
    return pref_embs, new_nonpref
```

每批替换 30% 的负样本为难样本（`hard_negative_ratio=0.3`）。

30% of negative samples per batch are replaced with hard negatives.

---

## Innovation ④ — 多数据集课程学习
## Innovation ④ — Multi-Dataset Curriculum Learning

### 问题 / Problem

原训练集仅使用 **20 道** MATH Level 5 题目（3个科目），PPM 几乎没有可泛化的训练信号。

The original setup used only **20 problems** from MATH Level 5 (3 subjects) — far too few for any meaningful generalization.

### 训练数据来源 / Training Data Sources

| 数据集 / Dataset | 来源 / Source | 规模 / Scale | 难度 / Difficulty |
|---|---|---|---|
| GSM8K | `openai/gsm8k` | ~8.5K | ⭐ (Level 1) |
| MATH All Levels | `EleutherAI/hendrycks_math` | ~12K | ⭐⭐–⭐⭐⭐⭐⭐ (L1-L5) |
| NuminaMath-CoT | `AI-MO/NuminaMath-CoT` | 860K (sampled) | ⭐⭐⭐ |
| OlympiadBench | `math-ai/OlympiadBench` | ~8K | ⭐⭐⭐⭐⭐ |
| AIME 1983-2024 | `Maxwell-Jia/AIME_1983_2024` | ~900 | ⭐⭐⭐⭐⭐⭐ |

### 课程学习策略 / Curriculum Learning Strategy

```
Easy → Medium → Hard training order (--curriculum flag)

difficulty 1  GSM8K          ──→  PPM learns: what is a valid reasoning step?
difficulty 2  MATH L1-L2     ──→  PPM learns: correct algebraic manipulation
difficulty 3  MATH L3-L4     ──→  PPM learns: multi-step proof structure
difficulty 4  MATH L5 / NuminaMath ──→  PPM learns: competition-level reasoning
difficulty 5  OlympiadBench  ──→  PPM learns: olympiad insight steps
difficulty 6  AIME           ──→  PPM sharpened on hardest problems
```

### 使用示例 / Usage

```bash
# Full multi-dataset training with curriculum learning
python experiments/run_experiment.py \
    --dataset math_all gsm8k numina olympiad aime \
    --curriculum \
    --train 200 --test 30
```

### 预期效果 / Expected Effect

- 训练样本量从 ~20 道扩充到 **数万道题**
- 跨数据集偏好对覆盖更广泛的数学子领域
- 课程排序使 PPM 先掌握基础推理模式，再泛化到竞赛级别

- Training set grows from ~20 to **tens of thousands** of problems  
- Cross-dataset preference pairs cover diverse mathematical sub-domains  
- Curriculum ordering lets the PPM master basic reasoning before competition-level problems

---

## Innovation ⑤ — Dropout + 梯度裁剪 + 余弦退火
## Innovation ⑤ — Regularization & Cosine LR Annealing

### 动机 / Motivation

小数据集（20 题）上的 PPM 训练极易过拟合；原代码无正则化机制，也无学习率调度。

Training on small datasets is highly prone to overfitting; the original code had no regularization and no LR schedule.

### 三个改进 / Three Improvements

**Dropout** — 在 StepEncoder 和 ValueHead 每层后加 Dropout(p=0.1)，防止神经元共适应。

```python
self.encoder = nn.Sequential(
    nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
    nn.Dropout(0.1),                          # ← NEW
    nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
    nn.Dropout(0.1),                          # ← NEW
)
```

**Gradient Clipping** — 裁剪梯度范数到 1.0，防止稀疏偏好对引起的梯度爆炸。

```python
torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)   # ← NEW
```

**Cosine LR Annealing** — 余弦退火调度，学习率从 lr 平滑降至 lr×0.01，避免末期震荡。

```python
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
```

```
LR schedule:
  5e-4 ──────╮
              ╰──╮
                  ╰──╮
                      ╰──╮
                          ╰── 5e-6  (eta_min)
  epoch: 0      25     50     80
```

---

## Innovation ⑥ — PPM 引导的蒙特卡洛展开采样
## Innovation ⑥ — PPM-Guided Monte Carlo Rollout

### 问题 / Problem

原 MCTS Rollout 阶段**永远选第一个 action**，完全是确定性的，丧失了 Monte Carlo 的随机探索特性。每次 simulation 走同一条路，值估计无法收敛到真实期望值。

Original MCTS rollout **always selected `actions[0]`** — purely deterministic. Every simulation followed the same path, making value estimates biased.

```python
# Original (wrong):
def _select_rollout_action(self, actions, state):
    return actions[0]                     # no Monte Carlo at all!
```

### 创新方案 / Innovation

用 PPM 分数作为权重进行 **softmax 采样**，不同 simulation 探索不同路径，恢复 Monte Carlo 属性。

Use PPM scores as **sampling weights** so different simulations explore different paths, restoring the Monte Carlo property.

```python
# Innovation (PPM-guided sampling):
def _select_rollout_action(self, actions, state):
    if self._ppm is not None and len(actions) > 1:
        problem = state.split('\n')[0]
        scores = [ppm.evaluate_step(a, model, problem=problem) for a in actions]
        probs  = softmax(scores)           # proportional weights
        idx    = rng.choice(len(actions), p=probs)   # stochastic!
        return actions[idx]
    return actions[0]                      # fallback without PPM
```

### 对比 / Comparison

```
Baseline rollout:         Innovation rollout:
  Sim 1: A→B→C→D          Sim 1: A→B→C→D  (PPM-weighted sample)
  Sim 2: A→B→C→D  same!   Sim 2: A→E→F→G  (different path!)
  Sim 3: A→B→C→D  same!   Sim 3: A→B→H→I  (different path!)

Value estimate: biased     Value estimate: unbiased (law of large numbers)
```

---

## 综合效果对比 / Combined Effect Summary

| 改进维度 / Dimension | 基准 / Baseline | 创新后 / With Innovations |
|---|---|---|
| PPM 评分依据 | 步骤文本 (无题目) | 题目 + 步骤 联合编码 |
| 损失函数 | 固定 margin=1.0 | 奖励自适应 margin |
| 训练样本 | ~20 道 MATH L5 | MATH + GSM8K + Olympiad + AIME |
| 负样本质量 | 随机配对 | 硬负样本挖掘 |
| 过拟合防护 | 无 | Dropout + 梯度裁剪 |
| LR 调度 | 固定 | 余弦退火 |
| Rollout 策略 | 确定性 (actions[0]) | PPM 加权随机采样 |

---

## 架构演进图 / Architecture Evolution

```
━━━━━━━━━━━━━━━━━━━━━━━━━ BASELINE ━━━━━━━━━━━━━━━━━━━━━━━━━
  step_text → embed(384) → Linear(384→256) → Linear(256→1)
  Loss: relu(v_neg - v_pos + 1.0)
  Data: 20 × MATH Level 5 problems
  Rollout: always actions[0]

━━━━━━━━━━━━━━━━━━━━━━━━ INNOVATION ━━━━━━━━━━━━━━━━━━━━━━━━
  problem_text ──→ embed(384) ──┐
                                 ├→ concat(768) → Encoder → Head → score
  step_text    ──→ embed(384) ──┘
  
  Encoder: Linear(768→256) + LN + ReLU + Dropout(0.1)
           Linear(256→256) + LN + ReLU + Dropout(0.1)
  Head:    Linear(256→128) + LN + ReLU + Dropout(0.1) + Linear(128→1)
  
  Loss: relu(v_neg - v_pos + clamp(r_pos - r_neg, min=0.1))
        + hard negatives (cosine-similarity mined, 30% of batch)
  
  LR:   CosineAnnealing(T_max=epochs, eta_min=lr×0.01)
  Grad: clip_norm(max=1.0)
  
  Data: MATH(all) + GSM8K + NuminaMath + OlympiadBench + AIME
        sorted by difficulty for curriculum learning
  
  Rollout: softmax(PPM_scores) sampling → different path per simulation
```

---

## 引用 / References

```
rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking
  Li et al., 2024

Solving Math Word Problems via Cooperative Reasoning induced Language Models
  Zhu et al., 2023

NuminaMath: A Large-Scale Math Competition Dataset
  AI-MO, 2024

OlympiadBench: A Challenging Benchmark for Promoting AGI with Olympiad-Level
  Bilingual Multimodal Scientific Problems
  He et al., 2024
```
