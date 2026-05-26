# Step 1: Energy-Based Models (EBM) — The Foundation

**Paper:** Yann LeCun, [*A Path Towards Autonomous Machine Intelligence*](papers/10356_a_path_towards_autonomous_mach.pdf) (2022) — **§4.1**  
**Curriculum:** `misc/JEPA_2022_PLAN_GPU.md` — Step 1  
**Code:** `experiments/step01_ebm/`

---

## 1. Paper anchor

Section 4.1 of the paper introduces the basic object used throughout the rest of the architecture:

> *"An EBM is a trainable system that, given two inputs x and y, produces a scalar energy F_w(x, y) that measures the degree of incompatibility between x and y."*

Important nuances from the same section (and surrounding discussion):

- **Low energy** means *compatible*; **high energy** means *incompatible*.
- The learnable weights **w** define an **energy landscape** over pairs (x, y) — especially over choices of y for a fixed x.
- An EBM is **not** required to be a normalized probability model. There is no partition function in this step; we are not doing softmax over y.

Step 1 is the **foundation** of the 10-step curriculum. Every later idea — contrastive training (§4.2), joint embeddings (§4.3), JEPA (§4.4), VICReg (§4.5), hierarchical stacks, and the full agent — assumes you already understand what F_w(x, y) is and why **training it naively fails**.

---

## 2. Problem we solved

### What question this experiment answers

Before we can learn *representations*, *predictions in embedding space*, or *planning through a world model*, we need a precise answer to:

**How do we score whether two variables “go together”?**

Step 1 implements that scorer as a neural network F_w(x, y) and trains it in the simplest possible way: **only on correct pairs**, minimizing energy on positives.

### What we deliberately do *not* solve yet

We do **not** train a useful compatibility function. That is intentional. The experiment is designed to fail in an instructive way so that §4.2 (contrastive methods) feels necessary rather than optional.

### The core failure mode we demonstrate: energy collapse

If the loss only says “make F(x, y) small for training pairs (x, y)”, the network is never told that **wrong** y values should have **high** energy. Gradient descent can therefore:

1. Drive energies on training pairs arbitrarily low (unbounded descent), and/or  
2. Produce a landscape that does **not** separate correct y from incorrect y.

The paper’s roadmap depends on recognizing this **collapse** early. Without Step 1, contrastive losses, regularizers, and JEPA training can look like arbitrary extra terms. After Step 1, they look like **fixes to a known bug**.

---

## 3. Data

We use **synthetic 2D manifolds** so we can plot the energy landscape F_w(x, y) as a heatmap over y ∈ ℝ².

### Default: Swiss roll

| Item | Detail |
|------|--------|
| **x** | Shape `(N, 1)`. A scalar in **[0, 1]** encoding normalized position along the roll (rank of arc parameter t among samples). |
| **y** | Shape `(N, 2)`. Point on the Swiss roll: y₁ = t cos t, y₂ = t sin t, with small Gaussian noise (σ = 0.05). |
| **N** | 10,000 pairs |
| **Pair type** | **Positive only**: each (x, y) is a matched pair from the same sample index. |

Implementation: `shared/data.py` → `swiss_roll_pairs()`.

### Alternative: concentric circles

| Item | Detail |
|------|--------|
| **x** | Angle θ normalized to [0, 1] |
| **y** | Point on inner (r = 1) or outer (r = 2) circle + noise |
| **Use** | `train.py --dataset circles` |

### Why this data?

- **Low-dimensional y** lets us visualize the full energy surface over a grid in ℝ².
- **Structured manifold** means a *good* EBM (after Step 2) should show **low energy near the true curve** and higher energy off-manifold.
- **1D x** models a “conditioning variable” (index, context, partial observation) without image complexity — same interface as later steps where x is richer.

We do **not** use negative pairs in Step 1. There is no ŷ sampled from other indices.

---

## 4. Strategy

### 4.1 Architecture

The EBM follows the plan’s template (`experiments/step01_ebm/model.py`):

```
x (1D) ──► x_encoder: MLP(1 → 64 → 128) ──► s_x (128D) ──┐
                                                          ├── concat (256D) ──► energy_head: MLP(256 → 128 → 64 → 1) ──► F_w(x,y)
y (2D) ──► y_encoder: MLP(2 → 64 → 128) ──► s_y (128D) ──┘
```

- Separate encoders let x and y live in different input spaces but meet in a shared space before the scalar energy is computed.
- ReLU MLPs (`shared/modules.py`) keep the build minimal and reusable in Step 2+.

**Interpretation:** The network learns arbitrary features of x and y, then a final MLP decides how “incompatible” the pair is. There is no explicit distance metric or cosine similarity yet (that appears in joint embedding architectures in Step 3).

### 4.2 Training (naive — by design)

| Hyperparameter | Default |
|----------------|---------|
| **Loss** | L = mean_batch F_w(x, y) |
| **Optimizer** | Adam, lr = 1e-3 |
| **Epochs** | 200 |
| **Batch size** | 256 |
| **Seed** | 42 |
| **Device** | `mps` if available (`shared/device.py`), else CPU |

Procedure (`train.py`):

1. Load all positive pairs into a `TensorDataset`.
2. For each epoch, shuffle batches, forward (x, y), compute mean energy, backprop, step.
3. Log per-epoch **mean** and **std** of F over all batch elements → `outputs/loss_history.json`.
4. Save weights → `outputs/checkpoint.pt`.

**What is missing from the loss:** any term that increases energy for (x, ŷ) when ŷ ≠ y. That omission is the entire pedagogical point.

### 4.3 Visualization strategy

`visualize.py` loads the checkpoint and:

1. Fixes one training sample’s x (default: index 0).
2. Evaluates F_w(x, y) on an 80×80 grid of y over [-3, 3]².
3. Writes:
   - `energy_heatmap.png` — raw energy over the plane (white scatter = training y points).
   - `energy_deviation_heatmap.png` — F − mean(F) on the grid (highlights flat vs structured landscape).
   - `energy_surface.png` — 3D plot of the same grid.
   - `training_curve.png` — mean energy ± batch std vs epoch.

This matches the plan’s three visualization bullets and adds a deviation map to make “flat collapse” easier to see when all energies are huge and negative.

### 4.4 Hardware (MPS)

Training and inference follow the **build-toy-model** skill: tensors moved to MPS, matplotlib inputs taken from `.cpu().numpy()`. On Apple Silicon (M5, 24 GB unified memory), this step is trivially small; the same `get_device()` helper carries forward to heavier steps (CIFAR, video).

---

## 5. What we implemented

| File | Role |
|------|------|
| `experiments/step01_ebm/model.py` | `EBM` module: dual encoders + energy head |
| `experiments/step01_ebm/train.py` | Naive training loop, checkpoint + history JSON |
| `experiments/step01_ebm/visualize.py` | Energy landscapes and training curve |
| `shared/device.py` | MPS/CPU selection |
| `shared/data.py` | Swiss roll & circles pair generators |
| `shared/modules.py` | Generic `MLP` builder |
| `shared/viz.py` | Matplotlib helpers for heatmaps and curves |

**Outputs directory** (`outputs/`):

| Artifact | Purpose |
|----------|---------|
| `checkpoint.pt` | Weights + metadata (dataset name, seed, epochs) |
| `loss_history.json` | Per-epoch mean/std energy |
| `*.png` | Figures referenced below |
| `explanation.md` | This document |

---

## 6. Results and evidence

### What “success” means for Step 1

Step 1 is successful when you **observe collapse**, not when you get a good classifier or density model.

Typical behavior after 200 epochs (Swiss roll, default hyperparameters):

| Epoch (approx.) | Mean F(x, y) | Interpretation |
|-----------------|--------------|----------------|
| 1 | ~−30 | Model already prefers lowering energy |
| 10–50 | Large negative, growing fast | Unbounded descent |
| 200 | Very large negative (e.g. −10¹³ order) | No floor in the architecture; optimizer keeps pushing F down |

The **training curve** (`training_curve.png`) should show a steep downward trend — often looking almost vertical on a linear y-axis once collapse accelerates. Batch std may also grow, reflecting runaway scales rather than meaningful structure.

### Energy landscape plots

For a **fixed x**:

- A **well-trained** compatibility model (after Step 2) should show a **valley** along the data manifold in y-space and higher energy elsewhere.
- After **naive** Step 1 training, you often see:
  - **Raw heatmap** (`energy_heatmap.png`): extremely negative values across much of the grid, with the manifold not clearly singled out.
  - **Deviation heatmap** (`energy_deviation_heatmap.png`): structure relative to the grid mean — if collapse is severe, the map can still look relatively flat (low coefficient of variation), meaning the model is not carving a sharp “compatibility funnel” over y.

The console reports grid statistics, e.g. energy range and CV = std / |mean|. A small CV suggests the landscape is **flat relative to its magnitude** — consistent with “everything is equally compatible” in the limit.

### Connection to the plan’s key intuition

> *Without a mechanism to push energies UP for incorrect y values, the model learns to make everything low energy.*

Our implementation shows a stronger version of that: energies are not merely “low and equal” — they can **diverge to large negative values** because nothing in L = F(x, y) penalizes making F more negative forever. That is the same class of problem contrastive methods address by **repelling** wrong pairs.

---

## 7. What this establishes

After completing Step 1, you should be able to:

1. **Define** F_w(x, y) and explain compatibility vs incompatibility in the paper’s language.
2. **Implement** a minimal EBM with separate encoders for heterogeneous inputs.
3. **Visualize** an energy landscape over the output space y for fixed x.
4. **Recognize collapse** from training logs and plots — not confuse it with “good convergence”.
5. **Articulate why** a second training signal (contrastive negatives, regularizers, etc.) is required before EBMs become useful for representation learning.

You have **not** yet established:

- Discriminative structure over y (Step 2).
- Embedding-space distance as energy (Step 3).
- Prediction in representation space (Step 5+).

---

## 8. Connection to the paper (deeper reading)

### §4.1 in practice

The paper’s EBM is a **scoring function**, not a generative model. Step 1 makes that concrete: we never sample y from exp(−F); we only **evaluate** F on pairs. That mindset carries through JEPA, where we care about **representation prediction** rather than pixel reconstruction.

### Why §4.2 exists

Table 1 in the appendix (contrastive losses: InfoNCE, hinge, logistic) assumes you already have F_w(x, y) and need a **training objective** that shapes the landscape. Step 1 explains *why* those rows exist: pairwise repulsion terms implement “pull up energy on contrastive samples” that naive minimization lacks.

### Bridge to joint embeddings (§4.3+)

Later, energy is often **distance in embedding space**: F(x, y) = ‖s_x − s_y‖². Step 1’s two encoders foreshadow that structure, but we keep a learned `energy_head` on the concatenation so the model is fully general — matching §4.1’s wording before the paper specializes to JEAs.

### Bridge to the full agent (§3)

The autonomous agent uses **cost** and **intrinsic cost** modules — scalar signals that shape behavior. EBMs are the low-level ancestor of “assign a scalar score to a configuration”. Understanding collapse prevents confusing “low cost everywhere” with “good planning”.

### Misconceptions this step clears up

| Misconception | Correction from Step 1 |
|---------------|-------------------------|
| “Lower loss = better model” | Here, lower loss = worse *useful* EBM (collapse). |
| “EBM = energy-based generative model” | We never normalize or sample; we only score pairs. |
| “We can skip contrastive training” | Naive training proves you cannot. |

---

## 9. Limitations of this toy

- **No negatives**, no MCMC, no score matching — only the broken positive-only baseline.
- **Tiny MLPs**, not the convnets or transformers used in real perception stacks.
- **Synthetic 2D data** — no high-dimensional y where contrastive sampling is expensive (the paper’s scaling argument in §4.2).
- **Unbounded energy** — no architectural constraint (e.g. softplus, bounded head) to force “flat zero” collapse instead of “−∞” collapse; both illustrate the same conceptual failure.
- **Single fixed x in default viz** — you can pass `--x-index` to explore others, but we do not aggregate landscapes over many x values.

These limits are addressed in **Step 2: Contrastive Training of EBMs** (`experiments/step02_contrastive_ebm/`), which reuses the same `EBM` class and adds hinge, InfoNCE, and logistic losses plus negative sampling strategies.

---

## 10. Next step

**Step 2** keeps the architecture identical and changes only the **objective**:

- Push **down** F(x, y) on positives (as now).
- Push **up** F(x, ŷ) on contrastive negatives ŷ (random, hard, or in-batch).

You should see energy landscapes with **valleys on the data manifold** instead of indiscriminate collapse. That is the first moment the EBM becomes a **selective** compatibility function — the prerequisite for joint embeddings and JEPA.

---

## Reproduce

From the repository root:

```bash
uv run python experiments/step01_ebm/train.py
uv run python experiments/step01_ebm/visualize.py
```

Optional:

```bash
uv run python experiments/step01_ebm/train.py --dataset circles --epochs 200
uv run python experiments/step01_ebm/visualize.py --x-index 500
```

After training, inspect `outputs/training_curve.png` first, then `outputs/energy_deviation_heatmap.png`, and read `outputs/loss_history.json` for numeric collapse evidence.
