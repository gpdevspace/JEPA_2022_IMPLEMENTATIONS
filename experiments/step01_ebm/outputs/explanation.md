# Step 1: Energy-Based Models (EBM) — The Foundation

**Paper:** Yann LeCun, [*A Path Towards Autonomous Machine Intelligence*](papers/10356_a_path_towards_autonomous_mach.pdf) (2022) — **§4.1**  
**Curriculum:** `misc/JEPA_2022_PLAN_GPU.md` — Step 1  
**Code:** `experiments/step01_ebm/`

---

## 1. Paper anchor

Section 4.1 introduces the object reused everywhere later in the stack:

> *"An EBM is a trainable system that, given two inputs x and y, produces a scalar energy F_w(x, y) that measures the degree of incompatibility between x and y."*

Key ideas for this step:

- **Low energy** = compatible pair; **high energy** = incompatible pair.
- **w** parameterizes an **energy landscape** — for fixed x, varying y traces a surface over the output space.
- An EBM is **not** a normalized density: we never compute Z = ∫ exp(−F) dy in Step 1.

Step 1 is the curriculum foundation. Contrastive training (§4.2), joint embeddings (§4.3), JEPA (§4.4), and the full agent (§3) all assume you understand F_w(x, y) and why **naive training collapses**.

---

## 2. Problem we solved

### Question

**How do we score whether two variables belong together?**

We implement F_w(x, y) as a neural network and train with the simplest objective: minimize energy on **correct** pairs only.

### Deliberate non-goal

We do **not** learn a useful compatibility function yet. Success = **observing failure** (collapse) so §4.2 feels mandatory.

### Failure mode: energy collapse

With loss L = mean F(x, y) on positives only:

1. Nothing pushes F(x, ŷ) **up** for wrong ŷ.
2. The network can drive F on training pairs toward **arbitrarily negative** values (unbounded descent).
3. The landscape over y for fixed x need not form a **valley on the data manifold** — everything can look “compatible.”

---

## 3. Data

### 3.1 What x and y represent

In the paper’s general notation, **x** and **y** are two pieces of information whose joint plausibility we score. They may live in different spaces (e.g. context vs outcome, observation vs prediction target).

| Variable | In this toy | Semantic role |
|----------|-------------|----------------|
| **x** | 1D scalar in [0, 1] | A **conditioning index** along the generative process — “where we are” along the manifold before we observe the 2D outcome. It stands in for partial context, time index, or latent coordinate. |
| **y** | 2D vector in ℝ² | The **outcome** we judge for compatibility with x — a point in output space. Later steps replace ℝ² with images, frames, or embeddings. |

The EBM answers: *given this x, is this y a plausible partner?* That is exactly the interface F_w(x, y) in §4.1, stripped to the smallest geometry where we can **see** the answer as a picture.

### 3.2 How the data is built

**Default: Swiss roll** (`shared/data.py` → `swiss_roll_pairs()`)

1. Sample arc parameter t ~ Uniform(0, 2π), N = 10,000.
2. Embed on the roll: y₁ = t cos t, y₂ = t sin t.
3. Add isotropic Gaussian noise with σ = 0.05.
4. Build x as the **normalized rank** of t among samples, mapping to [0, 1]. Same index i gives matched (xᵢ, yᵢ).

**Alternative: concentric circles** (`--dataset circles`)

- Half the points on radius 1, half on radius 2; x = θ / 2π for the polar angle.
- Same (x, y) pairing by index; useful for a simpler, symmetric manifold.

| Item | Swiss roll (default) |
|------|---------------------|
| x shape | (10000, 1) |
| y shape | (10000, 2) |
| Pairing | Positive only: (xᵢ, yᵢ) matched |
| Negatives | **None** in Step 1 |

### 3.3 Why this dataset was chosen

| Reason | Benefit for Step 1 |
|--------|-------------------|
| **y is 2D** | We can evaluate F_w(x, y) on a dense grid in the plane and draw heatmaps — the energy **landscape** from the paper becomes literal terrain. |
| **y lies on a 1D curve in 2D** | A *good* EBM (Step 2+) should concentrate low energy **on** the Swiss roll and higher energy **off** it. We can visually check whether the model learned a “compatibility valley.” |
| **x is low-dimensional** | We can fix one x (one conditioning value) and sweep all y — the standard “slice” of the landscape §4.1 describes. |
| **Synthetic + noiseless structure** | No dataset download, fast on MPS, reproducible seed — focus stays on **training dynamics**, not data engineering. |
| **Known ground-truth manifold** | White scatter overlays in plots show where true pairs live; you can judge whether low-energy regions align with the roll. |

We did **not** use MNIST, CIFAR, or video here because high-dimensional y would require slices, PCA, or many 2D projections — that obscures the first lesson (what F and collapse mean).

### 3.4 Analogy to the full paper setting

| This toy | Later in the curriculum / paper |
|----------|----------------------------------|
| x = scalar index along roll | x = past frames, partial observation, or state s_t |
| y = 2D point on manifold | y = future frame, augmented image, or target representation s_y |
| F_w(x, y) = MLP on encodings | F = ‖s_x − s_y‖², prediction error in embedding space, or planning cost |
| Heatmap over ℝ² | t-SNE / scalar summaries when y is high-D |

The **pairing structure** is the same: the model always scores **consistency between two inputs**, not reconstruction of one from the other alone.

### 3.5 What we excluded (and why)

- **No negative pairs ŷ** — Step 1 isolates the bug in positive-only training; Step 2 adds contrastive repulsion.
- **No held-out test split** — we care about qualitative landscape shape, not benchmark accuracy.
- **No normalization of energy** — reinforces that EBMs are not probabilities in this step.

---

## 4. Strategy

### 4.1 Architecture

```
x (1D) ──► x_encoder: MLP(1 → 64 → 128) ──► s_x ──┐
                                                   ├── concat ──► energy_head ──► F_w(x,y) ∈ ℝ
y (2D) ──► y_encoder: MLP(2 → 64 → 128) ──► s_y ──┘
```

Implemented in `experiments/step01_ebm/model.py`; MLP helper in `shared/modules.py`.

### 4.2 Training (naive)

| Hyperparameter | Value |
|----------------|--------|
| Loss | L = mean_batch F_w(x, y) |
| Optimizer | Adam, lr = 1e-3 |
| Epochs | 200 |
| Batch size | 256 |
| Seed | 42 |
| Device | MPS if available (`shared/device.py`), else CPU |

Logs → `outputs/loss_history.json`; weights → `outputs/checkpoint.pt`.

### 4.3 Hardware

On MacBook Pro M5 (24 GB unified memory), this step is tiny; `get_device()` selects MPS for consistency with later image/video steps. Matplotlib always plots from CPU arrays.

---

## 5. Visualizations

All figures are produced by `visualize.py` after loading `outputs/checkpoint.pt`. Default: **fix x** from training sample index 0; evaluate F on an **80×80** grid with y₁, y₂ ∈ [−3, 3]. Training data points are overlaid as **white scatter** (the Swiss roll in the plane).

---

### 5.1 `training_curve.png`

| | |
|--|--|
| **What is plotted** | **X:** epoch (1–200). **Y:** mean F(x, y) over all training pairs each epoch. Shaded band: ±1 standard deviation of per-sample energies within the epoch. |
| **How produced** | `train.py` writes `loss_history.json`; `visualize.py` calls `shared/viz.plot_training_curve()`. |
| **How to read it** | Downward curve = energies decreasing on positives. Steep dive = collapse accelerating. Widening band = energies spreading in scale (runaway magnitudes), not learning a sharper manifold structure. |
| **Expected in Step 1** | **Collapse:** curve plunges toward very large negative values (e.g. epoch 1 ≈ −31 → epoch 200 ≈ −2.56×10¹³ in our run). This is **not** “good convergence.” |
| **Paper link** | Motivates §4.2: without terms that **raise** energy on bad pairs, training can only push positives lower without bound. |

---

### 5.2 `energy_heatmap.png`

| | |
|--|--|
| **What is plotted** | **Axes:** y₁ (horizontal), y₂ (vertical). **Color:** F_w(x_fixed, y) at each grid cell. **Overlay:** all training y points (white dots) showing the Swiss roll. **Fixed:** one x from the dataset (default sample 0). |
| **How produced** | `energy_grid()` in `visualize.py`: replicate x_fixed across the grid, batch forward through `EBM`, reshape to 80×80, `plot_energy_heatmap()`. |
| **How to read it** | **Darker / “hotter” colors** in viridis = more negative energy in our run (color scale follows actual F values). You ask: *is there a **ridge or valley** that follows the white Swiss roll?* After Step 1, often **no** — large negative values appear broadly, not in a thin tube along the manifold. |
| **Expected in Step 1 (collapsed)** | Broad regions of similarly extreme negative energy; manifold not clearly singled out as the unique low-energy locus. |
| **Expected after Step 2 (contrastive)** | Valley **along** the roll, higher energy away from it — selective compatibility. |
| **Paper link** | Direct picture of the **energy landscape over output space** for fixed context x (§4.1). |

---

### 5.3 `energy_deviation_heatmap.png`

| | |
|--|--|
| **What is plotted** | Same grid and axes as `energy_heatmap.png`, but color encodes **F − mean(F)** over the grid (deviation from the grid’s mean energy). Same white Swiss roll overlay. |
| **How produced** | Same grid as heatmap; pass `energies - energies.mean()` to `plot_energy_heatmap()`. |
| **How to read it** | Removes the global offset when all F values are huge and negative. **Structure here** = relative compatibility: where is y slightly more/less preferred than average for this x? **Flat** map → model treats almost all y equally (relative collapse). **Structured** ridges along the roll → selective landscape. |
| **Expected in Step 1** | Often modest relative structure (coefficient of variation on grid may still be ~0.3+); the key lesson is comparing to Step 2, not achieving a perfect flat sheet. |
| **Paper link** | Clarifies “**constant** low energy everywhere” — collapse can mean flat *relative* landscape, not only identical absolute F. |

---

### 5.4 `energy_surface.png`

| | |
|--|--|
| **What is plotted** | **3D surface:** y₁, y₂ on the base plane; height = F_w(x_fixed, y). Same 80×80 grid as heatmaps. |
| **How produced** | `shared/viz.plot_energy_surface()` with the raw energy grid (not deviation). |
| **How to read it** | View angle shows whether energy forms a **channel** along the roll or a flat/degenerate bowl. Peaks and valleys in 3D match peaks and valleys in the heatmap. |
| **Expected in Step 1** | Often a dominated, skewed surface when F has run to extreme negatives — hard to interpret absolute height; use with heatmaps and deviation map. |
| **Paper link** | Same landscape as §4.1; alternative view for intuition about “terrain” over y. |

---

## 6. What we implemented

| File | Role |
|------|------|
| `experiments/step01_ebm/model.py` | `EBM` |
| `experiments/step01_ebm/train.py` | Naive training |
| `experiments/step01_ebm/visualize.py` | All figures in §5 |
| `shared/device.py` | MPS/CPU |
| `shared/data.py` | Swiss roll & circles |
| `shared/modules.py` | `MLP` |
| `shared/viz.py` | Plot helpers |

---

## 7. Results and evidence

### Training metrics (Swiss roll, seed 42, 200 epochs)

| Epoch | Mean F(x, y) | Std (batch) |
|-------|----------------|-------------|
| 1 | −30.93 | 45.56 |
| 200 | −2.56×10¹³ | 1.46×10¹³ |

See `outputs/loss_history.json` for the full series. Match against **`training_curve.png`**.

### Landscape metrics (console from `visualize.py`)

Typical after collapse: grid energies on the order of 10¹³ negative, with coefficient of variation (std / |mean|) often below ~0.4 — indicating limited **relative** structure vs the global plunge.

### How figures work together

1. **`training_curve.png`** — proves optimization is doing what we asked (minimize positive energy) without fixing the landscape.
2. **`energy_heatmap.png`** — asks whether that solution is **useful** (manifold-aligned valleys).
3. **`energy_deviation_heatmap.png`** — strips global offset to test “everything equally compatible.”
4. **`energy_surface.png`** — same story in 3D for geometric intuition.

---

## 8. What this establishes

You can now:

1. Define F_w(x, y) and compatibility in the paper’s terms.
2. Explain what x and y **mean** in the toy and in later pipelines.
3. Read each output plot and say what healthy vs collapsed behavior looks like.
4. Recognize collapse from **numbers and figures**, not from loss going down alone.
5. Justify contrastive training before representation learning.

Not yet established: discriminative landscapes (Step 2), embedding distance (Step 3), JEPA prediction (Step 5+).

---

## 9. Connection to the paper

- **§4.1:** EBM as scorer; landscapes over y; not a generative normalized model.
- **§4.2:** Table 1 losses exist because Step 1’s training signal is incomplete.
- **§4.3+:** Two encoders foreshadow s_x, s_y before we predict in representation space.
- **§3 (agent):** Scalar “cost” signals inherit the same collapse logic if everything is driven down without contrast.

| Misconception | Correction |
|---------------|------------|
| Lower loss = better model | Here, lower = worse *useful* EBM |
| EBM = sampler from exp(−F) | We only evaluate F |
| Skip contrastive training | Step 1 shows you cannot |

---

## 10. Limitations

- Positive-only loss; no negatives.
- 2D synthetic y only; no high-D contrastive sampling story yet.
- Unbounded F; collapse shows as −∞ drift, not only flat zero.
- Default viz uses one fixed x (`--x-index` to explore others).

---

## 11. Next step

**Step 2** (`step02_contrastive_ebm`): same `EBM`, add hinge / InfoNCE / logistic and negatives (random, hard, in-batch). Revisit **`energy_heatmap.png`** — you should see a valley tracking the white Swiss roll.

---

## Reproduce

```bash
cd /Users/gpmac/gpbuildspace/JEPA/jepa_2022
uv run python experiments/step01_ebm/train.py
uv run python experiments/step01_ebm/visualize.py
```

Optional: `--dataset circles`, `--x-index 500`.

After running, read this file alongside the four PNGs in order: training curve → heatmap → deviation heatmap → surface.
