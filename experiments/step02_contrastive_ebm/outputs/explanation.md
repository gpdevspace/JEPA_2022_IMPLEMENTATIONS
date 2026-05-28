# Step 2: Contrastive Training of EBMs

**Paper:** LeCun, *A Path Towards Autonomous Machine Intelligence* (2022) — §4.2, Appendix
**Curriculum:** `misc/JEPA_2022_PLAN_GPU.md` — Step 2
**Code:** `experiments/step02_contrastive_ebm/`

---

## 1. Paper anchor

Step 2 implements the contrastive training objective described in §4.2 and the appendix, where the EBM is taught not only to assign low energy to matching pairs `(x, y)` but also to assign higher energy to incorrect or contrastive samples `ŷ`.

The core idea is: *push down energy on positives, and push up energy on negatives.* This is the first training objective in the roadmap that prevents the collapse observed in Step 1.

## 2. Problem we solved

This experiment answers the question: *How do we make an energy-based model discriminate correct outputs from incorrect ones?* Step 1 showed that minimizing only `F_w(x, y)` drives the model to collapse. Step 2 fixes that by adding an explicit repulsion term for contrastive samples.

The main failure mode we demonstrate is the difference between:

- naive positive-only energy minimization, which collapses the landscape,
- contrastive training, which produces a selective low-energy valley around the true manifold.

## 3. Data

### 3.1 What the variables represent

- `x` is a scalar context variable in `[0, 1]` representing a position along the Swiss roll.
- `y` is the corresponding 2D point on the Swiss roll manifold in `ℝ^2`.
- `x` is a partial observation / conditioning input; `y` is the output to score.

### 3.2 How the data is built

| Item | Value |
|------|-------|
| Dataset | `swiss_roll` |
| x shape / meaning | `(N, 1)` normalized arc-length index along the roll |
| y shape / meaning | `(N, 2)` noisy 2D points on the roll |
| Samples | `10,000` |
| Positives / negatives | positive pairs are matched `(x, y)`; negative pairs are sampled randomly from the dataset |

The Swiss roll is generated with `shared/data.py:swiss_roll_pairs()`. `y` is computed as `t * cos(t), t * sin(t)` plus Gaussian noise (`σ=0.05`), and `x` is the normalized rank of `t` in `[0, 1]`.

### 3.3 Why this dataset was chosen

- The output space is low-dimensional and easily visualized.
- The Swiss roll has a curved manifold structure, so a correct EBM should place low energy along that curve and higher energy off the manifold.
- This makes the effect of contrastive training visible in 2D energy plots.

### 3.4 Analogy to the full paper setting

This toy maps to later JEPA settings by treating `x` as a conditioning observation and `y` as a candidate output. In higher steps, `x` and `y` become image or video representations instead of 1D/2D toy data, but the same contrastive training idea applies.

### 3.5 What we excluded (and why)

- No hard negative mining in the default run; we use random negatives first to keep the experiment simple.
- No high-dimensional image data, because the goal here is to validate the contrastive objective itself.
- No in-batch negative sampling in the default output, though the code supports it.

## 4. Strategy

### Architecture

- `experiments/step02_contrastive_ebm/model.py` defines `EBM`.
- `EBM` uses two MLP encoders: `x_encoder` for `x` and `y_encoder` for `y`.
- The encoded vectors are concatenated and passed through an energy head to produce a scalar energy.

### Training

| Hyperparameter | Value |
|----------------|-------|
| Loss | hinge contrastive loss: `max(0, F(x, y) - F(x, ŷ) + margin)` |
| Optimizer | Adam |
| Epochs | 120 |
| Batch size | 256 |
| Device | `mps` when available |
| Loss type | `hinge` |
| Negative strategy | `random` |

Training procedure in `experiments/step02_contrastive_ebm/train.py`:

1. Load the Swiss roll dataset.
2. For each batch, sample random negatives from the dataset.
3. Compute positive energy and negative energy.
4. Minimize the hinge contrastive loss.
5. Save checkpoint and `loss_history.json`.

## 5. Visualizations

### 5.1 `energy_heatmap.png`

- **What is plotted:** Energy values `F(x, y)` over a 2D grid of candidate `y` values for a fixed training `x`.
- **How produced:** `experiments/step02_contrastive_ebm/visualize.py` evaluates the trained model on a grid of `y` values and saves the heatmap.
- **How to read it:** Darker regions indicate lower energy. The true Swiss roll manifold should appear as a low-energy valley.
- **Expected in this step:** a visible low-energy path near the data manifold, instead of a flat collapsed surface.
- **Paper link:** contrastive training shapes the energy landscape to prefer correct pairs and repel incorrect pairs.

### 5.2 `energy_surface.png`

- **What is plotted:** A 3D surface of the same energy landscape over `y`.
- **How produced:** `visualize.py` renders the grid as a surface plot.
- **How to read it:** The surface height is energy; valleys correspond to high compatibility.
- **Expected in this step:** a distinct valley around the true data manifold with higher energy elsewhere.
- **Paper link:** this surface visualization makes the contrastive objective explicit in geometric terms.

### 5.3 `training_curve.png`

- **What is plotted:** Epoch-wise mean contrastive loss with ±1 std deviation across batches.
- **How produced:** `visualize.py` reads `loss_history.json` and plots the curve.
- **How to read it:** the mean loss measures how well positives are separated from negatives on average.
- **Expected in this step:** loss should fall and stabilize, indicating successful contrastive training.
- **Paper link:** the learning curve shows that the objective is being optimized, unlike the collapse in Step 1.

## 6. What we implemented

| File | Role |
|------|------|
| `experiments/step02_contrastive_ebm/model.py` | `EBM` model with dual encoders and energy head |
| `experiments/step02_contrastive_ebm/train.py` | Contrastive training loop, checkpointing, and history logging |
| `experiments/step02_contrastive_ebm/visualize.py` | Energy landscape and training curve visualizations |
| `shared/device.py` | MPS/CPU device selection |
| `shared/data.py` | Swiss roll and concentric circles generator |
| `shared/modules.py` | Reusable MLP builder |
| `shared/viz.py` | Plot helpers for heatmaps, surfaces, and curves |

## 7. Results and evidence

From the final run:

- `epoch 0` mean loss: `0.6383`, std loss: `0.2691`
- `epoch 119` mean loss: `0.0325`, std loss: `0.0149`
- Final loss dropped by ~95% from the first epoch.
- The training curve is smooth and stable after the first 20 epochs.

These numbers indicate that the contrastive objective is successfully learning a discriminative energy function rather than collapsing.

## 8. What this establishes

- Contrastive loss prevents the collapse seen in Step 1.
- The same EBM architecture now learns an energy surface that distinguishes correct `y` values from random negatives.
- The model can be trained with random negative sampling as a first contrastive baseline.
- This step establishes the need for contrastive supervision before moving to joint embedding architectures.

## 9. Connection to the paper

Step 2 corresponds to §4.2 and the contrastive loss table in the appendix. The code implements one of the paper’s proposed training objectives and shows how it changes the energy landscape from an indiscriminate collapse to a structured compatibility function.

## 10. Limitations of this toy

- Only random negatives are used by default; hard negatives and in-batch negatives are not explored here.
- The output space is 2D, so this does not yet demonstrate the scaling issues of contrastive training in higher-dimensional representations.
- The model still uses a toy `x` and `y` pair rather than image or video embeddings.

## 11. Next step

Step 3 will move from raw pair scoring to a joint embedding architecture, where the energy is defined as a distance between learned representations. That step will show how contrastive training can operate in embedding space and how it generalizes to richer inputs.

## Reproduce

From the repository root:

```bash
uv run python experiments/step02_contrastive_ebm/train.py --epochs 120 --batch-size 256 --loss-type hinge --negative-strategy random
uv run python experiments/step02_contrastive_ebm/visualize.py
```
