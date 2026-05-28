---
name: build-toy-model
description: >-
  Implements a single step from misc/JEPA_2022_PLAN_GPU.md as a runnable
  PyTorch toy on Apple Silicon (MPS). Use this skill to plan, implement,
  verify, and document one curriculum step end-to-end (code + `outputs/explanation.md`).
scope: workspace
disable-model-invocation: true
inputs:
  - name: step (string)
    description: "Short step identifier or number, e.g. 'step01_ebm' or 'Step 1'"
outputs:
  - name: explanation.md
    description: "Required written artifact under experiments/step*/outputs/"
---

# Build Toy Model (JEPA 2022 Curriculum)

This skill encapsulates the repeatable workflow to turn a single curriculum step
from `misc/JEPA_2022_PLAN_GPU.md` into a verified, documented experiment.

Outcome: runnable experiment directory under `experiments/stepNN_<name>/` with
training, visualization, checkpoint, and a completed `outputs/explanation.md`.

## Target hardware (required)

- **Machine:** MacBook Pro, Apple M5, **24 GB unified memory**
- **Backend:** PyTorch **MPS** (Metal Performance Shaders), not CUDA
- **Batching:** Prefer moderate batches (64–512 for 2D toys); 24 GB allows larger batches later
- **Fallback:** If MPS unavailable, use `cpu` and warn the user

Standard device helper (create in `shared/device.py` on first step, reuse after):

```python
import torch

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

### MPS rules for every step

1. **Create tensors on device:** `x = x.to(device)` after generation on CPU; or generate directly on device when cheap.
2. **Matplotlib:** Move results to CPU before plotting: `.detach().cpu().numpy()`.
3. **Avoid unsupported ops:** If an op fails on MPS, fix with a CPU fallback for that op only (document in a one-line comment), or use `float64` on CPU for that tensor — do not switch the whole run to CPU silently.
4. **Reproducibility:** Set seeds for `torch`, `numpy`; note MPS nondeterminism for some ops.
5. **Memory:** Call `torch.mps.empty_cache()` between heavy visualization passes if RSS grows during long runs.

## How the skill runs (agent steps)

1. Gather Graph Context — locate related nodes in `graphify-out/` and `misc/JEPA_2022_PLAN_GPU.md`.
2. Draft Implementation Plan — short, reviewable plan with files, model, data, and verification criteria.
3. Wait for user approval of the plan (ask to continue).
4. Implement files (`model.py`, `train.py`, `visualize.py`) following conventions.
5. Run training & visualization; produce `loss_history.json` and figures in `outputs/`.
6. Write `outputs/explanation.md` per template and commit artifacts.

The rest of this document expands the required checks and content for each phase.

## Mandatory workflow (do not skip)

### Phase A — Graph context (before any code)

1. Confirm `graphify-out/graph.json` exists in the repo root.
2. Run graphify queries (CLI or MCP) for the step topic, e.g.:
   - `graphify query "step N <concept> existing implementation shared modules"`
   - `graphify explain "<relevant node from plan>"`
3. Summarize for the user:
   - **Reuse:** existing modules/patterns to extend
   - **Greenfield:** what the graph shows is plan-only (pseudocode in markdown)
   - **Align:** naming and layout from `misc/JEPA_2022_PLAN_GPU.md` Project Structure

If the graph has no implementation nodes for the step, state that clearly — the repo is often plan-only until the step is built.

### Phase B — Implementation plan (wait for approval)

Read the step section in `misc/JEPA_2022_PLAN_GPU.md` (title, dataset, architecture, training, visualizations, key intuition).

Produce a short plan with:

| Section | Content |
|--------|---------|
| Step & paper ref | e.g. Step 1, §4.1 |
| Files to create | Under `experiments/stepNN_<name>/` and `shared/` |
| Reuse from graph/code | Or "none — first implementation" |
| Model & loss | Shapes, forward pass, loss (including intentional failure modes) |
| Data | Generator/dataset, MPS batching |
| Train script | Epochs, optimizer, logging |
| Viz | Plots that prove the paper intuition |
| Verify | What "success" and "expected failure" look like |
| Run commands | `uv run python ...` from repo root |

**Stop and ask the user to review the plan.** Do not write implementation code until they approve (unless they explicitly say "skip review" or "implement now").

Decision points / branching:
- If graph shows existing implementation: propose reuse and minimal edits.
- If graph is plan-only: propose full new experiment skeleton.
- If required dependencies are missing in `pyproject.toml`: ask user before adding new packages.

### Phase C — Implement

Layout (from plan):

```
experiments/stepNN_<short_name>/
  model.py
  train.py
  visualize.py
  outputs/
    explanation.md   # required — see Phase E
shared/
  device.py      # get_device()
  data.py        # datasets used by multiple steps
  viz.py         # plotting helpers
  modules.py     # MLP blocks, encoders reused across steps
```

Conventions:

- **Imports:** package-relative or add repo root to `PYTHONPATH` via `uv run` from project root.
- **Dependencies:** already in `pyproject.toml`; do not add packages unless the step requires it and user agrees.
- **Training scripts:** print `device` at start; use `tqdm` for epochs; save checkpoints to `experiments/stepNN_*/outputs/`.
- **Naive / collapse demos:** when the plan says training "will fail", implement that path first, plot it, then optional commented path for contrastive follow-up (Step 2).

Implementation checklist (quick):
- `device` printed at start of `train.py`.
- Checkpoint saved to `outputs/checkpoint.pt`.
- `loss_history.json` written with per-epoch metrics.
- `visualize.py` reads checkpoint and writes labeled figure files into `outputs/`.

### Phase D — Verify

1. `uv run python experiments/stepNN_*/train.py` completes without NaN.
2. `uv run python experiments/stepNN_*/visualize.py` writes figures to `outputs/`.
3. Check **key intuition** bullet from the plan (e.g. energy collapse for Step 1).
4. Optionally update graph: `graphify --update .` after code lands.

### Phase E — Write `outputs/explanation.md` (required)

After training and visualization succeed, write **`experiments/stepNN_*/outputs/explanation.md`**. This is a durable learning artifact — not a README stub. Target **detailed prose** (roughly 800–2000 words unless the step is trivial).

Use the outline in [explanation-template.md](explanation-template.md). Every section must be filled with step-specific content (numbers, file names, observed metrics from `loss_history.json`, what the plots show).

| Section | What to cover |
|--------|----------------|
| **Paper anchor** | Section number, quoted idea from LeCun (2022), how this step fits the roadmap |
| **Problem we solved** | What question or gap this toy answers; what would be unclear without running it |
| **Data** | See **Data section requirements** below — not just a table of shapes |
| **Strategy** | Architecture, loss, optimizer, hyperparameters, training procedure, MPS notes |
| **Visualizations** | See **Visualization section requirements** below — one subsection **per plot file** |
| **What we implemented** | File map (`model.py`, `train.py`, …) and how pieces connect |
| **Results & evidence** | Training curves, collapse/success criteria; tie numbers to the plots |
| **What this establishes** | Concrete capabilities proven before moving to the next step |
| **Connection to the paper** | How hands-on work maps to §X; misconceptions this step clears up |
| **Limitations** | What this toy does *not* show (saved for later steps) |
| **Next step** | One paragraph bridging to Step N+1 in the curriculum |

#### Data section requirements (mandatory detail)

The **Data** section must go beyond shapes and sample counts. Include:

1. **What x and y represent semantically** — e.g. “partial observation / index” vs “outcome in output space”, not only tensor shapes.
2. **How the data is generated** — equations, sampling process, noise, train/val split if any.
3. **Why this dataset was chosen for this step** — what property of the data makes the paper’s phenomenon visible (low-D plot, multimodality, temporal structure, etc.).
4. **Analogy to later paper settings** — how this toy data stands in for images, video frames, actions, or states in later steps.
5. **What we did *not* include** — e.g. no negatives in Step 1, and why that matters for the experiment.

#### Visualization section requirements (mandatory detail)

Add a dedicated **## Visualizations** section (separate from Strategy, or expand Strategy with a full subsection). For **every** figure written to `outputs/`:

| Per plot | Required content |
|----------|------------------|
| **Filename** | e.g. `energy_heatmap.png` |
| **What is plotted** | Axes, variables held fixed, grid/range, overlays |
| **How it was produced** | Script, function, checkpoint dependency |
| **How to read it** | What low/high/colors mean in terms of F_w or loss |
| **Healthy vs collapsed / success vs failure** | What you should see after a *good* step vs what Step 1 *expects* |
| **Link to paper intuition** | Which sentence or figure in the paper this illuminates |

Do not list filenames in one bullet line — give each plot its own `###` heading so the doc works as a standalone figure guide.

Commit `explanation.md` with the step (figures in `outputs/` may stay gitignored if large; the markdown should still describe them).

**Do not mark a step complete without `outputs/explanation.md`.**

## Implementation acceptance criteria

- Train script runs end-to-end without NaNs and with reproducible seed intent.
- Visualizations are present and documented with per-figure subsections.
- `loss_history.json` contains epoch-level numbers referenced in `explanation.md`.
- `outputs/explanation.md` follows the provided template and includes concrete evidence.

## Step index (quick lookup)

| Step | Dir | Focus |
|------|-----|--------|
| 1 | `step01_ebm` | EBM F(x,y), naive loss, collapse, energy landscape |
| 2 | `step02_contrastive_ebm` | Hinge, InfoNCE, logistic; negatives |
| 3 | `step03_joint_embedding` | Siamese JEA, CIFAR-10 |
| 4 | `step04_non_contrastive` | Barlow Twins, variance/covariance |
| 5 | `step05_jepa` | Predict in embedding space, Moving MNIST |
| 6 | `step06_vicreg` | VICReg on JEPA |
| 7 | `step07_latent_variable_jepa` | Latent z, forking paths |
| 8 | `step08_hierarchical_jepa` | Two-level H-JEPA |
| 9 | `step09_world_model_planning` | World model + planning, MiniGrid |
| 10 | `step10_full_agent` | IC, critic, configurator, memory |

Full specs: `misc/JEPA_2022_PLAN_GPU.md`.

## Example user prompts

- "Implement Step 1 using build-toy-model"
- "Build the Step 5 JEPA toy — plan first"
- "Step 2 contrastive EBM, MPS, follow the skill"

Suggested prompts for this skill (examples to include in the template):
- "Plan Step 1 using build-toy-model (show files and checks only)"
- "Implement Step 1 now — skip review and run training on MPS"
- "Draft explanation.md for step01_ebm using outputs in experiments/step01_ebm/outputs/"

## Additional reference

- Step-specific pitfalls and ablations: [steps-reference.md](steps-reference.md)
- Explanation outline: [explanation-template.md](explanation-template.md)
