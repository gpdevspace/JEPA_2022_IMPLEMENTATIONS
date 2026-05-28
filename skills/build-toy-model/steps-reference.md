# JEPA 2022 toy steps — reference

Condensed from `misc/JEPA_2022_PLAN_GPU.md`. Read the full section in the plan before implementing.

## Step 1 — EBM

- **Data:** Swiss roll or concentric circles; x = 1D parameter, y ∈ R²
- **Model:** `EBM`: encoders x→128, y→128; energy head on concat → scalar
- **Train:** L = F(x,y) only (positives) → **collapse** (constant low energy)
- **Viz:** 2D energy heatmap for fixed x; training curve; 3D surface
- **Intuition:** Without pushing up wrong y, everything becomes compatible

## Step 2 — Contrastive EBM

- Same `EBM`; add hinge, InfoNCE, logistic losses
- Negatives: random, hard, in-batch
- **Intuition:** Contrastive needs many negatives in high-D y

## Step 3 — JEA

- CIFAR-10 augmented pairs; ResNet-18-ish CNN; F = ||s_x - s_y||²
- InfoNCE + in-batch negatives; t-SNE, collapse monitor

## Step 4 — Non-contrastive

- Barlow Twins + variance/covariance; no negatives
- Ablations: remove decorrelation or variance → collapse

## Step 5 — JEPA

- Moving MNIST; predict s_y from s_x in embedding space
- Compare to generative pixel predictor

## Step 6 — VICReg JEPA

- Expander heads; invariance + variance + covariance terms
- Ablations per term

## Step 7 — Latent-variable JEPA

- Forking 2D paths; z with dim bottleneck / discrete / L1
- Without z regularization → cheat/collapse

## Step 8 — H-JEPA

- Two levels; short vs long horizon targets; Level 2 abstracts fast detail

## Step 9 — World model + planning

- MiniGrid pixels; differentiable rollout; Mode-1 vs Mode-2

## Step 10 — Full agent

- Intrinsic cost (fixed), critic, memory, configurator; visual multi-goal env
