# Building Autonomous Machine Intelligence from Scratch

## 🎯 Goal

Implement every core concept from Yann LeCun's *"A Path Towards Autonomous Machine Intelligence"* (2022) as a progression of **10 self-contained toy models**. Each step introduces one key idea, builds on the previous step, and includes visualizations so you can *see* the concept working.

By the end, you'll have a working **miniature autonomous agent** with a learned world model, hierarchical planning, and intrinsic motivation — the full architecture from the paper.

---

## Overview: The 10 Steps

| Step | Concept | Paper Section | Estimated Time |
|------|---------|---------------|----------------|
| 1 | Energy-Based Models (EBMs) | §4.1 | 2–3 hours |
| 2 | Contrastive Training of EBMs | §4.2, Appendix | 2–3 hours |
| 3 | Joint Embedding Architecture (JEA) | §4.3 | 2–3 hours |
| 4 | Non-Contrastive Training (collapse prevention) | §4.3 | 3–4 hours |
| 5 | JEPA — Joint Embedding Predictive Architecture | §4.4 | 3–4 hours |
| 6 | VICReg Loss for JEPA | §4.5.1 | 3–4 hours |
| 7 | Latent-Variable JEPA | §4.4, §4.5 | 4–5 hours |
| 8 | Hierarchical JEPA (H-JEPA) | §4.6 | 4–5 hours |
| 9 | World Model + Actor + Planning (Mode-2) | §3, §3.1 | 5–6 hours |
| 10 | Full Agent: Cost Module + Critic + Configurator | §3.2, §3.3 | 5–6 hours |

**Total: ~35–45 hours of focused work**

---

## Prerequisites & Setup

### Dependencies to add to `pyproject.toml`

```toml
dependencies = [
    "torch>=2.0",
    "torchvision>=0.15",
    "matplotlib>=3.7",
    "numpy>=1.24",
    "gymnasium>=0.29",       # for Step 9-10 environments
    "minigrid>=2.3.1",       # for visual RL environments
    "tqdm>=4.65",
    "scikit-learn>=1.3",     # for t-SNE / evaluation
]
```

### Hardware Acceleration (Apple Silicon)
Your **M5 Pro with 24GB Unified Memory** is highly capable. We will utilize the **Metal Performance Shaders (MPS)** backend in PyTorch to maximally leverage your GPU. 

```python
# Standard device configuration for all scripts:
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
```
Because of the 24GB unified memory, we can easily handle batches of images and sequences of frames without traditional VRAM bottlenecks.

### Project Structure

```
jepa_2022/
├── papers/                          # existing
├── experiments/
│   ├── step01_ebm/
│   │   ├── model.py
│   │   ├── train.py
│   │   └── visualize.py
│   ├── step02_contrastive_ebm/
│   ├── step03_joint_embedding/
│   ├── step04_non_contrastive/
│   ├── step05_jepa/
│   ├── step06_vicreg/
│   ├── step07_latent_variable_jepa/
│   ├── step08_hierarchical_jepa/
│   ├── step09_world_model_planning/
│   └── step10_full_agent/
├── shared/
│   ├── data.py              # shared data generation utilities
│   ├── viz.py               # shared visualization utilities
│   └── modules.py           # reusable neural net building blocks
├── pyproject.toml
└── main.py
```

---

## Step 1: Energy-Based Models (EBM) — The Foundation

> **Paper concept (§4.1):** *"An EBM is a trainable system that, given two inputs x and y, produces a scalar energy F_w(x, y) that measures the degree of incompatibility between x and y."*

### What you'll learn
- An EBM assigns a scalar **energy** to every (x, y) pair
- Low energy = compatible; high energy = incompatible
- The energy surface defines a "landscape" over the output space
- **This is NOT a probability model** — there is no normalization requirement

### Dataset
- **2D Swiss Roll** or **concentric circles**: generate `(x, y)` pairs where `x` is a 1D index/parameter and `y ∈ R²` is the corresponding 2D point
- This lets you visualize the energy surface as a heatmap over 2D space

### Architecture

**`model.py`**:
```python
class EBM(nn.Module):
    """
    F_w(x, y) → scalar energy
    - x_encoder: MLP(x_dim → 64 → 128)
    - y_encoder: MLP(2 → 64 → 128)
    - energy_head: MLP(256 → 128 → 64 → 1)
    Energy = energy_head(concat(x_enc, y_enc))
    """
```

### Training (naive — will fail!)
- First, try training with **only the positive term**: `L = F_w(x, y)` for matching pairs
- **Observe the collapse**: energy goes to a constant (negative infinity or zero) for ALL inputs
- This motivates Step 2

### Visualizations
1. **Energy landscape heatmap**: For a fixed x, plot `F_w(x, y)` over a 2D grid of y values
2. **Training curve**: energy over epochs — watch it collapse
3. **3D surface plot** of the energy landscape

### Key intuition to verify
> *Without a mechanism to push energies UP for incorrect y values, the model learns to make everything low energy — this is the "collapse" problem that drives the entire paper.*

---

## Step 2: Contrastive Training of EBMs

> **Paper concept (§4.2, Appendix Table 1):** *"Contrastive methods consist in constructing a loss functional whose minimization has the effect of pushing down the energies of training samples, and pulling up the energies of suitably-selected 'contrastive samples'."*

### What you'll learn
- Contrastive loss has **two terms**: push down on correct (x, y) and push up on incorrect (x, ŷ)
- Different contrastive losses: **pairwise hinge**, **InfoNCE**, **logistic**
- The difficulty of "hard negative mining"

### Changes from Step 1
- Same EBM architecture
- Add contrastive loss functions:

```python
# Pairwise Hinge (Row 5 in paper's Table 1)
L_hinge = max(0, F(x,y) - F(x,ŷ) + margin)

# InfoNCE / Softmax (Row 1 in paper's Table 1)
L_nce = F(x,y) + log Σ_ŷ exp(-F(x,ŷ))

# Logistic (Row 8)
L_logistic = log(1 + exp(F(x,y) - F(x,ŷ)))
```

### Negative sampling strategies
1. **Random negatives**: sample ŷ randomly from the dataset
2. **Hard negatives**: sample ŷ that currently has low energy (most "threatening")
3. **In-batch negatives**: use other samples in the batch as negatives

### Visualizations
1. **Energy landscape** after contrastive training — should show clear low-energy valleys near data
2. **Compare** the three loss functions: which produces the sharpest energy landscape?
3. **Negative sample efficiency**: how many negatives are needed for good training?

### Key intuition to verify
> *Contrastive methods work but scale poorly: as the dimensionality of y grows, you need exponentially more contrastive samples. This motivates non-contrastive methods in Step 4.*

---

## Step 3: Joint Embedding Architecture (JEA)

> **Paper concept (§4.3):** *"Joint Embedding Architectures... map x and y to representations s_x and s_y in the same embedding space, and define the energy as a distance between the embeddings."*

### What you'll learn
- Instead of measuring compatibility in input space, map both inputs to a **shared embedding space**
- Energy = distance between embeddings: `F(x,y) = D(s_x, s_y)`
- This is the Siamese network / twin network idea
- Collapse is even MORE dangerous here (all embeddings can converge to a single point)

### Dataset
- **CIFAR-10**: Create pairs (x, y) where y is a transformed version of x (random crop, horizontal flip, color jitter, grayscale). 
- Positive pair: (image, augmented version of same image)
- Negative pair: (image, different image from the batch)
- *Why upgrade from 2D?* Joint embeddings shine when extracting abstract representations from high-dimensional data. Your M5 Pro can easily handle CIFAR-10 CNN training.

### Architecture

**`model.py`**:
```python
class JointEmbeddingArch(nn.Module):
    """
    Siamese / Joint Embedding Architecture
    - encoder_x: ResNet-18 (modified for 32x32) or simple CNN → R^d
    - encoder_y: Shared weights with encoder_x
    - Energy F(x,y) = ||s_x - s_y||²
    """
```

### Training
- Train with **contrastive loss** (InfoNCE) using in-batch negatives
- Show that it works but requires many negatives

### Visualizations
1. **t-SNE** of the embedding space — same-class points should cluster
2. **Embedding collapse monitor**: track the standard deviation of embeddings across a batch
3. **Nearest neighbor retrieval**: given a query, find the closest embeddings

### Key intuition to verify
> *Joint embeddings are powerful but fragile. If you remove the contrastive term, all embeddings collapse to a single point. The next step shows how to prevent this WITHOUT contrastive samples.*

---

## Step 4: Non-Contrastive Training (Preventing Collapse)

> **Paper concept (§4.3):** *"Non-contrastive methods use a regularizer term R_w(y) whose minimization has the effect of preventing the energy surface from becoming 'too flat' near training samples."*

### What you'll learn
- You can train joint embeddings **without any negative samples**
- Three mechanisms to prevent collapse: **Barlow Twins**, **Whitening**, and **VICReg** (preview)
- The key insight: instead of contrasting **samples**, contrast **dimensions** (decorrelation)

### Methods to implement

```python
# Method 1: Barlow Twins
# Cross-correlation matrix between s_x and s_y over a batch
C = (s_x_norm.T @ s_y_norm) / batch_size
# Loss: make C close to identity
L = (diagonal_terms - 1)² + λ * off_diagonal_terms²

# Method 2: Simple variance + covariance regularization
# Variance: keep std of each dimension above threshold
L_var = mean(max(0, threshold - std(s, dim=batch)))
# Covariance: decorrelate dimensions
cov = (s - s.mean(0)).T @ (s - s.mean(0)) / (batch_size - 1)
L_cov = (off_diagonal(cov) ** 2).sum() / dim
```

### Experiments
1. Train JEA from Step 3 with Barlow Twins loss (**no negatives!**)
2. Compare embedding quality with contrastive method from Step 3
3. **Ablation**: remove the decorrelation term → watch collapse happen
4. **Ablation**: remove the variance term → watch collapse happen differently

### Visualizations
1. **Cross-correlation matrix** of embeddings: should approach identity
2. **Embedding dimension histograms**: each dimension should have non-trivial variance
3. **Collapse detector**: track the rank of the embedding matrix over training

### Key intuition to verify
> *Non-contrastive methods work by ensuring the embedding space is well-utilized — every dimension carries information, and dimensions are independent. This is "dimension-contrastive" vs "sample-contrastive".*

---

## Step 5: JEPA — Joint Embedding Predictive Architecture

> **Paper concept (§4.4):** *"A JEPA does not reconstruct y from s_x directly, but predicts the representation s_y from s_x... The predictor may use a latent variable z to represent the information necessary to predict s_y that is not present in s_x."*

### What you'll learn
- The **core innovation**: predict in **representation space**, not in input space
- JEPA ≠ autoencoder: you never reconstruct the input
- The **predictor** maps `s_x → ŝ_y` (predicted representation of y)
- Energy = distance between predicted and actual representation: `D(s_y, ŝ_y)`
- This allows the model to **ignore unpredictable details**

### Dataset
- **Moving MNIST**: A standard dataset for video prediction.
- x = past frames (e.g., frames at t-3, t-2, t-1, t)
- y = future frame (at t+1)
- *Why upgrade?* JEPA's primary motivation is video prediction where predicting every pixel (generative) is wasteful compared to predicting abstract representations. 

### Architecture (this is the key figure from the paper!)

**`model.py`**:
```python
class JEPA(nn.Module):
    """
    Joint-Embedding Predictive Architecture (Figure 12 in paper)

    Components:
    - encoder_x: 3D CNN or Spatio-temporal CNN (takes sequence of frames) → s_x (dim d)
    - encoder_y: 2D CNN (takes single future frame) → s_y (dim d)
    - predictor: MLP or small Transformer (d → hidden → d) → predicts ŝ_y from s_x

    Energy: F(x,y) = ||s_y - predictor(s_x)||²

    Training: minimize energy for correct pairs while preventing collapse
    """
```

### Comparison: JEPA vs Generative Model
- Also implement a simple **generative** (autoencoder) model that predicts y directly in input space
- Compare: which handles noisy/unpredictable details better?

### Visualizations
1. **Representation space**: plot s_x and s_y embeddings — predicted ŝ_y should be close to s_y
2. **Prediction errors**: in input space (generative) vs. in representation space (JEPA)
3. **What gets ignored**: show that JEPA embeddings discard unpredictable noise while the generative model struggles

### Key intuition to verify
> *By predicting in representation space, JEPA can learn to IGNORE irrelevant, unpredictable details (like texture, noise). A generative model must predict every detail, making it fragile.*

---

## Step 6: VICReg Loss for JEPA Training

> **Paper concept (§4.5.1):** *"VICReg ensures that different components of representations over a batch are different. VICReg is contrastive over components, while traditional contrastive methods are contrastive over vectors."*

### What you'll learn
- **V**ariance-**I**nvariance-**C**ovariance **Reg**ularization
- The three loss terms and their roles:
  - **Invariance**: make ŝ_y close to s_y (the prediction)
  - **Variance**: keep each embedding dimension alive (prevent collapse to constant)
  - **Covariance**: decorrelate dimensions (prevent collapse to low-rank)
- The **expander** head trick: map to a higher-dimensional space for the loss computation

### Architecture

**`model.py`**:
```python
class JEPA_VICReg(nn.Module):
    """
    JEPA + VICReg training (Figure 14 in paper)

    Same as Step 5 JEPA, plus:
    - expander_x: MLP(d → 256 → 256) (maps s_x to higher dim for loss)
    - expander_y: MLP(d → 256 → 256) (maps s_y to higher dim for loss)

    VICReg Loss:
      L = λ * L_invariance + μ * L_variance + ν * L_covariance

      L_invariance = ||ŝ_y - s_y||² (MSE)
      L_variance   = Σ_j max(0, γ - sqrt(Var(v_j) + ε)) (per-dim std hinge)
      L_covariance = (1/d) Σ_{i≠j} Cov(v_i, v_j)² (off-diag cov)
    """
```

### Experiments
1. Train JEPA with VICReg loss — **no contrastive samples needed!**
2. **Ablation study**: remove each VICReg term one at a time
   - Remove variance → embeddings collapse to constant
   - Remove covariance → embeddings collapse to low-rank
   - Remove invariance → embeddings become unpredictive
3. **Hyperparameter sensitivity**: vary λ, μ, ν weights

### Visualizations
1. **VICReg loss components** over training: three separate curves
2. **Embedding covariance matrix**: should approach diagonal
3. **Effective dimensionality**: track the rank of the covariance matrix
4. **Per-dimension variance histogram**: all dims should be active

### Key intuition to verify
> *VICReg replaces the need for contrastive samples entirely. Instead of ensuring different SAMPLES produce different embeddings, it ensures different DIMENSIONS of the embedding carry different information. This scales much better.*

---

## Step 7: Latent-Variable JEPA

> **Paper concept (§4.4):** *"The predictor may depend on a latent variable z that represents the information necessary to predict s_y that is not present in s_x."*

### What you'll learn
- When the future is **multimodal** (multiple valid outcomes), a deterministic predictor fails
- The **latent variable z** captures the "missing information" that determines which outcome occurs
- **Critical**: z must have **limited information content** (sparse, low-dim, or discrete) to prevent collapse
- Without regularization on z, the model can "cheat" by encoding ALL of y into z

### Dataset
- **Forking paths**: 2D trajectories that reach a fork point and can go left OR right
- x = trajectory before the fork; y = trajectory after (either branch)
- The latent z should capture WHICH branch was taken

### Architecture

**`model.py`**:
```python
class LatentVariableJEPA(nn.Module):
    """
    JEPA with latent variable z (Figure 12 in paper)

    - encoder_x: input x → s_x
    - encoder_y: input y → s_y (target)
    - latent_encoder: (s_x, s_y) → z [only used during training for inference]
    - predictor: (s_x, z) → ŝ_y

    Energy: F(x,y,z) = D(s_y, predictor(s_x, z)) + R(z)

    Inference: ž = argmin_z F(x,y,z)
    """
```

### Regularization options for z (implement all three!)

```python
# Option A: Low-dimensional z (e.g., dim=2 when s_y dim=128)
# — limits information by dimensionality bottleneck

# Option B: Discrete z (e.g., categorical with K=4 options)
# — uses Gumbel-Softmax for differentiable discrete sampling

# Option C: Sparse z with L1 regularization
# R(z) = α * ||z||_1
```

### Experiments
1. Train on forking-path data with each regularization option
2. **Ablation**: train WITHOUT z regularization → observe collapse (energy goes to 0 everywhere)
3. Visualize the latent space: does z capture the fork choice?

### Visualizations
1. **Latent space**: color-code z by which fork branch was taken — should separate cleanly
2. **Multi-modal predictions**: for a fixed x, sample different z values → should produce predictions on different branches
3. **Energy landscape over z**: for a fixed (x, y), plot energy as a function of z — should show a clear minimum

### Key intuition to verify
> *The latent variable z captures the "residual uncertainty" that x alone cannot resolve. But z MUST be information-bottlenecked, otherwise the model collapses. The fork-choice experiment makes this crystal clear.*

---

## Step 8: Hierarchical JEPA (H-JEPA)

> **Paper concept (§4.6):** *"JEPA-1 extracts low-level representations and performs short-term predictions. JEPA-2 takes the representations extracted by JEPA-1 as inputs and extracts higher-level representations with which longer-term predictions can be performed."*

### What you'll learn
- Stack JEPAs hierarchically: each level operates at a higher abstraction
- **Lower levels**: detailed representations, short-term predictions
- **Higher levels**: abstract representations, long-term predictions
- Higher levels learn to **ignore details** that don't matter for long-term prediction

### Dataset
- **Long-horizon Video (e.g., bouncing balls with physics)**:
  - **Fast component**: The exact pixel-level rotation or texture of the balls.
  - **Slow component**: The overall macro trajectory and collision events.
- Level 1 JEPA should capture short-term dynamics; Level 2 should capture long-term macro events.

### Architecture

**`model.py`**:
```python
class HierarchicalJEPA(nn.Module):
    """
    H-JEPA: Two-level hierarchical JEPA (Figure 15 in paper)

    Level 1 (fine-grained, short-term):
      - enc1_x: CNN on frames → s1_x 
      - enc1_y: CNN on frame → s1_y
      - pred1: s1_x → ŝ1_y (predict next step, t+1)

    Level 2 (abstract, long-term):
      - enc2_x: s1_x → s2_x (abstract representation, input = Level 1 output)
      - enc2_y: s1_y → s2_y
      - pred2: s2_x → ŝ2_y (predict multiple steps ahead)

    Both levels trained with VICReg.
    Level 2 uses LARGER temporal gaps for its prediction targets.
    """
```

### Training procedure
1. Train Level 1 JEPA first (or jointly) with short-term prediction targets (t → t+1)
2. Train Level 2 JEPA using Level 1 representations, with long-term prediction targets (t → t+5 or t+10)
3. Level 2 loss encourages it to learn representations where long-term prediction is possible → forces it to **abstract away** fast/unpredictable details

### Visualizations
1. **Level 1 vs Level 2 representations**: t-SNE plots showing that Level 2 clusters trajectories by their long-term behavior (drift direction) while Level 1 preserves fine detail
2. **Prediction accuracy at different horizons**: Level 1 is better at short-term, Level 2 is better at long-term
3. **Information content**: measure what each level encodes and what it discards

### Key intuition to verify
> *Hierarchical stacking with increasing prediction horizons forces higher levels to learn increasingly abstract representations. Level 2 literally cannot predict fine details 10 steps out, so it learns to ignore them — and that's the RIGHT thing to do.*

---

## Step 9: World Model + Actor + Planning (Mode-2)

> **Paper concept (§3, §3.1):** *"Mode-2 perception involves deliberate, reasoned... The agent uses its world model and searches for a sequence of actions that minimizes a given cost... This is akin to model-predictive control."*

### What you'll learn
- Use a learned JEPA as a **world model** that predicts state transitions
- An **actor** proposes action sequences
- **Planning by optimization**: find actions that minimize future cost using gradient descent through the differentiable world model
- Mode-1 (reactive) vs Mode-2 (deliberate) behavior

### Environment
- **MiniGrid (Visual RL)**: A pixel-based gridworld (e.g., `MiniGrid-Empty-8x8-v0` or `MiniGrid-DoorKey-8x8-v0`).
- State: (C, H, W) RGB pixel observation
- Action: Discrete (turn left, turn right, move forward)
- The agent must plan from visual inputs, not just coordinates.

### Architecture

**`model.py`**:
```python
class WorldModel(nn.Module):
    """
    Learned world model as a JEPA (from Steps 5-7)
    Predicts: ŝ_{t+1} = Predictor(s_t, a_t, z_t)
    where s_t = CNN_Encoder(image_observation_t)
    """

class Actor(nn.Module):
    """
    Mode-1: Policy network π(s) → a  (reactive, fast)
    Mode-2: Gradient-based planning through world model (deliberate, slow)

    Mode-2 planning:
      1. Initialize action sequence [a_0, a_1, ..., a_T]
      2. Rollout through world model: s_{t+1} = WorldModel(s_t, a_t)
      3. Compute total cost: C = Σ_t cost(s_t)
      4. Backprop gradient ∂C/∂a_t through world model
      5. Update actions via gradient descent
      6. Repeat for N iterations
      7. Execute first action a_0
    """

class CostModule(nn.Module):
    """
    Differentiable cost function:
    cost(s) = ||s - s_goal||² + obstacle_penalty(s)
    """
```

### Training procedure
1. **Phase 1 — Learn the world model**: collect random trajectories in the environment, train JEPA world model to predict next state from current state + action
2. **Phase 2 — Mode-2 planning**: use gradient-based optimization to plan action sequences through the learned world model
3. **Phase 3 — Mode-1 distillation**: train a reactive policy network to imitate the planned actions (amortized inference)

### Visualizations
1. **World model predictions**: show predicted vs actual trajectories (rollouts)
2. **Planning visualization**: animate the optimization of action sequences — watch paths get refined
3. **Mode-1 vs Mode-2**: reactive policy (fast, approximate) vs planned actions (slow, optimal)
4. **World model error**: how prediction error compounds over longer planning horizons

### Key intuition to verify
> *Planning through a differentiable world model lets the agent "imagine" outcomes before acting. Mode-2 (planning) finds good actions but is slow. Mode-1 (reactive policy) is fast but approximate. The agent can "compile" Mode-2 plans into Mode-1 reactions.*

---

## Step 10: Full Autonomous Agent — Cost Module + Critic + Configurator

> **Paper concept (§3.2, §3.3):** *"The cost module comprises the intrinsic cost module which is immutable and the critic, a trainable module that predicts future values of the intrinsic cost... The configurator module takes input from all other modules and configures them for the task at hand."*

### What you'll learn
- **Intrinsic Cost (IC)**: hard-wired, immutable drives (curiosity, safety, energy)
- **Trainable Critic (TC)**: learns to predict future intrinsic costs → enables long-horizon planning without full world-model rollout
- **Short-term memory**: stores (time, state, intrinsic_cost) triplets for critic training
- **Configurator**: switches between subgoals by adjusting cost weights
- The complete architecture from Figure 2 of the paper

### Architecture — The Full System

**`model.py`**:
```python
class IntrinsicCost(nn.Module):
    """
    IMMUTABLE — not trained.
    IC(s) = u₁·IC₁(s) + u₂·IC₂(s) + ...
    Sub-costs:
      IC_safety:    high energy near obstacles
      IC_curiosity: low energy for novel states (prediction error)
      IC_goal:      high energy far from goal
      IC_energy:    high energy for high-action states (laziness)
    """

class Critic(nn.Module):
    """
    TRAINABLE — predicts future intrinsic cost.
    TC(s_t) ≈ Σ_{δ} γ^δ · IC(s_{t+δ})
    Trained from short-term memory:
      retrieve (s_τ, IC(s_{τ+δ})) → minimize ||TC(s_τ) - IC(s_{τ+δ})||²
    """

class ShortTermMemory:
    """
    Stores triplets: (time τ, state s_τ, intrinsic_cost IC(s_τ))
    Supports retrieval by time or by state (nearest-neighbor)
    Acts as a differentiable key-value memory
    """

class Configurator(nn.Module):
    """
    Executive control: sets weights u_i, v_j for the cost submodules
    Input: current state, goal description, memory contents
    Output: cost weights, perception attention, world model configuration
    Implements subgoal decomposition:
      High-level goal → sequence of subgoals → weight adjustments
    """

class AutonomousAgent(nn.Module):
    """
    The complete system (Figure 2):
    - perception:     Encoder(obs) → s_t
    - world_model:    JEPA from Step 9
    - intrinsic_cost: IntrinsicCost (immutable)
    - critic:         Critic (trainable)
    - actor:          Mode-1 policy + Mode-2 planning
    - memory:         ShortTermMemory
    - configurator:   Configurator

    Main loop:
      1. Perceive: s_t = perception(obs_t)
      2. Configure: configurator adjusts cost weights for current subgoal
      3. Plan (Mode-2): optimize actions through world model + cost
      4. Act: execute first planned action
      5. Remember: store (t, s_t, IC(s_t)) in memory
      6. Train critic: from memory
      7. Update Mode-1 policy: imitate Mode-2 plans
    """
```

### Environment
- **Visual Multi-goal Navigation**: A custom MiniGrid environment or a PyGame visual environment with multiple goals, obstacles, and "energy" pickups. The agent receives RGB images as input.
- The agent must:
  - Navigate to goals (driven by IC_goal)
  - Avoid obstacles (driven by IC_safety)
  - Manage energy (driven by IC_energy)
  - Explore novel areas (driven by IC_curiosity)
- The **configurator** sets which subgoal to prioritize at each moment

### Training procedure
1. **Pre-train world model** on random exploration data (from Step 9)
2. **Initialize intrinsic costs** (hand-designed, NOT learned)
3. **Run agent loop**: perceive → plan → act → memorize → train critic
4. **Observe emergent behavior**: the agent should naturally learn to:
   - Avoid obstacles it has experienced
   - Seek goals efficiently
   - Explore when confident in local area
   - Manage energy by collecting pickups when low

### Visualizations
1. **Agent behavior montage**: show the agent navigating, avoiding, exploring over time
2. **Cost decomposition**: stacked plot of IC_safety, IC_goal, IC_curiosity, IC_energy over time
3. **Critic accuracy**: predicted future cost vs actual future cost
4. **Memory contents**: what states and costs are stored
5. **Configurator decisions**: which subgoal is active over time
6. **Mode-1 vs Mode-2 usage**: when does the agent plan vs react?

### Key intuition to verify
> *The full system demonstrates that combining a learned world model with intrinsic motivation and hierarchical planning produces an agent that autonomously learns useful behaviors without external reward signals. The critic provides "shortcuts" for planning, and the configurator enables flexible subgoal management.*

---

## Open Questions

> [!IMPORTANT]
> **Scope**: Should I implement all 10 steps, or would you prefer to start with a subset (e.g., Steps 1–6 for the JEPA/VICReg core, then decide whether to continue to the agent architecture)?

> [!IMPORTANT]
> **Notebook vs Scripts**: Would you prefer each step as a **Jupyter notebook** (more interactive, inline visualizations) or as **Python scripts** (more modular, easier to iterate on)?

---

## Verification Plan

### For each step:
1. **Train successfully**: loss decreases, no NaN, no collapse
2. **Visualize**: produce the listed plots that demonstrate the concept
3. **Ablation**: remove key components and verify the predicted failure mode
4. **Compare**: benchmark against the alternative approach (e.g., generative vs JEPA)

### End-to-end verification:
- The full agent in Step 10 should demonstrate autonomous goal-seeking, obstacle avoidance, and exploration behavior in the 2D environment
- The agent should get **better over time** as the critic and world model improve
