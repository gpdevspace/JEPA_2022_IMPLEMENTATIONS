# Step 3: Joint Embedding Architecture (JEA)

**Paper:** LeCun, *A Path Towards Autonomous Machine Intelligence* (2022) — §4.3
**Curriculum:** `misc/JEPA_2022_PLAN_GPU.md` — Step 3
**Code:** `experiments/step03_joint_embedding/`

---

## 1. Paper anchor

Step 3 implements a Joint Embedding Architecture (JEA) from §4.3. The paper describes a model that maps input pairs `(x, y)` into a shared embedding space and defines energy as a distance between the two representations. In the toy implementation here, the model learns embeddings for two augmented views of the same CIFAR-10 image. We compare two variants: a direct squared-distance energy loss that shows the collapse path, and an InfoNCE contrastive objective that recovers useful structure by using batch negatives.

## 2. Problem we solved

This experiment answers the question: *How do we train an energy model by comparing embeddings instead of raw input pairs?* The Step 1 and Step 2 experiments operate in input space; Step 3 lifts the compatibility function into a shared representation space. The first variant uses a direct squared-distance energy loss, and it demonstrates that the encoder can collapse even when both positive and negative batch relationships are present if the loss minimizes all pairwise distances. The second variant uses InfoNCE, which instead uses negatives to push different examples apart and so recovers a useful representation.

## 3. Data

### 3.1 What the variables represent

- `x` and `y` are two augmented views of the same CIFAR-10 image.
- In paper terms, `x` and `y` are different observations that should be compatible if they refer to the same underlying input.
- The label is not used in the loss, but it is used in visualization to color the learned embeddings.

### 3.2 How the data is built

| Item | Value |
|------|-------|
| Dataset | CIFAR-10 |
| x shape / meaning | `(3, 32, 32)` RGB image view 1 |
| y shape / meaning | `(3, 32, 32)` RGB image view 2 |
| Samples | `10,000` training examples (subset) |
| Positives / negatives | positives are two augmentations of the same image; negatives are other images in the batch |

The data loader uses two random augmentations for each image: random crop with padding, horizontal flip, and color jitter. This creates a pair of views that should map to the same semantic representation even though their pixels differ.

### 3.3 Why this dataset was chosen

- CIFAR-10 is a standard vision benchmark with rich visual structure but small enough to train quickly on Apple M5.
- The dataset is high-dimensional enough to make embedding learning meaningful and to justify the use of a joint embedding architecture.
- Using augmentations instead of synthetic 2D toy data matches the step's transition toward representation learning.

### 3.4 Analogy to the full paper setting

This toy maps the later JEPA setting because CIFAR-10 images replace the raw perceptual inputs, and the two augmented views represent two different observations of the same underlying scene. In later steps, `x` and `y` will become temporal or multimodal observations, but the principle of training a shared representation remains the same.

### 3.5 What we excluded (and why)

- No full CIFAR-10 training set was used, only a subset of 10,000 examples to keep runtime manageable.
- The model does not yet use a full ResNet backbone; it uses a small convolutional encoder appropriate for a toy experiment.
- We do not train a task-specific classifier; the goal is representation quality, not classification accuracy.

## 4. Strategy

### Architecture

- `experiments/step03_joint_embedding/model.py` defines `JointEmbeddingModel`.
- The model is a shared encoder for both views.
- The encoder consists of three convolutional blocks, batch normalization, ReLU, max pooling, and a linear projection into a 128-dimensional embedding.
- Energy between a pair is defined as the squared Euclidean distance between embeddings, consistent with the JEA idea.

### Training

| Hyperparameter | Value |
|----------------|-------|
| Loss | squared distance energy or InfoNCE on normalized embeddings |
| Optimizer | Adam |
| Epochs | 30 |
| Batch size | 128 |
| Dataset subset | 10,000 CIFAR-10 examples |
| Device | MPS when available |
| Temperature | 0.1 (InfoNCE only) |

Training is performed in `experiments/step03_joint_embedding/train.py`.
The compatibility energy is computed on the learned embeddings, not on raw images.
Each batch produces positive pairs from two augmentations of the same image and also exposes negative samples through the other batch examples.
The `distance` variant minimizes all squared distances across the batch, including negative pairs, which causes the embedding to collapse by making every point similar. The `info_nce` variant instead minimizes the positive pair similarity while maximizing the negative pair margin, which preserves structure.

## 5. Visualizations

### 5.1 `training_curve_distance.png` / `training_curve_info_nce.png`

- **What is plotted:** epoch-wise mean loss for each training variant.
- **How it was produced:** `visualize.py` reads method-specific histories and plots them separately.
- **How to read it:** a low direct-distance loss may hide collapse, while InfoNCE loss decline with preserved variance indicates recovery.
- **Expected in this step:** the `distance` curve should fall rapidly and can still represent a collapsed model; the `info_nce` curve should fall while preserving embedding structure.
- **Paper link:** shows that a naive energy loss is insufficient without negative structure.

### 5.2 `training_curve_comparison.png`

- **What is plotted:** direct comparison of the two training losses on the same axes.
- **How it was produced:** `visualize.py` overlays the `distance` and `info_nce` curves when both histories are present.
- **How to read it:** the comparison makes the collapse path visible: the `distance` loss can reach a low value even though the embeddings are collapsed, while the `info_nce` loss declines with preserved variance.
- **Expected in this step:** InfoNCE protects against collapse by keeping training aligned with a useful margin.

### 5.3 `embedding_tsne_distance.png` / `embedding_tsne_info_nce.png`

- **What is plotted:** 2D t-SNE embedding of 1,000 CIFAR-10 examples for each training variant.
- **How it was produced:** `visualize.py` computes embeddings for a held-out set and projects them with t-SNE separately for both training methods.
- **How to read it:** colors indicate CIFAR-10 class labels. The `distance` variant should show collapse or poor class separation, while `info_nce` should show more distinct clusters.
- **Expected in this step:** the `distance` variant may collapse toward a small region of the embedding space; the `info_nce` variant should recover semantic structure.
- **Paper link:** this visualizes the difference between a naive energy loss and contrastive recovery.

### 5.4 `embedding_variance_distance.png` / `embedding_variance_info_nce.png`

- **What is plotted:** variance of each embedding dimension over the sample subset, for both variants.
- **How it was produced:** `visualize.py` computes the variance across embedding dimensions.
- **How to read it:** low variance means a dimension is unused. Collapse appears as many near-zero variances in the `distance` variant.
- **Expected in this step:** the `distance` variant should have collapsed dimensions, while `info_nce` should retain more active dimensions.
- **Paper link:** helps verify that the joint embedding preserves dimensional structure when negatives are used.

### 5.5 `embedding_variance_comparison.png`

- **What is plotted:** sorted embedding variances for both methods on the same axes.
- **How it was produced:** `visualize.py` sorts the variance values and overlays them for direct comparison.
- **How to read it:** the plot shows how many dimensions remain active; a collapsed model will have a steep drop-off.
- **Expected in this step:** `info_nce` preserves more variance across dimensions than the collapsed `distance` baseline.

### 5.6 `embedding_correlation_distance.png` / `embedding_correlation_info_nce.png`

- **What is plotted:** the empirical correlation between embedding dimensions for each training variant.
- **How it was produced:** `visualize.py` computes the feature correlation matrix from the learned embeddings.
- **How to read it:** values near 1 on the diagonal and values near 0 off diagonal indicate a decorrelated representation. A collapsed `distance` model will show stronger off-diagonal structure.
- **Expected in this step:** the `distance` variant should show more correlated dimensions, while `info_nce` should produce a cleaner correlation pattern.
- **Paper link:** this plot highlights how InfoNCE provides a decorrelation effect that a raw energy loss does not.

### 5.7 `energy_surface_distance.png` / `energy_surface_info_nce.png`

- **What is plotted:** a 3D energy surface for a fixed reference embedding and a 2D manifold of candidate embeddings.
- **How it was produced:** `visualize.py` projects the learned embeddings into two principal components, reconstructs a local embedding grid, and evaluates the model's energy on that grid.
- **How to read it:** the surface shows where the model assigns low energy (high compatibility) around the reference. A collapsed `distance` model will have a broad flat basin, while the `info_nce` model should show a more structured compatibility landscape.
- **Expected in this step:** the `distance` surface should look nearly uniformly low-energy, indicating the representation has collapsed and cannot distinguish different directions. The `info_nce` surface should show a sharper minimum and more variation, indicating recovery of a useful embedding geometry.
- **Paper link:** this plot makes the collapse/recovery comparison explicit by visualizing the learned energy landscape rather than just summary statistics.

## 6. What we implemented

| File | Role |
|------|------|
| `experiments/step03_joint_embedding/model.py` | Shared encoder and energy / InfoNCE loss utilities |
| `experiments/step03_joint_embedding/train.py` | Joint embedding training loop on CIFAR-10 |
| `experiments/step03_joint_embedding/visualize.py` | Embedding t-SNE, variance, and training curve visualizations |
| `shared/data.py` | Added `CIFAR10PairDataset` for paired view generation |
| `shared/device.py` | MPS/CPU device selection |
| `shared/viz.py` | Training curve plotting utility |

## 7. Results and evidence

From the training runs:

- `distance` variant final loss: `0.000857`, indicating the batch-distance objective collapsed the embedding.
- `info_nce` variant final loss: `0.167889`, indicating a stable contrastive representation.

The direct distance objective minimizes all pairwise batch distances and therefore collapses the encoder even though negative samples are present. InfoNCE instead preserves structure by using negative batch examples to separate different images.

## 8. What this establishes

- A JEA can be trained on image data using paired augmentations and an InfoNCE loss.
- The shared encoder learns meaningful embeddings rather than collapsing to a constant when InfoNCE is used.
- The direct squared-distance variant demonstrates the collapse path when the loss minimizes all pairwise batch distances, including negatives.
- The learned embeddings show semantic structure in the CIFAR-10 toy setting for the InfoNCE variant.
- This step transitions the curriculum from raw energy scoring to representation-based compatibility.

## 9. Connection to the paper

This step implements §4.3’s key idea that energy can be defined by a distance in embedding space. It also demonstrates the failure mode of direct distance minimization and the recovery obtained by using InfoNCE. This comparison is the basis for later JEPA and non-contrastive training methods.

## 10. Limitations of this toy

- The encoder is a small CNN, not a full ResNet or transformer.
- Only a 10,000-image subset of CIFAR-10 was used for runtime efficiency.
- The visualization uses t-SNE on 1,000 samples, which is a local view of the embedding space.
- We do not yet introduce temporal prediction or latent-variable structure.

### Curse of dimensionality for negatives

In a 128-dimensional embedding space, explicitly sampling enough negatives is extremely difficult. The volume of a 128D ball grows so rapidly that random negatives concentrate on a thin shell, making them less informative. To cover the space meaningfully, you would need exponentially many negative samples in the embedding dimension. This is why InfoNCE uses batch-based implicit negatives instead of trying to exhaustively sample the full 128D space.

## 11. Next step

Step 4 will remove the need for explicit negative samples and introduce non-contrastive regularizers like Barlow Twins or VICReg. That step will show how to prevent collapse in a joint embedding setup when the loss no longer relies on cross-sample contrast.

## Reproduce

From the repository root:

```bash
uv run python experiments/step03_joint_embedding/train.py --subset-size 10000 --epochs 30 --batch-size 128 --lr 1e-3 --temperature 0.1
uv run python experiments/step03_joint_embedding/visualize.py --subset-size 1000 --batch-size 128
```
