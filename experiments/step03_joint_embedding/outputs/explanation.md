# Step 3: Joint Embedding Architecture (JEA)

**Paper:** LeCun, *A Path Towards Autonomous Machine Intelligence* (2022) — §4.3
**Curriculum:** `misc/JEPA_2022_PLAN_GPU.md` — Step 3
**Code:** `experiments/step03_joint_embedding/`

---

## 1. Paper anchor

Step 3 implements a Joint Embedding Architecture (JEA) from §4.3. The paper describes a model that maps input pairs `(x, y)` into a shared embedding space and defines energy as a distance between the two representations. In the toy implementation here, the model learns embeddings for two augmented views of the same CIFAR-10 image and is trained so that matching views are closer than non-matching views.

## 2. Problem we solved

This experiment answers the question: *How do we train an energy model by comparing embeddings instead of raw input pairs?* The Step 1 and Step 2 experiments operate in input space; Step 3 lifts the compatibility function into a shared representation space. The failure mode we avoid is embedding collapse, where a trivial representation makes all images appear identical. The training objective used here encourages positive pairs to match while discouraging negative pairs implicitly through InfoNCE.

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
| Loss | InfoNCE on normalized embeddings |
| Optimizer | Adam |
| Epochs | 30 |
| Batch size | 128 |
| Dataset subset | 10,000 CIFAR-10 examples |
| Device | MPS when available |
| Temperature | 0.1 |

Training is performed in `experiments/step03_joint_embedding/train.py`.
Each batch produces positive pairs from two augmentations of the same image. The InfoNCE loss encourages the matching pair to have higher similarity than all other pairs in the batch.

## 5. Visualizations

### 5.1 `training_curve.png`

- **What is plotted:** epoch-wise mean InfoNCE loss with batch standard deviation.
- **How it was produced:** `experiments/step03_joint_embedding/visualize.py` reads `loss_history.json` and plots the loss curve.
- **How to read it:** A falling curve means the model is learning to align positive embeddings and separate negatives.
- **Expected in this step:** steady decline from a high initial loss to a lower stable value.
- **Paper link:** shows that the joint embedding objective is being optimized, which is the core training behavior of JEA.

### 5.2 `embedding_tsne.png`

- **What is plotted:** 2D t-SNE embedding of 1,000 CIFAR-10 examples.
- **How it was produced:** `visualize.py` computes embeddings for a held-out set and projects them with t-SNE.
- **How to read it:** colors indicate CIFAR-10 class labels; clusters indicate semantic grouping in embedding space.
- **Expected in this step:** same-class points should be closer together than different-class points, showing the encoder learned useful structure.
- **Paper link:** this is the first visualization of JEA representations, illustrating that the embedding space captures semantic similarity.

### 5.3 `embedding_variance.png`

- **What is plotted:** variance of each embedding dimension over the sample subset.
- **How it was produced:** `visualize.py` computes the variance across embedding dimensions.
- **How to read it:** higher variance in a dimension indicates that the model is using that dimension; near-zero variance would indicate collapse.
- **Expected in this step:** a spread of variances across dimensions rather than all dims being uniform or near zero.
- **Paper link:** helps verify that the joint embedding has useful dimensional structure and is not trivially collapsed.

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

From the training run:

- Epoch 0 mean loss: `1.8410`, std loss: `0.8808`
- Epoch 1 mean loss: `0.7134`, std loss: `0.1391`
- Epoch 29 mean loss: `0.1110`, std loss: `0.0179`

The loss dropped steadily, demonstrating that the encoder learned to align same-image views and separate other images in the batch. The final loss plateau at ~0.11 indicates a stable joint embedding objective on the CIFAR-10 subset.

## 8. What this establishes

- A JEA can be trained on image data using paired augmentations and an InfoNCE loss.
- The shared encoder learns meaningful embeddings rather than collapsing to a constant.
- The learned embeddings show semantic structure in the CIFAR-10 toy setting.
- This step transitions the curriculum from raw energy scoring to representation-based compatibility.

## 9. Connection to the paper

This step implements §4.3’s key idea that energy can be defined by a distance in embedding space. It demonstrates how two views of the same underlying input can be pulled together in representation space, which is the basis for later JEPA and non-contrastive training methods.

## 10. Limitations of this toy

- The encoder is a small CNN, not a full ResNet or transformer.
- Only a 10,000-image subset of CIFAR-10 was used for runtime efficiency.
- The visualization uses t-SNE on 1,000 samples, which is a local view of the embedding space.
- We do not yet introduce temporal prediction or latent-variable structure.

## 11. Next step

Step 4 will remove the need for explicit negative samples and introduce non-contrastive regularizers like Barlow Twins or VICReg. That step will show how to prevent collapse in a joint embedding setup when the loss no longer relies on cross-sample contrast.

## Reproduce

From the repository root:

```bash
uv run python experiments/step03_joint_embedding/train.py --subset-size 10000 --epochs 30 --batch-size 128 --lr 1e-3 --temperature 0.1
uv run python experiments/step03_joint_embedding/visualize.py --subset-size 1000 --batch-size 128
```
