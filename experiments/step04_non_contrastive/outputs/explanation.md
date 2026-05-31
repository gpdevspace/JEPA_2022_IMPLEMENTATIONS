
# Step 4: Non-Contrastive Joint Embedding

**Paper:** LeCun, *A Path Towards Autonomous Machine Intelligence* (2022) — §4.3  
**Curriculum:** `misc/JEPA_2022_PLAN_GPU.md` — Step 4  
**Code:** `experiments/step04_non_contrastive/`

---

# 1. Paper anchor

Step 4 implements non-contrastive joint embedding training. Unlike contrastive learning, no explicit negative examples are used. Instead, the model learns invariant representations from positive pairs (two augmentations of the same image) while regularization prevents representational collapse.

The implementation evaluates:

- Barlow Twins
- Variance/Covariance regularization (VICReg-style)

Both methods belong to the family of collapse-prevention approaches that replace explicit negatives with statistical constraints on the embedding space.

---

# 2. Problem we solved

The central question is:

> How can a joint embedding architecture learn meaningful representations without negative samples?

If we only minimize:

z₁ ≈ z₂

the network can learn a trivial solution:

z = constant

for every image.

This is called **representation collapse**.

A collapsed model achieves perfect invariance but contains no information about image content.

The role of Barlow Twins and variance/covariance regularization is therefore:

1. Learn invariance between augmentations.
2. Preserve information content.
3. Prevent collapse.
4. Encourage different embedding dimensions to encode different information.

---

# 3. Initial failure mode

The original implementation contained two issues:

## 3.1 Incorrect normalization

Originally:

```python
z = z - z.mean(dim=1, keepdim=True)
std = z.std(dim=1, keepdim=True)
```

This normalizes each sample independently.

For Barlow Twins this is incorrect because the cross-correlation matrix is defined across the batch:

```python
z = z - z.mean(dim=0, keepdim=True)
std = z.std(dim=0, keepdim=True)
```

This computes:

- mean of feature i across samples
- variance of feature i across samples

which is exactly what the Barlow Twins objective requires.

After fixing normalization:

- diagonal correlations became meaningful
- off-diagonal decorrelation started working
- the correlation matrix became interpretable

---

## 3.2 Underpowered projection head

The original projection head:

```text
128 → 512 → 128
```

was too small.

The final configuration:

```text
128 → 1024 → 128
```

provides substantially more capacity.

This is important because Barlow Twins performs most of its decorrelation work in projection space.

A stronger projector allows:

- better feature disentanglement
- improved decorrelation
- higher effective rank
- more stable optimization

---

## 3.3 Weak redundancy penalty

Originally:

```python
lambda = 0.005
```

This was not strong enough to suppress correlated dimensions.

Final configuration:

```python
lambda = 0.05
```

Increasing λ places more weight on removing redundant dimensions.

Observed effect:

- cleaner correlation matrix
- higher effective rank
- flatter covariance spectrum
- improved embedding diversity

---

## 3.4 Small batch statistics

Barlow Twins estimates correlations using batch statistics.

Original:

```text
batch_size = 128
```

Final:

```text
batch_size = 512
```

Larger batches produce more reliable estimates of:

- feature means
- feature variances
- cross-correlations

which significantly stabilizes training.

---

# 4. Final training configuration

| Hyperparameter | Final value |
|---------------|-------------|
| Epochs | 40 |
| Optimizer | Adam |
| Batch size | 512 |
| Projector hidden width | 1024 |
| Projection dimension | 128 |
| Barlow λ | 0.05 |
| Dataset | CIFAR-10 |
| Samples | 10,000 |
| Loss | Barlow Twins |

---

# 5. Understanding the learned representation

A good non-contrastive representation should answer four questions.

## 5.1 Did invariance emerge?

Goal:

```text
augmentation A(image)
≈
augmentation B(image)
```

Measured through:

- Barlow diagonal correlations
- loss reduction

Evidence:

- mean diagonal ≈ 0.95
- steadily decreasing loss

This indicates that different views of the same image are mapped to similar representations.

---

## 5.2 Did redundancy decrease?

Goal:

Different embedding dimensions should encode different information.

Measured through:

- off-diagonal correlations
- covariance spectrum

Evidence:

- off-diagonal magnitude drops steadily
- correlation matrix approaches identity

This shows that dimensions are becoming less redundant.

---

## 5.3 Did collapse occur?

Collapse means:

```text
all images → same embedding
```

Symptoms:

- effective rank ≈ 1
- single dominant eigenvalue
- dense point cloud concentrated in one location

Evidence against collapse:

- effective rank rises from ~2 to >12
- many active covariance directions remain
- embeddings occupy a large region of space

The model is therefore not collapsed.

---

## 5.4 What does the geometry look like?

This is measured through PCA and t-SNE.

Good geometry means:

- images occupy many dimensions
- semantic structure emerges naturally
- representations are spread rather than compressed

---

# 6. Interpretation of final plots

## 6.1 Training loss and Barlow metrics

Observations:

- loss decreases smoothly
- diagonal correlation increases rapidly
- off-diagonal correlation decreases steadily

Interpretation:

The model simultaneously:

- learns invariance
- removes redundancy

This is the exact behavior Barlow Twins is designed to produce.

---

## 6.2 Cross-correlation matrix

Desired appearance:

- bright diagonal
- near-zero off-diagonal entries

Current result:

- strong diagonal line
- mostly weak off-diagonal structure
- dramatic improvement over earlier runs

Interpretation:

Each projection dimension is learning a distinct feature while maintaining agreement across augmentations.

This is one of the strongest indicators that training is now working correctly.

---

## 6.3 Effective rank

This plot is arguably the most important collapse detector.

Observed:

```text
~2  →  ~12.5
```

Interpretation:

The representation continuously expands into additional independent dimensions.

Earlier runs showed effective rank near 1–2, indicating near-collapse.

The final run demonstrates healthy dimensional utilization.

---

## 6.4 Covariance eigenvalue spectrum

Desired behavior:

- gradual decay
- many non-zero eigenvalues

Observed:

- first few eigenvalues dominate
- long tail remains active
- no catastrophic drop to zero

Interpretation:

Information is distributed across many dimensions rather than being concentrated into one dominant component.

This is consistent with successful collapse prevention.

---

## 6.5 PCA density map

The previous energy surface visualization was misleading because non-contrastive methods do not optimize an explicit energy landscape.

The PCA density map is a better diagnostic.

Observed:

- broad occupied region
- dense central manifold
- many valid directions

Interpretation:

Embeddings are spread throughout latent space rather than collapsing into a single point.

---

## 6.6 t-SNE visualization

Observed:

- large-scale structure emerges
- classes are not perfectly separated
- semantically related samples occupy nearby regions

Interpretation:

The model is learning meaningful visual features despite never seeing labels.

This is expected behavior for a self-supervised representation after only 40 epochs on a small CNN.

---

# 7. Why the new configuration works

The original collapse tendency was caused by a combination of:

1. incorrect feature normalization
2. weak redundancy penalty
3. limited projector capacity
4. noisy correlation estimates from small batches

The final system fixes all four issues:

| Change | Effect |
|----------|---------|
| dim=0 normalization | correct correlation statistics |
| λ: 0.005 → 0.05 | stronger decorrelation |
| projector: 512 → 1024 | higher representation capacity |
| batch: 128 → 512 | stable correlation estimates |

Together these changes:

- increase effective rank
- reduce feature redundancy
- improve correlation structure
- prevent collapse
- produce substantially cleaner embedding geometry

---

# 8. Conclusion

The final Barlow Twins implementation now exhibits the expected behavior of a healthy non-contrastive learner:

- invariance emerges between augmented views
- redundancy steadily decreases
- collapse is avoided
- embedding dimensionality increases throughout training
- learned representations occupy a rich latent manifold

The visual diagnostics strongly suggest that the model has transitioned from a near-collapsed representation to a stable and information-rich embedding space.
