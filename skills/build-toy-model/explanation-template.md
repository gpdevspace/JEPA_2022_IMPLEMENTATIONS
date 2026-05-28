# explanation.md template (copy to `experiments/stepNN_*/outputs/explanation.md`)

Replace all bracketed placeholders. Delete this header line in the final file.

---

# Step [N]: [Title]

**Paper:** LeCun, *A Path Towards Autonomous Machine Intelligence* (2022) — §[section]  
**Curriculum:** `misc/JEPA_2022_PLAN_GPU.md` — Step [N]  
**Code:** `experiments/step[NN]_[name]/`

## 1. Paper anchor

[Quote or paraphrase the core definition from the paper. State where this step sits in the 10-step arc.]

## 2. Problem we solved

[What conceptual problem does this experiment address? What failure mode or capability are we demonstrating?]

## 3. Data

### 3.1 What the variables represent

[Semantic meaning of x and y — what they stand for in the paper’s notation and in the real-world analogy.]

### 3.2 How the data is built

| Item | Value |
|------|--------|
| Dataset | [name] |
| x shape / meaning | […] |
| y shape / meaning | […] |
| Samples | [N] |
| Positives / negatives | [how pairs are built] |

[Equations or sampling procedure. Noise, normalization, splits.]

### 3.3 Why this dataset was chosen

[Explicit pedagogical reasons: dimensionality, visualize-ability, manifold structure, multimodality, temporal gap, etc.]

### 3.4 Analogy to the full paper setting

[How this toy maps to images / video / actions / states in later steps.]

### 3.5 What we excluded (and why)

[Missing negatives, held-out conditions, simplified world — tied to this step’s goal.]

## 4. Strategy

### Architecture

[Diagram or bullet list: encoders, heads, energy / loss.]

### Training

| Hyperparameter | Value |
|----------------|--------|
| Loss | [formula] |
| Optimizer | […] |
| Epochs | […] |
| Batch size | […] |
| Device | MPS / CPU |

[Intentional design choices: naive vs contrastive, ablations, etc.]

## 5. Visualizations

One subsection per output figure.

### 5.1 `[filename].png`

- **What is plotted:** [axes, fixed variables, range]
- **How produced:** [`visualize.py`, grid size, checkpoint]
- **How to read it:** [color scale, overlays, landmarks]
- **Expected in this step:** [collapse vs success pattern]
- **Paper link:** [§ / intuition]

### 5.2 `[next file].png`

[Repeat for every PNG.]

## 6. What we implemented

| File | Role |
|------|------|
| `model.py` | […] |
| `train.py` | […] |
| `visualize.py` | […] |
| `shared/…` | [reuse] |

## 7. Results and evidence

[Report numbers from your run. Reference section 5 by figure name.]

## 8. What this establishes

[Bullet list of concrete things you can now claim before Step N+1.]

## 9. Connection to the paper

[How §X reads differently after running this.]

## 10. Limitations of this toy

[What we simplified; what Step N+1 will add.]

## 11. Next step

[Bridge to next experiment.]

## Reproduce

```bash
cd /path/to/jepa_2022
uv run python experiments/step[NN]_*/train.py
uv run python experiments/step[NN]_*/visualize.py
```
