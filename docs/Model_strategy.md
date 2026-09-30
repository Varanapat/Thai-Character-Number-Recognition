# Model Strategy — Thai Character Classification

## 1. Purpose

This document defines the training and inference strategies for the Thai Character Classification project.

The goal is **not** to train as many models as possible. The goal is to find a training strategy that generalizes well to new sources and unseen data while keeping inference practical.

Our selected architecture from the initial model comparison is:

> **MobileNetV3-Small**

The architecture is kept fixed while we compare different **training strategies**.

---

# 2. Current Model Selection

The initial model comparison evaluated:

- Custom CNN
- ResNet18
- EfficientNet-B0
- DenseNet121
- MobileNetV3-Small

Current result:

| Model | Accuracy | Macro F1 | Params (M) | Inference (ms) |
|---|---:|---:|---:|---:|
| MobileNetV3-Small | 0.9929 | **0.9813** | 1.60 | 3.19 |
| DenseNet121 | **0.9937** | 0.9794 | 7.04 | 8.48 |
| ResNet18 | 0.9934 | 0.9792 | 11.22 | **1.35** |
| EfficientNet-B0 | 0.9929 | 0.9777 | 4.11 | 4.84 |
| Custom CNN | 0.9920 | 0.9631 | **1.19** | **0.75** |

### Why MobileNetV3-Small?

MobileNetV3-Small is not the highest-accuracy model in the current validation result.

However, it provides a strong balance between:

- High accuracy
- High Macro F1
- Small parameter count
- Fast inference
- Practical deployment

Therefore, we use **MobileNetV3-Small as the fixed backbone for the next experiments**.

---

# 3. Main Research Question

The next question is:

> **Which training strategy gives the best generalization to unseen data?**

We therefore compare:

```text
Same Architecture
      ↓
MobileNetV3-Small
      ↓
Different Training Strategies
      ↓
Cross-Source Evaluation
      ↓
Select Final Strategy
      ↓
Final Unseen Data Inference
```

The architecture should not change between these experiments.

This makes the comparison about **training strategy**, rather than architecture.

---

# 4. Dataset Sources

We currently have three datasets:

```text
D1
D2
D3
```

We treat them as different sources/domains.

The important observation so far is that generalization is not symmetric.

For example:

```text
Train D1 → Test D3 ≈ 40%
Train D3 → Other Source ≈ 80%
```

This suggests that the data distributions of the sources may be different.

Therefore, source-aware evaluation is important.

---

# 5. Strategy Candidates

## Strategy S1 — D1 Only

Train MobileNetV3-Small using D1 only.

```text
D1
 ↓
MobileNetV3-Small
 ↓
Test / Inference
```

### Purpose

Baseline for understanding how well D1 alone generalizes.

### Question

> How much does a model trained only on D1 depend on the characteristics of D1?

---

# 6. Strategy S2 — D3 Only

Train MobileNetV3-Small using D3 only.

```text
D3
 ↓
MobileNetV3-Small
 ↓
Test / Inference
```

### Purpose

This is particularly important because preliminary experiments suggest that D3 may generalize better across sources.

### Question

> Does D3 provide more transferable character representations?

We should not assume this is true before measuring it across the same evaluation protocol.

---

# 7. Strategy S3 — Combined Training

Train one MobileNetV3-Small using all three datasets.

```text
D1 ─┐
D2 ─┼──→ Combined Training → MobileNetV3-Small
D3 ─┘
```

### Purpose

Test whether exposing the model to multiple source distributions improves generalization.

### Important consideration

The datasets may have different sizes.

For example:

```text
D1 = 50,000
D2 = 10,000
D3 = 5,000
```

A naive combination would make D1 dominate training.

Therefore, record the number of samples per source and consider whether balanced sampling is needed.

Possible approaches:

- Natural sampling
- Equal source sampling
- Weighted sampling

Do not change the sampling strategy between experiments without recording it.

---

# 8. Strategy S4 — Separate Models + Majority Vote

Train three independent MobileNetV3-Small models:

```text
D1 → Model D1
D2 → Model D2
D3 → Model D3
```

For each new image:

```text
              New Image
                  │
        ┌─────────┼─────────┐
        ↓         ↓         ↓
     Model D1  Model D2  Model D3
        ↓         ↓         ↓
        ก         ก         ข
        └─────────┼─────────┘
                  ↓
            Majority Vote
                  ↓
                  ก
```

### Purpose

Test whether independent source-specific models can complement each other.

### Advantages

- Simple
- Easy to explain
- No retraining required after the three models exist
- Can reveal whether different sources provide complementary knowledge

### Limitation

All models have equal voting power even if one model is much more confident or generalizes better.

---

# 9. Strategy S5 — Probability Averaging

Instead of using only the predicted class, use the complete probability distribution from each model.

Example:

```text
Model D1:
ก = 0.70
ข = 0.20
ค = 0.10

Model D2:
ก = 0.40
ข = 0.50
ค = 0.10

Model D3:
ก = 0.30
ข = 0.60
ค = 0.10
```

Average the probabilities:

```text
P_final(class)
=
(P_D1 + P_D2 + P_D3) / 3
```

Then select the class with the highest final probability.

### Why test this?

It uses more information than hard majority voting.

---

# 10. Strategy S6 — Weighted Probability Ensemble

If cross-source experiments show that some source-specific models generalize better than others, assign weights based on **predefined validation/cross-source evidence**.

Example:

```text
D1 model = 0.20
D2 model = 0.30
D3 model = 0.50
```

Then:

```text
P_final
=
0.20 × P_D1
+
0.30 × P_D2
+
0.50 × P_D3
```

### Important rule

Weights must be determined **before the final unseen test**.

Do not choose weights based on the predictions or hidden labels of tomorrow's evaluation data.

---

# 11. Strategy S7 — Combined Model vs Ensemble

This is an important comparison.

### Approach A — Combine Knowledge During Training

```text
D1 + D2 + D3
        ↓
   One MobileNet
        ↓
    Prediction
```

### Approach B — Combine Knowledge During Inference

```text
D1 → Model
D2 → Model
D3 → Model
        ↓
    Ensemble
        ↓
    Prediction
```

### Research Question

> Is it better to expose one model to diverse sources during training, or preserve source-specific models and combine their predictions at inference time?

This can become a strong discussion point if the results differ.

---

# 12. Recommended Experiment Order

Do not run everything randomly.

Use the following progression:

```text
S1 — D1 Only
      ↓
S2 — D3 Only
      ↓
S3 — D1+D2+D3
      ↓
S4 — Majority Vote
      ↓
S5 — Probability Averaging
      ↓
S6 — Weighted Probability
```

If time is limited, prioritize:

```text
1. D1-only
2. D3-only
3. Combined
4. Majority Vote
```

Then add probability-based ensembles if useful.

---

# 13. Evaluation Protocol

The final unseen test set must not be used to select the strategy.

Use existing labeled data for experimentation.

Recommended evaluation:

```text
                 Training
                    ↓
              80% Train
                    │
                    ↓
                 Model
                    │
                    ↓
              20% Validation
```

For source-generalization experiments:

```text
Train Source
     ↓
MobileNet
     ↓
Different Source
     ↓
Evaluation
```

Examples:

```text
D1 → D2
D1 → D3

D3 → D1
D3 → D2
```

Use the same evaluation protocol when comparing strategies.

---

# 14. Metrics

Record at least:

- Accuracy
- Macro F1
- Inference time

For ensemble experiments, also record:

- Agreement rate between models
- Confidence
- Number of samples where models disagree

Recommended experiment table:

| Strategy | Test Source | Accuracy | Macro F1 | Inference ms | Notes |
|---|---|---:|---:|---:|---|
| D1-only | D2 | - | - | - | |
| D1-only | D3 | - | - | - | |
| D3-only | D1 | - | - | - | |
| D3-only | D2 | - | - | - | |
| Combined | D1 | - | - | - | |
| Combined | D2 | - | - | - | |
| Combined | D3 | - | - | - | |
| Majority Vote | D1 | - | - | - | |
| Majority Vote | D2 | - | - | - | |
| Majority Vote | D3 | - | - | - | |

---

# 15. Inference Output

For experiments, save detailed prediction results.

Recommended CSV:

```text
filename,d1_pred,d1_conf,d2_pred,d2_conf,d3_pred,d3_conf,final_pred,final_conf
image_001.png,ก,0.91,ก,0.83,ข,0.97,ก,0.91
image_002.png,ข,0.98,ค,0.55,ข,0.94,ข,0.96
```

This allows us to inspect disagreements between models.

For the actual submission, use the exact output format required by the instructor.

---

# 16. Project Directory Structure

Recommended structure:

```text
thai-character-classification/
│
├── README.md
├── Model_strategy.md
│
├── configs/
│   ├── d1.yaml
│   ├── d3.yaml
│   ├── combined.yaml
│   └── ensemble.yaml
│
├── data/
│   ├── D1/
│   ├── D2/
│   └── D3/
│
├── models/
│   ├── mobilenet_d1/
│   │   └── best.pth
│   ├── mobilenet_d2/
│   │   └── best.pth
│   ├── mobilenet_d3/
│   │   └── best.pth
│   └── mobilenet_combined/
│       └── best.pth
│
├── src/
│   ├── dataset.py
│   ├── transforms.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── ensemble/
│   ├── majority_vote.py
│   ├── probability_average.py
│   └── weighted_ensemble.py
│
├── experiments/
│   ├── model_comparison/
│   ├── cross_source/
│   └── inference_trials/
│
├── results/
│   ├── model_comparison.csv
│   ├── cross_source_results.csv
│   ├── confusion_matrix/
│   └── inference/
│
└── submission/
    └── final_predictions.csv
```

---

# 17. What Belongs Where?

## `src/model.py`

Only define the MobileNetV3-Small architecture and classifier head.

```text
Architecture
```

---

## `src/transforms.py`

Keep preprocessing and augmentation here.

```text
Resize
Crop
Tensor conversion
Normalization
Training augmentation
Validation transform
```

This is important because preprocessing must be consistent between training and inference.

---

## `src/dataset.py`

Responsible for:

- Loading images
- Reading labels
- Mapping class names to class IDs
- Dataset/source identification

---

## `src/train.py`

Responsible for training one strategy at a time.

Examples:

```text
train D1
train D3
train combined
```

---

## `src/evaluate.py`

Responsible for:

- Accuracy
- Macro F1
- Confusion matrix
- Per-class metrics

---

## `src/inference.py`

Responsible for:

```text
Image
 ↓
Preprocessing
 ↓
MobileNet
 ↓
Prediction
 ↓
Confidence
 ↓
CSV
```

---

## `ensemble/`

Only for combining predictions from already-trained models.

This means ensemble experiments do **not** require retraining the individual models.

---

# 18. Preprocessing and MobileNetV3-Small

Selecting MobileNetV3-Small does **not** mean preprocessing happens automatically inside the model.

The model and preprocessing are separate components.

Conceptually:

```text
Raw Image
    ↓
Preprocessing / Transform
    ↓
Tensor
    ↓
MobileNetV3-Small
    ↓
Logits
    ↓
Softmax
    ↓
Prediction
```

For torchvision's pretrained MobileNetV3-Small weights, the weight object provides the recommended preprocessing configuration. However, when implementing a training pipeline, the transform still needs to be explicitly applied to the input dataset.

Typical preprocessing includes:

```text
Resize / Crop
      ↓
Convert to Tensor
      ↓
Normalize
      ↓
MobileNetV3-Small
```

If the original project already has a preprocessing pipeline that produced the current model results, **keep the same preprocessing when comparing training strategies**.

Do not change preprocessing at the same time as the training strategy, otherwise the experiment becomes difficult to interpret.

---

# 19. Training Transform vs Validation/Inference Transform

These should be separated.

### Training

```text
Image
 ↓
Resize
 ↓
Thai-specific augmentation (if enabled)
 ↓
Tensor
 ↓
Normalize
 ↓
Model
```

### Validation

```text
Image
 ↓
Resize
 ↓
Tensor
 ↓
Normalize
 ↓
Model
```

### Inference

```text
New Image
 ↓
Same validation/inference preprocessing
 ↓
Tensor
 ↓
Model
 ↓
Prediction
```

Do not apply random augmentation to the validation or final inference images.

---

# 20. Important Rule for Fair Comparison

When comparing:

```text
D1-only
D3-only
Combined
Ensemble
```

keep the following as consistent as possible:

- MobileNetV3-Small architecture
- Input size
- Normalization
- Optimizer
- Learning rate
- Epochs / stopping rule
- Loss function
- Random seed
- Evaluation metrics

The main variable should be the **training strategy**.

---

# 21. Final Model Selection

The final strategy should be selected using existing labeled validation/cross-source experiments.

Do not select the final strategy using tomorrow's unseen data.

Decision process:

```text
Existing Labeled Data
        ↓
Cross-Source Evaluation
        ↓
Compare Strategies
        ↓
Select Strategy
        ↓
LOCK
        ↓
Tomorrow's Unseen Data
        ↓
Final Inference
```

---

# 22. Final Inference

Once the strategy is selected:

```text
LOCKED STRATEGY
      ↓
New Unseen Data
      ↓
Same Preprocessing
      ↓
MobileNet / Ensemble
      ↓
Prediction
      ↓
Required Submission Format
```

The final output format should follow the instructor's specification.

---

# 23. Decision Checklist

Before selecting the final strategy:

- [ ] Does it perform well on the standard validation set?
- [ ] Does it generalize across sources?
- [ ] Does it handle D1 → D3 failure?
- [ ] Is Macro F1 acceptable?
- [ ] Is inference fast enough?
- [ ] Is the pipeline reproducible?
- [ ] Is the strategy fixed before the final unseen test?

---

# 24. Current Working Hypothesis

Based on the preliminary results:

> D1-only may be highly source-dependent, while D3-only may provide stronger cross-source generalization.

However, this is only a hypothesis.

The experiments should determine whether:

```text
D1-only
vs
D3-only
vs
Combined
vs
Ensemble
```

provides the most robust performance on unseen sources.

---

# 25. Final Goal

The goal is not simply:

> **"Which model has the highest validation accuracy?"**

The goal is:

> **"Which training strategy allows MobileNetV3-Small to generalize most reliably to data from sources that were not used during training?"**

This distinction is the core of the project.
