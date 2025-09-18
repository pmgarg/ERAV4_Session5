
```bash


### View All Experiments
| Experiment | Notebook | Best Accuracy | View on |
|------------|----------|---------------|----------|
| **Exp 1: SGD + Scheduler** | [`Session5_model_training_SGD.ipynb`](Session5_model_training_SGD.ipynb) | 99.56% |
| **Exp 2: SGD Fixed LR** | [`Session5_model_training_SGD_no_scheduler.ipynb`](Session5_model_training_SGD_no_scheduler.ipynb) | 99.10%  |
| **Exp 3: Adam + Scheduler** ⭐ | [`Session5_model_training_adam.ipynb`](Session5_model_training_adam.ipynb) | **99.61%** |
| **Exp 4: AdamW + Scheduler** | [`Session5_model_training_adamw.ipynb`](Session5_model_training_adamw.ipynb) | 99.60% |
## 📊 Model Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│  Input: 28×28×1 MNIST Image                        │
├─────────────────────────────────────────────────────┤
│  Conv Block 1: Feature Extraction                   │
│  ├─ Conv1 (1→14) + BN → 26×26×14                  │
│  ├─ Conv2 (14→14) + BN → 24×24×14                 │
│  └─ Conv3 (14→16) + BN → 22×22×16                 │
├─────────────────────────────────────────────────────┤
│  MaxPool2d(2×2) → 11×11×16                         │
├─────────────────────────────────────────────────────┤
│  Conv Block 2: Deep Features                        │
│  ├─ Conv4 (16→18) + BN → 9×9×18                   │
│  └─ Conv5 (18→20) + BN → 7×7×20                   │
├─────────────────────────────────────────────────────┤
│  MaxPool2d(2×2) → 3×3×20                           │
├─────────────────────────────────────────────────────┤
│  Conv Block 3: Final Features                       │
│  ├─ Conv6 (20→22) + BN → 3×3×22                   │
│  └─ Conv7 (22→22) + BN → 1×1×22                   │
├─────────────────────────────────────────────────────┤
│  Global Average Pooling → 22                       │
│  Fully Connected → 10 classes                       │
└─────────────────────────────────────────────────────┘
```

## 🎯 Training Evolution: The Quest for 99.4%

### 📈 Experiment Timeline

<table>
<tr>
<td>

#### 🔬 Experiment 1: Baseline SGD
📓 **Notebook:** [`Session5_model_training_SGD.ipynb`](./Session5_model_training_SGD.ipynb)

**Configuration:**
- Optimizer: `SGD`
- Learning Rate: `0.006`
- Weight Decay: `1e-6`
- Scheduler: `OneCycleLR`
- Max LR: `0.02`

</td>
<td>

```python
# Performance Metrics
┌──────────┬──────────┬──────────┐
│  Epoch   │  Train   │   Test   │
├──────────┼──────────┼──────────┤
│    1     │  85.66%  │  98.20%  │
│    5     │  98.49%  │  99.21%  │
│   10     │  98.96%  │  99.47%  │
│   15     │  99.18%  │  99.50%  │
│   20     │  99.39%  │  99.56%  │
└──────────┴──────────┴──────────┘
```

</td>
</tr>
</table>

#### 🏆 Achievement Unlocked: **99.56%** Test Accuracy! ✨

---

<table>
<tr>
<td>

#### 🔬 Experiment 2: SGD Without Scheduler
📓 **Notebook:** [`Session5_model_training_SGD_no_scheduler.ipynb`](./Session5_model_training_SGD_no_scheduler.ipynb)

**Configuration:**
- Optimizer: `SGD`
- Learning Rate: `0.06` (fixed)
- Weight Decay: `1e-4`
- Scheduler: `None`

</td>
<td>

```python
# Performance Metrics
┌──────────┬──────────┬──────────┐
│  Epoch   │  Train   │   Test   │
├──────────┼──────────┼──────────┤
│    1     │  64.31%  │  90.57%  │
│    5     │  96.08%  │  98.26%  │
│   10     │  97.68%  │  98.63%  │
│   15     │  98.15%  │  99.02%  │
│   20     │  98.47%  │  99.10%  │
└──────────┴──────────┴──────────┘
```

</td>
</tr>
</table>

#### 📊 Result: **99.10%** - Slower convergence without scheduler

---

<table>
<tr>
<td>

#### 🔬 Experiment 3: Adam with Scheduler
📓 **Notebook:** [`Session5_model_training_adam.ipynb`](./Session5_model_training_adam.ipynb)

**Configuration:**
- Optimizer: `Adam`
- Learning Rate: `0.01`
- Weight Decay: `1e-4`
- Scheduler: `OneCycleLR`

</td>
<td>

```python
# Performance Metrics
┌──────────┬──────────┬──────────┐
│  Epoch   │  Train   │   Test   │
├──────────┼──────────┼──────────┤
│    1     │  92.00%  │  98.17%  │
│    5     │  98.00%  │  98.97%  │
│   10     │  98.12%  │  99.01%  │
│   15     │  98.75%  │  99.42%  │
│   20     │  99.41%  │  99.61%  │
└──────────┴──────────┴──────────┘
```

</td>
</tr>
</table>

#### 🎉 New Record: **99.61%** Test Accuracy! 🥇

---

<table>
<tr>
<td>

#### 🔬 Experiment 4: AdamW with Scheduler
📓 **Notebook:** [`Session5_model_training_adamw.ipynb`](./Session5_model_training_adamw.ipynb)

**Configuration:**
- Optimizer: `AdamW`
- Learning Rate: `0.02`
- Weight Decay: `1e-4`
- Scheduler: `OneCycleLR`

</td>
<td>

```python
# Performance Metrics
┌──────────┬──────────┬──────────┐
│  Epoch   │  Train   │   Test   │
├──────────┼──────────┼──────────┤
│    1     │  92.28%  │  96.43%  │
│    5     │  98.41%  │  98.80%  │
│   10     │  98.86%  │  99.39%  │
│   15     │  99.22%  │  99.55%  │
│   20     │  99.53%  │  99.60%  │
└──────────┴──────────┴──────────┘
```

</td>
</tr>
</table>

#### 🎯 Excellent Performance: **99.60%** Test Accuracy! 

---

## 📊 Comparative Analysis

### Convergence Speed Comparison

```
Test Accuracy Progress:
      99.6% ┤                                    ╭─────
      99.4% ┤                         ╭──────────╯
      99.2% ┤                   ╭─────╯
      99.0% ┤            ╭──────╯
      98.8% ┤       ╭────╯
      98.6% ┤   ╭───╯
      98.4% ┤  ╱
      98.2% ┤ ╱
            └─┬────┬────┬────┬────┬────┬────┬────┬────
              1    3    5    7    9   11   13   15   20
                            Epochs

    ─── Exp 1 (SGD+Scheduler)    Best: 99.56%
    ─── Exp 2 (SGD Only)         Best: 99.10%
    ─── Exp 3 (Adam+Scheduler)   Best: 99.61% ⭐
    ─── Exp 4 (AdamW+Scheduler)  Best: 99.60%
```

## 🔑 Key Insights

### ✅ What Worked Well

1. **OneCycleLR Scheduler** 
   - Dramatically improved convergence speed
   - Better final accuracy across all optimizers

2. **Adam Optimizer**
   - Fastest initial convergence
   - Achieved best overall accuracy (99.61%)
   
3. **Batch Normalization**
   - Stabilized training across all experiments
   - No signs of overfitting even without dropout

### 📈 Performance Summary

| Metric | Exp 1 | Exp 2 | Exp 3 | Exp 4 |
|--------|-------|-------|-------|-------|
| **Best Test Acc** | 99.56% | 99.10% | **99.61%** 🏆 | 99.60% |
| **Best Train Acc** | 99.39% | 98.47% | 99.41% | **99.53%** |
| **Epochs to 99%** | 4 | 15 | 10 | 9 |
| **Training Stability** | High | Medium | High | High |

## 🚀 Model Efficiency Stats

<div align="center">

| **Metric** | **Value** | **Status** |
|------------|-----------|------------|
| Total Parameters | 18,662 | ✅ Under 25K limit |
| Model Size | ~73 KB | 🎯 Ultra-lightweight |
| Training Time/Epoch | ~6.5 seconds | ⚡ Fast |
| Inference Time | <1 ms | 🚀 Real-time capable |
| Memory Usage | <50 MB | 💚 Mobile-ready |

</div>

## 🏗️ Detailed Architecture Breakdown

### Layer-by-Layer Analysis

| Layer | Type | Parameters | Input Size | Output Size | Receptive Field |
|-------|------|-----------|------------|-------------|-----------------|
| **conv1** | Conv2d(1→14, k=3) | (3×3×1+1)×14 = 140 | 28×28×1 | 26×26×14 | 3×3 |
| **bn1** | BatchNorm2d | 14×2 = 28 | 26×26×14 | 26×26×14 | 3×3 |
| **conv2** | Conv2d(14→14, k=3) | (3×3×14+1)×14 = 1,778 | 26×26×14 | 24×24×14 | 5×5 |
| **bn2** | BatchNorm2d | 14×2 = 28 | 24×24×14 | 24×24×14 | 5×5 |
| **conv3** | Conv2d(14→16, k=3) | (3×3×14+1)×16 = 2,032 | 24×24×14 | 22×22×16 | 7×7 |
| **bn3** | BatchNorm2d | 16×2 = 32 | 22×22×16 | 22×22×16 | 7×7 |
| **pool1** | MaxPool2d(2×2) | 0 | 22×22×16 | 11×11×16 | 14×14 |
| **conv4** | Conv2d(16→18, k=3) | (3×3×16+1)×18 = 2,610 | 11×11×16 | 9×9×18 | 18×18 |
| **bn4** | BatchNorm2d | 18×2 = 36 | 9×9×18 | 9×9×18 | 18×18 |
| **conv5** | Conv2d(18→20, k=3) | (3×3×18+1)×20 = 3,260 | 9×9×18 | 7×7×20 | 22×22 |
| **bn5** | BatchNorm2d | 20×2 = 40 | 7×7×20 | 7×7×20 | 22×22 |
| **pool2** | MaxPool2d(2×2) | 0 | 7×7×20 | 3×3×20 | 44×44 |
| **conv6** | Conv2d(20→22, k=3, p=1) | (3×3×20+1)×22 = 3,982 | 3×3×20 | 3×3×22 | 44×44 |
| **bn6** | BatchNorm2d | 22×2 = 44 | 3×3×22 | 3×3×22 | 44×44 |
| **conv7** | Conv2d(22→22, k=3) | (3×3×22+1)×22 = 4,378 | 3×3×22 | 1×1×22 | 48×48 |
| **bn7** | BatchNorm2d | 22×2 = 44 | 1×1×22 | 1×1×22 | 48×48 |
| **gap** | AdaptiveAvgPool2d | 0 | 1×1×22 | 1×1×22 | Full |
| **fc** | Linear(22→10) | 22×10+10 = 230 | 22 | 10 | Full |

**Total Parameters: 18,662**

## 💡 Architecture Decisions Explained

### Core Architecture Components

* **How many layers?** 7×Conv (all 3×3), 7×BN, 2×MaxPool, 1×GAP, 1×FC
* **MaxPooling:** 2 layers (after C3 and after C5)
* **1×1 Convolutions:** None used
* **3×3 Convolutions:** 7 (all convs are 3×3)
* **Softmax:** Not in model; `CrossEntropyLoss` applies `log_softmax` internally. Use `softmax` only for reporting probabilities

### Training Configuration Insights

* **Learning rate:** 
  - SGD base lr 0.006 with OneCycleLR(max_lr=0.02)
  - weight_decay=1e-4 
  - Adamw lr 0.001 with OneCycleLR(max_lr=0.02)
  - weight_decay=1e-4
  

* **Kernels & Channel Progression:** 14→14→16→18→20→22→22
  - Modest growth to stay within ~19k params

### Regularization & Normalization

* **BatchNorm:** After every conv, before ReLU
  - Stabilizes training
  - Allows higher learning rates
  - Adds tiny regularization effect

* **Image normalization:** `transforms.Normalize((0.1307,), (0.3081,))` for MNIST
  - Dataset-specific mean/std

* **Dropout:** None in this model
  - BN + label smoothing (0.05) sufficient for MNIST
  - Introduce small dropout (0.05–0.1) near head **only if** train≫test (clear overfitting)

### Architectural Positioning

* **Position of MaxPooling:** 
  - First Pool after 3 Conv layers
  - Second pool after next 2 conv layer

* **Transition layers (1×1 conv):** Not used here

* **"Distance" of MaxPool from prediction:** 
  - Last pool → **2 convs** → (GAP→FC)

* **"Distance" of BN from prediction:**
  - BN7 is right before GAP/FC (BN→ReLU→C7→BN→ReLU→GAP→FC)





</div>