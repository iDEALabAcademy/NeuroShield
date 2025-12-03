# Neural-Symbolic Traffic Sign Recognition with Adversarial Robustness

A hybrid neural-symbolic approach for robust traffic sign classification on the GTSRB dataset, combining deep learning with symbolic logic constraints and adversarial training.

## Overview

This project implements a novel architectural approach that integrates **symbolic knowledge** into neural network training to improve both classification accuracy and adversarial robustness. The model learns not just to classify traffic signs, but also to predict and respect symbolic attributes (shape, colors, category, icon type) through soft logic constraints.

### Main Features

- **Multi-headed Architecture**: ResNet-18 backbone with 7 prediction heads (class + 6 symbolic attributes)
- **Soft Logic Constraints**: Probabilistic penalties for violating symbolic rules (e.g., "stop signs are octagons")
- **Adversarial Training**: Toggle between FGSM or PGD attacks during training
- **Semantic Loss**: Restricts predictions to symbolically consistent classes
- **Adaptive Weighting**: Dynamically adjusts loss weights based on component difficulty and training progress
- **Comprehensive Evaluation**: Tracks both classification accuracy and symbolic rule satisfaction

## Architecture

### Model Structure

```
Input Image (64x64x3)
        ↓
   ResNet-18 Backbone
        ↓
   Feature Vector (512)
        ↓
    ┌───┴───┬───────┬────────┬─────────┬──────────┬──────────┬─────────┐
    ↓       ↓       ↓        ↓         ↓          ↓          ↓         ↓
  Class   Shape  Border   Fill      Item     Category    Icon
 (43 cls) (5)    Color    Color    Color      (7)        Type
                 (7)      (7)      (7)                   (12)
```

### Symbolic Knowledge Base

The model is trained to respect symbolic rules encoded in `constraints/rules.py`:

- **Shape Rules**: Maps each class to geometric shape (circle, triangle, octagon, diamond)
- **Color Rules**: Specifies border, fill, and item colors for each sign
- **Category Rules**: Groups signs into functional categories (speed, warning, mandatory, etc.)
- **Icon Rules**: Identifies the type of symbol (human, vehicle, arrow, number, etc.)

Example:
```python
# Stop sign (class 14)
Shape: octagon
Fill: red
Border: white  
Item: white (text)
Category: priority
Icon: text
```

### Loss Function

```
Total Loss = 0.5 × (Clean Loss + Adversarial Loss)

where each domain loss =
    CE Loss                                    # Classification
  + adaptive_λ × Universal Joint Logic Loss    # Symbolic constraints
  + λ_semantic × Semantic Loss                 # Valid class restriction

where adaptive_λ = min(1.0, (epoch+1)/3.0) × component_weights.mean()
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/GTSRB-neural-symbolic.git
cd GTSRB-neural-symbolic
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download GTSRB dataset**

Option A - Using Kaggle Hub (automatic):
```python
import kagglehub
dataset_path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
```

Option B - Manual download:
- Download from [Kaggle GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- Extract to `gtsrb-german-traffic-sign/versions/1/`

## Usage

### Training

```bash
python train.py --dataset-root "path/to/gtsrb-german-traffic-sign/versions/1"
```

**Default hyperparameters** (edit in `train.py`):
```python
# Adversarial attack configuration
use_pgd = False             # Toggle: True for PGD, False for FGSM
epsilon = 0.1               # L∞ perturbation magnitude
pgd_alpha = 0.01            # PGD step size (only used if use_pgd=True)
pgd_iters = 40              # PGD iterations (only used if use_pgd=True)

# Training configuration
num_epochs = 5              # Training epochs
lambda_semantic = 0.3       # Semantic loss weight
warmup_epochs = 0           # Epochs before enabling logic losses
```

**To switch between FGSM and PGD:**
```python
# For FGSM (faster, single-step attack)
use_pgd = False

# For PGD (stronger, iterative attack)
use_pgd = True
```

### Training Output

The training script will:
- Save model checkpoint to `checkpoints/best_model.pth` so make sure you have a checkpoints folder
- Log metrics to `logs/` directory if functions are uncommented
- Output training progress to console and `output.txt`


### Evaluation Metrics (some code may need to be uncommented to receive all of these)

The model tracks:

1. **Classification Accuracy**
   - Top-1 accuracy on clean test set
   - Top-1 accuracy on FGSM adversarial examples
   - Per-class accuracy breakdown

2. **Symbolic Accuracy** (`logs/symbolic_accuracy.json`)
   - Shape prediction accuracy
   - Color prediction accuracy (border, fill, item)
   - Category prediction accuracy
   - Icon type prediction accuracy

3. **Rule Satisfaction** (`logs/symbolic_metrics.json`)
   - Per-class logic rule satisfaction rate
   - Average logic loss per class
   - Component-wise violation analysis

## Project Structure

```
GTSRB-hybrid-backup/
├── train.py                      # Main training script
├── fgsm_attack.py               # FGSM attack implementation
├── pgd_attack.py                # PGD attack implementation
├── hybrid_evaluations.py        # Evaluation metrics
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
│
├── model/
│   ├── resnet_gtsrb.py         # Multi-headed ResNet architecture
│   └── dataset.py              # GTSRB dataset loader
│
├── constraints/
    ├── rules.py                # Symbolic knowledge base
    ├── logic_loss.py           # Individual soft logic losses
    ├── pylon_joint_rules.py   # Universal joint constraint
    └── semantic_loss.py        # Semantic loss implementation

```

This project is licensed under the MIT License - see the LICENSE file for details.

Contributions are welcome! Please feel free to submit a Pull Request.

For questions or issues, please open an issue on GitHub or contact [jschm25@uic.edu]

---

**Note from Dev**: First off this is research code. While we strive for correctness, please report any issues you encounter.

I would like to mention personally that as a young aspiring Machine Learning engineer, my work and process was not perfect. My goal was to research how Neuro Symbolic AI can help improve adversarial robustness in neural networks. My conclusion was that with my approach Neuro Symbolic alone had a negligible impact on adversarial robustness, however when coupled with an adversarial training approach it made that same adversarial training nearly 3x as effective while maintaining peak clean accuracy peformance (adversarial training tends to break/hurt clean accuracy). That being said, Neuro Symbolic AI does have the potential and can improve a models vulnerability to attacks. My architectural engineering was in no way perfect and plenty of more fine tuning could have been done to achieve better results, but my experimentation was sound and proves real results/findings. I think my basic and simple approach to developing and experimenting shows just how much more potential there is in an approach like this to obtain higher results in adversarial robustness.

I read many papers that lead me to craft my approach for this research. I started with designing man-made rules that captured known domain knowledge on traffic signs and implementing a standard adversarial training approach. My AT approach only yielded a ~10% increase in robustness. I then continued with designing pylon inspired loss functions with my rules as soft constraints. That is what led me to develop the original logic loss (logic_loss.py). Improvement in robustness was there, but negligible. Then I continued by strengthening the logic loss by combining them together to create the joint rule loss (pylon_joint_rules.py). This is where the major improvement was found and an increase in robustness was evident; AT impact nearly tripled (~10% improvement to a 30% improvement). I really enjoyed the semantic loss paper from Xu et al. so I wanted to see if a similar design would help. What I found wasn't necessarily improvement, but more so consistency in the model achieving peak performance numbers in training. These are just a few things I wanted to share.

-Jason Schmidt
