"""
Neural-Symbolic Training for GTSRB Traffic Sign Recognition with Adversarial Robustness

This script implements a hybrid neural-symbolic approach to train a robust traffic sign classifier.

KEY FEATURES:
1. Multi-headed ResNet: Predicts class + symbolic attributes (shape, colors, category, icon)
2. Soft Logic Constraints: Penalizes violations of symbolic rules (e.g., "stop signs are octagons")
3. Adversarial Training: Trains on both clean and FGSM-perturbed (Or PGD) images for robustness
4. Semantic Loss: Restricts probability mass to symbolically consistent classes
5. Adaptive Weighting: Dynamically adjusts loss weights based on component difficulty

TRAINING FLOW (per batch):
    1. Clean forward pass → compute CE loss + logic losses
    2. Generate FGSM adversarial examples
    3. Adversarial forward pass → compute CE loss + logic losses
    4. Combine losses: total = 0.5 * (clean_loss + adv_loss)
    5. Backpropagate and update weights

LOSS COMPONENTS:
    - Cross-Entropy: Standard classification loss
    - Logic Losses: Penalize violations of shape/color/category rules
    - Semantic Loss: KL divergence to valid symbolic classes
    - Adaptive Lambda: Increases logic weight when model is uncertain or incorrect
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import defaultdict
import sys


class Tee(object):
    """
    Utility class to duplicate stdout/stderr to both console and file.
    Allows logging all output to output.txt while still showing in terminal.
    """
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

output_file = open('output.txt', 'w', encoding='utf-8')

sys.stdout = Tee(sys.__stdout__, output_file)
sys.stderr = Tee(sys.__stderr__, output_file)

from model.resnet_gtsrb import ResNetWithShape
from model.dataset import GTSRBDataset
from fgsm_attack import evaluate_adversarial_attack, evaluate_adversarial_attack_stop_yield_only, evaluate_adversarial_attack_speed_only
from constraints.semantic_loss import semantic_loss

from constraints.rules import class_shape_map, class_color_parts, class_category_map, class_icon_type_map

from fgsm_attack import fgsm_attack
from pgd_attack import pgd_attack

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset-root",
    default=os.environ.get(
        "DATASET_ROOT", "gtsrb-german-traffic-sign/versions/1"
    ),
    help="Path to the GTSRB dataset",
)
args = parser.parse_args()


# ---------- CONFIGURATION ----------
dataset_root = args.dataset_root
train_csv = os.path.join(dataset_root, "Train.csv")
test_csv = os.path.join(dataset_root, "Test.csv")
train_img_dir = os.path.join(dataset_root, "Train")
test_img_dir = os.path.join(dataset_root, "Test")


# Hyperparameters
lambda_logic = 0.5  # Weight for logic losses (will be adapted dynamically)
num_epochs = 1  # Set to 1 for testing
batch_size = 128

# Adversarial attack configuration
use_pgd = False  # Toggle: True for PGD, False for FGSM
epsilon = 0.1  # L∞ perturbation magnitude for both FGSM and PGD
pgd_alpha = 0.01  # PGD step size (typically eps/4 to eps/10)
pgd_iters = 40  # Number of PGD iterations (more = stronger attack)

# ---------- LABELS ----------
classes = {  # 43 traffic sign classes
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield', 14: 'Stop',
    15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited', 17: 'No entry',
    18: 'General caution', 19: 'Dangerous curve to the left', 20: 'Dangerous curve to the right',
    21: 'Double curve', 22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End of all speed and passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead',
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right',
    39: 'Keep left', 40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End no passing veh > 3.5 tons'
}

# ---------- TRANSFORM ----------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------- DATALOADERS ----------
train_dataset = GTSRBDataset(train_csv, train_img_dir, transform)
test_dataset = GTSRBDataset(test_csv, test_img_dir, transform)
# Use the batch_size hyperparameter. Set num_workers=0 for Windows compatibility
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
print(f"Training with batch_size={batch_size}")# ---------- MODEL ----------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print   (f"Using device: {device}")
# model = ResNetWithShape(num_classes=43, num_shapes=5, num_colors=7).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
# criterion = nn.CrossEntropyLoss()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ResNetWithShape(num_classes=43, num_shapes=5, num_colors=7)

# 👇 Use all available GPUs if more than 1
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    model = nn.DataParallel(model)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()


def fgsm_attack_helper(model, images, labels, epsilon):
    """
    Helper to generate FGSM adversarial examples.
    
    FGSM (Fast Gradient Sign Method):
    1. Compute loss on clean images
    2. Backpropagate to get gradient w.r.t. input image
    3. Perturb image in direction that increases loss: x_adv = x + ε * sign(∇_x Loss)
    
    Args:
        model: Neural network
        images: Clean input images [B, 3, 64, 64]
        labels: True labels [B]
        epsilon: Perturbation magnitude (typically 0.1)
        
    Returns:
        Adversarial images [B, 3, 64, 64]
    """
    with torch.enable_grad():
        images = images.clone().detach().requires_grad_(True)
        outputs = model(images)[0]  # class_logits
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        data_grad = images.grad.data
        return fgsm_attack(images, epsilon, data_grad)
    

def pgd_attack_helper(model, images, labels, eps=0.1, alpha=0.01, iters=40):
    """
    Helper to generate PGD adversarial examples (stronger attack than FGSM).
    
    PGD (Projected Gradient Descent):
    - Iteratively applies small FGSM-like steps (alpha) for multiple iterations
    - Projects back to epsilon-ball after each step
    - More effective at finding adversarial examples than single-step FGSM
    """
    with torch.enable_grad():
        return pgd_attack(model, images, labels, eps=eps, alpha=alpha, iters=iters)


def compute_entropy(probs, epsilon=1e-8):
    """
    Compute normalized entropy for class probabilities.
    
    High entropy → model is uncertain (uniform distribution)
    Low entropy → model is confident (peaked distribution)
    
    Used for adaptive weighting: apply stronger logic constraints when model is uncertain.
    
    Returns:
        Normalized entropy in [0, 1]
    """
    log_probs = torch.log(probs + epsilon)
    entropy = -torch.sum(probs * log_probs, dim=1)  # Shape: [B]
    max_entropy = torch.log(torch.tensor(probs.size(1), device=probs.device))
    return entropy / max_entropy  # Normalize to [0, 1]



def normalize_weights(component_scores, smoothing=0.01):
    """
    Convert component accuracy scores to adaptive loss weights.
    
    INVERSE weighting: Components with LOWER accuracy get HIGHER weights.
    This focuses the model's attention on harder-to-learn symbolic attributes.
    
    Example:
        shape accuracy = 0.9 → weight = 1/(0.9+0.01) = 1.10
        color accuracy = 0.6 → weight = 1/(0.6+0.01) = 1.64 (higher!)
        
    Args:
        component_scores: Dict mapping component name -> accuracy score
        smoothing: Small constant to prevent division by zero
        
    Returns:
        Dict of normalized weights that sum to 1.0
    """
    if all(v == 0 for v in component_scores.values()):
        # All scores are zero, return uniform weights
        n = len(component_scores)
        return {k: 1.0 / n for k in component_scores}
    inverted = {k: 1.0 / (v + smoothing) for k, v in component_scores.items()}
    total = sum(inverted.values())
    return {k: v / total for k, v in inverted.items()}




# ---------- TRAIN ----------
# Print adversarial training configuration
attack_method = "PGD" if use_pgd else "FGSM"
print(f"\n{'='*60}")
print(f"  Adversarial Training Configuration")
print(f"{'='*60}")
print(f"  Attack Method: {attack_method}")
print(f"  Epsilon (ε): {epsilon}")
if use_pgd:
    print(f"  PGD Alpha: {pgd_alpha}")
    print(f"  PGD Iterations: {pgd_iters}")
print(f"  Number of Epochs: {num_epochs}")
print(f"  Batch Size: {batch_size}")
print(f"{'='*60}\n")

warmup_epochs = 0  # Number of epochs to disable logic losses (allows model to learn basics first)
for epoch in range(num_epochs):
    model.train()

    # --- Tracking variables for monitoring performance ---
    stop_total_all = 0
    stop_satisfied_all = 0
    stop_shape_correct_all = 0
    stop_fill_correct_all = 0
    stop_border_correct_all = 0
    stop_item_correct_all = 0
    stop_category_correct_all = 0
    stop_icon_none_all = 0

    true_stop_total_all = 0
    true_stop_correct_class_all = 0
    true_stop_logic_satisfied_all = 0
    
    total_loss, total_ce_loss, total_logic_loss_accum, total_samples = 0.0, 0.0, 0.0, 0

    # Add accumulators for each logic loss
    sum_shape, sum_border, sum_fill, sum_item, sum_category, sum_icon_type = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    sum_universal_joint = 0.0
    joint_count = 0


    # Disable dynamic weight computation for faster training
    # if epoch >= warmup_epochs:
    #     # Compute adaptive weights based on component performance
    #     # Components that are harder to predict get higher weights
    #     print(f"Computing dynamic weights before epoch {epoch}...")
    #     component_scores = evaluate_stop_component_stats(model, train_loader, device)
    #     component_weights = normalize_weights(component_scores)
    # else:
    #     # During warmup, don't use adaptive weights
    #     component_weights = None
    
    # Use uniform weights for all components
    component_weights = None

    # Training progress tracking
    total_batches = len(train_loader)
    batch_idx = 0
    print(f"\nEpoch {epoch+1}/{num_epochs} - Total batches: {total_batches}")
    print(f"{'='*60}")

    for images, labels in train_loader:
        batch_idx += 1
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        
        # Print progress every 10 batches
        if batch_idx % 10 == 0 or batch_idx == 1:
            print(f"Batch [{batch_idx}/{total_batches}] ({100*batch_idx/total_batches:.1f}%) - Processing...")

        # ==================== CLEAN PASS ====================
        # Forward pass on original (clean) images
        optimizer.zero_grad()
        outputs_clean = model(images)
        class_logits, shape_logits, border_logits, fill_logits, item_logits, category_logits, icon_type_logits = outputs_clean
        ce_loss_clean = criterion(class_logits, labels)

        # --- Compute adaptive weights based on model confidence ---
        class_logits_scaled = class_logits * 3.0  # Temperature scaling for sharper distribution

        class_probs = F.softmax(class_logits_scaled, dim=1)

        rules_dicts = {
            'color_parts': class_color_parts
            , 'shape_map': class_shape_map,
            'category_map': class_category_map,
            'icon_type_map': class_icon_type_map
        }

        # Semantic loss: restrict probability to symbolically valid classes
        sem_loss_clean = semantic_loss(class_probs, labels, rules_dicts)

# At the start of training
        # print("Example class probs:", class_probs[0])
        # print("Example shape probs:", shape_probs[0])
        # print("Logic loss (shape):", logic_loss_shape_clean.item())

        # Adaptive lambda: higher when model is uncertain or incorrect
        entropy = compute_entropy(class_probs)
        incorrect_mask = (class_logits.argmax(1) != labels).float()
        adaptive_weights = entropy + incorrect_mask * (1.0 - entropy)


        # Convert logits to probabilities for all symbolic attributes
        shape_probs = F.softmax(shape_logits, dim=1)
        border_probs = F.softmax(border_logits, dim=1)
        fill_probs = F.softmax(fill_logits, dim=1)
        item_probs = F.softmax(item_logits, dim=1)
        icon_type_probs = F.softmax(icon_type_logits, dim=1)
        category_probs = F.softmax(category_logits, dim=1)
        


        # --- Safety check for NaN in component weights ---
        if component_weights is not None and any(v != v for v in component_weights.values()):  # NaN check
            print("NaN detected in component_weights, resetting to uniform.")
            n = len(component_weights)
            component_weights = {k: 1.0 / n for k in component_weights}


        # --- Universal joint logic loss (combines all symbolic constraints) ---
        # This implements: IF class_predicted THEN (shape AND color AND category AND icon) are correct

        from constraints.pylon_joint_rules import universal_joint_soft_weighted#stop_joint_soft_weighted, yield_joint_soft_weighted, speed_limit_joint_soft_weighted

        labels_for_mask = labels.detach()  # or labels.cpu() if needed

        logic_loss_universal_clean = universal_joint_soft_weighted(
            class_probs, shape_probs, fill_probs, border_probs,
            item_probs, category_probs, icon_type_probs,
            component_weights=component_weights,
            label_mask=labels_for_mask
        )



        #DEBUGGING
        # At the start of training
        # print("Example class probs:", class_probs[0])
        # print("Example shape probs:", shape_probs[0])
        # print("Logic loss (shape):", logic_loss_shape_clean.item())


        logic_loss_clean = logic_loss_universal_clean

        # Accumulate joint loss (already averaged over batch)
        sum_universal_joint += logic_loss_universal_clean.item() * batch_size
        joint_count += batch_size


        # --- Adversarial Example Generation (FGSM or PGD) ---
        # Generate adversarial examples to make model robust to small perturbations
        if use_pgd:
            # PGD: Stronger iterative attack with multiple gradient steps
            adv_images = pgd_attack(
                model, images, labels,
                eps=epsilon, alpha=pgd_alpha, iters=pgd_iters,
                clamp_min=0, clamp_max=1
            )
        else:
            # FGSM: Fast single-step gradient attack
            images_adv = images.clone().detach().requires_grad_(True)
            model.zero_grad()
            outputs_for_grad = model(images_adv)
            class_logits_grad = outputs_for_grad[0]
            ce_loss_for_grad = criterion(class_logits_grad, labels)
            ce_loss_for_grad.backward()
            data_grad = images_adv.grad.data
            adv_images = fgsm_attack(images, epsilon, data_grad)

        # ==================== ADVERSARIAL PASS ====================
        # Forward pass on perturbed images
        outputs_adv = model(adv_images)
        class_logits_adv, shape_logits_adv, border_logits_adv, fill_logits_adv, item_logits_adv, category_logits_adv, icon_type_logits_adv = outputs_adv
        ce_loss_adv = criterion(class_logits_adv, labels)




        class_probs_adv = torch.softmax(class_logits_adv, dim=1)

        # Semantic loss on adversarial examples
        sem_loss_adv = semantic_loss(class_probs_adv, labels, rules_dicts)

        # Convert adversarial logits to probabilities
        shape_probs_adv = torch.softmax(shape_logits_adv, dim=1)
        border_probs_adv = F.softmax(border_logits_adv, dim=1)
        fill_probs_adv = F.softmax(fill_logits_adv, dim=1)
        item_probs_adv = F.softmax(item_logits_adv, dim=1)
        icon_type_probs_adv = F.softmax(icon_type_logits_adv, dim=1)
        category_probs_adv = F.softmax(category_logits_adv, dim=1)


        # Joint logic loss on adversarial examples
        # Use ground truth labels (not predicted class) to enforce correct symbolic rules even under attack
        logic_loss_universal_adv = universal_joint_soft_weighted(
            class_probs_adv, shape_probs_adv, fill_probs_adv, border_probs_adv,
            item_probs_adv, category_probs_adv, icon_type_probs_adv,
            component_weights=component_weights,
            label_mask=labels  # Enforce rules for true class, not predicted class
        )



        logic_loss_adv = logic_loss_universal_adv

        # Accumulate joint loss (already averaged over batch)
        sum_universal_joint += logic_loss_universal_adv.item() * batch_size
        joint_count += batch_size

        # --- Compute adaptive lambda: increases over epochs and when model struggles ---
        adaptive_lambda = 0 if epoch < warmup_epochs else min(1.0, (epoch + 1) / 3.0) * adaptive_weights.mean().item()

        # --- Combine all losses ---
        lambda_semantic = 0.3  # Weight for semantic loss (tune this)
        total_clean = ce_loss_clean + adaptive_lambda * logic_loss_clean
        total_clean += lambda_semantic * sem_loss_clean
        total_adv   = ce_loss_adv   + adaptive_lambda * logic_loss_adv
        total_adv   += lambda_semantic * sem_loss_adv


        # --- Final combined loss: average of clean and adversarial ---
        # This balances learning from both clean and perturbed data
        loss = 0.5 * (total_clean + total_adv)



        if torch.isnan(loss):
            print("NaN in loss at epoch", epoch)
            continue

        loss.backward()
        
        # Gradient clipping to prevent exploding gradients (must be after backward)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()

        # --- Tracking for logging ---
        total_loss += loss.item() * batch_size
        total_ce_loss += (ce_loss_clean.item() + ce_loss_adv.item()) * 0.5 * batch_size
        total_logic_loss_accum += (logic_loss_clean.item() + logic_loss_adv.item()) * 0.5 * batch_size
        total_samples += batch_size


    if component_weights is not None:
        print("[Component Weights] (higher weight = harder to learn)")
        for k, v in component_weights.items():
            print(f"  {k}: {v:.4f}")
    print(f"[Epoch {epoch + 1}/{num_epochs}] Avg Total Loss: {total_loss / total_samples:.4f}, "
          f"Logic Loss: {total_logic_loss_accum / total_samples:.4f}")
    print(f"[Universal Joint Loss] Avg: {sum_universal_joint / joint_count:.6f}")
    
    # # Evaluate symbolic accuracy on training set
    # compute_symbolic_accuracy(model, train_loader, device)
    # log_classwise_rule_satisfaction(model, train_loader, device)
    
    # # Evaluate symbolic accuracy on adversarial examples (FGSM)
    # compute_symbolic_accuracy_adv(
    #     model, test_loader, device,
    #     attack_fn=lambda m, x, y: fgsm_attack_helper(m, x, y, epsilon),
    #     name="fgsm"
    # )
    # log_classwise_rule_satisfaction_adv(
    #     model, test_loader, device,
    #     attack_fn=lambda m, x, y: fgsm_attack_helper(m, x, y, epsilon),
    #     name="fgsm"
    # )

    # PGD
    # compute_symbolic_accuracy_adv(
    #     model, test_loader, device,
    #     attack_fn=lambda m, x, y: pgd_attack_helper(m, x, y, eps=0.1, alpha=0.01, iters=40),
    #     name="pgd"
    # )
    # log_classwise_rule_satisfaction_adv(
    #     model, test_loader, device,
    #     attack_fn=lambda m, x, y: pgd_attack_helper(m, x, y, eps=0.1, alpha=0.01, iters=40),
    #     name="pgd"
    # )

# Save the trained model
os.makedirs('checkpoints', exist_ok=True)
torch.save(model.state_dict(), 'checkpoints/best_model.pth')
print("\nModel saved to checkpoints/best_model.pth")


# ==================== EVALUATION ====================
# ---------- EVALUATE CLEAN ----------
# Test accuracy on original (non-perturbed) images
model.eval()
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        class_logits, shape_logits, border_logits, fill_logits, item_logits, category_logits, icon_type_logits = model(images)
        preds = class_logits.argmax(1)
        correct += (preds == labels).sum().item()
clean_acc = 100. * correct / len(test_dataset)
print(f"Top-1 Accuracy (clean): {clean_acc:.2f}%")

# ---------- EVALUATE ADVERSARIAL (FGSM) ----------
# Test robustness against FGSM attacks
acc_adv, adv_examples = evaluate_adversarial_attack(model, test_loader, epsilon, device, classes)
print(f"Top-1 Accuracy (FGSM, ε={epsilon}): {acc_adv * 100:.2f}%")

# Test on specific sign categories
acc, examples = evaluate_adversarial_attack_stop_yield_only(model, test_loader, epsilon, device, classes)
acc, examples = evaluate_adversarial_attack_speed_only(model, test_loader, epsilon, device, classes)

# ---------- PER-CLASS ADVERSARIAL ACCURACY ----------
# Detailed breakdown of which classes are most vulnerable to FGSM
from collections import defaultdict

# Evaluate model on FGSM adversarial examples
model.eval()
correct_counts = defaultdict(int)
total_counts = defaultdict(int)

for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)

    # Generate FGSM adversarial images
    images.requires_grad = True
    class_logits = model(images)[0]  # <-- Unpack only class logits
    loss = criterion(class_logits, labels)
    model.zero_grad()
    loss.backward()
    data_grad = images.grad.data
    perturbed_images = images + epsilon * data_grad.sign()
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    # Get predictions on adversarial examples
    class_logits_adv = model(perturbed_images)[0]
    preds = class_logits_adv.argmax(1)

    for label, pred in zip(labels, preds):
        total_counts[label.item()] += 1
        if pred == label:
            correct_counts[label.item()] += 1

print(f"\n[FGSM Per-Class Accuracy (ε={epsilon})]")
for class_id in range(43):
    total = total_counts[class_id]
    correct = correct_counts[class_id]
    if total > 0:
        acc = 100. * correct / total
        print(f"  Class {class_id:2d}: {acc:.2f}% ({correct}/{total})")
    else:
        print(f"  Class {class_id:2d}: No samples.")


#EVALUATE PGD ATTACK
from pgd_attack import pgd_attack

print("\n[PGD Evaluation]")
model.eval()
correct_pgd = 0

for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)

    # PGD needs gradient flow
    adv_images = pgd_attack(model, images, labels, eps=0.1, alpha=0.01, iters=40)

    with torch.no_grad():
        outputs = model(adv_images)[0]  # class_logits
        preds = outputs.argmax(1)
        correct_pgd += (preds == labels).sum().item()

pgd_acc = 100. * correct_pgd / len(test_dataset)
print(f"Top-1 Accuracy (PGD, ε=0.1): {pgd_acc:.2f}%")


output_file.close()