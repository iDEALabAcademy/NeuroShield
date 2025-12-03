"""
FGSM (Fast Gradient Sign Method) Adversarial Attack Implementation

FGSM is a simple white-box adversarial attack that crafts perturbed images
by adding small perturbations in the direction of the gradient of the loss function.

Core Idea: 
    x_adversarial = x_clean + ε * sign(∇_x Loss(x, y_true))
    
The attack moves the input in the direction that maximizes the classification loss,
making the model more likely to misclassify.
"""

import torch
import torch.nn.functional as F

def fgsm_attack(image, epsilon, data_grad):
    """
    Generate FGSM adversarial perturbation.
    
    Args:
        image: Clean input image [B, 3, H, W]
        epsilon: Perturbation magnitude (e.g., 0.1 for 10% perturbation)
        data_grad: Gradient of loss w.r.t. input image ∇_x Loss
        
    Returns:
        Perturbed image clamped to [0, 1] range
    """
    # Get sign of gradient (direction that increases loss)
    sign_data_grad = data_grad.sign()
    # Add perturbation: move in direction that increases loss
    perturbed_image = image + epsilon * sign_data_grad
    # Clamp to valid image range [0, 1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def evaluate_adversarial_attack(model, data_loader, epsilon, device, class_names, joint_rule_fn=None, semantic_loss_fn=None, rules_dicts=None):
    """
    Evaluate model robustness against FGSM attacks on entire test set.
    
    Generates adversarial examples for all test images and measures:
    1. Classification accuracy on adversarial examples
    2. Joint logic rule violations (if joint_rule_fn provided)
    3. Semantic loss (if semantic_loss_fn provided)
    
    Args:
        model: Neural network model
        data_loader: Test data loader
        epsilon: FGSM perturbation magnitude
        device: cuda or cpu
        class_names: Dict mapping class id to name
        joint_rule_fn: Optional function to compute joint logic violations
        semantic_loss_fn: Optional function to compute semantic loss
        rules_dicts: Optional symbolic rules dictionary
        
    Returns:
        (accuracy, adversarial_examples): Tuple of float accuracy and list of example images
    """
    model.eval()
    correct = 0
    adv_examples = []
    total_samples = 0
    sum_joint = 0.0
    sum_semantic = 0.0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        outputs = model(images)
        class_logits = outputs[0]
        loss = F.cross_entropy(class_logits, labels)
        model.zero_grad()
        loss.backward()

        data_grad = images.grad.data
        perturbed_images = fgsm_attack(images, epsilon, data_grad)

        adv_outputs = model(perturbed_images)
        class_logits = adv_outputs[0]
        final_pred = class_logits.max(1, keepdim=True)[1]
        correct += final_pred.eq(labels.view_as(final_pred)).sum().item()
        batch_size = images.size(0)
        total_samples += batch_size

        # Joint rule violation (if provided)
        if joint_rule_fn is not None:
            # Unpack all outputs as needed for your joint rule
            sum_joint += joint_rule_fn(*[F.softmax(o, dim=1) for o in adv_outputs], label_mask=labels).item() * batch_size

        # Semantic loss (if provided)
        if semantic_loss_fn is not None and rules_dicts is not None:
            class_probs = F.softmax(class_logits, dim=1)
            sum_semantic += semantic_loss_fn(class_probs, labels, rules_dicts).item() * batch_size

        if len(adv_examples) < 5:
            adv_img = perturbed_images[0].detach().cpu().numpy().transpose(1, 2, 0)
            orig = class_logits.argmax(1)[0].item()
            adv = final_pred[0].item()
            adv_examples.append((orig, adv, adv_img))

    print(f"[FGSM] Top-1 Accuracy: {correct / total_samples:.4f}")
    if joint_rule_fn is not None:
        print(f"[FGSM] Avg Joint Rule Violation: {sum_joint / total_samples:.4f}")
    if semantic_loss_fn is not None:
        print(f"[FGSM] Avg Semantic Loss: {sum_semantic / total_samples:.4f}")

    final_acc = correct / len(data_loader.dataset)
    return final_acc, adv_examples


def evaluate_adversarial_attack_stop_yield_only(
    model, data_loader, epsilon, device, class_names,
    joint_rule_fn=None, semantic_loss_fn=None, rules_dicts=None
):
    """
    Evaluate FGSM robustness for STOP (class 14) and YIELD (class 13) signs only.
    
    These critical traffic signs are particularly important for safety,
    so we evaluate them separately to ensure they're robust to attacks.
    
    Tracks:
    - Classification accuracy on adversarial STOP/YIELD examples
    - Symbolic rule violations (shape, color, category)
    - Semantic loss values

    This was used for ablation experiments to understand how different signs were more
    vulnerable to adversarial attacks than others
    """
    model.eval()
    correct = 0
    adv_examples = []

    STOP_ID = 14
    YIELD_ID = 13

    total_samples = 0
    sum_joint = 0.0
    sum_semantic = 0.0

    for images, labels in data_loader:
        # Filter only STOP and YIELD samples
        mask = (labels == STOP_ID) | (labels == YIELD_ID)
        if mask.sum() == 0:
            continue

        images, labels = images[mask].to(device), labels[mask].to(device)
        images.requires_grad = True

        # Clean forward and gradient
        outputs = model(images)
        class_logits = outputs[0]
        loss = F.cross_entropy(class_logits, labels)
        model.zero_grad()
        loss.backward()

        # Generate adversarial examples
        data_grad = images.grad.data
        perturbed_images = fgsm_attack(images, epsilon, data_grad)

        # Forward on adversarial samples
        adv_outputs = model(perturbed_images)
        class_logits = adv_outputs[0]
        final_pred = class_logits.max(1, keepdim=True)[1]
        correct += final_pred.eq(labels.view_as(final_pred)).sum().item()

        batch_size = images.size(0)
        total_samples += batch_size

        # Joint rule violation (if provided)
        if joint_rule_fn is not None:
            sum_joint += joint_rule_fn(*[F.softmax(o, dim=1) for o in adv_outputs], label_mask=labels).item() * batch_size

        # Semantic loss (if provided)
        if semantic_loss_fn is not None and rules_dicts is not None:
            class_probs = F.softmax(class_logits, dim=1)
            sum_semantic += semantic_loss_fn(class_probs, labels, rules_dicts).item() * batch_size

        # Visual examples
        if len(adv_examples) < 5:
            adv_img = perturbed_images[0].detach().cpu().numpy().transpose(1, 2, 0)
            orig = class_logits.argmax(1)[0].item()
            adv = final_pred[0].item()
            adv_examples.append((orig, adv, adv_img))

    if total_samples == 0:
        print("No STOP or YIELD samples found in dataset.")
        return 0.0, []

    print(f"[FGSM - STOP & YIELD] Top-1 Accuracy: {correct / total_samples:.4f}")
    if joint_rule_fn is not None:
        print(f"[FGSM - STOP & YIELD] Avg Joint Rule Violation: {sum_joint / total_samples:.4f}")
    if semantic_loss_fn is not None:
        print(f"[FGSM - STOP & YIELD] Avg Semantic Loss: {sum_semantic / total_samples:.4f}")

    final_acc = correct / total_samples
    print(f"Top-1 Accuracy on STOP & YIELD (FGSM, ε={epsilon}): {final_acc * 100:.2f}%")
    return final_acc, adv_examples


def evaluate_adversarial_attack_speed_only(
    model, data_loader, epsilon, device, class_names,
    joint_rule_fn=None, semantic_loss_fn=None, rules_dicts=None
):
    """
    Evaluate FGSM robustness for speed limit signs (classes 0-8) only.
    
    Speed limit signs all share similar symbolic attributes (circular shape,
    red border, white fill, number icon), so we evaluate them as a group
    to assess how well symbolic constraints help with similar classes.
    
    Tracks:
    - Classification accuracy on adversarial speed sign examples
    - Symbolic rule violations
    - Semantic loss values
    """
    model.eval()
    correct = 0
    adv_examples = []

    SPEED_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8}

    total_samples = 0
    sum_joint = 0.0
    sum_semantic = 0.0

    for images, labels in data_loader:
        # Filter only SPEED samples
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for sid in SPEED_IDS:
            mask |= (labels == sid)
        if mask.sum() == 0:
            continue

        images, labels = images[mask].to(device), labels[mask].to(device)
        images.requires_grad = True

        # Clean forward and gradient
        outputs = model(images)
        class_logits = outputs[0]
        loss = F.cross_entropy(class_logits, labels)
        model.zero_grad()
        loss.backward()

        # Generate adversarial examples
        data_grad = images.grad.data
        perturbed_images = fgsm_attack(images, epsilon, data_grad)

        # Forward on adversarial samples
        adv_outputs = model(perturbed_images)
        class_logits = adv_outputs[0]
        final_pred = class_logits.max(1, keepdim=True)[1]
        correct += final_pred.eq(labels.view_as(final_pred)).sum().item()

        batch_size = images.size(0)
        total_samples += batch_size

        # Joint rule violation (if provided)
        if joint_rule_fn is not None:
            sum_joint += joint_rule_fn(*[F.softmax(o, dim=1) for o in adv_outputs], label_mask=labels).item() * batch_size

        # Semantic loss (if provided)
        if semantic_loss_fn is not None and rules_dicts is not None:
            class_probs = F.softmax(class_logits, dim=1)
            sum_semantic += semantic_loss_fn(class_probs, labels, rules_dicts).item() * batch_size

        # Visual examples
        if len(adv_examples) < 5:
            adv_img = perturbed_images[0].detach().cpu().numpy().transpose(1, 2, 0)
            orig = class_logits.argmax(1)[0].item()
            adv = final_pred[0].item()
            adv_examples.append((orig, adv, adv_img))

    if total_samples == 0:
        print("No SPEED samples found in dataset.")
        return 0.0, []

    print(f"[FGSM - SPEED] Top-1 Accuracy: {correct / total_samples:.4f}")
    if joint_rule_fn is not None:
        print(f"[FGSM - SPEED] Avg Joint Rule Violation: {sum_joint / total_samples:.4f}")
    if semantic_loss_fn is not None:
        print(f"[FGSM - SPEED] Avg Semantic Loss: {sum_semantic / total_samples:.4f}")

    final_acc = correct / total_samples
    print(f"Top-1 Accuracy on SPEED (FGSM, ε={epsilon}): {final_acc * 100:.2f}%")
    return final_acc, adv_examples