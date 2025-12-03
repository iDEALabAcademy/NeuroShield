# pgd_attack.py
"""
PGD (Projected Gradient Descent) Adversarial Attack Implementation

PGD is a stronger iterative variant of FGSM that applies small gradient steps
repeatedly, projecting back to the epsilon-ball after each step.

Core Idea:
    for i in range(iters):
        x_t+1 = Π_ε(x_t + α * sign(∇_x Loss(x_t, y_true)))
        
Where Π_ε projects back to the valid epsilon-ball around the original image.

PGD is considered one of the strongest first-order attacks and is commonly
used for adversarial training to improve model robustness.
"""

import torch

def pgd_attack(model, images, labels, eps=0.1, alpha=0.01, iters=40, clamp_min=0, clamp_max=1):
    """
    Perform PGD adversarial attack with multiple gradient descent iterations.
    
    Args:
        model: Neural network to attack
        images: Clean input images [B, 3, H, W]
        labels: True class labels [B]
        eps: Maximum perturbation (L∞ constraint, e.g., 0.1)
        alpha: Step size for each iteration (e.g., 0.01)
        iters: Number of gradient descent iterations (e.g., 40)
        clamp_min: Minimum valid pixel value (0)
        clamp_max: Maximum valid pixel value (1)
        
    Returns:
        Adversarial images within eps-ball of original images
    """
    # Clone and save original images for projection constraint
    images = images.clone().detach().to(labels.device)
    ori_images = images.clone().detach()
    images.requires_grad = True

    # Iterative gradient-based attack
    for i in range(iters):
        # Forward pass to get predictions
        outputs = model(images)[0]  # get class_logits
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()

        # Take gradient step in direction that increases loss
        grad = images.grad.data
        images = images + alpha * grad.sign()
        
        # Projection: ensure perturbation stays within eps-ball of original
        delta = torch.clamp(images - ori_images, min=-eps, max=eps)
        
        # Clamp to valid pixel range and detach for next iteration
        images = torch.clamp(ori_images + delta, min=clamp_min, max=clamp_max).detach_()
        images.requires_grad = True

    return images
