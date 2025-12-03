"""
Semantic Loss (Xu et al. inspired approach)

This implements a semantic loss that restricts the probability mass to only those
classes that satisfy the same symbolic constraints as the true label.

Key Idea: If a sign has specific attributes (shape, colors, category), the model 
should only assign probability to classes with those EXACT same attributes.

Example: If true label is "speed limit 50" (red circle, white fill, number icon),
then probability should be distributed only among OTHER speed limit signs, 
not distributed to triangular warning signs.
"""

from typing import List
from constraints.rules import (
    class_shape_map, class_color_parts, class_category_map,
    class_icon_type_map
)

def get_classes_satisfying_same_rule(label: int, rules_dicts: dict) -> List[int]:
    """
    Find all classes that share the same symbolic attributes as the given label.
    
    This creates equivalence classes: groups of traffic signs that are symbolically
    identical in their structure (shape, colors, category, icon type).
    
    Args:
        label: The true class label
        rules_dicts: Dictionary containing all symbolic rule mappings
        
    Returns:
        List of class indices that match ALL symbolic attributes of label
    """
    """Return all class indices that satisfy the same symbolic constraints as `label`."""
    matching_classes = []
    ref = {
        'shape': class_shape_map.get(label, None),
        'border': rules_dicts['color_parts'].get(label, {}).get('border', None),
        'fill': rules_dicts['color_parts'].get(label, {}).get('fill', None),
        'item': rules_dicts['color_parts'].get(label, {}).get('item', None),
        'category': class_category_map.get(label, None),
        'icon_type': class_icon_type_map.get(label, None),
    }

    for cls in range(43):  # Number of GTSRB classes
        same = True
        if class_shape_map.get(cls, None) != ref['shape']:
            same = False
        elif rules_dicts['color_parts'].get(cls, {}).get('border', None) != ref['border']:
            same = False
        elif rules_dicts['color_parts'].get(cls, {}).get('fill', None) != ref['fill']:
            same = False
        elif rules_dicts['color_parts'].get(cls, {}).get('item', None) != ref['item']:
            same = False
        elif class_category_map.get(cls, None) != ref['category']:
            same = False
        elif class_icon_type_map.get(cls, None) != ref['icon_type']:
            same = False

        if same:
            matching_classes.append(cls)

    return matching_classes


import torch
import torch.nn.functional as F

def semantic_loss(class_probs: torch.Tensor, labels: torch.Tensor, rules_dicts: dict) -> torch.Tensor:
    """
    Compute semantic loss by constraining probability to symbolically valid classes.
    
    This uses KL divergence to push the model's predictions toward a target distribution
    that only has mass on classes matching the true label's symbolic attributes.
    
    Example: True label = "speed 30" (class 1)
             Valid classes = all speed limits (classes 0-8)
             Target distribution = uniform over classes 0-8, zero elsewhere
             Model is penalized for putting probability on triangular warning signs
    
    Args:
        class_probs: [B, 43] predicted probability distribution over classes
        labels: [B] ground-truth class indices
        rules_dicts: Dictionary containing symbolic rule mappings
        
    Returns:
        Scalar KL divergence loss
    """
    """
    Compute semantic loss by penalizing probability mass outside valid symbolic assignments.
    
    class_probs: [B, 43] predicted probability distribution over classes
    labels: [B] ground-truth class indices
    """
    B, C = class_probs.size()
    device = class_probs.device
    targets = torch.zeros_like(class_probs)  # [B, 43]

    for i in range(B):
        valid_classes = get_classes_satisfying_same_rule(labels[i].item(), rules_dicts)
        if len(valid_classes) == 0:
            # fallback: uniform if no satisfying assignments (shouldn't happen if rules are exhaustive)
            targets[i] = torch.ones(C, device=device) / C
        else:
            targets[i, valid_classes] = 1.0 / len(valid_classes)

    # KL divergence: KL(targets || class_probs)
    # Add epsilon to avoid log(0)
    epsilon = 1e-8
    class_probs = class_probs + epsilon
    targets = targets + epsilon
    loss = F.kl_div(class_probs.log(), targets, reduction='batchmean')
    return loss
