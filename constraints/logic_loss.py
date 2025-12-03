"""
Logic Loss Functions for Neural-Symbolic GTSRB Training (LEGACY)

NOTE: These functions are LEGACY baseline logic losses not used in main experiments.
All logic constraints in the current training pipeline are handled by the universal 
joint constraint in pylon_joint_rules.py, which combines all symbolic attributes 
using fuzzy logic implications.

These individual loss functions are conceptually correct soft penalties of the form:
    Loss = 1.0 - P(expected_attribute | predicted_class)

They are retained here for:
- Baseline comparisons against the universal joint constraint
- Ablation studies testing individual vs. joint constraints
- Understanding the evolution of the architecture

Core Approach: Soft Probabilistic Constraints
- Instead of hard binary checks, we use probability distributions
- Loss = 1.0 - probability_of_expected_attribute
- Lower loss when model predictions align with symbolic rules
- Each function enforces one symbolic rule independently

Current Architecture (Main Experiments):
- Uses universal_joint_soft_weighted() from pylon_joint_rules.py
- Combines shape, color (fill/border/item), category, and icon_type in one constraint
- Based on fuzzy logic implication: P(A→B) = max(1-P(A), P(B))
- Automatically handles batch processing and label masking for adversarial training
"""

import torch
import torch.nn.functional as F



def logic_consistency_loss(class_logits, shape_logits, class_shape_map, shape_to_id, device='cpu'):
    """
    [LEGACY - NOT USED IN MAIN EXPERIMENTS]
    
    Penalize shape predictions that don't match the expected shape for predicted class.
    Superseded by universal_joint_soft_weighted() in main training pipeline.
    
    Example: If model predicts "stop sign", shape should be "octagon"
    
    Args:
        class_logits: [B, 43] raw logits for traffic sign classes
        shape_logits: [B, 5] raw logits for shape predictions (circle, triangle, octagon, diamond, other)
        class_shape_map: dict mapping class_id -> expected shape name (from rules.py)
        shape_to_id: dict mapping shape name -> shape id (from rules.py)
        
    Returns:
        Scalar loss (higher when shape doesn't match expected)
        
    Note: Uses 'other' (id=4) as fallback for undefined shapes
    """
    batch_size = class_logits.size(0)
    class_probs = F.softmax(class_logits, dim=1)
    shape_probs = F.softmax(shape_logits, dim=1)

    expected_shape_ids = torch.tensor([
        shape_to_id.get(class_shape_map.get(int(torch.argmax(c).item()), 'other'), shape_to_id['other'])
        for c in class_logits
    ], dtype=torch.long, device=device)

    # Now we use soft probabilities instead of argmax mismatch
    loss = 1.0 - shape_probs[torch.arange(batch_size), expected_shape_ids]


    return loss.mean()


def logic_color_part_loss(class_logits, part_logits, class_color_parts, color_label_to_id, part='fill', device='cpu'):
    """
    [LEGACY - NOT USED IN MAIN EXPERIMENTS]
    
    Penalize color predictions for a specific sign part (fill/border/item) that don't match
    the expected color for the predicted class.
    Superseded by universal_joint_soft_weighted() in main training pipeline.
    
    Example: If model predicts "stop sign", fill should be "red"
    
    Args:
        class_logits: [B, 43] raw logits for traffic sign classes
        part_logits: [B, 7] raw logits for color of specific part (red, white, blue, yellow, black, none, other)
        class_color_parts: dict mapping class_id -> {part: color_name} (from rules.py)
        color_label_to_id: dict mapping color name -> color id (from rules.py)
        part: which part we're checking ('fill', 'border', or 'item')
        
    Returns:
        Scalar loss (higher when color doesn't match expected)
        
    Note: Uses 'other' (id=6) as fallback for undefined colors
    """
    batch_size = class_logits.size(0)
    class_probs = F.softmax(class_logits, dim=1)
    part_probs = F.softmax(part_logits, dim=1)

    expected_color_ids = torch.tensor([
        color_label_to_id.get(class_color_parts.get(int(torch.argmax(c).item()), {}).get(part, 'other'), color_label_to_id['other'])
        for c in class_logits
    ], dtype=torch.long, device=device)

    # Soft constraint: encourage part_logits to place higher prob on expected color
    loss = 1.0 - part_probs[torch.arange(batch_size), expected_color_ids]
    return loss.mean()


def logic_category_loss(class_logits, category_logits, class_category_map, category_to_id, device='cpu'):
    """
    [LEGACY - NOT USED IN MAIN EXPERIMENTS]
    
    Penalize category predictions that don't match the expected category for predicted class.
    Superseded by universal_joint_soft_weighted() in main training pipeline.
    
    Example: If model predicts "speed limit 50", category should be "speed"
    
    Args:
        class_logits: [B, 43] raw logits for traffic sign classes
        category_logits: [B, 7] raw logits for sign category (speed, prohibition, mandatory, warning, information, priority, other)
        class_category_map: dict mapping class_id -> expected category name (from rules.py)
        category_to_id: dict mapping category name -> category id (from rules.py)
        
    Returns:
        Scalar loss (higher when category doesn't match expected)
        
    Note: Uses 'other' (id=6) as fallback for undefined categories
    """
    batch_size = class_logits.size(0)
    class_probs = F.softmax(class_logits, dim=1)
    category_probs = F.softmax(category_logits, dim=1)

    expected_category_ids = torch.tensor([
        category_to_id.get(class_category_map.get(int(torch.argmax(c).item()), 'other'), category_to_id['other'])
        for c in class_logits
    ], dtype=torch.long, device=device)

    loss = 1.0 - category_probs[torch.arange(batch_size), expected_category_ids]
    return loss.mean()


def logic_icon_type_loss(class_logits, icon_type_logits, class_icon_type_map, icon_type_to_id, device='cpu'):
    """
    [LEGACY - NOT USED IN MAIN EXPERIMENTS]
    
    Penalize icon type predictions that don't match the expected icon for predicted class.
    Superseded by universal_joint_soft_weighted() in main training pipeline.
    
    Example: If model predicts "pedestrian crossing", icon type should be "human"
    
    Args:
        class_logits: [B, 43] raw logits for traffic sign classes
        icon_type_logits: [B, num_icon_types] raw logits for icon type
        class_icon_type_map: dict mapping class_id -> expected icon type name (from rules.py)
        icon_type_to_id: dict mapping icon type name -> icon type id (from rules.py)
        
    Returns:
        Scalar loss (higher when icon type doesn't match expected)
        
    Note: Uses 'none' as fallback for signs without icons
    """
    class_preds = torch.argmax(class_logits, dim=1)
    expected_icon_ids = torch.tensor([
        icon_type_to_id.get(class_icon_type_map.get(int(c.item()), 'none'), icon_type_to_id['none'])
        for c in class_preds
    ], dtype=torch.long, device=device)

    probs = F.softmax(icon_type_logits, dim=1)
    loss = 1.0 - probs[torch.arange(class_preds.size(0)), expected_icon_ids]
    return loss.mean()
