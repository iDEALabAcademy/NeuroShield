"""
Pylon-style Joint Logic Constraints

This implements a universal joint constraint that combines ALL symbolic attributes
into a single implication-based soft logic loss.

Core Logic (Implication):
    IF model predicts class C 
    THEN (shape is correct) AND (fill color is correct) AND (border is correct) AND ...
    
The loss penalizes violations of this joint implication using fuzzy logic operators.

Approach:
- Compute conjunction (AND) of all component probabilities
- Form implication: class_confidence => joint_attribute_confidence
- Weight different components based on their reliability (adaptive weighting)
"""

import torch
def safe_pow(x, p, min_val=1e-2, max_val=1.0):
    """
    Safely raise probabilities to a power, avoiding numerical issues.
    Clamps values to prevent log(0) and overflow problems.
    """
    return torch.clamp(x, min=min_val, max=max_val).pow(p)


def universal_joint_soft_weighted(
    class_probs, shape_probs, fill_probs, border_probs,
    item_probs, category_probs, icon_type_probs,
    component_weights=None,
    label_mask=None
):
    """
    Universal joint soft logic constraint with adaptive component weighting.
    
    This function implements: P(class) => P(shape) ∧ P(fill) ∧ P(border) ∧ ... 
    
    The implication is computed as: min(P(consequent) / P(antecedent), 1.0)
    Where consequent is the conjunction (product) of all component probabilities.
    
    Adaptive Weighting: Components that are harder to predict (lower accuracy) 
    receive HIGHER weights, making the model focus more on difficult attributes.
    
    Args:
        class_probs: [B, 43] probability distribution over classes
        shape_probs: [B, 5] probability distribution over shapes
        fill_probs: [B, num_colors] probability for fill color
        border_probs: [B, num_colors] probability for border color
        item_probs: [B, num_colors] probability for item/icon color
        category_probs: [B, 7] probability distribution over categories
        icon_type_probs: [B, num_icon_types] probability distribution over icon types
        component_weights: Optional dict with weights for each component
        label_mask: Optional ground truth labels to use instead of predicted class
        
    Returns:
        Scalar loss: -log(implication_strength)
        Higher loss when symbolic rules are violated
    """
    import torch
    from constraints.rules import (
        class_shape_map, class_color_parts, class_category_map, class_icon_type_map,
        shape_to_id, color_label_to_id,
        category_to_id, icon_type_to_id
    )

    batch_size = class_probs.shape[0]
    pred_class = class_probs.argmax(dim=1)
    labels = label_mask if label_mask is not None else pred_class

    eps = 1e-6

    default_weights = {
        'shape': 1.0, 'fill': 1.0, 'border': 1.0,
        'item': 1.0, 'category': 1.0, 'icon': 1.0
    }
    weights = component_weights if component_weights is not None else default_weights

    # Process ALL samples in the batch
    batch_losses = []
    
    for b in range(batch_size):
        label = int(labels[b].item())
        class_conf = class_probs[b, pred_class[b]]
        component_vals = []

        # --- SHAPE ---
        if label in class_shape_map:
            expected = shape_to_id.get(class_shape_map[label], None)
            if expected is not None:
                conf = shape_probs[b, expected]
                component_vals.append(safe_pow(conf, weights['shape']))

        # --- FILL COLOR ---
        fill_color = class_color_parts.get(label, {}).get('fill', None)
        if fill_color in color_label_to_id:
            conf = fill_probs[b, color_label_to_id[fill_color]]
            component_vals.append(safe_pow(conf, weights['fill']))

        # --- BORDER COLOR ---
        border_color = class_color_parts.get(label, {}).get('border', None)
        if border_color in color_label_to_id:
            conf = border_probs[b, color_label_to_id[border_color]]
            component_vals.append(safe_pow(conf, weights['border']))

        # --- ITEM COLOR ---
        item_color = class_color_parts.get(label, {}).get('item', None)
        if item_color in color_label_to_id:
            conf = item_probs[b, color_label_to_id[item_color]]
            component_vals.append(safe_pow(conf, weights['item']))

        # --- CATEGORY ---
        expected_cat = category_to_id.get(class_category_map.get(label, 'other'), None)
        if expected_cat is not None:
            conf = category_probs[b, expected_cat]
            component_vals.append(safe_pow(conf, weights['category']))

        # --- ICON TYPE ---
        expected_icon = icon_type_to_id.get(class_icon_type_map.get(label, 'none'), None)
        if expected_icon is not None:
            conf = icon_type_probs[b, expected_icon]
            component_vals.append(safe_pow(conf, weights['icon']))

        # Compute loss for this sample
        if len(component_vals) == 0:
            # No rules apply to this class - zero loss
            batch_losses.append(torch.tensor(0.0, device=class_probs.device))
        else:
            logic_val = torch.stack(component_vals).prod()
            implication = torch.minimum(logic_val / (class_conf + eps), torch.tensor(1.0, device=class_probs.device))
            sample_loss = -torch.log(implication + eps)
            batch_losses.append(sample_loss)
    
    # Return mean loss across all samples in batch
    if len(batch_losses) == 0:
        return torch.tensor(0.0, device=class_probs.device)
    
    return torch.stack(batch_losses).mean()
