"""
Hybrid Evaluation Metrics for Neural-Symbolic Traffic Sign Recognition

This module evaluates how well the model satisfies symbolic logic constraints
on both clean and adversarial examples. It's a critical part of the pipeline
for measuring neural-symbolic integration.

Key Functions:
1. evaluate_stop/yield/speed_logic_on_dataset(): Per-class symbolic rule evaluation
2. evaluate_stop_component_stats(): Component-wise accuracy for adaptive weighting
3. compute_symbolic_accuracy(): Overall symbolic attribute accuracy
4. log_classwise_rule_satisfaction(): Per-class logic rule satisfaction metrics
5. *_adv versions: Same metrics on adversarial examples

These metrics help answer:
- Does the model learn correct symbolic attributes?
- Are symbolic constraints satisfied even under adversarial attack?
- Which components are hardest to learn?

Some of these functions are currently implemented and being displayed, while others not.
Most of these had to do with ablation experiments as well as understanding more about performance
"""

import torch
from torch.utils.data import DataLoader
from collections import defaultdict
import json
import os

def evaluate_stop_logic_on_dataset(model, dataloader, device):
    """
    Evaluate symbolic rule satisfaction for STOP signs (class 14).
    
    Dynamically pulls expected attribute IDs from rules.py to ensure consistency.
    """
    from constraints.rules import (
        class_shape_map, class_color_parts, class_category_map, class_icon_type_map,
        shape_to_id, color_label_to_id, category_to_id, icon_type_to_id
    )
    
    model.eval()
    total_stop = 0
    satisfied_stop = 0

    shape_correct = fill_correct = border_correct = 0
    item_color_correct = category_correct = icon_correct = 0
    
    # Get expected values from rules.py for class 14 (STOP)
    stop_class = 14
    expected_shape = shape_to_id[class_shape_map[stop_class]]  # octagon = 2
    expected_fill = color_label_to_id[class_color_parts[stop_class]['fill']]  # red = 0
    expected_border = color_label_to_id[class_color_parts[stop_class]['border']]  # white = 1
    expected_item = color_label_to_id[class_color_parts[stop_class]['item']]  # white = 1
    expected_category = category_to_id[class_category_map[stop_class]]  # priority = 5
    expected_icon = icon_type_to_id[class_icon_type_map[stop_class]]  # text = 7

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            class_logits, shape_logits, border_logits, fill_logits, item_logits, category_logits, icon_type_logits = outputs

            # Only look at ground-truth STOP labels
            is_stop = labels == stop_class
            stop_indices = torch.nonzero(is_stop).squeeze()

            if stop_indices.numel() == 0:
                continue

            class_preds = class_logits.argmax(dim=1)
            shape_preds = shape_logits.argmax(dim=1)
            fill_preds = fill_logits.argmax(dim=1)
            border_preds = border_logits.argmax(dim=1)
            item_color_preds = item_logits.argmax(dim=1)
            category_preds = category_logits.argmax(dim=1)
            icon_preds = icon_type_logits.argmax(dim=1)

            shape_correct += (shape_preds[stop_indices] == expected_shape).sum().item()
            fill_correct += (fill_preds[stop_indices] == expected_fill).sum().item()
            border_correct += (border_preds[stop_indices] == expected_border).sum().item()
            item_color_correct += (item_color_preds[stop_indices] == expected_item).sum().item()
            category_correct += (category_preds[stop_indices] == expected_category).sum().item()
            icon_correct += (icon_preds[stop_indices] == expected_icon).sum().item()

            # Joint rule: all components must match
            joint = (
                (shape_preds[stop_indices] == expected_shape) &
                (fill_preds[stop_indices] == expected_fill) &
                (border_preds[stop_indices] == expected_border) &
                (item_color_preds[stop_indices] == expected_item) &
                (category_preds[stop_indices] == expected_category) &
                (icon_preds[stop_indices] == expected_icon)
            )

            satisfied_stop += joint.sum().item()
            total_stop += is_stop.sum().item()

    print(f"\n[STOP Rule Evaluation on Full Set]")
    print(f"  Total STOP samples: {total_stop}")
    print(f"  STOP Rule Satisfaction: {satisfied_stop}/{total_stop} ({satisfied_stop/total_stop:.2%})")
    print(f"  Component Accuracies:")
    print(f"    Shape correct:       {shape_correct/total_stop:.2%}")
    print(f"    Fill color correct:  {fill_correct/total_stop:.2%}")
    print(f"    Border color correct:{border_correct/total_stop:.2%}")
    print(f"    Item color correct:  {item_color_correct/total_stop:.2%}")
    print(f"    Category correct:    {category_correct/total_stop:.2%}")
    print(f"    Icon type correct:   {icon_correct/total_stop:.2%}")

def evaluate_stop_component_stats(model, dataloader, device):
    from constraints.rules import (
        class_shape_map, class_color_parts, class_category_map, class_icon_type_map,
        shape_to_id, color_label_to_id, category_to_id, icon_type_to_id
    )
    
    model.eval()
    correct = {'shape': 0, 'fill': 0, 'border': 0, 'item': 0,
               'category': 0, 'icon': 0}
    total = 0
    
    # Get expected values from rules.py for class 14 (STOP)
    stop_class = 14
    expected_shape = shape_to_id[class_shape_map[stop_class]]
    expected_fill = color_label_to_id[class_color_parts[stop_class]['fill']]
    expected_border = color_label_to_id[class_color_parts[stop_class]['border']]
    expected_item = color_label_to_id[class_color_parts[stop_class]['item']]
    expected_category = category_to_id[class_category_map[stop_class]]
    expected_icon = icon_type_to_id[class_icon_type_map[stop_class]]

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            (
                class_logits, shape_logits, border_logits, fill_logits, item_logits, category_logits, icon_type_logits
            ) = outputs

            is_stop = labels == stop_class
            if not is_stop.any():
                continue
            stop_idx = torch.nonzero(is_stop, as_tuple=False).view(-1)
            # if stop_idx.numel() == 0:
#                 continue

            shape_preds = shape_logits.argmax(dim=1)[stop_idx]
            fill_preds = fill_logits.argmax(dim=1)[stop_idx]
            border_preds = border_logits.argmax(dim=1)[stop_idx]
            item_preds = item_logits.argmax(dim=1)[stop_idx]
            category_preds = category_logits.argmax(dim=1)[stop_idx]
            icon_preds = icon_type_logits.argmax(dim=1)[stop_idx]

            correct['shape'] += (shape_preds == expected_shape).sum().item()
            correct['fill'] += (fill_preds == expected_fill).sum().item()
            correct['border'] += (border_preds == expected_border).sum().item()
            correct['item'] += (item_preds == expected_item).sum().item()
            correct['category'] += (category_preds == expected_category).sum().item()
            correct['icon'] += (icon_preds == expected_icon).sum().item()
            total += len(stop_idx)

    if total == 0:
        # No STOP signs in this epoch, return uniform scores
        return {k: 1.0 for k in correct}
    return {k: v / (total + 1e-6) for k, v in correct.items()}

def evaluate_yield_logic_on_dataset(model, dataloader, device):
    from constraints.rules import (
        class_shape_map, class_color_parts, class_category_map, class_icon_type_map,
        shape_to_id, color_label_to_id, category_to_id, icon_type_to_id
    )
    
    model.eval()
    total_yield = 0
    satisfied_yield = 0

    shape_correct = fill_correct = border_correct = 0
    item_color_correct = category_correct = icon_correct = 0
    
    # Get expected values from rules.py for class 13 (YIELD)
    yield_class = 13
    expected_shape = shape_to_id[class_shape_map[yield_class]]  # triangle = 1
    expected_fill = color_label_to_id[class_color_parts[yield_class]['fill']]  # white = 1
    expected_border = color_label_to_id[class_color_parts[yield_class]['border']]  # red = 0
    expected_item = color_label_to_id[class_color_parts[yield_class]['item']]  # none = 5
    expected_category = category_to_id[class_category_map[yield_class]]  # priority = 5
    expected_icon = icon_type_to_id[class_icon_type_map[yield_class]]  # none = 11

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            class_logits, shape_logits, border_logits, fill_logits, item_logits, category_logits, icon_type_logits = outputs

            is_yield = labels == yield_class
            yield_indices = torch.nonzero(is_yield).squeeze()

            if yield_indices.numel() == 0:
                continue

            shape_preds = shape_logits.argmax(dim=1)
            fill_preds = fill_logits.argmax(dim=1)
            border_preds = border_logits.argmax(dim=1)
            item_color_preds = item_logits.argmax(dim=1)
            category_preds = category_logits.argmax(dim=1)
            icon_preds = icon_type_logits.argmax(dim=1)

            shape_correct += (shape_preds[yield_indices] == expected_shape).sum().item()
            fill_correct += (fill_preds[yield_indices] == expected_fill).sum().item()
            border_correct += (border_preds[yield_indices] == expected_border).sum().item()
            item_color_correct += (item_color_preds[yield_indices] == expected_item).sum().item()
            category_correct += (category_preds[yield_indices] == expected_category).sum().item()
            icon_correct += (icon_preds[yield_indices] == expected_icon).sum().item()

            joint = (
                (shape_preds[yield_indices] == expected_shape) &
                (fill_preds[yield_indices] == expected_fill) &
                (border_preds[yield_indices] == expected_border) &
                (item_color_preds[yield_indices] == expected_item) &
                (category_preds[yield_indices] == expected_category) &
                (icon_preds[yield_indices] == expected_icon)
            )

            satisfied_yield += joint.sum().item()
            total_yield += is_yield.sum().item()

    print(f"\n[YIELD Rule Evaluation on Full Set]")
    print(f"  Total YIELD samples: {total_yield}")
    print(f"  YIELD Rule Satisfaction: {satisfied_yield}/{total_yield} ({satisfied_yield/total_yield:.2%})")
    print(f"  Component Accuracies:")
    print(f"    Shape correct:       {shape_correct/total_yield:.2%}")
    print(f"    Fill color correct:  {fill_correct/total_yield:.2%}")
    print(f"    Border color correct:{border_correct/total_yield:.2%}")
    print(f"    Item color correct:  {item_color_correct/total_yield:.2%}")
    print(f"    Category correct:    {category_correct/total_yield:.2%}")
    print(f"    Icon type correct:   {icon_correct/total_yield:.2%}")

def evaluate_speed_logic_on_dataset(model, dataloader, device):
    from constraints.rules import (
        class_shape_map, class_color_parts, class_category_map, class_icon_type_map,
        shape_to_id, color_label_to_id, category_to_id, icon_type_to_id
    )
    
    model.eval()

    speed_labels = {
        0: 'Speed limit (20km/h)',
        1: 'Speed limit (30km/h)',
        2: 'Speed limit (50km/h)',
        3: 'Speed limit (60km/h)',
        4: 'Speed limit (70km/h)',
        5: 'Speed limit (80km/h)',
        6: 'End of speed limit (80km/h)',
        7: 'Speed limit (100km/h)',
        8: 'Speed limit (120km/h)',
    }

    with torch.no_grad():
        for label_id, label_name in speed_labels.items():
            total = satisfied = 0
            shape_correct = fill_correct = border_correct = 0
            item_color_correct = category_correct = icon_correct = 0
            
            # Get expected values from rules.py for this speed class
            expected_shape = shape_to_id[class_shape_map[label_id]]  # circle = 0
            expected_fill = color_label_to_id[class_color_parts[label_id]['fill']]  # white = 1
            expected_border = color_label_to_id[class_color_parts[label_id]['border']]  # red = 0 (or black for class 6)
            expected_item = color_label_to_id[class_color_parts[label_id]['item']]  # black = 4 (or black for class 6)
            expected_category = category_to_id[class_category_map[label_id]]  # speed = 0
            expected_icon = icon_type_to_id[class_icon_type_map[label_id]]  # number = 10

            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                class_logits, shape_logits, border_logits, fill_logits, item_logits, category_logits, icon_type_logits = outputs

                is_label = labels == label_id
                indices = torch.nonzero(is_label).squeeze()
                if indices.numel() == 0:
                    continue

                shape_preds = shape_logits.argmax(dim=1)
                fill_preds = fill_logits.argmax(dim=1)
                border_preds = border_logits.argmax(dim=1)
                item_color_preds = item_logits.argmax(dim=1)
                category_preds = category_logits.argmax(dim=1)
                icon_preds = icon_type_logits.argmax(dim=1)

                # Expected values for this specific speed sign (from rules.py)
                shape_correct += (shape_preds[indices] == expected_shape).sum().item()
                fill_correct += (fill_preds[indices] == expected_fill).sum().item()
                border_correct += (border_preds[indices] == expected_border).sum().item()
                item_color_correct += (item_color_preds[indices] == expected_item).sum().item()
                category_correct += (category_preds[indices] == expected_category).sum().item()
                icon_correct += (icon_preds[indices] == expected_icon).sum().item()

                joint = (
                    (shape_preds[indices] == expected_shape) &
                    (fill_preds[indices] == expected_fill) &
                    (border_preds[indices] == expected_border) &
                    (item_color_preds[indices] == expected_item) &
                    (category_preds[indices] == expected_category) &
                    (icon_preds[indices] == expected_icon)
                )

                satisfied += joint.sum().item()
                total += is_label.sum().item()

            if total > 0:
                print(f"\n[{label_name} Rule Evaluation]")
                print(f"  Total Samples: {total}")
                print(f"  Rule Satisfaction: {satisfied}/{total} ({satisfied/total:.2%})")
                print(f"  Component Accuracies:")
                print(f"    Shape correct:       {shape_correct/total:.2%}")
                print(f"    Fill color correct:  {fill_correct/total:.2%}")
                print(f"    Border color correct:{border_correct/total:.2%}")
                print(f"    Item color correct:  {item_color_correct/total:.2%}")
                print(f"    Category correct:    {category_correct/total:.2%}")
                print(f"    Icon type correct:   {icon_correct/total:.2%}")
            else:
                print(f"\n[{label_name} Rule Evaluation]")
                print(f"  No samples found in dataset.")

def compute_symbolic_accuracy(model, data_loader, device):
    from constraints.rules import (
        class_shape_map, class_color_parts, class_category_map,
        class_icon_type_map,
        shape_to_id, color_label_to_id,
        category_to_id, icon_type_to_id
    )

    model.eval()
    component_correct = defaultdict(lambda: defaultdict(int))
    component_total = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            (class_logits, shape_logits, border_logits, fill_logits, item_logits, category_logits, icon_type_logits) = outputs

            shape_preds = shape_logits.argmax(dim=1).cpu()
            border_preds = border_logits.argmax(dim=1).cpu()
            fill_preds = fill_logits.argmax(dim=1).cpu()
            item_preds = item_logits.argmax(dim=1).cpu()
            category_preds = category_logits.argmax(dim=1).cpu()
            icon_type_preds = icon_type_logits.argmax(dim=1).cpu()

            for i in range(labels.size(0)):
                label = labels[i].item()
                label_str = str(label)

                if label in class_shape_map:
                    gt = shape_to_id[class_shape_map[label]]
                    component_total['shape'][label_str] += 1
                    if shape_preds[i] == gt:
                        component_correct['shape'][label_str] += 1

                colors = class_color_parts.get(label, {})
                if 'border' in colors:
                    gt = color_label_to_id[colors['border']]
                    component_total['border'][label_str] += 1
                    if border_preds[i] == gt:
                        component_correct['border'][label_str] += 1
                if 'fill' in colors:
                    gt = color_label_to_id[colors['fill']]
                    component_total['fill'][label_str] += 1
                    if fill_preds[i] == gt:
                        component_correct['fill'][label_str] += 1
                if 'item' in colors:
                    gt = color_label_to_id[colors['item']]
                    component_total['item'][label_str] += 1
                    if item_preds[i] == gt:
                        component_correct['item'][label_str] += 1

                if label in class_category_map:
                    gt = category_to_id[class_category_map[label]]
                    component_total['category'][label_str] += 1
                    if category_preds[i] == gt:
                        component_correct['category'][label_str] += 1

                if label in class_icon_type_map:
                    gt = icon_type_to_id[class_icon_type_map[label]]
                    component_total['icon_type'][label_str] += 1
                    if icon_type_preds[i] == gt:
                        component_correct['icon_type'][label_str] += 1

    # Print summary to console
    print("\n[Symbolic Component Accuracy]")
    for comp in component_total:
        total = sum(component_total[comp].values())
        correct = sum(component_correct[comp].values())
        if total > 0:
            acc = 100.0 * correct / total
            print(f"  {comp:12s}: {acc:.2f}% ({correct}/{total})")
        else:
            print(f"  {comp:12s}: No samples")

    # Save per-class accuracy
    accuracy_result = {}
    for comp in component_total:
        accuracy_result[comp] = {}
        for cls in component_total[comp]:
            total = component_total[comp][cls]
            correct = component_correct[comp][cls]
            acc = correct / total if total > 0 else 0.0
            accuracy_result[comp][cls] = round(acc, 4)

    os.makedirs("logs", exist_ok=True)
    with open("logs/symbolic_accuracy.json", "w") as f:
        json.dump(accuracy_result, f, indent=2)

    print("✅ Saved symbolic accuracy to logs/symbolic_accuracy.json")


def log_classwise_rule_satisfaction(
    model, data_loader, device,
    output_path="symbolic_metrics.json"
):
    """
    Log per-class logic rule satisfaction rates and average logic loss.
    
    Uses the universal joint constraint to compute per-sample logic violations.
    A sample satisfies the rule if logic_loss < 0.1 (near zero).
    
    Results saved to logs/symbolic_metrics.json showing:
    - Satisfaction rate: % of samples with logic_loss < 0.1
    - Avg logic loss: Mean violation severity
    - Count: Number of samples per class
    
    Helps identify which classes struggle most with symbolic constraints.
    """
    from constraints.pylon_joint_rules import universal_joint_soft_weighted
    model.eval()
    class_counts = defaultdict(int)
    rule_satisfied_counts = defaultdict(int)
    total_logic_loss = defaultdict(float)

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            (class_logits, shape_logits, border_logits, fill_logits,
             item_logits,category_logits, icon_type_logits) = outputs

            class_probs = torch.softmax(class_logits, dim=1)
            shape_probs = torch.softmax(shape_logits, dim=1)
            border_probs = torch.softmax(border_logits, dim=1)
            fill_probs = torch.softmax(fill_logits, dim=1)
            item_probs = torch.softmax(item_logits, dim=1)
            category_probs = torch.softmax(category_logits, dim=1)
            icon_type_probs = torch.softmax(icon_type_logits, dim=1)

            batch_size = labels.size(0)
            for i in range(batch_size):
                class_id = labels[i].item()
                class_counts[class_id] += 1

                # Per-sample logic loss
                # logic_loss = universal_joint_soft_weighted(
                #     class_probs[i:i+1], shape_probs[i:i+1], fill_probs[i:i+1], border_probs[i:i+1],
                #     text_probs[i:i+1], None, category_probs[i:i+1], icon_type_probs[i:i+1],
                #     label_mask=labels[i:i+1]
                # ).item()
                logic_loss = universal_joint_soft_weighted(
                    class_probs[i:i+1], shape_probs[i:i+1], fill_probs[i:i+1], border_probs[i:i+1],
                    item_probs[i:i+1], category_probs[i:i+1], icon_type_probs[i:i+1],
                    label_mask=labels[i:i+1]
                ).item()

                total_logic_loss[class_id] += logic_loss
                if logic_loss < 0.1:  # rule satisfied (close to 0)
                    rule_satisfied_counts[class_id] += 1

    stats = {}
    for class_id in range(43):
        total = class_counts[class_id]
        satisfied = rule_satisfied_counts[class_id]
        loss = total_logic_loss[class_id]
        if total > 0:
            stats[class_id] = {
                "satisfaction_rate": satisfied / total,
                "avg_logic_loss": loss / total,
                "count": total
            }
        else:
            stats[class_id] = {
                "satisfaction_rate": None,
                "avg_logic_loss": None,
                "count": 0
            }

    print("\n[Rule Satisfaction by Class]")
    for cid, info in stats.items():
        if info["count"] > 0:
            print(f"  Class {cid:2d}: Satisfaction={info['satisfaction_rate']:.2f}, "
                  f"Logic Loss={info['avg_logic_loss']:.4f}, Count={info['count']}")

    os.makedirs("logs", exist_ok=True)
    with open(os.path.join("logs", output_path), "w") as f:
        json.dump(stats, f, indent=2)


def compute_symbolic_accuracy_adv(model, data_loader, device, attack_fn, name="fgsm"):
    import json
    import os
    from collections import defaultdict
    from constraints.rules import (
        class_shape_map, class_color_parts, class_category_map,
        class_icon_type_map,
        shape_to_id, color_label_to_id,
        category_to_id, icon_type_to_id
    )

    model.eval()
    component_correct = defaultdict(lambda: defaultdict(int))
    component_total = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            images_adv = attack_fn(model, images, labels)

            outputs = model(images_adv)
            (class_logits, shape_logits, border_logits, fill_logits, item_logits, category_logits, icon_type_logits) = outputs

            shape_preds = shape_logits.argmax(dim=1).cpu()
            border_preds = border_logits.argmax(dim=1).cpu()
            fill_preds = fill_logits.argmax(dim=1).cpu()
            item_preds = item_logits.argmax(dim=1).cpu()
            category_preds = category_logits.argmax(dim=1).cpu()
            icon_type_preds = icon_type_logits.argmax(dim=1).cpu()

            for i in range(labels.size(0)):
                label = labels[i].item()
                label_str = str(label)

                if label in class_shape_map:
                    gt = shape_to_id[class_shape_map[label]]
                    component_total['shape'][label_str] += 1
                    if shape_preds[i] == gt:
                        component_correct['shape'][label_str] += 1

                colors = class_color_parts.get(label, {})
                if 'border' in colors:
                    gt = color_label_to_id[colors['border']]
                    component_total['border'][label_str] += 1
                    if border_preds[i] == gt:
                        component_correct['border'][label_str] += 1
                if 'fill' in colors:
                    gt = color_label_to_id[colors['fill']]
                    component_total['fill'][label_str] += 1
                    if fill_preds[i] == gt:
                        component_correct['fill'][label_str] += 1
                if 'item' in colors:
                    gt = color_label_to_id[colors['item']]
                    component_total['item'][label_str] += 1
                    if item_preds[i] == gt:
                        component_correct['item'][label_str] += 1
                elif 'item' not in colors or colors['item'] not in color_label_to_id:
                    print(f"[WARN] Missing item color for class {label} (entry: {colors})")

                if label in class_category_map:
                    gt = category_to_id[class_category_map[label]]
                    component_total['category'][label_str] += 1
                    if category_preds[i] == gt:
                        component_correct['category'][label_str] += 1

                if label in class_icon_type_map:
                    gt = icon_type_to_id[class_icon_type_map[label]]
                    component_total['icon_type'][label_str] += 1
                    if icon_type_preds[i] == gt:
                        component_correct['icon_type'][label_str] += 1

    # Save results
    accuracy_result = {}
    for comp in component_total:
        accuracy_result[comp] = {}
        for cls in component_total[comp]:
            total = component_total[comp][cls]
            correct = component_correct[comp][cls]
            acc = correct / total if total > 0 else 0.0
            accuracy_result[comp][cls] = round(acc, 4)

    os.makedirs("logs", exist_ok=True)
    out_path = f"logs/symbolic_accuracy_{name}.json"
    with open(out_path, "w") as f:
        json.dump(accuracy_result, f, indent=2)
    print(f"✅ Saved adversarial symbolic accuracy to {out_path}")



def log_classwise_rule_satisfaction_adv(model, data_loader, device, attack_fn, name="fgsm"):
    from constraints.pylon_joint_rules import universal_joint_soft_weighted
    import os, json
    from collections import defaultdict

    model.eval()
    class_counts = defaultdict(int)
    rule_satisfied_counts = defaultdict(int)
    total_logic_loss = defaultdict(float)

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            images_adv = attack_fn(model, images, labels)

            outputs = model(images_adv)
            (class_logits, shape_logits, border_logits, fill_logits,
             item_logits, category_logits, icon_type_logits) = outputs

            class_probs = torch.softmax(class_logits, dim=1)
            shape_probs = torch.softmax(shape_logits, dim=1)
            border_probs = torch.softmax(border_logits, dim=1)
            fill_probs = torch.softmax(fill_logits, dim=1)
            item_probs = torch.softmax(item_logits, dim=1)
            category_probs = torch.softmax(category_logits, dim=1)
            icon_type_probs = torch.softmax(icon_type_logits, dim=1)

            for i in range(labels.size(0)):
                class_id = labels[i].item()
                class_counts[class_id] += 1

                # logic_loss = universal_joint_soft_weighted(
                #     class_probs[i:i+1], shape_probs[i:i+1], fill_probs[i:i+1], border_probs[i:i+1],
                #     text_probs[i:i+1], None, category_probs[i:i+1], icon_type_probs[i:i+1],
                #     label_mask=labels[i:i+1]
                # ).item()
                logic_loss = universal_joint_soft_weighted(
                    class_probs[i:i+1], shape_probs[i:i+1], fill_probs[i:i+1], border_probs[i:i+1],
                    item_probs[i:i+1], category_probs[i:i+1], icon_type_probs[i:i+1],
                    label_mask=labels[i:i+1]
                ).item()

                total_logic_loss[class_id] += logic_loss
                if logic_loss < 0.1:
                    rule_satisfied_counts[class_id] += 1

    stats = {}
    for class_id in range(43):
        total = class_counts[class_id]
        satisfied = rule_satisfied_counts[class_id]
        loss = total_logic_loss[class_id]
        if total > 0:
            stats[class_id] = {
                "satisfaction_rate": satisfied / total,
                "avg_logic_loss": loss / total,
                "count": total
            }
        else:
            stats[class_id] = {
                "satisfaction_rate": None,
                "avg_logic_loss": None,
                "count": 0
            }

    out_path = f"logs/symbolic_metrics_{name}.json"
    os.makedirs("logs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Saved adversarial rule satisfaction metrics to {out_path}")