"""
ResNet-18 with Multiple Prediction Heads for Neural-Symbolic Learning

This model predicts not just the traffic sign class, but also explicit symbolic
attributes (shape, colors, category, icon type). These auxiliary predictions enable
the application of symbolic logic constraints during training.

Architecture:
- Backbone: ResNet-18 (modified for small 64x64 images)
- Multiple heads: one for class, one for each symbolic attribute
- Each symbolic head is a small MLP for better feature transformation
"""

import torch.nn as nn
from torchvision import models
from constraints.rules import icon_type_to_id


def make_symbolic_head(output_dim):
    """
    Create a small MLP head for predicting symbolic attributes.
    
    Structure: Linear(512->128) -> BatchNorm -> ReLU -> Linear(128->output_dim)
    
    This provides better feature transformation than a single linear layer,
    helping the model learn distinct representations for different symbolic concepts.
    """
    return nn.Sequential(
        nn.Linear(512, 128),   # down-project
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, output_dim)  # final output
    )


class ResNetWithShape(nn.Module):
    """
    Multi-headed ResNet for traffic sign classification with symbolic attribute prediction.
    
    Outputs:
        1. class_logits: [B, 43] - main classification (which traffic sign)
        2. shape_logits: [B, num_shapes] - geometric shape prediction
        3. border_logits: [B, num_colors] - border color prediction
        4. fill_logits: [B, num_colors] - background/fill color prediction
        5. item_logits: [B, num_colors] - icon/text color prediction
        6. category_logits: [B, 7] - functional category prediction
        7. icon_type_logits: [B, num_icon_types] - type of icon/symbol prediction
    """
    def __init__(self, num_classes=43, num_shapes=5, num_colors=7):
        """
        Args:
            num_classes: Number of traffic sign classes (43 for GTSRB)
            num_shapes: Number of shape categories (5: circle, triangle, octagon, diamond, other)
            num_colors: Number of color categories (7: red, white, blue, yellow, black, none, other)
        """
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        # Classifier head (single-layer is fine here)
        self.classifier = nn.Linear(512, num_classes)

        # Symbolic heads using expanded MLPs
        self.shape_head = make_symbolic_head(num_shapes)
        self.border_color_head = make_symbolic_head(num_colors)
        self.fill_color_head = make_symbolic_head(num_colors)
        self.item_color_head = make_symbolic_head(num_colors)
        self.category_head = make_symbolic_head(7)  # 6 categories + 1 'other'
        self.icon_type_head = make_symbolic_head(len(icon_type_to_id))

    def forward(self, x):
        """
        Forward pass through backbone and all prediction heads.
        
        Args:
            x: [B, 3, 64, 64] input images (normalized)
            
        Returns:
            Tuple of 7 tensors (logits for class and 6 symbolic attributes)
        """
        features = self.backbone(x)  # [B, 512] feature vector

        # Main classification head
        class_logits = self.classifier(features)
        shape_logits = self.shape_head(features)
        border_logits = self.border_color_head(features)
        fill_logits = self.fill_color_head(features)
        item_logits = self.item_color_head(features)
        category_logits = self.category_head(features)
        # text_digit_logit = self.text_digit_head(features)
        icon_type_logits = self.icon_type_head(features)

        return (
            class_logits, shape_logits, border_logits, fill_logits,
            item_logits,
            category_logits, icon_type_logits
        )