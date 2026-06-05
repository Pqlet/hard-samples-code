from __future__ import annotations

from typing import Literal

import torch.nn as nn
from torchvision import models


ModelName = Literal["resnet18", "resnet50"]


def create_model(model_name: ModelName, num_classes: int, pretrained: bool = False) -> nn.Module:
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported model {model_name!r}")

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
