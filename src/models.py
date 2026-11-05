"""Model definitions (timm backbones, classifier head)."""
import timm
from torch import nn


def create_model(name, num_classes, pretrained=True, drop_rate=0.2):
    """An ImageNet-pretrained timm backbone with a fresh classification head."""
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    # Start the new head at zero. EfficientNet's init scales weights by 1/sqrt(num_classes),
    # which for 4 classes gives huge first logits (loss ~4 instead of ~1.4) and the
    # first epoch then wrecks the pretrained features.
    head = model.get_classifier()
    if isinstance(head, nn.Linear):
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
    return model
