"""timm backbone factory with a forward-hook feature tap for distillation."""
import torch
import torch.nn as nn
import timm

# Preferred intermediate feature sources. Fallbacks are resolved by name below.
FEATURE_CANDIDATES = {
    "resnet50": ["layer2", "layer3", "layer1"],
    "mobilenetv2_100": ["blocks.3", "blocks.4", "blocks.2", "features.5"],
    "squeezenet1_1": ["features.8", "features.5", "features"],
    "efficientnet_lite0": ["blocks.2", "blocks.1", "blocks.0"],
}


def _find_feature_layer(backbone, arch):
    names = list(dict(backbone.named_modules()).keys())
    for candidate in FEATURE_CANDIDATES.get(arch, []):
        if candidate in names:
            return candidate
    for prefix in ("features", "blocks", "layer"):
        matches = [n for n in names if n and n.startswith(prefix)]
        if matches:
            return matches[-1]
    return None


class FeatureModel(nn.Module):
    """Returns `(logits, intermediate_features)` when asked, else just logits."""

    def __init__(self, backbone, feature_layer=None):
        super().__init__()
        self.backbone = backbone
        self.feature_layer = feature_layer
        self._feat = None
        self._handle = None
        if feature_layer:
            target = backbone.get_submodule(feature_layer)
            self._handle = target.register_forward_hook(self._capture)

    def _capture(self, module, inp, out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        self._feat = out

    def forward(self, x, return_feats=False):
        self._feat = None
        logits = self.backbone(x)
        # torchvision SqueezeNet ends with a 4D (N, C, 1, 1) head; flatten it.
        if logits.dim() > 2:
            logits = logits.flatten(1)
        if return_feats:
            return logits, self._feat
        return logits

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def build_model(arch, num_classes):
    # Clean Fire-module SqueezeNet sized for CIFAR (the installed timm has none,
    # and torchvision's 224-oriented head collapses at 32x32).
    if arch.startswith("squeezenet"):
        from .squeezenet import SqueezeNetCIFAR
        backbone = SqueezeNetCIFAR(num_classes=num_classes)
        layer = "features"
    else:
        backbone = timm.create_model(arch, pretrained=False, num_classes=num_classes)
        layer = _find_feature_layer(backbone, arch)
    return FeatureModel(backbone, layer), layer
