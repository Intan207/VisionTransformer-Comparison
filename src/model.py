from typing import Optional

import torch.nn as nn
import torchvision.models as models

try:
    import timm  
except ImportError as e:
    raise ImportError(
    ) from e

def build_vit_base(num_classes: int, pretrained: bool = True) -> nn.Module:
    model_name = "vit_base_patch16_224"
    m = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,   
    )
    return m


def build_swin_tiny(num_classes: int, pretrained: bool = True) -> nn.Module:
    model_name = "swin_tiny_patch4_window7_224"
    m = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return m


def build_deit_small(num_classes: int, pretrained: bool = True) -> nn.Module:
    model_name = "deit_small_patch16_224"
    m = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return m


def build_timm_generic(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    m = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return m

def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    timm_name: Optional[str] = None,
) -> nn.Module:
   
    key = (name or "").lower().replace("_", "-")

    if key in ("vit-base", "vit"):
        return build_vit_base(num_classes=num_classes, pretrained=pretrained)

    if key in ("swin-tiny", "swin"):
        return build_swin_tiny(num_classes=num_classes, pretrained=pretrained)

    if key in ("deit-small", "deit"):
        return build_deit_small(num_classes=num_classes, pretrained=pretrained)

    if key == "timm":
        if not timm_name:
            raise ValueError("Jika name='timm', parameter timm_name wajib diisi.")
        return build_timm_generic(
            model_name=timm_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )

    raise ValueError(f"Unknown model name: {name}")
