# src/model.py — ResNet/ConvNeXt + Vision Transformer (ViT, Swin, DeiT)
from typing import Optional

import torch.nn as nn
import torchvision.models as models

try:
    import timm  # PyTorch Image Models (wajib untuk Vision Transformer)
except ImportError as e:
    raise ImportError(
        "Package 'timm' belum terinstall. "
        "Install dulu dengan: pip install timm"
    ) from e


# =========================
#  CNN BASELINES (TA LAMA)
# =========================

def build_resnet152(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Bangun model ResNet-152 untuk klasifikasi dengan num_classes output.
    Dipakai sebagai baseline CNN (opsional).
    """
    try:
        weights = models.ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
    except Exception:
        # Kalau versi torchvision beda dan enum weights tidak ada
        weights = None

    m = models.resnet152(weights=weights)
    # Ganti fully-connected terakhir sesuai jumlah kelas
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m


def build_convnext_base(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Bangun model ConvNeXt-Base untuk klasifikasi dengan num_classes output.
    Dipakai sebagai baseline CNN (opsional).
    """
    try:
        weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
    except Exception:
        weights = None

    m = models.convnext_base(weights=weights)

    # classifier di ConvNeXt-Base adalah nn.Sequential([... , Linear])
    classifier = m.classifier              # tipe aslinya nn.Sequential
    if isinstance(classifier, nn.Sequential):
        last_layer = classifier[-1]
        if isinstance(last_layer, nn.Linear):
            in_features = last_layer.in_features
            classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            # fallback kalau struktur beda
            raise RuntimeError("Lapisan terakhir ConvNeXt bukan nn.Linear seperti yang diharapkan.")
    else:
        raise RuntimeError("m.classifier bukan nn.Sequential seperti yang diharapkan.")

    m.classifier = classifier
    return m


# =========================
#  VISION TRANSFORMERS
# =========================

def build_vit_base(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Vision Transformer (ViT-Base, patch16, 224x224).
    Paper: 'An Image is Worth 16x16 Words'.
    """
    model_name = "vit_base_patch16_224"
    m = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,   # langsung ganti head ke num_classes
    )
    return m


def build_swin_tiny(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Swin Transformer Tiny (hierarchical, shifted window).
    Paper: 'Swin Transformer: Hierarchical Vision Transformer using Shifted Windows'.
    """
    model_name = "swin_tiny_patch4_window7_224"
    m = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return m


def build_deit_small(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    DeiT Small (Data-efficient Image Transformer).
    Paper: 'Training data-efficient image transformers & distillation through attention'.
    """
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
    """
    Helper generik: buat model apa pun yang ada di timm.create_model.

    Contoh:
        build_timm_generic("vit_small_patch16_224", num_classes=4, pretrained=True)
        build_timm_generic("swin_small_patch4_window7_224", num_classes=4, pretrained=True)
    """
    m = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return m


# =========================
#  WRAPPER UTAMA
# =========================

def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    timm_name: Optional[str] = None,
) -> nn.Module:
    """
    Wrapper sederhana: pilih model berdasarkan nama string.

    Parameter:
        name: nama pendek (lebih human-friendly) untuk pilih model.
            Contoh:
                "resnet-152"
                "convnext-base"
                "vit-base"
                "swin-tiny"
                "deit-small"
                "timm"  -> pakai argumen timm_name

        num_classes: jumlah kelas output
        pretrained: apakah pakai pre-trained ImageNet
        timm_name: kalau name == "timm", pakai string nama model timm original

    Catatan:
        - Untuk tugas Vision Transformer minimal pilih 2 dari:
            "vit-base", "swin-tiny", "deit-small", atau "timm" dengan model_name ViT/Swin/DeiT lain.
        - ResNet & ConvNeXt bisa dipakai sebagai baseline tambahan (opsional).
    """
    key = (name or "").lower().replace("_", "-")

    # ===== CNN BASELINES =====
    if key in ("resnet-152", "resnet152"):
        return build_resnet152(num_classes=num_classes, pretrained=pretrained)

    if key in ("convnext-base", "convnext"):
        return build_convnext_base(num_classes=num_classes, pretrained=pretrained)

    # ===== VISION TRANSFORMERS =====
    if key in ("vit-base", "vit"):
        return build_vit_base(num_classes=num_classes, pretrained=pretrained)

    if key in ("swin-tiny", "swin"):
        return build_swin_tiny(num_classes=num_classes, pretrained=pretrained)

    if key in ("deit-small", "deit"):
        return build_deit_small(num_classes=num_classes, pretrained=pretrained)

    # ===== GENERIC TIMM MODEL =====
    if key == "timm":
        if not timm_name:
            raise ValueError("Jika name='timm', parameter timm_name wajib diisi.")
        return build_timm_generic(
            model_name=timm_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )

    raise ValueError(f"Unknown model name: {name}")
