# model.py — definisi model pretrained untuk transfer learning
# Catatan:
# - Dua arsitektur: ResNet-152 & ConvNeXt-Base (pretrained ImageNet).
# - Hanya mengganti layer klasifikasi terakhir agar sesuai jumlah kelas dataset.
# - Tidak mengubah fungsionalitas asli; ini versi yang lebih rapi & terdokumentasi.

from __future__ import annotations

from torchvision import models
import torch
import torch.nn as nn


def build_resnet152(num_classes: int, pretrained: bool = True):
    """
    Membangun ResNet-152 untuk klasifikasi multi-kelas.

    Args:
        num_classes : jumlah kelas pada dataset (output logits).
        pretrained  : True -> pakai bobot ImageNet (transfer learning).

    Proses:
      1) Ambil backbone resnet152 dari torchvision.
      2) Jika pretrained=True, gunakan weights ImageNet default.
      3) Ganti layer fully-connected terakhir (m.fc) agar output = num_classes.
    """
    weights = models.ResNet152_Weights.DEFAULT if pretrained else None
    m = models.resnet152(weights=weights)
    # m.fc input dim mengikuti backbone; kita ganti head-nya saja
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build_convnext_base(num_classes: int, pretrained: bool = True):
    """
    Membangun ConvNeXt-Base untuk klasifikasi multi-kelas.

    Args:
        num_classes : jumlah kelas pada dataset (output logits).
        pretrained  : True -> pakai bobot ImageNet (transfer learning).

    Proses:
      1) Ambil backbone convnext_base dari torchvision.
      2) Jika pretrained=True, gunakan weights ImageNet default.
      3) Head ConvNeXt ada di m.classifier; ganti linear terakhir agar output = num_classes.
    """
    weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
    m = models.convnext_base(weights=weights)
    # Struktur head convnext_base: [LayerNorm2d, Flatten, Linear]
    # dapatkan in_features dengan aman; fallback ke bentuk weight jika stub typing tidak tersedia
    # gunakan children() untuk menghindari akses __getitem__ pada Module yang tidak ter-typing
    classifier_children = list(m.classifier.children())
    if len(classifier_children) == 0:
        raise ValueError("ConvNeXt classifier is empty; cannot replace head")
    last_layer = classifier_children[-1]
    if hasattr(last_layer, "in_features"):
        # Ensure we coerce to a plain int for the nn.Linear constructor
        in_f = int(getattr(last_layer, "in_features"))
    elif hasattr(last_layer, "weight") and getattr(last_layer, "weight") is not None:
        weight = getattr(last_layer, "weight")
        # If weight is a Tensor (or Parameter), read shape directly; otherwise try to get .shape if available
        if isinstance(weight, torch.Tensor):
            in_f = int(weight.shape[1])
        elif hasattr(weight, "shape"):
            in_f = int(tuple(weight.shape)[1])
        else:
            raise ValueError("Cannot determine weight shape for ConvNeXt classifier head")
    else:
        # Fallback: search backward for a module that exposes in_features
        # Fallback: search backward for a module that exposes in_features
        in_f = None
        for module in reversed(classifier_children):
            if hasattr(module, "in_features"):
                in_f = int(getattr(module, "in_features"))
                break
        if in_f is None:
            raise ValueError("Cannot determine in_features for ConvNeXt classifier head")
    # Final ensure in_f is an int (type-checkers may still infer a union)
    try:
        in_f = int(in_f)
    except Exception as exc:
        raise ValueError(f"Determined in_features is not convertible to int: {in_f!r}") from exc
    classifier_children[-1] = nn.Linear(in_f, num_classes)
    m.classifier = nn.Sequential(*classifier_children)
    return m
