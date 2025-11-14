import torch.nn as nn
import torchvision.models as models


def build_resnet152(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Bangun model ResNet-152 untuk klasifikasi dengan num_classes output.
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


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Wrapper sederhana: pilih model berdasarkan nama string.
    """
    key = (name or "").lower().replace("_", "-")

    if key in ("resnet-152", "resnet152"):
        return build_resnet152(num_classes=num_classes, pretrained=pretrained)

    if key in ("convnext-base", "convnext"):
        return build_convnext_base(num_classes=num_classes, pretrained=pretrained)

    raise ValueError(f"Unknown model name: {name}")
