"""Model zoo สำหรับ Experiment A — ทุกตัวรับ input 3 channel และคืน logits เท่าจำนวนคลาส

    custom_cnn          baseline เขียนเอง (Conv-BN-ReLU + GAP)
    resnet18 / resnet50
    efficientnet_b0
    densenet121
    mobilenet_v3_small
"""
import torch
import torch.nn as nn
from torchvision import models

MODEL_NAMES = ("custom_cnn", "resnet18", "resnet50", "efficientnet_b0",
               "densenet121", "mobilenet_v3_small")


class CustomCNN(nn.Module):
    """Baseline ตาม README ข้อ 7: (Conv-BN-ReLU) × convs_per_stage + Pool ต่อกันหลาย stage
    ปิดท้ายด้วย Global Average Pooling แล้ว Classifier

    ปรับความลึกได้จาก config เช่น
        model_kwargs: {custom_cnn: {widths: [32, 64], convs_per_stage: 1}}
    ซึ่งจะได้โครงตรงตามแผนภาพใน README เป๊ะ (Conv-BN-ReLU-Pool -> Conv-Pool -> GAP -> Classifier)
    ค่าเริ่มต้น 4 stage × 2 conv ลึกกว่าแผนภาพ เพราะงานนี้มี 81 คลาสและตัวอักษรหลายคู่คล้ายกันมาก
    """

    def __init__(self, num_classes, widths=(32, 64, 128, 256), in_ch=3, dropout=0.2,
                 convs_per_stage=2):
        super().__init__()
        layers, c = [], in_ch
        for w in widths:
            for _ in range(convs_per_stage):
                layers += [nn.Conv2d(c, w, 3, padding=1, bias=False), nn.BatchNorm2d(w),
                           nn.ReLU(True)]
                c = w
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(c, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name, num_classes, pretrained=True, **kwargs):
    """สร้างโมเดลพร้อมเปลี่ยนหัวให้ตรงจำนวนคลาส (custom_cnn ไม่มี pretrained weight)

    kwargs ส่งต่อให้ CustomCNN เท่านั้น เช่น widths, convs_per_stage, dropout
    """
    if name == "custom_cnn":
        return CustomCNN(num_classes, **kwargs)

    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "densenet121":
        m = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"ไม่รู้จักโมเดล: {name} (มีให้เลือก {MODEL_NAMES})")
    return m


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def inference_time_ms(model, device, img_size=96, batch_size=1, repeats=50, warmup=10):
    """เวลาเฉลี่ยต่อ 1 ภาพ (มิลลิวินาที) — Efficiency metric ตาม README ข้อ 16"""
    model.eval().to(device)
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(repeats):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - t0) * 1000 / (repeats * batch_size)
