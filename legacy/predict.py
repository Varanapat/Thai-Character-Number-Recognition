"""Predict Thai character codes for images using the trained ResNet50.

Usage:
    .venv/bin/python predict.py path/to/img1.jpg path/to/img2.png [--ckpt outputs/resnet50/best.pt]
"""
import argparse

import torch
import torch.nn as nn
from torchvision import models

from train_resnet50 import preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--ckpt", default="outputs/resnet50/best.pt")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes, size = ckpt["classes"], ckpt["img_size"]
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval()

    for path in args.images:
        x = torch.from_numpy(preprocess(path, size).copy()).float().div(255)
        x = ((x - 0.5) / 0.5)[None, None].expand(1, 3, -1, -1)
        with torch.no_grad():
            prob = model(x).softmax(1)[0]
        top = prob.topk(args.topk)
        preds = ", ".join(f"{classes[i]} ({p:.2%})" for p, i in zip(top.values.tolist(), top.indices.tolist()))
        print(f"{path}: {preds}")


if __name__ == "__main__":
    main()
