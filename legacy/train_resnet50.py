"""Train ResNet50 (ImageNet-pretrained) for Thai character + numeral recognition.

Data layout: Dataset/{character,number}/<code>/<D1|D2|D3>/<image>
Each <code> folder is one class. Empty classes are skipped. Dataset/Mixed is unlabeled and ignored.

Usage:
    .venv/bin/python train_resnet50.py --epochs 15
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2 as T

ROOT = Path(__file__).parent
DATA_DIRS = [ROOT / "Dataset" / "character", ROOT / "Dataset" / "number"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def preprocess(path, size):
    """Grayscale -> crop to ink bounding box -> pad to square -> resize. Returns uint8 HxW."""
    img = Image.open(path).convert("L")
    arr = np.asarray(img)
    ink = arr < 128  # all sources are dark ink on light background
    if ink.any():
        ys, xs = np.where(ink)
        arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = arr.shape
    side = int(max(h, w) * 1.15) + 2
    canvas = np.full((side, side), 255, np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = arr
    return np.asarray(Image.fromarray(canvas).resize((size, size), Image.BILINEAR))


def scan():
    samples = []
    for d in DATA_DIRS:
        for code_dir in sorted(d.iterdir()):
            if not code_dir.is_dir() or code_dir.name.startswith("_"):
                continue  # ข้ามโฟลเดอร์พิเศษ เช่น _review ที่ clean_d1.py สร้างไว้
            for f in code_dir.rglob("*"):
                if f.suffix.lower() in EXTS:
                    source = f.parent.name if f.parent != code_dir else "?"
                    samples.append((str(f), code_dir.name, source))
    return samples


def build_cache(size, cache_path):
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        return z["x"], z["codes"], z["sources"]
    samples = scan()
    print(f"Preprocessing {len(samples)} images to {size}x{size} (one-time cache)...")
    x = np.empty((len(samples), size, size), np.uint8)
    for i, (p, _, _) in enumerate(samples):
        x[i] = preprocess(p, size)
        if i % 20000 == 0:
            print(f"  {i}/{len(samples)}")
    codes = np.array([s[1] for s in samples])
    sources = np.array([s[2] for s in samples])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, x=x, codes=codes, sources=sources)
    return x, codes, sources


class CharDataset(Dataset):
    def __init__(self, x, y, train):
        self.x, self.y = x, y
        aug = [
            T.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.85, 1.1), shear=8, fill=255),
        ] if train else []
        self.tf = T.Compose([T.ToImage(), *aug, T.ToDtype(torch.float32, scale=True)])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.tf(self.x[i][:, :, None])  # 1xHxW in [0,1]
        img = (img - 0.5) / 0.5
        return img.expand(3, -1, -1), self.y[i]


def build_model(num_classes):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, loss_sum = [], 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss_sum += F.cross_entropy(out, yb, reduction="sum").item()
        preds.append(out.argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    return loss_sum / len(preds), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="outputs/resnet50")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    x, codes, sources = build_cache(args.img_size, ROOT / "data" / f"cache_{args.img_size}.npz")
    classes = sorted({str(c) for c in codes}, key=int)
    cls_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_idx[str(c)] for c in codes], np.int64)
    print(f"{len(y)} images, {len(classes)} classes, device={device}")

    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.2, stratify=y, random_state=args.seed)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=args.seed)
    np.savez(out / "split.npz", train=tr, val=va, test=te)

    dl = lambda ids, train: DataLoader(
        CharDataset(x[ids], y[ids], train), batch_size=args.batch_size, shuffle=train,
        num_workers=args.workers, persistent_workers=args.workers > 0, drop_last=train)
    train_dl, val_dl, test_dl = dl(tr, True), dl(va, False), dl(te, False)

    model = build_model(len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl), pct_start=0.15)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    meta = {"classes": classes, "img_size": args.img_size, "arch": "resnet50",
            "normalize": {"mean": 0.5, "std": 0.5}, "preprocess": "gray, crop ink bbox, pad square x1.15"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    best_acc, history = 0.0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, correct, n = time.time(), 0.0, 0, 0
        for step, (xb, yb) in enumerate(train_dl):
            xb, yb = xb.to(device), yb.to(device)
            out_ = model(xb)
            loss = crit(out_, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            run_loss += loss.item() * len(yb); n += len(yb)
            correct += (out_.argmax(1) == yb).sum().item()
            if step % 200 == 0:
                print(f"  ep{ep} step {step}/{len(train_dl)} loss {loss.item():.4f}", flush=True)
        val_loss, val_pred = evaluate(model, val_dl, device)
        val_acc = (val_pred == y[va]).mean()
        rec = {"epoch": ep, "train_loss": run_loss / n, "train_acc": correct / n,
               "val_loss": val_loss, "val_acc": float(val_acc), "sec": time.time() - t0}
        history.append(rec)
        print(f"Epoch {ep}/{args.epochs} train_loss {rec['train_loss']:.4f} train_acc {rec['train_acc']:.4f} "
              f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} ({rec['sec']:.0f}s)", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), **meta}, out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))

    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    _, test_pred = evaluate(model, test_dl, device)
    yt = y[te]
    report = {"best_val_acc": float(best_acc), "test_acc": float((test_pred == yt).mean()),
              "test_acc_by_source": {s: float((test_pred[sources[te] == s] == yt[sources[te] == s]).mean())
                                     for s in sorted(set(sources[te]))},
              "test_acc_by_class": {classes[c]: float((test_pred[yt == c] == c).mean())
                                    for c in range(len(classes))}}
    (out / "test_report.json").write_text(json.dumps(report, indent=2))
    print(f"TEST acc {report['test_acc']:.4f}  by source {report['test_acc_by_source']}")
    worst = sorted(report["test_acc_by_class"].items(), key=lambda kv: kv[1])[:10]
    print("Worst classes:", worst)


if __name__ == "__main__":
    main()
