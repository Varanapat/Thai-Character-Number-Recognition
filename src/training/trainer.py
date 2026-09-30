"""Training loop กลางที่ทุก experiment เรียกใช้

รองรับ class imbalance 2 วิธีตาม README ข้อ 5
    imbalance: none | class_weight | weighted_sampling
และบันทึกทุกอย่างที่ต้องใช้ reproduce ตาม README ข้อ 21
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.augmentation.policies import get_policy, to_input
from src.models.factory import build_model, count_params
from src.utils.common import get_device, save_json, set_seed


def _batches(x, y, ids, batch_size, shuffle, device, policy=None, rng=None):
    ids = np.asarray(ids)
    if shuffle:
        ids = ids[(rng or np.random.default_rng()).permutation(len(ids))]
    n = len(ids) // batch_size * batch_size if shuffle else len(ids)  # drop_last ตอนเทรน
    for s in range(0, max(n, 0), batch_size):
        b = ids[s:s + batch_size]
        xb = to_input(torch.from_numpy(x[b]), device, policy)
        yield xb, torch.from_numpy(y[b]).to(device)


def _sampling_probs(y, ids, power):
    """น้ำหนักการสุ่มสำหรับ weighted sampling: power=0 ไม่ปรับ, 1 = ทุกคลาสเท่ากัน"""
    yy = y[ids]
    cnt = np.bincount(yy, minlength=yy.max() + 1).astype(float)
    w = (cnt.max() / np.maximum(cnt[yy], 1)) ** power
    return w / w.sum()


@torch.no_grad()
def predict(model, data, ids, device, batch_size=256, return_probs=True):
    """คืน (pred, probs) ของ ids ที่ให้มา — ไม่มี augmentation"""
    model.eval()
    out = []
    for xb, _ in _batches(data.x, data.y, ids, batch_size, False, device):
        out.append(model(xb).softmax(1).cpu().numpy())
    probs = np.concatenate(out) if out else np.zeros((0, data.n_classes), np.float32)
    return probs.argmax(1), (probs if return_probs else None)


def train_model(data, split, cfg, model_name=None, aug=None, out_dir=None, verbose=True):
    """เทรน 1 run แล้วคืน dict สรุปผล (history + path ของ checkpoint)

    cfg ใช้คีย์: epochs, batch_size, lr, weight_decay, label_smoothing, img_size,
                 imbalance, sampling_power, pretrained, seed, device, early_stop_patience
    """
    model_name = model_name or cfg["model"]
    aug = aug or cfg.get("augmentation", "standard")
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device(cfg.get("device", "auto"))
    rng = np.random.default_rng(seed)

    tr, va = split["train"], split.get("val")
    model_kwargs = (cfg.get("model_kwargs") or {}).get(model_name, {})
    model = build_model(model_name, data.n_classes, cfg.get("pretrained", True),
                        **model_kwargs)
    if cfg.get("init_from"):  # fine-tune ต่อจากน้ำหนักเดิม ไม่ใช่เริ่มใหม่จากศูนย์
        ckpt = torch.load(cfg["init_from"], map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if verbose:
            print(f"  โหลดน้ำหนักจาก {cfg['init_from']} (epoch {ckpt.get('epoch')}, "
                  f"score {ckpt.get('score', 0):.4f}) มาเทรนต่อ")
    model = model.to(device)
    policy = get_policy(aug)

    steps = max(1, len(tr) // cfg["batch_size"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * steps, pct_start=0.15)

    weight = None
    if cfg.get("imbalance") == "class_weight":
        cnt = np.bincount(data.y[tr], minlength=data.n_classes)
        weight = torch.tensor((cnt.mean() / np.maximum(cnt, 1)) ** 0.5,
                              dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight,
                                    label_smoothing=cfg.get("label_smoothing", 0.1))

    if verbose:
        print(f"[{model_name} | aug={aug}] train {len(tr)} | val {len(va) if va is not None else 0}"
              f" | {data.n_classes} classes | {count_params(model) / 1e6:.1f}M params | {device}")

    history, best, best_epoch, bad = [], -1.0, 0, 0
    out_dir = Path(out_dir) if out_dir else None
    ckpt_path = (out_dir / "best.pt") if out_dir else None
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        t0, run_loss, correct, n = time.time(), 0.0, 0, 0
        epoch_ids = tr
        if cfg.get("imbalance") == "weighted_sampling":
            p = _sampling_probs(data.y, tr, cfg.get("sampling_power", 0.5))
            epoch_ids = rng.choice(tr, size=len(tr), replace=True, p=p)
        for step, (xb, yb) in enumerate(
                _batches(data.x, data.y, epoch_ids, cfg["batch_size"], True, device, policy, rng)):
            out = model(xb)
            loss = criterion(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            run_loss += loss.item() * len(yb); n += len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            if verbose and step % 200 == 0:
                print(f"    ep{ep} step {step}/{steps} loss {loss.item():.4f}", flush=True)

        rec = dict(epoch=ep, train_loss=run_loss / max(n, 1), train_acc=correct / max(n, 1),
                   sec=time.time() - t0)
        if va is not None and len(va):
            pred, probs = predict(model, data, va, device, cfg.get("eval_batch_size", 256))
            rec["val_acc"] = float((pred == data.y[va]).mean())
            true_p = probs[np.arange(len(va)), data.y[va]]
            rec["val_loss"] = float(-np.log(np.clip(true_p, 1e-12, None)).mean())
            score = rec["val_acc"]
        else:
            score = rec["train_acc"]
        history.append(rec)
        if verbose:
            msg = " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                           for k, v in rec.items() if k != "val_loss")
            print(f"  {msg}", flush=True)

        if score > best:
            best, best_epoch, bad = score, ep, 0
            if ckpt_path:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "model_name": model_name,
                            "classes": data.classes, "img_size": cfg["img_size"],
                            "augmentation": aug, "epoch": ep, "score": best}, ckpt_path)
        else:
            bad += 1
            if cfg.get("early_stop_patience") and bad >= cfg["early_stop_patience"]:
                if verbose:
                    print(f"  early stop ที่ epoch {ep} (ไม่ดีขึ้น {bad} epoch)")
                break

    if ckpt_path and ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    result = dict(model=model_name, augmentation=aug, best_score=float(best),
                  best_epoch=best_epoch, history=history,
                  checkpoint=str(ckpt_path) if ckpt_path else None,
                  config={k: cfg[k] for k in cfg if k not in ("splits", "models", "augmentations")})
    if out_dir:
        save_json({k: v for k, v in result.items()}, out_dir / "training.json")
    return model, result
