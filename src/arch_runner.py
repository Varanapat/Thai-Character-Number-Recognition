"""เครื่องยนต์เบื้องหลังไฟล์ arch_*.py — เทรน ประเมิน และทำนายจริง ด้วย architecture เดียว

ไฟล์ arch_*.py แต่ละไฟล์เรียก main("<ชื่อ architecture>") เท่านั้น
ทุกไฟล์ใช้ค่าเดียวกันจาก configs/architecture.yaml จึงเทียบกันได้อย่างยุติธรรม

โหมดการทำงาน
    train    เทรน + ประเมิน แล้วต่อท้ายผลลง results/tables/architecture_results.csv
    predict  โหลดโมเดลที่เทรนไว้ ทำนายรูปในโฟลเดอร์ แล้วเขียน CSV ส่งงาน
    both     ทำทั้งสองอย่างต่อกัน

ผลการประเมินมี 2 ส่วนเสมอ
    in-domain   test set ที่กันไว้ของแหล่งที่ใช้เทรน
    unseen      ทุกแหล่งที่ไม่ได้ใช้เทรน ประเมินทั้งแหล่ง (คอลัมน์ acc_<source>)
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.datasets.splits import random_split
from src.ensemble.combiners import load_model, predict_probs
from src.evaluation.metrics import compute_metrics, save_report
from src.models.factory import MODEL_NAMES, count_params, inference_time_ms
from src.training.trainer import train_model
from src.training.run_experiment import prepare
from src.utils.common import (EXPERIMENTS_DIR, RESULTS_DIR, ROOT, get_device, load_config,
                              save_json, thai_char)

MODELS_DIR = ROOT / "models"
TABLE = RESULTS_DIR / "tables" / "architecture_results.csv"
DEFAULT_CONFIG = ROOT / "configs" / "architecture.yaml"


def append_row(row, path=TABLE):
    """ต่อท้ายผลหนึ่งแถวลง CSV โดยรักษาคอลัมน์เดิมไว้ (ไฟล์เดียวรวมทุก architecture)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(path, encoding="utf-8"))) if path.exists() else []
    rows.append({k: v for k, v in row.items()})
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"  ต่อท้ายผลลง {path.relative_to(ROOT)} (ตอนนี้มี {len(rows)} แถว)")


def do_train(arch, cfg, args):
    data = prepare(cfg)
    sources = cfg["train_sources"]
    available = sorted({s for s in set(data.sources.tolist()) if s != "unknown"})
    unseen = [s for s in available if s not in sources]
    print(f"\n=== {arch} | เทรนด้วย {'+'.join(sources)} | แหล่งที่ไม่ได้เทรน: "
          f"{', '.join(unseen) if unseen else 'ไม่มี'} ===")

    splits = {s: random_split(data, sources=s, val_ratio=cfg["val_ratio"],
                              test_ratio=cfg["test_ratio"], seed=cfg["seed"], verbose=False)
              for s in sources}
    split = {k: np.concatenate([splits[s][k] for s in sources]) for k in ("train", "val", "test")}
    print(f"  train {len(split['train'])} | val {len(split['val'])} | test {len(split['test'])}")

    out_dir = MODELS_DIR / arch
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, result = train_model(data, split, cfg, arch, cfg["augmentation"], out_dir)
    minutes = (time.time() - t0) / 60
    device = get_device(cfg.get("device", "auto"))

    probs = predict_probs(model, data.x[split["test"]], device, cfg["eval_batch_size"])
    m = compute_metrics(data.y[split["test"]], probs.argmax(1), data.classes, True)
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "architecture": arch,
           "train_sources": "+".join(sources), "augmentation": cfg["augmentation"],
           "epochs": cfg["epochs"], "img_size": cfg["img_size"], "seed": cfg["seed"],
           "n_train": len(split["train"]), "n_test": len(split["test"]),
           "val_acc": round(result["best_score"], 4),
           "test_acc": round(m["accuracy"], 4), "test_macro_f1": round(m["macro_f1"], 4)}
    print(f"  in-domain: acc {m['accuracy']:.4f} | macro F1 {m['macro_f1']:.4f}")
    save_report(m, out_dir / "report_in_domain.json", extra={"architecture": arch})

    unseen_scores = []
    for s in unseen:
        ids = data.where(sources=s)
        pr = predict_probs(model, data.x[ids], device, cfg["eval_batch_size"])
        ms_ = compute_metrics(data.y[ids], pr.argmax(1), data.classes, True)
        row[f"acc_{s}"] = round(ms_["accuracy"], 4)
        row[f"macro_f1_{s}"] = round(ms_["macro_f1"], 4)
        unseen_scores.append(ms_["macro_f1"])
        print(f"  แหล่งที่ไม่เคยเห็น {s}: acc {ms_['accuracy']:.4f} | macro F1 {ms_['macro_f1']:.4f}")
        save_report(ms_, out_dir / f"report_unseen_{s}.json", extra={"test_source": s})
    if unseen_scores:
        row["unseen_macro_f1_avg"] = round(float(np.mean(unseen_scores)), 4)

    row |= {"params_m": round(count_params(model) / 1e6, 2),
            "inference_ms": round(inference_time_ms(model, device, cfg["img_size"]), 2),
            "train_minutes": round(minutes, 1),
            "checkpoint": str((out_dir / "best.pt").relative_to(ROOT))}
    append_row(row)
    save_json(row, out_dir / "summary.json")
    return row


def do_predict(arch, cfg, args):
    ckpt_path = Path(args.checkpoint) if args.checkpoint else MODELS_DIR / arch / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"ยังไม่มีโมเดลที่ {ckpt_path} — รันโหมด train ก่อน")
    from src.datasets.thai_chars import EXTS, preprocess

    device = get_device(cfg.get("device", "auto"))
    model, ckpt = load_model(ckpt_path, device)
    classes, size = ckpt["classes"], ckpt["img_size"]

    src = Path(args.images)
    paths = [src] if src.is_file() else sorted(f for f in src.rglob("*")
                                               if f.suffix.lower() in EXTS)
    if not paths:
        raise SystemExit(f"ไม่พบรูปใน {args.images}")
    print(f"\n=== ทำนาย {len(paths)} รูปด้วย {arch} ({ckpt_path.name}) ===")

    x = np.stack([preprocess(p, size) for p in paths])
    probs = predict_probs(model, x, device, cfg["eval_batch_size"])

    out = Path(args.out or f"submission/{arch}_predictions.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "predicted_code", "predicted_char", "confidence", "top_3"])
        for p, pr in zip(paths, probs):
            top = np.argsort(-pr)[:3]
            w.writerow([p.name, classes[top[0]], thai_char(classes[top[0]]),
                        round(float(pr[top[0]]), 4),
                        " | ".join(f"{classes[i]}({thai_char(classes[i])}):{pr[i]:.3f}"
                                   for i in top)])
    conf = probs.max(1)
    print(f"  เขียนผลลง {out}")
    print(f"  ความมั่นใจเฉลี่ย {conf.mean():.3f} | ต่ำกว่า 0.5 จำนวน {int((conf < 0.5).sum())} รูป")
    return out


def main(arch):
    if arch not in MODEL_NAMES:
        raise SystemExit(f"ไม่รู้จัก {arch} (มีให้เลือก {MODEL_NAMES})")
    ap = argparse.ArgumentParser(description=f"{arch} — เทรน / ประเมิน / ทำนาย")
    ap.add_argument("--mode", default="train", choices=["train", "predict", "both"])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--train-sources", nargs="+",
                    help="แหล่งที่ใช้เทรน เช่น --train-sources D1 D2 (ที่เหลือกลายเป็นแหล่งทดสอบ)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit-per-class", type=int, help="ใช้ทดสอบว่าโค้ดรันได้")
    ap.add_argument("--images", help="โฟลเดอร์รูปสำหรับโหมด predict")
    ap.add_argument("--out", help="ไฟล์ CSV ผลทำนาย")
    ap.add_argument("--checkpoint", help="ระบุ .pt เองแทนค่าเริ่มต้น models/<arch>/best.pt")
    args = ap.parse_args()

    cfg = load_config(args.config, epochs=args.epochs, limit_per_class=args.limit_per_class)
    cfg["model"] = arch
    if args.train_sources:
        cfg["train_sources"] = args.train_sources

    if args.mode in ("train", "both"):
        do_train(arch, cfg, args)
    if args.mode in ("predict", "both"):
        if not args.images:
            raise SystemExit("โหมด predict ต้องระบุ --images")
        do_predict(arch, cfg, args)
