"""ตัวรันการทดลองทั้ง 4 แบบ ใช้ได้ทั้งจาก command line และเรียกเป็นฟังก์ชันในโน้ตบุ๊ก

    python -m src.training.run_experiment --experiment model_comparison --config configs/baseline.yaml
    python -m src.training.run_experiment --experiment augmentation     --config configs/augmentation.yaml
    python -m src.training.run_experiment --experiment hard_character   --config configs/baseline.yaml
    python -m src.training.run_experiment --experiment generalization   --config configs/generalization.yaml

ทุก run เขียนผลลง experiments/<experiment>/<run_name>/ และสรุปเป็นตารางใน results/tables/
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from src.datasets.splits import build_split, describe
from src.datasets.thai_chars import load_data
from src.evaluation import hard_chars as hc
from src.evaluation.metrics import evaluate_split, generalization_gap, save_report
from src.models.factory import build_model, count_params, inference_time_ms
from src.training.trainer import train_model
from src.utils.common import (EXPERIMENTS_DIR, RESULTS_DIR, get_device, load_config,
                              run_dir, save_json, set_seed)


# ---------------------------------------------------------------- helpers


def _names(cfg, default):
    """ชื่อโฟลเดอร์ experiment และชื่อไฟล์ตาราง — ตั้ง experiment_name/table_name ใน config
    เพื่อให้การทดสอบแบบเร็วไม่เขียนทับผลรันจริง"""
    exp = cfg.get("experiment_name", default)
    return exp, cfg.get("table_name", exp)


def save_table(rows, name):
    """เขียนตารางสรุปเป็น CSV ใน results/tables/"""
    if not rows:
        return None
    path = RESULTS_DIR / "tables" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))  # union ของคีย์ทุกแถว
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"  ตาราง -> {path.relative_to(RESULTS_DIR.parent)}")
    return path


def prepare(cfg, verbose=True):
    """โหลดข้อมูลตาม config (รองรับ limit_per_class สำหรับทดสอบ pipeline)"""
    set_seed(cfg.get("seed", 42))
    data = load_data(cfg["img_size"], verbose=verbose)
    if cfg.get("limit_per_class"):
        ids = data.subset_per_class(cfg["limit_per_class"], cfg.get("seed", 42))
        data = type(data)(data.x[ids], data.codes[ids], data.sources[ids],
                          data.groups[ids], data.paths[ids])
        if verbose:
            print(f"[limit_per_class={cfg['limit_per_class']}] เหลือ {len(data.y)} รูป "
                  f"{data.n_classes} คลาส")
    return data


def train_and_eval(data, split, cfg, model_name, aug, experiment, run_name, verbose=True):
    """เทรน 1 run แล้ววัดผลบน test (ถ้าไม่มี test ใช้ val)"""
    out = run_dir(experiment, run_name)
    t0 = time.time()
    model, result = train_model(data, split, cfg, model_name, aug, out, verbose)
    device = get_device(cfg.get("device", "auto"))
    eval_ids = split.get("test") if split.get("test") is not None else split.get("val")
    metrics = evaluate_split(model, data, eval_ids, device, data.classes)
    np.save(out / "confusion.npy", metrics["confusion"])
    np.savez(out / "predictions.npz", y_true=metrics["y_true"], y_pred=metrics["y_pred"],
             ids=eval_ids)
    save_report(metrics, out / "report.json",
                extra=dict(model=model_name, augmentation=aug, run=run_name,
                           split=describe(data, split), train_seconds=time.time() - t0,
                           params=count_params(model),
                           inference_ms=inference_time_ms(model, device, cfg["img_size"])))
    if verbose:
        print(f"  {run_name}: acc {metrics['accuracy']:.4f} | macro F1 {metrics['macro_f1']:.4f}")
    return model, result, metrics


# ---------------------------------------------------------------- experiments


def run_model_comparison(cfg, data=None, verbose=True):
    """Experiment A — เทียบ architecture ภายใต้ setup เดียวกัน"""
    data = data or prepare(cfg, verbose)
    exp, table = _names(cfg, "model_comparison")
    split = build_split(data, cfg["split"], cfg.get("seed", 42), verbose)
    print("split:", describe(data, split))
    rows = []
    for name in cfg["models"]:
        print(f"\n=== model: {name} ===")
        model, result, metrics = train_and_eval(
            data, split, cfg, name, cfg.get("augmentation", "standard"),
            exp, name, verbose)
        device = get_device(cfg.get("device", "auto"))
        rows.append(dict(model=name, accuracy=round(metrics["accuracy"], 4),
                         macro_f1=round(metrics["macro_f1"], 4),
                         params_m=round(count_params(model) / 1e6, 2),
                         inference_ms=round(inference_time_ms(model, device, cfg["img_size"]), 2),
                         best_epoch=result["best_epoch"],
                         train_minutes=round(sum(h["sec"] for h in result["history"]) / 60, 1)))
        save_table(rows, table)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    return rows


def run_augmentation(cfg, data=None, verbose=True):
    """Experiment B — เทียบ augmentation 4 ระดับด้วยโมเดลเดียวกัน"""
    data = data or prepare(cfg, verbose)
    exp, table = _names(cfg, "augmentation")
    split = build_split(data, cfg["split"], cfg.get("seed", 42), verbose)
    rows = []
    for aug in cfg["augmentations"]:
        print(f"\n=== augmentation: {aug} ===")
        model, result, metrics = train_and_eval(
            data, split, cfg, cfg["model"], aug, exp, f'{cfg["model"]}_{aug}', verbose)
        row = dict(augmentation=aug, accuracy=round(metrics["accuracy"], 4),
                   macro_f1=round(metrics["macro_f1"], 4), best_epoch=result["best_epoch"])
        # RQ2 ตัวจริง: augmentation ช่วยตอนเจอ source ที่ไม่เคยเห็นหรือไม่
        device = get_device(cfg.get("device", "auto"))
        for t in cfg.get("cross_test_sources", []):
            mt = evaluate_split(model, data, data.where(sources=t), device, data.classes)
            row[f"acc_{t}"] = round(mt["accuracy"], 4)
            row[f"gap_{t}"] = round(generalization_gap(metrics["accuracy"], mt["accuracy"]), 4)
            save_report(mt, run_dir(exp, f'{cfg["model"]}_{aug}') / f"cross_test_{t}.json",
                        extra=dict(augmentation=aug, test_source=t))
            print(f"  -> ทดสอบกับ {t}: acc {mt['accuracy']:.4f} | macro F1 {mt['macro_f1']:.4f}")
        rows.append(row)
        save_table(rows, table)
        del model
    return rows


def run_hard_character(cfg, data=None, verbose=True):
    """Experiment C — หา hard character จาก confusion matrix แล้ว fine-tune ซ้ำ"""
    data = data or prepare(cfg, verbose)
    exp, table = _names(cfg, "hard_character")
    split = build_split(data, cfg["split"], cfg.get("seed", 42), verbose)
    aug = cfg.get("augmentation", "thai_specific")

    print("\n=== รอบที่ 1: baseline ===")
    model, base_result, metrics = train_and_eval(data, split, cfg, cfg["model"], aug,
                                                 exp, "baseline", verbose)
    hard = hc.find_hard_characters(metrics, data.classes,
                                   cfg.get("hard_f1_threshold", 0.9),
                                   cfg.get("hard_top_k") or 10,   # null ใน yaml = ใช้ค่า 10
                                   cfg.get("hard_min_support", 20))
    if not hard:
        print(f"  [เตือน] ไม่พบคลาสที่มีตัวอย่างใน test ถึง {cfg.get('hard_min_support', 20)} รูป "
              f"จึงหา hard character ไม่ได้ — ลดค่า hard_min_support หรือเพิ่มขนาด test set")
    pairs = hc.hard_pairs(metrics)
    hc.save_hard_set(hard, pairs, EXPERIMENTS_DIR / exp / "hard_set.json")
    print(hc.summarize(hard))

    print("\n=== รอบที่ 2: fine-tune ด้วย weighted sampling ของคลาสยาก ===")
    ft_cfg = dict(cfg, epochs=cfg.get("finetune_epochs", max(2, cfg["epochs"] // 3)),
                  lr=cfg["lr"] * cfg.get("finetune_lr_scale", 0.3),
                  imbalance="weighted_sampling",
                  sampling_power=cfg.get("finetune_sampling_power", 0.75),
                  init_from=base_result["checkpoint"])  # เทรนต่อจาก baseline จริง
    _, _, ft_metrics = train_and_eval(data, split, ft_cfg, cfg["model"], aug,
                                      exp, "finetuned", verbose)
    hard_codes = [h["code"] for h in hard]
    before = float(np.mean([metrics["per_class"][c]["f1"] for c in hard_codes])) if hard_codes else 0.0
    after = float(np.mean([ft_metrics["per_class"].get(c, {"f1": 0})["f1"]
                           for c in hard_codes])) if hard_codes else 0.0
    rows = [dict(stage="baseline", accuracy=round(metrics["accuracy"], 4),
                 macro_f1=round(metrics["macro_f1"], 4), hard_mean_f1=round(before, 4)),
            dict(stage="finetuned", accuracy=round(ft_metrics["accuracy"], 4),
                 macro_f1=round(ft_metrics["macro_f1"], 4), hard_mean_f1=round(after, 4))]
    save_table(rows, table)
    return rows


def run_generalization(cfg, data=None, verbose=True):
    """Experiment D — random split vs cross-source (6 ทิศ) vs unseen font"""
    data = data or prepare(cfg, verbose)
    exp, table = _names(cfg, "generalization")
    model_name, aug = cfg["model"], cfg.get("augmentation", "thai_specific")
    device = get_device(cfg.get("device", "auto"))
    sources = cfg.get("sources", ["D1", "D2", "D3"])
    rows, matrix = [], {}

    # D1. Random split (in-domain baseline) ต่อ source
    for s in sources:
        print(f"\n=== random split: {s} ===")
        split = build_split(data, {"type": "random", "sources": s}, cfg.get("seed", 42), verbose)
        model, _, m = train_and_eval(data, split, cfg, model_name, aug,
                                     exp, f"random_{s}", verbose)
        rows.append(dict(setting="random_split", train=s, test=s,
                         accuracy=round(m["accuracy"], 4), macro_f1=round(m["macro_f1"], 4)))
        matrix[(s, s)] = m["accuracy"]

        # D2. Cross-source: ใช้โมเดลตัวเดียวกันนี้ไปทดสอบกับ source อื่นที่ไม่เคยเห็น
        for t in sources:
            if t == s:
                continue
            ids = data.where(sources=t)
            mt = evaluate_split(model, data, ids, device, data.classes)
            rows.append(dict(setting="cross_source", train=s, test=t,
                             accuracy=round(mt["accuracy"], 4), macro_f1=round(mt["macro_f1"], 4),
                             gap=round(generalization_gap(m["accuracy"], mt["accuracy"]), 4)))
            matrix[(s, t)] = mt["accuracy"]
            save_report(mt, run_dir(exp, f"random_{s}") / f"cross_test_{t}.json",
                        extra=dict(train_source=s, test_source=t))
            print(f"  {s} -> {t}: acc {mt['accuracy']:.4f} | macro F1 {mt['macro_f1']:.4f}")
        save_table(rows, table)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    # D3. Unseen font test
    for spec in cfg.get("unseen_font", []):
        s = spec["source"]
        print(f"\n=== unseen font: {s} ===")
        try:
            split = build_split(data, dict(spec, type="unseen_font"), cfg.get("seed", 42), verbose)
        except ValueError as e:
            print(f"  ข้าม unseen font ของ {s}: {e}")
            continue
        _, _, m = train_and_eval(data, split, cfg, model_name, aug,
                                 exp, f"unseen_font_{s}", verbose)
        rows.append(dict(setting="unseen_font", train=s, test=f'{s}:heldout_groups',
                         accuracy=round(m["accuracy"], 4), macro_f1=round(m["macro_f1"], 4)))
        save_table(rows, table)

    save_json({f"{k[0]}->{k[1]}": round(v, 4) for k, v in matrix.items()},
              RESULTS_DIR / "tables" / f"cross_source_matrix{'' if table == 'generalization' else '_' + table}.json")
    return rows


EXPERIMENTS = {"model_comparison": run_model_comparison, "augmentation": run_augmentation,
               "hard_character": run_hard_character, "generalization": run_generalization}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, choices=list(EXPERIMENTS))
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, help="override epochs ใน config")
    ap.add_argument("--limit-per-class", type=int, help="ใช้ข้อมูลไม่เกิน N รูปต่อคลาสต่อ source")
    ap.add_argument("--models", nargs="+", help="override รายชื่อโมเดล")
    ap.add_argument("--device", help="cpu / mps / cuda (ค่าเริ่มต้น auto)")
    args = ap.parse_args()

    cfg = load_config(args.config, epochs=args.epochs, limit_per_class=args.limit_per_class,
                      models=args.models, device=args.device)
    print(f"=== {args.experiment} | config: {Path(args.config).name} ===")
    EXPERIMENTS[args.experiment](cfg)
    print("\nเสร็จแล้ว ดูผลที่ experiments/ และ results/tables/")


if __name__ == "__main__":
    main()
