"""เปรียบเทียบ training strategy S1-S7 ของ MobileNetV3-Small (Model_strategy.md)

สถาปัตยกรรม preprocessing optimizer seed epoch เหมือนกันหมด ตัวแปรเดียวคือ "วิธีเทรน/วิธีรวมผล"

กลยุทธ์ที่เทียบ
    S1  D1-only            โมเดลเดียว เทรนด้วย D1
    S2  D3-only            โมเดลเดียว เทรนด้วย D3
    S2b D2-only            เพิ่มเข้ามาเพื่อให้ครบทั้งสามแหล่ง
    S3  Combined           โมเดลเดียว เทรนด้วยทั้งสามแหล่ง (split สะอาด ไม่ทับ test)
    S4  Majority Vote      โหวตจาก 3 โมเดลรายแหล่ง
    S5  Prob Average       เฉลี่ย probability จาก 3 โมเดล
    S6  Weighted Prob      ถ่วงน้ำหนักตามหลักฐานจาก validation (ห้ามใช้ test)
    S7  S3 เทียบ S4/S5/S6  รวมความรู้ตอนเทรน เทียบกับ รวมตอน inference

สองสนามวัดผล
    held_out  test set ที่กันไว้ของทั้งสามแหล่ง — วัดผลในสภาพ "แหล่งที่เคยเห็น"
    loso      leave-one-source-out: เทรนด้วย 2 แหล่ง ทดสอบกับแหล่งที่ 3 ทั้งแหล่ง
              — จำลองสถานการณ์ "ข้อมูลพรุ่งนี้มาจากแหล่งใหม่" ใช้ตัวนี้เป็นตัวตัดสิน

วิธีรัน
    python -m src.training.run_strategy --stage all --config configs/strategy.yaml
    python -m src.training.run_strategy --stage ensemble   # ไม่ต้องเทรน ใช้โมเดลที่มีอยู่
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.datasets.splits import random_split
from src.ensemble.combiners import (agreement_stats, apply_mask, load_model, majority_vote,
                                    predict_probs, prob_average, seen_class_mask)
from src.evaluation.metrics import compute_metrics
from src.models.factory import inference_time_ms
from src.training.run_experiment import prepare, save_table, train_and_eval
from src.utils.common import EXPERIMENTS_DIR, get_device, load_config, save_json

EXP = "strategy"
SOURCES = ("D1", "D2", "D3")   # ค่าเริ่มต้น ถูกแทนที่ด้วยรายชื่อจริงจากข้อมูลตอนรัน


def set_sources(data, cfg):
    """กำหนดรายชื่อ source จากข้อมูลที่โหลดมาจริง หรือจาก key `sources` ใน config

    เพิ่มข้อมูลแหล่งใหม่ลง Dataset/ แล้วรันได้เลย ไม่ต้องแก้โค้ด
    """
    global SOURCES
    found = sorted({s for s in set(data.sources.tolist()) if s != "unknown"})
    SOURCES = tuple(cfg.get("sources") or found)
    unused = set(found) - set(SOURCES)
    print(f"source ที่ใช้: {', '.join(SOURCES)}" + (f" (ข้าม {', '.join(sorted(unused))})" if unused else ""))
    return SOURCES


# ---------------------------------------------------------------- splits


def source_splits(data, cfg):
    """แบ่งแต่ละแหล่งแบบ 80/10/10 ด้วย seed เดียวกัน

    ใช้ฟังก์ชันและ seed เดียวกับ Experiment D จึงได้ชุดเดียวกันเป๊ะ
    ทำให้ checkpoint ของ random_D1/D2/D3 ที่เทรนไว้แล้วนำมาใช้ต่อได้ทันที
    """
    return {s: random_split(data, sources=s, val_ratio=0.1, test_ratio=0.1,
                            seed=cfg.get("seed", 42), verbose=False) for s in SOURCES}


def union(splits, part, sources=SOURCES):
    return np.concatenate([splits[s][part] for s in sources])


# ---------------------------------------------------------------- โมเดลรายแหล่ง


def source_model_paths(cfg, splits, data):
    """หา checkpoint ของโมเดลรายแหล่ง ถ้ายังไม่มีให้เทรนใหม่ด้วยโปรโตคอลเดียวกัน"""
    paths = {}
    for s in SOURCES:
        reuse = EXPERIMENTS_DIR / "generalization" / f"random_{s}" / "best.pt"
        own = EXPERIMENTS_DIR / EXP / f"single_{s}" / "best.pt"
        if own.exists():
            paths[s] = own
        elif reuse.exists() and cfg.get("reuse_generalization_models", True):
            paths[s] = reuse
            print(f"  ใช้โมเดล {s} ที่เทรนไว้แล้วจาก experiments/generalization/random_{s}/")
        else:
            print(f"\n=== เทรนโมเดลรายแหล่ง: {s} ===")
            train_and_eval(data, splits[s], cfg, cfg["model"], cfg["augmentation"],
                           EXP, f"single_{s}")
            paths[s] = own
    return paths


# ---------------------------------------------------------------- probability cache


def probs_for(paths, data, ids, device, cfg, tag):
    """คำนวณ probability ของทุกโมเดลบน ids ชุดหนึ่ง แล้ว cache ไว้ใช้ซ้ำ"""
    cache = EXPERIMENTS_DIR / EXP / "probs" / f"{tag}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if len(z["ids"]) == len(ids) and (z["ids"] == ids).all():
            return {k: z[k] for k in z.files if k not in ("ids",)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, p in paths.items():
        model, _ = load_model(p, device)
        out[name] = predict_probs(model, data.x[ids], device, cfg.get("eval_batch_size", 256))
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    np.savez(cache, ids=ids, **out)
    return out


# ---------------------------------------------------------------- การประเมิน


def evaluate(name, probs, y_true, data, rows, extra=None):
    m = compute_metrics(y_true, probs.argmax(1), data.classes, labels_present_only=True)
    row = dict(strategy=name, accuracy=round(m["accuracy"], 4), macro_f1=round(m["macro_f1"], 4),
               n_test=m["n_samples"], **(extra or {}))
    rows.append(row)
    print(f"  {name:28} acc {m['accuracy']:.4f} | macro F1 {m['macro_f1']:.4f}")
    return row, m


def val_weights(paths, data, splits, device, cfg, members=None, verbose=True):
    """S6 — น้ำหนักจากหลักฐาน validation เท่านั้น (ห้ามแตะ test ตาม Model_strategy.md ข้อ 10)

    ให้คะแนนแต่ละโมเดลด้วย macro F1 เฉลี่ยบน val ของ "แหล่งอื่นที่มันไม่ได้เทรน"
    เพราะเรากำลังเลือกโมเดลที่ทำงานกับแหล่งแปลกหน้าได้ดี

    members: จำกัดวงให้เหลือเฉพาะโมเดลที่ใช้จริงในสถานการณ์นั้น
             ตอนทำ leave-one-source-out ต้องส่งเฉพาะสองแหล่งที่เทรน
             ห้ามให้ val ของแหล่งที่กันไว้ทดสอบหลุดเข้ามามีส่วนกำหนดน้ำหนัก
    """
    members = list(members or SOURCES)
    scores = {}
    for s in members:
        model, _ = load_model(paths[s], device)
        vals = []
        for other in members:
            if other == s:
                continue
            ids = splits[other]["val"]
            pr = predict_probs(model, data.x[ids], device, cfg.get("eval_batch_size", 256))
            vals.append(compute_metrics(data.y[ids], pr.argmax(1), data.classes,
                                        labels_present_only=True)["macro_f1"])
        scores[s] = float(np.mean(vals))
        del model
    total = sum(scores.values())
    weights = {k: v / total for k, v in scores.items()}
    if verbose:
        print(f"  น้ำหนักจาก val ของ {'+'.join(members)}:",
              {k: f"{v:.3f}" for k, v in weights.items()})
    return weights, scores


def ensemble_set(prob_dict, masks, weights, members):
    """คืน dict ของทุกวิธีรวมผล สำหรับสมาชิกที่กำหนด"""
    raw = [prob_dict[m] for m in members]
    masked = [apply_mask(prob_dict[m], masks[m]) for m in members]
    w = [weights[m] for m in members]
    return {
        "S4 majority_vote": np.eye(raw[0].shape[1])[majority_vote(raw)],
        "S5 prob_average": prob_average(raw),
        "S5b prob_avg_masked": prob_average(masked),
        "S6 weighted_prob": prob_average(raw, w),
        "S6b weighted_masked": prob_average(masked, w),
    }


# ---------------------------------------------------------------- stages


def stage_combined(data, splits, cfg):
    """S3 — เทรนด้วยทั้งสามแหล่ง โดยใช้เฉพาะ train ของแต่ละแหล่ง (ไม่แตะ test เลย)"""
    print("\n=== S3: Combined training (split สะอาด) ===")
    split = {"train": union(splits, "train"), "val": union(splits, "val"),
             "test": union(splits, "test")}
    print(f"  train {len(split['train'])} | val {len(split['val'])} | test {len(split['test'])}")
    train_and_eval(data, split, cfg, cfg["model"], cfg["augmentation"], EXP, "combined_clean")


def stage_loso(data, splits, cfg):
    """เทรนด้วย 2 แหล่ง ทดสอบกับแหล่งที่ 3 ทั้งแหล่ง — จำลอง 'แหล่งใหม่ที่ไม่เคยเห็น'"""
    for held in SOURCES:
        others = [s for s in SOURCES if s != held]
        print(f"\n=== LOSO: เทรนด้วย {'+'.join(others)} -> ทดสอบ {held} ===")
        split = {"train": union(splits, "train", others), "val": union(splits, "val", others),
                 "test": data.where(sources=held)}
        print(f"  train {len(split['train'])} | test {len(split['test'])} (ทั้งแหล่ง {held})")
        train_and_eval(data, split, cfg, cfg["model"], cfg["augmentation"],
                       EXP, f"loso_{held}")


def stage_ensemble(data, splits, cfg):
    """ประเมินทุกกลยุทธ์ในสองสนาม แล้วสรุปเป็นตาราง"""
    device = get_device(cfg.get("device", "auto"))
    paths = source_model_paths(cfg, splits, data)
    masks = {s: seen_class_mask(data, splits[s]["train"]) for s in SOURCES}
    for s in SOURCES:
        print(f"  โมเดล {s} เคยเห็น {masks[s].sum()}/{data.n_classes} คลาส")

    weights, raw_scores = val_weights(paths, data, splits, device, cfg)
    save_json(dict(weights=weights, val_cross_source_macro_f1=raw_scores,
                   note="น้ำหนักนี้มาจาก val ของทั้งสามแหล่ง ใช้กับ ensemble 3 โมเดลตอน inference จริง "
                        "ส่วนในสนาม LOSO จะคำนวณน้ำหนักใหม่จากเฉพาะสองแหล่งที่เทรน"),
              EXPERIMENTS_DIR / EXP / "ensemble_weights.json")

    # ---------- สนามที่ 1: test set ที่กันไว้ของทั้งสามแหล่ง ----------
    print("\n=== สนามที่ 1: held-out test ของทั้งสามแหล่ง ===")
    ids = union(splits, "test")
    y = data.y[ids]
    prob_dict = probs_for(paths, data, ids, device, cfg, "heldout")
    rows = []
    for s in SOURCES:
        evaluate(f"S1/S2 single_{s}", prob_dict[s], y, data, rows, dict(field="held_out"))
    comb = EXPERIMENTS_DIR / EXP / "combined_clean" / "best.pt"
    if comb.exists():
        model, _ = load_model(comb, device)
        p = predict_probs(model, data.x[ids], device, cfg.get("eval_batch_size", 256))
        evaluate("S3 combined", p, y, data, rows, dict(field="held_out"))
        del model
    for name, p in ensemble_set(prob_dict, masks, weights, list(SOURCES)).items():
        evaluate(name, p, y, data, rows, dict(field="held_out"))
    stats = agreement_stats([prob_dict[s] for s in SOURCES])
    print(f"  โมเดลทั้งสามเห็นตรงกัน {stats['agreement_rate']:.1%} "
          f"| ไม่ตรงกัน {stats['n_disagree']} รูป")
    save_json(stats, EXPERIMENTS_DIR / EXP / "agreement_heldout.json")

    # ---------- สนามที่ 2: leave-one-source-out (ตัวตัดสิน) ----------
    print("\n=== สนามที่ 2: leave-one-source-out (แหล่งที่ไม่เคยเห็น) ===")
    for held in SOURCES:
        others = [s for s in SOURCES if s != held]
        ids = data.where(sources=held)
        y = data.y[ids]
        print(f"\n  -- ทดสอบกับ {held} ทั้งแหล่ง ({len(ids)} รูป) โดยใช้ความรู้จาก {'+'.join(others)}")
        pd_ = probs_for(paths, data, ids, device, cfg, f"loso_{held}")
        for s in others:
            evaluate(f"single_{s} -> {held}", pd_[s], y, data, rows,
                     dict(field="loso", held_out=held))
        loso_ckpt = EXPERIMENTS_DIR / EXP / f"loso_{held}" / "best.pt"
        if loso_ckpt.exists():
            model, _ = load_model(loso_ckpt, device)
            p = predict_probs(model, data.x[ids], device, cfg.get("eval_batch_size", 256))
            evaluate(f"S3 combined({'+'.join(others)}) -> {held}", p, y, data, rows,
                     dict(field="loso", held_out=held))
            del model
        w_loso, _ = val_weights(paths, data, splits, device, cfg, members=others)
        for name, p in ensemble_set(pd_, masks, w_loso, others).items():
            evaluate(f"{name}({'+'.join(others)}) -> {held}", p, y, data, rows,
                     dict(field="loso", held_out=held))
        save_json(agreement_stats([pd_[s] for s in others]),
                  EXPERIMENTS_DIR / EXP / f"agreement_loso_{held}.json")

    save_table(rows, "strategy_comparison")
    summarize(rows, data, cfg, device)
    return rows


def summarize(rows, data, cfg, device):
    """สรุปว่ากลยุทธ์ไหนควรใช้จริง โดยตัดสินจากสนาม LOSO"""
    loso = [r for r in rows if r.get("field") == "loso"]
    by_strategy = {}
    for r in loso:
        key = r["strategy"].split("(")[0].split(" -> ")[0].strip()
        by_strategy.setdefault(key, []).append(r["macro_f1"])
    ranked = sorted(((k, float(np.mean(v)), len(v)) for k, v in by_strategy.items()),
                    key=lambda kv: -kv[1])
    print("\n=== สรุป: macro F1 เฉลี่ยบนแหล่งที่ไม่เคยเห็น (ตัวตัดสิน) ===")
    for k, v, n in ranked:
        print(f"  {k:34} {v:.4f}  (เฉลี่ยจาก {n} การทดสอบ)")
    ckpt = next(p for p in [EXPERIMENTS_DIR / EXP / "combined_clean" / "best.pt",
                            EXPERIMENTS_DIR / "generalization" / "random_D1" / "best.pt"]
                if p.exists())
    model, _ = load_model(ckpt, device)
    ms = round(inference_time_ms(model, device, cfg["img_size"]), 2)
    print(f"\n  inference โมเดลเดียว {ms} ms/ภาพ | ensemble 3 โมเดล ~{ms * 3:.1f} ms/ภาพ")
    save_json(dict(ranking=[dict(strategy=k, loso_macro_f1=v, n=n) for k, v, n in ranked],
                   single_model_inference_ms=ms,
                   note="ensemble 3 โมเดลใช้เวลา inference ประมาณ 3 เท่าของโมเดลเดียว"),
              EXPERIMENTS_DIR / EXP / "recommendation.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/strategy.yaml")
    ap.add_argument("--stage", default="all",
                    choices=["all", "combined", "loso", "ensemble"],
                    help="combined/loso ต้องเทรน ส่วน ensemble ใช้โมเดลที่มีอยู่แล้ว")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit-per-class", type=int)
    args = ap.parse_args()

    cfg = load_config(args.config, epochs=args.epochs, limit_per_class=args.limit_per_class)
    data = prepare(cfg)
    set_sources(data, cfg)
    splits = source_splits(data, cfg)
    for s in SOURCES:
        print(f"  {s}: train {len(splits[s]['train'])} | val {len(splits[s]['val'])} "
              f"| test {len(splits[s]['test'])}")

    t0 = time.time()
    if args.stage in ("all", "combined"):
        stage_combined(data, splits, cfg)
    if args.stage in ("all", "loso"):
        stage_loso(data, splits, cfg)
    if args.stage in ("all", "ensemble"):
        stage_ensemble(data, splits, cfg)
    print(f"\nใช้เวลาทั้งหมด {(time.time() - t0) / 60:.1f} นาที")
    print("ผล: results/tables/strategy_comparison.csv | experiments/strategy/recommendation.json")


if __name__ == "__main__":
    main()
