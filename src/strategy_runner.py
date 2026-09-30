"""เครื่องยนต์เบื้องหลังไฟล์ strategy_*.py (Model_strategy.md)

architecture ล็อกไว้ที่ MobileNetV3-Small ทุกไฟล์ ตัวแปรเดียวคือ "วิธีเทรน / วิธีรวมผล"
ค่าการเทรนทั้งหมดมาจาก configs/strategy_base.yaml ไฟล์เดียว จึงเทียบกันได้ยุติธรรม (ข้อ 20)

ทุกไฟล์บันทึกผลต่อท้ายลง results/tables/strategy_results.csv หนึ่งแถวต่อหนึ่ง test source
ตามตารางที่แนะนำในข้อ 14 ของเอกสาร

    strategy | test_source | scope | accuracy | macro_f1 | ...

scope มีสองแบบ
    in_domain  ทดสอบกับ test set ที่กันไว้ของแหล่งที่ใช้เทรน (โมเดลเคยเห็นแหล่งนี้)
    unseen     ทดสอบกับแหล่งที่ไม่ได้ใช้เทรนเลย ทั้งแหล่ง  <-- ใช้ตัวนี้ตัดสินสำหรับข้อมูลวันจริง
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.datasets.splits import random_split
from src.ensemble.combiners import (agreement_stats, load_model, majority_vote, predict_probs,
                                    prob_average)
from src.evaluation.metrics import compute_metrics, save_report
from src.models.factory import count_params, inference_time_ms
from src.training.run_experiment import prepare
from src.training.trainer import train_model
from src.utils.common import ROOT, get_device, load_config, save_json, thai_char

ARCH = "mobilenet_v3_small"          # ล็อกตาม Model_strategy.md ข้อ 2
MODELS_DIR = ROOT / "models"
TABLE = ROOT / "results" / "tables" / "strategy_results.csv"
DEFAULT_CONFIG = ROOT / "configs" / "strategy_base.yaml"
FALLBACK_CKPT = ROOT / "experiments" / "generalization" / "random_{s}" / "best.pt"
MAPPING_CSV = ROOT / "Mapping.csv"


def load_answer_mapping():
    """โหลด Mapping.csv (folder=รหัส TIS-620, class_id=คำตอบที่โจทย์ต้องการ)

    คืน dict {รหัส TIS-620 (str): class_id (str)} — คืน dict ว่างถ้ายังไม่มีไฟล์นี้
    """
    if not MAPPING_CSV.exists():
        return {}
    with open(MAPPING_CSV, encoding="utf-8-sig") as f:
        return {row["folder"].strip(): row["class_id"].strip() for row in csv.DictReader(f)}


# ---------------------------------------------------------------- พื้นฐาน


def setup(args):
    """โหลด config + ข้อมูล + split ของแต่ละแหล่ง (ใช้ seed เดียวกันทุกไฟล์)"""
    cfg = load_config(args.config, epochs=args.epochs, limit_per_class=args.limit_per_class)
    cfg["model"] = ARCH
    data = prepare(cfg)
    sources = sorted({s for s in set(data.sources.tolist()) if s != "unknown"})
    splits = {s: random_split(data, sources=s, val_ratio=cfg["val_ratio"],
                              test_ratio=cfg["test_ratio"], seed=cfg["seed"], verbose=False)
              for s in sources}
    return cfg, data, splits, sources


def append_row(row):
    """ต่อท้ายผลลง CSV กลาง (ทุก strategy ใช้ไฟล์เดียวกัน ไม่เขียนทับกัน)"""
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(TABLE, encoding="utf-8"))) if TABLE.exists() else []
    rows.append(row)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(TABLE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)


def model_path(source, smoke=False):
    """โมเดลรายแหล่ง: ของเราเองก่อน ถ้าไม่มีใช้ของ Experiment D ที่เทรนโปรโตคอลเดียวกัน

    โมเดลจากรอบทดสอบ (_smoke_*) จะถูกใช้ก็ต่อเมื่อรันในโหมดทดสอบเท่านั้น
    """
    if smoke:
        sm = MODELS_DIR / f"_smoke_mobilenet_{source.lower()}" / "best.pt"
        if sm.exists():
            return sm
    own = MODELS_DIR / f"mobilenet_{source.lower()}" / "best.pt"
    if own.exists():
        return own
    fallback = Path(str(FALLBACK_CKPT).format(s=source))
    return fallback if fallback.exists() else None


def write_per_image(code, rows, folder):
    """ผลรายรูป: path + คลาสที่ทำนาย (ไฟล์ละ strategy ตามที่ตกลงกันไว้)"""
    out = ROOT / folder / f"{code}_output.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"  ผลรายรูป -> {out.relative_to(ROOT)} ({len(rows)} แถว)")
    return out


def evaluate_and_log(code, strategy, probs_fn, data, splits, trained_sources, cfg, extra=None):
    """ประเมินกับทุกแหล่ง แล้วบันทึกทีละแถว — แหล่งที่ไม่ได้เทรนคือสนามตัดสิน"""
    device = get_device(cfg.get("device", "auto"))
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    unseen_scores, per_image = [], []
    for s in sorted(splits):
        in_domain = s in trained_sources
        if in_domain:
            ids = splits[s].get("test")
            if ids is None:
                ids = splits[s].get("val")
                if ids is None:
                    print(f"  [เตือน] {s}: ไม่มี test/val split ให้ประเมิน (test_ratio=val_ratio=0) ข้ามแหล่งนี้")
                    continue
        else:
            ids = data.where(sources=s)
        probs = probs_fn(ids)
        pred, conf = probs.argmax(1), probs.max(1)
        for i, idx in enumerate(ids):
            true_code = data.classes[data.y[idx]]
            pred_code = data.classes[pred[i]]
            per_image.append(dict(
                path=str(Path(data.paths[idx]).relative_to(ROOT)),
                predicted_class=pred_code,
                predicted_char=thai_char(pred_code),
                confidence=round(float(conf[i]), 4),
                true_class=true_code, true_char=thai_char(true_code),
                correct=int(pred_code == true_code), scope="in_domain" if in_domain else "unseen",
                source=s))
        m = compute_metrics(data.y[ids], probs.argmax(1), data.classes, True)
        scope = "in_domain" if in_domain else "unseen"
        row = dict(timestamp=stamp, strategy=strategy, trained_on="+".join(trained_sources),
                   test_source=s, scope=scope, accuracy=round(m["accuracy"], 4),
                   macro_f1=round(m["macro_f1"], 4), n_test=m["n_samples"],
                   **(extra or {}))
        append_row(row)
        print(f"  {s} ({scope}): acc {m['accuracy']:.4f} | macro F1 {m['macro_f1']:.4f} "
              f"| {m['n_samples']} รูป")
        if not in_domain:
            unseen_scores.append(m["macro_f1"])
    write_per_image(code, per_image, "results/predictions")
    if unseen_scores:
        print(f"  >> macro F1 เฉลี่ยบนแหล่งที่ไม่เคยเห็น: {np.mean(unseen_scores):.4f}")
    else:
        print("  >> ไม่มีแหล่งที่ไม่เคยเห็นเลย (เทรนครบทุกแหล่ง) "
              "— ถ้าอยากวัดสนามนี้ให้ใช้ --train-sources ตัดบางแหล่งออก")
    print(f"  บันทึกลง {TABLE.relative_to(ROOT)}")


# ---------------------------------------------------------------- strategy แบบโมเดลเดียว


def run_single(code, strategy, sources, args):
    cfg, data, splits, all_sources = setup(args)
    sources = [s for s in (args.train_sources or sources) if s in all_sources]
    if not sources:
        raise SystemExit(f"ไม่พบแหล่งที่ระบุในข้อมูล (มีอยู่: {all_sources})")
    unseen = [s for s in all_sources if s not in sources]
    print(f"\n=== {strategy} | {ARCH} | เทรนด้วย {'+'.join(sources)} "
          f"| แหล่งที่ไม่เคยเห็น: {', '.join(unseen) or 'ไม่มี'} ===")

    keys = [k for k in ("train", "val", "test") if all(k in splits[s] for s in sources)]
    split = {k: np.concatenate([splits[s][k] for s in sources]) for k in keys}
    print("  " + " | ".join(f"{k} {len(split[k])}" for k in ("train", "val", "test") if k in split))
    # รอบทดสอบ (--limit-per-class) เขียนแยกโฟลเดอร์ ไม่ให้ทับโมเดลจริงที่จะใช้วันงาน
    smoke = bool(cfg.get("limit_per_class"))
    if smoke:
        strategy += " [smoke]"
        print("  โหมดทดสอบ: ผลและโมเดลถูกเก็บแยก ไม่ทับของจริง")
    out_dir = MODELS_DIR / (("_smoke_" if smoke else "") + "mobilenet_"
                            + "_".join(s.lower() for s in sources))
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, result = train_model(data, split, cfg, ARCH, cfg["augmentation"], out_dir)
    minutes = round((time.time() - t0) / 60, 1)
    device = get_device(cfg.get("device", "auto"))
    ms = round(inference_time_ms(model, device, cfg["img_size"]), 2)

    def probs_fn(ids):
        return predict_probs(model, data.x[ids], device, cfg["eval_batch_size"])

    evaluate_and_log(code, strategy, probs_fn, data, splits, sources, cfg,
                     dict(inference_ms=ms, train_minutes=minutes,
                          params_m=round(count_params(model) / 1e6, 2),
                          val_acc=round(result["best_score"], 4),
                          checkpoint=str((out_dir / "best.pt").relative_to(ROOT)),
                          augmentation=cfg["augmentation"], epochs=cfg["epochs"],
                          seed=cfg["seed"]))
    save_json(dict(strategy=strategy, sources=sources, minutes=minutes), out_dir / "strategy.json")


# ---------------------------------------------------------------- strategy แบบ ensemble


def ensemble_weights(members, data, splits, cfg, device, smoke=False):
    """S6 — น้ำหนักจาก val ของแหล่งสมาชิกเท่านั้น (ข้อ 10: ห้ามใช้ข้อมูลของ test)

    คะแนนของแต่ละโมเดล = macro F1 บน val ของ "แหล่งอื่นในทีม" ซึ่งมันไม่ได้เทรน
    """
    scores = {}
    for s in members:
        model, _ = load_model(model_path(s, smoke), device)
        vals = []
        for other in members:
            if other == s:
                continue
            ids = splits[other]["val"]
            pr = predict_probs(model, data.x[ids], device, cfg["eval_batch_size"])
            vals.append(compute_metrics(data.y[ids], pr.argmax(1), data.classes, True)["macro_f1"])
        scores[s] = float(np.mean(vals)) if vals else 1.0
        del model
    total = sum(scores.values())
    weights = {k: v / total for k, v in scores.items()}
    print("  น้ำหนัก (จาก val ของสมาชิกเท่านั้น):", {k: f"{v:.3f}" for k, v in weights.items()})
    save_json(dict(weights=weights, val_macro_f1=scores),
              MODELS_DIR / "ensemble_weights.json")
    return weights


def run_ensemble(code, strategy, method, args):
    cfg, data, splits, all_sources = setup(args)
    members = [s for s in (args.members or all_sources) if s in all_sources]
    smoke = bool(cfg.get("limit_per_class"))
    missing = [s for s in members if model_path(s, smoke) is None]
    if missing:
        raise SystemExit(
            f"ยังไม่มีโมเดลของแหล่ง {', '.join(missing)}\n"
            f"ให้รันไฟล์ strategy_s1_d1_only.py / strategy_s2_d3_only.py / "
            f"strategy_s2b_d2_only.py ของแหล่งนั้นก่อน")
    unseen = [s for s in all_sources if s not in members]
    if cfg.get("limit_per_class"):
        strategy += " [smoke]"
        print("  โหมดทดสอบ: ผลถูกบันทึกแยกด้วยป้าย [smoke]")
    print(f"\n=== {strategy} | สมาชิก {'+'.join(members)} "
          f"| แหล่งที่ไม่เคยเห็น: {', '.join(unseen) or 'ไม่มี'} ===")
    for s in members:
        print(f"  โมเดล {s}: {model_path(s, smoke).relative_to(ROOT)}")

    device = get_device(cfg.get("device", "auto"))
    weights = (ensemble_weights(members, data, splits, cfg, device, smoke)
               if method == "weighted" else None)
    models = {s: load_model(model_path(s, smoke), device)[0] for s in members}
    ms = round(sum(inference_time_ms(m, device, cfg["img_size"]) for m in models.values()), 2)

    def probs_fn(ids):
        per_model = [predict_probs(models[s], data.x[ids], device, cfg["eval_batch_size"])
                     for s in members]
        stats = agreement_stats(per_model)
        print(f"    โมเดลเห็นตรงกัน {stats['agreement_rate']:.1%} "
              f"| ไม่ตรงกัน {stats['n_disagree']} รูป")
        if method == "majority_vote":
            return np.eye(len(data.classes))[majority_vote(per_model)]
        if method == "weighted":
            return prob_average(per_model, [weights[s] for s in members])
        return prob_average(per_model)

    evaluate_and_log(code, strategy, probs_fn, data, splits, members, cfg,
                     dict(inference_ms=ms, train_minutes=0, augmentation=cfg["augmentation"],
                          epochs=cfg["epochs"], seed=cfg["seed"],
                          checkpoint=" + ".join(str(model_path(s, smoke).relative_to(ROOT))
                                                for s in members)))


# ---------------------------------------------------------------- ทำนายรูปใหม่


def single_ckpt(ckpt_dir, sources):
    """หา checkpoint ของ strategy แบบโมเดลเดียว ถ้ายังไม่เคยเทรนเองให้ใช้ของ Experiment D

    โมเดลจาก Experiment D ใช้ architecture, augmentation, seed และจำนวน epoch ชุดเดียวกัน
    จึงใช้แทนกันได้ และทำให้ทดสอบ S1-S3 ได้ทันทีโดยไม่ต้องเทรนใหม่
    """
    own = ckpt_dir / "best.pt"
    if own.exists():
        return own
    if len(sources) == 1:
        alt = Path(str(FALLBACK_CKPT).format(s=sources[0]))
    else:
        alt = ROOT / "experiments" / "generalization" / "combined_all_sources" / "best.pt"
    if alt.exists():
        print(f"  ยังไม่มี {own.relative_to(ROOT)} -> ใช้โมเดลจาก Experiment D: "
              f"{alt.relative_to(ROOT)}")
        return alt
    return own


def predict(code, strategy, method, members, ckpt_dir, sources, args):
    """โหมดวันจริง — ใช้โมเดลที่เทรนไว้ ทำนายรูปใหม่ แล้วเขียน CSV ตามข้อ 15"""
    from src.datasets.thai_chars import EXTS, preprocess

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    mapping = load_answer_mapping()
    if mapping:
        print(f"  ใช้ {MAPPING_CSV.name} แปลงรหัส TIS-620 -> answer ({len(mapping)} คลาส)")
    if method == "single":
        paths_ckpt = {"model": Path(args.checkpoint) if args.checkpoint
                      else single_ckpt(ckpt_dir, sources)}
    else:
        paths_ckpt = {s: model_path(s) for s in (args.members or members)}
    for k, v in paths_ckpt.items():
        if v is None or not Path(v).exists():
            raise SystemExit(f"ไม่พบโมเดล {k}: {v} — ต้องรันโหมดเทรนก่อน")

    src = Path(args.images)
    files = [src] if src.is_file() else sorted(f for f in src.rglob("*")
                                               if f.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"ไม่พบรูปใน {args.images}")

    loaded = {k: load_model(v, device) for k, v in paths_ckpt.items()}
    classes = next(iter(loaded.values()))[1]["classes"]
    size = next(iter(loaded.values()))[1]["img_size"]
    print(f"\n=== {strategy}: ทำนาย {len(files)} รูป ด้วย {len(loaded)} โมเดล ===")

    x = np.stack([preprocess(p, size) for p in files])
    per_model = {k: predict_probs(m, x, device, cfg["eval_batch_size"])
                 for k, (m, _) in loaded.items()}
    probs = list(per_model.values())
    if method == "majority_vote":
        final = np.eye(len(classes))[majority_vote(probs)]
    elif method == "weighted":
        wfile = MODELS_DIR / "ensemble_weights.json"
        if not wfile.exists():
            raise SystemExit("ไม่พบ models/ensemble_weights.json — รันโหมดเทรน/ประเมินของ S6 ก่อน")
        import json
        w = json.loads(wfile.read_text())["weights"]
        final = prob_average(probs, [w[k] for k in per_model])
    elif method == "single":
        final = probs[0]
    else:
        final = prob_average(probs)

    out = Path(args.out or ROOT / "submission" / f"{code}_output.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    truth = {}
    if args.labels == "filename":
        from src.datasets.filename_labels import label_code, summarize
        info = summarize(files)
        print(f"  อ่านเฉลยจากชื่อไฟล์: {info['labeled']}/{info['total']} รูปมีคลาสที่โมเดลรู้จัก "
              f"| นอกคลาสที่รู้จัก {info['out_of_vocab']} รูป")
        if info["unparsed"]:
            print(f"  [เตือน] ชื่อที่แปลไม่ได้: {', '.join(info['unparsed'][:8])}")
        truth = {str(f): label_code(f)[0] for f in files}

    cols = ["path", "predicted_class", "predicted_char", "confidence"]
    if mapping:
        cols.insert(3, "answer")   # ก่อน confidence, หลัง predicted_char
    if truth:
        cols += ["true_class", "true_char", "correct"]
    for k in per_model:
        cols += [f"{k}_pred", f"{k}_conf"]
    cols += ["top_3", "agree"]
    unmapped = set()
    with open(out, "w", newline="", encoding="utf-8") as f:
        w_ = csv.DictWriter(f, fieldnames=cols)
        w_.writeheader()
        for i, p in enumerate(files):
            j = int(final[i].argmax())
            top = np.argsort(-final[i])[:3]
            picks = []
            row = {"path": str(p), "predicted_class": classes[j],
                   "predicted_char": thai_char(classes[j]),
                   "confidence": round(float(final[i][j]), 4)}
            if mapping:
                row["answer"] = mapping.get(classes[j], "")
                if not row["answer"]:
                    unmapped.add(classes[j])
            for k, pr in per_model.items():
                t = int(pr[i].argmax())
                picks.append(t)
                row[f"{k}_pred"] = classes[t]
                row[f"{k}_conf"] = round(float(pr[i][t]), 4)
            if truth:
                tc = truth.get(str(p))
                row |= {"true_class": tc or "", "true_char": thai_char(tc) if tc else "",
                        "correct": "" if tc is None else int(classes[j] == tc)}
            row |= {"top_3": " | ".join(f"{classes[t]}({thai_char(classes[t])}):{final[i][t]:.3f}"
                                        for t in top),
                    "agree": int(len(set(picks)) == 1)}
            w_.writerow(row)
    conf = final.max(1)
    if mapping and unmapped:
        print(f"  [เตือน] {len(unmapped)} คลาสที่โมเดลทำนายไม่มีใน {MAPPING_CSV.name}: "
              f"{', '.join(sorted(unmapped))}")
    print(f"  เขียนผลลง {out}")
    print(f"  ความมั่นใจเฉลี่ย {conf.mean():.3f} | ต่ำกว่า 0.5 จำนวน {int((conf < 0.5).sum())} รูป")

    if truth:
        idx = [i for i, f in enumerate(files) if truth.get(str(f))]
        y_true = np.array([classes.index(truth[str(files[i])]) for i in idx])
        y_pred = final[idx].argmax(1)
        m = compute_metrics(y_true, y_pred, classes, True)
        print(f"\n  === ผลกับ {Path(args.images).name} ({len(idx)} รูปที่มีเฉลยในคลาสของเรา) ===")
        print(f"  accuracy {m['accuracy']:.4f} | macro F1 {m['macro_f1']:.4f}")
        worst = sorted(m["per_class"].items(), key=lambda kv: kv[1]["f1"])[:5]
        print("  คลาสที่แย่สุด: " +
              ", ".join(f"{c}({v['char']}) F1 {v['f1']:.2f}" for c, v in worst))
        append_row(dict(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"), strategy=code,
                        trained_on="+".join(args.members or args.train_sources
                                            or members or sources or []),
                        test_source=Path(args.images).name, scope="unseen",
                        accuracy=round(m["accuracy"], 4), macro_f1=round(m["macro_f1"], 4),
                        n_test=len(idx), note="เฉลยจากชื่อไฟล์"))
        print(f"  สรุปต่อท้ายลง {TABLE.relative_to(ROOT)} แล้ว")


# ---------------------------------------------------------------- entry point


def main(code, strategy, method="single", sources=None, members=None):
    """method: single | majority_vote | prob_average | weighted"""
    ap = argparse.ArgumentParser(description=f"{strategy} ({ARCH})")
    ap.add_argument("images_pos", nargs="?", metavar="IMAGES",
                    help="โฟลเดอร์รูปที่จะทำนาย ใส่มาแล้วเข้าโหมด predict ให้อัตโนมัติ")
    ap.add_argument("--mode", default=None, choices=["train", "evaluate", "predict"],
                    help="ไม่ใส่ = หาคำตอบ | ใส่โฟลเดอร์รูป = ทำนาย")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--train-sources", nargs="+", help="แหล่งที่ใช้เทรน (strategy แบบโมเดลเดียว)")
    ap.add_argument("--members", nargs="+", help="แหล่งของโมเดลสมาชิก (strategy แบบ ensemble)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit-per-class", type=int, help="ใช้ตรวจว่าโค้ดรันได้")
    ap.add_argument("--images", help="โฟลเดอร์รูปสำหรับโหมด predict")
    ap.add_argument("--out", help="ไฟล์ CSV ผลทำนาย")
    ap.add_argument("--labels", choices=["filename"],
                    help="อ่านเฉลยจากชื่อไฟล์ (รูปแบบ PrintAksorn) เพื่อคิด accuracy ให้ด้วย")
    ap.add_argument("--checkpoint", help="ระบุ .pt เอง (strategy แบบโมเดลเดียว)")
    args = ap.parse_args()
    args.images = args.images_pos or args.images
    if args.mode is None:
        args.mode = "predict" if args.images else ("train" if method == "single" else "evaluate")

    if args.mode == "predict":
        if not args.images:
            raise SystemExit("โหมด predict ต้องระบุ --images")
        srcs = args.train_sources or sources or []
        ckpt_dir = MODELS_DIR / ("mobilenet_" + "_".join(s.lower() for s in srcs))
        predict(code, strategy, method, members or [], ckpt_dir, srcs, args)
    elif method == "single":
        run_single(code, strategy, sources, args)
    else:
        run_ensemble(code, strategy, method, args)
