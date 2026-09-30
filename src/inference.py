"""ทำนายรูปใหม่ด้วยกลยุทธ์ที่ล็อกไว้ แล้วเขียน CSV (Model_strategy.md ข้อ 15, 22)

ใช้ preprocessing ชุดเดียวกับตอนเทรนเสมอ และไม่มี augmentation

ตัวอย่าง
    # โมเดลเดียว
    python -m src.inference --images path/to/folder --strategy single \
        --models experiments/strategy/combined_clean/best.pt --out submission/predictions.csv

    # ensemble เฉลี่ย probability จากสามโมเดลรายแหล่ง
    python -m src.inference --images path/to/folder --strategy prob_average \
        --models experiments/generalization/random_D1/best.pt \
                 experiments/generalization/random_D2/best.pt \
                 experiments/generalization/random_D3/best.pt \
        --out submission/predictions.csv

    # ถ่วงน้ำหนักตามที่คำนวณไว้จาก validation
    python -m src.inference ... --strategy weighted --weights-file experiments/strategy/ensemble_weights.json
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.datasets.thai_chars import EXTS, preprocess
from src.ensemble.combiners import load_model, majority_vote, predict_probs, prob_average
from src.utils.common import get_device, thai_char

STRATEGIES = ("single", "majority_vote", "prob_average", "weighted")


def list_images(path):
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="ไฟล์เดียวหรือโฟลเดอร์")
    ap.add_argument("--models", nargs="+", required=True, help="path ของ checkpoint (.pt)")
    ap.add_argument("--strategy", default="prob_average", choices=STRATEGIES)
    ap.add_argument("--weights", nargs="+", type=float, help="น้ำหนักของแต่ละโมเดล (strategy=weighted)")
    ap.add_argument("--weights-file", help="ไฟล์ ensemble_weights.json จาก run_strategy")
    ap.add_argument("--out", default="submission/predictions.csv")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--topk", type=int, default=3, help="จำนวนอันดับที่เก็บไว้ในคอลัมน์ top_k")
    args = ap.parse_args()

    paths = list_images(args.images)
    if not paths:
        raise SystemExit(f"ไม่พบรูปใน {args.images}")
    device = get_device()
    print(f"{len(paths)} รูป | {len(args.models)} โมเดล | strategy={args.strategy} | {device}")

    models, names, classes, img_size = [], [], None, None
    for mp in args.models:
        model, ckpt = load_model(mp, device)
        models.append(model)
        names.append(Path(mp).parent.name)
        if classes is None:
            classes, img_size = ckpt["classes"], ckpt["img_size"]
        elif ckpt["classes"] != classes:
            raise SystemExit("checkpoint มีรายการคลาสไม่ตรงกัน รวมผลไม่ได้")

    x = np.stack([preprocess(p, img_size) for p in paths])
    probs = [predict_probs(m, x, device, args.batch_size) for m in models]

    if args.strategy == "single" or len(probs) == 1:
        final = probs[0]
    elif args.strategy == "majority_vote":
        final = np.eye(len(classes))[majority_vote(probs)]
    elif args.strategy == "prob_average":
        final = prob_average(probs)
    else:
        w = args.weights
        if not w and args.weights_file:
            wd = json.loads(Path(args.weights_file).read_text())["weights"]
            w = [wd[n.replace("random_", "").replace("single_", "")] for n in names]
        if not w:
            raise SystemExit("strategy=weighted ต้องระบุ --weights หรือ --weights-file")
        final = prob_average(probs, w)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["filename"]
    for n in names:
        cols += [f"{n}_pred", f"{n}_char", f"{n}_conf"]
    cols += ["final_pred", "final_char", "final_conf", "top_k", "agree"]

    with open(out, "w", newline="", encoding="utf-8") as f:
        w_ = csv.DictWriter(f, fieldnames=cols)
        w_.writeheader()
        for i, p in enumerate(paths):
            row = {"filename": p.name}
            per_model = []
            for n, pr in zip(names, probs):
                k = int(pr[i].argmax())
                per_model.append(k)
                row[f"{n}_pred"] = classes[k]
                row[f"{n}_char"] = thai_char(classes[k])
                row[f"{n}_conf"] = round(float(pr[i][k]), 4)
            k = int(final[i].argmax())
            top = np.argsort(-final[i])[:args.topk]
            row |= {"final_pred": classes[k], "final_char": thai_char(classes[k]),
                    "final_conf": round(float(final[i][k]), 4),
                    "top_k": " | ".join(f"{classes[j]}({thai_char(classes[j])}):{final[i][j]:.3f}"
                                        for j in top),
                    "agree": int(len(set(per_model)) == 1)}
            w_.writerow(row)

    conf = final.max(1)
    print(f"เขียนผลลง {out}")
    print(f"ความมั่นใจเฉลี่ย {conf.mean():.3f} | ต่ำกว่า 0.5 อยู่ {int((conf < 0.5).sum())} รูป")
    if len(probs) > 1:
        preds = np.stack([p.argmax(1) for p in probs])
        print(f"โมเดลเห็นตรงกันทุกตัว {float((preds == preds[0]).all(0).mean()):.1%} ของรูป")


if __name__ == "__main__":
    main()
