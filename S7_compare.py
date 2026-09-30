"""S7 — สรุปเปรียบเทียบทุก strategy จาก results/tables/strategy_results.csv

ไม่เทรนอะไรทั้งสิ้น แค่อ่านผลที่ไฟล์ S1-S6 บันทึกไว้แล้วจัดตารางให้อ่านง่าย
ตอบคำถามข้อ 11 ของ Model_strategy.md: รวมความรู้ตอนเทรน (S3) เทียบกับ รวมตอน inference (S4-S6)

    python3 S7_compare.py
    python3 S7_compare.py --scope in_domain    # ดูสนามในโดเมนแทน
    python3 S7_compare.py --scope all
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
TABLE = ROOT / "results" / "tables" / "strategy_results.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", default="unseen", choices=["unseen", "in_domain", "all"],
                    help="unseen = แหล่งที่โมเดลไม่เคยเห็น (ใช้ตัดสินสำหรับข้อมูลวันจริง)")
    ap.add_argument("--csv", default=str(TABLE))
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"ยังไม่มี {path} — รันไฟล์ S1-S6 อย่างน้อยหนึ่งไฟล์ก่อน")
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if "[smoke]" not in r["strategy"]]
    if args.scope != "all":
        rows = [r for r in rows if r["scope"] == args.scope]
    if not rows:
        raise SystemExit(
            f"ยังไม่มีผลใน scope = {args.scope}\n"
            "ถ้าอยากได้สนาม unseen ให้รันโดยตัดบางแหล่งออก เช่น\n"
            "  python3 S3_combined.py --train-sources D1 D2\n"
            "  python3 S5_prob_average.py --members D1 D2")

    latest = {}   # เก็บเฉพาะผลล่าสุดของแต่ละ (strategy, เทรนด้วย, ทดสอบกับ)
    for r in rows:
        latest[(r["strategy"], r["trained_on"], r["test_source"])] = r

    by_run = defaultdict(list)
    for (strategy, trained, _), r in latest.items():
        by_run[(strategy, trained)].append(r)

    print(f"\n=== ผลรายแหล่ง (scope = {args.scope}) ===")
    print(f"{'strategy':22} {'เทรนด้วย':12} {'ทดสอบกับ':9} {'acc':>8} {'macro F1':>9} {'ms':>7}")
    for (strategy, trained), rs in sorted(by_run.items()):
        for r in sorted(rs, key=lambda x: x["test_source"]):
            print(f"{strategy:22} {trained:12} {r['test_source']:9} "
                  f"{float(r['accuracy']):8.4f} {float(r['macro_f1']):9.4f} "
                  f"{float(r.get('inference_ms') or 0):7.2f}")

    print(f"\n=== อันดับ: macro F1 เฉลี่ยทุกแหล่งใน scope นี้ ===")
    ranked = sorted(((s, t, sum(float(r["macro_f1"]) for r in rs) / len(rs),
                      sum(float(r["accuracy"]) for r in rs) / len(rs), len(rs),
                      float(rs[0].get("inference_ms") or 0))
                     for (s, t), rs in by_run.items()), key=lambda x: -x[2])
    print(f"{'อันดับ':6} {'strategy':22} {'เทรนด้วย':12} {'macro F1':>9} {'acc':>8} "
          f"{'แหล่ง':>6} {'ms':>7}")
    for i, (s, t, f1, acc, n, ms) in enumerate(ranked, 1):
        print(f"{i:<6} {s:22} {t:12} {f1:9.4f} {acc:8.4f} {n:6} {ms:7.2f}")

    best = ranked[0]
    print(f"\nดีที่สุดใน scope นี้: {best[0]} (เทรนด้วย {best[1]}) "
          f"macro F1 {best[2]:.4f} | inference {best[5]:.2f} ms/ภาพ")
    print(f"วันจริงให้รัน: python3 {best[0]}.py path/to/new_images")

    singles = [r for r in ranked if r[0].startswith(("S1", "S2", "S3"))]
    ens = [r for r in ranked if r[0].startswith(("S4", "S5", "S6"))]
    if singles and ens:
        print(f"\n=== S7: รวมความรู้ตอนเทรน เทียบกับ รวมตอน inference ===")
        print(f"  โมเดลเดียวที่ดีที่สุด : {singles[0][0]:22} macro F1 {singles[0][2]:.4f} "
              f"| {singles[0][5]:.2f} ms")
        print(f"  ensemble ที่ดีที่สุด  : {ens[0][0]:22} macro F1 {ens[0][2]:.4f} "
              f"| {ens[0][5]:.2f} ms")
        diff = ens[0][2] - singles[0][2]
        print(f"  ส่วนต่าง {diff:+.4f} -> " +
              ("ensemble ดีกว่า" if diff > 0 else "โมเดลเดียวดีกว่า"))
        if 0 < diff < 0.005:
            print("  หมายเหตุ: ต่างกันน้อยมาก ถ้าเวลา inference สำคัญ เลือกโมเดลเดียวคุ้มกว่า")

    print("\nผลรายรูปของแต่ละ strategy อยู่ที่ results/predictions/<S?>_output.csv")


if __name__ == "__main__":
    main()
