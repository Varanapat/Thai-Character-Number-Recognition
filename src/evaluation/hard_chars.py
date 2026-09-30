"""Experiment C — Hard Character Mining

หา "ตัวอักษรยาก" จาก confusion matrix ของโมเดลจริง (ไม่กำหนดล่วงหน้า)
แล้วสร้าง sampling weight สำหรับ fine-tuning รอบถัดไป
"""
import numpy as np

from src.utils.common import save_json, thai_char


def find_hard_characters(metrics, classes, f1_threshold=0.9, top_k=10, min_support=20):
    """คลาสที่โมเดลทำได้แย่ที่สุด = hard character

    metrics: ผลจาก evaluation.metrics.evaluate_split

    กติกา
    - ไม่นับคลาสที่มีตัวอย่างใน test น้อยกว่า min_support เพราะ F1 ของคลาสที่มี 1-2 รูป
      แกว่งจนไม่มีความหมาย (คลาสแบบนี้คือปัญหา "ข้อมูลน้อย" ไม่ใช่ "ตัวอักษรยาก")
    - ปกติใช้เกณฑ์ F1 < f1_threshold แต่ถ้าโมเดลเก่งจนไม่มีคลาสไหนต่ำกว่าเกณฑ์เลย
      จะเลือก top_k คลาสที่ F1 ต่ำสุดมาแทน เพื่อให้การทดลองเดินต่อได้
    """
    rows = [dict(code=code, char=m["char"], f1=m["f1"], recall=m["recall"],
                 precision=m["precision"], support=m["support"])
            for code, m in metrics["per_class"].items() if m["support"] >= min_support]
    rows.sort(key=lambda r: r["f1"])
    hard = [r for r in rows if r["f1"] < f1_threshold]
    if len(hard) < (top_k or 0):
        hard = rows[:top_k]  # fallback: เอาคลาสที่แย่ที่สุดเท่าที่มี
    elif top_k:
        hard = hard[:top_k]
    return hard


def hard_pairs(metrics, min_count=5):
    """คู่ที่สับสนกันจริงจากผลโมเดล (ใช้ตั้ง targeted augmentation)"""
    return [p for p in metrics["confused_pairs"] if p["count"] >= min_count]


def build_sampling_weights(data, hard_codes, boost=3.0):
    """คืน weight ต่อ sample สำหรับ fine-tuning: คลาสยากถูกสุ่มบ่อยขึ้น boost เท่า"""
    hard_idx = {data.classes.index(c) for c in hard_codes if c in data.classes}
    w = np.ones(len(data.y), np.float64)
    w[np.isin(data.y, list(hard_idx))] = boost
    return w


def hard_subset(data, ids, hard_codes):
    """เลือกเฉพาะ sample ของคลาสยาก — ใช้ทำ fine-tune เฉพาะกลุ่ม"""
    hard_idx = [data.classes.index(c) for c in hard_codes if c in data.classes]
    return ids[np.isin(data.y[ids], hard_idx)]


def save_hard_set(hard, pairs, path):
    return save_json(dict(
        n_hard=len(hard),
        hard_characters=hard,
        confused_pairs=pairs,
        note="F1 ต่ำกว่าเกณฑ์ = hard character; ใช้ build_sampling_weights() ต่อสำหรับ fine-tuning",
    ), path)


def summarize(hard, top=10):
    lines = [f"hard characters: {len(hard)} คลาส"]
    for r in hard[:top]:
        lines.append(f"  {r['code']} ({r['char']}): F1 {r['f1']:.3f} | recall {r['recall']:.3f} "
                     f"| support {r['support']}")
    return "\n".join(lines)
