"""Custom CNN (baseline เขียนเอง ไม่มี pretrained weight)

เทรนและประเมิน (ผลต่อท้ายลง results/tables/architecture_results.csv)
    python arch_custom_cnn.py

เทรนเฉพาะบางแหล่ง แหล่งที่เหลือกลายเป็นชุดทดสอบแหล่งที่ไม่เคยเห็น
    python arch_custom_cnn.py --train-sources D1 D2

ทดสอบว่าโค้ดรันได้ก่อน (ไม่ถึงนาที)
    python arch_custom_cnn.py --epochs 1 --limit-per-class 20

วันจริง: ทำนายรูปใหม่ด้วยโมเดลที่เทรนไว้ -> submission/custom_cnn_predictions.csv
    python arch_custom_cnn.py --mode predict --images path/to/new_images
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # ให้ import src ได้จากโฟลเดอร์ย่อย

from src.arch_runner import main

if __name__ == "__main__":
    main("custom_cnn")
