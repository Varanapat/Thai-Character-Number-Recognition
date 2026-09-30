"""S1 — D1 Only

เทรน MobileNetV3-Small ด้วย D1 อย่างเดียว
ใช้เป็น baseline ว่าโมเดลติดลักษณะเฉพาะของ D1 แค่ไหน

รันเพื่อหาคำตอบ
    python3 strategies/S1_d1_only.py

ได้ output 2 ไฟล์
    results/predictions/S1_output.csv   ผลรายรูป: path + คลาสที่ทำนาย + เฉลยจริง + ถูก/ผิด
    results/tables/strategy_results.csv     สรุป accuracy และ macro F1 (ต่อท้ายรวมทุก strategy)

จำลองว่าเจอแหล่งใหม่: ตัดบางแหล่งออก แหล่งที่เหลือกลายเป็นชุดทดสอบที่ไม่เคยเห็น
    python3 strategies/S1_d1_only.py --train-sources D1 D2

ตรวจว่าโค้ดรันได้ก่อน (ไม่ถึงนาที ไม่ทับผลจริง)
    python3 strategies/S1_d1_only.py --epochs 1 --limit-per-class 20

วันจริง: ทำนายรูปใหม่ -> submission/S1_output.csv
    python3 strategies/S1_d1_only.py path/to/new_images
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.strategy_runner import main

if __name__ == "__main__":
    main("S1", "S1_d1_only", method="single", sources=["D1"])
