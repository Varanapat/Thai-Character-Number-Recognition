"""S2 — D3 Only

เทรนด้วย D3 อย่างเดียว
สมมติฐานคือ D3 อาจให้ตัวแทนของตัวอักษรที่ถ่ายทอดข้ามแหล่งได้ดีกว่า

รันเพื่อหาคำตอบ
    python3 strategies/S2_d3_only.py

ได้ output 2 ไฟล์
    results/predictions/S2_output.csv   ผลรายรูป: path + คลาสที่ทำนาย + เฉลยจริง + ถูก/ผิด
    results/tables/strategy_results.csv     สรุป accuracy และ macro F1 (ต่อท้ายรวมทุก strategy)

จำลองว่าเจอแหล่งใหม่: ตัดบางแหล่งออก แหล่งที่เหลือกลายเป็นชุดทดสอบที่ไม่เคยเห็น
    python3 strategies/S2_d3_only.py --train-sources D1 D2

ตรวจว่าโค้ดรันได้ก่อน (ไม่ถึงนาที ไม่ทับผลจริง)
    python3 strategies/S2_d3_only.py --epochs 1 --limit-per-class 20

วันจริง: ทำนายรูปใหม่ -> submission/S2_output.csv
    python3 strategies/S2_d3_only.py path/to/new_images
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.strategy_runner import main

if __name__ == "__main__":
    main("S2", "S2_d3_only", method="single", sources=["D3"])
