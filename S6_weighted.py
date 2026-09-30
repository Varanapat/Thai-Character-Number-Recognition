"""S6 — Weighted Probability Ensemble

ถ่วงน้ำหนักแต่ละโมเดลด้วยผลบน validation ของแหล่งอื่นในทีม (ข้อ 10: ห้ามใช้ข้อมูล test)
ต้องรัน S1, S2, S2b ให้มีโมเดลก่อน

รันเพื่อหาคำตอบ
    python3 S6_weighted.py

ได้ output 2 ไฟล์
    results/predictions/S6_output.csv   ผลรายรูป: path + คลาสที่ทำนาย + เฉลยจริง + ถูก/ผิด
    results/tables/strategy_results.csv     สรุป accuracy และ macro F1 (ต่อท้ายรวมทุก strategy)

จำลองว่าเจอแหล่งใหม่: ตัดบางแหล่งออก แหล่งที่เหลือกลายเป็นชุดทดสอบที่ไม่เคยเห็น
    python3 S6_weighted.py --members D1 D2

ตรวจว่าโค้ดรันได้ก่อน (ไม่ถึงนาที ไม่ทับผลจริง)
    python3 S6_weighted.py --epochs 1 --limit-per-class 20

วันจริง: ทำนายรูปใหม่ -> submission/S6_output.csv
    python3 S6_weighted.py path/to/new_images
"""
from src.strategy_runner import main

if __name__ == "__main__":
    main("S6", "S6_weighted", method="weighted", members=["D1", "D2", "D3"])
