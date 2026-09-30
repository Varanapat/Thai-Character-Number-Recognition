"""S4 — Majority Vote

โหวตจากโมเดลรายแหล่งสามตัว ไม่ต้องเทรนใหม่
ต้องรัน S1, S2, S2b ให้มีโมเดลก่อน

รันเพื่อหาคำตอบ
    python3 S4_majority_vote.py

ได้ output 2 ไฟล์
    results/predictions/S4_output.csv   ผลรายรูป: path + คลาสที่ทำนาย + เฉลยจริง + ถูก/ผิด
    results/tables/strategy_results.csv     สรุป accuracy และ macro F1 (ต่อท้ายรวมทุก strategy)

จำลองว่าเจอแหล่งใหม่: ตัดบางแหล่งออก แหล่งที่เหลือกลายเป็นชุดทดสอบที่ไม่เคยเห็น
    python3 S4_majority_vote.py --members D1 D2

ตรวจว่าโค้ดรันได้ก่อน (ไม่ถึงนาที ไม่ทับผลจริง)
    python3 S4_majority_vote.py --epochs 1 --limit-per-class 20

วันจริง: ทำนายรูปใหม่ -> submission/S4_output.csv
    python3 S4_majority_vote.py path/to/new_images
"""
from src.strategy_runner import main

if __name__ == "__main__":
    main("S4", "S4_majority_vote", method="majority_vote", members=["D1", "D2", "D3"])
