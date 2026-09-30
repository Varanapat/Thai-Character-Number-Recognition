"""S2b — D2 Only

เทรนด้วย D2 อย่างเดียว
เพิ่มให้ครบทั้งสามแหล่ง และจำเป็นสำหรับใช้เป็นสมาชิกของ ensemble

รันเพื่อหาคำตอบ
    python3 S2b_d2_only.py

ได้ output 2 ไฟล์
    results/predictions/S2b_output.csv   ผลรายรูป: path + คลาสที่ทำนาย + เฉลยจริง + ถูก/ผิด
    results/tables/strategy_results.csv     สรุป accuracy และ macro F1 (ต่อท้ายรวมทุก strategy)

จำลองว่าเจอแหล่งใหม่: ตัดบางแหล่งออก แหล่งที่เหลือกลายเป็นชุดทดสอบที่ไม่เคยเห็น
    python3 S2b_d2_only.py --train-sources D1 D2

ตรวจว่าโค้ดรันได้ก่อน (ไม่ถึงนาที ไม่ทับผลจริง)
    python3 S2b_d2_only.py --epochs 1 --limit-per-class 20

วันจริง: ทำนายรูปใหม่ -> submission/S2b_output.csv
    python3 S2b_d2_only.py path/to/new_images
"""
from src.strategy_runner import main

if __name__ == "__main__":
    main("S2b", "S2b_d2_only", method="single", sources=["D2"])
