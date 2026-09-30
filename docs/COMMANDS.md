# คำสั่งทั้งหมดที่ต้องใช้

architecture ล็อกที่ MobileNetV3-Small ทุกไฟล์ ค่าการเทรนอยู่ใน `configs/strategy_base.yaml`

---

## ขั้นที่ 0 — ตรวจว่าทุกอย่างรันได้ (ไม่ถึงนาทีต่อไฟล์ ไม่ทับผลจริง)

```bash
python3 strategies/S1_d1_only.py --epochs 1 --limit-per-class 20
```

---

## ขั้นที่ 1 — เทรนโมเดลรายแหล่ง (ต้องทำก่อน เพราะ S4/S5/S6 ใช้โมเดลพวกนี้)

```bash
python3 strategies/S1_d1_only.py
```
```bash
python3 strategies/S2_d3_only.py
```
```bash
python3 strategies/S2b_d2_only.py
```

## ขั้นที่ 2 — เทรนแบบรวมทุกแหล่ง

```bash
python3 strategies/S3_combined.py
```

## ขั้นที่ 3 — ensemble (ไม่ต้องเทรน ใช้โมเดลจากขั้นที่ 1)

```bash
python3 strategies/S4_majority_vote.py
```
```bash
python3 strategies/S5_prob_average.py
```
```bash
python3 strategies/S6_weighted.py
```

## ขั้นที่ 4 — จำลองว่าเจอแหล่งใหม่ (สำคัญที่สุดสำหรับการเลือกใช้วันจริง)

ตัด D3 ออกจากการเทรน แล้ว D3 กลายเป็นชุดทดสอบที่โมเดลไม่เคยเห็นเลย

```bash
python3 strategies/S3_combined.py --train-sources D1 D2
```
```bash
python3 strategies/S4_majority_vote.py --members D1 D2
```
```bash
python3 strategies/S5_prob_average.py --members D1 D2
```
```bash
python3 strategies/S6_weighted.py --members D1 D2
```

ทำซ้ำโดยตัดแหล่งอื่นออกได้ เช่น `--train-sources D1 D3` (D2 กลายเป็นแหล่งที่ไม่เคยเห็น)

## ขั้นที่ 5 — ดูผลสรุปว่าควรเลือก strategy ไหน

```bash
python3 strategies/S7_compare.py
```
```bash
python3 strategies/S7_compare.py --scope in_domain
```

---

## วันจริง — ทำนายรูปใหม่

ใส่ path ของโฟลเดอร์รูปต่อท้ายได้เลย

```bash
python3 strategies/S5_prob_average.py path/to/new_images
```

เปลี่ยนเป็นไฟล์ที่ชนะจาก S7 ได้ทุกตัว เช่น

```bash
python3 strategies/S3_combined.py path/to/new_images
```

กำหนดที่เก็บผลเองได้

```bash
python3 strategies/S5_prob_average.py path/to/new_images --out submission/final.csv
```

---

## ไฟล์ผลลัพธ์

| ไฟล์ | เนื้อหา |
|---|---|
| `results/predictions/S1_output.csv` … `S6_output.csv` | ผลรายรูปตอนหาคำตอบ: path, คลาสที่ทำนาย, เฉลยจริง, ถูก/ผิด, scope |
| `results/tables/strategy_results.csv` | สรุป accuracy และ macro F1 ของทุก strategy รวมในไฟล์เดียว |
| `submission/S5_output.csv` | ผลทำนายวันจริง (ชื่อไฟล์ตาม strategy ที่ใช้) |
| `models/mobilenet_d1/best.pt` ฯลฯ | โมเดลที่เทรนไว้ |

คอลัมน์ `scope` ในไฟล์ผลรายรูป
- `in_domain` = รูปจากแหล่งที่โมเดลเคยเห็นตอนเทรน
- `unseen` = รูปจากแหล่งที่ไม่เคยเห็นเลย ใช้ตัวนี้ตัดสินว่า strategy ไหนดีสำหรับข้อมูลวันจริง

---

## เวลาโดยประมาณ

| คำสั่ง | เวลา |
|---|---|
| S1 | ~10 นาที |
| S2 | ~5 นาที |
| S2b | ~12 นาที |
| S3 | ~25 นาที |
| S4 / S5 / S6 | ~2-3 นาทีต่อไฟล์ |
| S7 | ไม่กี่วินาที |
| ทำนายวันจริง | ไม่กี่วินาทีต่อรูปหลักร้อย |
