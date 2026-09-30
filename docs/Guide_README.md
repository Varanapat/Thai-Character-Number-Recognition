# Thai Character & Number Recognition — Cross-Source Generalization

โปรเจกต์สร้างโมเดลจำแนกอักขระไทย (พยัญชนะ สระ วรรณยุกต์ เลขไทย รวม 80 คลาส รหัสโฟลเดอร์เป็นรหัส TIS-620 เช่น `161` = ก) จากข้อมูล 3 แหล่งที่หน้าตาต่างกันมาก แล้วตอบคำถามหลักของโปรเจกต์:

> **Accuracy สูงในข้อมูลที่เคยเห็น ไม่ได้แปลว่าโมเดล generalize ได้ดีเมื่อเจอข้อมูลจากแหล่งใหม่**

สถาปัตยกรรมที่ใช้ทั้งโปรเจกต์คือ **MobileNetV3-Small** (เลือกจาก Experiment A) ส่วนตัวแปรที่เปรียบเทียบกันคือ **วิธีเทรน/วิธีรวมผลของโมเดล** (strategy S1–S7)

สรุปผลแบบละเอียดทั้งหมดอยู่ใน **[RESULT_README.md](RESULT_README.md)** — ไฟล์นี้เป็นแค่ภาพรวมโครงสร้างโปรเจกต์กับวิธีใช้งาน

---

## 1. ข้อมูล (3 แหล่ง)

| | D1 (DL - AJ) | D2 (THI - C168) | D3 (Burapha) |
|---|---|---|---|
| ที่มา | อาจารย์ให้มา | THI-C168 dataset | เก็บ/สร้างจาก Burapha |
| จำนวนรูป | 62,393 | 76,927 | 23,835 |
| ลักษณะภาพ | ขาวดำ เล็กมาก ~15×20 px | สี ~145×140 px ขอบขาวเยอะ | ขาวดำ จัตุรัส ~24–60 px |

ทั้งสามแหล่งถูกจัดเรียงรวมกันไว้ใต้ `Dataset/character/<รหัส TIS-620>/` โดยชื่อไฟล์ระบุแหล่งที่มา (`D1_P0001.pdf`, `D2_P0001.pdf`, …) รายละเอียดรูปแบบไฟล์และโครงสร้างโฟลเดอร์ดูที่ **[Dataset/README.md](../Dataset/README.md)**

D2 ไม่มีคลาสเลขไทยและขาดสระบางตัว จึงมีคลาสน้อยกว่า D1/D3 — มีผลต่อการตีความ macro F1 เวลาทดสอบข้าม source (ดูรายละเอียดใน RESULT_README.md ข้อ 5.2)

---

## 2. โครงสร้างโปรเจกต์

```text
.
├── README.md              ภาพรวม (ไฟล์นี้)
├── RESULT_README.md       สรุปผลการทดลองทั้งหมด — อ่านก่อนเขียนรายงาน/พรีเซนต์
├── RUNBOOK.md              ลำดับการรัน Experiment A–D แบบละเอียด (ได้ไฟล์อะไร อ่านผลยังไง)
├── COMMANDS.md             คำสั่งทั้งหมดของ pipeline กลยุทธ์ S1–S7
├── Model_strategy.md       เหตุผลเบื้องหลังการออกแบบ strategy S1–S7
│
├── configs/                ไฟล์ตั้งค่าการเทรน (.yaml) ต่อ experiment/strategy
├── Dataset/                 ข้อมูลดิบ จัดเรียงตาม character code (มี README.md ของตัวเอง)
├── data/                    cache ที่ preprocess แล้ว (.npz) สร้างใหม่ได้เสมอ ไม่ได้ commit
│
├── src/                     โค้ดหลัก
│   ├── datasets/            โหลด/แบ่ง/preprocess ข้อมูล
│   ├── augmentation/        augmentation policy (standard, thai_specific, thai_randaugment)
│   ├── models/               สถาปัตยกรรมโมเดล
│   ├── training/              training loop ของ Experiment A–D
│   ├── evaluation/            metric, confusion matrix
│   ├── ensemble/               majority vote / probability average / weighted
│   ├── strategy_runner.py     ตัวรันจริงของ S1–S6 (S*.py แค่เรียกไฟล์นี้)
│   └── inference.py            ทำนายรูปใหม่จริง ๆ (ใช้ตอน "วันจริง")
│
├── notebooks/                โน้ตบุ๊ก 01–05 คู่กับแต่ละ experiment ใน RUNBOOK.md
├── experiments/               ผลรายรอบของ Experiment A–D (best.pt, training.json, report.json, confusion.npy)
├── models/                    โมเดลจาก S1–S3 (mobilenet_d1, mobilenet_d2, mobilenet_d3, mobilenet_d1_d2_d3)
├── results/
│   ├── tables/                 ตารางสรุปผล (.csv/.json) ของทุก experiment/strategy
│   └── figures/                 กราฟทั้งหมด
├── submission/                  ผลทำนายจากแต่ละ strategy (S1_output.csv … S6_output.csv)
│
├── S1_d1_only.py, S2_d3_only.py, S2b_d2_only.py   เทรนโมเดลรายแหล่ง
├── S3_combined.py                                    เทรนรวมทั้ง 3 แหล่ง
├── S4_majority_vote.py, S5_prob_average.py, S6_weighted.py   ensemble จากโมเดล S1/S2/S2b
├── S7_compare.py                                       สรุปเปรียบเทียบทุก strategy
├── clean_d1.py / clean_d1source.py                     ตรวจ label ผิดใน D1 ด้วย confident learning
│
└── train_resnet50*.py, train_withRestnet_*.ipynb        งานเดิมก่อน pipeline นี้ (ResNet50 baseline อ้างอิงได้)
```

`Project/README.md` เป็นไฟล์ต้นแบบเก่า (เนื้อหาเดียวกับ README.md ก่อนแก้) ไม่ได้ใช้งานจริงกับโปรเจกต์นี้ ทิ้งไว้เป็นตัวอย่างเฉย ๆ

---

## 3. อ่านอะไรตอนไหน

| ต้องการ | ไปที่ |
|---|---|
| ภาพรวมโปรเจกต์ โครงสร้างไฟล์ | README.md (ไฟล์นี้) |
| ผลการทดลองทั้งหมด สรุปคำตอบ RQ1–RQ4 | [RESULT_README.md](RESULT_README.md) |
| รันซ้ำ Experiment A–D ทีละขั้น | [RUNBOOK.md](RUNBOOK.md) |
| คำสั่งรัน strategy S1–S7 ทั้งหมด | [COMMANDS.md](COMMANDS.md) |
| เหตุผลออกแบบ strategy S1–S7 | [Model_strategy.md](Model_strategy.md) |
| รูปแบบไฟล์ข้อมูล/โครงสร้าง Dataset | [Dataset/README.md](../Dataset/README.md) |

---

## 4. สถานะปัจจุบัน

- Experiment A–D (เปรียบเทียบสถาปัตยกรรม, augmentation, hard character mining, generalization) รันครบแล้ว — สรุปผลใน RESULT_README.md
- Strategy S1, S2, S2b, S3 (เทรนรวมทั้ง 3 แหล่ง), S4, S5, S6 รันครบแล้ว ผลอยู่ใน `results/tables/strategy_results.csv`
- โมเดลที่เทรนด้วยทั้ง 3 แหล่ง (`S3_combined.py`) อยู่ที่ `models/mobilenet_d1_d2_d3/best.pt` — val accuracy 94.2% (10 epoch, augmentation `thai_specific`)
- `S7_compare.py` (สนาม unseen — จำลองข้อมูลวันจริง) ให้อันดับล่าสุด:

  | อันดับ | strategy | เทรนด้วย | macro F1 |
  |---:|---|---|---:|
  | 1 | S3 (โมเดลเดียว เทรนรวม) | D1+D2+D3 | 0.8685 |
  | 2 | S5 (prob average ensemble) | D1+D2+D3 | 0.8186 |
  | 3 | S4 (majority vote ensemble) | D1+D2+D3 | 0.7968 |

  สรุป: ตอนนี้ **โมเดลเดียวที่เทรนรวมทั้ง 3 แหล่ง (S3) ดีกว่า ensemble ทุกแบบ** ผลอาจเปลี่ยนได้ถ้ารันซ้ำหรือเปลี่ยน augmentation (ดู RESULT_README.md หมายเหตุท้าย Experiment B) — รัน `python3 S7_compare.py` เพื่อดูอันดับล่าสุดเสมอ ก่อนตัดสินใจว่าจะใช้ strategy ไหนวันจริง

---

## 5. วันจริง — ทำนายข้อมูลใหม่

```bash
python3 S3_combined.py path/to/new_images
```

หรือเปลี่ยนเป็นไฟล์ที่ `S7_compare.py` แนะนำ ณ ตอนนั้น (เช่น `python3 S5_prob_average.py path/to/new_images`) กำหนดที่เก็บผลเองได้ด้วย `--out submission/final.csv` ไม่ต้อง preprocess รูปเอง — ทุก strategy script preprocess ให้อัตโนมัติด้วยขั้นตอนเดียวกับตอนเทรน (grayscale → crop กรอบหมึก → pad จัตุรัส → resize) รองรับนามสกุล `.jpg` `.jpeg` `.png` `.bmp`

ผลลัพธ์เป็น CSV ที่ `submission/<strategy>_output.csv` คอลัมน์หลัก:

| คอลัมน์ | ความหมาย |
|---|---|
| `path` | path ของรูปต้นฉบับ |
| `predicted_class` | รหัส TIS-620 ของตัวอักษรที่ทำนาย |
| `predicted_char` | ตัวอักษรไทยที่ถอดแล้ว (ผ่าน `thai_char()`) |
| `answer` | **คำตอบตาม `Mapping.csv`** (ดูหมายเหตุด้านล่าง) — มีคอลัมน์นี้เฉพาะตอนมี `Mapping.csv` อยู่ที่ root โปรเจกต์ |
| `confidence` | ความมั่นใจของคำตอบสุดท้าย |
| `top_3` | อันดับความน่าจะเป็นสูงสุด 3 อันดับ |
| `agree` | โมเดลย่อยทั้งหมด (ถ้าใช้ ensemble) ทายตรงกันไหม |

**Mapping.csv:** วาง `Mapping.csv` (คอลัมน์ `folder,gt_label,class_id` — `folder` คือรหัส TIS-620, `class_id` คือคำตอบที่โจทย์ต้องการ) ไว้ที่ root โปรเจกต์ สคริปต์ทำนาย (`S1`–`S7` ทุกตัว ผ่าน `src/strategy_runner.py`) จะโหลดไฟล์นี้อัตโนมัติแล้วเติมคอลัมน์ `answer` ให้เอง — ไม่ต้อง join เองภายหลัง

ตอนนี้ `Mapping.csv` ที่มีอยู่ครอบคลุม 72 รหัส แต่ข้อมูลใน `Dataset/` มีคลาสจริง 89 รหัส (character+number) แปลว่ามี **17 รหัสที่ยังไม่มีคำตอบใน Mapping.csv**: `165(ฅ) 166(ฆ) 172(ฌ) 174(ฎ) 198(ฦ) 208(ะ) 211(ำ) 218(ฺ) 219 220 221 222 223(฿) 235(๋) 237(ํ) 238(๎) 239(๏)` ถ้าโมเดลทายรูปวันจริงตรงกับคลาสพวกนี้ คอลัมน์ `answer` จะว่าง (สคริปต์จะพิมพ์เตือนรายชื่อคลาสที่ทายไม่เจอ mapping ทุกครั้งที่รัน) — ควรเช็คกับอาจารย์/ต้นทางของ Mapping.csv ก่อนวันจริงว่ารหัสพวกนี้ควรตอบเป็นอะไร

---

## 6. เริ่มต้นใช้งานครั้งแรก

```bash
pip install -r requirements.txt

# สร้าง cache ข้อมูล preprocess แล้ว (ทำครั้งเดียว ~5 นาที)
python -c "from src.datasets.thai_chars import load_data; load_data(96)"
```

จากนั้น:
- จะรัน experiment งานวิจัย (A–D) ทีละขั้น → ดู [RUNBOOK.md](RUNBOOK.md)
- จะรัน/ทดสอบ strategy S1–S7 หรือทำนายข้อมูลวันจริง → ดู [COMMANDS.md](COMMANDS.md)

ทุก run บันทึก seed, config, การแบ่งข้อมูล, epoch และเวลาเทรนไว้ใน `training.json`/`strategy.json` ข้าง ๆ checkpoint อยู่แล้ว ใช้อ้างอิงตอนเขียนรายงานได้เลย
