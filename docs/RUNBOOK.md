# RUNBOOK — ลำดับการรันการทดลอง

คู่มือปฏิบัติของโครงสร้างใน [README.md](../README.md) ข้อ 20
บอกว่า **รันอะไร → ได้ไฟล์อะไร → อ่านผลยังไง → ทำอะไรต่อ**

ทุกอย่างรันได้ 2 ทาง เลือกทางใดทางหนึ่ง
- **โน้ตบุ๊ก** `notebooks/01..05` เหมาะกับการดูกราฟและเขียนรายงาน
- **command line** เหมาะกับงานที่รันยาว ปิดหน้าจอทิ้งไว้ได้

```bash
pip install -r requirements.txt
```

---

## ขั้นที่ 0 — เตรียม cache (ทำครั้งเดียว)

```bash
python -c "from src.datasets.thai_chars import load_data; load_data(96)"
```

**ได้:** `data/research_cache_96.npz` (~1.4 GB) รูปทั้งหมด 163,155 รูปที่ preprocess แล้ว
ใช้เวลาราว 5 นาที ครั้งต่อไปทุก experiment โหลดจากไฟล์นี้ทันที
ไฟล์นี้สร้างใหม่ได้เสมอ และไม่ได้ commit ลง git

> ถ้าย้าย/เพิ่ม/ลบไฟล์ใน `Dataset/` ระบบจะตรวจพบเองแล้ว preprocess ใหม่ให้

---

## ขั้นที่ 1 — Dataset Analysis

```bash
jupyter lab notebooks/01_dataset_analysis.ipynb
```

**ได้:**
- `results/tables/dataset_summary.json` — จำนวนรูป/คลาส/imbalance ratio ของแต่ละ source
- `results/figures/class_distribution.png`, `samples_per_source.png`

**ต้องสรุปให้ได้ก่อนไปต่อ**
1. source ไหนใหญ่พอเป็น training set หลัก
2. คลาสไหนมีรูปน้อยจนต้องตัดทิ้ง (พารามิเตอร์ `min_count`)
3. imbalance ratio สูงไหม ถ้าสูงมากให้ตั้ง `imbalance: class_weight` หรือ `weighted_sampling` ใน config
4. สาม source ต่างกันจริงไหม ถ้าไม่ต่าง Experiment D จะไม่มีความหมาย

---

## ขั้นที่ 2 — Experiment A: Model Comparison

```bash
python -m src.training.run_experiment --experiment model_comparison --config configs/baseline.yaml
```

ทดสอบให้เร็วก่อนได้ด้วย (ควรทำทุกครั้งที่แก้ config)

```bash
python -m src.training.run_experiment --experiment model_comparison --config configs/baseline.yaml --epochs 1 --limit-per-class 20
```

**เวลา:** 5 โมเดล × 8 epoch บน D1 ประมาณ 2–4 ชั่วโมง (Mac M-series)

**ได้:**
- `results/tables/model_comparison.csv` — accuracy, macro F1, จำนวนพารามิเตอร์, inference time, เวลาเทรน
- `experiments/model_comparison/<model>/` — `best.pt`, `training.json`, `report.json`, `confusion.npy`, `predictions.npz`

**อ่านผล:** ดู **macro F1** เป็นหลัก ไม่ใช่ accuracy เพราะข้อมูลไม่สมดุล ถ้าสองตัวใกล้กันให้เลือกตัวที่เล็กและเร็วกว่า

**ทำต่อ:** เอาชื่อโมเดลที่ชนะไปใส่ `model:` ใน `configs/augmentation.yaml` และ `configs/generalization.yaml`

---

## ขั้นที่ 3 — Experiment B: Data Augmentation

```bash
python -m src.training.run_experiment --experiment augmentation --config configs/augmentation.yaml
```

**เวลา:** 4 policy × 10 epoch ประมาณ 2–3 ชั่วโมง

**ได้:**
- `results/tables/augmentation.csv`
- `results/figures/augmentation_examples.png` (จากโน้ตบุ๊ก 03) — เห็นว่าแต่ละ policy ทำอะไรกับรูปจริง
- `experiments/augmentation/<model>_<policy>/`

**อ่านผล:** ถ้า `thai_specific` ชนะ `standard` ไม่มาก อย่าเพิ่งสรุปว่าไม่จำเป็น เพราะประโยชน์ของมันมักโผล่ตอนทดสอบข้าม source ในขั้นที่ 5

**ทำต่อ:** เอา policy ที่ดีที่สุดไปใส่ `augmentation:` ใน `configs/generalization.yaml`

---

## ขั้นที่ 4 — Experiment C: Hard Character Mining

```bash
python -m src.training.run_experiment --experiment hard_character --config configs/baseline.yaml
```

**เวลา:** เทรน 2 รอบ (baseline + fine-tune) ประมาณ 1–2 ชั่วโมง

**ได้:**
- `experiments/hard_character/hard_set.json` — คลาสที่ F1 ต่ำกว่าเกณฑ์ และคู่ที่สับสนบ่อย
- `results/tables/hard_character.csv` — เทียบ accuracy / macro F1 / F1 เฉลี่ยของคลาสยาก ก่อนและหลัง fine-tune
- `results/figures/confusion_matrix.png` (จากโน้ตบุ๊ก 04)

**อ่านผล:** ถ้า `hard_mean_f1` เพิ่มขึ้นโดยที่ `macro_f1` รวมไม่ลดลง แปลว่าได้ผล
ถ้ารวมลดลง แสดงว่า oversampling แรงเกินไป ให้ลด `finetune_sampling_power`

**ทำต่อ:** เปิดดูรูปที่ทายผิดในโน้ตบุ๊ก 04 ถ้าพบว่าหลายรูปติด label ผิดตั้งแต่ต้น ให้รัน `clean_d1.py` (ดูท้ายไฟล์นี้)

---

## ขั้นที่ 5 — Experiment D: Generalization ⭐

```bash
python -m src.training.run_experiment --experiment generalization --config configs/generalization.yaml
```

**เวลา:** เทรน 5 ครั้ง (3 source + unseen font 2 ชุด) ประมาณ 3–5 ชั่วโมง

**ได้:**
- `results/tables/generalization.csv` — ทุกการทดสอบพร้อมค่า gap
- `results/tables/cross_source_matrix.json` — เมทริกซ์ 3×3
- `results/figures/cross_source_matrix.png` (จากโน้ตบุ๊ก 05)
- `experiments/generalization/random_<source>/cross_test_<other>.json`

**อ่านผล**
- เส้นทแยง = in-domain, นอกเส้นทแยง = cross-source
- `gap = in-domain − cross-source` ยิ่งกว้างยิ่งแปลว่าโมเดลติดลักษณะเฉพาะของ source
- unseen font ตกมากแค่ไหนเทียบกับ random split ของ source เดียวกัน คือคำตอบของ "จำ character หรือจำ font"

**ทำต่อ:** รันซ้ำโดยเปลี่ยน `augmentation:` เป็น `none` แล้วเทียบ gap สองรอบ จะได้หลักฐานว่า augmentation ช่วย generalization จริงไหม
และรันเซลล์ Combined Training ท้ายโน้ตบุ๊ก 05 เพื่อตอบ README ข้อ 13

---

## ลำดับที่แนะนำ

```text
0. cache  ->  1. dataset analysis  ->  2. model comparison  ->  เลือกโมเดล
                                              ↓
                                     3. augmentation  ->  เลือก policy
                                              ↓
                                     4. hard character mining
                                              ↓
                                     5. generalization (การทดลองหลัก)
```

ขั้นที่ 2–5 ต่อเนื่องกัน เพราะแต่ละขั้นเลือกค่าให้ขั้นถัดไป ถ้าเวลาจำกัดให้ลด `epochs` ลงก่อน อย่าข้ามขั้น

---

## แก้อะไรที่ไหน

| อยากเปลี่ยน | แก้ที่ |
|---|---|
| epoch / batch size / learning rate | ไฟล์ใน `configs/` |
| รายชื่อโมเดลที่เทียบ | `configs/baseline.yaml` คีย์ `models` |
| วิธีจัดการ class imbalance | คีย์ `imbalance` (`none`/`class_weight`/`weighted_sampling`) |
| สูตร augmentation | `src/augmentation/policies.py` |
| วิธีแบ่งข้อมูล | `src/datasets/splits.py` |
| โมเดลใหม่ | `src/models/factory.py` |
| ขนาดภาพ | `img_size` ใน config (จะสร้าง cache ใหม่ให้อัตโนมัติ) |

---

## หมายเหตุสำหรับ reproducibility (README ข้อ 21)

ทุก run บันทึก seed, config, split, จำนวน epoch, เวลาเทรน และ checkpoint ไว้ใน
`experiments/<experiment>/<run>/training.json` และ `report.json` อยู่แล้ว
เวลาเขียนรายงานให้อ้างไฟล์เหล่านี้แทนการจดตัวเลขเอง

---

## เครื่องมือเสริม: ตรวจ label ผิดด้วย confident learning

ถ้าสงสัยว่าไฟล์ใน `Dataset/character/<code>/D1/` ติด label ผิด

```bash
python clean_d1.py --k 4 --epochs 4
```

เป็น dry-run เสมอ สร้างแค่ CSV กับสำเนารูปใน `_review_snapshots/` ต้องสั่ง `--apply` เองถึงจะย้ายไฟล์จริง
และย้อนกลับได้ด้วย `--undo <csv>` รายละเอียดอยู่ใน docstring ของ `clean_d1.py`

---

## สคริปต์เดิมของโปรเจกต์

`train_resnet50.py`, `train_resnet50.ipynb` และ `train_withRestnet_d1/d2/d3.ipynb`
เป็นงานก่อนหน้าที่แยกจาก pipeline นี้ ใช้อ้างอิงผล baseline เดิม (ResNet50 ได้ val accuracy 94.93%) ได้
