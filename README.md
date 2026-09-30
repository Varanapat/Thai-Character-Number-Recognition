# Thai Character & Number Recognition

จำแนกตัวอักษรไทย 81 คลาส (พยัญชนะ สระ วรรณยุกต์ และเลขไทย) ด้วย CNN
โดยเน้นคำถามว่า **โมเดลทำงานกับข้อมูลจากแหล่งที่ไม่เคยเห็นได้ดีแค่ไหน** ไม่ใช่แค่ accuracy สูงในชุดเดิม

| | |
|---|---|
| โมเดลที่เลือกใช้ | MobileNetV3-Small (1.6M พารามิเตอร์, 2.5 ms/ภาพ) |
| กลยุทธ์ที่เลือกใช้ | S3 — เทรนรวมทั้งสามแหล่งไว้ในโมเดลเดียว |
| ข้อมูลเทรน | 163,155 รูป จาก 3 แหล่ง (D1 ลายมือ, D2 ลายมือ, D3 ลายมือ) |
| ผลในโดเมนเดิม | accuracy 99.1% (D1), 88.5% (D2), 98.5% (D3) |
| ผลกับฟอนต์พิมพ์ที่ไม่เคยเห็น | accuracy 83.8% / macro F1 0.742 |

---

## เริ่มใช้งานเร็ว ๆ

```bash
pip install -r requirements.txt
```

ทำนายรูปใหม่ด้วยโมเดลที่เทรนไว้แล้ว (ไม่ต้องเทรนใหม่ ไม่ต้องมี Dataset)

```bash
python3 strategies/S3_combined.py path/to/images
```

ได้ `submission/S3_output.csv` ที่มี path, คลาสที่ทำนาย, ตัวอักษร และความมั่นใจ

ถ้าชื่อไฟล์มีเฉลยอยู่ในตัว (เช่น `ko_kai_Athiti-Bold.png`) เติม `--labels filename` แล้วระบบจะคิด accuracy ให้ด้วย

คำสั่งทั้งหมดอยู่ใน [COMMANDS.md](docs/COMMANDS.md)

---

## โครงสร้างไฟล์

### ไฟล์ที่ใช้รัน — 1 ไฟล์ต่อ 1 กลยุทธ์

ทุกไฟล์ใช้ MobileNetV3-Small และค่าการเทรนชุดเดียวกัน ต่างกันแค่ "วิธีเทรน" หรือ "วิธีรวมผล"
ไม่ใส่ path = เทรนและวัดผล, ใส่ path = ทำนายรูปในโฟลเดอร์นั้น

| ไฟล์ | กลยุทธ์ | ต้องเทรนไหม |
|---|---|---|
| `S1_d1_only.py` | เทรนด้วย D1 อย่างเดียว | ใช่ ~35 นาที |
| `S2_d3_only.py` | เทรนด้วย D3 อย่างเดียว | ใช่ ~13 นาที |
| `S2b_d2_only.py` | เทรนด้วย D2 อย่างเดียว | ใช่ ~41 นาที |
| `S3_combined.py` | เทรนรวมทั้งสามแหล่ง **(ตัวที่เลือกใช้จริง)** | ใช่ ~57 นาที |
| `S4_majority_vote.py` | โหวตจาก 3 โมเดลรายแหล่ง | ไม่ ใช้โมเดลจาก S1+S2+S2b |
| `S5_prob_average.py` | เฉลี่ยความน่าจะเป็นจาก 3 โมเดล | ไม่ |
| `S6_weighted.py` | ถ่วงน้ำหนักตามผลบน validation | ไม่ |
| `S7_compare.py` | สรุปเทียบทุกกลยุทธ์จาก CSV | ไม่ |

### `src/` — โค้ดหลัก

| ไฟล์ | หน้าที่ |
|---|---|
| `strategy_runner.py` | เครื่องยนต์กลางที่ไฟล์ S1–S6 เรียกใช้ ทั้งโหมดเทรนและโหมดทำนาย |
| `datasets/thai_chars.py` | โหลดรูป, preprocess (ขาวดำ → ตัดกรอบหมึก → จัตุรัส → 96×96), ทำ cache |
| `datasets/splits.py` | แบ่ง train/val/test และ split แบบข้ามแหล่ง |
| `datasets/filename_labels.py` | อ่านเฉลยจากชื่อไฟล์แบบ PrintAksorn แล้วแปลงเป็นรหัส TIS-620 |
| `models/factory.py` | นิยามโมเดลทั้ง 6 แบบ, นับพารามิเตอร์, วัดเวลา inference |
| `augmentation/policies.py` | augmentation 4 ระดับ ทำงานบน GPU ทีละ batch |
| `training/trainer.py` | training loop, class weight, weighted sampling, early stopping |
| `training/run_experiment.py` | ตัวรัน Experiment A–D (เทียบ architecture, augmentation, hard character, generalization) |
| `training/run_strategy.py` | ตัวรันเปรียบเทียบกลยุทธ์แบบรวบยอดในคำสั่งเดียว |
| `evaluation/metrics.py` | accuracy, macro F1, metric รายคลาส, confusion matrix |
| `evaluation/hard_chars.py` | หาตัวอักษรที่โมเดลทำได้แย่ที่สุดจาก confusion matrix |
| `ensemble/combiners.py` | รวมผลหลายโมเดล: โหวต / เฉลี่ย / ถ่วงน้ำหนัก / ปิดคลาสที่ไม่เคยเห็น |
| `arch_runner.py` | เครื่องยนต์ของไฟล์ `arch_*.py` ที่ใช้เทียบ architecture |
| `inference.py` | ทำนายรูปใหม่ด้วย ensemble หลายโมเดลพร้อมกัน |
| `utils/common.py` | path, อ่าน config, ตั้ง seed, เลือก device, แปลงรหัส TIS-620 เป็นตัวอักษรไทย |

### `configs/` — ค่าการทดลอง แก้ที่นี่ ไม่ต้องแก้โค้ด

| ไฟล์ | ใช้กับ |
|---|---|
| `strategy_base.yaml` | ไฟล์ S1–S6 (ค่ากลางของทุกกลยุทธ์) |
| `baseline.yaml` | Experiment A เทียบ architecture และ Experiment C hard character |
| `augmentation.yaml` | Experiment B เทียบ augmentation |
| `generalization.yaml` | Experiment D ทดสอบข้ามแหล่งและ unseen font |
| `architecture.yaml` | ไฟล์ `arch_*.py` |
| `strategy.yaml` | `src/training/run_strategy.py` |

### `models/` — โมเดลที่เทรนแล้ว (ใช้ทำนายได้เลย)

```text
models/mobilenet_d1/best.pt          เทรนด้วย D1
models/mobilenet_d2/best.pt          เทรนด้วย D2
models/mobilenet_d3/best.pt          เทรนด้วย D3
models/mobilenet_d1_d2_d3/best.pt    เทรนรวมทุกแหล่ง (ตัวที่ใช้จริง)
```

แต่ละไฟล์เก็บทั้งน้ำหนัก, รายชื่อคลาส, ขนาดภาพ และ augmentation ที่ใช้เทรน

### `results/` — ผลการทดลอง

| โฟลเดอร์ | เนื้อหา |
|---|---|
| `tables/` | CSV สรุปทุกการทดลอง เช่น `strategy_results.csv`, `model_comparison.csv`, `generalization.csv` |
| `figures/` | กราฟ: confusion matrix, cross-source matrix, ตัวอย่าง augmentation, การกระจายของคลาส |
| `predictions_printaksorn/` | ผลรายรูปตอนทดสอบกับฟอนต์พิมพ์ |

### `notebooks/` — โน้ตบุ๊กสำหรับดูผลแบบมีกราฟ

```text
01_dataset_analysis.ipynb        สำรวจข้อมูลทั้ง 3 แหล่ง
02_model_comparison.ipynb        Experiment A
03_augmentation.ipynb            Experiment B
04_hard_character_mining.ipynb   Experiment C
05_generalization.ipynb          Experiment D
```

### `tools/` — เครื่องมือเสริม

| ไฟล์ | หน้าที่ |
|---|---|
| `clean_d1.py` | ตรวจหา label ที่น่าจะผิดด้วย confident learning (cleanlab) มีโหมด dry-run และย้อนกลับได้ |
| `clean_d1source.py` | เวอร์ชันที่ทำงานกับข้อมูลต้นทาง |
| `arch_*.py` | เทียบ architecture 6 แบบภายใต้ค่าเดียวกัน (ใช้ตอนเลือกโมเดล) |

รันจาก root ได้เลย เช่น `python3 tools/clean_d1.py --help`

ส่วน `Mapping.csv` ที่ root คือตารางแปลงรหัสโฟลเดอร์ (TIS-620) เป็นรหัสคลาสที่ใช้ส่งงาน เช่น 161 = ก = 101

### `legacy/` — งานรุ่นแรก (ResNet50) เก็บไว้อ้างอิง

```text
legacy/train_resnet50.py / .ipynb            เทรน ResNet50 กับข้อมูลรวมทุกแหล่ง
legacy/train_withRestnet_d1/d2/d3.ipynb      เทรนแยกรายแหล่ง
legacy/predict.py                            ทำนายด้วยโมเดล ResNet50
docs/README(RestNet50).md                    สรุปผลรุ่นแรก (val accuracy 94.93%)
```

### เอกสาร

ทั้งหมดอยู่ในโฟลเดอร์ `docs/`

| ไฟล์ | เนื้อหา |
|---|---|
| `Guide_README.md` | ภาพรวมโปรเจกต์และคำถามวิจัย |
| `Model_strategy.md` | แผนการเปรียบเทียบกลยุทธ์ S1–S7 |
| `RESULT_README.md` | ผลการทดลองทั้งหมด (Experiment A–D) พร้อมการตีความ |
| `CONFERENCE_BRIEF.md` | สรุปสำหรับนำเสนอ อธิบายว่าไฟล์ทำงานยังไงและผลเป็นอย่างไร |
| `COMMANDS.md` | คำสั่งทุกอย่างที่ต้องใช้ |
| `RUNBOOK.md` | ลำดับการรัน Experiment A–D |
| `Present_Readme.md` | เนื้อหาสำหรับทำสไลด์ |
| `Presentation_Script.md` | บทพูดพรีเซนต์แบ่งตามคน |
| `Inference_Guide.md` | คู่มือการทำ inference ด้วย MobileNetV3-Small |
| `DLM pre.pdf` | สไลด์นำเสนอ |

---

## ข้อมูล

ไฟล์ข้อมูลไม่ได้อยู่ใน repo เพราะใหญ่เกินขีดจำกัดของ GitHub (รวมกันกว่า 1 GB)

```text
Dataset/character/<รหัส TIS-620>/<D1|D2|D3>/*.jpg
Dataset/number/<รหัส TIS-620>/<D1|D2|D3>/*.jpg
```

| แหล่ง | ที่มา | จำนวนรูป | คลาส | ลักษณะ |
|---|---|---:|---:|---|
| D1 | DL - AJ | 62,393 | 76 | ขาวดำ เล็กมาก ~15×20 px |
| D2 | THI - C168 | 76,927 | 68 | ภาพสี ~145×140 px (ไม่มีเลขไทย) |
| D3 | Burapha | 23,835 | 77 | ขาวดำ จัตุรัส ~24–60 px |

ดาวน์โหลดได้จากหน้า Releases ของ repo นี้ แล้วแตกไฟล์ไว้ที่ `Dataset/` ระดับเดียวกับโฟลเดอร์ `src/`
โครงสร้างและที่มาของข้อมูลอธิบายไว้ใน [Dataset/README.md](Dataset/README.md)

**ถ้าจะแค่ทำนายรูปใหม่ ไม่ต้องมี Dataset** ใช้โมเดลใน `models/` ได้เลย

---

## ผลการทดลองโดยสรุป

### เทียบ architecture (Experiment A)

| Model | Accuracy | Macro F1 | Params | Inference |
|---|---:|---:|---:|---:|
| MobileNetV3-Small | 0.9929 | **0.9813** | 1.6M | 3.19 ms |
| DenseNet121 | **0.9937** | 0.9794 | 7.04M | 8.48 ms |
| ResNet18 | 0.9934 | 0.9792 | 11.2M | 1.35 ms |
| EfficientNet-B0 | 0.9929 | 0.9777 | 4.11M | 4.84 ms |
| Custom CNN | 0.9920 | 0.9631 | 1.19M | 0.75 ms |

ทุกตัวเสมอกันที่ 99% จึงเลือกจากต้นทุน → MobileNetV3-Small

### เทียบกลยุทธ์ ทดสอบกับฟอนต์พิมพ์ที่ไม่เคยเห็น (456 รูป)

| กลยุทธ์ | Accuracy | Macro F1 |
|---|---:|---:|
| **S3 รวมทุกแหล่ง** | **0.8377** | **0.7420** |
| S5 เฉลี่ยความน่าจะเป็น | 0.8202 | 0.6871 |
| S4 โหวต | 0.8158 | 0.6872 |
| S2 D3 อย่างเดียว | 0.7412 | 0.5855 |
| S1 D1 อย่างเดียว | 0.6447 | 0.5058 |
| S2b D2 อย่างเดียว | 0.5965 | 0.4457 |

**รวมข้อมูลตอนเทรน ดีกว่ารวมผลตอนทำนาย** และใช้โมเดลเดียวเร็วกว่า ensemble 3 เท่า

### ปัญหาการข้ามแหล่ง (Experiment D)

โมเดลที่เทรนแหล่งเดียวแล้วเอาไปทดสอบอีกแหล่ง (accuracy)

| เทรนด้วย ↓ / ทดสอบ → | D1 | D2 | D3 |
|---|---:|---:|---:|
| D1 | 0.9909 | 0.4350 | 0.4959 |
| D2 | 0.8330 | 0.8816 | 0.5545 |
| D3 | 0.8357 | 0.6808 | 0.9895 |

ตกจาก 99% เหลือ 43.5% เมื่อข้ามแหล่ง แต่การเปลี่ยนกลุ่มผู้เขียนภายในแหล่งเดียวกันแทบไม่กระทบ
แปลว่าตัวการคือวิธีเก็บและประมวลผลภาพของแต่ละแหล่ง ไม่ใช่ความต่างของลายมือ

รายละเอียดทั้งหมดอยู่ใน [RESULT_README.md](docs/RESULT_README.md)

---

## สภาพแวดล้อม

Python 3.12, PyTorch 2.14 รันบน Apple Silicon ผ่าน MPS (รองรับ CUDA และ CPU ด้วย)
ไลบรารีทั้งหมดอยู่ใน `requirements.txt`

ทุกการทดลองบันทึก seed, config, การแบ่งข้อมูล, จำนวน epoch และเวลาเทรนไว้ในไฟล์ JSON ของแต่ละรอบ
