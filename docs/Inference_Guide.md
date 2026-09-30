# MobileNetV3-Small Image Classification — Inference

This project is used to perform **image classification inference** with a trained **MobileNetV3-Small** model on a new dataset.

The inference pipeline is:

```text
Raw Image
    ↓
Preprocessing
    ↓
Resize / Tensor / Normalize
    ↓
MobileNetV3-Small
    ↓
Logits
    ↓
Prediction (Class ID)
    ↓
mapping.csv
    ↓
Answer
    ↓
submission.csv
```

---

## 1. Project Structure

Recommended project structure:

```text
project/
│
├── README.md
│
├── inference.py
├── mapping.csv
├── model.pth
│
├── dataset/
│   └── ...
│
└── output/
    └── submission.csv
```

The important files are:

| File             | Description                               |
| ---------------- | ----------------------------------------- |
| `inference.py`   | Script for running inference              |
| `model.pth`      | Trained MobileNetV3-Small model           |
| `mapping.csv`    | Mapping between model class ID and answer |
| `submission.csv` | Final prediction file                     |

---

# 2. Model Pipeline

The model receives an image and processes it through the following steps:

```text
                    Raw Image
                        ↓
                 Preprocessing
                        ↓
              Resize / Tensor / Normalize
                        ↓
             MobileNetV3-Small
                        ↓
                     Logits
                        ↓
                  Argmax / Prediction
                        ↓
                  Class ID / Index
                        ↓
                  mapping.csv
                        ↓
                     Answer
```

### Preprocessing

Each input image must go through the same preprocessing pipeline used during training.

Typical preprocessing:

```text
Image
  ↓
Resize
  ↓
Convert to Tensor
  ↓
Normalize
  ↓
Model Input
```

**Important:** The preprocessing used during inference must match the preprocessing used during training.

For example:

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[...],
        std=[...]
    )
])
```

Replace the normalization values with the values used during training.

---

# 3. Model

The classification model is:

```text
MobileNetV3-Small
```

The model produces logits for all classes:

```text
Image
  ↓
MobileNetV3-Small
  ↓
Logits

Example:

[-1.2, 0.3, 4.8, 0.7, ...]
```

The predicted class is obtained using `argmax`:

```python
predicted_class = logits.argmax(dim=1)
```

For example:

```text
Logits:
[-1.2, 0.3, 4.8, 0.7]

             ↓ argmax

Predicted Class:
2
```

The number `2` is the **model class index**.

It should not be directly treated as the final answer until it is converted using `mapping.csv`.

---

# 4. mapping.csv

`mapping.csv` is used to convert the model's class index into the required answer.

Example:

```csv
gt_path,class_id
test_dataset/image_100,101
test_dataset/image_1001,102
test_dataset/image_1002,103
test_dataset/image_1003,201
test_dataset/image_1004,204
```

If the model predicts:

```text
class_id = 161
```

The program looks up:

```text
mapping.csv

161 → ก → 101
```

Therefore:

```text
Prediction = 101
```

### Important

Do **not** manually assume 

unless this is exactly how `mapping.csv` is defined.

The final answer should always be generated from the provided `mapping.csv`.

---

# 5. Input Path

The inference script should accept the dataset path from the command line.

Example:

```bash
python inference.py --input_path ./dataset
```

For an absolute path:

```bash
python inference.py --input_path /path/to/test_dataset
```

This allows the input dataset to be changed without modifying the Python code.

For example, if the instructor gives:

```text
/path/to/new_dataset
```

run:

```bash
python inference.py --input_path /path/to/new_dataset
```

---

# 6. Recommended Command

The basic command is:

```bash
python inference.py \
    --input_path /path/to/test_dataset \
    --mapping_path ./mapping.csv \
    --model_path ./model.pth \
    --output_path ./output/submission.csv
```

### Parameters

| Argument         | Description                  |
| ---------------- | ---------------------------- |
| `--input_path`   | Path to the new test dataset |
| `--mapping_path` | Path to `mapping.csv`        |
| `--model_path`   | Path to trained model        |
| `--output_path`  | Path for the generated CSV   |

---

# 7. Example: Testing a New Dataset

Suppose the instructor gives:

```text
/Users/student/Desktop/test_dataset
```

Run:

```bash
python inference.py \
    --input_path "/Users/student/Desktop/test_dataset" \
    --mapping_path "./mapping.csv" \
    --model_path "./model.pth" \
    --output_path "./output/submission.csv"
```

The script will:

```text
Input Dataset
      ↓
Read Images
      ↓
Preprocessing
      ↓
MobileNetV3-Small
      ↓
Prediction
      ↓
Class ID
      ↓
mapping.csv
      ↓
Answer
      ↓
submission.csv
```

---

# 8. Input Dataset Format

The inference script should be able to read image files from the provided input directory.

Example:

```text
test_dataset/
├── image001.jpg
├── image002.jpg
├── image003.jpg
├── image004.png
└── ...
```

Supported image extensions should include:

```text
.jpg
.jpeg
.png
.bmp
.webp
```

The script should preserve the original filename so that the final CSV can be matched back to the corresponding image.

---

# 9. Output CSV

The final output should contain the image identifier and the predicted answer.

Example:

```csv
filename,answer
image001.jpg,101
image002.jpg,103
image003.jpg,201
image004.jpg,301
```

The exact column names must follow the format specified by the instructor.

If the instructor provides a required submission format, use that format instead.

---

# 10. Recommended Inference Logic

The inference process should conceptually work as follows:

```python
# 1. Load model
model = load_model(model_path)

# 2. Load mapping
mapping = load_mapping(mapping_path)

# 3. Find images
images = get_images(input_path)

# 4. For each image
for image_path in images:

    # Load image
    image = load_image(image_path)

    # Preprocess
    image = preprocess(image)

    # Model inference
    logits = model(image)

    # Get predicted class
    class_id = logits.argmax(dim=1).item()

    # Convert class ID → answer
    answer = mapping[class_id]

    # Save result
    results.append({
        "filename": image_path.name,
        "answer": answer
    })

# 5. Export CSV
save_csv(results, output_path)
```

---

# 11. Important: `model.eval()`

During inference, the model must be switched to evaluation mode:

```python
model.eval()
```

Inference should also be performed without calculating gradients:

```python
with torch.no_grad():
    logits = model(images)
```

Recommended pattern:

```python
model.eval()

with torch.no_grad():
    logits = model(images)
    predictions = logits.argmax(dim=1)
```

This is important because we are **testing/predicting**, not training the model.

---

# 12. Batch Inference

If the test dataset contains many images, the images should be processed in batches instead of loading every image into memory at once.

Example:

```text
Dataset
   ↓
Batch 1 → Model
Batch 2 → Model
Batch 3 → Model
Batch 4 → Model
   ↓
Combine Predictions
   ↓
mapping.csv
   ↓
submission.csv
```

Example:

```bash
python inference.py \
    --input_path ./test_dataset \
    --mapping_path ./mapping.csv \
    --model_path ./model.pth \
    --output_path ./output/submission.csv \
    --batch_size 32
```

---

# 13. GPU / CPU

The script should automatically select GPU when available.

Conceptually:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

The model and input tensors must be moved to the same device:

```python
model.to(device)
images = images.to(device)
```

If CUDA is unavailable, inference will run on CPU.

---

# 14. Before the Test

Before running the actual test dataset, verify the following:

### Model

* [ ] Correct `model.pth`
* [ ] Correct MobileNetV3-Small architecture
* [ ] Correct number of classes
* [ ] Model loads successfully

### Preprocessing

* [ ] Same image size as training
* [ ] Same normalization as training
* [ ] Same channel order
* [ ] RGB conversion is handled correctly

### Mapping

* [ ] Correct `mapping.csv`
* [ ] All predicted class IDs exist in `mapping.csv`
* [ ] Mapping direction is correct: `class_id → answer`

### Input

* [ ] `input_path` points to the correct dataset
* [ ] Image extensions are supported
* [ ] Images can be opened successfully

### Output

* [ ] CSV column names match the required format
* [ ] Every input image has exactly one prediction
* [ ] Filename/order format matches the instructor's requirement

---

# 15. Quick Test

Before using the actual dataset, test the command with a small folder:

```text
test_small/
├── image001.jpg
├── image002.jpg
└── image003.jpg
```

Run:

```bash
python inference.py \
    --input_path ./test_small \
    --mapping_path ./mapping.csv \
    --model_path ./model.pth \
    --output_path ./output/test_submission.csv
```

Check:

```text
output/
└── test_submission.csv
```

Expected:

```csv
filename,answer
image001.jpg,ก
image002.jpg,ข
image003.jpg,ค
```

---

# 16. Actual Test-Day Workflow

When the instructor gives a new dataset path:

### Step 1 — Check the input path

Example:

```text
/Users/student/Desktop/test_dataset
```

### Step 2 — Run inference

```bash
python inference.py \
    --input_path "/Users/student/Desktop/test_dataset" \
    --mapping_path "./mapping.csv" \
    --model_path "./model.pth" \
    --output_path "./output/submission.csv"
```

### Step 3 — Check the output

```bash
cat ./output/submission.csv
```

or open:

```text
output/submission.csv
```

### Step 4 — Submit the CSV

Use:

```text
output/submission.csv
```

---

# 17. One-Line Command

If `mapping.csv`, `model.pth`, and `inference.py` are already in the current directory:

```bash
python inference.py --input_path "/PATH/TO/NEW_DATASET"
```

The script can use default paths such as:

```text
model.pth
mapping.csv
output/submission.csv
```

This is recommended for test-day use because only the input path needs to be changed.

---

# 18. Recommended Test-Day Interface

The ideal command is:

```bash
python inference.py --input_path "/PATH/TO/DATASET"
```

The script should automatically:

```text
1. Load model.pth
2. Load mapping.csv
3. Detect available device
4. Find images in input_path
5. Preprocess images
6. Run MobileNetV3-Small
7. Get predicted class IDs
8. Map class IDs to answers
9. Generate submission.csv
```

Example terminal output:

```text
==================================================
MobileNetV3-Small Inference
==================================================

Input       : /path/to/test_dataset
Model       : ./model.pth
Mapping     : ./mapping.csv
Device      : cuda

Found images: 500

Running inference...
[####################] 100%

Inference completed.

Output      : ./output/submission.csv
Predictions : 500

==================================================
Done
==================================================
```

---

# 19. Important Notes

### Do not retrain the model during inference

The test process should only perform:

```text
Load Model
    ↓
Preprocess
    ↓
Predict
    ↓
Mapping
    ↓
CSV
```

There should be no:

```text
optimizer.step()
loss.backward()
model.train()
```

during inference.

### Do not change preprocessing

The preprocessing configuration should match the training configuration.

For example, if the model was trained with:

```text
224 × 224
RGB
ToTensor
Normalization
```

the test images should use the same configuration.

### Do not manually change class labels

Always use:

```text
mapping.csv
```

to convert:

```text
Class ID → Answer
```

---

# 20. Final Pipeline

```text
                         TEST DATASET
                              │
                              ▼
                         Raw Images
                              │
                              ▼
                       Load Image (RGB)
                              │
                              ▼
                         Preprocessing
                              │
                              ▼
                    Resize / Tensor / Normalize
                              │
                              ▼
                     MobileNetV3-Small
                              │
                              ▼
                            Logits
                              │
                              ▼
                      Argmax / Class ID
                              │
                              ▼
                         mapping.csv
                              │
                              ▼
                           Answer
                              │
                              ▼
                     submission.csv
```

The goal of the inference script is to make the test-day workflow as simple as:

```bash
python inference.py --input_path "/PATH/TO/NEW_DATASET"
```

Then submit:

```text
output/submission.csv
```
