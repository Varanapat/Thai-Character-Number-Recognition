"""Path, config, seed และ helper เล็ก ๆ ที่ทุก module ใช้ร่วมกัน"""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "Dataset"
CACHE_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
EXPERIMENTS_DIR = ROOT / "experiments"

# บทบาทของแต่ละ source ตาม README ข้อ 4.1
SOURCE_ROLES = {"A": "D1", "B": "D2", "C": "D3"}
SOURCE_NAMES = {"D1": "DL - AJ", "D2": "THI - C168", "D3": "Burapha"}


def load_config(path, **overrides):
    """อ่าน YAML แล้ว override ด้วย keyword ได้ เช่น load_config(p, epochs=1)"""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(name=None):
    if name and name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def thai_char(code):
    """รหัสโฟลเดอร์คือรหัส TIS-620: 161 -> ก, 240 -> ๐"""
    try:
        return bytes([int(code)]).decode("tis-620")
    except (ValueError, UnicodeDecodeError):
        return "?"


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_dir(experiment, run_name):
    """โฟลเดอร์เก็บผลของ 1 run เช่น experiments/model_comparison/resnet18_standard/"""
    d = EXPERIMENTS_DIR / experiment / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d
