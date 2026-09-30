"""รวมคำทำนายจากโมเดลหลายตัวตอน inference (Model_strategy.md ข้อ 8-10)

ทุกฟังก์ชันทำงานกับ probability matrix รูปทรง (n_samples, n_classes) ของแต่ละโมเดล
โมเดลทุกตัวในโปรเจกต์ใช้ class index ชุดเดียวกัน (81 คลาสจากข้อมูลทั้งหมด) จึงรวมกันได้ตรง ๆ
"""
import numpy as np
import torch

from src.models.factory import build_model
from src.augmentation.policies import to_input


@torch.no_grad()
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(ckpt["model_name"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def predict_probs(model, x_uint8, device, batch_size=256):
    """คืน probability (n, C) — ไม่มี augmentation (ตาม Model_strategy.md ข้อ 19)"""
    out = []
    for s in range(0, len(x_uint8), batch_size):
        xb = to_input(torch.from_numpy(x_uint8[s:s + batch_size]), device, None)
        out.append(model(xb).softmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def seen_class_mask(data, train_ids):
    """คลาสที่โมเดลเคยเห็นตอนเทรน — ใช้ปิดคลาสที่เป็นไปไม่ได้

    เช่น โมเดลที่เทรนด้วย D2 ไม่เคยเห็นเลขไทยเลย การปล่อยให้มันโหวตเลขไทย
    คือการเอา noise เข้าระบบ
    """
    mask = np.zeros(data.n_classes, bool)
    mask[np.unique(data.y[train_ids])] = True
    return mask


def apply_mask(probs, mask):
    """ตัดคลาสที่โมเดลไม่เคยเห็นออกแล้ว normalize ใหม่"""
    p = probs * mask[None, :]
    total = p.sum(1, keepdims=True)
    return np.divide(p, total, out=np.full_like(p, 1.0 / p.shape[1]), where=total > 1e-12)


def majority_vote(prob_list):
    """S4 — โหวตจากคลาสที่แต่ละโมเดลเลือก เสมอกันตัดสินด้วยผลรวม probability"""
    preds = np.stack([p.argmax(1) for p in prob_list])          # (n_models, n)
    n_classes = prob_list[0].shape[1]
    counts = np.zeros((preds.shape[1], n_classes), np.float64)
    for row in preds:
        counts[np.arange(len(row)), row] += 1
    counts += np.mean(prob_list, axis=0) * 1e-3                 # tie-break ด้วยความมั่นใจ
    return counts.argmax(1)


def prob_average(prob_list, weights=None):
    """S5 (weights=None) และ S6 (กำหนด weights) — เฉลี่ย probability ทั้งการแจกแจง"""
    w = np.ones(len(prob_list)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    return np.tensordot(w, np.stack(prob_list), axes=(0, 0))


def agreement_stats(prob_list):
    """สถิติความเห็นตรงกันของโมเดล (Model_strategy.md ข้อ 14)"""
    preds = np.stack([p.argmax(1) for p in prob_list])
    all_agree = (preds == preds[0]).all(0)
    n = preds.shape[1]
    return dict(agreement_rate=float(all_agree.mean()),
                n_disagree=int((~all_agree).sum()),
                mean_confidence=float(np.mean([p.max(1).mean() for p in prob_list])),
                n_samples=int(n))
