"""Metrics ตาม README ข้อ 16 — ไม่ดูแค่ Accuracy"""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from src.utils.common import save_json, thai_char


def compute_metrics(y_true, y_pred, classes, labels_present_only=False):
    """คืน accuracy, macro F1 และ metric ราย class"""
    labels = sorted(set(y_true.tolist())) if labels_present_only else list(range(len(classes)))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    per_class = {}
    for i, c in enumerate(labels):
        if support[i] == 0:
            continue
        code = classes[c]
        per_class[code] = dict(char=thai_char(code), precision=float(prec[i]), recall=float(rec[i]),
                               f1=float(f1[i]), support=int(support[i]),
                               accuracy=float(rec[i]))  # per-class accuracy = recall
    return dict(accuracy=float(accuracy_score(y_true, y_pred)),
                macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
                n_samples=int(len(y_true)), n_classes=int(len(per_class)), per_class=per_class)


def confusion(y_true, y_pred, n_classes):
    return confusion_matrix(y_true, y_pred, labels=range(n_classes))


def top_confused_pairs(cm, classes, top=15):
    """คู่ที่สับสนบ่อยที่สุด (true -> predicted) สำหรับ Experiment C"""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = []
    for flat in np.argsort(off, axis=None)[::-1][:top]:
        t, p = divmod(int(flat), len(classes))
        if off[t, p] == 0:
            break
        pairs.append(dict(true=classes[t], true_char=thai_char(classes[t]),
                          pred=classes[p], pred_char=thai_char(classes[p]),
                          count=int(off[t, p]),
                          rate=float(off[t, p] / max(cm[t].sum(), 1))))
    return pairs


def evaluate_split(model, data, ids, device, classes=None, batch_size=256):
    """predict + คำนวณ metric ครบชุดในฟังก์ชันเดียว"""
    from src.training.trainer import predict
    classes = classes or data.classes
    pred, probs = predict(model, data, ids, device, batch_size)
    y_true = data.y[ids]
    out = compute_metrics(y_true, pred, classes, labels_present_only=True)
    out["confusion"] = confusion(y_true, pred, len(classes))
    out["confused_pairs"] = top_confused_pairs(out["confusion"], classes)
    out["y_true"], out["y_pred"], out["probs"] = y_true, pred, probs
    return out


def save_report(result, path, extra=None):
    """เซฟเฉพาะส่วนที่ JSON ได้ (ตัด array ใหญ่ ๆ ออก)"""
    keep = {k: v for k, v in result.items()
            if k not in ("confusion", "y_true", "y_pred", "probs")}
    if extra:
        keep.update(extra)
    return save_json(keep, path)


def generalization_gap(in_domain_acc, out_domain_acc):
    """README ข้อ 12"""
    return float(in_domain_acc - out_domain_acc)
