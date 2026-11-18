"""Pooled cross-validation and split evaluation metrics (Steps 1.7 and 1.8).

Per Decision D23 (docs/decisions.md), session-disjoint CV has one gun session
per fold, so several folds lack validation clips for some classes (.38 is in
folds 1 and 3 only). All primary CV metrics (macro-F1, balanced accuracy,
per-class recall/precision/F1, confusion matrix, per-source accuracy) are
computed once on the pooled out-of-fold predictions across all five folds.
Per-fold scores over the classes present in each fold are also reported to
show fold-to-fold spread.
"""
from __future__ import annotations

import numpy as np


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    sources: list[str] | None = None,
    fold_ids: np.ndarray | None = None,
) -> dict:
    """Compute pooled macro-F1, balanced accuracy, per-class & per-source metrics."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n_classes = len(classes)
    cm = confusion_matrix_np(y_true, y_pred, n_classes)

    per_class = {}
    f1s, recalls = [], []
    for c, name in enumerate(classes):
        tp = int(cm[c, c])
        support = int(cm[c, :].sum())
        pred_pos = int(cm[:, c].sum())
        rec = tp / support if support > 0 else 0.0
        prec = tp / pred_pos if pred_pos > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[name] = {
            "support": support,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }
        if support > 0:
            f1s.append(f1)
            recalls.append(rec)

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    bal_acc = float(np.mean(recalls)) if recalls else 0.0
    acc = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0

    out: dict = {
        "n_samples": int(len(y_true)),
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "revolver_38spl_recall": per_class.get("revolver_38spl", {}).get("recall", 0.0),
        "revolver_38spl_f1": per_class.get("revolver_38spl", {}).get("f1", 0.0),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }

    if sources is not None:
        src_arr = np.asarray(sources)
        per_source = {}
        for s in sorted(set(sources)):
            mask = src_arr == s
            n_s = int(mask.sum())
            per_source[s] = {
                "n": n_s,
                "accuracy": float(np.mean(y_true[mask] == y_pred[mask])) if n_s else 0.0,
            }
        out["per_source"] = per_source

    if fold_ids is not None:
        fold_ids = np.asarray(fold_ids, dtype=np.int64)
        per_fold = {}
        fold_f1s, fold_bals = [], []
        for k in sorted(set(fold_ids.tolist())):
            m = fold_ids == k
            sub = classification_metrics(y_true[m], y_pred[m], classes)
            per_fold[int(k)] = {
                "n": int(m.sum()),
                "present_macro_f1": sub["macro_f1"],
                "present_balanced_accuracy": sub["balanced_accuracy"],
                "accuracy": sub["accuracy"],
                "revolver_38spl_recall": (
                    sub["per_class"]["revolver_38spl"]["recall"]
                    if sub["per_class"]["revolver_38spl"]["support"] > 0
                    else None
                ),
            }
            fold_f1s.append(sub["macro_f1"])
            fold_bals.append(sub["balanced_accuracy"])
        out["per_fold"] = per_fold
        out["fold_present_macro_f1_mean"] = float(np.mean(fold_f1s))
        out["fold_present_macro_f1_std"] = float(np.std(fold_f1s))
        out["fold_present_bal_acc_mean"] = float(np.mean(fold_bals))
        out["fold_present_bal_acc_std"] = float(np.std(fold_bals))

    return out


def aggregate_seed_metrics(seed_runs: list[dict], classes: list[str]) -> dict:
    """Aggregate metrics across multiple random seeds (mean and std)."""
    f1s = [r["macro_f1"] for r in seed_runs]
    bals = [r["balanced_accuracy"] for r in seed_runs]
    accs = [r["accuracy"] for r in seed_runs]
    r38 = [r["revolver_38spl_recall"] for r in seed_runs]
    per_class_summary = {}
    for c in classes:
        recs = [r["per_class"][c]["recall"] for r in seed_runs]
        cf1s = [r["per_class"][c]["f1"] for r in seed_runs]
        per_class_summary[c] = {
            "support": seed_runs[0]["per_class"][c]["support"],
            "recall_mean": float(np.mean(recs)),
            "recall_std": float(np.std(recs)),
            "f1_mean": float(np.mean(cf1s)),
            "f1_std": float(np.std(cf1s)),
        }
    per_source_summary = {}
    if "per_source" in seed_runs[0]:
        for s in seed_runs[0]["per_source"]:
            saccs = [r["per_source"][s]["accuracy"] for r in seed_runs]
            per_source_summary[s] = {
                "n": seed_runs[0]["per_source"][s]["n"],
                "accuracy_mean": float(np.mean(saccs)),
                "accuracy_std": float(np.std(saccs)),
            }
    return {
        "n_seeds": len(seed_runs),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "balanced_accuracy_mean": float(np.mean(bals)),
        "balanced_accuracy_std": float(np.std(bals)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "revolver_38spl_recall_mean": float(np.mean(r38)),
        "revolver_38spl_recall_std": float(np.std(r38)),
        "per_class": per_class_summary,
        "per_source": per_source_summary,
    }
