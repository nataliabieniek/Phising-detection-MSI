import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_curve


def fpr_fnr(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return fp / max(np.sum(y_true == 0), 1), fn / max(np.sum(y_true == 1), 1)


def best_f1_threshold(y_true, scores) -> float:
    """Próg wybierany na zbiorze walidacyjnym."""
    thresholds = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 197)))
    f1s = [f1_score(y_true, scores >= t) for t in thresholds]
    return float(thresholds[int(np.argmax(f1s))])


def evaluate(y_true, scores, threshold: float = 0.5) -> dict:
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)
    fpr, fnr = fpr_fnr(y_true, y_pred)
    roc_fpr, roc_tpr, _ = roc_curve(y_true, scores)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "fpr": fpr,
        "fnr": fnr,
        "fpr_at_recall95": float(roc_fpr[np.argmax(roc_tpr >= 0.95)]),
        "fnr_at_fpr1": float(1 - roc_tpr[roc_fpr <= 0.01].max()),
    }
