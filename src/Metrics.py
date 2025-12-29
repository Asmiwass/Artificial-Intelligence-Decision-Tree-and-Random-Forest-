import numpy as np


def _binary_metrics(y_true, y_pred, pos_label):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TP = np.sum((y_true == pos_label) & (y_pred == pos_label))
    TN = np.sum((y_true != pos_label) & (y_pred != pos_label))
    FP = np.sum((y_true != pos_label) & (y_pred == pos_label))
    FN = np.sum((y_true == pos_label) & (y_pred != pos_label))

    acc = (TP + TN) / max(TP + TN + FP + FN, 1)
    prec = TP / max(TP + FP, 1)
    rec = TP / max(TP + FN, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)

    return acc, prec, rec, f1


def _macro_multiclass_metrics(y_true, y_pred, labels=None):
    """
    Macro-averaged precision/recall/F1 for multi-class classification.
    Accuracy is standard overall accuracy.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    acc = np.mean(y_true == y_pred)

    precs, recs, f1s = [], [], []
    for c in labels:
        TP = np.sum((y_true == c) & (y_pred == c))
        FP = np.sum((y_true != c) & (y_pred == c))
        FN = np.sum((y_true == c) & (y_pred != c))

        prec = TP / max(TP + FP, 1)
        rec = TP / max(TP + FN, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)

        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    return acc, float(np.mean(precs)), float(np.mean(recs)), float(np.mean(f1s))


def compute_metrics(y_true, y_pred, pos="Yes"):
    """
    Compute metrics for binary or multi-class tasks.

    - If y_true has exactly 2 unique labels: compute binary metrics using pos label.
    - Otherwise: compute macro-averaged precision/recall/F1.
    """
    y_true_arr = np.asarray(y_true)
    unique_labels = np.unique(y_true_arr)

    if len(unique_labels) == 2:
        # If pos not in labels, fallback to the "second" label to avoid zeros
        pos_label = pos if pos in unique_labels else unique_labels[1]
        return _binary_metrics(y_true, y_pred, pos_label)

    # Multi-class (e.g. Penguins)
    return _macro_multiclass_metrics(y_true, y_pred, labels=unique_labels)
