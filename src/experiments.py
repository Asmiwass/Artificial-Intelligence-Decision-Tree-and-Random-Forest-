import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

# ===== IMPORT MODELS & METRICS (CORRECT PATH) =====
from src.models import (
    id3_train,
    id3_predict,
    train_random_forest_feature_bagging,
    predict_random_forest_feature_bagging,
    add_feature_noise,
    add_label_noise
)

from src.Metrics import compute_metrics


# Q6a — Training size experiment
def training_size_experiment(
    X_train, y_train,
    X_test, y_test,
    train_fractions,
    n_repeats=5
):
    results = []

    for frac in train_fractions:
        f1_values = []

        for seed in range(n_repeats):
            X_shuf, y_shuf = shuffle(
                X_train, y_train, random_state=seed
            )

            n_train = int(frac * len(X_shuf))

            tree = id3_train(
                X_shuf.iloc[:n_train],
                y_shuf.iloc[:n_train]
            )

            y_pred = id3_predict(tree, X_test)
            _, _, _, f1 = compute_metrics(y_test.values, y_pred)

            f1_values.append(f1)

        results.append((
            frac,
            np.mean(f1_values),
            np.std(f1_values)
        ))

    return pd.DataFrame(
        results,
        columns=["Train fraction", "F1 mean", "F1 std"]
    )


# Q6b — Feature noise experiment
def feature_noise_experiment(
    X_train, y_train,
    X_test, y_test,
    noise_levels,
    n_repeats=5
):
    results = []

    for noise in noise_levels:
        f1_values = []

        for seed in range(n_repeats):
            X_noisy = add_feature_noise(
                X_train,
                noise_level=noise,
                random_state=seed
            )

            tree = id3_train(X_noisy, y_train)
            y_pred = id3_predict(tree, X_test)

            _, _, _, f1 = compute_metrics(y_test.values, y_pred)
            f1_values.append(f1)

        results.append((
            noise,
            np.mean(f1_values),
            np.std(f1_values)
        ))

    return pd.DataFrame(
        results,
        columns=["Noise level", "F1 mean", "F1 std"]
    )


# Q6c — Label noise experiment
def label_noise_experiment(
    X_train, y_train,
    X_test, y_test,
    noise_levels,
    n_repeats=5
):
    results = []

    for noise in noise_levels:
        f1_values = []

        for seed in range(n_repeats):
            y_noisy = add_label_noise(
                y_train,
                noise_level=noise,
                random_state=seed
            )

            tree = id3_train(X_train, y_noisy)
            y_pred = id3_predict(tree, X_test)

            _, _, _, f1 = compute_metrics(y_test.values, y_pred)
            f1_values.append(f1)

        results.append((
            noise,
            np.mean(f1_values),
            np.std(f1_values)
        ))

    return pd.DataFrame(
        results,
        columns=["Label noise level", "F1 mean", "F1 std"]
    )


# Q6d — Pruning / max depth experiment
def pruning_experiment(
    X_train, y_train,
    X_test, y_test,
    depths
):
    results = []

    for d in depths:
        tree = id3_train(X_train, y_train, max_depth=d)
        y_pred = id3_predict(tree, X_test)

        acc, prec, rec, f1 = compute_metrics(y_test.values, y_pred)

        results.append((
            d,
            acc,
            prec,
            rec,
            f1
        ))

    return pd.DataFrame(
        results,
        columns=["Max depth", "Accuracy", "Precision", "Recall", "F1-score"]
    )


# Q7 — Random Forest cross-validation (feature bagging)
def rf_cv_experiment(
    X_train, y_train,
    param_grid,
    n_splits=5,
    random_state=42
):
    cv_results = []

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    for n_trees in param_grid["n_trees"]:
        for max_depth in param_grid["max_depth"]:
            for max_features in param_grid["max_features"]:

                f1_scores = []

                for train_idx, val_idx in kf.split(X_train):
                    X_tr = X_train.iloc[train_idx]
                    y_tr = y_train.iloc[train_idx]

                    X_val = X_train.iloc[val_idx]
                    y_val = y_train.iloc[val_idx]

                    forest = train_random_forest_feature_bagging(
                        X_tr,
                        y_tr,
                        n_trees=n_trees,
                        max_depth=max_depth,
                        max_features=max_features
                    )

                    y_val_pred = predict_random_forest_feature_bagging(
                        forest, X_val
                    )

                    _, _, _, f1 = compute_metrics(
                        y_val.values, y_val_pred
                    )
                    f1_scores.append(f1)

                cv_results.append((
                    n_trees,
                    max_depth,
                    max_features,
                    np.mean(f1_scores),
                    np.std(f1_scores)
                ))

    return pd.DataFrame(
        cv_results,
        columns=[
            "n_trees",
            "max_depth",
            "max_features",
            "F1 mean",
            "F1 std"
        ]
    )
