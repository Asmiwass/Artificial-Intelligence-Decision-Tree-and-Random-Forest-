import numpy as np
import pandas as pd
from collections import Counter


## ID3 Decision Tree

def entropy(y):
    """
    Compute Shannon entropy of a label vector.
    """
    counts = np.array(list(Counter(y).values()))
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-12))



def information_gain(y_parent, y_children):
    """
    Compute Information Gain between a parent node and its children.
    """
    H_parent = entropy(y_parent)
    n = len(y_parent)

    weighted_entropy = 0.0
    for child in y_children:
        if len(child) > 0:
            weighted_entropy += (len(child) / n) * entropy(child)

    return H_parent - weighted_entropy



def candidate_thresholds(x):
    """
    Generate candidate thresholds for a continuous feature.
    """
    values = np.sort(x.unique())
    if len(values) < 2:
        return []
    return (values[:-1] + values[1:]) / 2



def best_split(X, y, feature):
    """
    Find the best split for a given feature (categorical or continuous).
    """
    if X[feature].dtype.kind in "bifc":  # continuous
        best_ig, best_tau = -np.inf, None
        for tau in candidate_thresholds(X[feature]):
            left = y[X[feature] <= tau]
            right = y[X[feature] > tau]
            ig = information_gain(y, [left, right])
            if ig > best_ig:
                best_ig, best_tau = ig, tau
        return best_ig, ("continuous", best_tau)

    else:  # categorical
        children = [y[X[feature] == v] for v in X[feature].unique()]
        ig = information_gain(y, children)
        return ig, ("categorical", None)



from collections import Counter
import numpy as np

def id3_train(X, y, max_depth=None, depth=0):
    """
    Train an ID3 decision tree.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix at the current node
    y : pandas.Series
        Labels corresponding to X
    max_depth : int or None
        Maximum depth of the tree (None = no limit)
    depth : int
        Current depth (used internally for recursion)
    """

    # --- Pure node ---
    if y.nunique() == 1:
        return y.iloc[0]

    # --- Stopping conditions ---
    if X.empty or (max_depth is not None and depth >= max_depth):
        return y.value_counts().idxmax()

    # --- Find best feature to split ---
    best_feature = None
    best_info = None
    best_ig = -np.inf

    for feature in X.columns:
        ig, info = best_split(X, y, feature)
        if ig > best_ig:
            best_ig = ig
            best_feature = feature
            best_info = info

    # --- No informative split ---
    if best_ig <= 0 or best_feature is None:
        return y.value_counts().idxmax()

    # --- Create decision node ---
    tree = {best_feature: {}}

    # Store majority class for unseen values (robustness)
    tree["_majority_class"] = y.value_counts().idxmax()

    split_type, tau = best_info

    # --- Continuous feature split ---
    if split_type == "continuous":
        tree[best_feature]["threshold"] = tau

        mask_left = X[best_feature] <= tau
        mask_right = X[best_feature] > tau

        tree[best_feature]["<=tau"] = id3_train(
            X[mask_left],
            y[mask_left],
            max_depth=max_depth,
            depth=depth + 1
        )

        tree[best_feature][">tau"] = id3_train(
            X[mask_right],
            y[mask_right],
            max_depth=max_depth,
            depth=depth + 1
        )

    # --- Categorical feature split ---
    else:
        for value in X[best_feature].unique():
            mask = X[best_feature] == value

            subtree = id3_train(
                X[mask].drop(columns=[best_feature]),
                y[mask],
                max_depth=max_depth,
                depth=depth + 1
            )

            tree[best_feature][value] = subtree

    return tree





def id3_predict(tree, X):
    """
    Predict labels for a dataset.
    """
    return X.apply(lambda row: id3_predict_one(tree, row), axis=1).values




def id3_predict_one(tree, x):
    """
    Predict label for a single sample.
    """
    if not isinstance(tree, dict):
        return tree

    feature = next(iter(tree))

    # Continuous feature
    if "threshold" in tree[feature]:
        tau = tree[feature]["threshold"]
        if x[feature] <= tau:
            return id3_predict_one(tree[feature]["<=tau"], x)
        else:
            return id3_predict_one(tree[feature][">tau"], x)

    # Categorical feature
    else:
        value = x[feature]
        if value in tree[feature]:
            return id3_predict_one(tree[feature][value], x)
        else:
            # unseen category → fallback
            return tree["_majority_class"]



## Random Forest


def bootstrap_sample(X, y, random_state=None):
    rng = np.random.default_rng(random_state)
    n = len(X)
    indices = rng.integers(0, n, size=n)
    return X.iloc[indices], y.iloc[indices]
def train_random_forest_bagging(
    X_train, y_train,
    n_trees=50,
    max_depth=None,
    random_state=42
):
    forest = []
    rng = np.random.default_rng(random_state)

    for i in range(n_trees):
        X_boot, y_boot = bootstrap_sample(
            X_train, y_train,
            random_state=rng.integers(1e9)
        )

        tree = id3_train(
            X_boot, y_boot,
            max_depth=max_depth
        )

        forest.append(tree)

    return forest



def predict_random_forest(forest, X):
    predictions = []

    for _, row in X.iterrows():
        tree_preds = [id3_predict_one(tree, row) for tree in forest]
        vote = Counter(tree_preds).most_common(1)[0][0]
        predictions.append(vote)

    return np.array(predictions)



def train_random_forest_feature_bagging(
    X_train, y_train,
    n_trees=50,
    max_depth=None,
    max_features=None,
    random_state=42
):
    forest = []
    rng = np.random.default_rng(random_state)

    all_features = X_train.columns.tolist()
    d = len(all_features)

    # Handle max_features properly
    if max_features is None:
        n_features = int(np.sqrt(d))
    elif max_features == "sqrt":
        n_features = int(np.sqrt(d))
    else:
        n_features = int(max_features)

    for i in range(n_trees):
        # Bootstrap sampling
        X_boot, y_boot = bootstrap_sample(
            X_train, y_train,
            random_state=rng.integers(1e9)
        )

        # Random feature subset
        selected_features = rng.choice(
            all_features,
            size=n_features,
            replace=False
        )

        X_boot_sub = X_boot[selected_features]

        tree = id3_train(
            X_boot_sub,
            y_boot,
            max_depth=max_depth
        )

        forest.append((tree, selected_features))

    return forest


def predict_random_forest_feature_bagging(forest, X):
    predictions = []

    for _, row in X.iterrows():
        tree_preds = []
        for tree, features in forest:
            row_sub = row[features]
            tree_preds.append(id3_predict_one(tree, row_sub))

        vote = Counter(tree_preds).most_common(1)[0][0]
        predictions.append(vote)

    return np.array(predictions)



def add_label_noise(y, noise_level, random_state=None):
    """
    Add label noise by randomly changing the label of a fraction of samples.

    Works for binary AND multi-class classification.
    - With probability = noise_level, each label is replaced by a different
      label chosen uniformly among the other classes.

    Parameters
    ----------
    y : pandas.Series
        Original labels (can be strings or ints)
    noise_level : float
        Fraction of labels to corrupt (between 0 and 1)
    random_state : int or None
        Seed for reproducibility

    Returns
    -------
    y_noisy : pandas.Series
        Copy of y with noisy labels
    """
    if not (0.0 <= noise_level <= 1.0):
        raise ValueError("noise_level must be in [0, 1].")

    rng = np.random.default_rng(random_state)
    y_noisy = y.copy()

    classes = pd.unique(y_noisy)
    if len(classes) < 2:
        # nothing to flip
        return y_noisy

    mask = rng.random(len(y_noisy)) < noise_level
    idx = np.where(mask)[0]

    for i in idx:
        current = y_noisy.iloc[i]
        other_classes = classes[classes != current]
        y_noisy.iloc[i] = rng.choice(other_classes)

    return y_noisy

import numpy as np
import pandas as pd

def add_feature_noise(X, noise_level, random_state=None):
    """
    Add noise to feature matrix X.
    - Numerical features: add Gaussian noise N(0, (noise_level * std)^2)
    - Categorical features: with prob=noise_level, replace by a random category

    Works for both datasets (Thyroid + Penguins).

    Parameters
    ----------
    X : pandas.DataFrame
    noise_level : float in [0,1]
    random_state : int or None

    Returns
    -------
    X_noisy : pandas.DataFrame
    """
    if not (0.0 <= noise_level <= 1.0):
        raise ValueError("noise_level must be in [0, 1].")

    rng = np.random.default_rng(random_state)
    X_noisy = X.copy()

    num_cols = X_noisy.select_dtypes(include=[np.number]).columns
    cat_cols = X_noisy.select_dtypes(exclude=[np.number]).columns

    # ---- Numerical noise ----
    for col in num_cols:
        std = X_noisy[col].std()
        if pd.isna(std) or std == 0:
            continue
        noise = rng.normal(loc=0.0, scale=noise_level * std, size=len(X_noisy))
        X_noisy[col] = X_noisy[col] + noise

    # ---- Categorical noise ----
    for col in cat_cols:
        values = pd.unique(X_noisy[col].dropna())
        if len(values) == 0:
            continue

        mask = rng.random(len(X_noisy)) < noise_level
        if mask.sum() == 0:
            continue

        X_noisy.loc[mask, col] = rng.choice(values, size=int(mask.sum()), replace=True)

    return X_noisy
