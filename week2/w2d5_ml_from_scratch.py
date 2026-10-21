"""Week 2, Day 5 (part 2): w2d5_ml_from_scratch.py
The ML workflow + core algorithms written from scratch in NumPy, checked against scikit-learn.

Install once: uv pip install scikit-learn
Run: python w2d5_ml_from_scratch.py
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

np.set_printoptions(precision=3, suppress=True)

# ---------------------------------------------------------------------------
# Part 1: The ML workflow: data -> split -> baseline -> model -> evaluate
# ---------------------------------------------------------------------------
X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, weights=[0.7, 0.3], random_state=0)
print("data:", X.shape, "| positive rate:", y.mean())

rng = np.random.default_rng(0)
idx = rng.permutation(len(X))
train, val, test = idx[:600], idx[600:800], idx[800:]
X_tr, X_val, X_te = X[train], X[val], X[test]
y_tr, y_val, y_te = y[train], y[val], y[test]
print("split sizes  train/val/test:", len(train), len(val), len(test))

mean, std = X_tr.mean(axis=0), X_tr.std(axis=0)  # learned on TRAIN only
X_tr, X_val, X_te = (X_tr - mean) / std, (X_val - mean) / std, (X_te - mean) / std

majority = int(round(y_tr.mean()))
baseline_acc = np.mean(y_te == majority)
print(f"baseline (always predict {majority}): test accuracy {baseline_acc:.3f}")


# ---------------------------------------------------------------------------
# Part 2: Logistic regression from scratch (gradient descent + L2)
# ---------------------------------------------------------------------------
class LogisticRegressionScratch:
    def __init__(self, lr: float = 0.1, epochs: int = 2000, l2: float = 0.0) -> None:
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegressionScratch:
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.epochs):
            p = self.sigmoid(X @ self.w + self.b)
            self.w -= self.lr * (X.T @ (p - y) / n + self.l2 * self.w)
            self.b -= self.lr * np.mean(p - y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.sigmoid(X @ self.w + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


for l2 in (0.0, 0.1):
    model = LogisticRegressionScratch(lr=0.5, epochs=3000, l2=l2).fit(X_tr, y_tr)
    print(f"L2={l2}: validation accuracy {np.mean(model.predict(X_val) == y_val):.3f} | weights {model.w}")

mine = LogisticRegressionScratch(lr=0.5, epochs=3000).fit(X_tr, y_tr)
sk = LogisticRegression(C=1e6, max_iter=5000).fit(X_tr, y_tr)
print("scratch weights:", mine.w, "| sklearn:", sk.coef_[0])
print(f"TEST accuracy  scratch {np.mean(mine.predict(X_te) == y_te):.3f} | sklearn {sk.score(X_te, y_te):.3f}")


# ---------------------------------------------------------------------------
# Part 3: k-Nearest Neighbors (vectorized)
# ---------------------------------------------------------------------------
class KNNClassifier:
    def __init__(self, k: int = 5) -> None:
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> KNNClassifier:
        self.X, self.y = X, y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        d2 = (X**2).sum(1)[:, None] + (self.X**2).sum(1)[None, :] - 2 * X @ self.X.T
        nearest = np.argpartition(d2, self.k, axis=1)[:, : self.k]
        return np.array([np.bincount(row).argmax() for row in self.y[nearest]])


for k in (1, 7, 25):
    acc = np.mean(KNNClassifier(k).fit(X_tr, y_tr).predict(X_val) == y_val)
    print(f"k-NN k={k:<2}: validation accuracy {acc:.3f}")


# ---------------------------------------------------------------------------
# Part 4: k-means with k-means++ initialization (unsupervised)
# ---------------------------------------------------------------------------
def kmeans(X: np.ndarray, k: int, iters: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    r = np.random.default_rng(seed)
    centers = [X[r.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        centers.append(X[r.choice(len(X), p=d2 / d2.sum())])
    C = np.array(centers)
    for _ in range(iters):
        labels = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        new_C = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j] for j in range(k)])
        if np.allclose(new_C, C):
            break
        C = new_C
    return labels, C, float(((X - C[labels]) ** 2).sum())


Xb, _ = make_blobs(n_samples=600, centers=4, cluster_std=1.0, random_state=3)
print("k-means elbow (inertia):", {k: round(kmeans(Xb, k)[2]) for k in range(2, 7)})


# ---------------------------------------------------------------------------
# Part 5: Decision-tree split with Gini impurity
# ---------------------------------------------------------------------------
def gini(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)
    return 1.0 - float(np.sum(p**2))


def best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
    best = (-1, 0.0, gini(y))
    for j in range(X.shape[1]):
        for t in np.unique(np.quantile(X[:, j], np.linspace(0.05, 0.95, 19))):
            left, right = y[X[:, j] <= t], y[X[:, j] > t]
            w = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
            if w < best[2]:
                best = (j, float(t), w)
    return best


f, t, g = best_split(X_tr, y_tr)
print(f"best first split: feature {f} <= {t:.3f} | gini {gini(y_tr):.3f} -> {g:.3f}")


# ---------------------------------------------------------------------------
# Part 6: ROC-AUC from scratch (ranking formula)
# ---------------------------------------------------------------------------
def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    ranks = np.empty(len(scores))
    ranks[np.argsort(scores)] = np.arange(1, len(scores) + 1)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


scores = mine.predict_proba(X_te)
print(f"ROC-AUC scratch {roc_auc(y_te, scores):.4f} | sklearn {roc_auc_score(y_te, scores):.4f}")


# ---------------------------------------------------------------------------
# Part 7: Hands-on exercise (solutions)
# ---------------------------------------------------------------------------
def precision_at_threshold(y_true: np.ndarray, proba: np.ndarray, t: float) -> float:
    pred = proba >= t
    return float(y_true[pred].mean()) if pred.any() else 0.0


p_low, p_high = precision_at_threshold(y_te, scores, 0.3), precision_at_threshold(y_te, scores, 0.7)
assert p_high >= p_low

best_k = max((1, 3, 5, 7, 9, 15, 25), key=lambda k: np.mean(KNNClassifier(k).fit(X_tr, y_tr).predict(X_val) == y_val))
assert best_k in (1, 3, 5, 7, 9, 15, 25)

assert gini(np.array([0, 0, 1, 1])) == 0.5 and gini(np.array([1, 1, 1])) == 0.0
print(f"precision at 0.3: {p_low:.3f} | at 0.7: {p_high:.3f} | best k on validation: {best_k}")
print("All exercises passed!")
