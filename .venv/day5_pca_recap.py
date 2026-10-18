import numpy as np

np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(7)

n = 500
sqft = rng.normal(1800, 400, n)
rooms = sqft / 300 + rng.normal(0, 0.5, n)
noise = rng.normal(0, 1, n)
X = np.column_stack([sqft, rooms, noise])

Z = (X - X.mean(axis=0)) / X.std(axis=0)
cov = np.cov(Z, rowvar=False)
print("covariance matrix:\n", cov)


eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]
explained = eigvals / eigvals.sum()
print("eigenvalues:", eigvals)
print("explained variance ratio:", explained)
print("first component (weights on sqft, rooms, noise):", eigvecs[:, 0])

X_pca = Z @ eigvecs[:, :2]
print("shape before:", Z.shape, "| after PCA:", X_pca.shape)
print(f"2 components keep {explained[:2].sum():.1%} of the information")
