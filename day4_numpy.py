import time
import numpy as np

np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(42)

a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
print("shape:", a.shape, "| ndim:", a.ndim, "| dtype:", a.dtype, "| size:", a.size)


print("zeros:", np.zeros(3), "| ones:", np.ones(3), "| arange:", np.arange(0, 10, 2), "| linspace:", np.linspace(0, 1, 5))
print("identity:\n", np.eye(3))
print("random ints:\n", rng.integers(0, 10, size=(2, 3)))



X = np.array([[23, 91, 50, 12],
              [88, 34, 67, 45],
              [10, 76, 99, 31],
              [55, 18, 42, 84]])
print("X[1, 2] =", X[1, 2], "| row 0:", X[0], "| column 1:", X[:, 1])
print("block X[1:3, 2:]:\n", X[1:3, 2:])


print("values > 80:", X[X > 80])
print("rows 0 and 3:\n", X[[0, 3]])
clipped = X.copy()
clipped[clipped > 90] = 90
print("clipped at 90:\n", clipped)


view = X[0]
view[0] = -1
print("slice is a VIEW -> X[0, 0] is now", X[0, 0])
X[0, 0] = 23



print("total:", X.sum(), "| column means:", X.mean(axis=0), "| row sums:", X.sum(axis=1))
print("max per row:", X.max(axis=1), "| position of max per row:", X.argmax(axis=1))


data = rng.normal(loc=[50, 1000, 0.5], scale=[10, 300, 0.1], size=(1000, 3))
z = (data - data.mean(axis=0)) / data.std(axis=0)
print("standardized means:", z.mean(axis=0).round(6), "| stds:", z.std(axis=0).round(6))
minmax = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
print("min-max range:", minmax.min(axis=0), "to", minmax.max(axis=0))


prices = np.array([[10.0, 20.0], [30.0, 40.0]])
print("add tax row-wise:\n", prices * np.array([1.08, 1.10]))


v1, v2 = rng.random(1_000_000), rng.random(1_000_000)
t0 = time.perf_counter()
dot_loop = 0.0
for x, y in zip(v1, v2):
    dot_loop += x * y
t1 = time.perf_counter()
dot_np = v1 @ v2
t2 = time.perf_counter()
print(f"loop: {t1 - t0:.3f}s | numpy: {t2 - t1:.5f}s | same answer: {np.isclose(dot_loop, dot_np)}")


A = rng.normal(size=(100, 3))
true_w = np.array([2.0, -1.0, 0.5])
b = A @ true_w + rng.normal(0, 0.01, 100)
w, *_ = np.linalg.lstsq(A, b, rcond=None)
print("true weights:", true_w, "| learned:", w)


P = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
Q = np.array([[0.0, 1.0], [3.0, 3.0]])
dists = np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).sum(axis=-1))
print("pairwise distances (3 x 2):\n", dists)
print("nearest Q for each P:", dists.argmin(axis=1))


def cosine_similarity_matrix(M: np.ndarray, N: np.ndarray) -> np.ndarray:
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    N = N / np.linalg.norm(N, axis=1, keepdims=True)
    return M @ N.T

docs = np.array([[1.0, 1.0, 0.0], [0.9, 1.1, 0.1], [0.0, 0.0, 1.0]])
print("cosine similarity:\n", cosine_similarity_matrix(docs, docs))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def one_hot(y, n_classes):
    return np.eye(n_classes)[y]

print("sigmoid:", sigmoid(np.array([-5.0, 0.0, 5.0])))
print("softmax:\n", softmax(np.array([[2.0, 1.0, 0.1], [1000.0, 1001.0, 1002.0]])))
print("one-hot:\n", one_hot(np.array([0, 2, 1]), 3))


m = np.arange(12).reshape(3, 4)
print("reshape(3, 4):\n", m, "\ntranspose shape:", m.T.shape, "| flatten:", m.ravel())
print("where even keep else -1:\n", np.where(m % 2 == 0, m, -1))


def min_max_scale(M):
    return (M - M.min(axis=0)) / (M.max(axis=0) - M.min(axis=0))

def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

def top_k_indices(scores, k):
    return np.argsort(scores)[::-1][:k]
