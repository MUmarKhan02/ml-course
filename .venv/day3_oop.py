from __future__ import annotations
class Customer:
    """A bank customer with a balance."""

    bank_name = "NorthStar Pay"

    def __init__(self, name: str, balance: float = 0.0) -> None:
        self.name = name
        self.balance = balance

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("deposit must be positive")
        self.balance += amount

    def __repr__(self) -> str:
        return f"Customer(name={self.name!r}, balance={self.balance:,.2f})"


ava = Customer("Ava", 1000)
ben = Customer("Ben")
ava.deposit(250)
print(ava, ben)
print("Same bank:", ava.bank_name, "|", ben.bank_name)
try:
    ava.deposit(-5)
except ValueError as e:
    print("Blocked:", e)



class Dataset:
    def __init__(self, rows: list[list[float]], labels: list[int]) -> None:
        self.rows = rows
        self.labels = labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[list[float], int]:
        return self.rows[i], self.labels[i]

    @property
    def positive_rate(self) -> float:
        return sum(self.labels) / len(self.labels)

ds = Dataset([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], [0, 1, 0, 1])
print("len:", len(ds), "| ds[1]:", ds[1], "| positive_rate:", ds.positive_rate)
for features, label in ds:
    pass
print("Iteration works, last label:", label)




from dataclasses import asdict, dataclass, field
from enum import Enum

class Task(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

@dataclass(frozen=True)
class TrainConfig:
    task: Task
    learning_rate: float = 0.01
    epochs: int = 100
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not 0 < self.learning_rate < 1:
            raise ValueError(f"learning_rate must be in (0, 1), got {self.learning_rate}")


cfg = TrainConfig(task=Task.REGRESSION, learning_rate=0.05, epochs=2000, tags=("baseline",))
print(cfg)
print(asdict(cfg))
try:
    TrainConfig(task=Task.REGRESSION, learning_rate=5)
except ValueError as e:
    print("Validation:", e)
try:
    cfg.epochs = 10
except Exception as e:
    print("Frozen:", type(e).__name__)




from abc import ABC, abstractmethod
import numpy as np

class BaseEstimator(ABC):
    def __init__(self) -> None:
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted yet. Call .fit() first.")


class MeanBaseline(BaseEstimator):
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        self.is_fitted = True
        return self

    def predict(self, X):
        self.check_fitted()
        return np.full(len(X), self.mean_)

class LinearRegressionGD(BaseEstimator):
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        n, d = X.shape
        self.coef_ = np.zeros(d)
        self.intercept_ = 0.0
        for _ in range(self.epochs):
            error = X @ self.coef_ + self.intercept_ - y
            self.coef_ -= self.learning_rate * (2 / n) * (X.T @ error)
            self.intercept_ -= self.learning_rate * (2 / n) * error.sum()
        self.is_fitted = True
        return self

    def predict(self, X):
        self.check_fitted()
        return X @ self.coef_ + self.intercept_

try:
    BaseEstimator()
except TypeError as e:
    print("Abstract:", str(e).split(" with")[0])


class StandardScaler:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-12
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

class Pipeline:
    def __init__(self, scaler: StandardScaler, model: BaseEstimator) -> None:
        self.scaler = scaler
        self.model = model

    def fit(self, X, y):
        self.model.fit(self.scaler.fit(X).transform(X), y)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


from typing import Protocol, runtime_checkable

@runtime_checkable
class Predictor(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...

def rmse(model: Predictor, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((model.predict(X) - y) ** 2)))




rng = np.random.default_rng(42)
n = 500
sqft = rng.uniform(600, 4000, n)
beds = rng.integers(1, 6, n)
age = rng.uniform(0, 80, n)
X = np.column_stack([sqft, beds, age])
price = 50_000 + 180 * sqft + 12_000 * beds - 900 * age + rng.normal(0, 25_000, n)
X_tr, X_te, y_tr, y_te = X[:400], X[400:], price[:400], price[400:]

baseline = MeanBaseline().fit(X_tr, y_tr)
pipe = Pipeline(StandardScaler(), LinearRegressionGD(learning_rate=cfg.learning_rate, epochs=cfg.epochs)).fit(X_tr, y_tr)



print("Pipeline is a Predictor:", isinstance(pipe, Predictor))
print(f"Baseline RMSE: ${rmse(baseline, X_te, y_te):,.0f}")
print(f"Model RMSE:    ${rmse(pipe, X_te, y_te):,.0f}")
try:
    LinearRegressionGD().predict(X_te)
except RuntimeError as e:
    print("Guard rail:", e)



class BankAccount:
    ...
    def withdraw(self, amount):
        if amount > self.balance:
            raise ValueError("insufficient funds")
        self.balance -= amount

class SavingsAccount(BankAccount):
    def __init__(self, owner, balance=0.0, rate=0.04):
        super().__init__(owner, balance)
        self.rate = rate

    def add_interest(self):
        self.balance += self.balance * self.rate

# in Metric:
    def beats(self, other):
        return self.value > other.value if self.higher_is_better else self.value < other.value

class MedianBaseline(MeanBaseline):
    def fit(self, X, y):
        self.mean_ = float(np.median(y))
        self.is_fitted = True
        return self

