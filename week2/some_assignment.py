"""Capstone (simple version): Day 1 to Week 2 Day 4 on one small messy cafe dataset.

Run:  pip install pandas numpy scipy duckdb matplotlib polars pyarrow
      python capstone_cafe_simple.py
"""

import csv
import functools
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.width", 120)
np.set_printoptions(precision=3, suppress=True)


def title(text):
    print(f"\n{'=' * 60}\n{text}\n{'=' * 60}")


# ---------------------------------------------------------------
# Part 0: make the raw (messy) dataset
# ---------------------------------------------------------------
rng = np.random.default_rng(2026)
cities = ["Austin", "Denver", "Seattle", "Boston"]
cats = ["Coffee", "Tea", "Bakery", "Merch"]

customers_rows = []
for i in range(1, 31):
    city = rng.choice(cities)
    customers_rows.append({
        "customer_id": f"C{i:02d}",
        "email": f"user{i}@Example.com",
        "age": int(rng.integers(18, 65)) if rng.random() > 0.1 else None,   # some missing
        "city": rng.choice([city, f" {city.lower()} ", city.upper()]),      # messy text
        "is_member": bool(rng.random() < 0.45),
    })
pd.DataFrame(customers_rows).to_csv("customers_raw.csv", index=False)

orders_rows = []
for i in range(1, 121):
    cat = rng.choice(cats, p=[0.5, 0.2, 0.22, 0.08])
    price = rng.uniform(15, 60) if cat == "Merch" else rng.uniform(3, 8)
    day = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 240)))
    orders_rows.append({
        "order_id": f"O{i:03d}",
        "customer_id": f"C{int(rng.integers(1, 31)):02d}",
        "order_date": day.date().isoformat(),
        "category": cat,
        "amount": f"${price:,.2f}",                                          # text, not number
        "returned": "yes" if rng.random() < 0.07 else "no",
    })
orders_rows[10]["amount"] = "$480.00"                                        # outlier
orders_rows[20]["amount"] = None                                             # missing
orders_rows[50]["amount"] = None
orders_rows += [dict(orders_rows[j]) for j in (3, 33, 63)]                   # duplicates
pd.DataFrame(orders_rows).to_csv("orders_raw.csv", index=False)

# ---------------------------------------------------------------
# Part 1 (Day 1): basics and collections
# ---------------------------------------------------------------
title("DAY 1: basics and collections")
with open("orders_raw.csv", newline="") as f:
    rows = list(csv.DictReader(f))

amounts = [float(r["amount"].replace("$", "")) for r in rows if r["amount"]]
print("rows:", len(rows), "| top 3 amounts:", sorted(amounts, reverse=True)[:3])
print("clean text:", repr(" austin ".strip().title()))
print("labels:", ["big" if a >= 10 else "small" for a in amounts[:5]])

cat_counts = Counter(r["category"] for r in rows)
print("category counts:", cat_counts.most_common(2))

orders_by_customer = defaultdict(list)
for r in rows:
    orders_by_customer[r["customer_id"]].append(r["order_id"])
print("customer C01 orders:", orders_by_customer["C01"])

coffee = {r["customer_id"] for r in rows if r["category"] == "Coffee"}
bakery = {r["customer_id"] for r in rows if r["category"] == "Bakery"}
print("coffee AND bakery:", len(coffee & bakery), "| coffee only:", len(coffee - bakery))


def two_sum(nums, target):
    seen = {}
    for i, v in enumerate(nums):
        if target - v in seen:
            return seen[target - v], i
        seen[v] = i


print("two_sum:", two_sum([2, 7, 11, 15], 9))

# ---------------------------------------------------------------
# Part 2 (Day 2): functions, decorators, generators
# ---------------------------------------------------------------
title("DAY 2: functions, decorators, generators")


def parse_amount(text, default=0.0):
    """Turn '$1,204.50' into 1204.5."""
    return float(text.replace("$", "").replace(",", "")) if text else default


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"  {func.__name__} took {(time.perf_counter() - start) * 1000:.2f} ms")
        return result
    return wrapper


@timer
def total_revenue(data):
    return sum(parse_amount(r["amount"]) for r in data)


print("revenue (raw, with duplicates):", round(total_revenue(rows), 2))


@functools.lru_cache(maxsize=None)
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)


print("fib(60) with cache:", fib(60))


def big_orders(data, threshold=25):          # generator: yields one at a time
    for r in data:
        if parse_amount(r["amount"]) > threshold:
            yield r["order_id"]


print("big orders:", list(big_orders(rows)))

# ---------------------------------------------------------------
# Part 3 (Day 3): OOP
# ---------------------------------------------------------------
title("DAY 3: OOP")


def load_clean_data():
    """Same cleaning as Part 6, packaged as a function."""
    customers = pd.read_csv("customers_raw.csv")
    orders = pd.read_csv("orders_raw.csv", parse_dates=["order_date"])
    customers["city"] = customers["city"].str.strip().str.title()
    customers["age"] = customers["age"].fillna(customers["age"].median())
    orders["amount"] = pd.to_numeric(orders["amount"].str.replace(r"[$,]", "", regex=True))
    orders["returned"] = orders["returned"].map({"yes": True, "no": False})
    orders = orders.drop_duplicates().dropna(subset=["amount"])
    q1, q3 = orders["amount"].quantile([0.25, 0.75])
    orders["amount"] = orders["amount"].clip(upper=q3 + 3 * (q3 - q1))
    return customers, orders


customers, orders = load_clean_data()


@dataclass(frozen=True)
class Config:
    learning_rate: float = 0.05
    epochs: int = 1000


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def predict(self, X): ...


class MeanBaseline(BaseModel):
    def fit(self, X, y):
        self.mean_ = y.mean()
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class LinearModel(BaseModel):
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def fit(self, X, y):
        self.w, self.b = np.zeros(X.shape[1]), 0.0
        for _ in range(self.cfg.epochs):
            err = X @ self.w + self.b - y
            self.w -= self.cfg.learning_rate * 2 / len(X) * (X.T @ err)
            self.b -= self.cfg.learning_rate * 2 / len(X) * err.sum()
        return self

    def predict(self, X):
        return X @ self.w + self.b


# predict each customer's total spend from age, number of orders, membership
feat = (orders.groupby("customer_id").agg(n_orders=("order_id", "count"), spend=("amount", "sum"))
        .join(customers.set_index("customer_id")[["age", "is_member"]]))
X = feat[["age", "n_orders", "is_member"]].astype(float).to_numpy()
y = feat["spend"].to_numpy()
Xs = (X - X.mean(axis=0)) / X.std(axis=0)                  # standardise (NumPy broadcasting)
rmse = lambda m: float(np.sqrt(np.mean((m.predict(Xs) - y) ** 2)))
print(f"baseline RMSE ${rmse(MeanBaseline().fit(Xs, y)):.2f} | linear model RMSE ${rmse(LinearModel(Config()).fit(Xs, y)):.2f}")

# ---------------------------------------------------------------
# Part 4 (Day 4): NumPy
# ---------------------------------------------------------------
title("DAY 4: NumPy")
print("shape:", X.shape, "| column means:", X.mean(axis=0))
print("customers with more than 5 orders:", int((X[:, 1] > 5).sum()))   # boolean mask
print("top 3 spenders (row index):", np.argsort(y)[::-1][:3])

v1, v2 = rng.random(300_000), rng.random(300_000)
t0 = time.perf_counter(); sum(a * b for a, b in zip(v1, v2)); t1 = time.perf_counter()
v1 @ v2; t2 = time.perf_counter()
print(f"dot product: loop {t1 - t0:.3f}s vs numpy {t2 - t1:.5f}s")

profile = orders.pivot_table(index="customer_id", columns="category", values="amount", aggfunc="sum", fill_value=0).to_numpy()
unit = profile / np.linalg.norm(profile, axis=1, keepdims=True)
print("cosine similarity of first 3 customers' buying habits:\n", (unit[:3] @ unit[:3].T))

# ---------------------------------------------------------------
# Part 5 (Day 5): PCA
# ---------------------------------------------------------------
title("DAY 5: PCA")
eigvals, eigvecs = np.linalg.eigh(np.cov(Xs, rowvar=False))
eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]          # biggest first
print("explained variance:", eigvals / eigvals.sum())
print("after PCA to 2 components:", (Xs @ eigvecs[:, :2]).shape)

# ---------------------------------------------------------------
# Part 6 (W2D1): pandas cleaning, step by step
# ---------------------------------------------------------------
title("WEEK 2 DAY 1: pandas cleaning")
c = pd.read_csv("customers_raw.csv")
o = pd.read_csv("orders_raw.csv")
print("missing:", o["amount"].isna().sum(), "| duplicates:", o.duplicated().sum(), "| cities:", sorted(c["city"].unique()))

o["amount"] = pd.to_numeric(o["amount"].str.replace(r"[$,]", "", regex=True))   # "$4.50" -> 4.5
o["order_date"] = pd.to_datetime(o["order_date"])
o["returned"] = o["returned"].map({"yes": True, "no": False})
c["city"] = c["city"].str.strip().str.title()
o = o.drop_duplicates().dropna(subset=["amount"])
c["age"] = c["age"].fillna(c["age"].median())

q1, q3 = o["amount"].quantile([0.25, 0.75])
fence = q3 + 3 * (q3 - q1)
print(f"outliers above ${fence:.2f}: {(o['amount'] > fence).sum()} -> capped")
o["amount"] = o["amount"].clip(upper=fence)
print("clean cities:", sorted(c["city"].unique()), "| rows left:", len(o))
print("Merch orders over $10:", len(o.query("category == 'Merch' and amount > 10")))

# ---------------------------------------------------------------
# Part 7 (W2D2): joins and groupby
# ---------------------------------------------------------------
title("WEEK 2 DAY 2: joins and groupby")
df = orders.merge(customers, on="customer_id", how="left", validate="many_to_one")
df["net_revenue"] = df["amount"].where(~df["returned"], 0.0)
print(f"gross ${df['amount'].sum():,.0f} | net of returns ${df['net_revenue'].sum():,.0f}")

print(df.groupby("city").agg(customers=("customer_id", "nunique"), revenue=("net_revenue", "sum"),
                             return_rate=("returned", "mean")).round(2).sort_values("revenue", ascending=False))
print(df.pivot_table(index="city", columns="category", values="net_revenue", aggfunc="sum", fill_value=0).round(0))
df["order_number"] = df.groupby("customer_id")["order_date"].rank(method="first").astype(int)
monthly = df.set_index("order_date").resample("MS")["net_revenue"].sum()
print("monthly revenue:\n", monthly.round(0).to_dict())

# ---------------------------------------------------------------
# Part 8 (W2D3): SQL with DuckDB + one chart
# ---------------------------------------------------------------
title("WEEK 2 DAY 3: SQL")
con = duckdb.connect()
con.register("customers", customers)
con.register("orders", orders)


def q(label, sql):
    print(f"\n--- {label} ---\n{con.sql(sql).df().to_string(index=False)}")


q("revenue by category (GROUP BY)", """
SELECT category, COUNT(*) AS orders, ROUND(SUM(amount), 0) AS revenue
FROM orders GROUP BY category ORDER BY revenue DESC""")
q("customers who never ordered (LEFT JOIN)", """
SELECT COUNT(*) AS never_ordered FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id WHERE o.order_id IS NULL""")
q("month-over-month growth (CTE + LAG)", """
WITH m AS (SELECT DATE_TRUNC('month', order_date) AS month, SUM(amount) AS revenue FROM orders GROUP BY 1)
SELECT month, ROUND(revenue, 0) AS revenue,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month)) / LAG(revenue) OVER (ORDER BY month), 1) AS mom_pct
FROM m ORDER BY month""")
q("top customer per city (ROW_NUMBER)", """
WITH s AS (SELECT c.city, o.customer_id, SUM(o.amount) AS total
           FROM orders o JOIN customers c USING (customer_id) GROUP BY 1, 2),
r AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY city ORDER BY total DESC) AS rn FROM s)
SELECT city, customer_id, ROUND(total, 2) AS total FROM r WHERE rn = 1 ORDER BY city""")

fig, ax = plt.subplots(figsize=(8, 4))
monthly.plot(ax=ax, marker="o")
ax.set(title="Monthly revenue (USD)")
fig.tight_layout()
fig.savefig("monthly_revenue.png", dpi=120)
print("\nsaved monthly_revenue.png")

# ---------------------------------------------------------------
# Part 9 (W2D4): stats and A/B testing
# ---------------------------------------------------------------
title("WEEK 2 DAY 4: stats and A/B testing")
spend = df.groupby(["customer_id", "is_member"])["amount"].sum().reset_index()
members, others = spend.loc[spend["is_member"], "amount"], spend.loc[~spend["is_member"], "amount"]
t, p = stats.ttest_ind(members, others, equal_var=False)
print(f"members ${members.mean():.2f} vs others ${others.mean():.2f} | Welch t-test p={p:.3f}")
chi2, p_chi, _, _ = stats.chi2_contingency(pd.crosstab(df["category"], df["returned"]))
print(f"return rate differs by category? chi-square p={p_chi:.3f}  (small sample: 'not significant' = not enough evidence)")


def sample_size_per_arm(p1, p2, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pb = (p1 + p2) / 2
    return int(np.ceil((za * np.sqrt(2 * pb * (1 - pb)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p2 - p1) ** 2))


n = sample_size_per_arm(0.04, 0.044)
print(f"users needed per arm to detect 4.0% -> 4.4%: {n:,}")

# simulated test: new "reorder" button in the cafe app
control, treatment = rng.binomial(1, 0.040, n), rng.binomial(1, 0.045, n)
pc, pt = control.mean(), treatment.mean()
pool = (control.sum() + treatment.sum()) / (2 * n)
z = (pt - pc) / np.sqrt(pool * (1 - pool) * 2 / n)
p_value = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"control {pc:.2%} vs treatment {pt:.2%} | z={z:.2f} p={p_value:.4f} -> {'SHIP' if p_value < 0.05 else 'do not ship yet'}")

# ---------------------------------------------------------------
# Part 10 (W2D4): Polars
# ---------------------------------------------------------------
title("WEEK 2 DAY 4: Polars")
try:
    import polars as pl

    op = (pl.read_csv("orders_raw.csv").unique(maintain_order=True)
          .with_columns(pl.col("amount").str.replace_all(r"[$,]", "").cast(pl.Float64),
                        (pl.col("returned") == "yes").alias("returned"))
          .drop_nulls("amount"))
    print(op.group_by("category")
          .agg(pl.len().alias("orders"), pl.col("amount").sum().round(0).alias("revenue"),
               pl.col("returned").mean().round(3).alias("return_rate"))
          .sort("revenue", descending=True))
except ImportError:
    print("polars not installed (pip install polars), skipping")

# ---------------------------------------------------------------
# Part 11 (W2D4): PySpark (skipped automatically if not available)
# ---------------------------------------------------------------
title("WEEK 2 DAY 4: PySpark")
try:
    import shutil
    if shutil.which("java") is None:
        raise RuntimeError("Java 17+ not found")
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = SparkSession.builder.master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    sp = (spark.read.csv("orders_raw.csv", header=True)
          .dropDuplicates()
          .withColumn("amount", F.regexp_replace("amount", r"[$,]", "").cast("double"))
          .dropna(subset=["amount"]))                         # lazy: nothing has run yet
    print("clean orders:", sp.count())                        # action: runs the plan
    sp.groupBy("category").agg(F.round(F.sum("amount"), 0).alias("revenue")).orderBy(F.desc("revenue")).show()
    spark.stop()
except (ImportError, RuntimeError) as exc:
    print("Skipping PySpark:", exc)

title("DONE")
