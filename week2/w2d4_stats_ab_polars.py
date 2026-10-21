"""Week 2, Day 4 (part 1): w2d4_stats_ab_polars.py
Hypothesis tests, A/B testing end-to-end, and Polars.

Needs customers_raw.csv and orders_raw.csv in the same folder (python make_raw_data.py).
Install once: uv pip install scipy polars

Run: python w2d4_stats_ab_polars.py
"""

import time

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

rng = np.random.default_rng(2026)


# ---------------------------------------------------------------------------
# Part 1: Hypothesis tests on our store data (are members really different?)
# ---------------------------------------------------------------------------
def load_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    customers = pd.read_csv("customers_raw.csv", parse_dates=["signup_date"])
    orders = pd.read_csv("orders_raw.csv", parse_dates=["order_date"])
    customers["state"] = customers["state"].str.strip().str.upper()
    customers["age"] = customers["age"].fillna(customers["age"].median())
    customers = customers.drop(columns=["email"])
    orders["amount"] = pd.to_numeric(orders["amount"].str.replace(r"[$,]", "", regex=True))
    orders["returned"] = orders["returned"].map({"yes": True, "no": False})
    orders = orders.drop_duplicates().dropna(subset=["amount"])
    upper = orders["amount"].quantile(0.75) + 3 * (orders["amount"].quantile(0.75) - orders["amount"].quantile(0.25))
    orders["amount"] = orders["amount"].clip(upper=upper)
    return customers, orders


customers, orders = load_clean_data()
df = orders.merge(customers, on="customer_id", how="left")
spend = df.groupby(["customer_id", "is_member"])["amount"].sum().reset_index()
members = spend.loc[spend["is_member"], "amount"]
others = spend.loc[~spend["is_member"], "amount"]
print(f"members: n={len(members)} mean=${members.mean():.2f} | non-members: n={len(others)} mean=${others.mean():.2f}")

t_stat, p_t = stats.ttest_ind(members, others, equal_var=False)
u_stat, p_u = stats.mannwhitneyu(members, others)
print(f"Welch t-test: t={t_stat:.2f} p={p_t:.3f}")
print(f"Mann-Whitney U: p={p_u:.3f}")

table = pd.crosstab(df["category"], df["returned"])
chi2, p_chi, dof, _ = stats.chi2_contingency(table)
print(f"Chi-square (return rate differs by category?): chi2={chi2:.2f} dof={dof} p={p_chi:.3f}")

# ---------------------------------------------------------------------------
# Part 2: A/B test design: how many users do we need?
# ---------------------------------------------------------------------------
baseline, mde, alpha, power = 0.040, 0.004, 0.05, 0.80


def sample_size_per_arm(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / (p2 - p1) ** 2))


n = sample_size_per_arm(baseline, baseline + mde, alpha, power)
print(f"\nneed {n:,} users per arm to detect {baseline:.1%} -> {baseline + mde:.1%}")
print(f"at 20,000 visitors/day split 50/50: {2 * n / 20_000:.1f} days -> run 7 full days")

# ---------------------------------------------------------------------------
# Part 3: Run and analyze the test (simulated: true conversion 4.0% vs 4.5%)
# ---------------------------------------------------------------------------
n_c, n_t = n, n + 211
conv_c = rng.binomial(1, 0.040, n_c)
conv_t = rng.binomial(1, 0.045, n_t)

srm = stats.chisquare([n_c, n_t])
print(f"\nSRM check: control={n_c:,} treatment={n_t:,} p={srm.pvalue:.3f} -> {'OK' if srm.pvalue > 0.001 else 'STOP: broken split'}")

p_c, p_t2 = conv_c.mean(), conv_t.mean()
p_pool = (conv_c.sum() + conv_t.sum()) / (n_c + n_t)
z = (p_t2 - p_c) / np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
p_value = 2 * (1 - stats.norm.cdf(abs(z)))
se = np.sqrt(p_c * (1 - p_c) / n_c + p_t2 * (1 - p_t2) / n_t)
ci = (p_t2 - p_c - 1.96 * se, p_t2 - p_c + 1.96 * se)
print(f"control {p_c:.3%} vs treatment {p_t2:.3%} | lift {p_t2 - p_c:+.3%} ({(p_t2 - p_c) / p_c:+.1%} relative)")
print(f"z={z:.2f} p={p_value:.4f} | 95% CI [{ci[0]:+.3%}, {ci[1]:+.3%}]")
print("decision:", "SHIP" if p_value < alpha and ci[0] > 0 else "do not ship yet")


# ---------------------------------------------------------------------------
# Part 4: Pitfalls: peeking, multiple comparisons, novelty effect
# ---------------------------------------------------------------------------
def false_positive_rate(peek: bool, sims: int = 1000, days: int = 14, per_day: int = 1500) -> float:
    hits = 0
    for _ in range(sims):
        a = rng.binomial(1, 0.04, (days, per_day))
        b = rng.binomial(1, 0.04, (days, per_day))
        for d in (range(1, days + 1) if peek else [days]):
            x1, x2, m = a[:d].sum(), b[:d].sum(), d * per_day
            pp = (x1 + x2) / (2 * m)
            if abs((x2 - x1) / m) / np.sqrt(pp * (1 - pp) * 2 / m) > 1.96:
                hits += 1
                break
    return hits / sims


print(f"\nA/A test, look once: {false_positive_rate(False):.1%} false positives")
print(f"A/A test, peek daily and stop at p<0.05: {false_positive_rate(True):.1%} false positives")

rng_m = np.random.default_rng(6)
pvals = np.array([stats.ttest_ind(rng_m.normal(0, 1, 500), rng_m.normal(0, 1, 500)).pvalue for _ in range(20)])
print(f"20 metrics, no real effect: {np.sum(pvals < 0.05)} 'significant' | after Bonferroni: {np.sum(pvals < 0.05 / 20)}")

daily_lift = 0.010 * np.exp(-np.arange(14) / 3) + 0.002
print("novelty effect, daily lift (points):", np.round(daily_lift * 100, 2)[[0, 3, 7, 13]].tolist())

# ---------------------------------------------------------------------------
# Part 5: CUPED variance reduction
# ---------------------------------------------------------------------------
m = 5000
pre = rng.gamma(2, 50, 2 * m)
group = np.repeat([0, 1], m)
post = 0.8 * pre + rng.normal(0, 40, 2 * m) + 3 * group
theta = np.cov(post, pre)[0, 1] / np.var(pre, ddof=1)
post_cuped = post - theta * (pre - pre.mean())
for label, y in [("raw", post), ("CUPED", post_cuped)]:
    res = stats.ttest_ind(y[group == 1], y[group == 0])
    print(f"{label:<5}: effect ${y[group == 1].mean() - y[group == 0].mean():.2f} | std {y.std():.1f} | p={res.pvalue:.4f}")

# ---------------------------------------------------------------------------
# Part 6: Polars: same cleaning and analysis, new engine
# ---------------------------------------------------------------------------
orders_pl = (
    pl.read_csv("orders_raw.csv")
    .unique(maintain_order=True)
    .with_columns(
        pl.col("amount").str.replace_all(r"[$,]", "").cast(pl.Float64),
        pl.col("order_date").str.to_date(),
        (pl.col("returned") == "yes").alias("returned"),
    )
    .drop_nulls("amount")
)
print("\npolars orders:", orders_pl.shape)
print(orders_pl.head(3))

by_cat = (
    orders_pl.group_by("category")
    .agg(
        pl.len().alias("orders"),
        pl.col("amount").sum().round(0).alias("revenue"),
        pl.col("returned").mean().round(3).alias("return_rate"),
    )
    .sort("revenue", descending=True)
)
print(by_cat)

customers_pl = pl.read_csv("customers_raw.csv").with_columns(pl.col("state").str.strip_chars().str.to_uppercase())
lazy = (
    orders_pl.lazy()
    .join(customers_pl.lazy(), on="customer_id")
    .filter(~pl.col("returned"))
    .group_by("state")
    .agg(pl.col("amount").sum().alias("revenue"), pl.col("customer_id").n_unique().alias("customers"))
    .with_columns((pl.col("revenue") / pl.col("customers")).round(2).alias("rev_per_customer"))
    .sort("rev_per_customer", descending=True)
)
print(lazy.collect().head(3))

ranked = orders_pl.with_columns(pl.col("order_date").rank("ordinal").over("customer_id").alias("order_number"))
print("second orders:", ranked.filter(pl.col("order_number") == 2).height)

n_big = 5_000_000
big = pd.DataFrame({"store": rng.integers(0, 500, n_big), "product": rng.integers(0, 20_000, n_big),
                    "price": rng.random(n_big) * 100, "qty": rng.integers(1, 10, n_big)})
big_pl = pl.DataFrame({c: big[c].to_numpy() for c in big.columns})
big_pl.group_by("store").agg(pl.len())  # warm-up: start Polars' thread pool once

t0 = time.perf_counter()
big.groupby("store")["price"].sum()
t1 = time.perf_counter()
big_pl.group_by("store").agg(pl.col("price").sum())
t2 = time.perf_counter()
print(f"5M rows, one simple sum:       pandas {1000 * (t1 - t0):.0f} ms | polars {1000 * (t2 - t1):.0f} ms")

t0 = time.perf_counter()
(big[big["qty"] > 3].assign(revenue=lambda d: d["price"] * d["qty"])
    .groupby("store").agg(revenue=("revenue", "sum"), avg=("revenue", "mean"), products=("product", "nunique")))
t1 = time.perf_counter()
(big_pl.lazy().filter(pl.col("qty") > 3).with_columns((pl.col("price") * pl.col("qty")).alias("revenue"))
    .group_by("store").agg(pl.col("revenue").sum(), pl.col("revenue").mean().alias("avg"), pl.col("product").n_unique())
    .collect())
t2 = time.perf_counter()
print(f"5M rows, filter+derive+3 aggs: pandas {1000 * (t1 - t0):.0f} ms | polars {1000 * (t2 - t1):.0f} ms")

# ---------------------------------------------------------------------------
# Part 7: Hands-on exercise (solutions)
# ---------------------------------------------------------------------------
n_small = sample_size_per_arm(0.10, 0.12)
assert n_small < n

ca_age = customers.loc[customers["state"] == "CA", "age"]
tx_age = customers.loc[customers["state"] == "TX", "age"]
p_age = stats.ttest_ind(ca_age, tx_age, equal_var=False).pvalue
assert 0 <= p_age <= 1

state_counts = customers_pl.group_by("state").agg(pl.len().alias("customers")).sort("customers", descending=True)
assert state_counts.height == 6
print(f"\nsample size for 10% -> 12%: {n_small:,} per arm")
print(f"CA vs TX age p-value: {p_age:.3f}")
print("All exercises passed!")
