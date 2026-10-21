"""Week 2, Day 5 (part 1): w2d5_metrics_pandas.py
Product metrics, the "revenue dropped" investigation, and pandas interview topics.

Needs customers_raw.csv and orders_raw.csv in the same folder (python make_raw_data.py).

Run: python w2d5_metrics_pandas.py
"""

import time
import warnings

import numpy as np
import pandas as pd

pd.set_option("display.width", 120)


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
orders["net_revenue"] = orders["amount"].where(~orders["returned"], 0.0)
as_of = orders["order_date"].max()
print("data runs until:", as_of.date())

# ---------------------------------------------------------------------------
# Part 1: Engagement: monthly active customers and stickiness
# ---------------------------------------------------------------------------
orders["month"] = orders["order_date"].dt.to_period("M")
mau = orders.groupby("month")["customer_id"].nunique()
dau = orders.groupby("order_date")["customer_id"].nunique()
avg_dau = dau.groupby(dau.index.to_period("M")).mean()
stickiness = (avg_dau / mau).round(3)
print("\nmonthly active customers (last 4 full months):", mau.loc["2025-05":"2025-08"].to_dict())
print("stickiness DAU/MAU (Dec 2024):", stickiness.loc["2024-12"])

# ---------------------------------------------------------------------------
# Part 2: Conversion, AOV, frequency, repeat rate
# ---------------------------------------------------------------------------
first_order = orders.groupby("customer_id")["order_date"].min().rename("first_order")
funnel = customers.join(first_order, on="customer_id")
converted_30d = ((funnel["first_order"] - funnel["signup_date"]).dt.days <= 30).mean()
aov = orders["amount"].mean()
orders_per_customer = orders.groupby("customer_id").size().mean()
repeat_rate = (orders.groupby("customer_id").size() > 1).mean()
print(f"\nsign-up -> first order within 30 days: {converted_30d:.1%}")
print(f"AOV ${aov:.2f} | orders per buying customer {orders_per_customer:.2f} | repeat rate {repeat_rate:.1%}")

# ---------------------------------------------------------------------------
# Part 3: Churn, LTV and CAC
# ---------------------------------------------------------------------------
last_order = orders.groupby("customer_id")["order_date"].max()
churned = (as_of - last_order).dt.days > 90
print(f"churned (no order in last 90 days): {churned.mean():.1%} of buying customers")

gross_margin = 0.40
revenue_per_customer = orders.groupby("customer_id")["net_revenue"].sum().mean()
ltv = revenue_per_customer * gross_margin
marketing_spend = 9_000
new_customers = funnel["first_order"].notna().sum()
cac = marketing_spend / new_customers
print(f"LTV (margin) ${ltv:.2f} | CAC ${cac:.2f} | LTV:CAC = {ltv / cac:.1f}")

# ---------------------------------------------------------------------------
# Part 4: Case study: "Revenue dropped last month. Why?"
# ---------------------------------------------------------------------------
monthly = orders.groupby("month").agg(revenue=("net_revenue", "sum"), orders=("order_id", "count"),
                                      customers=("customer_id", "nunique"))
monthly["aov"] = monthly["revenue"] / monthly["orders"]
dec, jan = monthly.loc["2024-12"], monthly.loc["2025-01"]
print("\n", monthly.loc["2024-11":"2025-02"].round(2))
change = jan["revenue"] - dec["revenue"]
volume_effect = (jan["orders"] - dec["orders"]) * dec["aov"]
price_effect = (jan["aov"] - dec["aov"]) * jan["orders"]
print(f"revenue change Dec->Jan: ${change:,.0f} = orders effect ${volume_effect:,.0f} + order-value effect ${price_effect:,.0f}")

by_cat = orders[orders["month"].isin([pd.Period("2024-12"), pd.Period("2025-01")])].pivot_table(
    index="category", columns="month", values="net_revenue", aggfunc="sum", fill_value=0)
by_cat["change"] = by_cat.iloc[:, 1] - by_cat.iloc[:, 0]
print(by_cat.sort_values("change").round(0))

# ---------------------------------------------------------------------------
# Part 5: pandas interview topics
# ---------------------------------------------------------------------------
dec_orders = orders[orders["month"] == pd.Period("2024-12")]
jan_orders = orders[orders["month"] == pd.Period("2025-01")]
stacked = pd.concat([dec_orders, jan_orders], ignore_index=True)
print(f"\nconcat stacks rows: {len(dec_orders)} + {len(jan_orders)} = {len(stacked)}")

side = pd.concat([mau.rename("mau"), monthly["revenue"]], axis=1).tail(2)
print("concat axis=1 aligns on the index:\n", side.round(0))

cust_idx = customers.set_index("customer_id")
joined = dec_orders.join(cust_idx[["state"]], on="customer_id")
merged = dec_orders.merge(customers[["customer_id", "state"]], on="customer_id", how="left")
print("join (on index) == merge (on column):", joined["state"].tolist() == merged["state"].tolist())

big = pd.DataFrame({"price": np.random.default_rng(0).random(200_000) * 100, "qty": np.arange(200_000) % 9 + 1})
t0 = time.perf_counter()
big["total_apply"] = big.apply(lambda r: r["price"] * r["qty"], axis=1)
t1 = time.perf_counter()
big["total_vec"] = big["price"] * big["qty"]
t2 = time.perf_counter()
print(f"row-wise apply {1000 * (t1 - t0):.0f} ms | vectorized {1000 * (t2 - t1):.1f} ms | same: {np.allclose(big['total_apply'], big['total_vec'])}")

demo = pd.DataFrame({"state": ["CA", "TX", "CA"], "discount": [0, 0, 0]})
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    demo[demo["state"] == "CA"]["discount"] = 10
print("chained assignment changed nothing:", demo["discount"].tolist(), "| warning:", caught[0].category.__name__ if caught else "none")
demo.loc[demo["state"] == "CA", "discount"] = 10
print("with .loc:", demo["discount"].tolist())

# ---------------------------------------------------------------------------
# Part 6: Hands-on exercise (solutions)
# ---------------------------------------------------------------------------
member_ids = customers.loc[customers["is_member"], "customer_id"]
member_aov = orders.loc[orders["customer_id"].isin(member_ids), "amount"].mean()
assert member_aov > 0

monthly_new = funnel.dropna(subset=["first_order"]).groupby(funnel["first_order"].dt.to_period("M")).size()
assert monthly_new.sum() == new_customers

cat_change = by_cat["change"].idxmin()
assert cat_change in {"Electronics", "Home", "Apparel", "Beauty"}
print(f"\nmember AOV ${member_aov:.2f} | biggest Dec->Jan drop: {cat_change}")
print("All exercises passed!")
