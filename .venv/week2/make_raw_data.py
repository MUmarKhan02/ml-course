"""Create two messy CSV files for Week 2 (synthetic US e-commerce data, realistic data-quality problems).

    customers_raw.csv  - 300 customers: inconsistent state codes, missing ages, emails (PII)
    orders_raw.csv     - ~1,000 orders: amounts stored as text ("$1,204.50"), missing values,
                         duplicate rows, a few extreme outliers, dates as text

Run once: python make_raw_data.py
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(2024)

# ----------------------------- customers -----------------------------
n_cust = 300
states = rng.choice(["CA", "TX", "NY", "FL", "WA", "IL"], n_cust, p=[0.25, 0.2, 0.18, 0.15, 0.12, 0.10]).astype(object)
messy = rng.random(n_cust) < 0.08
states[messy] = [f" {s.lower()} " for s in states[messy]]          # " ca " instead of "CA"
age = rng.normal(38, 12, n_cust).clip(18, 80).round()
age[rng.random(n_cust) < 0.06] = np.nan                            # missing ages
customers = pd.DataFrame({
    "customer_id": np.arange(1001, 1001 + n_cust),
    "email": [f"user{i}@example.com" for i in range(n_cust)],
    "state": states,
    "age": age,
    "signup_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n_cust), unit="D"),
    "is_member": rng.random(n_cust) < 0.3,
})

# ----------------------------- orders -----------------------------
rows = []
oid = 50001
categories = {"Electronics": 220, "Home": 90, "Apparel": 55, "Beauty": 35}
for c in customers.itertuples():
    for _ in range(rng.poisson(3.3)):
        cat = rng.choice(list(categories))
        amount = max(5.0, rng.gamma(2.0, categories[cat] / 2))
        rows.append({
            "order_id": oid,
            "customer_id": c.customer_id,
            "order_date": (c.signup_date + pd.Timedelta(days=int(rng.integers(0, 270)))).strftime("%Y-%m-%d"),
            "category": cat,
            "amount": round(amount, 2),
            "returned": "yes" if rng.random() < 0.08 else "no",
        })
        oid += 1
orders = pd.DataFrame(rows)

orders.loc[rng.choice(len(orders), 4, replace=False), "amount"] *= 50          # extreme outliers
amount_text = orders["amount"].map(lambda v: f"${v:,.2f}").astype(object)      # money stored as text
amount_text[rng.random(len(orders)) < 0.04] = np.nan                           # missing amounts
orders["amount"] = amount_text
orders = pd.concat([orders, orders.sample(15, random_state=1)], ignore_index=True)  # duplicate rows

customers.to_csv("customers_raw.csv", index=False)
orders.to_csv("orders_raw.csv", index=False)
print(f"Wrote customers_raw.csv ({len(customers)} rows) and orders_raw.csv ({len(orders)} rows)")
