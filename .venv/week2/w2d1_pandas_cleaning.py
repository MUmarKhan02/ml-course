"""Week 2, Day 1: w2d1_pandas_cleaning.py
pandas basics & data cleaning on messy US e-commerce data.

First run:  python make_raw_data.py   (creates customers_raw.csv and orders_raw.csv)
Then run:   python w2d1_pandas_cleaning.py
"""

import hashlib

import numpy as np
import pandas as pd

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 10)

# ---------------------------------------------------------------------------
# Part 1: Load and take a first look
# ---------------------------------------------------------------------------
customers = pd.read_csv("customers_raw.csv", parse_dates=["signup_date"])
orders = pd.read_csv("orders_raw.csv")
print("shapes:", customers.shape, orders.shape)
print(orders.head(3))
print(orders.dtypes)
customers.info()

# ---------------------------------------------------------------------------
# Part 2: Find the problems
# ---------------------------------------------------------------------------
print("\nmissing values per column:\n", orders.isna().sum())
print("missing ages:", customers["age"].isna().sum())
print("duplicate order rows:", orders.duplicated().sum())
print("state values:", sorted(customers["state"].unique()))
print("returned values:", orders["returned"].value_counts().to_dict())

# ---------------------------------------------------------------------------
# Part 3: Fix data types
# ---------------------------------------------------------------------------
orders["amount"] = pd.to_numeric(orders["amount"].str.replace(r"[$,]", "", regex=True))
orders["order_date"] = pd.to_datetime(orders["order_date"])
orders["returned"] = orders["returned"].map({"yes": True, "no": False})
print("\nfixed dtypes:\n", orders.dtypes)

# ---------------------------------------------------------------------------
# Part 4: Clean text, duplicates and missing values
# ---------------------------------------------------------------------------
customers["state"] = customers["state"].str.strip().str.upper()
print("clean states:", sorted(customers["state"].unique()))

before = len(orders)
orders = orders.drop_duplicates()
print(f"removed {before - len(orders)} duplicate rows")

customers["age_missing"] = customers["age"].isna()
customers["age"] = customers["age"].fillna(customers["age"].median())
orders = orders.dropna(subset=["amount"])
print("rows left:", len(orders), "| missing amounts left:", orders["amount"].isna().sum())

# ---------------------------------------------------------------------------
# Part 5: Outliers (IQR rule and z-scores)
# ---------------------------------------------------------------------------
print("\namount summary:\n", orders["amount"].describe().round(2))
q1, q3 = orders["amount"].quantile([0.25, 0.75])
iqr = q3 - q1
upper = q3 + 3 * iqr
print(f"IQR upper fence: ${upper:,.2f} | outliers: {(orders['amount'] > upper).sum()}")
z = (orders["amount"] - orders["amount"].mean()) / orders["amount"].std()
print("rows with |z| > 4:", (z.abs() > 4).sum())
orders["amount_capped"] = orders["amount"].clip(upper=upper)
print(f"max before: ${orders['amount'].max():,.2f} | after capping: ${orders['amount_capped'].max():,.2f}")

# ---------------------------------------------------------------------------
# Part 6: Selecting and filtering
# ---------------------------------------------------------------------------
print("\ncolumns:", orders[["order_id", "category", "amount"]].head(2).to_dict("records"))
print("iloc first 2 rows, first 3 cols:\n", orders.iloc[:2, :3])
big_electronics = orders.loc[(orders["category"] == "Electronics") & (orders["amount"] > 500), ["order_id", "amount"]]
print("Electronics over $500:", len(big_electronics))
print("query() version:", len(orders.query("category == 'Electronics' and amount > 500")))
ca_members = customers.loc[(customers["state"] == "CA") & customers["is_member"], "customer_id"]
print("California members:", len(ca_members))
print("top states:\n", customers["state"].value_counts().head(3))

# ---------------------------------------------------------------------------
# Part 7: New columns and sorting
# ---------------------------------------------------------------------------
orders["order_month"] = orders["order_date"].dt.to_period("M")
orders["is_weekend"] = orders["order_date"].dt.dayofweek >= 5
orders["size"] = pd.cut(orders["amount_capped"], bins=[0, 50, 200, np.inf], labels=["small", "medium", "large"])
print("\norder sizes:", orders["size"].value_counts().to_dict())
print(orders.sort_values("amount", ascending=False)[["order_id", "category", "amount"]].head(3))

# ---------------------------------------------------------------------------
# Part 8: Protect PII, then save the clean data
# ---------------------------------------------------------------------------
customers["email_hash"] = customers["email"].map(lambda e: hashlib.sha256(("course-salt:" + e).encode()).hexdigest()[:12])
customers = customers.drop(columns=["email"])
print("\nhashed:", customers["email_hash"].head(2).tolist())

try:
    orders.to_parquet("orders_clean.parquet", index=False)
    customers.to_parquet("customers_clean.parquet", index=False)
    print("saved Parquet files")
except ImportError:
    orders.to_csv("orders_clean.csv", index=False)
    customers.to_csv("customers_clean.csv", index=False)
    print("pyarrow not installed: saved CSV files instead")

# ---------------------------------------------------------------------------
# Part 9: Hands-on exercise (solutions)
# ---------------------------------------------------------------------------
member_share = customers["is_member"].mean()
assert 0 < member_share < 1

return_rate = orders.groupby("category")["returned"].mean().round(3)
assert set(return_rate.index) == {"Electronics", "Home", "Apparel", "Beauty"}


def clean_amount(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")


assert clean_amount(pd.Series(["$1,204.50", "$3.00", "N/A"])).tolist()[:2] == [1204.5, 3.0]
print(f"\nmember share: {member_share:.1%}")
print("return rate by category:", return_rate.to_dict())
print("All exercises passed!")
