from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    customers = pd.read_csv("customers_raw.csv", parse_dates=["signup_date"])
    orders = pd.read_csv("orders_raw.csv", parse_dates=["order_date"])
    customers["state"] = customers["state"].str.strip().str.upper()
    customers["age"] = customers["age"].fillna(customers["age"].median())
    customers = customers.drop(columns=["email"])
    orders["amount"] = pd.to_numeric(orders["amount"].str.replace(r"[$,]", "", regex=True))
    orders["returned"] = orders["returned"].map({"yes": True, "no": False})
    orders = orders.drop_duplicates().dropna(subset=["amount"])
    
    # Keep raw amount before capping for specific queries
    orders["amount_raw"] = orders["amount"]
    
    upper = orders["amount"].quantile(0.75) + 3 * (orders["amount"].quantile(0.75) - orders["amount"].quantile(0.25))
    orders["amount"] = orders["amount"].clip(upper=upper)
    return customers, orders

customers, orders = load_clean_data()
con = duckdb.connect()
con.register("customers", customers)
con.register("orders", orders)

def q(title: str, sql: str) -> pd.DataFrame:
    result = con.sql(sql).df()
    print(f"\n=== {title} ===\n{result.to_string(index=False)}")
    return result


q("1. Five biggest Electronics orders", """
SELECT order_id, customer_id, amount_raw
FROM orders
WHERE category = 'Electronics' AND NOT returned
ORDER BY amount_raw DESC
LIMIT 5
""")

q("2. Revenue and orders by category", """
SELECT category,
       COUNT(*)              AS orders,
       ROUND(SUM(amount), 0) AS revenue,
       ROUND(AVG(amount), 2) AS avg_order
FROM orders
GROUP BY category
ORDER BY revenue DESC
""")


q("3. States with more than 40 customers", """
SELECT state, COUNT(*) AS customers
FROM customers
GROUP BY state
HAVING COUNT(*) > 40
ORDER BY customers DESC
""")


q("4. Order size bands (CASE WHEN)", """
SELECT CASE WHEN amount < 50  THEN 'small'
            WHEN amount < 200 THEN 'medium'
            ELSE 'large' END AS size,
       COUNT(*) AS orders
FROM orders
GROUP BY size
ORDER BY orders DESC
""")


q("5. Revenue per customer by state (INNER JOIN)", """
SELECT c.state,
       COUNT(DISTINCT c.customer_id)                           AS customers,
       ROUND(SUM(o.amount) / COUNT(DISTINCT c.customer_id), 2) AS revenue_per_customer
FROM customers AS c
JOIN orders AS o ON o.customer_id = c.customer_id
GROUP BY c.state
ORDER BY revenue_per_customer DESC
""")


q("6. Customers who never ordered (LEFT JOIN + IS NULL)", """
SELECT COUNT(*) AS never_ordered
FROM customers AS c
LEFT JOIN orders AS o ON o.customer_id = c.customer_id
WHERE o.order_id IS NULL
""")


q("7. Month-over-month revenue growth (CTE + LAG)", """
WITH monthly AS (
    SELECT DATE_TRUNC('month', order_date) AS month,
           SUM(amount)                     AS revenue
    FROM orders
    WHERE NOT returned
    GROUP BY 1
)
SELECT month,
       ROUND(revenue, 0) AS revenue,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
             / LAG(revenue) OVER (ORDER BY month), 1) AS mom_pct
FROM monthly
ORDER BY month
LIMIT 6
""")


q("8. Top 2 customers per state (ROW_NUMBER)", """
WITH spend AS (
    SELECT c.state, o.customer_id, SUM(o.amount) AS total
    FROM orders AS o
    JOIN customers AS c USING (customer_id)
    GROUP BY c.state, o.customer_id
),
ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY state ORDER BY total DESC) AS rn
    FROM spend
)
SELECT state, customer_id, ROUND(total, 2) AS total
FROM ranked
WHERE rn <= 2
ORDER BY state, total DESC
""")


q("9. Running total of revenue in 2024 (window SUM)", """
SELECT DATE_TRUNC('month', order_date) AS month,
       ROUND(SUM(SUM(amount)) OVER (ORDER BY DATE_TRUNC('month', order_date)), 0) AS running_total
FROM orders
WHERE order_date < '2024-07-01'
GROUP BY 1
ORDER BY 1
""")

q("10. Second-highest order value (classic interview)", """
SELECT DISTINCT amount
FROM orders
ORDER BY amount DESC
LIMIT 1 OFFSET 1
""")


q("11. Days between first and second order", """
WITH ranked AS (
    SELECT customer_id, order_date,
           ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) AS rn
    FROM orders
)
SELECT ROUND(AVG(DATE_DIFF('day', f.order_date, s.order_date)), 1) AS avg_days_to_second_order,
       COUNT(*) AS customers_with_2_orders
FROM ranked AS f
JOIN ranked AS s ON s.customer_id = f.customer_id AND f.rn = 1 AND s.rn = 2
""")


q("12. Bought Electronics but never Apparel (EXCEPT)", """
SELECT COUNT(*) AS customers FROM (
    SELECT customer_id FROM orders WHERE category = 'Electronics'
    EXCEPT
    SELECT customer_id FROM orders WHERE category = 'Apparel'
)
""")

q("13. Median vs average order value (PERCENTILE_CONT)", """
SELECT category,
       ROUND(AVG(amount_raw), 2)                                         AS mean_order,
       ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount_raw), 2) AS median_order
FROM orders
GROUP BY category
ORDER BY mean_order DESC
""")


q("14. Return rate: members vs non-members side by side (conditional aggregation)", """
SELECT o.category,
       ROUND(AVG(CASE WHEN c.is_member     THEN o.returned::INT END), 3) AS member_return_rate,
       ROUND(AVG(CASE WHEN NOT c.is_member THEN o.returned::INT END), 3) AS nonmember_return_rate
FROM orders AS o
JOIN customers AS c USING (customer_id)
GROUP BY o.category
ORDER BY o.category
""")

q("15. Top 3 distinct order values per category (DENSE_RANK + QUALIFY)", """
SELECT category, amount, DENSE_RANK() OVER (PARTITION BY category ORDER BY amount DESC) AS rnk
FROM orders
WHERE category IN ('Beauty', 'Home')
QUALIFY rnk <= 3
ORDER BY category, rnk
""")


logins = pd.DataFrame({
    "user_id": [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
    "login_date": pd.to_datetime([
        "2025-03-01", "2025-03-02", "2025-03-03", "2025-03-05",
        "2025-03-01", "2025-03-03", "2025-03-04",
        "2025-03-10", "2025-03-11", "2025-03-12", "2025-03-13", "2025-03-15",
    ]),
})
con.register("logins", logins)
q("16. Longest streak of consecutive login days (gaps and islands)", """
WITH numbered AS (
    SELECT user_id,
           CAST(login_date AS DATE) AS d,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS rn
    FROM logins
),
islands AS (
    SELECT user_id, d - CAST(rn AS INTEGER) AS grp
    FROM numbered
)
SELECT user_id, MAX(streak) AS longest_streak
FROM (SELECT user_id, grp, COUNT(*) AS streak FROM islands GROUP BY user_id, grp)
GROUP BY user_id
ORDER BY user_id
""")


charts = Path("charts")
charts.mkdir(exist_ok=True)
df = orders.merge(customers, on="customer_id", how="left")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(df["amount"], bins=40, color="#3b6fb6")
axes[0].set(title="Order value (USD)", xlabel="USD", ylabel="orders")
axes[1].hist(np.log1p(df["amount"]), bins=40, color="#2a9d8f")
axes[1].set(title="log(1 + order value)", xlabel="log USD")
fig.tight_layout()
fig.savefig(charts / "1_distribution.png", dpi=120)
print(f"skew raw: {df['amount'].skew():.2f} | skew after log: {np.log1p(df['amount']).skew():.2f}")



monthly = df.set_index("order_date").resample("MS")["amount"].sum()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(monthly.index, monthly.values, marker="o", color="#3b6fb6")
ax.set(title="Monthly revenue (USD)", ylabel="USD")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(charts / "2_monthly_revenue.png", dpi=120)

rr = df.groupby("category")["returned"].mean().sort_values() * 100
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(rr.index, rr.values, color=["#e76f51" if v == rr.max() else "#8d99ae" for v in rr.values])
ax.set(title="Return rate by category (%)", xlabel="%")
fig.tight_layout()
fig.savefig(charts / "3_return_rate.png", dpi=120)

spend = df.groupby(["customer_id", "is_member"])["amount"].sum().reset_index()
fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot([spend.loc[~spend["is_member"], "amount"], spend.loc[spend["is_member"], "amount"]],
           tick_labels=["Non-member", "Member"], showfliers=False)
ax.set(title="Total spend per customer (USD)", ylabel="USD")
fig.tight_layout()
fig.savefig(charts / "4_member_spend.png", dpi=120)


feat = df.groupby("customer_id").agg(age=("age", "first"), orders=("order_id", "count"),
                                     spend=("amount", "sum"), avg_order=("amount", "mean")).astype(float)
corr = feat.corr().round(2)
print("correlation with spend:\n", corr["spend"].sort_values(ascending=False))
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45)
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, corr.iloc[i, j], ha="center", va="center", fontsize=8)
fig.colorbar(im)
fig.tight_layout()
fig.savefig(charts / "5_correlation.png", dpi=120)
plt.close("all")



ex1 = con.sql("""
SELECT category, ROUND(AVG(returned::INT), 3) AS return_rate
FROM orders GROUP BY category ORDER BY return_rate DESC LIMIT 1
""").df()

ex2 = con.sql("""
SELECT c.is_member, ROUND(AVG(o.amount), 2) AS avg_order
FROM orders AS o JOIN customers AS c USING (customer_id)
GROUP BY c.is_member ORDER BY c.is_member
""").df()

ex3 = con.sql("""
WITH ranked AS (
    SELECT category, order_id, amount_raw,
           RANK() OVER (PARTITION BY category ORDER BY amount_raw DESC) AS rnk
    FROM orders
)
SELECT category, order_id, amount_raw FROM ranked WHERE rnk = 1 ORDER BY category
""").df()



