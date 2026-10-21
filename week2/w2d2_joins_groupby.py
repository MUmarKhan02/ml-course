import pandas as pd

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 12)

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
print("customers:", customers.shape, "| orders:", orders.shape)


df = orders.merge(customers, on="customer_id", how="left", validate="many_to_one")
print("merged:", df.shape)
print(df[["order_id", "customer_id", "state", "is_member", "amount"]].head(3))

no_orders = customers.merge(orders[["customer_id"]].drop_duplicates(), on="customer_id", how="left", indicator=True)
print("join check:", no_orders["_merge"].value_counts().to_dict())
print("customers who never ordered:", (no_orders["_merge"] == "left_only").sum())


df["net_revenue"] = df["amount"].where(~df["returned"], 0.0)
print(f"gross ${df['amount'].sum():,.0f} | net ${df['net_revenue'].sum():,.0f}")


by_state = (
    df.groupby("state")
    .agg(
        customers=("customer_id", "nunique"),
        orders=("order_id", "count"),
        revenue=("net_revenue", "sum"),
        avg_order=("amount", "mean"),
        return_rate=("returned", "mean"),
    )
    .assign(revenue_per_customer=lambda d: d["revenue"] / d["customers"])
    .sort_values("revenue", ascending=False)
    .round(2)
)
print("revenue by state:\n", by_state)


by_cat_member = df.groupby(["category", "is_member"])["net_revenue"].mean().round(2).unstack()
print("average order value by category, members vs non-members:\n", by_cat_member)


df["customer_total"] = df.groupby("customer_id")["net_revenue"].transform("sum")
df["order_number"] = df.groupby("customer_id")["order_date"].rank(method="first").astype(int)
df["share_of_customer"] = (df["net_revenue"] / df["customer_total"]).fillna(0).round(3)
print(df.sort_values(["customer_id", "order_date"])[["customer_id", "order_date", "net_revenue", "customer_total", "order_number"]].head(4))



top5 = df.groupby("customer_id")["net_revenue"].sum().nlargest(5)
print("top 5 customers:\n", top5.round(2))


pivot = df.pivot_table(index="state", columns="category", values="net_revenue", aggfunc="sum",
                       fill_value=0, margins=True, margins_name="Total").round(0)
print("revenue: state x category\n", pivot)


monthly = df.set_index("order_date").resample("MS")["net_revenue"].sum().to_frame("revenue")
monthly["mom_growth_%"] = (monthly["revenue"].pct_change() * 100).round(1)
monthly["rolling_3m"] = monthly["revenue"].rolling(3).mean().round(0)
print("monthly revenue (last 6 months):\n", monthly.round(0).tail(6))
best = monthly["revenue"].idxmax()
print("best month:", best.strftime("%B %Y"), f"${monthly['revenue'].max():,.0f}")


df["weekday"] = df["order_date"].dt.day_name()
print("orders by weekday:", df["weekday"].value_counts().to_dict())



df["order_month"] = df["order_date"].dt.to_period("M")
df["cohort"] = df.groupby("customer_id")["order_date"].transform("min").dt.to_period("M")
df["months_since_first"] = (df["order_month"] - df["cohort"]).apply(lambda p: p.n)
cohort_counts = df.pivot_table(index="cohort", columns="months_since_first", values="customer_id", aggfunc="nunique")
retention = cohort_counts.div(cohort_counts[0], axis=0).round(2)
print("cohort sizes:", cohort_counts[0].head(4).to_dict())
print("retention (first 4 cohorts, months 0-4):\n", retention.iloc[:4, :5])



cat_share = (df.groupby("category")["net_revenue"].sum() / df["net_revenue"].sum()).round(3)

repeat_rate = (df.groupby("customer_id")["order_id"].nunique() > 1).mean()

state_month = df.pivot_table(index="order_month", columns="state", values="order_id",
                             aggfunc="count", fill_value=0)
