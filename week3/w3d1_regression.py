"""Week 3, Day 1 (part 1): w3d1_regression.py
Regression end-to-end with scikit-learn: Pipeline, ColumnTransformer, cross-validation, model comparison,
error analysis, regularization (L1 vs L2) and saving the model.

Install once: uv pip install scikit-learn
Run: python w3d1_regression.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pd.set_option("display.width", 120)


# ---------------------------------------------------------------------------
# Part 1: Data: predict US home sale prices
# ---------------------------------------------------------------------------
def make_housing(n: int = 4000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metro = rng.choice(["San Jose", "Seattle", "Austin", "Atlanta", "Phoenix", "Boston"], n)
    premium = pd.Series(metro).map({"San Jose": 3.0, "Seattle": 1.9, "Boston": 2.0, "Austin": 1.3,
                                    "Atlanta": 1.0, "Phoenix": 1.05}).to_numpy()
    sqft = rng.normal(1900, 600, n).clip(500, 6000)
    beds = np.clip(np.round(sqft / 600 + rng.normal(0, 0.7, n)), 1, 7)
    baths = np.clip(np.round(beds * 0.7 + rng.normal(0, 0.5, n)), 1, 6)
    year_built = rng.integers(1920, 2025, n)
    school = rng.integers(1, 11, n).astype(float)
    has_pool = rng.random(n) < np.where(np.isin(metro, ["Phoenix", "Austin"]), 0.35, 0.08)
    price = (premium * (150 * sqft + 15_000 * baths) + 9_000 * school * premium + 400 * (year_built - 1920)
             + 25_000 * has_pool + rng.normal(0, 40_000, n))
    df = pd.DataFrame(dict(metro=metro, sqft=sqft.round(), beds=beds, baths=baths, year_built=year_built,
                           school_rating=school, has_pool=has_pool, price=price.round(-2)))
    df.loc[rng.random(n) < 0.04, "school_rating"] = np.nan  # realistic missing values
    return df


df = make_housing()
print(df.head(3))
print("missing school ratings:", df["school_rating"].isna().sum(), "| price range:", f"${df['price'].min():,.0f} to ${df['price'].max():,.0f}")

X, y = df.drop(columns="price"), df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("train/test rows:", len(X_train), len(X_test))

# ---------------------------------------------------------------------------
# Part 2: Preprocessing with ColumnTransformer + Pipeline
# ---------------------------------------------------------------------------
numeric = ["sqft", "beds", "baths", "year_built", "school_rating"]
categorical = ["metro", "has_pool"]
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
])
demo = preprocess.fit(X_train).transform(X_train)
print("\nafter preprocessing:", demo.shape, "(5 numeric + 6 metros + 2 pool flags = 13 columns)")

# ---------------------------------------------------------------------------
# Part 3: Compare models with 5-fold cross-validation
# ---------------------------------------------------------------------------
models = {
    "Baseline (mean)": DummyRegressor(),
    "Linear Regression": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=0),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0),
}
cv = KFold(n_splits=5, shuffle=True, random_state=0)
rows = []
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    s = cross_validate(pipe, X_train, y_train, cv=cv, n_jobs=-1,
                       scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"})
    rows.append({"model": name, "RMSE": -s["test_rmse"].mean(), "RMSE std": s["test_rmse"].std(),
                 "MAE": -s["test_mae"].mean(), "R2": s["test_r2"].mean()})
results = pd.DataFrame(rows).sort_values("RMSE")
print("\n5-fold cross-validation on the training set:")
print(results.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# Part 4: Final model on the test set (touched once), error analysis, save
# ---------------------------------------------------------------------------
best_name = results.iloc[0]["model"]
best = Pipeline([("prep", preprocess), ("model", models[best_name])]).fit(X_train, y_train)
pred = best.predict(X_test)
print(f"\nbest model: {best_name}")
print(f"TEST: RMSE ${np.sqrt(mean_squared_error(y_test, pred)):,.0f} | MAE ${mean_absolute_error(y_test, pred):,.0f} | R2 {r2_score(y_test, pred):.3f}")

err = X_test.assign(price=y_test, pred=pred)
err["abs_pct_err"] = (err["pred"] - err["price"]).abs() / err["price"] * 100
print("\nmedian absolute % error by metro:")
print(err.groupby("metro")["abs_pct_err"].median().round(1).sort_values(ascending=False).to_string())
worst = err.sort_values("abs_pct_err", ascending=False).head(3)[["metro", "sqft", "price", "pred", "abs_pct_err"]]
print("\n3 worst predictions:\n", worst.round(0).to_string())

log_ridge = Pipeline([("prep", preprocess), ("model", TransformedTargetRegressor(Ridge(), func=np.log1p, inverse_func=np.expm1))])
s = cross_validate(log_ridge, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
print(f"\nRidge on log(price): CV RMSE ${-s['test_score'].mean():,.0f}")

joblib.dump(best, "house_price_model.joblib")
loaded = joblib.load("house_price_model.joblib")
one_house = pd.DataFrame([{"metro": "Seattle", "sqft": 2200, "beds": 4, "baths": 3, "year_built": 2005,
                           "school_rating": 8.0, "has_pool": False}])
print(f"saved and reloaded model predicts a 2,200 sqft Seattle house at ${loaded.predict(one_house)[0]:,.0f}")

# ---------------------------------------------------------------------------
# Part 5: Regularization: L1 (Lasso) vs L2 (Ridge) with noisy extra features
# ---------------------------------------------------------------------------
rng = np.random.default_rng(3)
n_reg = 300
signal = rng.normal(size=(n_reg, 3))
noise_features = rng.normal(size=(n_reg, 12))
X_reg = np.hstack([signal, noise_features])
y_reg = 5 * signal[:, 0] - 3 * signal[:, 1] + 2 * signal[:, 2] + rng.normal(0, 1, n_reg)
for label, m in [("OLS (no regularization)", LinearRegression()), ("Ridge (L2)", Ridge(alpha=30)), ("Lasso (L1)", Lasso(alpha=0.3))]:
    m.fit(X_reg, y_reg)
    print(f"{label:<24} weights on the 12 useless features: max |w| = {np.abs(m.coef_[3:]).max():.3f}"
          f" | exactly zero: {int(np.sum(m.coef_[3:] == 0))}/12")

# ---------------------------------------------------------------------------
# Part 6: Hands-on exercise (solutions)
# ---------------------------------------------------------------------------
print ("\n--- Hands-on exercise (solutions) ---")

ridge_scores = {}
for alpha in (0.01, 1, 100, 1000):
    p = Pipeline([("prep", preprocess), ("model", Ridge(alpha=alpha))])
    ridge_scores[alpha] = -cross_validate(p, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")["test_score"].mean()
best_alpha = min(ridge_scores, key=ridge_scores.get)
assert best_alpha in ridge_scores

metro_median = err.groupby("metro")["abs_pct_err"].median()
hardest_metro = metro_median.idxmax()
assert hardest_metro in set(df["metro"])

baseline_rmse = results.loc[results["model"] == "Baseline (mean)", "RMSE"].iloc[0]
best_rmse = results.iloc[0]["RMSE"]
improvement = 1 - best_rmse / baseline_rmse
assert improvement > 0.5
print(f"\nbest Ridge alpha: {best_alpha} | hardest metro to predict: {hardest_metro} | improvement over baseline: {improvement:.0%}")
print("All exercises passed!")
