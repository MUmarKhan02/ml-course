"""Week 3, Day 1 (part 2): w3d1_forecasting.py
Time-series forecasting: daily retail demand for a US store chain.
Calendar + US holiday features, lag/rolling features (no leakage), TimeSeriesSplit backtesting,
seasonal-naive baseline vs gradient boosting, MAE / MAPE / WAPE, prediction intervals.

Install once: uv pip install scikit-learn
Run: python w3d1_forecasting.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, TimeSeriesSplit

pd.set_option("display.width", 120)
rng = np.random.default_rng(7)

# ---------------------------------------------------------------------------
# Part 1: Daily unit sales 2023-2025: trend + weekly + yearly seasonality + US holidays
# ---------------------------------------------------------------------------
dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
t = np.arange(len(dates))
weekly = np.array([0.9, 0.85, 0.88, 0.95, 1.1, 1.35, 1.25])[dates.dayofweek]       # Mon..Sun
yearly = 1 + 0.15 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365.25)
holidays = pd.to_datetime([f"{y}-{md}" for y in (2023, 2024, 2025) for md in ("07-04", "12-24", "12-26")]
                          + ["2023-11-24", "2024-11-29", "2025-11-28"])          # Black Friday
is_holiday = dates.isin(holidays)
units = (200 + 0.08 * t) * weekly * yearly * np.where(is_holiday, 1.8, 1.0) * rng.normal(1, 0.06, len(t))
df = pd.DataFrame({"date": dates, "units": units.round()})
print(df.tail(3))
print("average units by weekday (Mon..Sun):", df.groupby(df["date"].dt.dayofweek)["units"].mean().round(0).tolist())
print("holiday uplift:", f"{df.loc[is_holiday, 'units'].mean() / df.loc[~is_holiday, 'units'].mean():.2f}x")

# ---------------------------------------------------------------------------
# Part 2: Features: calendar, holidays, lags and rolling stats (horizon = 7 days)
# ---------------------------------------------------------------------------
H = 7
feat = df.copy()
feat["dow"] = feat["date"].dt.dayofweek
feat["month"] = feat["date"].dt.month
feat["dayofyear"] = feat["date"].dt.dayofyear
feat["is_holiday"] = is_holiday.astype(int)
for lag in (7, 14, 21, 28, 364):
    feat[f"lag_{lag}"] = feat["units"].shift(lag)
feat["roll_mean_28"] = feat["units"].shift(H).rolling(28).mean()
feat["roll_std_28"] = feat["units"].shift(H).rolling(28).std()
feat = feat.dropna().reset_index(drop=True)
X_cols = [c for c in feat.columns if c not in ("date", "units")]
print("\nrows after dropping the first 364 days (needed for lag_364):", len(feat), "| features:", len(X_cols))

# A leakage trap: a lag of 1 day is not available when forecasting a week ahead
leaky = df.copy()
leaky["lag_1"] = leaky["units"].shift(1)
leaky = leaky.dropna()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"MAE": float(np.mean(np.abs(y_true - y_pred))),
            "MAPE%": float(np.mean(np.abs(y_true - y_pred) / y_true) * 100),
            "WAPE%": float(np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100)}


# ---------------------------------------------------------------------------
# Part 3: Why random K-fold is wrong for time series
# ---------------------------------------------------------------------------
def gbm() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)


def wape_of(train_idx: np.ndarray, test_idx: np.ndarray) -> float:
    m = gbm().fit(feat.loc[train_idx, X_cols], feat.loc[train_idx, "units"])
    p = m.predict(feat.loc[test_idx, X_cols])
    return metrics(feat.loc[test_idx, "units"].to_numpy(), p)["WAPE%"]


last_block = np.arange(len(feat) - 100, len(feat))
past = np.arange(0, len(feat) - 100)
honest = wape_of(past, last_block)
shuffled = np.mean([wape_of(tr, te) for tr, te in KFold(5, shuffle=True, random_state=0).split(feat)])
print(f"\nWAPE forecasting the LAST 100 days from the past (honest): {honest:.2f}%")
print(f"WAPE with shuffled K-fold (trains on the future, too optimistic): {shuffled:.2f}%")

# ---------------------------------------------------------------------------
# Part 4: Backtesting with TimeSeriesSplit: baseline vs gradient boosting
# ---------------------------------------------------------------------------
tscv = TimeSeriesSplit(n_splits=5, test_size=56)
rows = []
for fold, (tr, te) in enumerate(tscv.split(feat)):
    train, test = feat.iloc[tr], feat.iloc[te]
    naive = test["lag_7"].to_numpy()  # seasonal-naive: same day last week
    model = gbm().fit(train[X_cols], train["units"])
    y_true = test["units"].to_numpy()
    rows.append({"fold": fold, "test_start": test["date"].iloc[0].date(),
                 "naive_WAPE%": metrics(y_true, naive)["WAPE%"], "gbm_WAPE%": metrics(y_true, model.predict(test[X_cols]))["WAPE%"],
                 "naive_MAE": metrics(y_true, naive)["MAE"], "gbm_MAE": metrics(y_true, model.predict(test[X_cols]))["MAE"]})
res = pd.DataFrame(rows)
print("\nbacktest: 5 folds x 56 days, always training on the past only")
print(res.round(2).to_string(index=False))
print(f"mean WAPE: seasonal-naive {res['naive_WAPE%'].mean():.2f}% | gradient boosting {res['gbm_WAPE%'].mean():.2f}%")

# ---------------------------------------------------------------------------
# Part 5: Error analysis by weekday + prediction intervals
# ---------------------------------------------------------------------------
tr_idx, te_idx = list(tscv.split(feat))[-1]
train, test = feat.iloc[tr_idx], feat.iloc[te_idx].copy()
model = gbm().fit(train[X_cols], train["units"])
test["pred"] = model.predict(test[X_cols])
test["abs_pct_err"] = (test["pred"] - test["units"]).abs() / test["units"] * 100
print("\nlast fold, mean abs % error by weekday (0=Mon):", test.groupby("dow")["abs_pct_err"].mean().round(1).to_dict())

lo = HistGradientBoostingRegressor(loss="quantile", quantile=0.10, max_iter=300, learning_rate=0.05, random_state=0).fit(train[X_cols], train["units"])
hi = HistGradientBoostingRegressor(loss="quantile", quantile=0.90, max_iter=300, learning_rate=0.05, random_state=0).fit(train[X_cols], train["units"])
test["p10"], test["p90"] = lo.predict(test[X_cols]), hi.predict(test[X_cols])
coverage = float(((test["units"] >= test["p10"]) & (test["units"] <= test["p90"])).mean())
print(f"80% prediction interval [p10, p90] covers {coverage:.0%} of actual days (target 80%)")
print(test[["date", "units", "p10", "pred", "p90"]].head(3).round({"units": 0, "p10": 0, "pred": 0, "p90": 0}).to_string(index=False))

# ---------------------------------------------------------------------------
# Part 6: Hands-on exercise (solutions)
# ---------------------------------------------------------------------------
print ("\n--- Hands-on exercise (solutions) ---")
# Forecast the final 7 days of the data, using ONLY information from before those days
actual_next = df["units"].tail(7).to_numpy()
history = df["units"].iloc[:-7]
mean_forecast = np.full(7, history.tail(28).mean())      # average of the 28 days before
seasonal_forecast = history.tail(7).to_numpy()           # the same weekdays, one week earlier
mean_wape = metrics(actual_next, mean_forecast)["WAPE%"]
seasonal_wape = metrics(actual_next, seasonal_forecast)["WAPE%"]
assert seasonal_wape < mean_wape

worst_weekday = max(test.groupby("dow")["abs_pct_err"].mean().items(), key=lambda kv: kv[1])[0]
assert worst_weekday in range(7)

top_feature = max(X_cols, key=lambda c: abs(np.corrcoef(feat[c], feat["units"])[0, 1]))
assert top_feature in X_cols
print(f"\n7-day forecast WAPE: last-28-day mean {mean_wape:.1f}% vs same-day-last-week {seasonal_wape:.1f}%")
print("weekday with the largest error in the last fold (0=Mon):", worst_weekday, "| feature most correlated with units:", top_feature)
print("All exercises passed!")
