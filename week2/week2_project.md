# Week 2 Store Data Business Report

**Data period:** January 6–August 25, 2025  
**Methods:** pandas resampling, grouping and pivot tables, cohort retention, repeat/churn metrics, and hypothesis tests.

## 1. How is revenue trending, and is the latest month complete?

| Month | Net revenue | Orders | MoM growth |
|---|---:|---:|---:|
| January | $144.04 | 22 | — |
| February | $81.16 | 12 | -43.7% |
| March | $83.69 | 16 | +3.1% |
| April | $54.66 | 11 | -34.7% |
| May | $95.92 | 19 | +75.5% |
| June | $86.43 | 11 | -9.9% |
| July | $106.69 | 18 | +23.4% |
| August* | $47.47 | 9 | -55.5% |

*August is incomplete; the latest order is August 25. January had the highest revenue in the period. Revenue fluctuated rather than following a steady trend, with recoveries in May and July.

**Recommendation:** Track revenue alongside order and customer counts. Compare partial months with the same number of days in previous months.

## 2. Which geographies and categories matter most, in total and per customer?

The customer file provides **city**, not state, so city-level results are used.

| City | Customers | Net revenue | Revenue per customer |
|---|---:|---:|---:|
| Denver | 11 | $233.75 | $21.25 |
| Austin | 8 | $233.42 | **$29.18** |
| Boston | 8 | $175.58 | $21.95 |
| Seattle | 3 | $57.30 | $19.10 |

Denver generated the most revenue, while Austin had the highest revenue per customer.

| Category | Orders | Net revenue | Return rate |
|---|---:|---:|---:|
| Coffee | 48 | **$250.79** | 8.3% |
| Bakery | 37 | $209.40 | 2.7% |
| Tea | 26 | $132.85 | 7.7% |
| Merch | 7 | $107.01 | 0.0% |

Coffee was the top category by revenue. Merch had high revenue per category buyer ($15.29), but that result is based on only seven orders.

**Recommendation:** Protect Coffee sales, test Bakery cross-selling, and gather more Merch data before expanding it.

## 3. Do customers come back?

- **Repeat rate:** 27 of 30 customers ordered more than once (**90.0%**).
- **Average orders per customer:** 3.93.
- **90-day inactivity:** 8 of 30 customers (26.7%) had not ordered within 90 days of August 25. This is a churn indicator, not proof they will never return.

Retention varied by first-order cohort. For example, 37.5% of January's 16 customers ordered in month 1, compared with 20.0% of February's cohort. Later cohorts have less follow-up data, and small cohorts make percentages unstable.

**Recommendation:** Test a win-back reminder for inactive customers and continue tracking retention.

## 4. Are member/non-member spending and category return-rate differences real?

Members averaged **$29.28 per customer**, compared with **$21.11** for non-members. The Welch t-test returned **p = 0.058**, above the 0.05 significance threshold, so the observed difference is not statistically significant at that level.

There were 7 returns among 118 orders (**5.9%** overall). Category return rates ranged from 0% for Merch to 8.3% for Coffee. The chi-square test gave **p = 0.621**, showing no statistically significant category difference. With only seven returns, this result is inconclusive.

**Recommendation:** Do not change membership or return policies based on this small sample. Collect more data and reassess.

## 5. Why did revenue fall from December to January, and what should we do?

The files contain orders from **January 6, 2025 onward**, with no December observations. The requested December-to-January comparison therefore cannot be calculated.

As a substitute example, revenue fell from **$144.04 in January to $81.16 in February** (-43.7%). Orders decreased from 22 to 12, while average net revenue per order increased slightly from about $6.55 to $6.76. The decline appears primarily associated with fewer orders, not a lower average order value.

**Recommendation:** Obtain December data, then compare revenue, order count, customer count, average order value, and category-level changes to identify the actual cause.

## Conclusion and data notes

Priorities are to monitor complete-month performance, protect leading categories, encourage repeat purchases, and collect more data before drawing conclusions from statistical tests.

The analysis removed duplicate orders and rows with missing amounts, capped extreme amounts using the supplied Week 2 method, and counted returned orders as **$0 net revenue**. The customer file has `city` rather than `state` and no `signup_date`, limiting state analysis and sign-up conversion calculations.
