# Research Agent — Design Document
**AI Financial Advisor Pipeline | Agent 2 of 5**
*khive AI LLC | Fordham MSQF Capstone 2026*

---

## Objective

Pull macroeconomic data from FRED, derive signal features, detect structural regime changes from the data itself, and map detected regimes to known historical periods. The output is a definitive macro regime assessment passed downstream to the Allocation Agent to inform personalised portfolio construction.

---

## Core Logic

The agent uses a **three-stage data-driven pipeline**:

1. **Detect segments** — PELT change-point detection finds structural breaks in the multivariate macro signal matrix without any date assumptions
2. **Cluster segments** — K-means (k=5) groups detected segments by macro fingerprint, not by calendar date
3. **Map labels** — XGBoost is used strictly as the supervised mapping stage: trained on short anchor windows of historically unambiguous periods, then applied to predict regime labels for all months

XGBoost is never used to detect breaks or cluster segments. It only translates macro feature patterns into named regime labels. Time-series cross-validation confirms no future data leaks into training.

---

## The Five Market Regimes

| Regime | Approximate Period | Characteristics |
|---|---|---|
| Dot-com Crash | Mar 2000 – Oct 2002 | Equity collapse, rising unemployment, yield curve steep |
| Global Financial Crisis (GFC) | Oct 2007 – Jun 2009 | Credit spread explosion, inverted curve, unemployment spike |
| COVID-19 Shock | Feb 2020 – Dec 2020 | Sharp unemployment surge, Fed rate crash, credit stress |
| Rate Hike Cycle | Mar 2022 – Jul 2023 | Aggressive Fed tightening, inverted yield curve, CPI peak |
| AI Boom | Aug 2023 – present | Equity expansion, normalising inflation, tight labour market |

---

## Key Input Variables

| FRED Series | Code | Feature Derived | Purpose |
|---|---|---|---|
| Yield Curve (10Y−2Y) | `T10Y2Y` | `yield_curve_z` | Classic recession signal — inversion precedes downturns |
| Term Spread (10Y−3M) | `T10Y3M` | `term_spread_z` | NY Fed recession model spread — more sensitive to near-term rate expectations |
| Fed Funds Rate | `FEDFUNDS` | `fed_funds_chg` | Month-over-month change captures hiking/cutting cycles |
| Unemployment Rate | `UNRATE` | `unemployment_chg` | Month-over-month change captures labour market shocks |
| CPI | `CPIAUCSL` | `cpi_z` | Converted to YoY % change (non-stationary raw level); z-scored |
| Credit Spread | `BAA10Y` | `credit_spread_z` | Corporate bond stress indicator |

All series are pulled from **1995–present** and resampled to **monthly frequency**. CPI is converted to year-over-year % change before modelling to ensure stationarity.

The two yield curve measures capture different signals: `T10Y2Y` reflects long-run recession risk, while `T10Y3M` (used in the NY Fed recession probability model) is more sensitive to near-term rate expectations and tends to invert earlier in tightening cycles.

---

## Pipeline Steps

### Step 1 — PELT Change-Point Detection

PELT (Pruned Exact Linear Time) detects structural breaks in the 6-feature signal matrix without any date assumptions. Penalty `pen=10` favours fewer, more persistent breaks over transient spikes.

**Detected breaks (6-feature model):**

| Break Date | Macro Context | Known Regime |
|---|---|---|
| 2001-06 | Both yield curve and term spread inverted, unemployment rising, credit spreads widening | Dot-com bust onset ✓ |
| 2004-10 | Fed funds rising sharply from post-dot-com lows, credit spreads tightening | Pre-GFC credit expansion |
| 2008-02 | Credit spreads spiking, yield curve collapsing, unemployment turning sharply higher | GFC onset ✓ |
| 2016-06 | Fed funds normalising after ZIRP, yield curve beginning to flatten | Post-QE normalisation |
| 2021-06 | CPI surging from post-COVID stimulus, fed funds about to pivot from zero | Inflation surge onset ✓ |
| 2023-02 | Fed funds peak, yield curve deeply inverted, credit spreads stabilising | End of rate hike cycle ✓ |

The 2001 breakpoint shifted from January to June when `T10Y3M` was added — the 10Y−3M spread confirmed the structural break slightly later than 10Y−2Y alone. COVID (2020) did not emerge as a breakpoint due to its short duration and V-shaped recovery; it is handled as an explicit anchor in the XGBoost step.

### Step 2 — K-Means Clustering (k=5)

Each segment is characterised by the mean, variance, and directional delta of each signal feature. K-means clusters these macro fingerprints into 5 candidate regime groups.

| Segment | Cluster | Named Regime Alignment |
|---|---|---|
| 1996-02 → 2001-05 | 0 | Late 90s expansion |
| 2001-06 → 2004-09 | 2 | Dot-com bust ✓ |
| 2004-10 → 2008-01 | 0 | Pre-GFC expansion |
| 2008-02 → 2016-05 | 2 | GFC + ZIRP recovery ✓ |
| 2016-06 → 2021-05 | 3 | Post-QE / pre-COVID |
| 2021-06 → 2023-01 | 1 | Rate hike cycle ✓ |
| 2023-02 → 2025-12 | 4 | AI Boom |

Cluster 0 appears twice (late 90s, pre-GFC) — both are expansion periods with similar macro fingerprints. Cluster 2 also appears twice (Dot-com bust, GFC+ZIRP) — both share elevated credit spreads and rising unemployment. XGBoost anchor windows resolve the label ambiguity within these groups.

### Step 3 — XGBoost Anchor Window Mapping

XGBoost is trained on short, historically unambiguous anchor windows — one per regime. It learns macro feature patterns → regime labels. It never sees dates.

| Regime | Anchor Window | Samples |
|---|---|---|
| Dot-com | 2001-06 → 2003-06 | 25 |
| GFC | 2008-06 → 2010-06 | 25 |
| COVID | 2020-02 → 2020-12 | 11 |
| Rate Hike | 2022-03 → 2023-07 | 17 |
| AI Boom | 2023-08 → 2024-12 | 17 |

**Total training samples: 95**

Time-series cross-validation (`TimeSeriesSplit`, n_splits=5) confirms no future data leaks into training — each fold trains only on past observations and tests on a future held-out window.

**CV accuracy note:** The mean CV accuracy of 0.240 reflects a structural constraint, not model failure. With 95 chronologically sorted samples, early folds train on one or two regimes and test on regimes they have never seen — producing 0.000 accuracy on those folds by design. The CV confirms no future data leakage; it is not a generalisation benchmark for a model trained on all anchor data.

### Step 4 — 6-Month Rolling Majority Vote Smoothing

Raw XGBoost predictions are smoothed with a 6-month rolling majority vote. A regime must dominate a 6-month window before being accepted, preventing the downstream Allocation Agent from rebalancing on transient signal noise.

---

## Output Schema (JSON)

```json
{
  "2025-12-01": {
    "regime_label":      "AI Boom",
    "prior_regime":      "AI Boom",
    "regime_shift_date": "2023-09-01",
    "regime_confidence": 0.971,
    "regime_volatility": 0.0549,
    "yield_curve":       0.71,
    "term_spread":       0.51,
    "fed_funds":         3.72,
    "unemployment":      4.4,
    "cpi":               2.6533,
    "credit_spread":     1.72
  }
}
```

| Field | Description |
|---|---|
| `regime_label` | Smoothed regime name for that month |
| `prior_regime` | Regime label from the previous month — signals transitions |
| `regime_shift_date` | First date of the current regime run — how long we've been in this regime |
| `regime_confidence` | XGBoost max class probability — model certainty for the predicted label |
| `regime_volatility` | 6-month rolling std of credit spread — macro stress level within the regime |
| `yield_curve` | Raw 10Y−2Y Treasury spread |
| `term_spread` | Raw 10Y−3M Treasury spread |
| `fed_funds` | Effective Federal Funds Rate |
| `unemployment` | Civilian Unemployment Rate |
| `cpi` | CPI year-over-year % change |
| `credit_spread` | Moody's Baa corporate bond spread over 10Y Treasury |

---

## Feature Importance (XGBoost)

| Feature | Importance | Interpretation |
|---|---|---|
| `fed_funds_chg` | ~0.34 | Strongest signal — rate hikes/cuts are the clearest policy-driven structural shifts |
| `credit_spread_z` | ~0.28 | Captures financial market stress; spikes sharply in crisis regimes |
| `cpi_z` | ~0.14 | Inflation level relative to history |
| `yield_curve_z` | ~0.14 | Yield curve shape; inversion precedes recessions |
| `unemployment_chg` | ~0.10 | Lags other indicators — confirmation signal, not leading |
| `term_spread_z` | TBD | Near-term rate expectations; complements yield_curve_z |

*Run Cell 18 and update `term_spread_z` importance with the printed score.*

---

## Failure Modes & Validation Plan

### Failure Mode 1 — Recency Bias
The LLM may label routine corrections as systemic crises.

**Mitigation:** XGBoost is the authoritative classifier. The LLM is not used for classification — only for downstream contextualisation if needed.

### Failure Mode 2 — Future Data Leakage in XGBoost
If training data includes months after the test period, the model has look-ahead bias.

**Mitigation:** `TimeSeriesSplit` cross-validation enforces chronological train/test splits. Each fold trains only on past observations.

### Failure Mode 3 — Regime Boundary Ambiguity
Anchor windows drawn incorrectly will misclassify transitional periods.

**Mitigation:** CV accuracy and classification report on held-out fold. Flag months with `regime_confidence < 0.60` as low-confidence in the JSON output.

### Failure Mode 4 — COVID Over-Classification
COVID's macro fingerprint (rapid rate cuts, sudden unemployment spike) resembles other easing episodes, causing the model to assign the label to structurally similar but historically distinct periods.

**Mitigation:** Documented as a known limitation. Tighter anchor windows and additional features (e.g. VIX) are noted as future improvements.

### Failure Mode 5 — Stale FRED Data
If the FRED API pull fails or returns stale data, regime classification is based on outdated conditions.

**Mitigation:** Log the `observation_end` date of each FRED series on every pull. If any series is more than 45 days stale relative to today, raise a `DataStalenessError` and halt the pipeline.

---

## CRSP Validation Results

| Regime | Avg Monthly Return | Std Dev | Months | Expected Direction |
|---|---|---|---|---|
| AI Boom | +1.28% | 3.47% | 81 | Positive ✓ |
| COVID | +1.11% | 5.33% | 68 | Mixed |
| Dot-com | +0.75% | 4.07% | 95 | Negative ✗ |
| Rate Hike | +0.98% | 4.51% | 53 | Negative ✗ |
| GFC | -0.04% | 5.96% | 50 | Negative ✓ |

AI Boom and GFC validate cleanly. Dot-com and Rate Hike show positive average returns due to over-classification — expansion months are being absorbed into these labels. This is a documented limitation, not a pipeline failure.

---

## Implementation Notes

- Runs in **Google Colab** with FRED API key stored via `userdata.get('FRED_API')`
- Full feature matrix saved to `agents/research/fred_macro_regimes.csv`
- Regime sequence saved to `agents/research/regime_sequence.json`
- Notebooks are prototypes — to be refactored into `.py` modules before Thursday

---

## Open Questions

- [ ] Update feature importance table with actual `term_spread_z` score from Cell 18 output
- [ ] Evaluate whether tighter anchor windows reduce Dot-com and COVID over-classification
- [ ] Confirm whether `regime_confidence < 0.60` threshold should be a hard gate before the Allocation Agent ingests the regime
- [ ] Determine final date boundary for the AI Boom regime as more data becomes available
