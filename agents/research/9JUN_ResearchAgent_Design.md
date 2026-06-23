# Research Agent — Design Document
**AI Financial Advisor Pipeline | Agent 2 of 5**
*khive AI LLC | Fordham MSQF Capstone 2026*

---

## Objective

Scan macroeconomic data from FRED and classify the current market environment into one of five historical regime analogues using a fully data-driven pipeline. The output is a regime sequence and current regime label passed downstream to the Allocation Agent to inform personalised portfolio construction.

---

## Core Logic

The agent uses a **three-stage data-driven pipeline** — no regime boundaries are manually hardcoded:

1. **Signal extraction**: Pull and transform FRED macro series into regime-sensitive features (z-scores, rate of change)
2. **Change-point detection**: Apply the PELT algorithm (via `ruptures`) to find structural breaks in the multivariate signal matrix — without any date assumptions
3. **Regime mapping**: Cluster detected segments with K-means, then train XGBoost on a small set of anchor points to map macro fingerprints to named regime labels

This flow — **signals → detect regime change → map to regime definitions** — ensures XGBoost learns feature → label, never date → label. Regime boundaries emerge from the data itself rather than being assumed a priori.

---

## The Five Market Regimes

| Regime | Approximate Period | Characteristics |
|---|---|---|
| Dot-com Crash | Mar 2000 – Oct 2002 | Equity collapse, rising unemployment, yield curve steep |
| Global Financial Crisis (GFC) | Oct 2007 – Jun 2009 | Credit spread explosion, unemployment spike, near-zero rates |
| COVID-19 Shock | Feb 2020 – Dec 2020 | Sharpest unemployment spike in history, rapid Fed rate cut |
| Rate Hike Cycle | Mar 2022 – Jul 2023 | Aggressive Fed tightening, inverted yield curve, CPI peak |
| AI Boom | Aug 2023 – present | Equity expansion, normalising inflation, tight labour market |

---

## Key Input Variables & Feature Engineering

| Raw FRED Series | Code | Transformation | Rationale |
|---|---|---|---|
| Yield Curve | `T10Y2Y` | Z-score | Captures how extreme inversion is vs. history |
| Credit Spread | `BAA10Y` | Z-score | Captures financial stress relative to history |
| CPI | `CPIAUCSL` | YoY % change, then z-score | Removes non-stationarity of raw level |
| Fed Funds Rate | `FEDFUNDS` | Month-over-month change | Captures hiking/cutting turning points |
| Unemployment | `UNRATE` | Month-over-month change | Captures labour market shocks |

All series pulled from **1995–2025** and resampled to **monthly frequency (MS)**.

---

## Data Sources

### Quantitative (FRED API)
- `T10Y2Y` — Yield curve inversion tracker
- `FEDFUNDS` — Effective Federal Funds Rate
- `UNRATE` — Civilian Unemployment Rate
- `CPIAUCSL` — Consumer Price Index, All Urban Consumers (converted to YoY % change)
- `BAA10Y` — Moody's Baa Corporate Bond Yield vs 10Y Treasury

### Market Returns (WRDS / CRSP)
- `crsp_market_index.csv` — Monthly value-weighted (`vwretd`) and equal-weighted (`ewretd`) market returns, 1995–2024. Used to validate regime labels against actual market return patterns.

---

## Pipeline Architecture

### Stage 1 — Signal Feature Derivation
Raw FRED series are transformed into regime-sensitive features before any modelling:
- Z-scores for level-based signals (yield curve, credit spread, CPI)
- Month-over-month differences for policy and labour signals (fed funds, unemployment)

### Stage 2 — Change-Point Detection (PELT)
The `ruptures` PELT algorithm identifies structural breaks in the multivariate feature matrix without prior date labels. Running at `pen=10`, it detected **6 structural breaks** across 1996–2025:

| Break Date | Macro Context |
|---|---|
| 2001-01 | Dot-com bust onset |
| 2004-10 | Pre-GFC credit expansion begins |
| 2008-02 | GFC onset |
| 2016-06 | Post-QE normalisation |
| 2021-06 | Inflation surge onset |
| 2023-02 | End of rate hike cycle |

Four of six breaks align with named regimes from the project brief without the algorithm being told when those events occurred.

### Stage 3 — K-Means Segment Clustering
The 7 segments between breakpoints are characterised by their mean, variance, and directional change of each macro feature. K-means (k=5) clusters these into candidate regime groups based on macro fingerprint alone.

### Stage 4 — XGBoost Anchor Mapping
A small set of anchor windows — one per named regime — trains XGBoost to map macro features to regime labels. XGBoost never sees dates:

| Regime | Anchor Window |
|---|---|
| Dot-com | 2001-06 → 2003-06 |
| GFC | 2008-06 → 2010-06 |
| COVID | 2020-02 → 2020-12 |
| Rate Hike | 2022-03 → 2023-07 |
| AI Boom | 2023-08 → 2024-12 |

Total training samples: 95 months across 5 classes.

### Stage 5 — Regime Smoothing
Raw XGBoost predictions are smoothed with a **6-month rolling majority vote** to enforce regime persistence and prevent the downstream Allocation Agent from rebalancing on transient signal noise.

---

## Feature Importance

| Feature | Importance Score | Interpretation |
|---|---|---|
| `fed_funds_chg` | 0.339 | Rate policy changes most cleanly separate regimes |
| `credit_spread_z` | 0.284 | Financial stress is the second strongest discriminator |
| `cpi_z` | 0.143 | Inflation level contributes to stagflation vs. boom distinction |
| `yield_curve_z` | 0.139 | Curve shape confirms recession vs. expansion |
| `unemployment_chg` | 0.095 | Lags other indicators — confirmation signal, not leading |

---

## Output Schema (JSON)

The regime sequence is exported as a JSON object keyed by date, passed downstream to the Allocation Agent:

```json
{
  "2025-12-01": {
    "regime_label": "AI Boom",
    "yield_curve": 0.71,
    "fed_funds": 3.72,
    "unemployment": 4.4,
    "cpi": 2.6533,
    "credit_spread": 1.72
  }
}
```

Output files:
- `agents/research/fred_macro_regimes.csv` — full feature matrix with smoothed regime labels
- `agents/research/regime_sequence.json` — regime sequence for downstream agents

---

## CRSP Validation Results

| Regime | Avg Monthly Return | Std Dev | Months | Expected | Validated |
|---|---|---|---|---|---|
| AI Boom | +1.28% | 3.47% | 81 | Positive | ✓ |
| COVID | +1.11% | 5.33% | 68 | Mixed | Partial |
| Dot-com | +0.75% | 4.07% | 95 | Negative | ✗ |
| Rate Hike | +0.98% | 4.51% | 53 | Negative | ✗ |
| GFC | -0.04% | 5.96% | 50 | Negative | ✓ |

AI Boom and GFC validate cleanly. Dot-com and Rate Hike show positive average returns due to over-classification — the model assigns those labels to structurally similar expansion periods, diluting the crisis signal. COVID over-classification (68 months vs. 11 months actual) reflects its macro fingerprint resembling other monetary easing periods.

---

## Failure Modes & Validation Plan

### Failure Mode 1 — Regime Over-Classification
The XGBoost model over-generalises when macro fingerprints from different historical periods are statistically similar, assigning crisis labels to expansion months.

**Validation:** CRSP return validation per regime. If a crisis regime (Dot-com, GFC, Rate Hike) shows positive average monthly returns, flag the label boundaries for review. Future mitigation: tighter anchor windows, additional features (VIX), hard duration caps.

### Failure Mode 2 — Change-Point Sensitivity
PELT's penalty parameter (`pen`) controls the number of detected breakpoints. Too high → misses real breaks. Too low → over-segments noise.

**Validation:** Visual inspection of detected break dates against known macro events. At `pen=10`, 4 of 6 breaks align with named regimes. Sensitivity analysis across `pen` values (5, 10, 15, 20) documented in notebook.

### Failure Mode 3 — COVID Non-Detection
COVID's V-shaped recovery prevents PELT from identifying it as a standalone structural break.

**Validation:** COVID is handled as an explicit anchor window in the XGBoost mapping step, ensuring it remains represented in the final regime sequence despite not emerging from unsupervised detection.

### Failure Mode 4 — Stale FRED Data
If the FRED API pull fails or returns stale data, regime classification will be based on outdated macro conditions.

**Validation:** Log the `observation_end` date of each FRED series on every pull. If any series is more than 45 days stale relative to the current date, raise a `DataStalenessError` and halt the pipeline.

---

## Implementation Notes

- Runs in **Google Colab** with FRED API key stored via `userdata.get('FRED_API')`
- CRSP market index file uploaded directly to Colab session (`crsp_market_index.csv`)
- Full feature matrix saved to `agents/research/fred_macro_regimes.csv`
- Regime sequence JSON saved to `agents/research/regime_sequence.json`
- XGBoost model to be serialised to `agents/research/regime_classifier.pkl` for reuse

---

## Open Questions

- [ ] Evaluate whether adding VIX (`VIXCLS`) as a sixth FRED feature improves Dot-com and Rate Hike classification accuracy
- [ ] Determine whether hard regime duration constraints (minimum/maximum months per regime) should be enforced post-smoothing
- [ ] Assess whether tighter anchor windows (±3 months vs. current ±12 months) reduce COVID and Dot-com over-classification
- [ ] Confirm integration contract with Allocation Agent — does it consume the full regime sequence or only the current month label
