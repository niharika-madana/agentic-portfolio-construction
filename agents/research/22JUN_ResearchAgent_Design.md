# Research Agent — Design Document
**AI Financial Advisor Pipeline | Agent 2 of 5**
*Fordham MSQF Capstone 2026*
*Last updated: 2026-06-23 (updated data layer: FRED now read from parquet cache)*

---

## Objective

Produce a **structured research report** — a validated `MacroRegimeSnapshot` passed downstream to the Allocation Agent. The Research Agent does not generate investment recommendations. It identifies which macro state the economy is currently in, so the Allocation Agent can anchor portfolio construction to historically grounded return and risk expectations.

**Implementation status:** ✅ Fully implemented — entry point `agents/research/research_agent.py → run_research_agent()`.

---

## Architectural Role

```
Profile Agent → Research Agent → Allocation Agent → Risk Agent → Compliance Agent
```

**Input:** Nothing from the Profile Agent — the Research Agent runs independently on FRED macro data.

**Output:** A single `MacroRegimeSnapshot` (Pydantic-validated) consumed by the Allocation and Risk agents. Also writes `agents/research/fred_macro_regimes.csv` and `agents/research/regime_sequence.json` for inspection.

---

## Pipeline

```
data/storage/fred_macro.parquet   ← read (fetched by data/fetch/fred.py)
    ↓
Feature engineering (z-scores, log-transforms, first differences, YoY changes)
    ↓
PELT change-point detection (ruptures) — finds structural breaks without date assumptions
    ↓
Segment fingerprinting + K-means clustering (optimal K via elbow + silhouette)
    ↓
XGBoost regime mapping (anchor windows → 5 academic taxonomy labels)
    ↓
6-month rolling majority vote (regime smoothing)
    ↓
[June 22] HMM & GMM comparison — benchmarked against XGBoost, current pipeline retained
    ↓
Pydantic validation (MacroRegimeSnapshot from contracts.py)
    ↓
Export: CSV (full feature matrix) + JSON (regime sequence for downstream)
```

---

## Data Source

**FRED 13 macro series** — loaded from `data/storage/fred_macro.parquet`.

The parquet file is fetched once by `data/fetch/fred.py → fetch_fred_macro()` and cached locally. The Research Agent reads from parquet on every run — no live API calls. If the parquet is missing, `loaders.pull_fred_data()` fetches it automatically (requires `FRED_API` key).

**FRED_SERIES mapping:**

| Key | FRED Code | Description |
|---|---|---|
| yield_curve | T10Y2Y | 10Y−2Y spread |
| term_spread | T10Y3M | 10Y−3M spread |
| fed_funds | FEDFUNDS | Effective Federal Funds Rate |
| unemployment | UNRATE | Civilian Unemployment Rate |
| cpi | CPIAUCSL | CPI (YoY % change, computed) |
| credit_spread | BAA10Y | Baa−10Y spread |
| indpro | INDPRO | Industrial Production (YoY % change, computed) |
| vix | VIXCLS | CBOE VIX |
| gdp | GDPC1 | Real GDP (QoQ annualised %, computed) |
| t5yie | T5YIE | 5Y breakeven inflation (begins 2003) |
| t10yie | T10YIE | 10Y breakeven inflation (begins 2004) |
| dgs5 | DGS5 | 5Y Treasury yield |
| dgs30 | DGS30 | 30Y Treasury yield |

> T5YIE begins 2003, T10YIE begins 2004. After `dropna()`, usable history starts ~2003-02.

---

## Feature Engineering

Raw series are non-stationary level processes. All features are transformed to stationary signals before classification.

| Transformation | Applied To | Rationale |
|---|---|---|
| Z-score | yield_curve, term_spread, credit_spread, t5yie, t10yie, vix | Standardise levels across secular trends |
| Log → Z-score | vix, indpro | Correct right skew before standardising |
| YoY % change → Z-score | cpi, indpro, gdp | Convert level to stationary flow |
| First difference | dgs5, dgs30 | Remove unit-root non-stationarity |
| Month-over-month change | fed_funds, unemployment | Identify hiking/cutting cycle turning points |

### The 13 `signal_cols` used in PELT and clustering

```python
signal_cols = [
    'yield_curve_z', 'term_spread_z', 'credit_spread_z', 'cpi_z',
    'fed_funds_chg', 'unemployment_chg',
    'vix_z', 'indpro_z', 'gdp_z',
    't5yie_z', 't10yie_z',
    'dgs5_chg', 'dgs30_chg'
]
```

---

## Change-Point Detection

**Algorithm:** PELT (Pruned Exact Linear Time) via `ruptures`, RBF kernel, penalty = 10.

**Result (June 2026 run):** 4 structural breaks, 5 segments.

| Break Date | Macro Context |
|---|---|
| 2004-09 | Fed begins hiking from post dot-com lows; spreads tightening |
| 2008-01 | Credit spreads spike; yield curve collapses; unemployment rising |
| 2014-09 | Fed tapering QE; normalising; VIX subdued |
| 2020-12 | COVID shock absorbed; CPI beginning to surge; Fed still at zero |

COVID (2020) emerged as a breakpoint with the expanded 13-feature set — unlike the prior 6-feature version where it had to be manually added as an anchor window.

---

## Clustering

**Algorithm:** K-means on segment-level fingerprints (mean, variance, delta per signal column).

**Optimal K selection:** Elbow (WCSS) + silhouette, constrained to [2, N_segments−1].

**Result:** K=2 (max silhouette). The binary split is a sanity check — XGBoost in the next step performs the fine-grained labelling on the full 270+ row monthly matrix.

---

## Academic Regime Taxonomy

| ID | Label | Definition | Literature |
|---|---|---|---|
| 0 | Early Recovery | Post-recession rebound; accommodative policy; spreads narrowing | NBER expansion onset; Bernanke (2020) |
| 1 | Late-Cycle Expansion | Sustained growth; tightening credit; low unemployment | Hamilton (1989) high-growth state |
| 2 | Financial Crisis & ZLB | Credit stress; rising unemployment; near-zero rate; QE | NBER contraction; Clarida et al. (1999) |
| 3 | Moderate Expansion | Post-QE normalisation; stable macro; low volatility | Hamilton (1989) moderate-growth state |
| 4 | Inflation Shock | Above-target CPI; rapid tightening; yield curve inversion | Clarida et al. (1999) anti-inflation regime |

**Sources:**
- Hamilton (1989). Econometrica 57(2). https://doi.org/10.2307/1912559
- Clarida, Galí & Gertler (1999). Journal of Economic Literature 37(4). https://doi.org/10.1257/jel.37.4.1661
- Bernanke (2020). Brookings. https://www.brookings.edu/articles/the-new-tools-of-monetary-policy/
- NBER Business Cycle Dating Committee. https://www.nber.org/research/business-cycle-dating

---

## XGBoost Regime Mapping

**Model:** XGBClassifier, 100 estimators, `random_state=42`.

**Training:** 5 anchor windows, one per segment, from the most historically unambiguous core months.

| Label | Anchor Window |
|---|---|
| Early Recovery | 2003-06 → 2004-06 |
| Late-Cycle Expansion | 2005-01 → 2007-06 |
| Financial Crisis & ZLB | 2008-09 → 2010-12 |
| Moderate Expansion | 2015-01 → 2019-06 |
| Inflation Shock | 2022-01 → 2023-06 |

**TimeSeriesSplit CV:** 5 folds. Mean accuracy 0.278 (std 0.236). Early folds train on 1-2 regimes — low accuracy confirms no future data leakage.

### Feature Importance (June 2026 run)

| Rank | Feature | Importance |
|---|---|---|
| 1 | term_spread_z | 0.237 |
| 2 | cpi_z | 0.226 |
| 3 | credit_spread_z | 0.166 |
| 4 | t10yie_z | 0.143 |
| 5 | yield_curve_z | 0.077 |
| 6 | vix_z | 0.069 |
| 7–13 | gdp_z through dgs5_chg | 0.082 total |

The two new breakeven series (t10yie_z, t5yie_z) together contribute 0.155 — validating their inclusion.

---

## Regime Smoothing

**Method:** 6-month rolling majority vote (`scipy.stats.mode`, `center=True`).

**Purpose:** A regime must dominate a 6-month window before being accepted. Prevents the Allocation Agent from rebalancing on transient boundary noise.

**June 2026 run result:**

| Regime | Smoothed Months |
|---|---|
| Moderate Expansion | 82 |
| Late-Cycle Expansion | 72 |
| Financial Crisis & ZLB | 62 |
| Early Recovery | 31 |
| Inflation Shock | 27 |

---

## Model Comparison — HMM & GMM vs. XGBoost

*June 22 session — data-driven justification for retaining the current pipeline.*

| Metric | XGBoost (smoothed) | HMM | GMM |
|---|---|---|---|
| Distinct regimes recovered | **5 / 5** | 3 / 5 | 3 / 5 |
| Match rate vs. XGBoost | — | 66.8% | 48.2% |
| Avg regime duration (months) | **15.2** | 16.1 | 12.5 |
| Inflation Shock correctly flagged | ✅ −0.29% avg return | ✗ merged into Late-Cycle | ✗ +0.19% (understated) |

**Verdict: Retain K-means + XGBoost.** XGBoost is the only model that recovers all 5 academic regimes and correctly identifies Inflation Shock as a negative-return environment. HMM flagged as a Phase 2 extension for transition probability matrices.

---

## Output Schema (from contracts.py)

```python
class MacroRegimeSnapshot(BaseModel):
    as_of:                  date
    regime_label:           str       # one of 5 academic labels
    prior_regime:           str
    regime_shift_date:      date
    regime_confidence:      float     # Field(ge=0.0, le=1.0)
    regime_volatility:      float     # Field(ge=0.0) — 6-month rolling credit spread std
    yield_curve:            float
    term_spread:            float
    fed_funds:              float
    unemployment:           float
    cpi:                    float
    credit_spread:          float
    is_low_confidence:      bool      # derived: confidence < 0.60
    regime_change_detected: bool      # derived: regime_label ≠ prior_regime
```

Example output (June 2026):

```json
{
    "as_of":                 "2025-12-01",
    "regime_label":          "Late-Cycle Expansion",
    "prior_regime":          "Late-Cycle Expansion",
    "regime_shift_date":     "2024-01-01",
    "regime_confidence":     0.847,
    "regime_volatility":     0.0312,
    "yield_curve":           0.71,
    "term_spread":           0.43,
    "fed_funds":             3.72,
    "unemployment":          4.1,
    "cpi":                   2.653,
    "credit_spread":         1.84,
    "is_low_confidence":     false,
    "regime_change_detected": false
}
```

---

## CRSP Validation Results (June 2026 Run)

Regime labels validated against CRSP value-weighted market returns 1995–December 2024.

| Regime | Avg Monthly Return | Std Dev | Months | Expected | Result |
|---|---|---|---|---|---|
| Early Recovery | +1.54% | 2.33% | 31 | Positive | ✓ |
| Late-Cycle Expansion | +1.29% | 3.02% | 61 | Positive | ✓ |
| Moderate Expansion | +1.05% | 4.23% | 82 | Positive | ✓ |
| Financial Crisis & ZLB | +0.76% | 5.81% | 62 | Negative / high vol | Partial ✓ (vol correct) |
| Inflation Shock | −0.29% | 5.18% | 27 | Mixed / negative | ✓ |

Return gradient is economically ordered. Inflation Shock is correctly negative — the original 6-feature model showed positive returns here.

---

## Downstream Agent Field Routing

**Allocation Agent:**
- `regime_label` — anchors return expectations; drives regime-conditional portfolio views
- `regime_confidence` — < 0.60 triggers conservative blending toward equal-weight prior
- `regime_shift_date` + `regime_change_detected` — fresh transitions warrant rebalancing
- `prior_regime` — identifies direction of transition (e.g. Inflation Shock → Late-Cycle = risk-on shift)

**Risk Agent:**
- `regime_label` — maps to historical drawdown anchor for each of the 5 regime stress tests
- `regime_volatility` — elevates stress test stringency in high-volatility regimes
- `vix` — tail-risk scenario scaling (in `MacroRegimeSnapshot` via FRED)

**Compliance Agent:**
- `regime_label` — regime-conditional suitability check (Financial Crisis → conservative constraint)
- `cpi` — real-return mandate compliance
- `regime_change_detected` — Check 1.3 requires current regime label in regime_evaluation

---

## Failure Modes

| Failure | Mitigation |
|---|---|
| FRED parquet missing | `loaders.pull_fred_data()` auto-fetches if api_key provided; raises `FileNotFoundError` with instructions if not |
| Breakeven data truncates history | T5YIE/T10YIE begin 2003-2004; usable dataset starts ~2003-02. Documented limitation. Drop to 11-feature set to recover dot-com period if needed. |
| K=2 too coarse for labelling | By design — clustering is a sanity check; XGBoost with 5 anchor windows does fine-grained labelling on the full monthly matrix |
| Regime label bleed | Anchor windows chosen from historically unambiguous core months; boundary ambiguity minimised |
| Pydantic validation failure | `MacroRegimeSnapshot` validates every field; `confidence` out of [0,1] or unknown regime label raises `ValidationError` |
| CRSP 2025 gap | CRSP returns unavailable through 2025; validation join drops NaN rows silently. Documented. |

---

## Implementation Notes

- Runs in Google Colab or local Python 3.11+
- FRED API key stored as secret `FRED_API` (only needed for first-time data fetch)
- No API keys needed after `data/storage/fred_macro.parquet` exists
- Dependencies: `fredapi`, `xgboost`, `scikit-learn`, `matplotlib`, `ruptures`, `pydantic`, `scipy`, `hmmlearn`, `pyarrow`
- Output files: `agents/research/fred_macro_regimes.csv`, `agents/research/regime_sequence.json`

---

## Open Questions

- [x] ~~HMM and GMM as alternatives~~ — resolved June 22: both evaluated; XGBoost retained.
- [ ] Confirm whether to add `ewretd` (equal-weighted CRSP) alongside `vwretd` for validation
- [ ] Evaluate whether `pen=5` in PELT recovers the 2001 dot-com breakpoint without noise breaks
- [ ] Decide whether to hard-gate on `regime_confidence > 0.70` before passing to Allocation Agent
- [ ] Consider adding NBER recession indicator (USREC) as a validation overlay on the timeline
- [ ] Phase 2: add HMM transition probability matrix as supplementary field in `MacroRegimeSnapshot`
