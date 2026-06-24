# Research Agent — Design Document
**AI Financial Advisor Pipeline | Agent 2 of 5**
*Fordham MSQF Capstone 2026*
*Last updated: 2026-06-25 (June 25 session — FRED and CRSP data migrated to Parquet cache; MacroRegimeSnapshot contract added; is_low_confidence and regime_change_detected derived fields implemented)*

---

## Objective

Produce a **structured research report** — a validated `MacroRegimeSnapshot` passed downstream to the Allocation and Risk agents. The Research Agent does not generate investment recommendations. It identifies which macro state the economy is currently in, so the Allocation Agent can anchor portfolio construction to historically grounded return and risk expectations.

**Implementation status:** ✅ Fully implemented — entry point `agents/research/research_agent.py → run_research_agent()`.

---

## Architectural Role

The Research Agent sits between the Profile Agent and the Allocation–Risk–Compliance loop:

```
Profile Agent → Research Agent → Allocation Agent → Risk Agent → Compliance Agent
```

**Input:** nothing from the Profile Agent directly — the Research Agent runs independently on FRED macro data.

**Output:** A single `MacroRegimeSnapshot` (Pydantic-validated) consumed by the Allocation and Risk agents. Also writes:
- `agents/research/fred_macro_regimes.csv` — full feature matrix for inspection
- `agents/research/regime_sequence.json` — date-keyed validated regime sequence
- `agents/research/macro_regime_snapshot.json` — most recent month snapshot
- `data/storage/fred_macro_regimes.parquet` — compressed feature matrix for downstream agents

> *Ocean (June 11 meeting): "The research agent follows the persona/profile report. Its sole deliverable is a research report, which then becomes the input for the allocation–proposal–risk loop."*

---

## Pipeline

```
data/storage/fred_macro.parquet   ← read (fetched by FRED API on first run, cached thereafter)
    ↓
Feature engineering (z-scores, log-transforms, first differences, YoY changes)
    ↓
PELT change-point detection (ruptures) — finds structural breaks without date assumptions
    ↓
Segment fingerprinting + K-means clustering (optimal K via elbow + silhouette)
    ↓
XGBoost regime mapping (anchor windows → academic taxonomy labels)
    ↓
6-month rolling majority vote (regime smoothing)
    ↓
[June 22] HMM & GMM comparison — benchmarked against XGBoost, current pipeline retained
    ↓
Pydantic validation (RegimeRecord — full sequence; MacroRegimeSnapshot — most recent month)
    ↓
Export: CSV + JSON (regime sequence) + Parquet (feature matrix) + MacroRegimeSnapshot JSON
```

---

## Macro Feature Set

13 FRED series pulled from 1995-01-01 to 2025-12-31 and resampled to monthly frequency.

| Series | FRED Code | Transformation | Purpose |
|--------|-----------|---------------|---------|
| Yield Curve (10Y−2Y) | T10Y2Y | Z-score | Classic recession signal — inversion precedes downturns |
| Term Spread (10Y−3M) | T10Y3M | Z-score | NY Fed recession model spread — most important feature (#1 importance) |
| Credit Spread | BAA10Y | Z-score | Corporate bond stress indicator (#3 importance) |
| CPI YoY | CPIAUCSL | YoY % change → Z-score | Inflation regime signal (#2 importance) |
| 10Y Breakeven Inflation | T10YIE | Z-score | Long-run inflation expectations (#4 importance) |
| 5Y Breakeven Inflation | T5YIE | Z-score | Near-term inflation expectations |
| VIX | VIXCLS | Log-transform → Z-score | Equity fear gauge — skew-corrected |
| Real GDP | GDPC1 | QoQ annualised % change → Z-score | Output growth proxy (quarterly, forward-filled) |
| Industrial Production | INDPRO | YoY % change → Log-transform → Z-score | Real-economy output |
| Fed Funds Rate | FEDFUNDS | Month-over-month change | Identifies hiking/cutting cycles |
| Unemployment | UNRATE | Month-over-month change | Labour market turning points (lagging indicator) |
| 5Y Treasury Yield | DGS5 | First difference | Mid-curve rate cycle signal |
| 30Y Treasury Yield | DGS30 | First difference | Long-end anchor |

> **Note:** T5YIE begins 2003, T10YIE begins 2004. After `dropna()`, the usable dataset starts ~2003-02, trimming the pre-2003 period (dot-com bust onset).

---

## Feature Engineering

Raw FRED series are non-stationary level processes. All features are transformed to stationary signals before classification.

### Transformation Logic

- **Z-scores** — standardise absolute levels so the model compares macro states across time rather than reacting to secular trends (rates were structurally higher in the 1990s than the 2010s).
- **Log-transforms** — correct right skew in VIX and INDPRO before z-scoring. Untransformed, a VIX spike to 80 would dominate all other signals.
- **First differences** — remove unit-root non-stationarity from DGS5 and DGS30. These yields are not co-integrated with the 10Y−2Y spread and must be differenced rather than z-scored.
- **YoY % changes** — convert non-stationary level series (CPI, INDPRO, GDP) into stationary flow measures before further transformation.

### The 13 `signal_cols` Used in PELT and Clustering

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

**Algorithm:** PELT (Pruned Exact Linear Time) via `ruptures` library, RBF kernel, penalty = 10.

**Result (June 2026 run):** 4 structural breaks detected, dividing 2003–2025 into 5 segments.

| Break Date | Macro Context |
|-----------|--------------|
| 2004-09 | Fed begins hiking from post dot-com lows; credit spreads tightening |
| 2008-01 | Credit spreads spike; yield curve collapses; unemployment turns higher |
| 2014-09 | Fed tapering QE; normalising rates; VIX subdued |
| 2020-12 | COVID shock absorbed; CPI beginning to surge; Fed still at zero |

With 13 features (vs. 6 previously), PELT requires a larger multivariate shift to trigger a break — filtering out minor transitions and retaining only the four most economically significant structural shifts.

**Notable:** COVID-19 (2020) emerged as a breakpoint with the expanded feature set, unlike the 6-feature version where it had to be added manually as an anchor window.

---

## Clustering

**Algorithm:** K-means on segment-level fingerprints (mean, variance, delta per signal column).

**Optimal K selection:** Elbow method (WCSS vs K) + silhouette score, K constrained to [2, N_segments − 1].

**Result:** K=2 selected (max silhouette). With only 5 segments, silhouette monotonically favours fewer clusters — this is expected, not a failure. The binary split is economically meaningful:

| Cluster | Character | Segments |
|---------|-----------|---------|
| 0 | Stress / low-growth | Post dot-com recovery, GFC+ZIRP, post-QE normalisation |
| 1 | Expansion / tightening | Pre-GFC credit boom, post-COVID inflation shock |

The coarse K=2 grouping is resolved into the 5-regime academic taxonomy by XGBoost in the next step, which operates on the full 270+ row monthly feature matrix rather than 5 segment fingerprints.

---

## Academic Regime Taxonomy

Ad-hoc event labels (Dot-com, COVID, AI Boom) replaced with macro-state definitions grounded in the NBER business cycle and monetary policy literature.

| ID | Academic Label | Definition | Literature |
|----|---------------|-----------|-----------|
| 0 | Early Recovery | Post-recession rebound; accommodative policy; credit spreads narrowing | NBER expansion onset; Bernanke (2020) |
| 1 | Late-Cycle Expansion | Sustained growth; tightening credit; low unemployment | Hamilton (1989) high-growth state |
| 2 | Financial Crisis & ZLB | Credit stress; rising unemployment; near-zero policy rate; QE | NBER contraction; Clarida et al. (1999) |
| 3 | Moderate Expansion | Post-QE normalisation; stable macro; low volatility | Hamilton (1989) moderate-growth state |
| 4 | Inflation Shock | Above-target CPI; rapid monetary tightening; yield curve inversion | Clarida et al. (1999) anti-inflation regime |

**Sources:**
- NBER Business Cycle Dating Committee: https://www.nber.org/research/business-cycle-dating
- Bernanke, B. (2020). *The New Tools of Monetary Policy*: https://www.brookings.edu/articles/the-new-tools-of-monetary-policy/
- Hamilton, J.D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*. Econometrica, 57(2): https://doi.org/10.2307/1912559
- Clarida, R., Galí, J., & Gertler, M. (1999). *The Science of Monetary Policy*. Journal of Economic Literature, 37(4): https://doi.org/10.1257/jel.37.4.1661

---

## XGBoost Regime Mapping

**Model:** XGBClassifier, 100 estimators, random_state=42.

**Training:** 5 anchor windows (one per segment), chosen from the most historically unambiguous months inside each segment boundary.

| Label | Anchor Window | Rationale |
|-------|-------------|-----------|
| Early Recovery | 2003-06 → 2004-06 | Post dot-com, rates low, spreads narrowing |
| Late-Cycle Expansion | 2005-01 → 2007-06 | Pre-GFC boom, rising rates, low VIX |
| Financial Crisis & ZLB | 2008-09 → 2010-12 | Acute GFC + early ZIRP |
| Moderate Expansion | 2015-01 → 2019-06 | Post-QE normalisation, stable macro |
| Inflation Shock | 2022-01 → 2023-06 | Peak CPI, aggressive Fed tightening |

**Time-series CV:** TimeSeriesSplit (5 folds). Mean accuracy 0.278 (std 0.236) — early folds train on 1–2 regimes and test on unseen ones by design. This confirms no future data leakage; it is not a generalisation benchmark.

### Feature Importance (Actual Values, June 2026 Run)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | term_spread_z | 0.237 |
| 2 | cpi_z | 0.226 |
| 3 | credit_spread_z | 0.166 |
| 4 | t10yie_z | 0.143 |
| 5 | yield_curve_z | 0.077 |
| 6 | vix_z | 0.069 |
| 7 | gdp_z | 0.034 |
| 8 | unemployment_chg | 0.014 |
| 9 | t5yie_z | 0.012 |
| 10 | indpro_z | 0.012 |
| 11 | fed_funds_chg | 0.006 |
| 12 | dgs30_chg | 0.003 |
| 13 | dgs5_chg | 0.002 |

**Key insight:** In the original 6-feature model, `fed_funds_chg` was the top signal. With 13 features, `term_spread_z` and `cpi_z` take the top two positions. The two new breakeven series (`t10yie_z`, `t5yie_z`) together contribute 0.155 importance — validating their inclusion.

---

## Regime Smoothing

**Method:** 6-month rolling majority vote (scipy.stats.mode, center=True).

**Purpose:** Raw XGBoost predictions switch month-to-month near regime boundaries. The majority vote enforces regime persistence — a regime must dominate a 6-month window before being accepted — preventing the Allocation Agent from rebalancing on transient signal noise.

**Result (June 2026 run):**

| Regime | Smoothed Months |
|--------|----------------|
| Moderate Expansion | 82 |
| Late-Cycle Expansion | 72 |
| Financial Crisis & ZLB | 62 |
| Early Recovery | 31 |
| Inflation Shock | 27 |

---

## Model Comparison — HMM & GMM vs. XGBoost

*Added June 22 session. Meeting action item: explore HMM and GMM as alternatives or produce a data-driven justification for retaining the current pipeline.*

### Models Evaluated

**Gaussian HMM** (`hmmlearn`, diagonal covariance, n_components=5, n_iter=1000, random_state=42) — a probabilistic state-space model that encodes regime persistence via a Markov transition matrix. Diagonal covariance used because full covariance requires estimating 13×13 = 169 parameters per state, infeasible with ~270 observations.

**Gaussian Mixture Model** (`sklearn`, full covariance, n_components=5, n_init=10, random_state=42) — an unsupervised density estimator that models the joint distribution of macro features as a mixture of 5 Gaussians. No temporal structure — classifies each month independently.

Both models were run on the same 13 `signal_cols` feature set. Cluster IDs were mapped to academic regime labels by majority-overlap against the smoothed XGBoost labels on the anchor windows.

### Quantitative Results

| Metric | XGBoost (smoothed) | HMM | GMM |
|---|---|---|---|
| Distinct regimes recovered | **5 / 5** | 3 / 5 | 3 / 5 |
| Match rate vs. XGBoost | — | 66.8% | 48.2% |
| Avg regime duration (months) | **15.2** | 16.1 | 12.5 |
| Regimes missing | none | Early Recovery, Inflation Shock | Early Recovery, Moderate Expansion |

### CRSP Return Validation — All Three Models

| Regime | XGBoost Avg Return | HMM Avg Return | GMM Avg Return | Expected |
|---|---|---|---|---|
| Early Recovery | **+1.54%** | — | — | Positive |
| Late-Cycle Expansion | **+1.29%** | +0.85% | +1.40% | Positive |
| Moderate Expansion | **+1.05%** | +1.19% | — | Positive |
| Financial Crisis & ZLB | +0.76% | +0.89% | +1.00% | Negative / high vol |
| Inflation Shock | **−0.29%** | — | +0.19% | Negative |

### Key Findings

**HMM (match rate 66.8%, 3/5 regimes recovered):**
- Fails to recover `Early Recovery` and `Inflation Shock` — the two regimes most critical for portfolio differentiation
- Two clusters both map to `Financial Crisis & ZLB` (cluster collision); cluster 3 contains only 1 month (degenerate solution)
- Missing `Inflation Shock` is the most damaging failure — HMM absorbs those months into `Late-Cycle Expansion` (+0.85%), masking the negative-return signal of the 2022–2023 tightening cycle
- Avg duration of 16.1 months is marginally higher than XGBoost's 15.2 — the Markov transition matrix adds modest persistence but not meaningfully more than the 6-month majority vote already achieves

**GMM (match rate 48.2%, 3/5 regimes recovered):**
- Lowest match rate of all three models; three of five clusters collapse into `Financial Crisis & ZLB`
- Without temporal structure, GMM over-fits to the largest cluster and absorbs structurally distinct periods into the same label
- `Inflation Shock` partially recovered but shown as +0.19% avg return vs. XGBoost's correct −0.29% — understates drawdown risk
- Avg duration of 12.5 months is the lowest of all three models — more regime churn than XGBoost with no mechanism to prevent it

### Verdict: Retain K-means + XGBoost Pipeline

The comparison provides a clear, data-driven justification for keeping the current approach:

1. **Taxonomy completeness:** XGBoost recovers all 5 academic regimes. HMM and GMM recover only 3, losing the two regimes most critical for portfolio differentiation.
2. **Inflation Shock identification:** XGBoost is the only model that correctly flags the 2022–2023 tightening cycle as a negative-return environment (−0.29%). This is load-bearing for the Allocation Agent.
3. **Match rate:** HMM achieves 66.8%, GMM only 48.2%. Neither clears a threshold that would justify replacing the existing pipeline.
4. **Regime churn:** GMM's avg duration of 12.5 months is lower than XGBoost's 15.2, producing more rebalancing noise downstream.
5. **Interpretability:** XGBoost anchor windows are tied to historically unambiguous periods — every label is defensible in the oral defense. HMM and GMM cluster IDs have no inherent economic meaning and require post-hoc mapping that itself introduces ambiguity.

**HMM as a future extension:** The one capability HMM adds that XGBoost lacks is a transition probability matrix — a forward-looking probability of moving from the current regime to each of the others. This could be added as a supplementary field in `MacroRegimeSnapshot` without replacing the XGBoost classifier. Flagged as a Phase 2 enhancement for the paper's Future Work section.

---

## Pydantic Validation

### RegimeRecord — Per-Row Validation
Every row is validated against `RegimeRecord` before writing to disk. No row with an invalid regime label, out-of-range confidence score, or negative volatility can enter the output JSON.

```python
class RegimeRecord(BaseModel):
    regime_label:      str        # must be one of 5 valid academic labels
    prior_regime:      str        # valid label or "None"
    regime_shift_date: str
    regime_confidence: float = Field(ge=0.0, le=1.0)
    regime_volatility: float = Field(ge=0.0)
    yield_curve:       float
    term_spread:       float
    fed_funds:         float
    unemployment:      float
    cpi:               float
    credit_spread:     float
    vix:               float
    indpro:            float
```

### MacroRegimeSnapshot — Downstream Contract
Single validated Pydantic object for the most recent month, passed directly to the orchestrator and consumed by the Allocation and Risk agents. Two derived boolean fields are enforced by `model_validator`:

```python
class MacroRegimeSnapshot(BaseModel):
    as_of:                  date
    regime_label:           str
    prior_regime:           str
    regime_shift_date:      date
    regime_confidence:      float = Field(ge=0.0, le=1.0)
    regime_volatility:      float = Field(ge=0.0)
    yield_curve:            float
    term_spread:            float
    fed_funds:              float
    unemployment:           float
    cpi:                    float
    credit_spread:          float
    is_low_confidence:      bool   # derived: confidence < 0.60
    regime_change_detected: bool   # derived: regime_label ≠ prior_regime
```

> *Ocean (June 11 meeting): "The research agent's output must be validated because LLM-generated numbers can be fabricated; proper validation via Pydantic models is needed before feeding the research report downstream."*

---

## Output Schema

### regime_sequence.json — Full Sequence
Date-keyed JSON of all validated monthly regime records:

```json
{
  "2025-12-01": {
    "regime_label":      "Late-Cycle Expansion",
    "prior_regime":      "Late-Cycle Expansion",
    "regime_shift_date": "2024-01-01",
    "regime_confidence": 0.847,
    "regime_volatility": 0.0312,
    "yield_curve":       0.71,
    "term_spread":       0.43,
    "fed_funds":         3.72,
    "unemployment":      4.1,
    "cpi":               2.653,
    "credit_spread":     1.84,
    "vix":               14.95,
    "indpro":            1.23
  }
}
```

### macro_regime_snapshot.json — Most Recent Month
Single `MacroRegimeSnapshot` object passed to the orchestrator:

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

Regime labels validated against CRSP value-weighted market returns (1995–December 2024). 2025 months excluded — CRSP data not yet publicly available.

| Regime | Avg Monthly Return | Std Dev | Months | Expected | Result |
|--------|-------------------|---------|--------|----------|--------|
| Early Recovery | +1.54% | 2.33% | 31 | Positive | ✓ |
| Late-Cycle Expansion | +1.29% | 3.02% | 61 | Positive | ✓ |
| Moderate Expansion | +1.05% | 4.23% | 82 | Positive | ✓ |
| Financial Crisis & ZLB | +0.76% | 5.81% | 62 | Negative / high vol | Partial ✓ |
| Inflation Shock | −0.29% | 5.18% | 27 | Mixed / negative | ✓ |

Four of five regimes validate directionally. Financial Crisis & ZLB validates on volatility (5.81% — highest of all regimes). The return gradient is economically ordered: Early Recovery > Late-Cycle > Moderate > Financial Crisis > Inflation Shock.

**Improvement over previous version:** The original 6-feature model showed positive average returns for Dot-com, GFC, and Rate Hike — all expected to be negative. The academic taxonomy with 13 features resolves this: Inflation Shock correctly goes negative (−0.29%) and no label dominates implausibly (max 82 months vs. prior 110 months).

---

## Downstream Agent Field Routing

**Allocation Agent** consumes:
- `regime_label` — anchors portfolio weight construction to academic regime state
- `regime_confidence` — weights the regime signal (low confidence → more conservative positioning)
- `is_low_confidence` — triggers conservative blending toward equal-weight prior if True (confidence < 0.60)
- `regime_shift_date` + `regime_change_detected` — fresh transitions warrant rebalancing
- `prior_regime` — identifies transition direction (e.g. Inflation Shock → Late-Cycle = risk-on shift)

**Risk Agent** consumes:
- `regime_label` — maps to historical drawdown distributions per regime for stress testing
- `regime_volatility` — macro stress proxy; elevates VaR threshold in high-volatility regimes
- `vix` — equity fear gauge for tail-risk scenario construction
- `is_low_confidence` — elevates stress test stringency when regime classification is uncertain

**Compliance Agent** consumes:
- `regime_label` — regime-conditional suitability checks (e.g. Financial Crisis & ZLB triggers conservative allocation constraints)
- `cpi` — inflation threshold check for real-return mandate compliance
- `regime_change_detected` — Check 1.3 requires current regime label in regime_evaluation

---

## Failure Modes & Validation Plan

### Failure Mode 1 — Breakeven Data Truncates History
T5YIE and T10YIE begin in 2003–2004, trimming the usable dataset. The pre-2003 dot-com bust period is outside the analysis window.

**Mitigation:** Documented limitation. If pre-2003 coverage is required, drop the breakeven series and revert to the 11-feature set — the dot-com period becomes available at the cost of losing inflation expectation signals.

### Failure Mode 2 — K=2 Is Too Coarse for Regime Labelling
With only 5 PELT segments, silhouette always favours K=2. The clustering step cannot produce 5 meaningfully distinct clusters.

**Mitigation:** By design — clustering is a sanity check, not the labelling mechanism. XGBoost with 5 anchor windows performs the fine-grained labelling on the full monthly matrix.

### Failure Mode 3 — Regime Label Bleed
Macro fingerprints from different historical periods may be statistically similar (e.g. post dot-com recovery and post-GFC recovery both show accommodative policy and narrowing spreads), causing XGBoost to apply the same label to distinct historical episodes.

**Mitigation:** Anchor windows are chosen from the most historically unambiguous core of each segment, avoiding boundary ambiguity. Documented in 10a as a known limitation for the paper.

### Failure Mode 4 — Pydantic Validation Failure
A computed row may have `regime_confidence` outside [0, 1] due to floating-point edge cases, or an unlabelled month if `regime_names` mapping is incomplete.

**Mitigation:** `RegimeRecord` validates every row before saving. Failed rows are logged with date and error message; pass/fail counts are printed. The JSON only contains validated rows.

### Failure Mode 5 — CRSP Data Gap (2025)
CRSP value-weighted returns are only available through December 2024. The 12 most recent months of `features_df` cannot be validated against market returns.

**Mitigation:** The validation join drops NaN rows silently. Documented in 10a. Not correctable until CRSP releases 2025 data.

### Failure Mode 6 — FRED API Failure
FRED is unreachable or the API key is invalid.

**Mitigation:** `data/storage/fred_macro.parquet` is checked first on every run — no API call is made if the parquet cache exists. If the parquet is missing and the API key is unavailable, a `FileNotFoundError` is raised with clear instructions. On first run, each series is fetched independently in a loop so a single series failure does not block others.

---

## Implementation Notes

- Runs in **Google Colab** with API keys stored via `userdata.get()`
- FRED API key stored as secret `FRED_API` — only needed on first run
- No API keys needed after `data/storage/fred_macro.parquet` exists
- Dependencies: `fredapi`, `xgboost`, `scikit-learn`, `matplotlib`, `ruptures`, `pydantic`, `scipy`, `hmmlearn`, `pyarrow`
- Output files:
  - `agents/research/fred_macro_regimes.csv` — full feature matrix with smoothed regime labels
  - `agents/research/regime_sequence.json` — validated regime sequence passed downstream
  - `agents/research/macro_regime_snapshot.json` — most recent month `MacroRegimeSnapshot`
  - `data/storage/fred_macro.parquet` — cached FRED macro data (fetched once)
  - `data/storage/crsp_market_index.parquet` — cached CRSP returns (converted from CSV once)
  - `data/storage/fred_macro_regimes.parquet` — full feature matrix for downstream agents

---

## Open Questions

- [x] ~~Explore HMM and GMM as alternative regime detection models~~ — resolved June 22: both models evaluated on same 13-feature set. XGBoost retained — only model to recover all 5 academic regimes and correctly identify Inflation Shock as negative-return. HMM flagged as Phase 2 extension for transition probability matrix.
- [ ] Confirm whether to add `ewretd` (equal-weighted CRSP) alongside `vwretd` for validation — equal-weighted returns are more sensitive to small-cap regimes
- [ ] Evaluate whether reducing `pen` in PELT (e.g. pen=5) recovers the 2001 dot-com breakpoint without re-introducing noise breaks
- [ ] Decide whether to hard-gate on `regime_confidence` threshold before passing to Allocation Agent (e.g. only act on regime if confidence > 0.70)
- [ ] Consider adding NBER recession indicator (USREC) as a validation overlay on the smoothed regime timeline
- [ ] Assess whether `crsp_monthly.csv` (individual stock data) should be used downstream in the Allocation Agent for portfolio construction
