# Research Agent — Design Document
**AI Financial Advisor Pipeline | Agent 2 of 5**
*Fordham MSQF Capstone 2026*
*Last updated: 2026-06-30 (June 30 session — PELT parameters exposed as named constants; cluster_segments() refactored with race-condition guard and input validation; HMM/GMM isolated to non-production path; schema alignment notes added; export paths consolidated; pandas pinned to 2.x; Cell 10 segment duration corrected to 19 mo)*

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

All output files must use the paths above. Any new output file must also be registered in `orchestrator/pipeline.py DATA_PATHS`.

> *Ocean (June 11 meeting): "The research agent follows the persona/profile report. Its sole deliverable is a research report, which then becomes the input for the allocation–proposal–risk loop."*

---

## Pipeline

```
data/storage/fred_macro.parquet   ← read (fetched by FRED API on first run, cached thereafter)
    ↓
Feature engineering (z-scores, log-transforms, first differences, YoY changes)
    ↓
PELT change-point detection (ruptures, PELT_MODEL="rbf", PELT_PEN=10) — structural breaks without date assumptions
    ↓
Segment fingerprinting → cluster_segments() [diagnostic only, not consumed downstream]
    ↓
XGBoost regime mapping (anchor windows → academic taxonomy labels)
    ↓
6-month rolling majority vote (regime smoothing)
    ↓
[NOT IN PRODUCTION PATH] HMM & GMM comparison — benchmarked and rejected; kept in notebook for Future Work section only
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

> **Note:** T5YIE begins 2003, T10YIE begins 2004. After `dropna()`, the usable dataset starts ~2003-02, trimming the pre-2003 period (dot-com bust onset). Total rows after dropna: 274 monthly observations (2003-02 through 2025-12).

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

**Algorithm:** PELT (Pruned Exact Linear Time) via `ruptures` library.

**Parameters** (exposed as named constants in `run_research_agent()` so callers can tune them):

```python
PELT_PEN   = 10     # regularisation penalty — higher → fewer breakpoints (coarser segmentation)
                    # lower → more breakpoints (finer; may over-segment on noisy data)
PELT_MODEL = "rbf"  # kernel — "rbf" captures non-linear signal shifts
                    # alternatives: "l1" (robust to outliers), "l2" (fast, sensitive to outliers)
```

**Result (June 2026 run):** 4 structural breaks detected, dividing 2003–2025 into 5 segments.

| Break Date | Macro Context |
|-----------|--------------|
| 2004-09 | Fed begins hiking from post dot-com lows; credit spreads tightening |
| 2008-01 | Credit spreads spike; yield curve collapses; unemployment turns higher |
| 2014-09 | Fed tapering QE; normalising rates; VIX subdued |
| 2020-12 | COVID shock absorbed; CPI beginning to surge; Fed still at zero |

### Resulting Segments (June 2026 Run)

| # | Period | Duration | Macro Character |
|---|---|---|---|
| 1 | 2003-02 → 2004-08 | ~19 mo | Post dot-com recovery, rates low, spreads tightening |
| 2 | 2004-09 → 2008-01 | ~40 mo | Credit boom, rising rates, low unemployment, low VIX |
| 3 | 2008-01 → 2014-09 | ~80 mo | GFC crisis + ZIRP recovery, elevated spreads, high unemployment |
| 4 | 2014-09 → 2020-12 | ~75 mo | Fed normalisation, low volatility, stable breakevens |
| 5 | 2020-12 → 2025-12 | ~60 mo | Post-COVID inflation surge, aggressive tightening, AI-driven recovery |

With 13 features (vs. 6 previously), PELT requires a larger multivariate shift to trigger a break — filtering out minor transitions and retaining only the four most economically significant structural shifts.

---

## Clustering — `cluster_segments()` [Diagnostic Only]

**Algorithm:** K-means on segment-level fingerprints (mean, variance, delta per signal column).

**Optimal K selection:** Elbow method (WCSS vs K) + silhouette score, K constrained to [2, N_segments − 1].

**Result:** K=2 selected (max silhouette). With only 5 segments, silhouette monotonically favours fewer clusters — this is expected, not a failure.

### `cluster_segments()` Function Contract

```python
def cluster_segments(features_df, signal_cols, break_dates) -> pd.DataFrame:
```

- **Returns:** A segment-level DataFrame with a `cluster` column added. **The return value is for downstream diagnostics only — it is NOT consumed by `run_research_agent()`.** XGBoost uses the full monthly feature matrix, not segment fingerprints.
- **Thread safety:** Protected by a module-level `threading.Lock()` — safe to call from parallel pipelines.
- **Input validation:** Logs the offending condition and raises `ValueError` (before any clustering work) if:
  - the input feature matrix is empty
  - any `signal_cols` entry is absent from the matrix
  - any signal column contains NaN values (offending columns named in the message)
  - any `break_dates` entry is not in the feature index, or break indices are non-monotonic
  - fewer than 2 segments result (cannot run K-means)

The coarse K=2 binary split (Cluster 0: stress/low-growth; Cluster 1: expansion/tightening) is resolved into the 5-regime academic taxonomy by XGBoost in the next step, which operates on the full 274-row monthly feature matrix.

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
|------|---------| -----------|
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

> ⚠️ **NOT IN PRODUCTION PATH.** This section documents the June 22 meeting action item results. HMM/GMM code is excluded from `run_research_agent()` and from `agents/research/research_agent.py`. It is retained in the notebook as a separate manually-executed section for the paper's Future Work discussion only.

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

Field names and types here are the authoritative schema. The root `contracts.py` is the single definition site — the per-agent `contracts.py` files this section used to reference (carrying `MacroContextForRisk` / `MacroContextForAllocation`) no longer exist, and neither do those types; a rename now needs updating only in `contracts.py` and its consumers. `VALID_REGIME_LABELS` is the single source of truth for regime label strings — no inline string literals elsewhere.

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
Date-keyed JSON of all validated monthly regime records. June 2026 run: 274 rows, 0 validation failures.

```json
{
  "2025-12-01": {
    "regime_label":      "Late-Cycle Expansion",
    "prior_regime":      "Late-Cycle Expansion",
    "regime_shift_date": "2023-08-01",
    "regime_confidence": 0.902,
    "regime_volatility": 0.0549,
    "yield_curve":       0.71,
    "term_spread":       0.51,
    "fed_funds":         3.72,
    "unemployment":      4.4,
    "cpi":               2.6533,
    "credit_spread":     1.72,
    "vix":               14.95,
    "indpro":            1.163
  }
}
```

### macro_regime_snapshot.json — Most Recent Month
Single `MacroRegimeSnapshot` object passed to the orchestrator. June 2026 run (as_of 2025-12-01):

```json
{
  "as_of":                  "2025-12-01",
  "regime_label":           "Late-Cycle Expansion",
  "prior_regime":           "Late-Cycle Expansion",
  "regime_shift_date":      "2023-08-01",
  "regime_confidence":      0.902,
  "regime_volatility":      0.0549,
  "yield_curve":            0.71,
  "term_spread":             0.51,
  "fed_funds":              3.72,
  "unemployment":           4.4,
  "cpi":                    2.6533,
  "credit_spread":          1.72,
  "is_low_confidence":      false,
  "regime_change_detected": false
}
```

---

## CRSP Validation Results (June 2026 Run)

Regime labels validated against CRSP value-weighted market returns (2003–December 2024). 2025 months excluded — CRSP data not yet publicly available.

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

**Mitigation:** By design — `cluster_segments()` is a diagnostic sanity check, not the labelling mechanism. XGBoost with 5 anchor windows performs the fine-grained labelling on the full monthly matrix.

### Failure Mode 3 — Regime Label Bleed
Macro fingerprints from different historical periods may be statistically similar, causing XGBoost to apply the same label to distinct historical episodes.

**Mitigation:** Anchor windows are chosen from the most historically unambiguous core of each segment, avoiding boundary ambiguity. Documented as a known limitation for the paper.

### Failure Mode 4 — Pydantic Validation Failure
A computed row may have `regime_confidence` outside [0, 1] due to floating-point edge cases, or an unlabelled month if `regime_names` mapping is incomplete.

**Mitigation:** `RegimeRecord` validates every row before saving. Failed rows are logged with date and error message; pass/fail counts are printed. The JSON only contains validated rows. June 2026 run: 274 passed, 0 failed.

### Failure Mode 5 — CRSP Data Gap (2025)
CRSP value-weighted returns are only available through December 2024. The 12 most recent months of `features_df` cannot be validated against market returns.

**Mitigation:** The validation join drops NaN rows silently. Documented. Not correctable until CRSP releases 2025 data.

### Failure Mode 6 — FRED API Failure
FRED is unreachable or the API key is invalid.

**Mitigation:** `data/storage/fred_macro.parquet` is checked first on every run — no API call is made if the parquet cache exists. If the parquet is missing and the API key is unavailable, a `FileNotFoundError` is raised with clear instructions.

### Failure Mode 7 — cluster_segments() Race Condition
Multiple pipeline workers writing segment fingerprints simultaneously could produce partial writes or corrupt state.

**Mitigation:** `cluster_segments()` is protected by `threading.Lock()`. Input validation raises `ValueError` with descriptive messages before any write occurs if: (a) the input DataFrame is empty, (b) any feature column contains NaN, or (c) fewer than 2 segments are present.

---

## Rebalance Evaluation — Justified or Churn? (Week 9 Deliverable)

Detection alone is not the deliverable. `MacroRegimeSnapshot.regime_change_detected` is a one-line string comparison (`regime_label != prior_regime`) and is **not a trading signal**. `agents/research/rebalance.py` answers the question that matters: should a detected change move the client's book?

### The problem, measured

The smoothed 1995–2025 sequence contains **18 regime runs with a median length of 5 months; 10 of the 18 last 6 months or fewer.** Between 2011-11 and 2013-10 the label flips six times, with runs of 2, 1, 3, 1, 2 and 3 months. A system rebalancing on `regime_change_detected` would have turned the book over six times in twenty months inside what is structurally one stretch of post-GFC normalisation.

### Five gates

Four persona-independent evidence gates, then one persona-conditioned materiality gate. All deterministic; no LLM participates.

| Gate | Test | Threshold |
|------|------|-----------|
| `persistence` | months the new label has held | `MIN_DWELL_MONTHS = 3` |
| `confidence` | mean XGBoost probability across the run | `CONFIDENCE_FLOOR = 0.60` |
| `structural_break` | a PELT break sits near the shift date | `BREAK_TOLERANCE_MONTHS = 6` |
| `reversion_base_rate` | how often this transition historically reverted | `MAX_REVERSION_RATE = 0.50` |
| `materiality` | the client's own equity target actually moves | `MIN_MATERIAL_DELTA = 0.02` |

`structural_break` does the most work. PELT runs on the multivariate signal matrix and never sees the XGBoost labels, so a label flip with no break behind it is a classifier wobble inside a stable macro environment rather than a change of regime. This independence is what makes the gate informative.

`reversion_base_rate` uses the specific prior→current transition when it has at least `MIN_TRANSITION_SAMPLE = 2` prior occurrences, otherwise falls back to the destination regime's short-run frequency, and reports the base rate as unavailable below `MIN_DESTINATION_SAMPLE = 3` prior runs. Without that floor a single short prior run yields a 100% reversion rate and vetoes the March 2008 move into Financial Crisis & ZLB — a transition no one would call churn.

### The regime tilt is derived, not declared

`agents/research/regime_returns.py` sizes the tilt from CRSP value-weighted returns joined to the regime sequence (263 overlapping months):

| Regime | n | ann. mean | SE(mean) | ann. vol | vol ratio | tilt |
|--------|---|-----------|----------|----------|-----------|------|
| Early Recovery | 31 | 20.1% | 5.0% | 8.1% | 1.87 | +15.0% (capped) |
| Late-Cycle Expansion | 61 | 16.7% | 4.6% | 10.5% | 1.44 | +13.3% |
| Moderate Expansion | 82 | 13.4% | 5.6% | 14.7% | 1.03 | +0.9% |
| Inflation Shock | 27 | −3.4% | 12.0% | 18.0% | 0.84 | −4.8% |
| Financial Crisis & ZLB | 62 | 9.5% | 8.9% | 20.1% | 0.75 | −7.5% |

**The tilt uses volatility only, and deliberately ignores the means.** Standard errors of 5–12% straddle almost the entire 23-point spread between the highest and lowest regime mean — the means are not statistically separable. Feeding them into a Merton weight produces w = 9.2 for Early Recovery (920% equity), which is estimation error, not signal. The volatilities *are* separable: Financial Crisis & ZLB realises 2.5× the volatility of Early Recovery. So the tilt holds the risk premium fixed at the full-sample estimate and scales equity by `sigma_base / sigma_regime` — the constant-Sharpe form of volatility targeting — then squashes the result into `±TILT_CAP`. One of five regimes hits the cap, and capping is reported rather than hidden.

### Harness result

`agents/research/rebalance_backtest.py` replays the evaluator over all 16 completed transitions, committing at `DECISION_LAG = 3` months after each shift date and scoring against whether the destination regime went on to hold ≥ 12 months.

| | evaluator | always-act baseline |
|---|---|---|
| accuracy | **81.2%** | 31.2% |
| churn avoided | **90.9%** (10 of 11) | 0% |
| durable captured | 60.0% (3 of 5) | 100% |
| precision | **75.0%** | 31.2% |
| trades taken | **4** | 16 |

The entire 2011–2013 churn cluster is correctly declined. Cost: two durable transitions missed (2003-06 Early Recovery, 2022-01 Inflation Shock) and one false positive (2021-05 Inflation Shock, which lasted 3 months).

**Disclosed leak:** PELT and XGBoost are fitted on the full sample upstream, so the label path and break locations at date *t* embed post-*t* information. The gates themselves are truncated point-in-time in `build_evidence()`, but that upstream leak remains and flatters `structural_break` and `confidence`. These numbers are an upper bound on live performance; a walk-forward refit is future work.

### The corner-solution finding

The materiality gate applies the tilt to the client's own `portfolio_equity_target`, converted from total-wealth to financial-wealth units by `× (financial + human) / financial` and clipped to the achievable [0, 1] portfolio range.

**Every one of the nine current BLS personas sits at a corner.** Human capital is 6–29× financial capital, so a total-wealth target of +0.48 maps to +8.9 in financial units and −0.25 maps to −6.7; both saturate. No regime tilt of any size moves these books — the constraint sets the allocation, not the regime. Regime-conditioned rebalancing only bites once financial capital is comparable to human capital (roughly fc ≳ hc, reached around $2M financial against $2.3M human for the equity-like persona, where the same transition produces a −9.2pp equity move and a JUSTIFIED verdict).

This is a result, not a bug, and it bears directly on the sensitivity question in *Now and Forward* §5: for asset-poor clients the intake precision of income variability cannot move the portfolio, because the portfolio is pinned at a bound regardless.

---

## Ticker Betas (24 Jul action item)

**"Implement beta-calculation routine (no weights) and produce ticker-beta table for review."**

`agents/research/ticker_betas.py` → `data/outputs/ticker_betas.{json,csv}`. Weights are deliberately not computed: the module estimates and reports exposures so beta caps and overlap constraints can be set from measured numbers, and the optimiser stays where it is.

Two regressions per ticker on monthly CRSP total returns, 25 years:

```
CAPM      r_i − rf = alpha + beta_mkt * (r_m − rf) + e     (Fama-French mktrf)
vs SPY    r_i − rf = alpha + beta_spy * (r_spy − rf) + e
```

`beta_spy` and its R² are the direct evidence for review §3 — that the book is diversified by ticker count and not by underlying exposure:

| ticker | n | β_mkt | SE | β_SPY | R²_SPY | resid vol |
|---|---|---|---|---|---|---|
| XLK | 300 | 1.248 | 0.041 | 1.281 | 0.745 | 11.5% |
| IWM | 295 | 1.166 | 0.031 | 1.163 | 0.765 | 9.8% |
| XLI | 300 | 1.070 | 0.030 | 1.124 | **0.828** | 7.8% |
| XLY | 300 | 1.068 | 0.035 | 1.113 | 0.769 | 9.4% |
| XLC | 78 | 0.967 | 0.058 | 1.005 | 0.785 | 9.2% |
| EFA | 280 | 0.945 | 0.033 | 0.978 | 0.755 | 8.4% |

**7 of 24 holdings sit at R² ≥ 0.70 against SPY.** A sector fund with β_SPY ≈ 1.3 and R² ≈ 0.75 is not adding a risk factor to a book that already holds SPY — it is levering the one already there. XLI at R² 0.83 is the clearest case.

Standard errors are reported next to every beta, because a cap set from a point estimate with a wide standard error is a cap set on noise. `MIN_MONTHS = 36` flags thin histories rather than hiding them — XLC (78 months) and XLRE (110) rest on far less data than SPY's 300.

Two notes for readers of the table. SPY's own `beta_mkt` is 0.954, not 1.0, because `mktrf` is the CRSP value-weighted market — broader than the S&P 500 — so SPY genuinely has slightly less than unit exposure to it. And the bond sleeve prices as expected: TLT −0.107, IEF −0.053, SHY −0.014, all with R² below 0.02, which is the diversification the equity sleeve is not providing.

---

## Regime → Asset-Class Sleeve Tilts (24 Jul decision 5)

The minutes asked to "map each regime to **predefined** asset-class tilts (e.g. increase bond weight in Inflation Shock)". `agents/research/regime_sleeves.py` **derives** them instead — a declared table would be the same unfalsifiable pattern the review objected to in the stress drawdowns. Same construction as the equity tilt: bounded `sigma_baseline / sigma_regime` per sleeve, on equal-weighted monthly CRSP returns for the ETFs in each sleeve.

| sleeve | Early Recovery | Financial Crisis & ZLB | Inflation Shock | Late-Cycle | Moderate Exp. |
|---|---|---|---|---|---|
| broad_equity | +0.150 | −0.088 | −0.020 | +0.128 | +0.034 |
| sector_equity | +0.150 | −0.071 | −0.047 | +0.125 | +0.002 |
| credit (HYG) | +0.150 | **−0.103** | +0.017 | +0.145 | +0.144 |
| fixed_income | −0.005 | −0.021 | **−0.089** | +0.017 | +0.081 |
| real_assets | +0.109 | −0.084 | +0.001 | +0.036 | +0.082 |
| cash | 0 | 0 | 0 | 0 | 0 |

**The data contradicts the example in the minutes.** Fixed income is the *worst* sleeve in Inflation Shock (−0.089, the most negative tilt in that column), not the shelter. That is economically right — 2022 is in the sample, and bonds fell alongside equities precisely because the shock was inflationary. "Increase bond weight in Inflation Shock" is the intuition a declared table would have encoded, and it would have been wrong.

Two more results worth keeping. **Credit is the most negative sleeve in the crisis** (−0.103, worse than broad equity), which is the correct read on high yield: HYG behaves like equity exactly when a bond sleeve is supposed to help, which is why it is split out of `fixed_income` rather than averaged into it. And **broad and sector equity are kept apart** so the mapping can express a preference between them — review §3 objects to layering one on the other, and a merged sleeve could not say so.

**Cash is held at zero tilt by construction** (`NON_TILTABLE`). BIL has almost no volatility in any regime, so the vol ratio is two near-zero numbers divided — it returned 9.4 in Early Recovery and gave cash a maximum positive tilt in the calmest regime *and* in the financial crisis. That is a divide-by-nearly-zero, not a signal. Cash is sized by the optimiser's cash floor and by what the risky sleeves leave behind.

Each tilt is reported next to the standard error of the mean it rests on, and flagged `mean_supports_tilt` when the return evidence agrees with the volatility evidence by more than one SE. Only 5 of 30 clear that bar — the rest rest on volatility alone, which is not disqualifying but should be visible.

**Scope:** this module produces the mapping only. Consuming it in the optimiser is review §4.2 and lives in `agents/allocation`. `regime_sleeve_tilts()` returns a plain `{regime: {sleeve: tilt}}` dict, also persisted to `data/outputs/regime_sleeve_tilts.json`.

---

## Snapshot Surface — Deprecation (24 Jul Research action)

The 24 Jul minutes ask this contract to **"expose only the fields needed for allocation (label, confidence, volatility)."** It currently exposes 15.

Verified by grep across `agents/risk`, `agents/allocation`, `agents/compliance` and `agents/orchestrator`: `yield_curve`, `term_spread`, `fed_funds`, `unemployment`, `cpi` and `credit_spread` are consumed in **zero** places outside this agent. The action item is correct — they are dead weight on a public contract and a schema-drift risk. (The "Downstream Agent Field Routing" section above claims Risk and Compliance read some of them. That table is aspirational; the code does not.)

**They are deprecated, not removed.** `tests/test_research.py` and `tests/test_compliance.py` construct snapshots with all six (5–6 references each), and `tests/` is outside the Profile/Research edit scope. Deleting them would break the suite with no way to fix it.

Use **`snapshot.for_allocation()`** instead — it returns the intended minimal view:

```python
{"regime_label": ..., "regime_confidence": ..., "regime_volatility": ...}
```

Code written against `for_allocation()` will not need changing when the deprecated fields are finally deleted, which is a one-line-per-field edit once the test fixtures can be updated. Nothing is lost by dropping them: the full signal matrix remains in `RegimeRecord` and in `data/outputs/fred_macro_regimes.csv`.

### Consumption

`MacroRegimeSnapshot.regime_change_evidence` carries the persona-independent block and flows into `ComplianceInput` and `AdvisorPackage` automatically. The persona-conditioned verdict comes from `evaluate_rebalance(snapshot, profile, regime_stats)` and is the natural source for `RegimeChangeFlag.rebalance_proposed`, which the contract has always declared and no agent has ever populated.

---

## Implementation Notes

- Runs in **Google Colab** with API keys stored via `userdata.get()`
- FRED API key stored as secret `FRED_API` — only needed on first run
- No API keys needed after `data/storage/fred_macro.parquet` exists
- **Environment:** pandas pinned to `>=2.0,<3.0` across all agents (pandas 3.x caused parquet load errors — team decision June 25 2026)
- Dependencies: `fredapi`, `xgboost`, `scikit-learn`, `matplotlib`, `ruptures`, `pydantic`, `scipy`, `pyarrow`, `"pandas>=2.0,<3.0"`
- `hmmlearn` — installed but **not imported in the production path**; only used in the manually-executed HMM/GMM comparison section of the notebook
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
