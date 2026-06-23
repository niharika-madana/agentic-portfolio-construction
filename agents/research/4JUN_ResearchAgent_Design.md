# Research Agent — Design Document
**AI Financial Advisor Pipeline | Agent 2 of 5**
*khive AI LLC | Fordham MSQF Capstone 2026*

---

## Objective

Scan macroeconomic data from FRED and classify the current market environment into one of five historical regime analogues. The output is a definitive macro regime assessment passed downstream to the Allocation Agent to inform personalised portfolio construction.

---

## Core Logic

The agent uses a **hybrid deterministic + LLM approach**:

1. Pull quantitative macro indicators from FRED
2. Run an **XGBoost classifier** trained on labelled historical months to identify the current regime
3. Use the LLM only for qualitative contextualisation and explanation — not for the classification decision itself

This architecture prevents recency bias and hallucinated regime labels by keeping the classification grounded in real data.

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

| FRED Series | Code | Purpose |
|---|---|---|
| Yield Curve | `T10Y2Y` | 10Y minus 2Y Treasury spread — inversion signals recession |
| Fed Funds Rate | `FEDFUNDS` | Identifies rate hike cycles |
| Unemployment Rate | `UNRATE` | Labour market shocks |
| CPI | `CPIAUCSL` | Inflation tracking |
| Credit Spread | `BAA10Y` | Corporate bond stress indicator |

All series are pulled from **1995–present** and resampled to **monthly frequency**.

---

## Data Sources

### Quantitative (FRED API)
- `T10Y2Y` — Yield curve inversion tracker
- `FEDFUNDS` — Effective Federal Funds Rate
- `UNRATE` — Civilian Unemployment Rate
- `CPIAUCSL` — Consumer Price Index, All Urban Consumers
- `BAA10Y` — Moody's Baa Corporate Bond Yield vs 10Y Treasury

### Qualitative (Unstructured, for LLM ingestion)
- **FOMC Meeting Minutes** — via RSS feed; used for sentiment parsing to supplement quantitative regime signals
- **Bloomberg Terminal** — Implied Volatility (VIX history) and corporate bond yield spreads across credit tiers (AAA to High Yield)

---

## Output Schema (JSON)

```json
{
  "current_macro_regime": "Rate_Hike_Cycle",
  "analogous_historical_regime": "2022_Inflation_Response",
  "regime_confidence_score": 0.85,
  "macro_indicators": {
    "yield_curve_status": "Inverted",
    "fed_funds_rate": 5.33,
    "unemployment_rate": 3.7,
    "cpi_trend": "Increasing",
    "credit_spread_bps": 180
  },
  "llm_context": "The current environment closely mirrors the 2022–2023 rate hike cycle, characterised by aggressive Fed tightening in response to post-pandemic inflation. Historical analogue suggests elevated duration risk and sector rotation away from long-duration growth equities."
}
```

---

## Regime Detection Model — XGBoost Classifier

### Step 1 — Label Training Data
Manually assign regime labels to known historical date ranges. These serve as ground truth.

```python
def assign_regime(date):
    if pd.Timestamp('2000-03-01') <= date <= pd.Timestamp('2002-10-31'):
        return 0  # Dot-com
    elif pd.Timestamp('2007-10-01') <= date <= pd.Timestamp('2009-06-30'):
        return 1  # GFC
    elif pd.Timestamp('2020-02-01') <= date <= pd.Timestamp('2020-12-31'):
        return 2  # COVID
    elif pd.Timestamp('2022-03-01') <= date <= pd.Timestamp('2023-07-31'):
        return 3  # Rate Hike
    elif pd.Timestamp('2023-08-01') <= date <= pd.Timestamp('2024-12-31'):
        return 4  # AI Boom
    else:
        return -1  # Unlabelled / normal
```

Months labelled `-1` (non-regime periods) are excluded from training but receive a predicted label when the model is applied to the full dataset.

### Step 2 — Train Classifier

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

features = ['yield_curve', 'fed_funds', 'unemployment', 'cpi', 'credit_spread']
model = XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### Step 3 — Predict and Visualise
Apply the trained model across all monthly observations (1995–present) and produce a colour-coded timeline of predicted regimes.

---

## Failure Modes & Validation Plan

### Failure Mode 1 — Recency Bias
The LLM may label routine market corrections as systemic crises, or over-weight recent events in its regime assessment.

**Validation:** XGBoost classifier is the authoritative source for regime classification. The LLM is permitted only to explain the regime — not to override or reframe it.

### Failure Mode 2 — Hallucinated Historical Analogues
The LLM may reference historical regimes that don't match the quantitative data, fabricating similarity where none exists.

**Validation:** Hybrid approach — regime classification is fully deterministic (XGBoost output). Any LLM-generated analogue description is cross-checked against the model's `regime_confidence_score`. If score is below **0.6**, the output is flagged as low-confidence and the LLM context block is suppressed.

### Failure Mode 3 — Regime Boundary Ambiguity
The XGBoost model is trained on manually defined date boundaries. If boundaries are drawn incorrectly (e.g. COVID end date too early), the model may misclassify transitional periods.

**Validation:** Evaluate model using `classification_report` on the held-out 20% test split. Target F1 ≥ 0.90 per regime class before using the model in the pipeline. Flag borderline months (confidence < 0.70) in the output schema.

### Failure Mode 4 — Stale FRED Data
If the FRED API pull fails or returns stale data, the regime classification will be based on outdated macro conditions.

**Validation:** Log the `observation_end` date of each FRED series on every pull. If any series is more than 45 days stale relative to the current date, raise a `DataStalenessError` and halt the pipeline.

---

## Implementation Notes

- Runs in **Google Colab** with FRED API key stored via `userdata.get('FRED_API')`
- Macro data saved to `data/processed/fred_macro.csv`
- Regime-labelled output saved to `data/processed/fred_macro_with_regimes.csv`
- XGBoost model to be serialised and saved to `models/regime_classifier.pkl` for reuse across pipeline runs

---

## Open Questions

- [ ] Test Bloomberg and Gemini API access (noted as pending in June 1 meeting)
- [ ] Decide on final date boundaries for the "AI Boom" regime as more data becomes available
- [ ] Evaluate whether HMM should be implemented as a fallback or benchmark model alongside XGBoost
- [ ] Determine how to handle the unlabelled months (`regime = -1`) in evaluation — should they form a sixth "Normal" regime class?
- [ ] Confirm whether FOMC RSS feeds are sufficiently up-to-date for real-time regime detection
