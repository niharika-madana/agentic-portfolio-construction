"""Run this script to regenerate the session PDF: python generate_doc.py"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

OUTPUT = "Capstone_Session_1_Documentation.pdf"

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

NAVY   = colors.HexColor("#1B2A4A")
BLUE   = colors.HexColor("#2E5EAA")
LBLUE  = colors.HexColor("#EAF0FB")
GREY   = colors.HexColor("#F5F5F5")
DKGREY = colors.HexColor("#444444")
GREEN  = colors.HexColor("#1A6B3C")

h1 = ParagraphStyle("H1", parent=styles["Heading1"],
    fontSize=20, textColor=NAVY, spaceAfter=8, spaceBefore=20,
    fontName="Helvetica-Bold")
h2 = ParagraphStyle("H2", parent=styles["Heading2"],
    fontSize=14, textColor=BLUE, spaceAfter=6, spaceBefore=14,
    fontName="Helvetica-Bold")
h3 = ParagraphStyle("H3", parent=styles["Heading3"],
    fontSize=11, textColor=NAVY, spaceAfter=4, spaceBefore=10,
    fontName="Helvetica-Bold")
body = ParagraphStyle("Body", parent=styles["Normal"],
    fontSize=9.5, leading=14, textColor=DKGREY,
    alignment=TA_JUSTIFY, spaceAfter=6)
code = ParagraphStyle("Code", parent=styles["Code"],
    fontSize=8, leading=12, fontName="Courier",
    backColor=GREY, borderPadding=(4, 4, 4, 4),
    textColor=colors.HexColor("#222222"), spaceAfter=6)
bullet = ParagraphStyle("Bullet", parent=body,
    leftIndent=18, bulletIndent=6, spaceAfter=3)
caption = ParagraphStyle("Caption", parent=body,
    fontSize=8, textColor=colors.HexColor("#666666"), alignment=TA_CENTER)
title_style = ParagraphStyle("Title", parent=styles["Title"],
    fontSize=26, textColor=NAVY, alignment=TA_CENTER,
    fontName="Helvetica-Bold", spaceAfter=6)
subtitle_style = ParagraphStyle("Sub", parent=styles["Normal"],
    fontSize=12, textColor=BLUE, alignment=TA_CENTER, spaceAfter=4)

def HR(): return HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=8, spaceBefore=4)
def B(t): return f"<b>{t}</b>"
def I(t): return f"<i>{t}</i>"
def C(t): return f"<font name='Courier' size='8'>{t}</font>"

def section_table(rows, col_widths=None):
    """Styled two-column table for key/value sections."""
    style = TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LBLUE]),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0),(-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ])
    t = Table(rows, colWidths=col_widths or [2.2*inch, 4.4*inch])
    t.setStyle(style)
    return t

def code_block(text):
    lines = [Paragraph(line.replace(" ", "&nbsp;").replace("<","&lt;").replace(">","&gt;"),
                        code) for line in text.strip().split("\n")]
    return lines

# ── Document content ──────────────────────────────────────────────────────────
story = []

# Cover page
story += [
    Spacer(1, 1.2*inch),
    Paragraph("Multi-Agent Portfolio Construction System", title_style),
    Paragraph("Session 1 — Deterministic Core: Complete Technical Documentation", subtitle_style),
    Spacer(1, 0.2*inch),
    Paragraph("Aidan Altorelli &nbsp;|&nbsp; Fordham MSQF Capstone &nbsp;|&nbsp; June 2026", subtitle_style),
    Spacer(1, 0.4*inch),
    HR(),
    Spacer(1, 0.2*inch),
    Paragraph(
        "This document covers every file built in Session 1: what each module does, "
        "why the design choices were made, the mathematical foundations behind each "
        "function, and how the pieces connect. It is intended as a complete reference "
        "that could be handed to a new engineer or a thesis committee member.",
        body),
    PageBreak(),
]

# ── 1. SYSTEM OVERVIEW ────────────────────────────────────────────────────────
story += [
    Paragraph("1. System Overview", h1), HR(),
    Paragraph(
        "The system is a <b>multi-agent LLM pipeline</b> for personalized portfolio "
        "construction that treats human capital as an asset on the investor's total "
        "economic balance sheet. The three-agent pipeline is:", body),
    Paragraph("• <b>Allocation Agent</b> — builds Black-Litterman optimized weights", bullet),
    Paragraph("• <b>Risk Agent</b> — evaluates those weights against deterministic thresholds", bullet),
    Paragraph("• <b>Compliance Agent</b> — final sign-off (not yet built)", bullet),
    Spacer(1, 0.1*inch),
    Paragraph(B("Core architectural rule: the LLM is a pure interface."), h3),
    Paragraph(
        "Every number the system produces is computed in code inside "
        f"{C('core/')}. The LLM's only permitted jobs are: (1) validate inputs, "
        "(2) route data to the right function, (3) render the deterministic output "
        "as readable English. It never invents or modifies a number. "
        "This is what makes the system fiduciary-defensible and auditable.", body),
    Spacer(1, 0.1*inch),
    Paragraph("File layout built in this session:", h3),
    section_table([
        ["File / Folder", "Purpose"],
        [C("portfolio_system/schemas.py"),      "Pydantic data contracts for every agent handoff"],
        [C("portfolio_system/data/loaders.py"), "WRDS pulls: CRSP, Compustat, Fama-French"],
        [C("core/human_capital.py"),            "BMS (1992) human-capital math — shared by both agents"],
        [C("core/constraints.py"),              "Single source of truth for all four concentration limits"],
        [C("core/allocation.py"),               "Black-Litterman optimizer + equilibrium returns"],
        [C("core/risk.py"),                     "Rule engine: VaR/CVaR, drawdown, stress, decision logic"],
        [C("tests/test_human_capital.py"),      "Unit tests for human_capital.py (17 tests)"],
        [C("tests/test_constraints.py"),        "Unit tests for constraints.py (16 tests)"],
        [C("tests/test_allocation.py"),         "Unit tests for allocation.py (17 tests)"],
        [C("tests/test_risk.py"),               "Unit tests for risk.py (29 tests)"],
        [C("pyproject.toml"),                   "Makes the project pip-installable; editable install active"],
    ], col_widths=[2.8*inch, 3.8*inch]),
    PageBreak(),
]

# ── 2. SCHEMAS ────────────────────────────────────────────────────────────────
story += [
    Paragraph("2. schemas.py — Pydantic Data Contracts", h1), HR(),
    Paragraph(
        f"{C('schemas.py')} is the single file every other module imports types from. "
        "It defines what data looks like at every boundary in the system: "
        "user inputs, allocation outputs, risk outputs, and the FLAG-loop feedback. "
        "Using Pydantic means bad data is rejected at the boundary, not silently "
        "propagated into core math.", body),

    Paragraph("Enums", h2),
    Paragraph(
        f"{C('RiskProfile')} (Conservative / Moderate / Aggressive) — maps to "
        "drawdown caps and risk-aversion coefficients used throughout the system. "
        f"{C('RiskDecision')} (APPROVE / FLAG / REJECT) — the three outcomes of the "
        f"Risk Agent. {C('StressSeverity')} (LOW / MEDIUM / HIGH / CRITICAL) — "
        "grades how badly a stress scenario hurts relative to the investor's own "
        f"drawdown cap. {C('ConstraintType')} — labels what kind of concentration "
        "rule was violated, so the FLAG feedback loop knows how to tighten the "
        "optimizer.", body),

    Paragraph("User Inputs", h2),
    section_table([
        ["Class", "What it holds"],
        [C("HumanCapitalInput"),  "PV of future labor income, employer ticker/sector, income volatility, income beta (correlation with market), years to retirement, discount rate"],
        [C("UserProfile"),        "Financial wealth W, the HumanCapitalInput, and risk profile"],
        [C("InstrumentUniverse"), "Pre-approved ticker list, their market-cap weights (for BL prior), and GICS sector mapping"],
        [C("AllocationInput"),    "UserProfile + universe + flag_constraints (empty on first run, populated on FLAG re-entry) + flag_iteration counter"],
    ]),
    Spacer(1, 0.1*inch),

    Paragraph("Allocation Output (→ Risk Agent)", h2),
    section_table([
        ["Class", "What it holds"],
        [C("WeightDecomposition"), "Per-ticker: total_weight, equilibrium_baseline, view_tilt, human_capital_offset — breaks each weight into its three sources"],
        [C("FactorExposures"),     "market_beta, smb, hml, mom — Fama-French + momentum betas from OLS regression"],
        [C("PortfolioStatistics"), "expected_return, volatility, sharpe_ratio, factor_exposures — all computed in core/"],
        [C("AllocationOutput"),    "The full handoff: weights list (validated to sum to 1.0), risky_weight, safe_weight, portfolio stats, rationale string (LLM fills this in)"],
    ]),
    Spacer(1, 0.1*inch),

    Paragraph("Why the two model_validators on AllocationOutput?", h3),
    Paragraph(
        "The weights-sum-to-one check is a mathematical invariant, not a business "
        "rule. Pydantic validators run on every construction, so if a bug in "
        f"{C('core/allocation.py')} produces weights that sum to 0.99, the schema "
        "catches it immediately with a clear error rather than silently propagating "
        "a wrong portfolio through the rest of the pipeline.", body),

    Paragraph("Risk Output (→ Compliance)", h2),
    section_table([
        ["Class", "What it holds"],
        [C("VaRMetrics"),                  "var_95, var_99, cvar_95, cvar_99 — daily VaR and CVaR at both Basel confidence levels"],
        [C("ConcentrationFlags"),          "Lists of breached tickers/sectors and a boolean for employer breach — populated by constraints.py"],
        [C("StressResult"),                "One row per scenario: name, portfolio_loss, threshold_breached, severity"],
        [C("HumanCapitalAdjustedMetrics"), "The HC balance-sheet view: total_wealth, hc_fraction, effective equity exposure, economic sector exposures, employer concentration"],
        [C("RiskMetrics"),                 "All metrics in one object: vol, VaR/CVaR, MDD, factor exposures, liquidity, concentration flags, HC metrics, stress results"],
        [C("RiskOutput"),                  "Decision + AllocationOutput + RiskMetrics + reasoning_trace + constraints_violated + flag_iteration"],
        [C("AllocationConstraint"),        "One violated constraint: type, target (ticker or sector), current_value, limit — the feedback the Risk Agent sends to Allocation on FLAG"],
    ]),
    PageBreak(),
]

# ── 3. DATA LOADERS ───────────────────────────────────────────────────────────
story += [
    Paragraph("3. data/loaders.py — WRDS Data Pulls", h1), HR(),
    Paragraph(
        "This is the only file in the system that touches a database. Every "
        f"function takes a {C('wrds.Connection')} and returns a clean "
        f"{C('pd.DataFrame')} or {C('dict')}. No math happens here — that "
        "separation means every core function is unit-testable with hand-built "
        "fixtures, no database access required in CI.", body),

    section_table([
        ["Function", "WRDS Table", "Returns", "Used by"],
        [C("get_connection()"),             C("—"),                    "Open WRDS connection",                              "Agent layer"],
        [C("load_permno_map()"),            C("crsp.dsenames"),        "DataFrame: ticker → PERMNO",                        "All CRSP callers"],
        [C("tickers_to_permnos()"),         C("crsp.dsenames"),        "dict {ticker: permno}",                             "Convenience wrapper"],
        [C("load_crsp_daily()"),            C("crsp.dsf"),             "Daily ret, price, volume",                          "core/risk.py VaR + stress"],
        [C("load_crsp_monthly()"),          C("crsp.msf"),             "Monthly ret + mktcap",                              "core/allocation.py cov matrix"],
        [C("load_risk_free_rate()"),        C("ff.factors_monthly"),   "pd.Series: date → rf",                              "HC discount rate, BL prior"],
        [C("load_gics_sectors()"),          C("comp.company"),         "dict {ticker: sector_name}",                        "concentration checks"],
        [C("load_fama_french_factors()"),   C("ff.factors_monthly"),   "DataFrame: mktrf, smb, hml, umd, rf",               "BL views + factor regression"],
        [C("load_market_cap_weights()"),    C("crsp.msf"),             "dict {permno: weight}",                             "BL equilibrium prior"],
    ], col_widths=[1.8*inch, 1.5*inch, 1.5*inch, 1.7*inch]),

    Spacer(1, 0.15*inch),
    Paragraph("Key implementation details", h3),
    Paragraph(
        f"<b>PERMNO mapping:</b> CRSP uses its own integer key (PERMNO), not ticker "
        "symbols. Every call to a CRSP table must first map tickers to PERMNOs via "
        f"{C('crsp.dsenames')}. The query filters by {C('namedt')} and {C('nameendt')} "
        "to get the mapping as-of a specific date — tickers change over time.", body),
    Paragraph(
        f"<b>ABS(prc):</b> CRSP stores price as negative when the quote is a "
        "bid-ask midpoint rather than an actual trade price. Every price pull uses "
        f"{C('ABS(prc)')} to normalize this quirk.", body),
    Paragraph(
        f"<b>Risk-free rate source:</b> The spec names 'CRSP Treasury files' but the "
        "implementation uses {C('ff.factors_monthly.rf')} which is the standard "
        "academic source and essentially identical to 30-day T-bill yields.", body),
    PageBreak(),
]

# ── 4. HUMAN CAPITAL ─────────────────────────────────────────────────────────
story += [
    Paragraph("4. core/human_capital.py — BMS Framework", h1), HR(),
    Paragraph(
        "This module implements the Bodie-Merton-Samuelson (1992) human capital "
        "framework extended by Campbell & Viceira (2002). It is the shared "
        "mathematical engine used by both agents: the Allocation Agent uses it to "
        "compute the optimal risky asset weight, and the Risk Agent uses it to "
        "build the total economic balance sheet.", body),

    Paragraph("Mathematical foundations", h2),
    section_table([
        ["Formula", "Source", "What it computes"],
        ["α = (μ − r_f) / (γ · σ²)",       "Merton (1971)",          "Optimal Merton risky share without human capital"],
        ["w_fin = α · (1 + H/W)",           "BMS (1992)",             "Optimal financial risky share when HC is risk-free"],
        ["w_fin = α·(1+H/W) − (H/W)·β_h",  "Campbell & Viceira (2002)", "Extended to risky HC: income beta hedges away equity demand"],
        ["Total wealth = W + H",             "BMS balance sheet",      "Financial wealth plus present value of human capital"],
        ["HC fraction = H / (W + H)",        "BMS balance sheet",      "Human capital as share of total economic wealth"],
    ]),

    Spacer(1, 0.1*inch),
    Paragraph("Function-by-function explanation", h2),

    Paragraph(B("merton_risky_share(expected_excess_return, portfolio_volatility, risk_profile)"), h3),
    Paragraph(
        "Implements the Merton (1971) formula for the optimal fraction of financial "
        "wealth to hold in the risky portfolio, ignoring human capital. "
        f"Risk aversion γ is mapped from {C('RiskProfile')}: Conservative → 5.0, "
        "Moderate → 3.0, Aggressive → 2.0. These are standard academic calibrations. "
        "The result is capped at 1.0 — no leverage in the base case. "
        "This value becomes the input α to the HC-adjusted formulas below.", body),

    Paragraph(B("compute_w_fin(alpha, human_capital_pv, financial_wealth, income_beta)"), h3),
    Paragraph(
        "The central function of the BMS framework. When income_beta = 0 "
        "(labor income is like a bond — certain, uncorrelated with the market), "
        "the investor can afford to take more financial risk because their HC already "
        "acts as a safe asset: w_fin = α · (1 + H/W). When income_beta > 0 "
        "(e.g., a tech worker whose salary correlates with the NASDAQ), the HC "
        "already provides implicit equity exposure, so the optimal financial risky "
        "share is reduced by the hedge term (H/W) · β_h. The result is clipped to "
        "[0, 1]: no short-selling the risky portfolio, no leverage.", body),

    Paragraph(B("employer_concentration(employer_financial_weight, financial_wealth, human_capital_pv)"), h3),
    Paragraph(
        "Computes total economic exposure to the employer as a fraction of total "
        "wealth. Human capital is treated as 100% employer exposure — this is the "
        "conservative, fiduciary-defensible assumption. The 15% limit from the spec "
        "is checked in constraints.py against this output. For a tech executive with "
        "$500k financial wealth and $1M human capital, even holding zero employer "
        "stock gives 67% employer concentration — well above the 15% limit, which "
        "is exactly the structurally-unfixable REJECT case in risk.py.", body),

    Paragraph(B("economic_sector_exposures(financial_weights, sectors, financial_wealth, human_capital_pv, employer_sector)"), h3),
    Paragraph(
        "Builds the total economic balance sheet view of sector exposures. "
        "For each ticker, its financial weight × financial_wealth is tallied by "
        "GICS sector. Then human capital (the full HC present value) is added to the "
        "employer's sector, because labor income is economically equivalent to a long "
        "position in that sector. The result is expressed as fractions of total wealth "
        "(W + H) and sums to 1.0. This is the input to the economic sector "
        "concentration check (25% limit).", body),
    PageBreak(),
]

# ── 5. CONSTRAINTS ────────────────────────────────────────────────────────────
story += [
    Paragraph("5. core/constraints.py — Concentration Limits", h1), HR(),
    Paragraph(
        "This is the single source of truth for all four concentration limits. "
        "The spec explicitly requires this: the same numbers must be used as "
        f"optimizer bounds in {C('allocation.py')} and as evaluation checks in "
        f"{C('risk.py')}. Having them in one file means you change a limit in one "
        "place and both agents automatically use the new value.", body),

    section_table([
        ["Constant", "Value", "Source", "How enforced"],
        [C("SINGLE_NAME_LIMIT"),     "10%", "Russell Investments, CFA Institute", "Optimizer upper bound + risk check"],
        [C("SECTOR_LIMIT"),          "20%", "HDFC TRU, industry practice",        "Optimizer linear constraint + risk check"],
        [C("ECONOMIC_SECTOR_LIMIT"), "25%", "Internal",                            "Risk check only (HC involved, can't optimize directly)"],
        [C("EMPLOYER_LIMIT"),        "15%", "Internal",                            "Risk check only (translated to financial bounds on FLAG)"],
    ]),

    Spacer(1, 0.1*inch),
    Paragraph("Function-by-function explanation", h2),

    Paragraph(B("aggregate_sector_weights(weights, sectors)"), h3),
    Paragraph(
        "Simple utility that sums instrument weights by GICS sector. Used internally "
        "by check_sector and also imported by allocation.py when building the sector "
        "linear constraints for scipy.", body),

    Paragraph(B("check_single_name / check_sector / check_economic_sector / check_employer"), h3),
    Paragraph(
        "Each function runs one concentration check and returns a list of "
        f"{C('AllocationConstraint')} objects — one per violation. The list is empty "
        "when the check passes. All comparisons use strict greater-than (>), not "
        "greater-than-or-equal, because a portfolio sitting exactly at the limit is "
        "compliant. This is the correct interpretation for all four limits.", body),

    Paragraph(B("run_all_checks(weights, sectors, economic_sector_exposures, employer_concentration)"), h3),
    Paragraph(
        f"Runs all four checks and aggregates the results into a {C('ConcentrationFlags')} "
        "object. This is what goes into RiskMetrics — it's the reporting view. "
        "The lists of breached names are for human-readable output (reasoning trace).", body),

    Paragraph(B("all_violations(...)"), h3),
    Paragraph(
        f"Also runs all four checks but returns a flat {C('list[AllocationConstraint]')} "
        "rather than a ConcentrationFlags object. This is the machine-readable view "
        "used to populate RiskOutput.constraints_violated on FLAG decisions, which "
        "then flows into AllocationInput.flag_constraints on re-entry.", body),
    PageBreak(),
]

# ── 6. ALLOCATION ─────────────────────────────────────────────────────────────
story += [
    Paragraph("6. core/allocation.py — Black-Litterman Optimizer", h1), HR(),
    Paragraph(
        "This is the most mathematically complex module. It implements the "
        "Black-Litterman (1990) model, which combines a CAPM-derived equilibrium "
        "prior with Fama-French factor views to produce posterior expected returns, "
        "then optimizes portfolio weights subject to concentration constraints.", body),

    Paragraph("The Black-Litterman pipeline", h2),
    section_table([
        ["Step", "Formula", "Function"],
        ["1. Build covariance",       "Σ = cov(excess_returns) × 12",              C("build_covariance_matrix()")],
        ["2. Equilibrium returns",    "Π = δ · Σ · w_mkt   (δ = 2.5)",             C("compute_equilibrium_returns()")],
        ["3. Factor views",           "OLS: r_i = α_i + B_i·F + ε_i → Q, P, Ω",   C("build_ff_views()")],
        ["4. BL posterior returns",   "μ_BL = A⁻¹[(τΣ)⁻¹Π + P'Ω⁻¹Q]  (τ=0.025)", C("black_litterman()")],
        ["5. BL posterior covariance","Σ_BL = Σ + A⁻¹",                            C("black_litterman()")],
        ["6. Optimize",               "max w'μ − (δ/2)·w'Σw s.t. constraints",    C("optimize_weights()")],
        ["7. HC scaling",             "w_fin from compute_w_fin(); risky/safe split", C("run_allocation()")],
        ["8. Factor exposures",       "OLS: r_p = α + β·F",                         C("compute_factor_exposures()")],
    ]),

    Spacer(1, 0.1*inch),
    Paragraph("Why Black-Litterman instead of mean-variance?", h3),
    Paragraph(
        "Vanilla mean-variance optimization (Markowitz) is notorious for producing "
        "'corner solutions' — portfolios with extreme concentration in a few assets "
        "and zero weight in many others. These solutions are extremely sensitive to "
        "small estimation errors in expected returns. Black-Litterman solves this by "
        "anchoring the prior to the CAPM equilibrium (which is diversified by "
        "construction) and only tilting away from it to the extent that the "
        "investor's views are confident. The result is a stable, diversified "
        "portfolio that is also fiduciary-defensible.", body),

    Paragraph("Why tau = 0.025?", h3),
    Paragraph(
        "τ controls how much weight the model places on the equilibrium prior vs "
        "the investor's views. τ = 0.025 is the value recommended by Walters (2013) "
        "based on empirical calibration. A small τ means high confidence in the "
        "equilibrium prior — conservative, appropriate for a fiduciary system.", body),

    Paragraph("How Fama-French views are constructed", h3),
    Paragraph(
        "For each asset i, OLS is run: r_i = α_i + B_i · F + ε_i, where F contains "
        "the four FF factors (MktRf, SMB, HML, UMD). The view Q_i is the "
        "factor-model-implied expected return: α_i·12 + B_i·λ_F where λ_F are the "
        "annualized mean factor premiums. P = I_N (one absolute view per asset). "
        "Ω = diag(annualized residual variances) / τ — larger residual variance means "
        "less confidence in that stock's view.", body),

    Paragraph("Optimizer constraints and FLAG re-entry", h3),
    Paragraph(
        "The optimizer uses scipy SLSQP with two types of constraints: "
        "(1) box bounds — 0 ≤ w_i ≤ 10% per ticker, (2) linear inequality "
        "constraints — one per GICS sector capping combined sector weight at 20%. "
        "On FLAG re-entry, violated constraints arrive in flag_constraints. "
        "SINGLE_NAME violations tighten that ticker's upper bound to 99% of the "
        "limit. SECTOR violations tighten the sector cap. EMPLOYER violations are "
        "translated to a max financial weight for the employer ticker via "
        "_max_employer_financial_weight(). ECONOMIC_SECTOR violations scale down "
        "the financial sector limit proportional to the financial/total wealth ratio.",
        body),

    Paragraph("Weight decomposition", h3),
    Paragraph(
        "The spec requires each weight to be broken into three sources. "
        "equilibrium_baseline is the weight from optimizing with the CAPM prior Π "
        "only (views set to zero by inflating Ω). view_tilt is the shift caused by "
        "the FF views: w_BL − w_eq. human_capital_offset is the shift caused by "
        "FLAG constraints from the previous Risk Agent run: w_BL_constrained − "
        "w_BL_unconstrained. On the first run with no FLAG constraints all three "
        "of these terms sum to the final weight and the HC offset is zero.", body),
    PageBreak(),
]

# ── 7. RISK ───────────────────────────────────────────────────────────────────
story += [
    Paragraph("7. core/risk.py — Deterministic Rule Engine", h1), HR(),
    Paragraph(
        "The Risk Agent evaluates a portfolio produced by the Allocation Agent "
        "against a fixed set of deterministic rules. It never modifies weights. "
        "Its only output is a decision (APPROVE / FLAG / REJECT) plus the metrics "
        "and reasoning trace that support that decision.", body),

    Paragraph("Risk measures", h2),
    section_table([
        ["Measure", "Method", "Standard"],
        ["VaR 95%",  "Historical simulation: 5th percentile of daily portfolio P&L over last 5 years (1,260 trading days)", "Basel III/IV"],
        ["VaR 99%",  "Historical simulation: 1st percentile", "Basel III/IV"],
        ["CVaR 95%", "Mean of portfolio returns below the 5th percentile (Expected Shortfall)", "Basel III/IV"],
        ["CVaR 99%", "Mean of portfolio returns below the 1st percentile", "Basel III/IV"],
        ["Max Drawdown", "Maximum peak-to-trough loss of cumulative portfolio return", "Uniform Prudent Investor Act"],
    ]),

    Spacer(1, 0.1*inch),
    Paragraph("Why CVaR in addition to VaR?", h3),
    Paragraph(
        "VaR tells you the loss at the threshold percentile. CVaR (also called "
        "Expected Shortfall) tells you the average loss in the worst scenarios — "
        "the tail of the distribution. CVaR is a coherent risk measure (it satisfies "
        "subadditivity), which VaR is not. Basel III/IV requires both. For fat-tailed "
        "return distributions (which financial returns are), CVaR is the more "
        "informative measure.", body),

    Paragraph("Stress scenarios", h2),
    section_table([
        ["Scenario", "Date range", "Benchmark loss", "Method"],
        ["S&P 500, 2008",         "2008-10-01 – 2009-03-09", "54%",   "Actual CRSP daily returns during period"],
        ["COVID, 2020",           "2020-02-19 – 2020-03-23", "34%",   "Actual CRSP daily returns during period"],
        ["60/40 Portfolio, 2008", "2008-01-01 – 2008-12-31", "23.7%", "Actual CRSP daily returns during period"],
        ["Employer shock",        "N/A (synthetic)",          "Computed", "50% employer stock drop + 30% HC hit"],
    ]),

    Paragraph(
        "For each historical scenario, the function pulls actual CRSP daily returns "
        "for the portfolio's holdings during that date window, compounds them, and "
        "computes the cumulative loss. If CRSP data doesn't cover the period (e.g., "
        "a recent IPO), the function falls back to a beta-scaled approximation: "
        "portfolio_loss ≈ beta × benchmark_loss. The employer shock is synthetic: "
        "it models a 50% drop in employer stock × financial weight in that stock, "
        "plus a 30% hit to HC × HC fraction of total wealth.", body),

    Paragraph("Drawdown caps by risk profile", h2),
    section_table([
        ["Profile", "Cap", "If breached"],
        ["Conservative", "15%", "FLAG: tighten all per-ticker limits proportionally"],
        ["Moderate",     "20%", "FLAG: tighten all per-ticker limits proportionally"],
        ["Aggressive",   "25%", "FLAG: tighten all per-ticker limits proportionally"],
    ]),

    Spacer(1, 0.1*inch),
    Paragraph("Stress severity thresholds", h3),
    Paragraph(
        "Severity is assigned relative to the investor's own drawdown cap (not "
        "absolute thresholds), so a 20% portfolio loss is CRITICAL for a conservative "
        "investor (cap = 15%) but only HIGH for an aggressive one (cap = 25%). "
        "LOW: loss < 50% of cap. MEDIUM: 50–100% of cap. HIGH: 100–150%. "
        "CRITICAL: >150%.", body),

    Paragraph("Decision logic: APPROVE / FLAG / REJECT", h2),
    Paragraph(B("REJECT conditions (issue is upstream of Allocation — not fixable by reoptimizing):"), h3),
    Paragraph("1. flag_iteration ≥ 3 — three FLAG loops exhausted with no clean solution.", bullet),
    Paragraph(
        "2. hc_fraction > EMPLOYER_LIMIT (15%) — human capital alone exceeds the "
        "employer concentration limit. Even if the investor holds zero employer "
        "stock, they still breach the limit. No portfolio reallocation can fix this.", bullet),
    Paragraph(
        "3. CRITICAL stress result with no concentration violations and no drawdown "
        "breach — the overall risk level is just too high for any feasible portfolio.", bullet),

    Paragraph(B("FLAG conditions (fixable by sending constraints back to Allocation):"), h3),
    Paragraph(
        "Any concentration violation (from constraints.py) or drawdown exceeding the "
        "cap. The violated constraints are packaged as AllocationConstraint objects "
        "and returned in RiskOutput.constraints_violated, which flows into "
        "AllocationInput.flag_constraints on the next iteration.", body),

    Paragraph(B("APPROVE:"), h3),
    Paragraph("No violations, MDD within cap, no structurally unfixable conditions.", body),
    PageBreak(),
]

# ── 8. UNIT TESTS ─────────────────────────────────────────────────────────────
story += [
    Paragraph("8. Unit Tests — 79 Tests, All Passing", h1), HR(),
    Paragraph(
        "Tests are written with pytest, using no mocks, no database access, and "
        "no LLM calls. Every test uses hand-built numeric fixtures. This is the "
        "spec's explicit requirement: 'so every rule is unit-testable with "
        "hand-built fixtures, no database access in CI.'", body),

    section_table([
        ["Test file", "Tests", "What is verified"],
        [C("test_human_capital.py"), "17",
         "Merton formula correctness; BMS/C&V w_fin bounds and monotonicity; "
         "employer concentration; sector exposure arithmetic"],
        [C("test_constraints.py"),   "16",
         "All four limit constants; strict > vs >=; single/multi-breach detection; "
         "all_violations flat list; clean portfolio returns empty"],
        [C("test_allocation.py"),    "17",
         "Covariance symmetry and PSD; equilibrium formula; FF views shape and "
         "structure; BL posterior with zero views equals prior; optimizer "
         "constraints respected; FLAG constraint tightening"],
        [C("test_risk.py"),          "29",
         "CVaR > VaR ordering; MDD bounds; severity tiers vs drawdown cap; "
         "all APPROVE/FLAG/REJECT decision paths; HC metrics arithmetic"],
    ], col_widths=[2.0*inch, 0.6*inch, 3.8*inch]),

    Spacer(1, 0.15*inch),
    Paragraph("Notable test design decisions", h3),
    Paragraph(
        f"<b>Exact-at-limit tests:</b> Every check function uses strict > (not ≥). "
        "Tests explicitly verify that a weight of exactly 10% is clean, not a "
        "breach. This documents the boundary behaviour for future readers.", body),
    Paragraph(
        f"<b>15-ticker universe for optimizer tests:</b> With a 10% single-name "
        "limit, you need at least 10 tickers for the optimizer to be feasible "
        "(10 × 10% = 100%). Tests use 15 tickers to give the optimizer room to work.", body),
    Paragraph(
        f"<b>BL zero-views test:</b> When views are sent to zero by inflating Ω by "
        "1e10, the BL posterior must converge to the equilibrium prior Π. This "
        "validates the BL formula numerically.", body),
    PageBreak(),
]

# ── 9. WHAT'S NEXT ────────────────────────────────────────────────────────────
story += [
    Paragraph("9. What Comes Next", h1), HR(),
    Paragraph(
        "Per the spec's build sequencing, the deterministic core is now locked and "
        "tested. The remaining work is the LLM wrapper layer:", body),
    section_table([
        ["Step", "File", "What it does"],
        ["Step 4 (done)", C("tests/"), "Adversarial test set for Risk — near-100% rejection on clearly bad portfolios"],
        ["Step 5a", C("agents/allocation_agent.py"), "Thin LLM wrapper: validate AllocationInput, call run_allocation(), render rationale string"],
        ["Step 5b", C("agents/risk_agent.py"),       "Thin LLM wrapper: validate AllocationOutput, call run_risk(), render reasoning_trace"],
        ["Step 5c", C("pipeline.py"),                "Orchestration: AllocationAgent → RiskAgent → ComplianceAgent + FLAG loop (max 3 iterations)"],
    ]),
    Spacer(1, 0.1*inch),
    Paragraph(
        "The agents must stay thin. If a number appears in agent code it belongs in "
        f"{C('core/')}. The LLM's only job in each agent is: (1) parse the user "
        "message into a validated schema object, (2) call the relevant core function, "
        "(3) turn the output back into readable prose. This boundary is what makes "
        "the system auditable.", body),
]

# ── Build ─────────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=letter,
    rightMargin=0.85*inch, leftMargin=0.85*inch,
    topMargin=0.9*inch,    bottomMargin=0.9*inch,
)
doc.build(story)
print(f"PDF written: {OUTPUT}")
