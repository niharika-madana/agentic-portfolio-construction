import sys, json
sys.path.insert(0, '.')
from agents.compliance.test_compliance import _make_compliance_input, _make_risk_output
from agents.compliance.compliance_agent import run_compliance
from contracts import RiskDecision

personas = [
    dict(persona="biology_professor",    risk_tolerance="conservative",
         hc_type="bond-like",            income_equity_beta=0.05,
         implicit_equity_exposure=0.039, rsu_concentration=0.00, employer_sector="education"),
    dict(persona="tech_executive",        risk_tolerance="moderate",
         hc_type="equity-like",          income_equity_beta=1.20,
         implicit_equity_exposure=0.906, rsu_concentration=0.40, employer_sector="technology"),
    dict(persona="financial_professional",risk_tolerance="aggressive",
         hc_type="mixed",                income_equity_beta=0.60,
         implicit_equity_exposure=0.501, rsu_concentration=0.00, employer_sector="finance"),
]

for p in personas:
    ci = _make_compliance_input(**p)
    ro = _make_risk_output(hc_type=p["hc_type"], rsu_concentration=p["rsu_concentration"],
                           portfolio_volatility_annual=0.15)
    result = run_compliance(ci, ro)
    print(f"\n{'='*60}")
    print(f"  {p['persona'].upper()}")
    print(f"  Status:   {result.compliance_status.value}")
    print(f"  Cleared:  {result.clearance}")
    print(f"  Severity: {result.overall_severity.value}")
    print(f"  Passed:   {len(result.passed_checks)} checks")
    print(f"  Violations:")
    for v in result.violations:
        print(f"    [{v.severity.value}] {v.check}: {v.description[:80]}...")
    print(f"  Recommendation: {result.recommendation}")
