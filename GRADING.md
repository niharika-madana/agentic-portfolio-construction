# Capstone Project II — Grading Criteria

**Project**: AI Financial Advisor — Agentic Portfolio Construction
**Mentor**: Haiyang Li (Ocean) | **Faculty**: Prof. Qing Sheng
**Cohort**: Summer 2026 | **Duration**: 12 weeks

> **Tentative.** This rubric is a working draft. Prof. Sheng owns the final grades; the
> channel weights below are provisional and may be adjusted once confirmed with her. Treat
> the *dimensions of quality* as stable — the philosophy won't change — and the *exact
> percentages* as subject to revision.

---

## Philosophy

Building a five-agent system is easy now. An agent can scaffold the Profile → Research →
Allocation → Risk → Compliance pipeline in an afternoon. That is not what you are graded on.

What we care about is whether the portfolio the system produces is *good*, whether you can
tell when the AI is wrong, and whether you understand why a tech executive should hold a
fundamentally different portfolio than a biology professor. The pipeline is plumbing. The
judgment is the project.

This is an agentic project, so one rule governs everything: **unevaluated output from an
agent might as well be noise.** Before you tune a prompt, swap a model, or add a tool, you
must be able to say how you will measure whether that change made the portfolio better. If
you can't measure it, you can't claim it. Evaluation is not the last step — it is the first
thing you design.

You cannot vibe your way through this one. If a vibed system were the goal, an agent already
builds it. Your work has to be the part the agent can't hand you: knowing what to build,
and knowing whether to trust what it builds back.

---

## What We Grade

Four dimensions describe what "good" looks like. The point weights are in the grade table
further down. The five-agent system working is the floor — the real weight sits in modeling
rigor, the trust-and-evaluation framework, and how well you defend the work in the room.

### 1. Technical Foundation

This is the floor. Clearing it earns a passing baseline and nothing more.

| Requirement | Expectation |
|-------------|-------------|
| **System works end-to-end** | Profile → Research → Allocation → Risk → Compliance runs start to finish on all three personas. No broken handoffs, no placeholder agents, no stubs. The bare minimum is that it produces a portfolio. |
| **Architecture is sound** | Clean separation between agents, with defined communication protocols and validation gates between phases. Someone should be able to swap the Allocation agent without rewriting Profile or Risk. We can tell a designed pipeline from a vibe-coded one. |
| **Real data pipeline** | WRDS (CRSP/Compustat) for historical returns and fundamentals, FRED for macro (yield curves, credit spreads, unemployment, CPI). Live, reproducible ingestion — not hand-pasted numbers. |
| **Evaluation harness exists** | There is an actual mechanism that scores portfolio quality, not just a pile of agent outputs. Every tuning decision (prompt, model, tooling) is justified against it. |
| **Reproducibility** | A teammate can clone the repo, run the pipeline, and get the same allocations. Dependencies documented, data paths clear, API usage explained. |

### 2. Modeling & Quantitative Rigor

Every allocation decision needs a derivation behind it. "The agent suggested it and it looked
reasonable" falls apart the moment someone asks why.

| Requirement | Expectation |
|-------------|-------------|
| **Human capital modeled, not hand-waved** | Total wealth = financial capital + human capital (present value of future earnings). Show how you estimate it per persona and how it changes the allocation. A bond-like income (the professor) and an equity-like income (the tech exec) must drive *meaningfully* different portfolios — and you must derive why. |
| **Regimes defined rigorously** | 4–5 historical regimes (dot-com, GFC, COVID, rate-hike, AI boom). Define them by data, not by vibes. What macro signals mark each regime's boundaries? |
| **Tied to your coursework** | Connect the work to what you've studied — portfolio theory, time series, fixed income, risk management, microstructure. This is a capstone; pull your degree together. |
| **Parameter choices justified** | Why this allocation method? Why these constraints? Why this rebalancing trigger? Grid search is fine if you can explain what it taught you. |

### 3. Trust, Evaluation & Fiduciary Rigor

**This is the distinguishing dimension of the project, and where it will be judged most
carefully.** The proposal says it plainly: the deliverable is not portfolio construction —
it is the validation pipeline. How do you *know* the AI's recommendation serves the client?

| Requirement | Expectation |
|-------------|-------------|
| **Hallucination detection** | How do you catch the agent inventing alpha, citing a regime analogue that doesn't fit, or recommending a position it can't justify? Show the validation gate, not just the claim that you "checked." |
| **Benchmarked against reality** | Backtest each personalized allocation through every regime against an 80/20 benchmark. Answer the key question: *when does personalization actually matter, and when is it noise?* |
| **Agent vs. industry comparison** | Document what a typical commission-driven advisor would recommend for each persona vs. what your system produces. Where do they diverge, and is your divergence defensible? |
| **"Already priced in"** | Grapple honestly with market efficiency. If the macro regime is public information, why isn't the personalization already reflected in prices? An honest engagement scores higher than pretending the question doesn't exist. |
| **Fiduciary meaning** | What does "fiduciary duty" mean when the advisor is a machine? Your Compliance agent should encode something real — suitability, concentration limits, conflict-free reasoning — not a rubber stamp. |
| **Honest conclusions** | If personalization doesn't beat 80/20 after costs for a given persona, *say so.* A rigorous "it didn't help here, and here's why" scores higher than a fabricated win. |

### 4. Oral Defense

The AI can write your paper. It can't sit in the chair and answer for you. Part of this is a
group presentation; the larger part is each of you defending the agent area you own (see the
grade table below).

**Example questions you should be able to answer:**

- Walk me through how you estimate human capital for the tech executive. What's the present
  value, what discount rate, and why?
- Explain why the tech executive got a different allocation than the professor — and what
  regime change would flip that.
- Your Research agent matched the current macro state to the 2018 rate-hike regime. How do
  you know that analogue is right and not a hallucination?
- Your Risk agent is "adversarial." Show me a recommendation it actually killed, and why.
- The personalized portfolio beat 80/20 in your backtest. Is that edge, or is it survivorship
  / overfitting to the regimes you happened to pick?
- What does your Compliance agent check that a human compliance officer would, and what does
  it miss?
- If the macro regime is public, why isn't this allocation already priced in?
- How do you know any single number in the final recommendation isn't fabricated? Walk me
  through your validation chain.

The defense goes deeper on **whatever you personally built**. You each own an agent area —
Profile, Research, Allocation, Risk, or Compliance — and you defend it. Divide the work so
each person owns a piece they can stand behind.

---

## How Your Grade Is Computed

> *Provisional — pending Prof. Sheng's confirmation.*

Your grade is assembled from five channels across the whole term, so steady work counts and
there's no single make-or-break moment.

| Channel | Weight | What it captures |
|---------|-------:|------------------|
| **Final paper & codebase** | 40% | The deliverable itself — the paper and the five-agent repository as submitted, judged against the four quality dimensions above. |
| **Individual oral defense** | 25% | Each person defends the agent area they own under questioning. Graded per student. This is the component the mentor weights most heavily. |
| **Ongoing participation** | 15% | Tracked throughout, at the mentor's discretion. Weekly engagement, demonstrated progress, the quality of your questions, how the work improves across the 12 weeks. |
| **Group presentation** | 10% | The team presents framing, methodology, and results together. |
| **Teammate evaluation** | 10% | Confidential peer assessment of each member's contribution and collaboration. |
| **Total** | **100%** | |

Two things stand out. **Half your grade is individual** — your own defense, your participation,
and how your teammates rate you — so a strong team can't carry a quiet member. And **owning
an agent area is part of the structure**: the area you pick (Profile, Research, Allocation,
Risk, Compliance) is the area you defend.

---

## Scope & Freedom

Agentic portfolio construction with human capital is the theme. Within it, you are free to
chase what genuinely interests you. Some directions worth considering:

- **When does personalization pay?** — Find the regimes and personas where accounting for
  human capital beats 80/20 by the widest margin, and the ones where it doesn't matter at all.
- **Regime-change detection** — Can the system reliably flag a macro shift and propose a
  justified rebalance, or does it churn on noise?
- **The fourth persona** — The proposal gives three (professor, tech exec, financial pro).
  Design a fourth that stresses the system — early-career, concentrated founder equity, near
  retirement — and show what breaks.
- **Unstructured data as a feature** — Agents are good at unstructured input. What text
  (filings, news, earnings calls) could sharpen the profile or research stage?
- **Evaluation methodology itself** — Is there a better way to score "is this portfolio good
  for *this* client" than backtest-vs-benchmark? That question alone could be a contribution.

Pick what excites you. Push it past what the agent hands you.

---

## Deliverables

| Deliverable | Format |
|-------------|--------|
| **Five-agent system** | Python — Profile, Research, Allocation, Risk, Compliance. Runs on a laptop. |
| **Three-persona portfolios** | Meaningfully different allocations for the professor, the tech executive, and the financial professional. |
| **Regime backtesting** | Personalized allocations vs. 80/20 benchmark across 4–5 historical regimes. |
| **Trust evaluation** | Hallucination detection, agent-vs-industry comparison, fiduciary methodology. |
| **Academic paper** | 12–15 pages, LaTeX or equivalent. Problem statement, literature, methodology, results, honest conclusion. |
| **Code repository** | Clean, documented, reproducible. README with setup. Data pipeline included. |
| **Group presentation + individual defense** | Team presents; each member defends their agent area. |

---

## Timeline

| Phase | Weeks | Focus |
|-------|-------|-------|
| **System Design & Data Infrastructure** | 1–4 | Design the agent architecture (roles, protocols, validation gates). Build the WRDS + FRED pipeline. Define regimes. Implement the Profile agent with total-wealth intake. |
| **Construction & Backtesting** | 5–8 | Run all three personas. Backtest through each regime vs. 80/20. Build regime-change detection and evaluate proposed rebalances. |
| **Critical Evaluation & Defense** | 9–12 | Compare against industry-standard advice. Address "already priced in." Write the paper, prepare the defense. |

Weekly meetings: **Fridays, ~10:00 AM ET** (confirm recurring invite). Progress shown in
these meetings feeds the participation grade.

---

## A Note on AI Usage

You may — and should — use AI tools (Claude, ChatGPT, Cursor) for coding and research. That
reflects how the work is actually done now, and this project is *about* working with agents.

But grading measures **demonstrated understanding**, surfaced through the oral defense. If an
agent built your Allocation logic and you can't explain why the tech executive is underweight
tech, that becomes obvious fast in the room. The tools do the typing. The understanding, and
the judgment about what to trust, still has to be yours.

The whole point of this project: building the system is easy. Knowing whether its advice is
genuinely good for the client is the hard part — and the only part worth grading.
