# OU Mean-Reversion on Spreads: Equities, Commodities, FX

*Research only. No broker connection and no live orders. Data comes from Yahoo Finance via yfinance and was downloaded on 2026-11-26.*

## 1. Summary

We fitted an Ornstein-Uhlenbeck (OU) model to 14 spreads and traded each one with a walk-forward z-score rule:

- Entry at |z| > 2, exit at |z| < 0.5, stop at |z| > 4.
- Trades fill one day after the signal.
- Positions are dollar-neutral.
- Costs are 5 bps per leg in the base case.
- All tests and parameters were fixed in-sample (IS, 2015–2021) and were not changed after looking at the out-of-sample period (OOS, 2022 to 2026-11).

1. **Only three spreads made money in both periods and stayed profitable at 10 bps: Brent–WTI (1:1 and hedged) and the 3-2-1 crack spread.** All three are physically linked energy spreads with economic reasons to mean-revert. OOS Sharpe is 1.07 for hedged Brent–WTI, 0.98 for 1:1 Brent–WTI and 0.63 for the crack spread. The result is fragile:
   - 56–74% of each pair's OOS P&L comes from a single year (2022, or 2026 for the hedged pair).
   - The futures series are not adjusted for contract rolls, and the different roll dates of Brent and WTI may create some of the apparent mean reversion (§6.1).
2. **Passing a cointegration test in-sample did not predict profit.** Of the 7 spreads that passed, 4 lost money in-sample at the base cost (KO/PEP, V/MA, AUD/NZD, USD/CAD vs crude), and hedged Brent–WTI only broke even (Sharpe 0.02). AUD/NZD had the strongest test result (EG p = 0.009) and lost money in both periods.
3. **Several cointegration relationships broke down OOS.** KO/PEP, AUD/NZD and the crack spread no longer pass ADF on OOS data (p = 0.47, 0.17, 0.08), and their OOS half-lives roughly doubled. HD/LOW, which failed the test in-sample, went from IS Sharpe +0.35 to OOS −0.42.
4. **Trades are held for less than one half-life.** The median holding period is 0.70× the estimated half-life in-sample and 0.65× OOS. The exit at |z| < 0.5 on a 60-day rolling z-score closes trades well before the spread has moved halfway back to its mean.
5. **Costs matter about as much as the signal.** Each spread trades about 5 round trips a year. Going from 0 to 10 bps lowers OOS annual return by about 1.0 point and OOS Sharpe by a median of 0.21.

## 2. Method

| Item | Choice (all in `config/`) |
|---|---|
| Split | IS 2015-01-01 → 2021-12-31; OOS 2022-01-01 → latest (2026-11-25). 2014 data is used only to warm up the rolling windows. |
| Spread | Hedged spreads: `log y − β_t log x − α_t`, with β from a rolling 252-day OLS fit (causal). Brent–WTI 1:1 and crack: fixed weights on prices in $/bbl (RB and HO ×42). Gold/silver ratio: `log(GC/SI)`. EUR/CHF: `log` of the rate. |
| Tests | ADF on the IS spread; Engle-Granger on the IS legs (2 legs); Johansen (3+ legs, used for the crack spread). |
| OU fit | AR(1) OLS (primary) and exact MLE, on the IS spread. The verdict counts a spread as mean-reverting if b ∈ (0,1), the lag-0 Dickey-Fuller p < 0.05 and the AIC-lag ADF p < 0.05. |
| Signal | Rolling 60-day z-score. Entry \|z\| > 2, exit \|z\| < 0.5, stop \|z\| > 4. After a stop, no re-entry until \|z\| < 0.5. No new entries while the trailing 252-day OU fit is not mean-reverting (refit every 21 days). |
| Execution | Signal at close t, filled at close t+1. Shares are fixed at entry. Dollar-neutral sizing: $0.50 long and $0.50 short per $1 of notional. |
| Costs | 5 bps per leg on traded notional at entry and exit (base); sensitivity at 0, 5 and 10 bps. |
| Quality gate | 114 pytest tests, including OU parameter recovery (OLS and MLE), look-ahead and shift tests, costs strictly lowering returns, and output schema validation. |

Returns are measured on a fixed $1 notional and do not compound. Sharpe ratios are annualised with 252 trading days.

## 3. Results: which spreads held up OOS and which broke down

The base cost is 5 bps per leg. Half-lives are in trading days. The OOS ADF p-value and OOS half-life are **diagnostics only** and were never used to choose anything.

| Asset | Pair | IS test p | Mean-reverting (IS) | IS half-life | IS Sharpe | OOS Sharpe | OOS ADF p | OOS half-life | Outcome |
|---|---|---|---|---|---|---|---|---|---|
| Commodities | brent_wti_hedged | EG 0.024 | yes | 16.1 | 0.02 | **1.07** | 0.002 | 6.3 | Held up OOS. IS edge ≈ 0 after costs. |
| Commodities | brent_wti (1:1) | EG 0.024 | yes | 30.8 | 0.36 | **0.98** | 0.013 | 7.2 | **Held up** |
| Commodities | crack_321 | Johansen rank 2 | yes | 34.4 | 0.25 | **0.63** | 0.081 | 70.8 | **Held up**, but weakening (OOS half-life ×2) |
| FX | usdcad_crude | EG 0.003 | yes | 21.2 | −0.23 | 0.35 | 0.003 | 28.1 | OOS gain without an IS edge; β ≈ −0.08, so this is effectively USD/CAD alone |
| Equities | ko_pep | EG 0.026 | yes | 27.0 | −0.40 | 0.31 | 0.468 | 65.1 | **Broke down**: no longer stationary OOS |
| Equities | v_ma | EG 0.018 | yes | 22.7 | −0.20 | 0.04 | 0.003 | 31.2 | No edge after costs |
| FX | aud_nzd | EG 0.009 | yes | 35.6 | −0.38 | −0.42 | 0.165 | 76.6 | **Broke down**: strongest IS test, lost in both periods |
| Equities | xom_cvx | EG 0.748 | yes* | 19.1 | 0.16 | 0.32 | 0.017 | 39.7 | Failed IS test; would not be traded |
| Equities | hd_low | EG 0.286 | yes* | 30.0 | 0.35 | −0.42 | 0.015 | 54.2 | Failed IS test; **IS gain reversed OOS** |
| Equities | xlf_kbe | EG 0.834 | yes* | 34.9 | −0.57 | −0.17 | 0.030 | 44.9 | Failed |
| Commodities | gold_silver_hedged | EG 0.589 | yes* | 68.0 | −0.31 | −0.14 | 0.004 | 30.0 | Failed |
| Commodities | gold_silver_ratio | EG 0.589 | no | — | −0.20 | −0.03 | 0.215 | 72.1 | Failed |
| FX | eurchf | ADF 0.610 | no | — | −0.25 | 0.64 | 0.176 | 108.0 | Failed IS test; the OOS gain is not evidence (§3.3) |
| FX | eurchf_post_snb | ADF 0.538 | no | — | −0.25 | 0.64 | 0.176 | 108.0 | Failed (same OOS as eurchf) |

\* "Mean-reverting" here describes the **rolling-hedge residual**, not the price relationship. See §6.

### 3.1 What held up
- **Brent–WTI.** Both versions pass the IS cointegration test, stay stationary OOS (ADF p ≤ 0.013) and keep a positive OOS Sharpe at every cost level. OOS hit rates are 75–78%. OOS mean reversion was *faster* than IS (half-life 7 days vs 16–31), so trades were short, about 10 days on average. The two crudes are close substitutes priced off the same physical arbitrage (shipping and pipeline capacity), which is the textbook reason for a stationary spread.
- **3-2-1 crack spread.** Johansen finds rank 2 among CL, RB and HO, and the fixed 3-2-1 combination traded profitably in both periods. The OOS ADF p of 0.08 and a half-life that doubled to 71 days suggest the refining margin has become slower to revert, or has shifted regime, since 2022.

**Concentration caveat.** Most of the OOS P&L comes from one year:

| Pair | 2022 | 2023 | 2024 | 2025 | 2026 (to Nov) |
|---|---|---|---|---|---|
| brent_wti | +0.093 | −0.001 | +0.022 | +0.005 | +0.047 |
| brent_wti_hedged | +0.043 | +0.006 | +0.020 | +0.006 | +0.130 |
| crack_321 | +0.159 | +0.030 | +0.036 | +0.033 | −0.043 |

(Values are summed daily returns on $1 notional.) The crack spread earned 74% of its OOS P&L in the 2022 energy shock. Outside the best year, these pairs earn about 1.5–2% a year at roughly 3–7% volatility. That is positive, but much weaker than the headline Sharpe suggests.

### 3.2 What broke down
- **AUD/NZD** had the strongest IS cointegration test (EG p = 0.009, Johansen rank 2) and lost money both IS and OOS. OOS the spread is no longer stationary (ADF p 0.17), and its half-life more than doubled (36 → 77 days). In 2022–2023 the RBA and RBNZ rate paths diverged, and the cross trended instead of reverting.
- **KO/PEP** passed the IS test but lost money IS (−0.40). The OOS spread is non-stationary (ADF p 0.47, half-life 27 → 65 days). The OOS Sharpe of 0.31 comes almost entirely from 2025, which earned more than the whole OOS total, so it doesn't confirm the model.
- **HD/LOW** is the classic overfitting trap. It failed the IS test, made money IS (+0.35) and lost OOS (−0.42, hit rate 35%). A rule that chose pairs by IS P&L rather than by IS tests would have picked it.

### 3.3 What looked good OOS but isn't evidence
- **EUR/CHF** failed the IS ADF test (p 0.61 on the full window, 0.54 after the SNB break) and lost money IS, but made +0.64 Sharpe OOS. The OOS gain happened despite the model, not because of it, so we don't count it. The two EUR/CHF variants have identical OOS numbers because they share the same OOS data and signals. They differ only in where the IS window starts.
- **XOM/CVX** failed Engle-Granger (p 0.75) but was profitable in both periods (IS 0.16, OOS 0.32). This is a candidate for further study, but under the project's rules it isn't tradeable.

## 4. Half-life vs holding period

| | IS | OOS |
|---|---|---|
| Median holding / IS half-life | 0.70 | 0.65 |
| Range | 0.42 – 1.08 | 0.32 – 0.97 |
| Exit reasons (all pairs, both samples) | 796 exits at \|z\| < 0.5 · 37 stops · 6 still open at end of data | |

Trades last about two-thirds of a half-life. One might expect them to last longer, since decaying from |z| = 2 to |z| = 0.5 takes about two half-lives under a stationary OOS process. Three effects explain the gap:

1. **The z-score's anchor moves.** It is measured against a 60-day rolling mean. Spread half-lives are about 20–35 days, so the rolling mean has caught up with a good part of the spread's move by the time the trade exits. Part of the exit comes from the mean moving towards the price, not from the price reverting.
2. **The rolling hedge re-centres the spread.** β_t and α_t are re-estimated daily over 252 days, which pulls the residual back towards 0.
3. **Half-lives are not stable.** For most pairs the OOS half-life was 1.3–2.4× the IS estimate (equities 19–35 → 31–65 days, AUD/NZD 36 → 77, crack 34 → 71). Brent–WTI (16–31 → 6–7) and hedged gold/silver (68 → 30) went the other way. Holding periods hardly changed, so the strategy's timing is set by the z-score window, not by the OU fit.

**Implication:** the OU half-life is a useful diagnostic, but this rule doesn't use it to time trades. Section 7 suggests tying the z-score window and a time stop to the estimated half-life.

OLS and exact MLE give the same half-lives to three decimals. For a Gaussian AR(1) sampled at equal intervals, the conditional MLE of (a, b) is the OLS solution, so the two methods differ only in σ, through the degrees-of-freedom correction. The MLE comparison works as a check of the implementation but adds no new information for daily data.

## 5. Transaction-cost sensitivity (Sharpe at 0 / 5 / 10 bps per leg)

| Pair | IS 0 | IS 5 | IS 10 | OOS 0 | OOS 5 | OOS 10 |
|---|---|---|---|---|---|---|
| brent_wti_hedged | 0.10 | 0.02 | **−0.06** | 1.24 | 1.07 | 0.90 |
| brent_wti | 0.45 | 0.36 | 0.27 | 1.17 | 0.98 | 0.79 |
| crack_321 | 0.31 | 0.25 | 0.19 | 0.70 | 0.63 | 0.56 |
| usdcad_crude | −0.17 | −0.23 | −0.29 | 0.41 | 0.35 | 0.29 |
| ko_pep | −0.29 | −0.40 | −0.51 | 0.40 | 0.31 | 0.22 |
| v_ma | −0.07 | −0.20 | −0.33 | 0.20 | 0.04 | **−0.12** |
| aud_nzd | −0.17 | −0.38 | −0.59 | −0.14 | −0.42 | −0.69 |
| xom_cvx | 0.26 | 0.16 | 0.07 | 0.43 | 0.32 | 0.21 |
| hd_low | 0.45 | 0.35 | 0.24 | −0.32 | −0.42 | −0.53 |
| xlf_kbe | −0.47 | −0.57 | −0.68 | −0.06 | −0.17 | −0.28 |
| gold_silver_hedged | −0.25 | −0.31 | −0.37 | −0.10 | −0.14 | −0.18 |
| gold_silver_ratio | −0.15 | −0.20 | −0.25 | −0.01 | −0.03 | −0.06 |
| eurchf | −0.12 | −0.25 | −0.37 | 0.79 | 0.64 | 0.49 |
| eurchf_post_snb | −0.12 | −0.25 | −0.38 | 0.79 | 0.64 | 0.49 |

- Each round trip costs 2 legs × (entry + exit) × the bps rate on $0.50 per leg. That is 10 bps of notional at 5 bps per leg. At about 5 round trips a year, 0 → 10 bps costs about **1.0% a year** (median 0.99%, range 0.87–1.31%). Brent–WTI trades most often and pays the most.
- The drop in Sharpe from that cost is largest where volatility is low. AUD/NZD, with 1.6% OOS volatility, loses 0.55 Sharpe going from 0 to 10 bps.
- **Brent–WTI (1:1) and the crack spread are the only pairs that pass the IS test and are positive in both samples at all three cost levels.** Hedged Brent–WTI turns negative IS at 10 bps.
- Real costs for these instruments are probably *above* 5 bps per leg for retail-sized trades. Futures roll costs aren't modelled, and FX and ETF spreads widen in stress, which is exactly when entries trigger.

## 6. Known limitations

1. **Futures roll gaps.** The yfinance `=F` series splice the front contract and are **not back-adjusted**. Price jumps at the roll stay in both the spread and the P&L. A 5σ jump rule flags them; it doesn't correct them. In the evaluation window it flags 9 distinct dates for the Brent–WTI pairs, 29 for the crack spread (19 of them on RB, many at the March and September changes to and from summer-grade gasoline) and 9 for gold/silver. P&L on the flagged dates:

   | Pair | IS P&L total / on flagged days | OOS P&L total / on flagged days |
   |---|---|---|
   | brent_wti | +0.170 / +0.047 | +0.165 / 0.000 |
   | brent_wti_hedged | +0.010 / −0.009 | +0.204 / 0.000 |
   | crack_321 | +0.136 / −0.002 | +0.216 / **−0.069** |
   | gold_silver_ratio | −0.111 / −0.072 | −0.028 / **+0.237** |
   | gold_silver_hedged | −0.133 / −0.051 | −0.075 / 0.000 |

   The *large* flagged jumps don't explain the energy results: Brent–WTI earned nothing OOS on them, and the crack spread lost money on them. The flagged gold/silver days in 2025–26 look like real silver moves rather than rolls. **The bigger risk is the roll jumps too small to flag.** Brent and WTI front months expire on different calendars. When one leg rolls before the other, the spread shifts by that leg's contango or backwardation, then shifts back when the other leg rolls. This roughly monthly sawtooth *looks like* fast mean reversion. Brent–WTI's very short OOS half-life (6–7 days) and ~10-day holding period are consistent with the strategy partly trading this artefact. Treat the Brent–WTI and crack results as unproven until they are re-run on back-adjusted or constant-maturity series. See `ou_model/assets/commodities/DATA_NOTES.md`.
2. **Survivorship and hindsight bias.** The equity pairs (KO/PEP, XOM/CVX, HD/LOW, V/MA, XLF/KBE) are well-known pairs of today's large caps, chosen with hindsight. A selection made in 2015 could have included pairs that later diverged or were delisted. Our equity results would likely *overstate* what a real selection would achieve, and even so they are poor.
3. **Structural breaks.**
   - The SNB removed the EUR/CHF 1.20 floor on 2015-01-15. This is flagged, and an `eurchf_post_snb` variant starts its estimation in February 2015. Yahoo stamps the −18.9% move on **2015-01-16**, one day late.
   - CL=F settled at **−37.63 on 2020-04-20**. That price is masked to NaN for spreads and sizing. P&L forward-fills across it, so a position held through it books the full 04-17 → 04-21 move.
   - The 2022 energy shock dominates the commodity OOS results (§3.1).
   - No regime filter was used.
4. **Rolling-hedge residuals look mean-reverting by construction.** With β_t and α_t re-fitted over 252 days, the residual tends to pass ADF and give a finite half-life even when the legs are not cointegrated. XOM/CVX, HD/LOW, XLF/KBE and hedged gold/silver all fail Engle-Granger but count as "mean-reverting". The Engle-Granger or Johansen test on the **raw legs** is the meaningful filter, and our cointegration verdict uses it.
5. **Data quality.**
   - Adjusted closes (`Adj Close`) are back-adjusted from the download date, so equity price levels, though not returns, shift slightly between downloads.
   - FX and futures calendars differ. For USD/CAD vs crude, CL=F needed 116 one-day forward fills (US futures holidays) and CAD=X needed 4.
   - Yahoo's FX closes are not a single fixing time.
   - The data has no bid-ask information, so costs are a flat bps assumption.
6. **Weak cross-asset link.** USD/CAD vs crude passes Engle-Granger, but the IS β is about −0.08. The "spread" is almost entirely log USD/CAD, so it shows little about CAD's link to oil.
7. **Model and backtest simplifications.**
   - Returns are on a fixed notional and do not compound.
   - Dollar-neutral sizing overrides the hedge ratio's beta-weighting for P&L. The hedge ratio defines the signal, and both legs get equal dollars.
   - The 6 trades still open at the end of the data are marked to market without an exit cost.
   - Financing, carry and the FX interest differential are not modelled. For AUD/NZD and USD/CAD these can be comparable in size to the edge.
8. **Multiple comparisons.** Of the 14 spreads, 11 are distinct. With that many, one or two would look good OOS by chance (EUR/CHF and XOM/CVX are candidates). No correction for this was applied.

## 7. Suggested next steps

1. **Kalman-filter hedge ratio.** It is already implemented in `core/hedge.py` (`hedge.method: kalman`, with parameters in `config/core.yaml`). Compare it to rolling OLS on the IS period only, then apply the IS-chosen setting to OOS once.
2. **Back-adjusted or constant-maturity futures.** This would remove the roll-gap artefacts in Brent–WTI and the crack spread before trusting their Sharpe ratios.
3. **Regime and break filters.** Pause entries when a rolling Engle-Granger or ADF p-value rises above a threshold, or after a structural-break test such as CUSUM or Bai–Perron. Many of the OOS losses (AUD/NZD, KO/PEP) came after the relationship had already weakened.
4. **Tie timing to the OU fit.** Set the z-score window to k × half-life, measure z against the OU equilibrium (μ, σ/√(2κ)) rather than a fixed 60-day window, and turn on the existing `max_holding_multiple` time stop. Choose k in-sample.
5. **Realistic costs and execution.** Link to an order-flow and execution model: spread- and impact-aware fills, futures roll costs, FX carry, and borrow costs for the short equity leg. Re-rank pairs on net-of-execution results.
6. **Wider universe with out-of-sample selection.** Screen sector peers each year using only data available at the time, to measure the selection effect free of survivorship bias.

## 8. How to reproduce

The commands below rebuild the environment, run all 114 tests, re-download the data and regenerate all 14 result folders, then launch the dashboard.

```bash
make install
make clean-results && make all
make ui
```

The repo is organised as follows:
- `ou_model/core/`: the shared engine.
- `ou_model/assets/{equities,commodities,fx}/`: data loading, cleaning and spread construction.
- `ou_model/ui/`: the dashboard, which reads only `results/`. Screenshots are in `ou_model/ui/screenshots/`.
- `CONTRACTS.md`: the interfaces between components.
- `.claude/agents/`: the agent roles.
