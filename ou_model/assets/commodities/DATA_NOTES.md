# Commodities data notes

Source: yfinance continuous front-month futures (`BZ=F`, `CL=F`, `RB=F`, `HO=F`, `GC=F`, `SI=F`),
`Close` (settlement). The data is downloaded once, cached in `data_cache/commodities/*.parquet` (gitignored), and refreshed with
`run(refresh=True)`. Today's unsettled bar is dropped (`data.drop_incomplete_last_bar`).

## 1. Not back-adjusted: roll gaps
Yahoo's `=F` series splices the front contract onto the next one on the roll date without any
adjustment. The price change between the expiring and the new contract therefore appears as a one-day
"return" that nobody could have earned. The jumps are largest where the curve is steep, or where
the contract specification changes:
- **RB=F (RBOB)** jumps around 1 March and 1 September every year. This is the summer/winter
  vapour-pressure grade switch, so the spliced contracts are different products.
- **CL=F / HO=F** jumps occur in stress periods (March 2020, March 2022) and at some rolls.

**Detection** (`config/commodities.yaml → roll_gap`): a day is flagged when
`|log return_t| / std(log returns over the previous 60 days) > 5`. The std uses only *prior*
returns, so a jump cannot inflate its own yardstick. Flags are reported per pair in
`summary.json → data.flags.roll_gaps` as `{date, leg, log_return, zscore}`. They can also catch
genuine shocks (e.g. COVID in March 2020, the 2022 invasion of Ukraine); the rule does not tell these
apart from rolls. Some flags fall in the 2014 warm-up period, before the in-sample start.

**Not corrected.** The flagged days stay in the data, so spreads, z-scores and P&L include the
jumps. For a 1:1 Brent–WTI spread the two legs roll on different dates, so a roll shows up as a
spread jump. A proper fix needs back-adjusted (ratio or difference) continuous series built from
individual contracts, which yfinance does not provide. Treat P&L on flagged days with care.

## 2. Negative WTI (2020-04-20)
`CL=F` settled at **−37.63** on 2020-04-20, the expiry of the May 2020 contract. The engine sizes
positions by dividing by price, and log transforms need positive prices. For these reasons every
non-positive price is
masked to NaN **before** any spread is built, level spreads included. That date has NaN
spread, z-score and leg weights, and it is listed in `data.flags.non_positive_prices`. No other leg has
a non-positive price.
P&L across the masked date: the engine forward-fills leg prices for P&L, so a position open
across 2020-04-20 books the full 04-17 -> 04-21 move of the masked leg on 04-21 (nothing is lost).

## 3. Units: crack spread
RB=F and HO=F are quoted in $/gallon, and CL=F in $/barrel. The crack legs are multiplied by
`scale = [1, 42, 42]` (`data.barrels_per_gallon_factor`), so all three legs are in $/bbl.
The 3-2-1 spread is `2·RB·42 + 1·HO·42 − 3·CL` ($ per 3 bbl). The scaled prices are the
tradable legs passed to the engine, so P&L and sizing use the same units as the spread.
The leg weights are the constant shares `(−3, 2, 1)`.

## 4. Calendar alignment
Each pair uses only the dates on which all of its legs have a settlement (inner join,
`max_ffill_days: 0`, no forward fill). Because the calendars differ, the gold/silver pairs have 2,992
rows from 2015 onward and the energy pairs have 2,993.

## 5. Constructions
| pair | construction | spread | shares per +1 spread unit |
|---|---|---|---|
| brent_wti | fixed_weights, level | BZ − CL | (1, −1) |
| brent_wti_hedged | hedged_spread, level | BZ − β_t CL − α_t (rolling OLS, 252d) | (1, −β_t) |
| crack_321 | fixed_weights, level | −3 CL + 2·42 RB + 1·42 HO | (−3, 2, 1) on scaled legs |
| gold_silver_ratio | ratio, log | log(GC / SI) | (1/P_GC, −1/P_SI) |
| gold_silver_hedged | hedged_spread, log | log GC − β_t log SI − α_t | (1/P_GC, −β_t/P_SI) |
