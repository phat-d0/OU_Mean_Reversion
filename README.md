# OU Mean-Reversion Research

Research-only project: Ornstein-Uhlenbeck mean-reversion models fitted to spreads in
equities, commodities and FX, backtested walk-forward with transaction costs.
**No broker connection and no live orders.**

## Layout
- `ou_model/core/` – shared engine (OU fit/simulation, stationarity tests, hedge ratios, signals, backtest, metrics)
- `ou_model/assets/{equities,commodities,fx}/` – data loading, cleaning and pair construction per asset class
- `ou_model/ui/app.py` – Streamlit dashboard reading `results/` only
- `config/` – every parameter (one YAML per asset class + `core.yaml` + `ui.yaml`)
- `results/<asset_class>/<pair>/` – `spread.parquet`, `trades.parquet`, `summary.json`
- `CONTRACTS.md` – interfaces between components; `.claude/agents/` – subagent definitions

## Quick start
```bash
make install   # python3.11 venv + editable install
make test      # pytest (core quality gate + asset/UI unit tests)
make run       # download data, build all pairs, write results/
make ui        # launch the dashboard
```

## Branches
`main` holds reviewed, merged work only. Each agent works on `agent/<name>`
(`quant`, `equities`, `commodities`, `fx`, `ui`); the Lead merges into `main` after review.

## Data split
In-sample 2015-01-01 → 2021-12-31 (tests, hedge ratios, parameter choice);
out-of-sample 2022-01-01 → present (evaluation only).
