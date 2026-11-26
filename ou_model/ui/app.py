"""Streamlit dashboard for the OU mean-reversion results.

Run with ``.venv/bin/python -m streamlit run ou_model/ui/app.py``.
Reads only ``results/`` and ``config/ui.yaml``. Optional query parameters
``?asset_class=fx&pair=eurchf`` preselect a pair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from ou_model.ui import charts as C
from ou_model.ui import data as D

REPO_ROOT = Path(__file__).resolve().parents[2]
UI_CONFIG_PATH = REPO_ROOT / "config" / "ui.yaml"


@st.cache_data(show_spinner=False)
def _config() -> dict[str, Any]:
    return D.load_ui_config(UI_CONFIG_PATH)


def _results_root(cfg: dict[str, Any]) -> Path:
    root = Path(cfg.get("results_root", "results"))
    return root if root.is_absolute() else REPO_ROOT / root


@st.cache_data(show_spinner=False)
def _load_pair(directory: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    return D.load_summary(directory), D.load_spread(directory), D.load_trades(directory)


def _preselect(options: list[str], key: str) -> int:
    wanted = st.query_params.get(key)
    return options.index(wanted) if wanted in options else 0


def sidebar(pairs: dict[str, list[str]]) -> tuple[str, str, dict[str, bool]]:
    """Asset class, then pair (only those present), then marker toggles."""
    st.sidebar.title("OU mean reversion")
    classes = sorted(pairs)
    asset_class = st.sidebar.selectbox("Asset class", classes, index=_preselect(classes, "asset_class"))
    names = pairs[asset_class]
    pair = st.sidebar.selectbox("Pair", names, index=_preselect(names, "pair"))
    st.sidebar.markdown("**Chart markers**")
    opts = {
        "show_roll_gaps": st.sidebar.toggle("Roll gaps", value=True),
        "show_other_flags": st.sidebar.toggle("Non-positive prices / bad ticks", value=True),
    }
    st.sidebar.caption("Research use only. Data read from results/.")
    return asset_class, pair, opts


def render_pair(cfg: dict[str, Any], directory: Path, marker_opts: dict[str, bool]) -> None:
    """Spread, z-score and equity panels plus parameter/metric tables for one pair."""
    summary, spread, trades = _load_pair(str(directory))
    if not summary:
        st.warning(f"No readable summary.json in {directory}.")
        return
    color = (cfg.get("asset_class_colors") or {}).get(summary.get("asset_class"), "inherit")
    legs = ", ".join(map(str, summary.get("legs") or []))
    st.markdown(f"### <span style='color:{color}'>{summary.get('asset_class', '')}</span> / "
                f"{summary.get('pair', directory.name)} &nbsp; <small>{legs}</small>", unsafe_allow_html=True)
    _headline(summary, cfg)
    if spread.empty:
        st.info("spread.parquet is missing; charts unavailable.")
    else:
        st.plotly_chart(C.spread_figure(spread, summary, cfg, **marker_opts), use_container_width=True)
        st.plotly_chart(C.zscore_figure(spread, summary, cfg, **marker_opts), use_container_width=True)
        st.plotly_chart(C.equity_figure(spread, summary, cfg), use_container_width=True)
    if trades.empty:
        st.info("No round-trip trades for this pair.")

    left, right = st.columns(2)
    with left:
        st.subheader("OU parameters")
        st.dataframe(D.ou_params_table(summary, cfg), use_container_width=True)
        st.subheader("Verdict")
        st.dataframe(D.verdict_table(summary, cfg), use_container_width=True)
    with right:
        st.subheader("Tests")
        st.dataframe(D.tests_table(summary, cfg), use_container_width=True, hide_index=True)
        st.subheader("Metrics (base cost)")
        st.dataframe(D.metrics_table(D.dig(summary, "metrics", default={}), cfg), use_container_width=True)
    st.subheader("Cost sensitivity")
    st.dataframe(D.cost_sensitivity_table(summary, cfg), use_container_width=True, hide_index=True)
    _flags_section(summary, trades)


def _headline(summary: dict[str, Any], cfg: dict[str, Any]) -> None:
    missing, nf = D.format_settings(cfg)
    oos = D.dig(summary, "metrics", "out_of_sample", default={})
    is_ = D.dig(summary, "metrics", "in_sample", default={})
    v = D.dig(summary, "verdict", default={})
    cols = st.columns(5)
    cols[0].metric("OOS Sharpe", D.fmt(oos.get("sharpe"), nf["default"], missing))
    cols[1].metric("IS Sharpe", D.fmt(is_.get("sharpe"), nf["default"], missing))
    cols[2].metric("Half-life (days)", D.fmt(is_.get("half_life_days"), nf["default"], missing))
    cols[3].metric("Cointegrated (IS)", D.fmt(v.get("cointegrated_in_sample"), "{}", missing))
    cols[4].metric("Mean-reverting (IS)", D.fmt(v.get("mean_reverting_in_sample"), "{}", missing))


def _flags_section(summary: dict[str, Any], trades: pd.DataFrame) -> None:
    counts = D.flag_counts(summary)
    label = "Data flags and notes (" + ", ".join(f"{k}: {n}" for k, n in counts.items()) + ")"
    with st.expander(label):
        for note in D.flag_notes(summary):
            st.markdown(f"- {note}")
        for kind in D.POINT_FLAGS:
            recs = D.flag_records(summary, kind)
            if not recs.empty:
                st.markdown(f"**{kind}**")
                st.dataframe(recs, use_container_width=True, hide_index=True)
        alignment = D.dig(summary, "data", "flags", "alignment")
        if alignment:
            st.markdown("**alignment**")
            st.json(alignment, expanded=False)
    if not trades.empty:
        with st.expander(f"Trades ({len(trades)})"):
            st.dataframe(trades, use_container_width=True, hide_index=True)


def render_cross_asset(cfg: dict[str, Any], root: Path) -> None:
    """All pairs ranked by OOS Sharpe, coloured by asset class."""
    df = D.cross_asset_frame(root)
    if df.empty:
        st.info("No results found.")
        return
    st.plotly_chart(C.cross_asset_figure(df, cfg), use_container_width=True)
    colors = cfg.get("asset_class_colors") or {}
    table = D.format_cross_asset(df, cfg)
    styled = table.style.apply(
        lambda row: [f"color: {colors.get(row['asset class'], 'inherit')}; font-weight: 600"
                     if c in ("asset class", "pair") else "" for c in row.index], axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption("Half-life from the in-sample OU fit; holding / half-life uses OOS average holding days. "
               "OOS figures are reported, never used for selection.")


def render_simulator(cfg: dict[str, Any]) -> None:
    """Sliders for κ, μ, σ (ranges from config) and simulated OU paths."""
    from ou_model.core.ou import simulate_ou  # the single allowed core import

    sim = cfg.get("simulator") or {}
    cols = st.columns(3)
    params = {}
    for col, name, label in zip(cols, ("kappa", "mu", "sigma"), ("κ (per year)", "μ", "σ (per √year)")):
        rng = sim.get(name) or {}
        params[name] = col.slider(label, float(rng["min"]), float(rng["max"]), float(rng["default"]))
    seed = st.number_input("Seed", min_value=0, value=0, step=1)
    dt = float(cfg.get("simulator_dt"))
    n_steps, n_paths = int(sim.get("n_steps")), int(sim.get("n_paths"))
    paths = np.vstack([
        simulate_ou(params["kappa"], params["mu"], params["sigma"], params["mu"], n_steps, dt, seed=int(seed) + i)
        for i in range(n_paths)
    ])
    hl = D.implied_half_life_days(params["kappa"], cfg.get("trading_days_per_year"))
    st.plotly_chart(C.simulator_figure(paths, dt, params["mu"], hl, cfg), use_container_width=True)
    st.caption("Exact discretisation; each path starts at μ. Half-life = ln 2 / κ.")


def main() -> None:
    """Entry point."""
    st.set_page_config(page_title="OU mean reversion", layout="wide")
    cfg = _config()
    root = _results_root(cfg)
    pairs = D.discover_pairs(root)
    if not pairs:
        st.error(f"No results found under {root}. Run the pipeline first.")
        render_simulator(cfg)
        return
    asset_class, pair, opts = sidebar(pairs)
    tab_pair, tab_cross, tab_sim = st.tabs(["Pair view", "Cross-asset summary", "Simulator"])
    with tab_pair:
        render_pair(cfg, D.pair_dir(root, asset_class, pair), opts)
    with tab_cross:
        render_cross_asset(cfg, root)
    with tab_sim:
        render_simulator(cfg)


main()
