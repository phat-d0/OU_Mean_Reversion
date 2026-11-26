"""plotly figure builders. Inputs are frames/dicts from :mod:`ou_model.ui.data`; styling comes from ui config."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ou_model.ui import data as D


def _chart(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("chart") or {}


def _base_layout(fig: go.Figure, cfg: dict[str, Any], title: str, height_key: str = "height") -> go.Figure:
    ch = _chart(cfg)
    fig.update_layout(
        title=title,
        height=ch.get(height_key),
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 50, "r": 20, "t": 70, "b": 40},
    )
    return fig


def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def add_markers(
    fig: go.Figure,
    summary: dict[str, Any],
    cfg: dict[str, Any],
    show_roll_gaps: bool = True,
    show_other_flags: bool = True,
) -> go.Figure:
    """Add the IS/OOS boundary, structural breaks, roll gaps and other point flags as vertical lines."""
    ch = _chart(cfg)
    boundary = D.oos_start(summary)
    if boundary is not None:
        fig.add_vline(x=boundary, line={"color": ch.get("boundary_color"), "dash": "dash", "width": 1.5})
        fig.add_annotation(x=boundary, y=0, yref="paper", text="OOS →", showarrow=False,
                           xanchor="left", yanchor="bottom", font={"color": ch.get("boundary_color")})
    for _, rec in D.flag_records(summary, "structural_breaks").iterrows():
        fig.add_vline(x=rec["date"], line={"color": ch.get("structural_break_color"), "width": 2})
        fig.add_annotation(x=rec["date"], y=0.98, yref="paper", text=str(rec.get("label", "break")),
                           showarrow=False, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.8)",
                           font={"color": ch.get("structural_break_color")})
    if show_roll_gaps:
        _rug(fig, D.flag_records(summary, "roll_gaps"), ch.get("roll_gap_color"), ch.get("roll_gap_opacity"),
             ch.get("roll_gap_width"), "roll gap")
    if show_other_flags:
        for kind in ("non_positive_prices", "bad_ticks"):
            _rug(fig, D.flag_records(summary, kind), ch.get("non_positive_color"), ch.get("roll_gap_opacity"),
                 ch.get("roll_gap_width"), kind.replace("_", " "))
    return fig


def _rug(fig: go.Figure, recs: pd.DataFrame, color: str, opacity: float, width: float, name: str) -> None:
    """Thin full-height vertical lines in one legend-toggleable trace (cheap even for dense flags)."""
    if recs.empty:
        return
    xs: list[Any] = []
    ys: list[Any] = []
    for d in recs["date"]:
        xs += [d, d, None]
        ys += [0, 1, None]
    hover = [_flag_hover(r) for _, r in recs.iterrows()]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", yaxis="y2", name=name, hoverinfo="skip",
                             line={"color": color, "width": width}, opacity=opacity))
    fig.add_trace(go.Scatter(x=recs["date"], y=[1] * len(recs), yaxis="y2", mode="markers", showlegend=False,
                             marker={"color": color, "size": 5, "symbol": "line-ns-open"},
                             name=name, text=hover, hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(yaxis2={"overlaying": "y", "range": [0, 1], "visible": False, "fixedrange": True})


def _flag_hover(rec: pd.Series) -> str:
    parts = [f"{k}={v}" for k, v in rec.items() if k != "date" and not D.is_missing(v)]
    return f"{rec['date']:%Y-%m-%d} " + ", ".join(parts)


def spread_figure(spread: pd.DataFrame, summary: dict[str, Any], cfg: dict[str, Any], **marker_opts: bool) -> go.Figure:
    """Spread with rolling μ and ±entry_z·σ band, plus boundary and flag markers."""
    ch = _chart(cfg)
    fig = go.Figure()
    if spread.empty:
        return _base_layout(fig, cfg, "Spread — no data")
    entry_z = D.signal_thresholds(summary)["entry"]
    bands = D.entry_bands(spread, entry_z)
    band_rgba = _hex_rgba(ch.get("band_color", "#2E6FDB"), ch.get("band_opacity", 0.1))
    lw = ch.get("line_width")
    fig.add_trace(go.Scatter(x=bands["date"], y=bands["upper"], mode="lines", line={"width": 0},
                             name=f"μ + {entry_z}σ", showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=bands["date"], y=bands["lower"], mode="lines", line={"width": 0},
                             fill="tonexty", fillcolor=band_rgba, name=f"μ ± {entry_z}σ (entry band)"))
    fig.add_trace(go.Scatter(x=bands["date"], y=bands["mean"], mode="lines", name="rolling μ",
                             line={"color": ch.get("mean_color"), "width": lw}))
    fig.add_trace(go.Scatter(x=spread["date"], y=spread["spread"], mode="lines", name="spread",
                             line={"color": ch.get("spread_color"), "width": lw}))
    add_markers(fig, summary, cfg, **marker_opts)
    return _base_layout(fig, cfg, "Spread with rolling mean and entry band")


def zscore_figure(spread: pd.DataFrame, summary: dict[str, Any], cfg: dict[str, Any], **marker_opts: bool) -> go.Figure:
    """Z-score with ±entry/exit/stop lines and event markers."""
    ch = _chart(cfg)
    fig = go.Figure()
    if spread.empty or "zscore" not in spread.columns:
        return _base_layout(fig, cfg, "Z-score — no data")
    fig.add_trace(go.Scatter(x=spread["date"], y=spread["zscore"], mode="lines", name="z-score",
                             line={"color": ch.get("zscore_color"), "width": ch.get("line_width")}))
    colors = ch.get("threshold_colors") or {}
    for name, z in D.signal_thresholds(summary).items():
        if D.is_missing(z):
            continue
        for sign in (1, -1):
            fig.add_hline(y=sign * z, line={"color": colors.get(name), "dash": "dot", "width": 1})
        fig.add_annotation(x=1, xref="paper", y=z, text=f"±{name} {z:g}", showarrow=False, xanchor="left",
                           font={"color": colors.get(name), "size": 10})
    styles = ch.get("event_markers") or {}
    for ev, pts in D.event_points(spread, "zscore").items():
        st = styles.get(ev) or {}
        fig.add_trace(go.Scatter(x=pts["date"], y=pts["zscore"], mode="markers", name=ev,
                                 marker={"symbol": st.get("symbol"), "color": st.get("color"),
                                         "size": ch.get("marker_size")}))
    add_markers(fig, summary, cfg, **marker_opts)
    fig = _base_layout(fig, cfg, "Z-score, thresholds and trade events")
    fig.update_layout(margin={"r": 90})
    return fig


def equity_figure(spread: pd.DataFrame, summary: dict[str, Any], cfg: dict[str, Any]) -> go.Figure:
    """Net equity curve with IS and OOS segments in distinct colours."""
    colors = cfg.get("sample_colors") or {}
    fig = go.Figure()
    if spread.empty or "equity" not in spread.columns:
        return _base_layout(fig, cfg, "Equity — no data")
    sample = spread["sample"] if "sample" in spread.columns else pd.Series("IS", index=spread.index)
    for label, key in (("IS", "in_sample"), ("OOS", "out_of_sample")):
        mask = sample == label
        if label == "OOS":  # bridge the gap so the two segments join
            idx = np.flatnonzero(mask.to_numpy())
            if idx.size and idx[0] > 0:
                mask.iloc[idx[0] - 1] = True
        seg = spread.loc[mask]
        if seg.empty:
            continue
        fig.add_trace(go.Scatter(x=seg["date"], y=seg["equity"], mode="lines", name=f"{label} equity",
                                 line={"color": colors.get(key), "width": _chart(cfg).get("line_width", 1) * 1.5}))
    add_markers(fig, summary, cfg, show_roll_gaps=False, show_other_flags=False)
    return _base_layout(fig, cfg, "Net equity curve (base cost)")


def cross_asset_figure(df: pd.DataFrame, cfg: dict[str, Any]) -> go.Figure:
    """Horizontal bars of OOS Sharpe per pair, ranked, coloured by asset class."""
    ch = _chart(cfg)
    colors = cfg.get("asset_class_colors") or {}
    fig = go.Figure()
    if df.empty:
        return _base_layout(fig, cfg, "No results")
    ordered = df.iloc[::-1]  # best on top
    for ac, grp in ordered.groupby("asset_class", sort=False):
        fig.add_trace(go.Bar(
            y=grp["label"], x=grp["oos_sharpe"].fillna(0.0), orientation="h", name=ac,
            marker_color=colors.get(ac),
            customdata=np.stack([grp["half_life_days"], grp["oos_holding_to_half_life"]], axis=-1),
            hovertemplate="%{y}<br>OOS Sharpe %{x:.3f}<br>half-life %{customdata[0]:.1f} d"
                          "<br>holding/half-life %{customdata[1]:.2f}<extra></extra>",
        ))
    fig.update_yaxes(categoryorder="array", categoryarray=list(ordered["label"]))
    fig.update_xaxes(title="OOS Sharpe (base cost)", zeroline=True)
    height = max(ch.get("cross_asset_min_height", 0), ch.get("cross_asset_height_per_pair", 0) * len(df))
    fig = _base_layout(fig, cfg, "Pairs ranked by out-of-sample Sharpe")
    fig.update_layout(height=height, hovermode="closest", barmode="overlay")
    return fig


def simulator_figure(paths: np.ndarray, dt: float, mu: float, half_life_days: float | None,
                     cfg: dict[str, Any]) -> go.Figure:
    """Simulated OU paths (rows) against time in years, with μ and the implied half-life."""
    fig = go.Figure()
    t = np.arange(paths.shape[1]) * dt
    for i, p in enumerate(paths):
        fig.add_trace(go.Scatter(x=t, y=p, mode="lines", name=f"path {i + 1}",
                                 line={"width": _chart(cfg).get("line_width")}))
    fig.add_hline(y=mu, line={"dash": "dash", "color": _chart(cfg).get("mean_color")},
                  annotation_text="μ", annotation_position="top left")
    hl = "∞ (no mean reversion)" if half_life_days is None else f"{half_life_days:.1f} trading days"
    fig = _base_layout(fig, cfg, f"Simulated OU paths — implied half-life {hl}", "simulator_height")
    fig.update_xaxes(title="years")
    fig.update_layout(hovermode="closest")
    return fig
