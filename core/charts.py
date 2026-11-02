"""Gráficos do dashboard (Plotly) com a paleta validada: status para veredito e severidade, azul para magnitude."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .config import SEVERIDADES
from .viz import SEV_COLOR, SEV_LABEL, VERDICT_COLOR

BLUE = "#2a78d6"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
MUTED = "#898781"
GRID = "rgba(137,135,129,0.25)"


def _layout(fig: go.Figure, height: int = 280) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13), hoverlabel=dict(font_size=13))
    return fig


def gauge(conformity: float, verdict: str, cfg, title: str) -> go.Figure:
    nc = (1 - cfg.threshold_nc) * 100
    su = (1 - cfg.threshold_susp) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=conformity, number=dict(suffix="%", font=dict(size=40)),
        title=dict(text=f"{title}<br><span style='font-size:0.8em;color:{VERDICT_COLOR.get(verdict, BLUE)}'>{verdict}</span>", font=dict(size=15)),
        gauge=dict(axis=dict(range=[0, 100], tickwidth=1, tickcolor=MUTED), bar=dict(color=VERDICT_COLOR.get(verdict, BLUE), thickness=0.35),
                   bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   steps=[dict(range=[0, nc], color="rgba(208,59,59,0.18)"), dict(range=[nc, su], color="rgba(250,178,25,0.22)"),
                          dict(range=[su, 100], color="rgba(12,163,12,0.18)")])))
    return _layout(fig, 270)


def verdict_donut(docs: list[dict]) -> go.Figure:
    order = ["CONFORME", "SUSPEITO", "NÃO CONFORME"]
    vals = [sum(1 for d in docs if d["verdict"] == o) for o in order]
    fig = go.Figure(go.Pie(labels=[o.title() for o in order], values=vals, hole=0.55, sort=False, direction="clockwise",
                           marker=dict(colors=[VERDICT_COLOR[o] for o in order], line=dict(color="rgba(0,0,0,0)", width=2)),
                           textinfo="label+value", textfont=dict(size=13), hovertemplate="%{label}: %{value} documento(s)<extra></extra>"))
    fig.update_layout(title=dict(text="Documentos por veredito", font=dict(size=15)), showlegend=True,
                      legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"))
    return _layout(fig, 270)


def severity_bars(df: pd.DataFrame) -> go.Figure:
    counts = [int((df["severidade_id"] == s).sum()) if len(df) else 0 for s in SEVERIDADES]
    fig = go.Figure(go.Bar(x=[SEV_LABEL[s] for s in SEVERIDADES], y=counts, marker=dict(color=[SEV_COLOR[s] for s in SEVERIDADES], cornerradius=4),
                           text=counts, textposition="outside", cliponaxis=False, hovertemplate="%{x}: %{y} achado(s)<extra></extra>"))
    fig.update_layout(title=dict(text="Achados por severidade", font=dict(size=15)), yaxis=dict(gridcolor=GRID, zeroline=False, dtick=1 if max(counts, default=0) < 6 else None),
                      xaxis=dict(showgrid=False), bargap=0.45)
    return _layout(fig, 270)


def hbar(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    vc = df[col].value_counts().sort_values() if len(df) else pd.Series(dtype=int)
    fig = go.Figure(go.Bar(x=vc.values, y=vc.index, orientation="h", marker=dict(color=BLUE, cornerradius=4), text=vc.values, textposition="outside",
                           cliponaxis=False, hovertemplate="%{y}: %{x} achado(s)<extra></extra>"))
    fig.update_layout(title=dict(text=title, font=dict(size=15)), xaxis=dict(gridcolor=GRID, zeroline=False), yaxis=dict(showgrid=False), bargap=0.4)
    return _layout(fig, max(220, 60 + 38 * max(len(vc), 1)))


def heatmap(df: pd.DataFrame, row: str, col: str, title: str, col_title: str) -> go.Figure:
    if len(df) == 0:
        return _layout(go.Figure(), 200)
    pt = df.pivot_table(index=row, columns=col, values="rule", aggfunc="count", fill_value=0)
    pt = pt.reindex(sorted(pt.columns), axis=1)
    z = pt.values
    zmax = max(int(z.max()), 1)
    fig = go.Figure(go.Heatmap(z=z, x=[str(c) for c in pt.columns], y=list(pt.index), colorscale=[[0, "#f3f6fb"], [0.001, SEQ[0]], [1, SEQ[-1]]],
                               zmin=0, zmax=zmax, xgap=2, ygap=2, showscale=False, hovertemplate=f"{col_title} %{{x}}<br>%{{y}}: %{{z}} achado(s)<extra></extra>",
                               text=[[str(v) if v else "" for v in r] for r in z], texttemplate="%{text}", textfont=dict(size=13)))
    fig.update_layout(title=dict(text=title, font=dict(size=15)), xaxis=dict(title=col_title, type="category"), yaxis=dict(autorange="reversed"))
    return _layout(fig, max(220, 90 + 36 * len(pt.index)))
