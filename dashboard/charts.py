"""Plotly chart builders untuk dashboard."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_DARK_TEMPLATE = "plotly_dark"
_MA_COLORS = {
    "ma20": "#38bdf8",   # sky blue
    "ma50": "#fb923c",   # orange
    "ma200": "#a78bfa",  # purple
}


def price_chart(
    df_raw: pd.DataFrame,
    ticker: str,
    signal_date: str | None = None,
    lookback_days: int = 120,
) -> go.Figure:
    """Candlestick + MA overlay + Volume bar chart.

    Args:
        df_raw: DataFrame OHLCV dari data_loader.load_raw()
        ticker: label untuk judul chart
        signal_date: tanggal scan (dipakai untuk marker vertikal), format "YYYY-MM-DD"
        lookback_days: berapa hari terakhir yang ditampilkan

    Returns:
        Plotly Figure dengan dua row (price + volume)
    """
    if df_raw.empty:
        return _empty_figure(f"Tidak ada data raw untuk {ticker}")

    df = df_raw.tail(lookback_days).copy()
    df["date"] = pd.to_datetime(df["date"])

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.72, 0.28],
    )

    # --- Candlestick ---
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
            increasing_fillcolor="#22c55e",
            decreasing_fillcolor="#ef4444",
        ),
        row=1, col=1,
    )

    # --- MA overlays (hanya jika kolom ada di df_raw) ---
    # Hitung MA dari raw OHLCV jika tidak ada di df_raw
    close = df["close"]
    for window, col, label in [(20, "ma20", "MA20"), (50, "ma50", "MA50"), (200, "ma200", "MA200")]:
        if col in df.columns:
            ma_series = df[col]
        else:
            ma_series = close.rolling(window, min_periods=max(window // 2, 10)).mean()
        color = _MA_COLORS.get(col, "#ffffff")
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=ma_series,
                name=label, line=dict(color=color, width=1.2),
                opacity=0.9,
            ),
            row=1, col=1,
        )

    # --- Volume bar ---
    vol_colors = [
        "#22c55e" if c >= o else "#ef4444"
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df["date"], y=df["volume"],
            name="Volume",
            marker_color=vol_colors,
            opacity=0.7,
            showlegend=False,
        ),
        row=2, col=1,
    )

    # --- Signal date marker ---
    if signal_date:
        sig_ts = pd.to_datetime(signal_date)
        if df["date"].min() <= sig_ts <= df["date"].max():
            fig.add_vline(
                x=sig_ts,
                line_width=1.5,
                line_dash="dash",
                line_color="#facc15",
                annotation_text="Signal",
                annotation_position="top right",
                annotation_font_color="#facc15",
            )

    fig.update_layout(
        template=_DARK_TEMPLATE,
        title=dict(text=ticker, font_size=16),
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
    )
    fig.update_yaxes(title_text="Harga", row=1, col=1, gridcolor="#1e293b")
    fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor="#1e293b")
    fig.update_xaxes(gridcolor="#1e293b", showgrid=True)

    return fig


def score_radar(row: pd.Series, ticker: str) -> go.Figure:
    """Radar chart komponen score untuk satu ticker."""
    categories = ["Trend", "Momentum", "Breakout", "Volume"]
    col_map = {
        "Trend": "trend_score",
        "Momentum": "momentum_score",
        "Breakout": "breakout_score",
        "Volume": "volume_score",
    }
    values = [float(row.get(col_map[c], 0) or 0) for c in categories]
    values_closed = values + [values[0]]
    cats_closed = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=cats_closed,
        fill="toself",
        fillcolor="rgba(56, 189, 248, 0.25)",
        line=dict(color="#38bdf8", width=2),
        name=ticker,
    ))
    fig.update_layout(
        template=_DARK_TEMPLATE,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], gridcolor="#1e293b", tickfont_size=9),
            angularaxis=dict(gridcolor="#334155"),
            bgcolor="#0f172a",
        ),
        paper_bgcolor="#0f172a",
        height=260,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
    )
    return fig


def history_timeline(df_history: pd.DataFrame) -> go.Figure:
    """Scatter plot total_score vs tanggal untuk history view."""
    if df_history.empty:
        return _empty_figure("Belum ada history")

    signal_colors = {
        "BREAKOUT": "#22c55e",
        "PRE_MARKUP": "#38bdf8",
        "WATCH": "#fb923c",
        "AVOID": "#ef4444",
        "NONE": "#64748b",
    }

    fig = go.Figure()
    for signal, grp in df_history.groupby("signal"):
        color = signal_colors.get(signal, "#64748b")
        hover = (
            grp.get("ticker", pd.Series(dtype=str)).astype(str)
            + " | score: "
            + grp.get("total_score", pd.Series(dtype=float)).round(1).astype(str)
        )
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(grp["date"]),
            y=grp["total_score"],
            mode="markers",
            name=signal,
            marker=dict(color=color, size=9, opacity=0.85),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        template=_DARK_TEMPLATE,
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(title="Total Score", range=[0, 10.5], gridcolor="#1e293b"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="#94a3b8"),
    )
    fig.update_layout(
        template=_DARK_TEMPLATE,
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig
