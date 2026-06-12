"""Plotly chart builders untuk dashboard.

Charts read their colours from module-level theme globals so they match the
active dashboard theme (light/dark). Call ``set_chart_theme(palette)`` once per
run (the app does this) before building figures. Defaults are the dark palette
so importing the module standalone still produces sensible charts.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Theme globals (overridden by set_chart_theme) ---
_TEMPLATE = "plotly_dark"
_PAPER    = "#0F1115"   # figure background
_PLOT     = "#0F1115"   # plotting area background
_SURFACE  = "#161A20"   # raised plot area (bars / fundamentals)
_GRID     = "#222831"   # gridlines
_TEXT     = "#E6E8EC"   # primary text
_MUTED    = "#9BA4B0"   # secondary text / titles
_UP       = "#22C55E"   # bullish candle
_DOWN     = "#EF4444"   # bearish candle
_ACCENT   = "#3B82F6"   # markers / highlights

# Moving-average accents read well on both light and dark surfaces.
_MA_COLORS = {
    "ma20": "#38bdf8",   # sky blue
    "ma50": "#f59e0b",   # amber
    "ma200": "#8b5cf6",  # violet
}


def set_chart_theme(palette: dict) -> None:
    """Point all chart builders at the active theme palette.

    Args:
        palette: dict from dashboard.theme.chart_palette(mode) with keys
            template, paper, plot, surface, grid, text, muted, up, down.
    """
    global _TEMPLATE, _PAPER, _PLOT, _SURFACE, _GRID, _TEXT, _MUTED, _UP, _DOWN, _ACCENT
    _TEMPLATE = palette.get("template", _TEMPLATE)
    _PAPER    = palette.get("paper",    _PAPER)
    _PLOT     = palette.get("plot",     _PLOT)
    _SURFACE  = palette.get("surface",  _SURFACE)
    _GRID     = palette.get("grid",     _GRID)
    _TEXT     = palette.get("text",     _TEXT)
    _MUTED    = palette.get("muted",    _MUTED)
    _UP       = palette.get("up",       _UP)
    _DOWN     = palette.get("down",     _DOWN)
    _ACCENT   = palette.get("accent",   _ACCENT)


def _apply_text_theme(_f: go.Figure) -> go.Figure:
    """Force readable, theme-matched colours on EVERY text layer of a figure:
    global font, axis ticks + titles, legend, and hover tooltips. Called on every
    figure before it is returned so light/dark mode text always meets contrast.
    """
    _f.update_layout(
        font=dict(color=_TEXT, size=12),
        title_font=dict(color=_TEXT),  # title isn't covered by layout.font; Streamlit's
                                       # pinned-dark theme would otherwise tint it.
        legend=dict(font=dict(color=_TEXT)),
        hoverlabel=dict(bgcolor=_SURFACE, bordercolor=_GRID, font=dict(color=_TEXT, size=12)),
    )
    # Cartesian axes (no-op on polar/pie figures).
    try:
        _f.update_xaxes(tickfont=dict(color=_MUTED), title_font=dict(color=_MUTED))
        _f.update_yaxes(tickfont=dict(color=_MUTED), title_font=dict(color=_MUTED))
    except Exception:  # noqa: BLE001
        pass
    return _f


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
    pct_chg = df["close"].pct_change() * 100
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color=_UP,
            decreasing_line_color=_DOWN,
            increasing_fillcolor=_UP,
            decreasing_fillcolor=_DOWN,
        ),
        row=1, col=1,
    )

    # Invisible scatter overlay for % change hover (Candlestick doesn't support hovertemplate)
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["close"],
            mode="markers",
            marker=dict(opacity=0, size=6, color="rgba(0,0,0,0)"),
            customdata=pct_chg.values,
            name="Δ%",
            hovertemplate=(
                "<b>%{x|%d %b %Y}</b><br>"
                "Close: %{y:,.0f}<br>"
                "Perubahan: %{customdata:+.2f}%"
                "<extra></extra>"
            ),
            showlegend=False,
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
        _UP if c >= o else _DOWN
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
    # Pass x as ISO string, NOT pd.Timestamp, to avoid:
    #   TypeError: Addition/subtraction of integers and integer-arrays
    #   with Timestamp is no longer supported  (pandas 2.0+, Plotly
    #   annotation-anchor arithmetic internally hits this).
    # Label text added via a separate add_annotation call.
    if signal_date:
        sig_ts = pd.to_datetime(signal_date)
        if df["date"].min() <= sig_ts <= df["date"].max():
            sig_str = sig_ts.strftime("%Y-%m-%d")
            fig.add_vline(
                x=sig_str,
                line_width=1.5,
                line_dash="dash",
                line_color=_ACCENT,
            )
            fig.add_annotation(
                x=sig_str,
                y=1,
                yref="paper",
                text="Signal",
                showarrow=False,
                xanchor="left",
                bgcolor=_SURFACE,
                bordercolor=_ACCENT,
                borderwidth=1,
                borderpad=2,
                font=dict(color=_TEXT, size=11),
            )

    fig.update_layout(
        template=_TEMPLATE,
        title=dict(text=ticker, font_size=16),
        height=540,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PLOT,
    )
    fig.update_yaxes(title_text="Harga", row=1, col=1, gridcolor=_GRID)
    fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor=_GRID)

    # Range selector buttons on the price axis (row 1)
    fig.update_xaxes(
        gridcolor=_GRID,
        showgrid=True,
        rangeselector=dict(
            buttons=[
                dict(count=7,  label="1W", step="day",   stepmode="backward"),
                dict(count=1,  label="1M", step="month", stepmode="backward"),
                dict(count=3,  label="3M", step="month", stepmode="backward"),
                dict(count=6,  label="6M", step="month", stepmode="backward"),
                dict(count=1,  label="1Y", step="year",  stepmode="backward"),
                dict(count=1,  label="YTD",step="year",  stepmode="todate"),
                dict(step="all", label="ALL"),
            ],
            bgcolor=_SURFACE,
            activecolor="#38bdf8",
            bordercolor=_GRID,
            font=dict(color=_MUTED, size=11),
            x=0.0,
            y=1.0,
        ),
        row=1, col=1,
    )
    fig.update_xaxes(gridcolor=_GRID, showgrid=True, row=2, col=1)

    return _apply_text_theme(fig)


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
        template=_TEMPLATE,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], gridcolor=_GRID, tickfont_size=9),
            angularaxis=dict(gridcolor=_GRID),
            bgcolor=_PLOT,
        ),
        paper_bgcolor=_PAPER,
        height=260,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
    )
    return _apply_text_theme(fig)


def history_timeline(df_history: pd.DataFrame) -> go.Figure:
    """Scatter plot total_score vs tanggal untuk history view."""
    if df_history.empty:
        return _empty_figure("Belum ada history")

    signal_colors = {
        "BREAKOUT": _UP,
        "PRE_MARKUP": "#38bdf8",
        "WATCH": "#fb923c",
        "AVOID": _DOWN,
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
        template=_TEMPLATE,
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PLOT,
        xaxis=dict(gridcolor=_GRID),
        yaxis=dict(title="Total Score", range=[0, 10.5], gridcolor=_GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return _apply_text_theme(fig)


def shareholder_pie(df_composition: pd.DataFrame, ticker: str) -> go.Figure:
    """Pie chart for shareholder composition (local vs foreign breakdown).

    Args:
        df_composition: DataFrame with columns: category, shares, percentage
        ticker: label for the chart title
    """
    if df_composition.empty:
        return _empty_figure(f"Tidak ada data komposisi pemegang saham untuk {ticker}")

    df = df_composition.copy()
    pct_col = "percentage" if "percentage" in df.columns else df.columns[-1]
    lbl_col = "category" if "category" in df.columns else df.columns[0]

    _COLORS = ["#38bdf8", "#fb923c", "#a78bfa", "#4ade80", "#f472b6", "#facc15"]

    fig = go.Figure(go.Pie(
        labels=df[lbl_col],
        values=df[pct_col],
        hole=0.40,
        marker=dict(colors=_COLORS[:len(df)], line=dict(color=_PAPER, width=2)),
        textinfo="label+percent",
        textfont_size=11,
        hovertemplate="<b>%{label}</b><br>%{value:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        template=_TEMPLATE,
        title=dict(text=f"Komposisi Pemegang Saham — {ticker}", font_size=13),
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=_PAPER,
        legend=dict(orientation="v", yanchor="middle", y=0.5, font_size=11),
        showlegend=True,
    )
    return _apply_text_theme(fig)


def monthly_holders_chart(df_monthly: pd.DataFrame, ticker: str) -> go.Figure:
    """Line chart of monthly shareholder count with MoM growth % as hover labels.

    Args:
        df_monthly: DataFrame with columns: month (YYYY-MM), shareholder_count, growth_pct
        ticker: label for the chart title
    """
    if df_monthly.empty:
        return _empty_figure(f"Tidak ada data bulanan pemegang saham untuk {ticker}")

    df = df_monthly.copy()

    # Build hover text
    def _hover(row: pd.Series) -> str:
        base = f"{row['month']}<br>{int(row['shareholder_count']):,} holder"
        if pd.notna(row.get("growth_pct")):
            sign = "+" if row["growth_pct"] >= 0 else ""
            color = "lime" if row["growth_pct"] >= 0 else "#f87171"
            base += f"<br><span style='color:{color}'>{sign}{row['growth_pct']:.2f}% MoM</span>"
        return base

    hover_texts = df.apply(_hover, axis=1).tolist()

    # Marker colors: green if growth ≥ 0, red otherwise
    colors = [
        _UP if pd.isna(g) or g >= 0 else _DOWN
        for g in df.get("growth_pct", pd.Series(dtype=float))
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["month"],
        y=df["shareholder_count"],
        mode="lines+markers",
        line=dict(color="#38bdf8", width=2),
        marker=dict(color=colors, size=8),
        hovertemplate="%{text}<extra></extra>",
        text=hover_texts,
        name="Shareholders",
    ))

    fig.update_layout(
        template=_TEMPLATE,
        title=dict(text=f"Jumlah Pemegang Saham — {ticker}", font_size=13),
        height=260,
        margin=dict(l=10, r=10, t=40, b=30),
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PLOT,
        xaxis=dict(title="Bulan", gridcolor=_GRID, tickangle=-30),
        yaxis=dict(title="Jumlah Holder", gridcolor=_GRID),
        showlegend=False,
    )
    return _apply_text_theme(fig)


def broker_chart(df_broker: pd.DataFrame, ticker: str) -> go.Figure:
    """Grouped bar chart top broker buy vs sell activity.

    Args:
        df_broker: DataFrame dari data_loader.load_broker_for_ticker()
                   Kolom: broker_code, broker_name, buy_lot, sell_lot, net_lot
        ticker: label untuk judul
    """
    if df_broker.empty:
        return _empty_figure(f"Tidak ada data broker untuk {ticker}")

    df = df_broker.copy()

    # Gunakan broker_name jika ada, fallback ke broker_code
    label_col = "broker_name" if "broker_name" in df.columns else "broker_code"
    labels = df[label_col].astype(str).tolist()
    buy_col = "buy_lot" if "buy_lot" in df.columns else df.columns[2]
    sell_col = "sell_lot" if "sell_lot" in df.columns else df.columns[3]

    buy_vals = pd.to_numeric(df[buy_col], errors="coerce").fillna(0).tolist()
    sell_vals = pd.to_numeric(df[sell_col], errors="coerce").fillna(0).tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Buy",
        y=labels,
        x=buy_vals,
        orientation="h",
        marker_color=_UP,
        opacity=0.85,
        hovertemplate="%{y}: %{x:,.0f} lot<extra>Buy</extra>",
    ))
    fig.add_trace(go.Bar(
        name="Sell",
        y=labels,
        x=[-v for v in sell_vals],   # negatif → ke kiri untuk mirror chart
        orientation="h",
        marker_color=_DOWN,
        opacity=0.85,
        hovertemplate="%{y}: %{customdata:,.0f} lot<extra>Sell</extra>",
        customdata=sell_vals,
    ))

    # Garis vertikal tengah
    fig.add_vline(x=0, line_width=1, line_color="#475569")

    fig.update_layout(
        template=_TEMPLATE,
        title=dict(text=f"Broker Activity — {ticker}", font_size=13),
        barmode="overlay",
        height=max(200, len(labels) * 32 + 80),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PLOT,
        xaxis=dict(
            title="Lot (Buy →  | ← Sell)",
            gridcolor=_GRID,
            zeroline=True, zerolinecolor="#475569",
        ),
        yaxis=dict(gridcolor=_GRID, autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
    )
    return _apply_text_theme(fig)


def broker_net_flow_chart(
    broker_history_df: pd.DataFrame,
    ticker: str,
) -> go.Figure:
    """Stacked bar chart of daily net broker flow by type (foreign / big-local / retail).

    Args:
        broker_history_df: Multi-day broker DataFrame from ``load_broker_history()``.
                           Must have: date, broker_code, net_lot.
        ticker: Ticker label for the chart title.

    Returns:
        Plotly Figure, or empty figure if data unavailable.
    """
    if broker_history_df is None or broker_history_df.empty:
        return _empty_figure(f"Tidak ada data broker flow untuk {ticker}")

    df = broker_history_df.copy()
    if "broker_code" not in df.columns or "net_lot" not in df.columns:
        return _empty_figure(f"Kolom broker tidak lengkap untuk {ticker}")

    from stock_scanner.pipeline.broker_intelligence import classify_broker
    df["broker_type"] = df["broker_code"].apply(classify_broker)
    df["net_lot"]     = pd.to_numeric(df["net_lot"], errors="coerce").fillna(0)

    if "date" not in df.columns:
        return _empty_figure("Data broker tidak memiliki kolom tanggal")

    df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["_date"])

    # Aggregate daily net by broker_type
    agg = (
        df.groupby(["_date", "broker_type"])["net_lot"]
        .sum()
        .reset_index()
        .pivot(index="_date", columns="broker_type", values="net_lot")
        .fillna(0)
        .sort_index()
    )

    type_colors = {
        "foreign":   "#38bdf8",
        "big_local": "#fb923c",
        "local":     "#94a3b8",
        "unknown":   "#475569",
    }
    type_labels = {
        "foreign":   "🌍 Asing",
        "big_local": "🏦 Big Local",
        "local":     "🏠 Retail/Lokal",
        "unknown":   "❓ Lainnya",
    }

    fig = go.Figure()
    for btype in ["foreign", "big_local", "local", "unknown"]:
        if btype not in agg.columns:
            continue
        fig.add_trace(go.Bar(
            x=agg.index,
            y=agg[btype],
            name=type_labels.get(btype, btype),
            marker_color=type_colors.get(btype, "#475569"),
            opacity=0.85,
            hovertemplate=f"<b>{type_labels.get(btype, btype)}</b><br>%{{x|%d %b}}: %{{y:+,.0f}} lot<extra></extra>",
        ))

    # Zero line
    fig.add_hline(y=0, line_width=1, line_color="#475569")

    fig.update_layout(
        template=_TEMPLATE,
        title=dict(text=f"Daily Net Broker Flow — {ticker}", font=dict(size=12, color=_MUTED)),
        barmode="group",
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=_PAPER,
        plot_bgcolor=_SURFACE,
        xaxis=dict(title="Tanggal", gridcolor=_GRID, tickangle=-30),
        yaxis=dict(title="Net Lot", gridcolor=_GRID, zeroline=True, zerolinecolor="#475569"),
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=10)),
    )
    return _apply_text_theme(fig)


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color=_MUTED),
    )
    fig.update_layout(
        template=_TEMPLATE,
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PLOT,
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return _apply_text_theme(fig)


def fundamental_trend_chart(
    financial_data: dict,
    ticker: str,
    metric: str = "revenue",
    label: str = "Revenue",
    unit: str = "Rp Triliun",
) -> go.Figure:
    """Bar chart for multi-period fundamental data (revenue, net income, etc.).

    Args:
        financial_data: dict from long_term.compare_financial_statements()
        ticker        : ticker label
        metric        : key in financial_data["annual"] (e.g. "revenue", "net_income")
        label         : display name for the metric
        unit          : y-axis unit label (default "Rp Triliun")

    Returns:
        Plotly Figure, or empty figure if data unavailable.
    """
    series = financial_data.get("annual", {}).get(metric, [])

    if not series:
        fig = go.Figure()
        fig.add_annotation(
            text="Data tidak tersedia",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13, color=_MUTED),
        )
        fig.update_layout(
            template=_TEMPLATE,
            paper_bgcolor=_PAPER,
            plot_bgcolor=_PLOT,
            height=220,
            margin=dict(l=10, r=10, t=30, b=10),
            title=dict(text=f"{label} ({ticker})", font=dict(size=13, color=_TEXT)),
        )
        return _apply_text_theme(fig)

    years  = [item["year"] for item in series]
    values = [item["value"] for item in series]

    # Color bars: most recent = accent, older = muted
    colors = ["#38bdf8"] + ["#334155"] * (len(years) - 1)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=years,
        y=values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
        textfont=dict(size=11, color=_TEXT),
    ))

    fig.update_layout(
        template=_TEMPLATE,
        paper_bgcolor=_PAPER,
        plot_bgcolor=_SURFACE,
        height=220,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text=f"{label} ({unit})", font=dict(size=12, color=_MUTED)),
        yaxis=dict(showgrid=True, gridcolor=_GRID, tickfont=dict(size=10)),
        xaxis=dict(tickfont=dict(size=10)),
        showlegend=False,
    )
    return _apply_text_theme(fig)


def price_chart_longterm(
    df_raw: pd.DataFrame,
    ticker: str,
) -> go.Figure:
    """Simplified long-term price chart showing all available history.

    Uses a line chart (cleaner for multi-year view) with annual MA.

    Args:
        df_raw: Full OHLCV DataFrame (no lookback limit — shows all years).
        ticker: ticker label for title.

    Returns:
        Plotly Figure.
    """
    if df_raw is None or df_raw.empty or "close" not in df_raw.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Data OHLCV tidak tersedia",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13, color=_MUTED),
        )
        fig.update_layout(
            template=_TEMPLATE,
            paper_bgcolor=_PAPER, plot_bgcolor=_PLOT,
            height=280, margin=dict(l=10, r=10, t=30, b=10),
        )
        return _apply_text_theme(fig)

    df = df_raw.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

    date_col = "date" if "date" in df.columns else df.index

    xs = df[date_col] if "date" in df.columns else df.index
    pct_chg_lt = df["close"].pct_change() * 100

    fig = go.Figure()

    # Price line with hover %
    fig.add_trace(go.Scatter(
        x=xs,
        y=df["close"],
        mode="lines",
        name="Harga",
        line=dict(color="#38bdf8", width=1.5),
        customdata=pct_chg_lt.values,
        hovertemplate=(
            "<b>%{x|%d %b %Y}</b><br>"
            "Harga: %{y:,.0f}<br>"
            "Perubahan: %{customdata:+.2f}%"
            "<extra></extra>"
        ),
    ))

    # MA 200
    if len(df) >= 200:
        ma200 = df["close"].rolling(200).mean()
        fig.add_trace(go.Scatter(
            x=xs,
            y=ma200,
            mode="lines",
            name="MA200",
            line=dict(color="#a78bfa", width=1, dash="dot"),
            hovertemplate="MA200: %{y:,.0f}<extra></extra>",
        ))

    fig.update_layout(
        template=_TEMPLATE,
        paper_bgcolor=_PAPER,
        plot_bgcolor=_SURFACE,
        height=310,
        margin=dict(l=10, r=10, t=50, b=10),
        title=dict(text=f"{ticker} — Harga Jangka Panjang", font=dict(size=12, color=_MUTED)),
        yaxis=dict(showgrid=True, gridcolor=_GRID, tickformat=","),
        xaxis=dict(
            showgrid=False,
            rangeselector=dict(
                buttons=[
                    dict(count=1,  label="1M",  step="month", stepmode="backward"),
                    dict(count=3,  label="3M",  step="month", stepmode="backward"),
                    dict(count=6,  label="6M",  step="month", stepmode="backward"),
                    dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
                    dict(count=3,  label="3Y",  step="year",  stepmode="backward"),
                    dict(count=1,  label="YTD", step="year",  stepmode="todate"),
                    dict(step="all", label="ALL"),
                ],
                bgcolor=_SURFACE,
                activecolor="#38bdf8",
                bordercolor=_GRID,
                font=dict(color=_MUTED, size=11),
            ),
            rangeslider=dict(
                visible=True,
                bgcolor=_PLOT,
                thickness=0.06,
            ),
            type="date",
        ),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10)),
    )
    return _apply_text_theme(fig)


# ---------------------------------------------------------------------------
# BROKER ANALYTICS CHARTS (STEP 3)
# ---------------------------------------------------------------------------

def broker_concentration_pie(
    df_broker: pd.DataFrame,
    broker_config: dict | None = None,
) -> go.Figure:
    """Pie chart: broker concentration by broker type.

    Tujuan: Visualisasi kontribusi setiap broker group (foreign, institution, retail, local, big_local)
    terhadap total net lot positif atau aktivitas keseluruhan.

    Args:
        df_broker       : DataFrame dengan kolom broker_code, broker_type, net_lot
        broker_config   : Optional dict untuk mengambil warna dari config
                         (format: config['display']['chart_colors'])

    Returns:
        Plotly Figure pie chart.
    """
    if df_broker.empty or "broker_type" not in df_broker.columns:
        return _empty_figure("Tidak ada data broker untuk chart.")

    # Aggregate by broker_type
    df = df_broker.copy()
    df["net_lot"] = pd.to_numeric(df["net_lot"], errors="coerce").fillna(0)

    agg = df.groupby("broker_type")["net_lot"].sum().reset_index()
    agg = agg[agg["net_lot"] != 0]  # Filter zeros
    agg = agg.sort_values("net_lot", key=abs, ascending=False)

    if agg.empty:
        return _empty_figure("Semua broker memiliki net_lot = 0.")

    # Color mapping
    colors = {
        "foreign": "#3b82f6",
        "institution": "#8b5cf6",
        "retail": "#f97316",
        "big_local": "#64748b",
        "local": "#94a3b8",
    }

    # Override dari config jika tersedia
    if broker_config and "display" in broker_config and "chart_colors" in broker_config["display"]:
        chart_colors = broker_config["display"]["chart_colors"]
        colors.update(chart_colors)

    # Build chart
    labels = agg["broker_type"].tolist()
    values = agg["net_lot"].tolist()
    plot_colors = [colors.get(btype, "#64748b") for btype in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=plot_colors, line=dict(color=_PAPER, width=2)),
        hovertemplate="<b>%{label}</b><br>Net Lot: %{value:,.0f}<br>%{percent}<extra></extra>",
        textposition="auto",
        textinfo="label+percent",
        textfont=dict(size=11, color=_TEXT),
    )])

    fig.update_layout(
        template=_TEMPLATE,
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PLOT,
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Konsentrasi Broker (Net Lot)", font=dict(size=12, color=_MUTED)),
        showlegend=True,
        legend=dict(font=dict(size=10), x=1.02, y=1),
    )
    return _apply_text_theme(fig)


def broker_history_trend_line(
    df_history: pd.DataFrame,
) -> go.Figure:
    """Line chart: broker history dengan 3 series (daily, rolling 5d, rolling 20d).

    Tujuan: Visualisasi trend aktivitas broker sepanjang periode dengan rolling averages.

    Args:
        df_history  : DataFrame hasil aggregate_broker_history() dengan kolom:
                     date, net_lot, rolling_5d, rolling_20d, cumulative_net_lot

    Returns:
        Plotly Figure line chart 3 series.
    """
    if df_history.empty or "date" not in df_history.columns:
        return _empty_figure("Tidak ada history data untuk chart.")

    df = df_history.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Ensure numeric columns
    for col in ["net_lot", "rolling_5d", "rolling_20d"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    fig = go.Figure()

    # Daily net_lot
    if "net_lot" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["net_lot"],
            mode="lines+markers",
            name="Daily Net",
            line=dict(color="#38bdf8", width=2),
            marker=dict(size=6),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Daily: %{y:,.0f}<extra></extra>",
        ))

    # Rolling 5d
    if "rolling_5d" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["rolling_5d"],
            mode="lines",
            name="5D Rolling",
            line=dict(color="#fb923c", width=2, dash="dash"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>5D Avg: %{y:,.0f}<extra></extra>",
        ))

    # Rolling 20d
    if "rolling_20d" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["rolling_20d"],
            mode="lines",
            name="20D Rolling",
            line=dict(color="#a78bfa", width=2, dash="dot"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>20D Avg: %{y:,.0f}<extra></extra>",
        ))

    fig.update_layout(
        template=_TEMPLATE,
        paper_bgcolor=_PAPER,
        plot_bgcolor=_SURFACE,
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
        title=dict(text="Trend Aktivitas Broker", font=dict(size=12, color=_MUTED)),
        xaxis=dict(title="Tanggal", showgrid=True, gridcolor=_GRID),
        yaxis=dict(title="Net Lot", showgrid=True, gridcolor=_GRID, zeroline=True, zerolinecolor="#475569"),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10)),
        hovermode="x unified",
    )
    return _apply_text_theme(fig)
