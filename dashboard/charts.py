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
    pct_chg = df["close"].pct_change() * 100
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
                line_color="#facc15",
            )
            fig.add_annotation(
                x=sig_str,
                y=1,
                yref="paper",
                text="Signal",
                showarrow=False,
                xanchor="left",
                font=dict(color="#facc15", size=11),
            )

    fig.update_layout(
        template=_DARK_TEMPLATE,
        title=dict(text=ticker, font_size=16),
        height=540,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
    )
    fig.update_yaxes(title_text="Harga", row=1, col=1, gridcolor="#1e293b")
    fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor="#1e293b")

    # Range selector buttons on the price axis (row 1)
    fig.update_xaxes(
        gridcolor="#1e293b",
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
            bgcolor="#1e293b",
            activecolor="#38bdf8",
            bordercolor="#334155",
            font=dict(color="#94a3b8", size=11),
            x=0.0,
            y=1.0,
        ),
        row=1, col=1,
    )
    fig.update_xaxes(gridcolor="#1e293b", showgrid=True, row=2, col=1)

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
        marker=dict(colors=_COLORS[:len(df)], line=dict(color="#0f172a", width=2)),
        textinfo="label+percent",
        textfont_size=11,
        hovertemplate="<b>%{label}</b><br>%{value:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        template=_DARK_TEMPLATE,
        title=dict(text=f"Komposisi Pemegang Saham — {ticker}", font_size=13),
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0f172a",
        legend=dict(orientation="v", yanchor="middle", y=0.5, font_size=11),
        showlegend=True,
    )
    return fig


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
        "#22c55e" if pd.isna(g) or g >= 0 else "#ef4444"
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
        template=_DARK_TEMPLATE,
        title=dict(text=f"Jumlah Pemegang Saham — {ticker}", font_size=13),
        height=260,
        margin=dict(l=10, r=10, t=40, b=30),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        xaxis=dict(title="Bulan", gridcolor="#1e293b", tickangle=-30),
        yaxis=dict(title="Jumlah Holder", gridcolor="#1e293b"),
        showlegend=False,
    )
    return fig


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
        marker_color="#22c55e",
        opacity=0.85,
        hovertemplate="%{y}: %{x:,.0f} lot<extra>Buy</extra>",
    ))
    fig.add_trace(go.Bar(
        name="Sell",
        y=labels,
        x=[-v for v in sell_vals],   # negatif → ke kiri untuk mirror chart
        orientation="h",
        marker_color="#ef4444",
        opacity=0.85,
        hovertemplate="%{y}: %{customdata:,.0f} lot<extra>Sell</extra>",
        customdata=sell_vals,
    ))

    # Garis vertikal tengah
    fig.add_vline(x=0, line_width=1, line_color="#475569")

    fig.update_layout(
        template=_DARK_TEMPLATE,
        title=dict(text=f"Broker Activity — {ticker}", font_size=13),
        barmode="overlay",
        height=max(200, len(labels) * 32 + 80),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        xaxis=dict(
            title="Lot (Buy →  | ← Sell)",
            gridcolor="#1e293b",
            zeroline=True, zerolinecolor="#475569",
        ),
        yaxis=dict(gridcolor="#1e293b", autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
    )
    return fig


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
        template=_DARK_TEMPLATE,
        title=dict(text=f"Daily Net Broker Flow — {ticker}", font=dict(size=12, color="#94a3b8")),
        barmode="group",
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        xaxis=dict(title="Tanggal", gridcolor="#1e293b", tickangle=-30),
        yaxis=dict(title="Net Lot", gridcolor="#1e293b", zeroline=True, zerolinecolor="#475569"),
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=10)),
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
            showarrow=False, font=dict(size=13, color="#94a3b8"),
        )
        fig.update_layout(
            template=_DARK_TEMPLATE,
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            height=220,
            margin=dict(l=10, r=10, t=30, b=10),
            title=dict(text=f"{label} ({ticker})", font=dict(size=13, color="#e2e8f0")),
        )
        return fig

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
        textfont=dict(size=11, color="#e2e8f0"),
    ))

    fig.update_layout(
        template=_DARK_TEMPLATE,
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        height=220,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text=f"{label} ({unit})", font=dict(size=12, color="#94a3b8")),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", tickfont=dict(size=10)),
        xaxis=dict(tickfont=dict(size=10)),
        showlegend=False,
    )
    return fig


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
            showarrow=False, font=dict(size=13, color="#94a3b8"),
        )
        fig.update_layout(
            template=_DARK_TEMPLATE,
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            height=280, margin=dict(l=10, r=10, t=30, b=10),
        )
        return fig

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
        template=_DARK_TEMPLATE,
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        height=310,
        margin=dict(l=10, r=10, t=50, b=10),
        title=dict(text=f"{ticker} — Harga Jangka Panjang", font=dict(size=12, color="#94a3b8")),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", tickformat=","),
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
                bgcolor="#1e293b",
                activecolor="#38bdf8",
                bordercolor="#334155",
                font=dict(color="#94a3b8", size=11),
            ),
            rangeslider=dict(
                visible=True,
                bgcolor="#0f172a",
                thickness=0.06,
            ),
            type="date",
        ),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10)),
    )
    return fig
