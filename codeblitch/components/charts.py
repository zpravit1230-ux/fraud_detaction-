"""Plotly chart builders shared across investigation pages."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PALETTE = ["#ed7054", "#e3ad55", "#29a89c", "#5e91cf", "#ad79d0"]
LAYOUT = {"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)", "font": {"color": "#dce5e5", "family": "Aptos, Segoe UI, sans-serif"}, "margin": {"l": 12, "r": 12, "t": 32, "b": 8}}


def risk_distribution(vendors: pd.DataFrame):
    counts = vendors.risk_level.value_counts().reindex(["Low", "Medium", "High", "Critical Review"], fill_value=0).reset_index()
    counts.columns = ["Review priority", "Vendors"]
    fig = px.bar(counts, x="Review priority", y="Vendors", color="Review priority", color_discrete_sequence=PALETTE)
    fig.update_layout(**LAYOUT, showlegend=False, xaxis_title=None, yaxis_title="Vendor count")
    return fig


def timeline_chart(transactions: pd.DataFrame):
    if transactions.empty: return go.Figure().update_layout(**LAYOUT)
    monthly = transactions.assign(month=transactions.transaction_date.dt.to_period("M").dt.to_timestamp()).groupby("month", as_index=False).agg(value=("amount", "sum"), count=("transaction_id", "count"))
    fig = px.area(monthly, x="month", y="value", color_discrete_sequence=["#29a89c"], markers=True)
    fig.update_layout(**LAYOUT, xaxis_title=None, yaxis_title="Transaction value ($)")
    return fig


def amount_histogram(transactions: pd.DataFrame):
    fig = px.histogram(transactions, x="amount", nbins=42, color_discrete_sequence=["#e3ad55"])
    fig.update_layout(**LAYOUT, xaxis_title="Amount ($)", yaxis_title="Transactions")
    return fig


def signal_type_chart(scored: pd.DataFrame):
    signal_counts = {
        "Model anomaly": int((scored.anomaly_score >= 55).sum()) if len(scored) else 0,
        "Shared bank": int((scored.shared_bank_account_count > 0).sum()) if len(scored) else 0,
        "Shared address": int((scored.shared_address_count > 0).sum()) if len(scored) else 0,
        "Potential duplicate": int((scored.duplicate_invoice_count > 0).sum()) if len(scored) else 0,
        "High amount vs history": int((scored.amount_vs_historical_average >= 2.5).sum()) if len(scored) else 0,
    }
    frame = pd.DataFrame({"Signal type": list(signal_counts), "Records": list(signal_counts.values())})
    fig = px.bar(frame, x="Records", y="Signal type", orientation="h", color="Signal type", color_discrete_sequence=PALETTE)
    fig.update_layout(**LAYOUT, showlegend=False, xaxis_title="Transaction records", yaxis_title=None)
    return fig