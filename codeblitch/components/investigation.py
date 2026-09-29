"""Vendor profile and evidence-led investigation workflow."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.graph_view import render_graph
from src.explainability import vendor_explanation
from src.graph_engine import graph_metrics
from src.utils import money, vendor_report_html


def render_vendor_investigation(data: dict[str, pd.DataFrame], scored: pd.DataFrame, vendor_scores: pd.DataFrame, graph) -> None:
    st.markdown("## Vendor investigation")
    if data["vendors"].empty:
        st.warning("No vendor records are available.")
        return
    controls = st.columns([1.2, 1, 1])
    search = controls[0].text_input("Search vendor", placeholder="ID or vendor name", key="vendor_search").strip().lower()
    level = controls[1].selectbox("Review priority", ["All", "Low", "Medium", "High", "Critical Review"])
    choices = data["vendors"].merge(vendor_scores, on="vendor_id", how="left", suffixes=("", "_risk"))
    if level != "All": choices = choices.loc[choices.risk_level == level]
    if search: choices = choices.loc[choices.vendor_id.str.lower().str.contains(search) | choices.vendor_name.str.lower().str.contains(search)]
    if choices.empty:
        st.info("No vendors match the current search and priority filters.")
        return
    default_index = choices.vendor_id.tolist().index("VENDOR_017") if "VENDOR_017" in choices.vendor_id.tolist() and not search else 0
    vendor_id = controls[2].selectbox("Selected vendor", choices.vendor_id.tolist(), index=default_index, format_func=lambda value: f"{value} · {choices.loc[choices.vendor_id.eq(value), 'vendor_name'].iloc[0]}")
    vendor = data["vendors"].loc[data["vendors"].vendor_id.eq(vendor_id)].iloc[0]
    transactions = scored.loc[scored.vendor_id.eq(vendor_id)].copy()
    summary = vendor_scores.loc[vendor_scores.vendor_id.eq(vendor_id)]
    score = float(summary.iloc[0].risk_score) if not summary.empty else 0
    risk = str(summary.iloc[0].risk_level) if not summary.empty else "Low"
    signals = vendor_explanation(vendor_id, vendor, transactions, data["vendors"])
    st.markdown(f"### {vendor.vendor_name} <span class='entity-id'>{vendor_id}</span>", unsafe_allow_html=True)
    st.caption("Investigation lead · risk score is a review-priority indicator, not a conclusion.")
    metric_cols = st.columns(4)
    age = max(0, (pd.Timestamp("2026-09-28") - pd.to_datetime(vendor.registration_date, errors="coerce")).days) if pd.notna(vendor.registration_date) else 0
    values = [("Priority score", f"{score:.1f} / 100 · {risk}"), ("Vendor age", f"{age:,} days"), ("Transactions", f"{len(transactions):,}"), ("Total value", money(transactions.amount.sum()))]
    for column, (label, value) in zip(metric_cols, values):
        with column: st.metric(label, value)
    details, explanation = st.columns([1, 1])
    with details:
        st.markdown("#### Entity profile")
        st.dataframe(pd.DataFrame({"Field": ["Registration date", "Industry", "Country", "Address", "Bank account", "Average transaction"], "Value": [str(vendor.registration_date), vendor.industry, vendor.country, vendor.address_id, vendor.bank_account_id, money(transactions.amount.mean() if len(transactions) else 0)]}), use_container_width=True, hide_index=True)
    with explanation:
        st.markdown("#### Why this vendor is prioritized")
        for signal in signals: st.markdown(f"- {signal}")
    metrics = graph_metrics(graph, vendor_id)
    st.markdown(f"**Relationship footprint:** {metrics['connected_vendors']} connected vendor(s) · {metrics['connected_employees']} employee(s) · {metrics['component_size']} entities in component")
    st.markdown("#### Transaction timeline")
    if not transactions.empty:
        st.line_chart(transactions.sort_values("transaction_date").set_index("transaction_date")["amount"], height=210, color="#29a89c")
        recent = transactions.sort_values("transaction_date", ascending=False).head(15).copy()
        recent["signals"] = recent.apply(lambda row: " · ".join(row.investigation_signals), axis=1)
        st.dataframe(recent[["transaction_id", "transaction_date", "amount", "employee_id", "invoice_id", "risk_score", "risk_level", "signals"]], use_container_width=True, hide_index=True)
    graph_col, export_col = st.columns([1.3, 0.7])
    with graph_col:
        st.markdown("#### Connected entities")
        render_graph(graph, vendor_id, depth=2, height=480)
    with export_col:
        st.markdown("#### Investigation signals")
        st.dataframe(pd.DataFrame({"Review area": [f"Signal {index + 1}" for index in range(len(signals))], "Evidence": signals}), use_container_width=True, hide_index=True)
        st.download_button("Download vendor report", vendor_report_html(vendor, score, risk, signals, transactions), file_name=f"{vendor_id.lower()}_investigation.html", mime="text/html", use_container_width=True)
        st.download_button("Export vendor transactions", transactions.to_csv(index=False), file_name=f"{vendor_id.lower()}_transactions.csv", mime="text/csv", use_container_width=True)