"""Executive overview and analytical chart sections."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import amount_histogram, risk_distribution, signal_type_chart, timeline_chart
from src.utils import money


def render_dashboard(data: dict[str, pd.DataFrame], scored: pd.DataFrame, vendors: pd.DataFrame) -> None:
    st.markdown("## Portfolio overview")
    total_value = float(data["transactions"].amount.sum())
    high_risk = int(vendors.risk_level.isin(["High", "Critical Review"]).sum()) if not vendors.empty else 0
    duplicate_count = int(data["invoices"].invoice_id.duplicated(keep=False).sum())
    signal_rows = int((scored.risk_score >= 30).sum()) if not scored.empty else 0
    cards = [("Vendors", f"{len(data['vendors']):,}", "Active registry"), ("Transactions", f"{len(data['transactions']):,}", "Synthetic demo records"),
             ("Transaction value", money(total_value), "Across selected dataset"), ("Investigation signals", f"{signal_rows:,}", "Transactions prioritized"),
             ("High-priority vendors", f"{high_risk:,}", "High or critical review"), ("Repeated invoice IDs", f"{duplicate_count:,}", "Potential duplicate records")]
    cols = st.columns(3)
    for index, (label, value, caption) in enumerate(cards):
        with cols[index % 3]:
            st.markdown(f"<div class='metric-panel'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-caption'>{caption}</div></div>", unsafe_allow_html=True)
    st.markdown("### Patterns across the portfolio")
    left, right = st.columns([1, 1.2])
    with left:
        st.plotly_chart(risk_distribution(vendors), use_container_width=True, config={"displayModeBar": False})
        st.plotly_chart(signal_type_chart(scored), use_container_width=True, config={"displayModeBar": False})
    with right:
        st.plotly_chart(timeline_chart(data["transactions"]), use_container_width=True, config={"displayModeBar": False})
        st.plotly_chart(amount_histogram(data["transactions"]), use_container_width=True, config={"displayModeBar": False})
    st.markdown("### Vendors to review")
    table = vendors.sort_values(["risk_score", "total_amount"], ascending=False).head(10).copy()
    if not table.empty:
        table["total_amount"] = table.total_amount.map(money)
        st.dataframe(table[["vendor_id", "risk_level", "risk_score", "transaction_count", "total_amount", "shared_bank_account_count", "shared_address_count"]], use_container_width=True, hide_index=True)
    st.caption("Scores prioritize review effort. A signal is not a finding of misconduct.")