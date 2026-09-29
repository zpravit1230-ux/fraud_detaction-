"""Searchable and filterable transaction investigation table."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_transactions(scored: pd.DataFrame) -> None:
    st.markdown("## Transaction investigation")
    if scored.empty:
        st.info("No transaction records are available.")
        return
    controls = st.columns(3)
    priority = controls[0].multiselect("Review priority", ["Low", "Medium", "High", "Critical Review"], default=["High", "Critical Review"])
    vendors = controls[1].multiselect("Vendor", sorted(scored.vendor_id.dropna().unique()))
    employees = controls[2].multiselect("Employee", sorted(scored.employee_id.dropna().unique()))
    amount_range = st.slider("Amount range", min_value=0, max_value=max(1000, int(scored.amount.max())), value=(0, max(1000, int(scored.amount.max()))), step=1000, format="$%d")
    date_min, date_max = scored.transaction_date.min().date(), scored.transaction_date.max().date()
    selected_dates = st.date_input("Transaction date", value=(date_min, date_max), min_value=date_min, max_value=date_max)
    payment_methods = st.multiselect("Payment method", sorted(scored.payment_method.dropna().unique()))
    filtered = scored.loc[scored.risk_level.isin(priority)] if priority else scored.copy()
    filtered = filtered.loc[filtered.amount.between(*amount_range)]
    if vendors: filtered = filtered.loc[filtered.vendor_id.isin(vendors)]
    if employees: filtered = filtered.loc[filtered.employee_id.isin(employees)]
    if payment_methods: filtered = filtered.loc[filtered.payment_method.isin(payment_methods)]
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        filtered = filtered.loc[filtered.transaction_date.dt.date.between(selected_dates[0], selected_dates[1])]
    st.caption(f"Showing {len(filtered):,} of {len(scored):,} transactions · unusual records are investigation leads")
    view = filtered.sort_values(["risk_score", "transaction_date"], ascending=[False, False]).copy()
    view["investigation_signals"] = view.investigation_signals.map(lambda signals: " · ".join(signals))
    st.dataframe(view[["transaction_id", "vendor_id", "employee_id", "transaction_date", "amount", "invoice_id", "payment_method", "anomaly_score", "risk_score", "risk_level", "investigation_signals"]], use_container_width=True, hide_index=True, height=560,
                 column_config={"amount": st.column_config.NumberColumn(format="$%.2f"), "risk_score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"), "anomaly_score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f")})
    st.download_button("Export filtered transactions", view.to_csv(index=False), file_name="transaction_investigation_results.csv", mime="text/csv")