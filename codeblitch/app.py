"""AI-assisted corporate fraud investigation dashboard."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px

from components.charts import PALETTE, LAYOUT
from components.dashboard import render_dashboard
from components.graph_view import render_graph
from components.investigation import render_vendor_investigation
from components.transaction_view import render_transactions
from data.generate_data import write_demo_data
from src.anomaly_detection import CONTAMINATION, MODEL_DIR, load_model, score_anomalies, train_model
from src.data_loader import DATA_DIR, load_data, validate_data
from src.database import DB_PATH, get_investigation_signals, initialize_database, insert_data, save_risk_score
from src.explainability import transaction_signals
from src.feature_engineering import engineer_features
from src.graph_engine import build_graph, graph_metrics
from src.risk_engine import aggregate_vendor_risks, score_transactions
from src.utils import search_entities

st.set_page_config(page_title="AI Fraud Investigation System", page_icon="◉", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--ink:#dce5e5;--muted:#8a9a9e;--panel:#132126;--line:#26393d;--mint:#54d6bb;--coral:#ed7054;--amber:#e3ad55}
html,body,[class*="css"]{font-family:'Manrope',sans-serif}
.stApp{background:radial-gradient(ellipse at 94% 2%,rgba(41,168,156,.11),transparent 34%),linear-gradient(135deg,#10191d 0%,#111d21 56%,#142226 100%);color:var(--ink)}
[data-testid="stSidebar"]{background:#0d171b;border-right:1px solid var(--line)}
h1,h2,h3{letter-spacing:0!important;color:#f1f5f2!important}
.brand{font-weight:800;font-size:19px;line-height:1.22;color:#f2f5ef}.brand span{color:var(--mint)}
.eyebrow{font-family:'DM Mono',monospace;color:var(--mint);font-size:11px;text-transform:uppercase;letter-spacing:0.08em;padding-bottom:9px}
.metric-panel{min-height:114px;padding:17px 18px;margin:3px 0 14px;background:linear-gradient(145deg,rgba(24,42,47,.96),rgba(17,31,36,.9));border:1px solid #2c4145;border-left:3px solid var(--mint);border-radius:5px}
.metric-label{color:#a6b6b8;font-size:12px}.metric-value{font-size:23px;font-weight:800;color:#f4f5ee;padding:6px 0 2px;overflow-wrap:anywhere}.metric-caption{font-family:'DM Mono',monospace;color:#789092;font-size:10px}
.entity-id{font:12px 'DM Mono',monospace;color:var(--mint);border:1px solid #315b58;padding:4px 8px;border-radius:3px;vertical-align:middle}
.stButton>button,.stDownloadButton>button{border-radius:4px;border:1px solid #42615f;background:#193432;color:#dbefea;font-weight:700}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--mint);color:#fff}
div[data-testid="stMetric"]{background:#152429;border:1px solid var(--line);padding:12px;border-radius:4px}
div[data-testid="stDataFrame"]{border:1px solid #293a3e;border-radius:4px}
section[data-testid="stSidebar"] .stRadio label{font-weight:600}
[data-testid="stCaptionContainer"]{color:#93a3a5}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_data():
    return load_data()


@st.cache_data(show_spinner="Building investigation features...")
def get_features(data):
    return engineer_features(data)


@st.cache_data(show_spinner=False)
def get_graph(data):
    return build_graph(data)


def analyze(data, features, model_bundle):
    anomaly_scores, anomalous, model_bundle = score_anomalies(features, model_bundle)
    scored = score_transactions(features, anomaly_scores)
    scored["anomalous"] = anomalous.reindex(scored.index).fillna(False).astype(bool)
    scored["investigation_signals"] = scored.apply(transaction_signals, axis=1)
    vendor_scores = aggregate_vendor_risks(scored)
    vendors = data["vendors"].merge(vendor_scores, on="vendor_id", how="left", suffixes=("", "_risk"))
    vendors["risk_score"] = vendors.risk_score.fillna(0)
    vendors["risk_level"] = vendors.risk_level.fillna("Low")
    return scored, vendor_scores, vendors, model_bundle


data = get_data()
features = get_features(data)
if "model_bundle" not in st.session_state:
    st.session_state.model_bundle = load_model()
if st.session_state.model_bundle is None and not features.empty:
    st.session_state.model_bundle = train_model(features, st.session_state.get("contamination", CONTAMINATION))
scored, vendor_scores, vendor_table, model_bundle = analyze(data, features, st.session_state.model_bundle)
graph = get_graph(data)

try:
    initialize_database()
    if "database_seeded" not in st.session_state:
        insert_data(data)
        for row in vendor_scores.itertuples(index=False):
            signals = []
            vendor_txns = scored.loc[scored.vendor_id.eq(row.vendor_id)]
            if not vendor_txns.empty:
                signals = sorted({signal for items in vendor_txns.investigation_signals for signal in items if "No elevated" not in signal})
            save_risk_score(row.vendor_id, row.risk_score, row.risk_level, signals)
        st.session_state.database_seeded = True
except Exception as error:
    st.sidebar.warning(f"SQLite is unavailable; CSV-backed analysis remains active. ({error})")

with st.sidebar:
    st.markdown("<div class='eyebrow'>Risk intelligence / 01</div><div class='brand'>AI FRAUD<br><span>INVESTIGATION</span><br>SYSTEM</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    page = st.radio("Workspace", ["Dashboard", "Vendor Investigation", "Transaction Investigation", "Fraud Graph", "Analytics", "Model / Data", "About"], label_visibility="collapsed")
    st.divider()
    global_query = st.text_input("Global entity search", placeholder="Vendor, employee, invoice, bank…")
    if global_query:
        matches = search_entities(global_query, data)
        if matches.empty: st.caption("No matching entities.")
        else: st.dataframe(matches.head(8), use_container_width=True, hide_index=True)
    st.markdown("<div class='eyebrow'>DEMO ENVIRONMENT · SYNTHETIC DATA</div>", unsafe_allow_html=True)
    st.caption("Find patterns. Follow relationships. Review the evidence.")

st.markdown("<div class='eyebrow'>DETECTION &nbsp; / &nbsp; RELATIONSHIPS &nbsp; / &nbsp; EXPLANATION &nbsp; / &nbsp; INVESTIGATION</div>", unsafe_allow_html=True)
st.title("AI Fraud Investigation System")
st.markdown("Detect suspicious entities. Reveal hidden relationships. Explain investigation signals.")
st.info("All findings are risk signals and investigation leads generated from synthetic data. They are not proof of fraud.", icon=":material/info:")

if page == "Dashboard":
    render_dashboard(data, scored, vendor_table)
elif page == "Vendor Investigation":
    render_vendor_investigation(data, scored, vendor_scores, graph)
elif page == "Transaction Investigation":
    render_transactions(scored)
elif page == "Fraud Graph":
    st.markdown("## Relationship investigation")
    if not vendor_table.empty:
        choices = vendor_table.sort_values("risk_score", ascending=False)
        left, depth_col, score_col = st.columns([1.2, 1, 1])
        selected_vendor = left.selectbox("Focus vendor", choices.vendor_id.tolist(), index=choices.vendor_id.tolist().index("VENDOR_017") if "VENDOR_017" in choices.vendor_id.tolist() else 0)
        depth = depth_col.slider("Relationship depth", 1, 4, 2)
        minimum_risk = score_col.slider("Minimum transaction risk", 0, 100, 0)
        entity_types = st.multiselect("Entity types", ["Vendor", "Employee", "Bank Account", "Address", "Transaction", "Invoice"], default=["Vendor", "Employee", "Bank Account", "Address", "Transaction", "Invoice"])
        review_vendor = vendor_table.loc[vendor_table.vendor_id.eq(selected_vendor)].iloc[0]
        st.markdown(f"**{selected_vendor}** · {review_vendor.vendor_name} · priority {review_vendor.risk_score:.1f}/100 · {review_vendor.risk_level}")
        vendor_txns = scored.loc[scored.vendor_id.eq(selected_vendor)]
        if minimum_risk and not vendor_txns.empty and vendor_txns.risk_score.max() < minimum_risk:
            st.warning("No transactions in this relationship focus meet the selected minimum priority.")
        else:
            render_graph(graph, selected_vendor, depth, set(entity_types), 620)
        st.caption("Select and drag nodes to explore. Pan and zoom controls are built into the graph.")
elif page == "Analytics":
    st.markdown("## Vendor and transaction analytics")
    tabs = st.tabs(["Risk patterns", "Relationships", "Duplicate invoices", "Employee interaction"])
    with tabs[0]:
        left, right = st.columns(2)
        with left: st.plotly_chart(px.box(scored, x="risk_level", y="amount", color="risk_level", color_discrete_sequence=PALETTE).update_layout(**LAYOUT), use_container_width=True)
        with right: st.plotly_chart(px.scatter(vendor_scores, x="transaction_count", y="risk_score", size="total_amount", color="risk_level", hover_name="vendor_id", color_discrete_sequence=PALETTE).update_layout(**LAYOUT), use_container_width=True)
        st.plotly_chart(px.bar(vendor_scores.nlargest(15, "total_amount"), x="vendor_id", y="average_amount", color="risk_level", color_discrete_sequence=PALETTE).update_layout(**LAYOUT), use_container_width=True)
    with tabs[1]:
        banks = data["vendors"].groupby("bank_account_id").agg(vendors=("vendor_id", "nunique")).query("vendors > 1").sort_values("vendors", ascending=False).reset_index()
        addresses = data["vendors"].groupby("address_id").agg(vendors=("vendor_id", "nunique")).query("vendors > 1").sort_values("vendors", ascending=False).reset_index()
        left, right = st.columns(2)
        with left: st.markdown("**Shared bank accounts**"); st.dataframe(banks, use_container_width=True, hide_index=True)
        with right: st.markdown("**Shared addresses**"); st.dataframe(addresses, use_container_width=True, hide_index=True)
        connected = vendor_scores.nlargest(15, "shared_bank_account_count")
        st.plotly_chart(px.bar(connected, x="vendor_id", y="shared_bank_account_count", color="risk_level", color_discrete_sequence=PALETTE).update_layout(**LAYOUT), use_container_width=True)
    with tabs[2]:
        duplicate_ids = data["invoices"].groupby("invoice_id").agg(record_count=("vendor_id", "count"), vendors=("vendor_id", "nunique"), amount=("invoice_amount", "sum")).query("record_count > 1").reset_index()
        st.metric("Invoice IDs appearing more than once", f"{len(duplicate_ids):,}")
        st.dataframe(duplicate_ids.sort_values("record_count", ascending=False), use_container_width=True, hide_index=True)
    with tabs[3]:
        interactions = scored.groupby(["employee_id", "vendor_id"]).agg(transactions=("transaction_id", "count"), amount=("amount", "sum"), highest_priority=("risk_score", "max")).reset_index().sort_values(["transactions", "amount"], ascending=False)
        st.dataframe(interactions.head(50), use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(interactions.head(20), x="employee_id", y="transactions", color="vendor_id", color_discrete_sequence=PALETTE).update_layout(**LAYOUT), use_container_width=True)
elif page == "Model / Data":
    st.markdown("## Model & data operations")
    model_status = "Loaded" if st.session_state.model_bundle is not None else "Not trained"
    columns = st.columns(4)
    columns[0].metric("Model", "Isolation Forest")
    columns[1].metric("Training samples", f"{len(features):,}")
    columns[2].metric("Numerical features", f"{len(model_bundle['columns']) if model_bundle else 0}")
    columns[3].metric("Model status", model_status)
    contamination = st.slider("Isolation Forest contamination", 0.005, 0.15, float(st.session_state.get("contamination", CONTAMINATION)), 0.005, help="Expected fraction of unusual observations used when fitting the model.")
    st.session_state.contamination = contamination
    st.caption(f"Anomalous transactions detected: {int(scored.anomalous.sum()) if len(scored) else 0:,} · Current fit contamination: {contamination:.1%}")
    actions = st.columns(3)
    if actions[0].button("Train model", use_container_width=True):
        st.session_state.model_bundle = train_model(features, contamination)
        st.rerun()
    if actions[1].button("Reload model", use_container_width=True):
        st.session_state.model_bundle = load_model()
        st.rerun()
    if actions[2].button("Generate demo data", use_container_width=True):
        write_demo_data(DATA_DIR)
        get_data.clear(); get_features.clear(); get_graph.clear()
        st.session_state.pop("database_seeded", None)
        st.rerun()
    if st.button("Reset demo data", type="secondary"):
        write_demo_data(DATA_DIR, seed=42)
        get_data.clear(); get_features.clear(); get_graph.clear()
        st.session_state.pop("database_seeded", None)
        st.rerun()
    st.markdown("#### Data quality review")
    warnings = validate_data(data)
    if warnings:
        for item in warnings: st.warning(item["message"])
    else: st.success("No blocking data-quality issues found.")
    st.markdown("#### Current feature set")
    st.code(", ".join(model_bundle["columns"] if model_bundle else []), language="text")
    st.markdown("#### Persistence")
    st.caption(f"SQLite store: {DB_PATH}")
    try:
        stored_signals = get_investigation_signals()
        st.caption(f"Persisted vendor review signals: {len(stored_signals):,}")
    except Exception as error:
        st.warning(f"SQLite persistence is unavailable: {error}")
    st.download_button("Download suspicious vendors", vendor_table.loc[vendor_table.risk_score >= 30].to_csv(index=False), "suspicious_vendors.csv", "text/csv")
    st.download_button("Download suspicious transactions", scored.loc[scored.risk_score >= 30].to_csv(index=False), "suspicious_transactions.csv", "text/csv")
    signals_frame = scored.loc[scored.risk_score >= 30, ["transaction_id", "vendor_id", "risk_score", "investigation_signals"]].copy()
    signals_frame["investigation_signals"] = signals_frame.investigation_signals.map(lambda items: " | ".join(items))
    st.download_button("Download investigation signals", signals_frame.to_csv(index=False), "investigation_signals.csv", "text/csv")
    st.download_button("Download vendor investigation report data", vendor_table.sort_values("risk_score", ascending=False).to_csv(index=False), "vendor_investigation_report.csv", "text/csv")
elif page == "About":
    st.markdown("## An investigation workspace, not a verdict engine")
    st.markdown("Traditional screening asks whether one payment looks unusual. This system connects machine-learning anomalies with identity overlaps, vendor behavior, and employee relationships so investigators can ask why a record was prioritized and what evidence to review next.")
    st.markdown("### Analysis pipeline")
    st.code("Synthetic CSV / SQLite → validation → behavioral features → Isolation Forest → relationship graph → weighted risk score → evidence-led review", language="text")
    st.markdown("### Methodology")
    st.markdown("- **Detection:** Isolation Forest ranks records that differ from the synthetic population; it is not trained on confirmed fraud labels.\n- **Relationships:** NetworkX models vendors, employees, bank accounts, addresses, invoices, and transactions. Shared identity is a lead for review, not proof of coordination.\n- **Scoring:** configurable weights combine model output and interpretable rule signals into a 0–100 review-priority score.\n- **Explainability:** each visible signal is generated from observed features in the loaded data.\n- **Privacy:** bundled information is synthetic and uses reserved `.test` email domains.")

st.markdown("<hr><span class='eyebrow'>AI-ASSISTED REVIEW · SYNTHETIC DATA · HUMAN INVESTIGATION</span>", unsafe_allow_html=True)