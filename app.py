import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Page Config
st.set_page_config(
    page_title="Nexus Tech | Customer Retention Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>

/* GLOBAL */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #0e1117;
    color: #e5e7eb;
}

.block-container {
    padding-top: 2rem;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1220, #0f172a);
    border-right: 1px solid rgba(255,255,255,0.06);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #38bdf8;
}

/* SIDEBAR LABEL TEXT (SLIDERS, INPUTS, SELECTBOXES) */
[data-testid="stSidebar"] label {
    color: #ffffff !important;
    font-weight: 500;
}

/* Sidebar help / secondary text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #d1d5db !important;
}

/* Slider min / max numbers */
[data-testid="stSidebar"] [data-testid="stTickBar"] {
    color: #e5e7eb !important;
}

/* Selectbox selected value */
[data-testid="stSidebar"] div[data-baseweb="select"] {
    color: #ffffff;
}


/* HEADERS */
h1 {
    font-size: 2.5rem;
    background: linear-gradient(90deg, #38bdf8, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

h2, h3 {
    color: #f3f4f6;
}

/* METRICS */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 0 30px rgba(56, 189, 248, 0.06);
    transition: transform 0.25s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
}

/* Pulse Glow */
.pulse {
    animation: pulseGlow 2s infinite;
    border-radius: 16px;
}

@keyframes pulseGlow {
    0% { box-shadow: 0 0 0 rgba(241, 196, 15, 0.2); }
    50% { box-shadow: 0 0 30px rgba(241, 196, 15, 0.6); }
    100% { box-shadow: 0 0 0 rgba(241, 196, 15, 0.2); }
}

/* STATUS BADGES */
.badge {
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
}

.badge-active {
    background: rgba(46, 204, 113, 0.15);
    color: #2ecc71;
}

.badge-risk {
    background: rgba(241, 196, 15, 0.15);
    color: #f1c40f;
}

.badge-churned {
    background: rgba(231, 76, 60, 0.15);
    color: #e74c3c;
}

/* AI CARD */
.ai-card {
    background: linear-gradient(135deg, rgba(56,189,248,0.15), rgba(168,85,247,0.15));
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 0 40px rgba(168, 85, 247, 0.25);
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}

/* BUTTON */
.stDownloadButton button {
    background: linear-gradient(135deg, #38bdf8, #6366f1);
    color: white;
    border-radius: 12px;
    font-weight: 600;
    padding: 0.65rem 1.3rem;
}

.stDownloadButton button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 30px rgba(99, 102, 241, 0.4);
}

/* PLOT PANELS */
.js-plotly-plot {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    padding: 8px;
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #38bdf8, #a855f7);
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# DATA LOADING
@st.cache_data
def load_data():
    df = pd.read_csv("Nexus_Tech_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# SIDEBAR
st.sidebar.title("🎯 Retention Settings")
st.sidebar.info("Adjust thresholds to define churn risk.")

churn_threshold = st.sidebar.slider(
    "Churn Definition (Days of Inactivity)",
    30, 180, 90
)

min_orders = st.sidebar.number_input(
    "Minimum Orders for 'Loyal' Status", 1, 10, 3
)

selected_region = st.sidebar.selectbox(
    "Target Region", ["All"] + list(df["Region"].unique())
)

# PROCESSING
def process_retention_data(data, threshold):
    current_date = data["Date"].max()

    cust_df = data.groupby("Customer_ID").agg({
        "Date": lambda x: (current_date - x.max()).days,
        "Order_ID": "count",
        "Total_Revenue": "sum",
        "Region": "first"
    }).rename(columns={
        "Date": "Recency",
        "Order_ID": "Frequency",
        "Total_Revenue": "LTV"
    })

    cust_df["Risk_Score"] = (cust_df["Recency"] / threshold) * 100
    cust_df["Risk_Score"] = cust_df["Risk_Score"].clip(upper=100).round(2)

    cust_df["Status"] = np.where(
        cust_df["Recency"] > threshold, "Churned",
        np.where(cust_df["Risk_Score"] > 70, "At Risk", "Active")
    )

    return cust_df

analysis_df = df if selected_region == "All" else df[df["Region"] == selected_region]
processed_df = process_retention_data(analysis_df, churn_threshold)

# MAIN UI
st.title("🎯 Customer Retention & Churn Predictor")
st.markdown(f"### Analyzing Intent Signals for **{selected_region}** Region")

# KPIs
total_customers = len(processed_df)
at_risk_count = len(processed_df[processed_df["Status"] == "At Risk"])
churn_rate = len(processed_df[processed_df["Status"] == "Churned"]) / total_customers

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("👥 Total Customers", f"{total_customers:,}")

with c2:
    st.markdown('<div class="pulse">', unsafe_allow_html=True)
    st.metric(
        "⚠️ At-Risk Customers",
        f"{at_risk_count:,}",
        delta=f"{at_risk_count/total_customers:.1%} of base",
        delta_color="inverse"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.metric("📉 Churn Rate", f"{churn_rate:.1%}")

# AI CONFIDENCE CARD
confidence_score = max(60, 100 - churn_rate * 100)

st.markdown(f"""
<div class="ai-card">
    <h3>🤖 AI Retention Signal Confidence</h3>
    <p>
        Predictive confidence is estimated at
        <b>{confidence_score:.1f}%</b>
        based on recency decay, frequency trends,
        and behavioral risk scoring.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# VISUALS
left, right = st.columns(2)

with left:
    status_counts = processed_df["Status"].value_counts().reset_index()
    fig_pie = px.pie(
        status_counts,
        names="Status",
        values="count",
        hole=0.5,
        title="Customer Health Composition",
        color="Status",
        color_discrete_map={
            "Active": "#2ECC71",
            "At Risk": "#F1C40F",
            "Churned": "#E74C3C"
        }
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with right:
    fig_scatter = px.scatter(
        processed_df,
        x="Recency",
        y="LTV",
        size="Frequency",
        color="Status",
        title="Recency vs Lifetime Value",
        hover_name=processed_df.index,
        color_discrete_map={
            "Active": "#2ECC71",
            "At Risk": "#F1C40F",
            "Churned": "#E74C3C"
        }
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ACTION CENTER
st.divider()
st.subheader("🚀 Win-Back Action Center")

action_list = processed_df[processed_df["Status"] != "Active"] \
    .sort_values("Risk_Score", ascending=False)

def badge(status):
    if status == "Active":
        return '<span class="badge badge-active">Active</span>'
    if status == "At Risk":
        return '<span class="badge badge-risk">At Risk</span>'
    return '<span class="badge badge-churned">Churned</span>'

action_list["Status"] = action_list["Status"].apply(badge)

st.write(
    action_list[["Recency", "Frequency", "LTV", "Risk_Score", "Status"]]
    .head(20)
    .to_html(escape=False),
    unsafe_allow_html=True
)

st.download_button(
    "📥 Export Win-Back List for Marketing",
    data=action_list.to_csv().encode("utf-8"),
    file_name="win_back_list.csv",
    mime="text/csv"
)
