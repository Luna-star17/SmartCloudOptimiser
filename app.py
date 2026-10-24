"""
UTHO SmartCloud Optimizer - Frontend Dashboard
A modern, safety-aware cloud cost optimization dashboard built with Streamlit, Pandas, and Plotly.
Provides resource inventory, historical telemetry analytics (avg/p95/p99), evidence-based
rightsizing, workload scheduling, orphan detection, and what-if financial simulations.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page
st.set_page_config(
    page_title="SmartCloud Optimizer | Cloud FinOps Dashboard",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# BACKEND INTEGRATION & DATA PATHS
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Attempt backend engine imports with fallback protections
try:
    from optimizer import calculate_vm_stats, generate_recommendations, DEFAULT_THRESHOLDS
    from simulator import ActionSimulator
    from alerts import detect_alerts
    BACKEND_AVAILABLE = True
except ImportError as err:
    BACKEND_AVAILABLE = False
    BACKEND_ERR = str(err)


# -----------------------------------------------------------------------------
# CUSTOM CSS & STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global font & container improvements */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top simulation notice banner */
    .sim-banner {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: #ffffff;
        padding: 10px 18px;
        border-radius: 8px;
        font-size: 0.88rem;
        font-weight: 500;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
    }
    
    /* KPI Metric Cards */
    .kpi-container {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 12px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    .kpi-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .kpi-subtext {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    
    /* Badges & Pills */
    .badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        line-height: 1.2;
    }
    .badge-low {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .badge-med {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .badge-high {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    .badge-guardrail {
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }
    .badge-status-accepted {
        background-color: #d1fae5;
        color: #065f46;
        border: 1px solid #34d399;
    }
    .badge-status-rejected {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #f87171;
    }
    .badge-status-pending {
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #cbd5e1;
    }

    /* Recommendation Card Styling */
    .rec-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .rec-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .rec-action {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 6px;
    }
    .rec-evidence {
        background-color: #f8fafc;
        border-left: 3px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
        color: #334155;
        margin: 10px 0;
        line-height: 1.45;
    }
    .rec-savings {
        font-size: 1.2rem;
        font-weight: 700;
        color: #059669;
    }
    
    /* Help Tooltip Box */
    .tooltip-box {
        background-color: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 0.8rem;
        color: #475569;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# HELPER FORMATTING FUNCTIONS
# -----------------------------------------------------------------------------
def format_inr(val) -> str:
    """Formats numeric values into Indian Rupee currency format (₹)."""
    if pd.isna(val) or val is None:
        return "₹0"
    try:
        val = float(val)
        return f"₹{val:,.0f}"
    except (ValueError, TypeError):
        return f"₹{val}"


def get_risk_badge_html(risk: str) -> str:
    """Generates styled HTML badge for risk levels."""
    r_lower = str(risk).lower()
    if "low" in r_lower:
        return f'<span class="badge badge-low">🛡️ Low Risk</span>'
    elif "med" in r_lower:
        return f'<span class="badge badge-med">⚠️ Medium Risk</span>'
    elif "high" in r_lower:
        return f'<span class="badge badge-high">🚨 High Risk</span>'
    return f'<span class="badge badge-guardrail">ℹ️ {risk}</span>'


def get_status_badge_html(status: str) -> str:
    """Generates styled HTML badge for recommendation decision status."""
    s_lower = str(status).lower()
    if s_lower == "accepted":
        return f'<span class="badge badge-status-accepted">✓ Accepted</span>'
    elif s_lower == "rejected":
        return f'<span class="badge badge-status-rejected">✕ Rejected</span>'
    return f'<span class="badge badge-status-pending">⏳ Pending Review</span>'


# -----------------------------------------------------------------------------
# DATA LOADING & CACHING PIPELINE
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading cloud telemetry and inventory datasets...")
def load_all_data():
    """
    Loads all data sources and executes backend statistical computations.
    Provides robust fallbacks if files are missing.
    """
    vms_path = os.path.join(DATA_DIR, "vms.csv")
    resources_path = os.path.join(DATA_DIR, "resources.csv")
    inventory_path = os.path.join(DATA_DIR, "inventory.csv")
    metrics_path = os.path.join(DATA_DIR, "metrics.csv")
    pricing_path = os.path.join(DATA_DIR, "pricing.csv")

    errors = []
    
    # 1. Load VMs
    if os.path.exists(vms_path):
        df_vms = pd.read_csv(vms_path)
    else:
        errors.append(f"Missing file: {vms_path}")
        df_vms = pd.DataFrame()

    # 2. Load Resources (Storage & Static IPs)
    if os.path.exists(resources_path):
        df_resources = pd.read_csv(resources_path)
    else:
        df_resources = pd.DataFrame()

    # 3. Load Unified Inventory
    if os.path.exists(inventory_path):
        df_inventory = pd.read_csv(inventory_path)
    elif not df_vms.empty:
        # Generate on-the-fly if inventory.csv not yet written
        inv_rows = []
        for _, vm in df_vms.iterrows():
            inv_rows.append({
                "resource_id": vm["vm_id"],
                "name": vm["vm_name"],
                "type": f"VM ({vm.get('instance_type', 'Virtual Machine')})",
                "region": vm["region"],
                "environment": vm["environment"],
                "owner": vm["team"],
                "vcpu": vm.get("vcpu", 2),
                "ram_gb": vm.get("memory_gb", 4),
                "disk_gb": 100,
                "creation_date": "2026-08-15",
                "monthly_cost": vm["monthly_cost"]
            })
        for _, res in df_resources.iterrows():
            is_ip = "IP" in res.get("resource_type", "")
            inv_rows.append({
                "resource_id": res["resource_id"],
                "name": res.get("notes", res["resource_id"]),
                "type": res["resource_type"],
                "region": res["region"],
                "environment": "Production" if res.get("attached") else "Staging",
                "owner": res["team"],
                "vcpu": np.nan,
                "ram_gb": np.nan,
                "disk_gb": np.nan if is_ip else res.get("size_gb", 0),
                "creation_date": "2026-07-01",
                "monthly_cost": res["monthly_cost"]
            })
        df_inventory = pd.DataFrame(inv_rows)
    else:
        df_inventory = pd.DataFrame()

    # 4. Load Pricing Catalog
    if os.path.exists(pricing_path):
        df_pricing = pd.read_csv(pricing_path)
    else:
        df_pricing = pd.DataFrame()

    # 5. Load Historical Metrics (optimize dtypes and parse timestamps)
    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)
        if "timestamp" in df_metrics.columns:
            df_metrics["timestamp"] = pd.to_datetime(df_metrics["timestamp"])
    else:
        errors.append(f"Missing file: {metrics_path}")
        df_metrics = pd.DataFrame()

    # Compute Statistics, Recommendations, and Alerts using backend engines
    df_stats = pd.DataFrame()
    recommendations = []
    alerts = []

    if BACKEND_AVAILABLE and not df_metrics.empty and not df_vms.empty:
        try:
            df_stats = calculate_vm_stats(df_metrics, df_vms)
            recommendations = generate_recommendations(
                df_vms=df_vms,
                df_stats=df_stats,
                df_resources=df_resources,
                df_pricing=df_pricing
            )
            alerts = detect_alerts(
                df_vms=df_vms,
                df_stats=df_stats,
                df_resources=df_resources
            )
        except Exception as e:
            errors.append(f"Backend computation error: {str(e)}")

    return {
        "df_vms": df_vms,
        "df_resources": df_resources,
        "df_inventory": df_inventory,
        "df_metrics": df_metrics,
        "df_pricing": df_pricing,
        "df_stats": df_stats,
        "recommendations": recommendations,
        "alerts": alerts,
        "errors": errors
    }


# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "rec_decisions" not in st.session_state:
    st.session_state["rec_decisions"] = {}

if "selected_resource_for_perf" not in st.session_state:
    st.session_state["selected_resource_for_perf"] = None

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Overview"


def accept_recommendation(rec_id: str):
    """Marks recommendation as accepted in session state."""
    st.session_state["rec_decisions"][rec_id] = {
        "status": "Accepted",
        "reason": "Approved by Administrator / FinOps Reviewer",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def reject_recommendation(rec_id: str, reason: str = "Unspecified"):
    """Marks recommendation as rejected in session state."""
    st.session_state["rec_decisions"][rec_id] = {
        "status": "Rejected",
        "reason": reason,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def reset_recommendation(rec_id: str):
    """Reverts recommendation back to pending review."""
    if rec_id in st.session_state["rec_decisions"]:
        del st.session_state["rec_decisions"][rec_id]


# -----------------------------------------------------------------------------
# MAIN APP EXECUTION
# -----------------------------------------------------------------------------
data_bundle = load_all_data()

# Check for critical dataset errors
if data_bundle["errors"]:
    for err in data_bundle["errors"]:
        st.error(f"⚠️ {err}")
    st.info("Tip: You can generate synthetic datasets by executing `python data_generator.py` in your terminal.")

df_inventory = data_bundle["df_inventory"]
df_vms = data_bundle["df_vms"]
df_resources = data_bundle["df_resources"]
df_metrics = data_bundle["df_metrics"]
df_stats = data_bundle["df_stats"]
recommendations = data_bundle["recommendations"]
alerts = data_bundle["alerts"]


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & GLOBAL FILTERS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ☁️ SmartCloud Optimizer")
    st.caption("College Hackathon Project | Cloud FinOps")
    
    st.markdown("""
    <div style="background-color: #f1f5f9; padding: 6px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 600; color: #475569; margin-bottom: 14px;">
        🧪 DEMO MODE — SIMULATED CLOUD DATA
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation Radio
    nav_options = [
        "Overview",
        "Resource Inventory",
        "Performance Analysis",
        "Optimization Recommendations",
        "Decision History & What-If"
    ]
    
    # Sync with session state if set externally
    current_nav_index = nav_options.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in nav_options else 0
    selected_page = st.radio(
        "Navigation",
        nav_options,
        index=current_nav_index,
        key="nav_selection"
    )
    st.session_state["nav_page"] = selected_page
    
    st.divider()
    st.markdown("#### 🔍 Dynamic Scope Filters")
    
    # Reset filters button
    if st.button("↺ Reset All Filters", use_container_width=True):
        st.session_state["filter_region"] = []
        st.session_state["filter_env"] = []
        st.session_state["filter_team"] = []
        st.session_state["filter_type"] = []
        st.rerun()

    # Dynamic filter options extracted from active inventory
    all_regions = sorted(df_inventory["region"].dropna().unique().tolist()) if not df_inventory.empty else []
    all_envs = sorted(df_inventory["environment"].dropna().unique().tolist()) if not df_inventory.empty else []
    all_teams = sorted(df_inventory["owner"].dropna().unique().tolist()) if not df_inventory.empty else []
    all_types = sorted(df_inventory["type"].dropna().unique().tolist()) if not df_inventory.empty else []

    selected_regions = st.multiselect("Region", all_regions, key="filter_region")
    selected_envs = st.multiselect("Environment", all_envs, key="filter_env")
    selected_teams = st.multiselect("Team / Owner", all_teams, key="filter_team")
    selected_types = st.multiselect("Resource Type", all_types, key="filter_type")
    
    # Date Range Filter for Telemetry
    min_date = df_metrics["timestamp"].min().date() if not df_metrics.empty and pd.api.types.is_datetime64_any_dtype(df_metrics["timestamp"]) else datetime.now().date() - timedelta(days=30)
    max_date = df_metrics["timestamp"].max().date() if not df_metrics.empty and pd.api.types.is_datetime64_any_dtype(df_metrics["timestamp"]) else datetime.now().date()
    
    date_range = st.date_input(
        "Historical Range (Telemetry)",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="filter_date_range"
    )

    st.divider()
    # Sidebar quick FinOps summary
    accepted_ids = [r_id for r_id, d in st.session_state["rec_decisions"].items() if d["status"] == "Accepted"]
    accepted_savings = sum(r["monthly_savings"] for r in recommendations if r["rec_id"] in accepted_ids)
    
    st.markdown("##### 💼 What-If Session Progress")
    st.write(f"• Accepted Actions: **{len(accepted_ids)}** / {len(recommendations)}")
    st.write(f"• Potential Savings: **{format_inr(accepted_savings)}** / mo")
    
    st.caption("UTHO Cloud Optimizer Engine v1.0 • Built with Streamlit & Plotly")


# -----------------------------------------------------------------------------
# APPLY SCOPE FILTERS
# -----------------------------------------------------------------------------
df_inv_filtered = df_inventory.copy() if not df_inventory.empty else pd.DataFrame()

if not df_inv_filtered.empty:
    if selected_regions:
        df_inv_filtered = df_inv_filtered[df_inv_filtered["region"].isin(selected_regions)]
    if selected_envs:
        df_inv_filtered = df_inv_filtered[df_inv_filtered["environment"].isin(selected_envs)]
    if selected_teams:
        df_inv_filtered = df_inv_filtered[df_inv_filtered["owner"].isin(selected_teams)]
    if selected_types:
        df_inv_filtered = df_inv_filtered[df_inv_filtered["type"].isin(selected_types)]

valid_resource_ids = set(df_inv_filtered["resource_id"].unique()) if not df_inv_filtered.empty else set()

# Filter recommendations matching current inventory scope
recs_filtered = [r for r in recommendations if r["resource_id"] in valid_resource_ids]
alerts_filtered = [a for a in alerts if a["resource_id"] in valid_resource_ids]


# -----------------------------------------------------------------------------
# TOP BANNER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="sim-banner">
    <div>
        <strong>Demo Mode — Simulated Cloud Telemetry:</strong>
        All displayed optimization actions, cost estimates, and simulations represent safe recommendations.
        Decisions made in this dashboard are simulated in-memory and <u>never alter live infrastructure</u>.
    </div>
    <span style="background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 4px; font-size: 0.75rem;">Safety Guardrails Active</span>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# PAGE 1: MAIN DASHBOARD OVERVIEW
# =============================================================================
if selected_page == "Overview":
    st.markdown("## 📊 Infrastructure & FinOps Overview")
    st.write("Real-time summary of monitored cloud estate, detected cost optimization candidates, and peak-safe capacity safeguards.")

    if df_inv_filtered.empty:
        st.warning("⚠️ No cloud resources match your currently selected filters. Please adjust or reset the sidebar filters.")
    else:
        # Calculate Core Metrics
        total_resources = len(df_inv_filtered)
        total_vms = len(df_inv_filtered[df_inv_filtered["type"].str.contains("VM", na=False)])
        flagged_resources = len(set(r["resource_id"] for r in recs_filtered))
        
        # Calculate raw potential monthly savings
        total_potential_savings = sum(r["monthly_savings"] for r in recs_filtered)
        
        # Count high-risk recommendations
        high_risk_count = sum(1 for r in recs_filtered if "high" in str(r.get("risk_level", "")).lower())
        total_monthly_cost = float(df_inv_filtered["monthly_cost"].sum())
        savings_pct = (total_potential_savings / total_monthly_cost * 100.0) if total_monthly_cost > 0 else 0.0

        # Display 6 Metric Cards
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Total Resources</div>
                <div class="kpi-value">{total_resources}</div>
                <div class="kpi-subtext">Active cloud assets</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Virtual Machines</div>
                <div class="kpi-value">{total_vms}</div>
                <div class="kpi-subtext">Compute instances</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Flagged for Opt.</div>
                <div class="kpi-value" style="color: #2563eb;">{flagged_resources}</div>
                <div class="kpi-subtext">Candidates evaluated</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c4:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Potential Savings</div>
                <div class="kpi-value" style="color: #059669;">{format_inr(total_potential_savings)}</div>
                <div class="kpi-subtext">~{savings_pct:.1f}% monthly reduction</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c5:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">High-Risk / Guardrails</div>
                <div class="kpi-value" style="color: {'#dc2626' if high_risk_count > 0 else '#475569'};">{high_risk_count}</div>
                <div class="kpi-subtext">Protected by safety engine</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c6:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Total Cloud Spend</div>
                <div class="kpi-value">{format_inr(total_monthly_cost)}</div>
                <div class="kpi-subtext">Baseline monthly cost</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Overview Charts (2x2 Grid)
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.markdown("#### 💰 Baseline vs. Optimized Monthly Spend")
            # Waterfall or Grouped comparison
            sim_projected_spend = max(total_monthly_cost - total_potential_savings, 0)
            df_spend_comp = pd.DataFrame({
                "Scenario": ["Current Spend", "Optimized Target", "Potential Savings"],
                "Amount (INR)": [total_monthly_cost, sim_projected_spend, total_potential_savings],
                "Type": ["Current", "Target", "Savings"]
            })
            fig_spend = px.bar(
                df_spend_comp,
                x="Scenario",
                y="Amount (INR)",
                color="Type",
                color_discrete_map={"Current": "#64748b", "Target": "#3b82f6", "Savings": "#10b981"},
                text_auto=".2s",
                title="Monthly Spend Impact Projection"
            )
            fig_spend.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
            fig_spend.update_yaxes(title="Monthly Cost (INR)", tickprefix="₹")
            st.plotly_chart(fig_spend, use_container_width=True)

        with row1_col2:
            st.markdown("#### 🌍 Potential Savings by Environment")
            if recs_filtered:
                df_recs = pd.DataFrame(recs_filtered)
                df_env_savings = df_recs.groupby("environment")["monthly_savings"].sum().reset_index()
                fig_env = px.pie(
                    df_env_savings,
                    names="environment",
                    values="monthly_savings",
                    hole=0.45,
                    color_discrete_sequence=["#10b981", "#3b82f6", "#f59e0b", "#8b5cf6"]
                )
                fig_env.update_traces(textposition='inside', textinfo='percent+label')
                fig_env.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_env, use_container_width=True)
            else:
                st.info("No recommendations available for current filter selection.")

        row2_col1, row2_col2 = st.columns(2)
        
        with row2_col1:
            st.markdown("#### 👥 Monthly Savings Opportunity by Team")
            if recs_filtered:
                df_recs = pd.DataFrame(recs_filtered)
                df_team_savings = df_recs.groupby("team")["monthly_savings"].sum().reset_index().sort_values(by="monthly_savings", ascending=True)
                fig_team = px.bar(
                    df_team_savings,
                    x="monthly_savings",
                    y="team",
                    orientation="h",
                    color="monthly_savings",
                    color_continuous_scale="Blues",
                    text_auto=".2s"
                )
                fig_team.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
                fig_team.update_xaxes(title="Estimated Monthly Savings (INR)", tickprefix="₹")
                fig_team.update_yaxes(title="")
                st.plotly_chart(fig_team, use_container_width=True)
            else:
                st.info("No recommendations available for current filter selection.")

        with row2_col2:
            st.markdown("#### 🏷️ Optimization Strategy Breakdown")
            if recs_filtered:
                df_recs = pd.DataFrame(recs_filtered)
                df_cat_savings = df_recs.groupby("category")["monthly_savings"].sum().reset_index()
                fig_cat = px.pie(
                    df_cat_savings,
                    names="category",
                    values="monthly_savings",
                    hole=0.45,
                    color="category",
                    color_discrete_map={
                        "Safe Rightsizing": "#3b82f6",
                        "Workload Scheduling": "#10b981",
                        "Orphaned Cleanup": "#f59e0b"
                    }
                )
                fig_cat.update_traces(textposition='inside', textinfo='percent+label')
                fig_cat.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("No recommendations available for current filter selection.")

        st.divider()

        # Active Safety Guardrails Feed
        st.markdown("### 🛡️ Active Safety Guardrails & Anomaly Feed")
        st.write("Our engine constantly audits workloads to protect against blind, naive downsizing of bursty or critical infrastructure.")
        
        if alerts_filtered:
            for alert in alerts_filtered[:5]:
                severity = alert.get("severity", "Info")
                bg_color = "#fffbeb" if severity == "Warning" else "#fef2f2" if severity == "Critical" else "#f8fafc"
                border_color = "#fde68a" if severity == "Warning" else "#fecaca" if severity == "Critical" else "#e2e8f0"
                icon = "⚠️" if severity == "Warning" else "🚨" if severity == "Critical" else "ℹ️"
                
                st.markdown(f"""
                <div style="background-color: {bg_color}; border: 1px solid {border_color}; border-left: 4px solid {'#f59e0b' if severity == 'Warning' else '#ef4444' if severity == 'Critical' else '#3b82f6'}; border-radius: 6px; padding: 12px 16px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: #1e293b;">{icon} {alert['type']}</span>
                        <span class="badge badge-guardrail">{alert['resource_id']} ({alert.get('team', '')})</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #334155; margin-top: 5px;">
                        {alert['message']}
                    </div>
                    <div style="font-size: 0.8rem; font-weight: 600; color: #64748b; margin-top: 4px;">
                        Recommended Action: {alert['recommended_action']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("All resources within current scope conform to standard utilization guardrails.")


# =============================================================================
# PAGE 2: CLOUD RESOURCE INVENTORY
# =============================================================================
elif selected_page == "Resource Inventory":
    st.markdown("## 📋 Cloud Resource Inventory")
    st.write("Comprehensive asset inventory across compute, block storage, and elastic static IPs.")

    if df_inv_filtered.empty:
        st.warning("⚠️ No cloud resources match the selected filters.")
    else:
        # Search Box
        inv_search = st.text_input("🔍 Quick Search by Resource ID, Name, or Owner", "", placeholder="e.g. VM-014, checkout, marketing, vol-0091")
        
        df_display = df_inv_filtered.copy()
        if inv_search:
            search_str = inv_search.lower()
            df_display = df_display[
                df_display["resource_id"].str.lower().str.contains(search_str, na=False) |
                df_display["name"].str.lower().str.contains(search_str, na=False) |
                df_display["owner"].str.lower().str.contains(search_str, na=False) |
                df_display["type"].str.lower().str.contains(search_str, na=False)
            ]

        st.caption(f"Showing **{len(df_display)}** of **{len(df_inventory)}** total cloud resources")

        # Format dataframe for clean table view
        df_table = df_display.copy()
        
        # Format monthly cost column
        df_table["Monthly Cost (INR)"] = df_table["monthly_cost"].apply(format_inr)
        
        # Clean formatting for vCPU and RAM: Display missing as "N/A"
        df_table["vCPU"] = df_table["vcpu"].apply(lambda x: "N/A" if pd.isna(x) else f"{int(x)} vCPU")
        df_table["RAM"] = df_table["ram_gb"].apply(lambda x: "N/A" if pd.isna(x) else f"{int(x)} GB")
        df_table["Disk Size"] = df_table["disk_gb"].apply(lambda x: "N/A" if pd.isna(x) else f"{int(x)} GB")

        # Rename columns to standard readable names
        columns_to_show = [
            "resource_id", "name", "type", "region", "environment",
            "owner", "vCPU", "RAM", "Disk Size", "creation_date", "Monthly Cost (INR)"
        ]
        
        # Column formatting in Streamlit dataframe
        st.dataframe(
            df_table[columns_to_show].rename(columns={
                "resource_id": "Resource ID",
                "name": "Resource Name",
                "type": "Resource Type",
                "region": "Region",
                "environment": "Environment",
                "owner": "Team / Owner",
                "creation_date": "Creation Date"
            }),
            use_container_width=True,
            hide_index=True,
            height=420
        )

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Interactive Resource Deep-Dive Selector
        st.markdown("#### 🔍 Select Resource to Inspect Telemetry")
        res_list = df_display["resource_id"].tolist()
        
        col_sel, col_btn = st.columns([3, 1])
        with col_sel:
            selected_res = st.selectbox(
                "Choose a resource to load into Performance Analysis:",
                res_list,
                key="inventory_drilldown_selector"
            )
        with col_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("📈 Analyze Performance ➔", use_container_width=True):
                st.session_state["selected_resource_for_perf"] = selected_res
                st.session_state["nav_page"] = "Performance Analysis"
                st.rerun()

        # CSV Download Button
        csv_data = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Export Filtered Inventory (CSV)",
            data=csv_data,
            file_name=f"smartcloud_inventory_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


# =============================================================================
# PAGE 3: HISTORICAL PERFORMANCE METRICS & STATISTICS
# =============================================================================
elif selected_page == "Performance Analysis":
    st.markdown("## 📈 Historical Performance Telemetry & Capacity Safeguards")
    st.write("Detailed 30-day time-series telemetry analysis featuring average, p95, and p99 peak-demand safeguards.")

    if df_metrics.empty:
        st.error("⚠️ Historical metrics dataset is not available. Please ensure `data/metrics.csv` is populated.")
    else:
        # Determine active selected resource
        available_vms = sorted(df_metrics["vm_id"].dropna().unique().tolist())
        
        # Check if pre-selected from inventory or recommendations
        default_index = 0
        if st.session_state.get("selected_resource_for_perf") in available_vms:
            default_index = available_vms.index(st.session_state["selected_resource_for_perf"])

        c_pick1, c_pick2 = st.columns([2, 3])
        with c_pick1:
            active_vm_id = st.selectbox(
                "Select Compute Instance (VM):",
                available_vms,
                index=default_index,
                key="active_vm_picker"
            )
            st.session_state["selected_resource_for_perf"] = active_vm_id

        # Lookup VM specification
        vm_meta = df_vms[df_vms["vm_id"] == active_vm_id].iloc[0] if not df_vms.empty and active_vm_id in df_vms["vm_id"].values else None
        
        with c_pick2:
            if vm_meta is not None:
                pattern = vm_meta.get("workload_pattern", "Normal")
                st.markdown(f"""
                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; margin-top: 4px;">
                    <div style="font-size: 0.82rem; color: #64748b; font-weight: 600;">ACTIVE INSTANCE METADATA</div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: #0f172a;">
                        {vm_meta['vm_name']} <span class="badge badge-guardrail">{vm_meta['instance_type']}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #475569; margin-top: 2px;">
                        • Team: <b>{vm_meta['team']}</b> | Env: <b>{vm_meta['environment']}</b> | Cost: <b>{format_inr(vm_meta['monthly_cost'])}/mo</b> | Pattern: <b>{pattern}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Filter telemetry for selected VM and date range
        v_metrics = df_metrics[df_metrics["vm_id"] == active_vm_id].sort_values(by="timestamp").copy()
        
        if len(date_range) == 2:
            start_d, end_d = date_range
            v_metrics = v_metrics[
                (v_metrics["timestamp"].dt.date >= start_d) &
                (v_metrics["timestamp"].dt.date <= end_d)
            ]

        if v_metrics.empty:
            st.warning(f"No metric readings recorded for {active_vm_id} within the selected date range.")
        else:
            # Calculate Statistical Metrics (Average, p95, p99, Max)
            cpu_series = v_metrics["cpu_pct"]
            mem_series = v_metrics["memory_pct"]
            disk_series = v_metrics["disk_used_pct"]
            iops_series = v_metrics["disk_iops"]
            net_series = v_metrics["network_mbps"]

            avg_cpu = float(cpu_series.mean())
            p95_cpu = float(np.percentile(cpu_series, 95))
            p99_cpu = float(np.percentile(cpu_series, 99))
            max_cpu = float(cpu_series.max())

            avg_mem = float(mem_series.mean())
            p95_mem = float(np.percentile(mem_series, 95))
            p99_mem = float(np.percentile(mem_series, 99))
            max_mem = float(mem_series.max())

            avg_iops = float(iops_series.mean())
            max_iops = float(iops_series.max())
            avg_net = float(net_series.mean())
            max_net = float(net_series.max())

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📊 Statistical Capacity Summary")
            
            # Beginner-friendly tooltips container
            st.markdown("""
            <div class="tooltip-box">
                💡 <b>Statistical Concepts Guide:</b><br>
                • <b>Average:</b> Typical everyday utilization across the observation period.<br>
                • <b>p95 (95th Percentile):</b> 95% of all recorded readings fall at or below this value. Essential for capacity rightsizing.<br>
                • <b>p99 (99th Percentile):</b> 99% of readings fall below this value. Identifies bursty spikes and protects mission-critical performance.<br>
                • <b>Max:</b> Peak extreme reading observed during heavy traffic bursts.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)

            # 4 Metric Cards for CPU & Memory percentiles
            stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
            with stat_c1:
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-title">CPU: Average vs p95</div>
                    <div class="kpi-value">{avg_cpu:.1f}% <span style="font-size: 1rem; color: #2563eb;">(p95: {p95_cpu:.1f}%)</span></div>
                    <div class="kpi-subtext">Peak p99: <b>{p99_cpu:.1f}%</b> | Max: <b>{max_cpu:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)

            with stat_c2:
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-title">RAM: Average vs p95</div>
                    <div class="kpi-value">{avg_mem:.1f}% <span style="font-size: 1rem; color: #2563eb;">(p95: {p95_mem:.1f}%)</span></div>
                    <div class="kpi-subtext">Peak p99: <b>{p99_mem:.1f}%</b> | Max: <b>{max_mem:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)

            with stat_c3:
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-title">Disk Storage & IOPS</div>
                    <div class="kpi-value">{avg_iops:.0f} <span style="font-size: 0.9rem; color: #64748b;">IOPS</span></div>
                    <div class="kpi-subtext">Disk Used: <b>{disk_series.iloc[-1]:.1f}%</b> | Max IOPS: <b>{max_iops:.0f}</b></div>
                </div>
                """, unsafe_allow_html=True)

            with stat_c4:
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-title">Network Throughput</div>
                    <div class="kpi-value">{avg_net:.1f} <span style="font-size: 0.9rem; color: #64748b;">Mbps</span></div>
                    <div class="kpi-subtext">Max Peak: <b>{max_net:.1f} Mbps</b></div>
                </div>
                """, unsafe_allow_html=True)

            # Bursty Safeguard Callout Banner
            if avg_cpu < 25.0 and p99_cpu >= 80.0:
                st.markdown(f"""
                <div style="background-color: #fffbeb; border: 1px solid #fde68a; border-left: 4px solid #f59e0b; border-radius: 6px; padding: 12px; margin: 12px 0;">
                    <strong>🛡️ Bursty Workload Safety Guardrail Active:</strong>
                    This machine exhibits a deceptively low average CPU ({avg_cpu:.1f}%), but sharp p99 bursts reaching {p99_cpu:.1f}%.
                    Naive algorithms would blindly downsize this instance, risking production latency spikes.
                    Our engine requires candidate instances to satisfy strict p95 and p99 headroom safety targets before suggesting downsizing.
                </div>
                """, unsafe_allow_html=True)

            # Interactive Plotly Charts
            st.markdown("#### 📉 CPU and Memory Utilization Timeline")
            
            fig_cpu_mem = go.Figure()
            fig_cpu_mem.add_trace(go.Scatter(
                x=v_metrics["timestamp"],
                y=v_metrics["cpu_pct"],
                mode="lines",
                name="CPU Utilization (%)",
                line=dict(color="#2563eb", width=1.5),
                hovertemplate="<b>CPU:</b> %{y:.1f}%<br>Time: %{x}<extra></extra>"
            ))
            fig_cpu_mem.add_trace(go.Scatter(
                x=v_metrics["timestamp"],
                y=v_metrics["memory_pct"],
                mode="lines",
                name="Memory Utilization (%)",
                line=dict(color="#10b981", width=1.5),
                hovertemplate="<b>RAM:</b> %{y:.1f}%<br>Time: %{x}<extra></extra>"
            ))
            
            # Target p95 & p99 safety thresholds
            fig_cpu_mem.add_hline(
                y=70, line_dash="dash", line_color="#f59e0b",
                annotation_text="Target p95 Safety Ceiling (70%)",
                annotation_position="top left"
            )
            fig_cpu_mem.add_hline(
                y=85, line_dash="dot", line_color="#ef4444",
                annotation_text="Critical p99 Headroom Limit (85%)",
                annotation_position="top left"
            )

            fig_cpu_mem.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(title="Utilization (%)", range=[0, 105]),
                xaxis=dict(title="Timestamp")
            )
            st.plotly_chart(fig_cpu_mem, use_container_width=True)

            # Chart 2: Disk IOPS & Network Activity
            ch_col1, ch_col2 = st.columns(2)
            
            with ch_col1:
                st.markdown("#### 💽 Disk IOPS Activity")
                fig_iops = px.line(
                    v_metrics,
                    x="timestamp",
                    y="disk_iops",
                    labels={"disk_iops": "Disk IOPS", "timestamp": "Time"},
                    title="Storage Operations per Second (IOPS)"
                )
                fig_iops.update_traces(line_color="#8b5cf6")
                fig_iops.update_layout(height=280, margin=dict(l=20, r=20, t=35, b=20))
                st.plotly_chart(fig_iops, use_container_width=True)

            with ch_col2:
                st.markdown("#### 🌐 Network Throughput (Mbps)")
                fig_net = px.line(
                    v_metrics,
                    x="timestamp",
                    y="network_mbps",
                    labels={"network_mbps": "Bandwidth (Mbps)", "timestamp": "Time"},
                    title="Network In/Out Traffic Volume"
                )
                fig_net.update_traces(line_color="#06b6d4")
                fig_net.update_layout(height=280, margin=dict(l=20, r=20, t=35, b=20))
                st.plotly_chart(fig_net, use_container_width=True)

            # Hourly Pattern Profile (Day of Week & Business Hours)
            st.markdown("#### 🕒 Workhour vs. Off-Hours Utilization Profile")
            v_metrics["hour"] = v_metrics["timestamp"].dt.hour
            v_metrics["day_name"] = v_metrics["timestamp"].dt.day_name()
            hourly_avg = v_metrics.groupby(["day_name", "hour"])["cpu_pct"].mean().reset_index()
            
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            fig_heatmap = px.density_heatmap(
                hourly_avg,
                x="hour",
                y="day_name",
                z="cpu_pct",
                category_orders={"day_name": days_order},
                color_continuous_scale="Viridis",
                labels={"hour": "Hour of Day (0-23)", "day_name": "Day of Week", "cpu_pct": "Avg CPU %"}
            )
            fig_heatmap.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_heatmap, use_container_width=True)


# =============================================================================
# PAGE 4: OPTIMIZATION RECOMMENDATIONS
# =============================================================================
elif selected_page == "Optimization Recommendations":
    st.markdown("## 💡 Evidence-Based Optimization Recommendations")
    st.write("Transparent recommendations generated with strict p95/p99 capacity safeguards and non-destructive simulation.")

    if not recs_filtered:
        st.info("No optimization recommendations match the current filters or dataset.")
    else:
        # Category and Status Filter Pills
        r_col1, r_col2, r_col3 = st.columns([2, 2, 2])
        with r_col1:
            cat_filter = st.selectbox(
                "Filter by Strategy:",
                ["All Categories", "Safe Rightsizing", "Workload Scheduling", "Orphaned Cleanup"],
                key="rec_cat_filter"
            )
        with r_col2:
            risk_filter = st.selectbox(
                "Filter by Risk Level:",
                ["All Risk Levels", "Low", "Medium", "High"],
                key="rec_risk_filter"
            )
        with r_col3:
            status_filter = st.selectbox(
                "Filter by Decision Status:",
                ["All Decisions", "Pending Review", "Accepted", "Rejected"],
                key="rec_status_filter"
            )

        # Apply recommendation filters
        displayed_recs = []
        for rec in recs_filtered:
            rec_id = rec["rec_id"]
            rec_status = st.session_state["rec_decisions"].get(rec_id, {}).get("status", "Pending Review")

            if cat_filter != "All Categories" and rec.get("category") != cat_filter:
                continue
            if risk_filter != "All Risk Levels" and rec.get("risk_level") != risk_filter:
                continue
            if status_filter != "All Decisions" and rec_status != status_filter:
                continue
            displayed_recs.append((rec, rec_status))

        st.caption(f"Showing **{len(displayed_recs)}** recommendations matching selected criteria")
        st.markdown("<br>", unsafe_allow_html=True)

        # Render Recommendation Cards
        for rec, rec_status in displayed_recs:
            rec_id = rec["rec_id"]
            r_id = rec["resource_id"]
            savings = rec["monthly_savings"]
            savings_pct = rec.get("savings_pct", 0)
            risk = rec.get("risk_level", "Low")
            category = rec.get("category", "Optimization")
            
            # Card Container
            with st.container():
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-header">
                        <div>
                            <span style="font-size: 0.82rem; font-weight: 700; color: #2563eb; text-transform: uppercase;">
                                {category}
                            </span>
                            <span style="color: #64748b; font-size: 0.82rem;"> • Resource: <b>{r_id}</b> ({rec.get('resource_name', '')})</span>
                        </div>
                        <div>
                            {get_risk_badge_html(risk)}
                            {get_status_badge_html(rec_status)}
                        </div>
                    </div>
                    <div class="rec-action">
                        {rec['action']}
                    </div>
                    <div style="display: flex; gap: 24px; font-size: 0.85rem; color: #475569; margin: 8px 0;">
                        <div>Current Spec: <b>{rec.get('current_spec', 'N/A')}</b></div>
                        <div>➔ Proposed Spec: <b>{rec.get('proposed_spec', 'N/A')}</b></div>
                    </div>
                    <div class="rec-evidence">
                        <b>Audit & Telemetry Evidence:</b> {rec.get('evidence', '')}
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 10px;">
                        <div>
                            <span style="font-size: 0.82rem; color: #64748b;">Current: {format_inr(rec.get('current_cost', 0))}/mo ➔ Projected: {format_inr(rec.get('projected_cost', 0))}/mo</span><br>
                            <span class="rec-savings">Potential Monthly Savings: {format_inr(savings)}</span>
                            <span style="font-size: 0.85rem; color: #059669; font-weight: 600;">({savings_pct}% reduction)</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Interactive Decision Controls
                btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1.2, 1.2, 1.5, 3])
                
                with btn_col1:
                    if rec_status != "Accepted":
                        if st.button("✓ Accept Action", key=f"btn_acc_{rec_id}", use_container_width=True):
                            accept_recommendation(rec_id)
                            st.toast(f"Accepted recommendation for {r_id}!")
                            st.rerun()
                    else:
                        st.button("✓ Accepted", disabled=True, key=f"btn_acc_dis_{rec_id}", use_container_width=True)

                with btn_col2:
                    if rec_status != "Rejected":
                        with st.popover("✕ Reject Action"):
                            st.markdown(f"**Reject Recommendation for {r_id}**")
                            reject_reason = st.selectbox(
                                "Reason for Rejection:",
                                [
                                    "Workload experiences seasonal peaks not in 30-day window",
                                    "Vendor application architecture requirement",
                                    "Scheduled decommissioning in near future",
                                    "Critical SLA requires oversized buffer",
                                    "Other / Custom Reason"
                                ],
                                key=f"sel_rej_{rec_id}"
                            )
                            if st.button("Confirm Rejection", key=f"btn_conf_rej_{rec_id}"):
                                reject_recommendation(rec_id, reject_reason)
                                st.toast(f"Rejected recommendation for {r_id}")
                                st.rerun()
                    else:
                        if st.button("↺ Reconsider", key=f"btn_rec_{rec_id}", use_container_width=True):
                            reset_recommendation(rec_id)
                            st.rerun()

                with btn_col3:
                    if st.button("📈 View Telemetry", key=f"btn_tel_{rec_id}", use_container_width=True):
                        st.session_state["selected_resource_for_perf"] = r_id
                        st.session_state["nav_page"] = "Performance Analysis"
                        st.rerun()

                with btn_col4:
                    if rec_status == "Accepted":
                        decision_info = st.session_state["rec_decisions"].get(rec_id, {})
                        st.caption(f"Accepted on {decision_info.get('timestamp', 'recently')}")
                    elif rec_status == "Rejected":
                        decision_info = st.session_state["rec_decisions"].get(rec_id, {})
                        st.caption(f"Rejected: {decision_info.get('reason', 'N/A')}")

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# =============================================================================
# PAGE 5: DECISION HISTORY & WHAT-IF SIMULATION
# =============================================================================
elif selected_page == "Decision History & What-If":
    st.markdown("## ⚖️ Decision History & What-If Financial Simulation")
    st.write("Track accepted versus rejected actions, simulate compounding savings without double-counting, and export FinOps audit reports.")

    # Execute What-If Simulation
    accepted_rec_ids = [r_id for r_id, d in st.session_state["rec_decisions"].items() if d["status"] == "Accepted"]
    rejected_rec_ids = [r_id for r_id, d in st.session_state["rec_decisions"].items() if d["status"] == "Rejected"]

    # Use ActionSimulator to accurately calculate compounded savings without double-counting
    if BACKEND_AVAILABLE and not df_vms.empty:
        simulator = ActionSimulator(df_vms, df_resources, recommendations)
        sim_result = simulator.simulate_plan(accepted_rec_ids)
    else:
        # Fallback simulation calculation
        raw_savings = sum(r["monthly_savings"] for r in recommendations if r["rec_id"] in accepted_rec_ids)
        baseline = float(df_inventory["monthly_cost"].sum()) if not df_inventory.empty else 508600.0
        sim_result = {
            "baseline_spend": baseline,
            "projected_spend": baseline - raw_savings,
            "total_savings": raw_savings,
            "savings_pct": round(raw_savings / baseline * 100.0, 1) if baseline > 0 else 0,
            "compounded_count": 0,
            "compounded_details": [],
            "savings_by_category": {"Safe Rightsizing": 0, "Workload Scheduling": 0, "Orphaned Cleanup": 0}
        }

    # Summary KPI Cards
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    with s_col1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Accepted Actions</div>
            <div class="kpi-value" style="color: #059669;">{len(accepted_rec_ids)}</div>
            <div class="kpi-subtext">Out of {len(recommendations)} available</div>
        </div>
        """, unsafe_allow_html=True)

    with s_col2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Rejected Actions</div>
            <div class="kpi-value" style="color: #dc2626;">{len(rejected_rec_ids)}</div>
            <div class="kpi-subtext">Preserved with justification</div>
        </div>
        """, unsafe_allow_html=True)

    with s_col3:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Approved Monthly Savings</div>
            <div class="kpi-value" style="color: #059669;">{format_inr(sim_result['total_savings'])}</div>
            <div class="kpi-subtext">~{sim_result['savings_pct']:.1f}% reduction in spend</div>
        </div>
        """, unsafe_allow_html=True)

    with s_col4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Projected Monthly Spend</div>
            <div class="kpi-value" style="color: #2563eb;">{format_inr(sim_result['projected_spend'])}</div>
            <div class="kpi-subtext">Baseline: {format_inr(sim_result['baseline_spend'])}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Compounded Savings Explanation (Differentiator)
    if sim_result.get("compounded_count", 0) > 0:
        st.markdown("""
        <div style="background-color: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;">
            <div style="font-weight: 700; color: #065f46; font-size: 0.95rem;">
                ✨ Compounded Optimization Active (Double-Counting Prevented)
            </div>
            <div style="font-size: 0.85rem; color: #047857; margin-top: 4px;">
                You have accepted both Rightsizing and Workload Scheduling on one or more instances.
                Our simulation engine automatically downsizes the instance first, and then calculates the off-hours scheduling savings on the downsized instance rate.
                This prevents naive double-counting common in standard cloud cost tools!
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Accepted Actions Table
    st.markdown("### 📋 Approved Optimization Plan")
    if not accepted_rec_ids:
        st.info("No recommendations have been accepted yet. Visit the **Optimization Recommendations** page to review and approve actions.")
    else:
        accepted_records = []
        for r_id in accepted_rec_ids:
            rec = next((r for r in recommendations if r["rec_id"] == r_id), None)
            if rec:
                decision_info = st.session_state["rec_decisions"].get(r_id, {})
                accepted_records.append({
                    "Recommendation ID": rec["rec_id"],
                    "Resource ID": rec["resource_id"],
                    "Category": rec["category"],
                    "Action": rec["action"],
                    "Monthly Savings": format_inr(rec["monthly_savings"]),
                    "Risk Level": rec.get("risk_level", "Low"),
                    "Approved At": decision_info.get("timestamp", "Recent")
                })
        st.dataframe(pd.DataFrame(accepted_records), use_container_width=True, hide_index=True)

    st.divider()

    # Rejected Actions Table
    st.markdown("### 🛑 Rejected Actions & Governance Log")
    if not rejected_rec_ids:
        st.caption("No recommendations have been rejected.")
    else:
        rejected_records = []
        for r_id in rejected_rec_ids:
            rec = next((r for r in recommendations if r["rec_id"] == r_id), None)
            if rec:
                decision_info = st.session_state["rec_decisions"].get(r_id, {})
                rejected_records.append({
                    "Recommendation ID": rec["rec_id"],
                    "Resource ID": rec["resource_id"],
                    "Category": rec["category"],
                    "Action": rec["action"],
                    "Monthly Savings Lost": format_inr(rec["monthly_savings"]),
                    "Justification / Rejection Reason": decision_info.get("reason", "Unspecified"),
                    "Rejected At": decision_info.get("timestamp", "Recent")
                })
        st.dataframe(pd.DataFrame(rejected_records), use_container_width=True, hide_index=True)

    st.divider()

    # Export & Reset Controls
    st.markdown("### 📤 Export & FinOps Reporting")
    col_exp1, col_exp2, col_exp3 = st.columns([2, 2, 2])
    
    with col_exp1:
        if accepted_rec_ids:
            # Build CSV report for accepted decisions
            export_df = pd.DataFrame([
                {
                    "rec_id": r["rec_id"],
                    "resource_id": r["resource_id"],
                    "action": r["action"],
                    "category": r["category"],
                    "monthly_savings": r["monthly_savings"],
                    "risk_level": r.get("risk_level", "Low"),
                    "approved_by": "FinOps Lead",
                    "status": "Approved"
                }
                for r in recommendations if r["rec_id"] in accepted_rec_ids
            ])
            st.download_button(
                label="📥 Export Approved Plan (CSV)",
                data=export_df.to_csv(index=False),
                file_name=f"approved_finops_plan_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
    with col_exp2:
        if st.button("↺ Reset All Accepted/Rejected Decisions", use_container_width=True):
            st.session_state["rec_decisions"] = {}
            st.toast("All decision records have been reset to Pending Review.")
            st.rerun()


# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("<br><hr style='margin: 20px 0;'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.8rem; padding: 10px 0;">
    SmartCloud Optimizer • College Hackathon Edition • Developed with Streamlit, Pandas & Plotly<br>
    <i>Transparent Capacity Safeguards • Anti-Blind-Optimization Guardrails • Non-Destructive In-Memory Simulation</i>
</div>
""", unsafe_allow_html=True)
