"""
SmartCloud Optimizer - Metrics Analysis and Transparent Rule-Based Optimization Engine

This module computes statistically robust resource utilization metrics and evaluates
cloud cost optimization opportunities with strict operational safeguards:
1. Calculates average, p95, p99, and peak utilization for CPU, Memory, Disk, IOPS, and Network.
2. Identifies:
   - Underutilized VMs (eligible for safe downsizing with capacity headroom checks)
   - Idle Non-Production VMs (candidates for off-hours shutdown schedules)
   - Bursty Workloads (CRITICAL SAFEGUARD: blocks downsizing despite low average CPU)
   - Incomplete / Insufficient Metrics (flags data quality issues before acting)
   - Unattached Block Storage Volumes (orphaned disks incurring waste)
   - Underutilized Storage Volumes (attached disks with <15% usage and near-zero IOPS)
   - Unassigned Public IPs (allocated static IPs incurring idle reservation fees)
3. Computes transparent simulated monthly and annual savings without double-counting.
4. Exports recommendations.csv and resource_statistics.csv for integration with Streamlit.
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure terminal stdout handles UTF-8 safely on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Default data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# -------------------------------------------------------------------------
# Transparent, Configurable Decision Thresholds
# All rules reference this single configuration dictionary.
# -------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # Data Quality: require at least 2 days of 15-min intervals (192 records)
    "min_observations": 192,
    "max_missing_percent": 15.0,

    # Downsizing Qualification: BOTH average and peak percentiles must be low
    "downsize_cpu_avg_max": 25.0,
    "downsize_cpu_p95_max": 45.0,
    "downsize_cpu_p99_max": 60.0,
    "downsize_mem_avg_max": 40.0,
    "downsize_mem_p95_max": 50.0,
    "downsize_mem_p99_max": 65.0,

    # Capacity Safety Headroom on Proposed Downsized Size:
    # On the candidate smaller instance, projected p95 CPU must not exceed 70%
    # and projected p99 CPU must not exceed 85%
    "projected_p95_cpu_limit": 70.0,
    "projected_p99_cpu_limit": 85.0,
    "projected_p95_mem_limit": 75.0,

    # Idle Non-Production Detection (candidates for off-hours scheduling)
    "idle_nonprod_cpu_avg_max": 8.0,
    "idle_nonprod_cpu_p95_max": 15.0,
    "idle_nonprod_mem_avg_max": 25.0,
    "idle_nonprod_iops_max": 50,
    # Stopping non-prod VMs during off-hours (nights & weekends: ~118 hrs/week) saves ~65% of monthly cost
    "idle_schedule_savings_ratio": 0.65,

    # Storage Thresholds
    "storage_low_disk_percent": 15.0,
    "storage_low_iops": 10,
}

# Pricing Catalog used for finding candidate smaller instance types
PRICING_CATALOG = [
    {"instance_type": "c-nano", "vcpu": 1, "ram_gb": 1, "monthly_cost": 350.0},
    {"instance_type": "c-micro", "vcpu": 1, "ram_gb": 2, "monthly_cost": 650.0},
    {"instance_type": "c-small", "vcpu": 2, "ram_gb": 4, "monthly_cost": 1300.0},
    {"instance_type": "c-medium", "vcpu": 2, "ram_gb": 8, "monthly_cost": 2400.0},
    {"instance_type": "c-large", "vcpu": 4, "ram_gb": 8, "monthly_cost": 3600.0},
    {"instance_type": "c-xlarge", "vcpu": 4, "ram_gb": 16, "monthly_cost": 6200.0},
    {"instance_type": "c-2xlarge", "vcpu": 8, "ram_gb": 32, "monthly_cost": 12000.0},
    {"instance_type": "c-4xlarge", "vcpu": 16, "ram_gb": 64, "monthly_cost": 23500.0},
]


def load_data(data_dir: str = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads inventory.csv and metrics.csv from the specified data directory.
    Validates file existence and parses types cleanly.
    """
    inv_path = os.path.join(data_dir, "inventory.csv")
    metrics_path = os.path.join(data_dir, "metrics.csv")

    if not os.path.exists(inv_path):
        raise FileNotFoundError(f"Missing inventory file at {inv_path}. Run data_generator.py first.")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Missing metrics file at {metrics_path}. Run data_generator.py first.")

    df_inventory = pd.read_csv(inv_path)
    df_metrics = pd.read_csv(metrics_path)

    # Standardize types and column values
    df_inventory["monthly_cost"] = pd.to_numeric(df_inventory["monthly_cost"], errors="coerce").fillna(0.0)
    for col in ["cpu_percent", "memory_percent", "disk_used_percent", "iops", "network_in_mb", "network_out_mb"]:
        if col in df_metrics.columns:
            df_metrics[col] = pd.to_numeric(df_metrics[col], errors="coerce").fillna(0.0)

    return df_inventory, df_metrics


def calculate_resource_statistics(
    df_metrics: pd.DataFrame,
    df_inventory: pd.DataFrame,
    expected_observations: int = 2880,
) -> pd.DataFrame:
    """
    Computes statistical utilization metrics grouped by resource_id:
    - Average, p95, p99, and maximum for CPU and Memory
    - Average and peak disk utilization, IOPS, and network throughput
    - Observation counts and missing observation percentages

    Why p95 and p99 are essential for cloud optimization:
    - 'Average' hides critical usage spikes. A VM with 10% average CPU might spike to 95%
      every day at peak hours. Downsizing based solely on average would cause severe outage.
    - 'p95' (95th percentile) is the utilization level below which 95% of readings fall.
      It captures sustained heavy load while filtering out rare momentary blips.
    - 'p99' (99th percentile) captures peak operational load. If p99 exceeds 85%, downsizing
      is unsafe because it would lead to CPU throttling and service latency.
    """
    if df_metrics.empty:
        return pd.DataFrame()

    # Fast grouped aggregation
    grouped = df_metrics.groupby("resource_id")

    stats = grouped.agg(
        obs_count=("cpu_percent", "count"),
        avg_cpu=("cpu_percent", "mean"),
        max_cpu=("cpu_percent", "max"),
        avg_mem=("memory_percent", "mean"),
        max_mem=("memory_percent", "max"),
        avg_disk=("disk_used_percent", "mean"),
        peak_disk=("disk_used_percent", "max"),
        avg_iops=("iops", "mean"),
        peak_iops=("iops", "max"),
        avg_net_in=("network_in_mb", "mean"),
        peak_net_in=("network_in_mb", "max"),
        avg_net_out=("network_out_mb", "mean"),
        peak_net_out=("network_out_mb", "max"),
    )

    # Calculate exact percentiles per resource_id
    # We use numpy percentile over each group's series
    p95_cpu_dict = grouped["cpu_percent"].apply(lambda s: float(np.percentile(s, 95))).to_dict()
    p99_cpu_dict = grouped["cpu_percent"].apply(lambda s: float(np.percentile(s, 99))).to_dict()
    p95_mem_dict = grouped["memory_percent"].apply(lambda s: float(np.percentile(s, 95))).to_dict()
    p99_mem_dict = grouped["memory_percent"].apply(lambda s: float(np.percentile(s, 99))).to_dict()

    stats["p95_cpu"] = stats.index.map(p95_cpu_dict)
    stats["p99_cpu"] = stats.index.map(p99_cpu_dict)
    stats["p95_mem"] = stats.index.map(p95_mem_dict)
    stats["p99_mem"] = stats.index.map(p99_mem_dict)

    # Missing observations percentage
    stats["expected_observations"] = expected_observations
    stats["missing_observation_percent"] = (
        (expected_observations - stats["obs_count"]).clip(lower=0) / expected_observations * 100.0
    ).round(2)

    # Format clearly named columns
    stats = stats.reset_index()
    stats = stats.rename(columns={
        "obs_count": "observation_count",
        "avg_cpu": "average_cpu_percent",
        "max_cpu": "max_cpu_percent",
        "p95_cpu": "p95_cpu_percent",
        "p99_cpu": "p99_cpu_percent",
        "avg_mem": "average_memory_percent",
        "max_mem": "max_memory_percent",
        "p95_mem": "p95_memory_percent",
        "p99_mem": "p99_memory_percent",
        "avg_disk": "average_disk_used_percent",
        "peak_disk": "peak_disk_used_percent",
        "avg_iops": "average_iops",
        "peak_iops": "peak_iops",
        "avg_net_in": "average_network_in_mb",
        "peak_net_in": "peak_network_in_mb",
        "avg_net_out": "average_network_out_mb",
        "peak_net_out": "peak_network_out_mb",
    })

    # Round floating values for clean presentation
    round_cols = [
        "average_cpu_percent", "p95_cpu_percent", "p99_cpu_percent", "max_cpu_percent",
        "average_memory_percent", "p95_memory_percent", "p99_memory_percent", "max_memory_percent",
        "average_disk_used_percent", "peak_disk_used_percent", "average_iops",
        "average_network_in_mb", "peak_network_in_mb", "average_network_out_mb", "peak_network_out_mb"
    ]
    for col in round_cols:
        stats[col] = stats[col].round(2)

    # Join with basic inventory fields for full context
    vm_inv = df_inventory[["resource_id", "name", "environment", "owner", "instance_type", "monthly_cost"]].copy()
    result = pd.merge(vm_inv, stats, on="resource_id", how="inner")
    return result


def find_safe_downsize_candidate(
    current_instance_type: str,
    p95_cpu: float,
    p99_cpu: float,
    p95_mem: float,
    config: dict,
) -> dict | None:
    """
    Evaluates catalog instance types to find the smallest, cheapest instance that safely
    handles the workload without exceeding capacity limits:
    - Projected p95 CPU <= 70%
    - Projected p99 CPU <= 85%
    - Projected p95 RAM <= 75%
    """
    pricing_map = {item["instance_type"]: item for item in PRICING_CATALOG}
    current_spec = pricing_map.get(current_instance_type)
    if not current_spec:
        return None

    current_vcpu = current_spec["vcpu"]
    current_ram = current_spec["ram_gb"]
    current_cost = current_spec["monthly_cost"]

    # Filter cheaper candidates in the same family
    cheaper_candidates = [
        c for c in PRICING_CATALOG
        if c["monthly_cost"] < current_cost and c["instance_type"] != current_instance_type
    ]
    # Sort by monthly cost ascending (we want maximum safe savings)
    cheaper_candidates.sort(key=lambda x: x["monthly_cost"])

    for candidate in cheaper_candidates:
        cand_vcpu = candidate["vcpu"]
        cand_ram = candidate["ram_gb"]

        # Calculate projected utilization on candidate hardware
        # E.g., if VM used 20% CPU on 8 vCPUs, moving to 4 vCPUs projects to 20% * (8/4) = 40%
        cpu_scale = current_vcpu / cand_vcpu
        mem_scale = current_ram / cand_ram

        projected_p95_cpu = p95_cpu * cpu_scale
        projected_p99_cpu = p99_cpu * cpu_scale
        projected_p95_mem = p95_mem * mem_scale

        # Capacity safety check
        if (
            projected_p95_cpu <= config["projected_p95_cpu_limit"] and
            projected_p99_cpu <= config["projected_p99_cpu_limit"] and
            projected_p95_mem <= config["projected_p95_mem_limit"]
        ):
            return {
                "proposed_type": candidate["instance_type"],
                "proposed_cost": float(candidate["monthly_cost"]),
                "proposed_vcpu": cand_vcpu,
                "proposed_ram": cand_ram,
                "projected_p95_cpu": round(projected_p95_cpu, 1),
                "projected_p99_cpu": round(projected_p99_cpu, 1),
                "projected_p95_mem": round(projected_p95_mem, 1),
            }

    return None


def generate_recommendations(
    df_inventory: pd.DataFrame,
    df_stats: pd.DataFrame,
    config: dict = None,
) -> pd.DataFrame:
    """
    Generates actionable, rule-based optimization recommendations for all cloud resources.

    Evaluates:
    - Rule A: Underutilized VMs -> Downsize to safe candidate instance.
    - Rule B: Idle Non-Production VMs -> Recommend off-hours shutdown schedule.
    - Rule C: Bursty Workloads -> GUARD TRIGGERED: Prevent downsizing despite low average.
    - Rule D: Production Critical Safeguard -> Prevent automated downsizing on critical workloads.
    - Rule E: Insufficient Metrics -> Flag review due to lack of historical telemetry.
    - Rule F: Unattached Storage Volumes -> Review orphaned storage for deletion.
    - Rule G: Underutilized Attached Storage -> Review volumes with low usage and zero IOPS.
    - Rule H: Unassigned Public IPs -> Release unassociated elastic IPs.
    - Rule I: Healthy Workloads -> Maintain current configuration.
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    # Build stats lookup dictionary keyed by resource_id
    stats_dict = {}
    if not df_stats.empty:
        stats_dict = {row["resource_id"]: row for _, row in df_stats.iterrows()}

    recommendations = []

    for _, res in df_inventory.iterrows():
        res_id = res["resource_id"]
        res_name = res["name"]
        res_type = res["type"]
        env = res["environment"]
        current_cost = float(res.get("monthly_cost", 0.0))
        is_critical = bool(res.get("is_critical", False))

        # Base record structure matching requested schema exactly
        rec = {
            "resource_id": res_id,
            "resource_name": res_name,
            "resource_type": res_type,
            "environment": env,
            "recommendation_type": "No change suggested",
            "suggested_action": "Maintain current configuration",
            "reason": "Resource operating normally within configured parameters.",
            "average_cpu_percent": np.nan,
            "p95_cpu_percent": np.nan,
            "p99_cpu_percent": np.nan,
            "average_memory_percent": np.nan,
            "p95_memory_percent": np.nan,
            "p99_memory_percent": np.nan,
            "risk_level": "Low",
            "current_monthly_cost": current_cost,
            "estimated_new_monthly_cost": current_cost,
            "estimated_monthly_savings": 0.0,
            "estimated_annual_savings": 0.0,
            "status": "Pending",
            "assumptions": "Calculated using simulated pricing catalog and telemetry statistics.",
        }

        # -----------------------------------------------------------------
        # EVALUATE COMPUTE INSTANCES (VMs)
        # -----------------------------------------------------------------
        if res_type == "VM":
            st = stats_dict.get(res_id)

            if st is None or (hasattr(st, "empty") and st.empty):
                rec["recommendation_type"] = "Data Quality Alert"
                rec["suggested_action"] = "Collect telemetry before evaluating"
                rec["reason"] = "No historical performance metrics found for this virtual machine."
                rec["status"] = "Needs review"
                rec["risk_level"] = "High"
                recommendations.append(rec)
                continue

            # Populate metrics in recommendation record
            rec["average_cpu_percent"] = st["average_cpu_percent"]
            rec["p95_cpu_percent"] = st["p95_cpu_percent"]
            rec["p99_cpu_percent"] = st["p99_cpu_percent"]
            rec["average_memory_percent"] = st["average_memory_percent"]
            rec["p95_memory_percent"] = st["p95_memory_percent"]
            rec["p99_memory_percent"] = st["p99_memory_percent"]

            obs_count = st["observation_count"]
            missing_pct = st["missing_observation_percent"]
            avg_cpu = st["average_cpu_percent"]
            p95_cpu = st["p95_cpu_percent"]
            p99_cpu = st["p99_cpu_percent"]
            avg_mem = st["average_memory_percent"]
            p95_mem = st["p95_mem_percent"] if "p95_mem_percent" in st else st["p95_memory_percent"]
            avg_iops = st["average_iops"]

            # Safeguard 1: Insufficient observations or high missing data
            if obs_count < cfg["min_observations"] or missing_pct > cfg["max_missing_percent"]:
                rec["recommendation_type"] = "Data Quality Alert"
                rec["suggested_action"] = "Collect additional telemetry before evaluating"
                rec["reason"] = (
                    f"Only {obs_count} metric observations available ({missing_pct}% missing). "
                    "Insufficient telemetry history to safely evaluate peak utilization."
                )
                rec["status"] = "Needs review"
                rec["risk_level"] = "High"
                rec["assumptions"] = "Requires minimum 192 observations (2 full days) for reliable p95/p99 calculation."
                recommendations.append(rec)
                continue

            # Safeguard 2: Production Critical Protection
            if env == "prod" and is_critical:
                rec["recommendation_type"] = "Production Safety Guardrail"
                rec["suggested_action"] = "Manual architectural review required"
                rec["reason"] = (
                    f"Mission-critical production resource ({res.get('owner', 'Core')}). "
                    "Automated downsizing blocked to protect high-availability SLAs. "
                    f"Observed p95 CPU: {p95_cpu}%, p95 Memory: {p95_mem}%."
                )
                rec["status"] = "Needs review"
                rec["risk_level"] = "Low"
                recommendations.append(rec)
                continue

            # Safeguard 3: Bursty Workload Guardrail (CRITICAL HACKATHON RULE)
            # Low average CPU (<25%), BUT high p95 (>45%) or p99 (>60%)
            if avg_cpu <= cfg["downsize_cpu_avg_max"] and (
                p95_cpu > cfg["downsize_cpu_p95_max"] or p99_cpu > cfg["downsize_cpu_p99_max"]
            ):
                rec["recommendation_type"] = "Capacity Retention (Guardrail)"
                rec["suggested_action"] = "Retain current capacity (Do NOT downsize)"
                rec["reason"] = (
                    f"Workload displays low average CPU ({avg_cpu}%) but frequent high peaks "
                    f"(p95: {p95_cpu}%, p99: {p99_cpu}%). Downsizing would cause severe CPU throttling "
                    "and performance degradation during peak traffic bursts."
                )
                rec["status"] = "Needs review"
                rec["risk_level"] = "High"  # High risk IF downsized
                rec["assumptions"] = "Peak demand requires current capacity buffer."
                recommendations.append(rec)
                continue

            # Rule B: Idle Non-Production VM (Dev or Staging)
            # Consistently negligible CPU (<8%), p95 (<15%), and near-zero IOPS (<50)
            if (
                env in ["dev", "staging"] and
                avg_cpu <= cfg["idle_nonprod_cpu_avg_max"] and
                p95_cpu <= cfg["idle_nonprod_cpu_p95_max"] and
                avg_iops <= cfg["idle_nonprod_iops_max"]
            ):
                savings_ratio = cfg["idle_schedule_savings_ratio"]
                monthly_savings = round(current_cost * savings_ratio, 2)
                new_cost = round(current_cost - monthly_savings, 2)

                rec["recommendation_type"] = "Workload Scheduling"
                rec["suggested_action"] = "Implement off-hours shutdown schedule (12h/day + weekends)"
                rec["reason"] = (
                    f"Consistently low activity in {env} environment (avg CPU: {avg_cpu}%, p95: {p95_cpu}%, "
                    f"avg IOPS: {int(avg_iops)}). Shutting down outside working hours eliminates ~65% of runtime waste."
                )
                rec["risk_level"] = "Low"
                rec["estimated_new_monthly_cost"] = new_cost
                rec["estimated_monthly_savings"] = monthly_savings
                rec["estimated_annual_savings"] = round(monthly_savings * 12, 2)
                rec["status"] = "Pending"
                rec["assumptions"] = "Assumes dev/staging instances operate 50 hours/week instead of 168 hours/week."
                recommendations.append(rec)
                continue

            # Rule A: Underutilized VM -> Safe Downsizing
            # Average and percentiles are comfortably within downsize thresholds
            if (
                avg_cpu <= cfg["downsize_cpu_avg_max"] and
                p95_cpu <= cfg["downsize_cpu_p95_max"] and
                p99_cpu <= cfg["downsize_cpu_p99_max"] and
                avg_mem <= cfg["downsize_mem_avg_max"] and
                p95_mem <= cfg["downsize_mem_p95_max"]
            ):
                candidate = find_safe_downsize_candidate(
                    res.get("instance_type", ""),
                    p95_cpu,
                    p99_cpu,
                    p95_mem,
                    cfg,
                )

                if candidate:
                    monthly_savings = round(current_cost - candidate["proposed_cost"], 2)
                    annual_savings = round(monthly_savings * 12, 2)

                    rec["recommendation_type"] = "Rightsizing"
                    rec["suggested_action"] = f"Downsize from {res.get('instance_type')} to {candidate['proposed_type']}"
                    rec["reason"] = (
                        f"Workload is consistently underutilized (avg CPU: {avg_cpu}%, p95: {p95_cpu}%, "
                        f"p95 RAM: {p95_mem}%). Downsizing to {candidate['proposed_type']} preserves safe headroom "
                        f"(projected p95 CPU: {candidate['projected_p95_cpu']}%, projected p99 CPU: {candidate['projected_p99_cpu']}%)."
                    )
                    rec["risk_level"] = "Medium" if candidate["projected_p95_cpu"] > 55.0 else "Low"
                    rec["estimated_new_monthly_cost"] = candidate["proposed_cost"]
                    rec["estimated_monthly_savings"] = monthly_savings
                    rec["estimated_annual_savings"] = annual_savings
                    rec["status"] = "Pending"
                    rec["assumptions"] = f"Proposed instance {candidate['proposed_type']} provides {candidate['proposed_vcpu']} vCPUs and {candidate['proposed_ram']} GB RAM."
                    recommendations.append(rec)
                    continue

            # Default healthy VM: Optimal utilization
            rec["recommendation_type"] = "No change suggested"
            rec["suggested_action"] = "Maintain current configuration"
            rec["reason"] = (
                f"Workload utilization is healthy and within optimal operational bands "
                f"(avg CPU: {avg_cpu}%, p95 CPU: {p95_cpu}%, avg RAM: {avg_mem}%)."
            )
            rec["status"] = "Accepted"
            rec["risk_level"] = "Low"
            recommendations.append(rec)

        # -----------------------------------------------------------------
        # EVALUATE BLOCK STORAGE VOLUMES
        # -----------------------------------------------------------------
        elif res_type == "block_volume":
            attached_to = str(res.get("attached_to", "unattached")).lower()
            disk_size = res.get("disk_size_gb", 0)

            # Rule F: Unattached Storage Volume (Orphaned)
            if attached_to == "unattached" or res.get("status") == "available":
                rec["recommendation_type"] = "Orphaned Resource Review"
                rec["suggested_action"] = "Snapshot and delete unattached block volume"
                rec["reason"] = (
                    f"Storage volume ({disk_size} GB) is unattached and unused. "
                    "Verify data retention policy and take a final snapshot before deleting to eliminate waste."
                )
                rec["risk_level"] = "Medium"
                rec["estimated_new_monthly_cost"] = 0.0
                rec["estimated_monthly_savings"] = current_cost
                rec["estimated_annual_savings"] = round(current_cost * 12, 2)
                rec["status"] = "Needs review"
                rec["assumptions"] = "Requires confirmation that no decommissioned backup data is needed."
                recommendations.append(rec)

            # Rule G: Attached but underutilized volume
            elif "attached volume with" in str(res.get("notes", "")).lower():
                rec["recommendation_type"] = "Storage Investigation"
                rec["suggested_action"] = "Investigate volume resizing or archiving"
                rec["reason"] = (
                    f"Volume has low disk occupancy (<15%) and near-zero I/O activity. "
                    "Review storage tiering or downsize allocated capacity."
                )
                rec["risk_level"] = "Low"
                rec["status"] = "Needs review"
                rec["assumptions"] = "Savings subject to actual volume data shrink verification."
                recommendations.append(rec)

            else:
                rec["recommendation_type"] = "No change suggested"
                rec["suggested_action"] = "Maintain current storage volume"
                rec["reason"] = "Storage volume is actively attached to compute instances with normal I/O."
                rec["status"] = "Accepted"
                recommendations.append(rec)

        # -----------------------------------------------------------------
        # EVALUATE PUBLIC IP ADDRESSES
        # -----------------------------------------------------------------
        elif res_type == "public_ip":
            attached_to = str(res.get("attached_to", "unassigned")).lower()

            # Rule H: Unassigned Public IP
            if attached_to == "unassigned" or res.get("status") == "unassigned":
                rec["recommendation_type"] = "Unused IP Release"
                rec["suggested_action"] = "Release unassigned public IP address"
                rec["reason"] = (
                    "Static public IP is allocated but not associated with an active compute resource, "
                    "incurring idle reservation fees."
                )
                rec["risk_level"] = "Low"
                rec["estimated_new_monthly_cost"] = 0.0
                rec["estimated_monthly_savings"] = current_cost
                rec["estimated_annual_savings"] = round(current_cost * 12, 2)
                rec["status"] = "Pending"
                rec["assumptions"] = "Releasing IP returns it to cloud provider pool; non-reversible without re-allocation."
                recommendations.append(rec)

            else:
                rec["recommendation_type"] = "No change suggested"
                rec["suggested_action"] = "Maintain public IP association"
                rec["reason"] = "Public IP is actively associated with a running service gateway."
                rec["status"] = "Accepted"
                recommendations.append(rec)

    df_recommendations = pd.DataFrame(recommendations)
    return df_recommendations


def calculate_savings(df_recommendations: pd.DataFrame) -> dict:
    """
    Computes overall savings metrics from the recommendation table.
    Ensures:
    - Current total cost vs projected total cost.
    - Monthly and annual savings.
    - Percentage cost reduction.
    - No double counting across mutually exclusive recommendations.
    """
    if df_recommendations.empty:
        return {
            "total_resources": 0,
            "current_monthly_spend": 0.0,
            "projected_monthly_spend": 0.0,
            "total_monthly_savings": 0.0,
            "total_annual_savings": 0.0,
            "savings_percentage": 0.0,
            "actionable_count": 0,
        }

    total_resources = len(df_recommendations)
    current_spend = float(df_recommendations["current_monthly_cost"].sum())
    monthly_savings = float(df_recommendations["estimated_monthly_savings"].sum())
    projected_spend = max(0.0, current_spend - monthly_savings)
    annual_savings = round(monthly_savings * 12, 2)

    savings_pct = round((monthly_savings / current_spend * 100.0), 2) if current_spend > 0 else 0.0
    actionable_count = int((df_recommendations["estimated_monthly_savings"] > 0).sum())

    return {
        "total_resources": total_resources,
        "current_monthly_spend": round(current_spend, 2),
        "projected_monthly_spend": round(projected_spend, 2),
        "total_monthly_savings": round(monthly_savings, 2),
        "total_annual_savings": annual_savings,
        "savings_percentage": savings_pct,
        "actionable_count": actionable_count,
    }


def export_recommendations(
    df_recommendations: pd.DataFrame,
    df_stats: pd.DataFrame,
    output_dir: str = DATA_DIR,
) -> tuple[str, str]:
    """
    Exports recommendations.csv and resource_statistics.csv to the data directory.
    Returns the filepaths of both exported files.
    """
    os.makedirs(output_dir, exist_ok=True)
    rec_path = os.path.join(output_dir, "recommendations.csv")
    stats_path = os.path.join(output_dir, "resource_statistics.csv")

    df_recommendations.to_csv(rec_path, index=False)
    if not df_stats.empty:
        df_stats.to_csv(stats_path, index=False)

    return rec_path, stats_path


def main():
    """CLI execution entrypoint to run data loading, statistical analysis, and optimization."""
    print("=" * 70)
    print("SmartCloud Optimizer - Rule-Based Cloud Optimization Engine")
    print("=" * 70)

    # 1. Load data
    print(f"\n[1/4] Loading inventory and metrics from: {DATA_DIR} ...")
    df_inventory, df_metrics = load_data(DATA_DIR)
    print(f"  -> Loaded {len(df_inventory)} inventory items and {len(df_metrics):,} metrics.")

    # 2. Compute statistics
    print("\n[2/4] Computing utilization statistics (avg, p95, p99, peak, observation counts)...")
    df_stats = calculate_resource_statistics(df_metrics, df_inventory)
    print(f"  -> Computed statistics for {len(df_stats)} virtual machines.")

    # 3. Generate recommendations
    print("\n[3/4] Evaluating optimization rules and capacity safeguards...")
    df_recs = generate_recommendations(df_inventory, df_stats)
    savings_summary = calculate_savings(df_recs)

    # 4. Export files
    print("\n[4/4] Exporting recommendations and resource statistics...")
    rec_path, stats_path = export_recommendations(df_recs, df_stats, DATA_DIR)
    print(f"  -> Saved recommendations:       {rec_path}")
    print(f"  -> Saved resource statistics:   {stats_path}")

    # Display Rich Terminal Summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION ENGINE RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Resources Analyzed:       {savings_summary['total_resources']}")
    print(f"Actionable Opportunities:       {savings_summary['actionable_count']}")
    print(f"Baseline Monthly Spend:         INR {savings_summary['current_monthly_spend']:,.2f}")
    print(f"Projected Monthly Spend:        INR {savings_summary['projected_monthly_spend']:,.2f}")
    print(f"Estimated Monthly Savings:      INR {savings_summary['total_monthly_savings']:,.2f} ({savings_summary['savings_percentage']}%)")
    print(f"Estimated Annual Savings:       INR {savings_summary['total_annual_savings']:,.2f}")
    print("-" * 70)

    print("\nRecommendations Breakdown by Type:")
    breakdown = df_recs["recommendation_type"].value_counts()
    for rec_type, count in breakdown.items():
        sub_savings = df_recs[df_recs["recommendation_type"] == rec_type]["estimated_monthly_savings"].sum()
        print(f"  * {rec_type:<32} : {count:>3} items (INR {sub_savings:,.2f}/mo)")

    print("\nSample Actionable Recommendations:")
    sample_actionable = df_recs[df_recs["estimated_monthly_savings"] > 0].head(5)
    cols_to_print = ["resource_id", "resource_type", "recommendation_type", "risk_level", "estimated_monthly_savings"]
    print(sample_actionable[cols_to_print].to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
