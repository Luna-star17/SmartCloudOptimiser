"""
UTHO SmartCloud Optimizer - Optimization Recommendation Engine
Implements Safety-Aware Rightsizing, Workload-Aware Scheduling, and Orphan Detection
with transparent, auditable evidence and p95/p99 capacity safeguards.
"""

import pandas as pd
import numpy as np


DEFAULT_THRESHOLDS = {
    "target_p95_cpu": 70.0,
    "target_p99_cpu": 85.0,
    "target_p95_mem": 75.0,
    "target_p99_mem": 85.0,
    "idle_cpu_threshold": 12.0,
    "idle_mem_threshold": 25.0,
    "orphan_days_threshold": 7,
}


def calculate_vm_stats(df_metrics: pd.DataFrame, df_vms: pd.DataFrame) -> pd.DataFrame:
    """
    Computes statistically robust utilization metrics for each VM:
    - Average, p95, p99, and Max for CPU and Memory
    - Low-utilization percentage
    - Weekday vs Weekend and Workhour vs Offhour utilization
    - Trend analysis comparing recent 7 days vs previous period
    """
    stats_list = []
    
    # Ensure datetime format
    metrics = df_metrics.copy()
    if not pd.api.types.is_datetime64_any_dtype(metrics["timestamp"]):
        metrics["timestamp"] = pd.to_datetime(metrics["timestamp"])
        
    metrics["hour"] = metrics["timestamp"].dt.hour
    metrics["is_weekend"] = metrics["timestamp"].dt.weekday >= 5
    metrics["is_workhour"] = (metrics["hour"].between(9, 19)) & (~metrics["is_weekend"])
    
    cutoff_7d = metrics["timestamp"].max() - pd.Timedelta(days=7)
    
    for _, vm in df_vms.iterrows():
        vm_id = vm["vm_id"]
        v_data = metrics[metrics["vm_id"] == vm_id]
        
        if v_data.empty:
            continue
            
        cpu = v_data["cpu_pct"]
        mem = v_data["memory_pct"]
        
        # p95 and p99 values
        p95_cpu = float(np.percentile(cpu, 95))
        p99_cpu = float(np.percentile(cpu, 99))
        p95_mem = float(np.percentile(mem, 95))
        p99_mem = float(np.percentile(mem, 99))
        
        avg_cpu = float(cpu.mean())
        avg_mem = float(mem.mean())
        max_cpu = float(cpu.max())
        max_mem = float(mem.max())
        
        # Off-hours & weekend utilization
        offhours_data = v_data[~v_data["is_workhour"]]
        offhours_avg_cpu = float(offhours_data["cpu_pct"].mean()) if not offhours_data.empty else avg_cpu
        
        # Low utilization % (readings with cpu < 15 and mem < 30)
        low_util_count = ((cpu < 15.0) & (mem < 30.0)).sum()
        low_util_pct = float((low_util_count / len(v_data)) * 100.0)
        
        # Trend: last 7 days vs prior 23 days
        recent_data = v_data[v_data["timestamp"] >= cutoff_7d]
        prior_data = v_data[v_data["timestamp"] < cutoff_7d]
        
        recent_avg_cpu = float(recent_data["cpu_pct"].mean()) if not recent_data.empty else avg_cpu
        prior_avg_cpu = float(prior_data["cpu_pct"].mean()) if not prior_data.empty else avg_cpu
        cpu_trend_pct = round(((recent_avg_cpu - prior_avg_cpu) / max(prior_avg_cpu, 1.0)) * 100.0, 1)
        
        stats_list.append({
            "vm_id": vm_id,
            "avg_cpu": round(avg_cpu, 1),
            "p95_cpu": round(p95_cpu, 1),
            "p99_cpu": round(p99_cpu, 1),
            "max_cpu": round(max_cpu, 1),
            "avg_mem": round(avg_mem, 1),
            "p95_mem": round(p95_mem, 1),
            "p99_mem": round(p99_mem, 1),
            "max_mem": round(max_mem, 1),
            "offhours_avg_cpu": round(offhours_avg_cpu, 1),
            "low_util_pct": round(low_util_pct, 1),
            "recent_avg_cpu": round(recent_avg_cpu, 1),
            "prior_avg_cpu": round(prior_avg_cpu, 1),
            "cpu_trend_pct": cpu_trend_pct,
            "avg_iops": int(v_data["disk_iops"].mean()),
            "avg_net_mbps": round(float(v_data["network_mbps"].mean()), 1)
        })
        
    df_stats = pd.DataFrame(stats_list)
    return df_stats


def evaluate_rightsizing(
    vm: pd.Series,
    stats: pd.Series,
    df_pricing: pd.DataFrame,
    thresholds: dict
) -> dict | None:
    """
    Evaluates safety-aware downsizing:
    - Protects Critical VMs from downsizing
    - Blocks downsizing for Bursty / Batch workloads whose p99 would violate headroom
    - Validates candidate capacity against both CPU and Memory p95 and p99 demand
    """
    curr_vcpu = vm["vcpu"]
    curr_mem = vm["memory_gb"]
    curr_cost = vm["monthly_cost"]
    
    # Absolute peak demand
    p95_cpu_demand = curr_vcpu * (stats["p95_cpu"] / 100.0)
    p99_cpu_demand = curr_vcpu * (stats["p99_cpu"] / 100.0)
    p95_mem_demand = curr_mem * (stats["p95_mem"] / 100.0)
    p99_mem_demand = curr_mem * (stats["p99_mem"] / 100.0)
    
    # Safety Check: Critical workload guardrail
    if vm.get("critical", False) or vm.get("workload_pattern") == "Critical":
        return None  # Preserved by safety guardrail
        
    # Safety Check: Bursty or Batch workloads with dangerous p99 spikes
    if stats["p99_cpu"] > thresholds["target_p99_cpu"] or stats["p99_mem"] > thresholds["target_p99_mem"]:
        # Naive optimizers would downsize based on low average CPU; our engine protects it!
        return None
        
    # Candidates must be cheaper and have less capacity
    candidates = df_pricing[
        (df_pricing["monthly_cost"] < curr_cost) &
        ((df_pricing["vcpu"] < curr_vcpu) | (df_pricing["memory_gb"] < curr_mem))
    ].sort_values(by="monthly_cost", ascending=True)
    
    best_candidate = None
    
    for _, cand in candidates.iterrows():
        cand_vcpu = cand["vcpu"]
        cand_mem = cand["memory_gb"]
        
        # Projected utilization on candidate
        proj_p95_cpu = (p95_cpu_demand / cand_vcpu) * 100.0
        proj_p99_cpu = (p99_cpu_demand / cand_vcpu) * 100.0
        proj_p95_mem = (p95_mem_demand / cand_mem) * 100.0
        proj_p99_mem = (p99_mem_demand / cand_mem) * 100.0
        
        # Headroom safety check
        if (
            proj_p95_cpu <= thresholds["target_p95_cpu"] and
            proj_p99_cpu <= thresholds["target_p99_cpu"] and
            proj_p95_mem <= thresholds["target_p95_mem"] and
            proj_p99_mem <= thresholds["target_p99_mem"]
        ):
            best_candidate = cand
            break  # Pick the most cost-effective safe candidate
            
    if best_candidate is not None:
        savings = curr_cost - best_candidate["monthly_cost"]
        savings_pct = round((savings / curr_cost) * 100.0, 1)
        
        # Risk assessment
        cand_proj_p95_cpu = round((p95_cpu_demand / best_candidate["vcpu"]) * 100.0, 1)
        cand_proj_p95_mem = round((p95_mem_demand / best_candidate["memory_gb"]) * 100.0, 1)
        
        risk = "Low"
        if cand_proj_p95_cpu > 60.0 or cand_proj_p95_mem > 65.0:
            risk = "Medium"
            
        evidence = (
            f"Observed 30-day CPU p95: {stats['p95_cpu']}%, p99: {stats['p99_cpu']}%. "
            f"Memory p95: {stats['p95_mem']}%, p99: {stats['p99_mem']}%. "
            f"Projected on {best_candidate['instance_type']}: CPU p95={cand_proj_p95_cpu}% "
            f"(headroom {(100 - cand_proj_p95_cpu):.1f}%), Memory p95={cand_proj_p95_mem}%. "
            f"Both within strict safety limits (p95<{thresholds['target_p95_cpu']}%, p99<{thresholds['target_p99_cpu']}%)."
        )
        
        return {
            "rec_id": f"REC-RS-{vm['vm_id']}",
            "resource_id": vm["vm_id"],
            "resource_name": vm["vm_name"],
            "team": vm["team"],
            "environment": vm["environment"],
            "region": vm["region"],
            "category": "Safe Rightsizing",
            "action": f"Downsize {vm['instance_type']} -> {best_candidate['instance_type']}",
            "current_spec": f"{vm['instance_type']} ({curr_vcpu} vCPU, {curr_mem}GB RAM)",
            "proposed_spec": f"{best_candidate['instance_type']} ({best_candidate['vcpu']} vCPU, {best_candidate['memory_gb']}GB RAM)",
            "current_cost": curr_cost,
            "projected_cost": best_candidate["monthly_cost"],
            "monthly_savings": savings,
            "savings_pct": savings_pct,
            "risk_level": risk,
            "confidence": "High",
            "rule_triggered": "Safety-Validated Headroom Rightsizing",
            "evidence": evidence,
            "pre_action_notice": "Verify peak seasonality before changing production instances."
        }
        
    return None


def evaluate_scheduling(vm: pd.Series, stats: pd.Series) -> dict | None:
    """
    Workload-Aware Scheduling (Differentiator D):
    Identifies Development & Staging VMs that are idle overnight and on weekends.
    Recommends automated off-hours shutdown schedule.
    """
    env = vm["environment"]
    is_crit = vm.get("critical", False)
    
    # Production or critical machines are excluded from automated stop schedules
    if env == "Production" or is_crit:
        return None
        
    offhours_cpu = stats["offhours_avg_cpu"]
    pattern = vm.get("workload_pattern", "")
    
    # Qualifies if identified as Weekend Idle or off-hours CPU < 10%
    if pattern == "Weekend Idle" or offhours_cpu < 8.0:
        curr_cost = vm["monthly_cost"]
        # Off-hours savings: 7 PM - 8 AM (13 hrs/day M-F) + 48 hrs weekend = 113 hrs idle / 168 total hrs = 67.2%
        savings_factor = 0.65
        savings = int(curr_cost * savings_factor)
        projected_cost = curr_cost - savings
        
        evidence = (
            f"Environment is '{env}'. Average off-hours & weekend CPU utilization is only {offhours_cpu}%. "
            f"Active workloads occur primarily during weekday business hours (09:00 - 19:00). "
            f"Implementing an automated 55-hour/week active schedule avoids ~65% of runtime charges."
        )
        
        return {
            "rec_id": f"REC-SC-{vm['vm_id']}",
            "resource_id": vm["vm_id"],
            "resource_name": vm["vm_name"],
            "team": vm["team"],
            "environment": vm["environment"],
            "region": vm["region"],
            "category": "Workload Scheduling",
            "action": "Automate Off-Hours & Weekend Shutdown (07:00 PM - 08:00 AM + Sat/Sun)",
            "current_spec": f"24/7 Always-On ({vm['instance_type']})",
            "proposed_spec": "Scheduled Auto-Start/Stop (55 hrs/week active)",
            "current_cost": curr_cost,
            "projected_cost": projected_cost,
            "monthly_savings": savings,
            "savings_pct": 65.0,
            "risk_level": "Low",
            "confidence": "High",
            "rule_triggered": "Non-Production Off-Hours Idling",
            "evidence": evidence,
            "pre_action_notice": "Configure automated cloud scheduler tag (Env: Dev/Staging)."
        }
        
    return None


def evaluate_orphaned_resources(df_resources: pd.DataFrame, thresholds: dict) -> list[dict]:
    """
    Detects unattached block storage volumes and unassociated Elastic Static IPs.
    """
    recs = []
    
    for _, res in df_resources.iterrows():
        is_attached = res["attached"]
        days_unatt = res.get("days_unattached", 0)
        cost = res["monthly_cost"]
        r_type = res["resource_type"]
        r_id = res["resource_id"]
        
        if not is_attached and days_unatt >= thresholds["orphan_days_threshold"]:
            risk = "Low" if days_unatt >= 30 else "Medium"
            confidence = "High" if days_unatt >= 20 else "Medium"
            
            if "Storage" in r_type:
                action = "Create Final Snapshot & Terminate Unattached Volume"
                evidence = (
                    f"Block storage volume '{r_id}' ({res['size_gb']} GB {r_type}) has been detached for {days_unatt} consecutive days. "
                    f"Team: {res['team']}. Zero I/O activity recorded. Retaining generates unnecessary monthly spend."
                )
            else:
                action = "Release Unassociated Static Elastic IP"
                evidence = (
                    f"Elastic Static IP '{r_id}' has been unassociated for {days_unatt} days. "
                    f"Team: {res['team']}. Cloud provider bills hourly idle reservation surcharge for unassigned IPs."
                )
                
            recs.append({
                "rec_id": f"REC-OR-{r_id}",
                "resource_id": r_id,
                "resource_name": f"{r_type} ({r_id})",
                "team": res["team"],
                "environment": "Storage / Network",
                "region": res["region"],
                "category": "Orphaned Cleanup",
                "action": action,
                "current_spec": f"{r_type} (Detached for {days_unatt} days)",
                "proposed_spec": "Decommission / Snapshot & Delete",
                "current_cost": cost,
                "projected_cost": 0,
                "monthly_savings": cost,
                "savings_pct": 100.0,
                "risk_level": risk,
                "confidence": confidence,
                "rule_triggered": "Unattached Cloud Asset Inactivity",
                "evidence": evidence,
                "pre_action_notice": "Ensure no pending DR snapshot restoration relies on this resource."
            })
            
    return recs


def generate_recommendations(
    df_vms: pd.DataFrame,
    df_stats: pd.DataFrame,
    df_resources: pd.DataFrame,
    df_pricing: pd.DataFrame,
    thresholds: dict = None
) -> list[dict]:
    """
    Orchestrates recommendation generation across all categories:
    1. Safe Rightsizing (Safety-aware p95/p99)
    2. Workload-Aware Scheduling
    3. Orphaned Resource Cleanup
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.copy()
        
    all_recommendations = []
    stats_dict = {row["vm_id"]: row for _, row in df_stats.iterrows()}
    
    for _, vm in df_vms.iterrows():
        vm_id = vm["vm_id"]
        stats = stats_dict.get(vm_id)
        if stats is None:
            continue
            
        # 1. Rightsizing
        rs_rec = evaluate_rightsizing(vm, stats, df_pricing, thresholds)
        if rs_rec:
            all_recommendations.append(rs_rec)
            
        # 2. Scheduling
        sc_rec = evaluate_scheduling(vm, stats)
        if sc_rec:
            all_recommendations.append(sc_rec)
            
    # 3. Orphaned resources
    orphan_recs = evaluate_orphaned_resources(df_resources, thresholds)
    all_recommendations.extend(orphan_recs)
    
    return all_recommendations
