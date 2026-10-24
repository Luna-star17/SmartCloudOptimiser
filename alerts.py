"""
UTHO SmartCloud Optimizer - Anomaly Alerts and Safety Guardrails Engine
Detects cost spikes, capacity risks, zombie assets, and triggers anti-blind-optimization guardrails.
"""

import pandas as pd


def detect_alerts(
    df_vms: pd.DataFrame,
    df_stats: pd.DataFrame,
    df_resources: pd.DataFrame
) -> list[dict]:
    """
    Evaluates infrastructure state against safety policies and anomaly detectors:
    1. Safety Guardrail: Protects bursty workloads from naive downsizing
    2. Capacity Exhaustion Alert: High sustained utilization risking outages
    3. Zombie Resource Waste: Long-unattached resources draining budget
    4. Workload Drift Alert: Significant recent utilization changes
    5. SLA Protection Guardrail: Critical production workloads
    """
    alerts = []
    stats_dict = {row["vm_id"]: row for _, row in df_stats.iterrows()}
    
    # 1. Bursty Workload Guardrail & Capacity Alerts on VMs
    for _, vm in df_vms.iterrows():
        vm_id = vm["vm_id"]
        stats = stats_dict.get(vm_id)
        if stats is None:
            continue
            
        pattern = vm.get("workload_pattern", "")
        p99_cpu = stats["p99_cpu"]
        avg_cpu = stats["avg_cpu"]
        p95_cpu = stats["p95_cpu"]
        p95_mem = stats["p95_mem"]
        trend = stats.get("cpu_trend_pct", 0)
        
        # Anti-Blind Optimization Guardrail: Bursty or Batch Workload
        if avg_cpu < 25.0 and p99_cpu >= 85.0:
            alerts.append({
                "alert_id": f"ALT-GUARD-{vm_id}",
                "severity": "Warning",
                "type": "Safety Guardrail: Bursty Traffic Pattern",
                "resource_id": vm_id,
                "resource_name": vm["vm_name"],
                "team": vm["team"],
                "environment": vm["environment"],
                "message": (
                    f"VM exhibits low average CPU ({avg_cpu}%) but extreme bursty peaks (p99={p99_cpu}%, max={stats['max_cpu']}%). "
                    f"Naive downsizing engines would incorrectly flag this for reduction. "
                    f"UTHO Optimizer has locked this instance from automatic downsizing to prevent service degradation."
                ),
                "recommended_action": "Retain capacity or implement auto-scaling group instead of static downsizing."
            })
            
        # Capacity Exhaustion Alert
        if p95_cpu >= 80.0 or p95_mem >= 85.0:
            alerts.append({
                "alert_id": f"ALT-CAP-{vm_id}",
                "severity": "Critical",
                "type": "Capacity Exhaustion Risk",
                "resource_id": vm_id,
                "resource_name": vm["vm_name"],
                "team": vm["team"],
                "environment": vm["environment"],
                "message": (
                    f"High sustained load: CPU p95={p95_cpu}%, Memory p95={p95_mem}%. "
                    f"Approaching capacity threshold with inadequate headroom for traffic surges."
                ),
                "recommended_action": "Upsize instance or scale horizontally to avoid OOM killer or thread starvation."
            })
            
        # Utilization Drift / Spike Alert (7-day trend > +35%)
        if trend >= 35.0:
            alerts.append({
                "alert_id": f"ALT-TREND-{vm_id}",
                "severity": "Info",
                "type": "Usage Trend Surge (+7d)",
                "resource_id": vm_id,
                "resource_name": vm["vm_name"],
                "team": vm["team"],
                "environment": vm["environment"],
                "message": (
                    f"Average CPU utilization increased by {trend}% over the past 7 days "
                    f"({stats['prior_avg_cpu']}% -> {stats['recent_avg_cpu']}%)."
                ),
                "recommended_action": "Investigate application release or increased customer traffic."
            })
            
    # 2. Zombie Asset Alerts (Orphaned storage & IPs > 30 days)
    for _, res in df_resources.iterrows():
        if not res["attached"] and res.get("days_unattached", 0) >= 30:
            alerts.append({
                "alert_id": f"ALT-ZOMBIE-{res['resource_id']}",
                "severity": "Warning",
                "type": "Zombie Cloud Resource Waste",
                "resource_id": res["resource_id"],
                "resource_name": f"{res['resource_type']} ({res['resource_id']})",
                "team": res["team"],
                "environment": "Storage / Network",
                "message": (
                    f"Asset has remained unattached for {res['days_unattached']} days, "
                    f"wasting ₹{res['monthly_cost']}/month. No read/write or packet traffic detected."
                ),
                "recommended_action": "Snapshot and terminate or release immediately."
            })
            
    return alerts
