"""
UTHO SmartCloud Optimizer - Synthetic Data Generator
Generates realistic, repeatable 30-day telemetry and cloud resource inventory
highlighting distinct workload patterns for safe rightsizing and scheduling.
"""

import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Set deterministic random seed for repeatable demo
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Realistic UTHO Cloud Pricing Catalog (INR / month)
PRICING_CATALOG = [
    {"instance_type": "u1.nano", "vcpu": 1, "memory_gb": 1, "monthly_cost": 350},
    {"instance_type": "u1.micro", "vcpu": 1, "memory_gb": 2, "monthly_cost": 650},
    {"instance_type": "u1.small", "vcpu": 2, "memory_gb": 4, "monthly_cost": 1300},
    {"instance_type": "u1.medium", "vcpu": 2, "memory_gb": 8, "monthly_cost": 2400},
    {"instance_type": "u1.large", "vcpu": 4, "memory_gb": 8, "monthly_cost": 3600},
    {"instance_type": "u1.xlarge", "vcpu": 4, "memory_gb": 16, "monthly_cost": 6200},
    {"instance_type": "u1.2xlarge", "vcpu": 8, "memory_gb": 32, "monthly_cost": 12000},
    {"instance_type": "u1.4xlarge", "vcpu": 16, "memory_gb": 64, "monthly_cost": 23500},
    {"instance_type": "u1.8xlarge", "vcpu": 32, "memory_gb": 128, "monthly_cost": 46000},
    # Compute optimized
    {"instance_type": "c1.large", "vcpu": 4, "memory_gb": 4, "monthly_cost": 3100},
    {"instance_type": "c1.xlarge", "vcpu": 8, "memory_gb": 8, "monthly_cost": 5900},
    # Memory optimized
    {"instance_type": "m1.large", "vcpu": 2, "memory_gb": 16, "monthly_cost": 4200},
    {"instance_type": "m1.xlarge", "vcpu": 4, "memory_gb": 32, "monthly_cost": 8100},
]

REGIONS = ["in-delhi-01", "in-mumbai-01", "in-bengaluru-01", "in-noida-01"]
TEAMS = ["Checkout-Core", "Search-Engine", "Data-Platform", "DevOps-Infra", "Marketing-Web", "Analytics-BI"]
ENVIRONMENTS = ["Production", "Staging", "Development"]


def generate_pricing_csv():
    df_pricing = pd.DataFrame(PRICING_CATALOG)
    path = os.path.join(DATA_DIR, "pricing.csv")
    df_pricing.to_csv(path, index=False)
    print(f"Generated {path} with {len(df_pricing)} instance types.")
    return df_pricing


def generate_vms_csv():
    """
    Generates 50 distinct VMs with structured workload classifications:
    - Underutilized (ideal for safe downsizing)
    - Bursty (low average, but sharp p99/peak spikes: naive optimizer would downsize, safety-engine BLOCKS)
    - Weekend Idle (Dev/Staging servers idle outside office hours: perfect for scheduling)
    - Batch Workload (Quiet during day, heavy crunch at night)
    - Critical (Core production components tagged critical, protected from aggressive downsizing)
    - Normal (Healthy, optimal utilization)
    """
    vms = []
    
    # Pre-defined archetypes to tell clear hackathon stories
    archetypes = [
        # (pattern, env, team, count, instance_type, critical, schedule)
        ("Underutilized", "Development", "Marketing-Web", 6, "u1.xlarge", False, "Unscheduled"),
        ("Underutilized", "Staging", "DevOps-Infra", 5, "u1.2xlarge", False, "Unscheduled"),
        ("Weekend Idle", "Development", "Checkout-Core", 7, "u1.large", False, "Always-On"),
        ("Weekend Idle", "Staging", "Search-Engine", 5, "u1.xlarge", False, "Always-On"),
        ("Bursty", "Production", "Checkout-Core", 5, "u1.2xlarge", False, "Always-On"),
        ("Bursty", "Production", "Search-Engine", 4, "u1.4xlarge", False, "Always-On"),
        ("Batch", "Production", "Data-Platform", 4, "u1.4xlarge", False, "Batch-Nightly"),
        ("Critical", "Production", "Checkout-Core", 4, "u1.2xlarge", True, "Always-On"),
        ("Critical", "Production", "DevOps-Infra", 3, "u1.large", True, "Always-On"),
        ("Normal", "Production", "Analytics-BI", 4, "u1.large", False, "Always-On"),
        ("Normal", "Production", "Data-Platform", 3, "u1.xlarge", False, "Always-On"),
    ]
    
    vm_counter = 1
    pricing_map = {item["instance_type"]: item for item in PRICING_CATALOG}
    
    for pattern, env, team, count, inst_type, is_crit, sched in archetypes:
        for _ in range(count):
            vm_id = f"VM-{vm_counter:03d}"
            inst = pricing_map[inst_type]
            region = random.choice(REGIONS)
            
            vms.append({
                "vm_id": vm_id,
                "vm_name": f"{team.lower()}-{env.lower()}-{vm_id.lower()}",
                "region": region,
                "team": team,
                "environment": env,
                "instance_type": inst_type,
                "vcpu": inst["vcpu"],
                "memory_gb": inst["memory_gb"],
                "monthly_cost": inst["monthly_cost"],
                "critical": is_crit,
                "schedule": sched,
                "workload_pattern": pattern
            })
            vm_counter += 1
            
    df_vms = pd.DataFrame(vms)
    path = os.path.join(DATA_DIR, "vms.csv")
    df_vms.to_csv(path, index=False)
    print(f"Generated {path} with {len(df_vms)} VMs.")
    return df_vms


def generate_metrics_csv(df_vms):
    """
    Generates 30 days of hourly metrics (720 intervals per VM).
    Total rows = 50 * 720 = 36,000 rows.
    Calculates realistic CPU, Memory, Disk, IOPS, and Network.
    """
    end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(days=30)
    timestamps = [start_time + timedelta(hours=i) for i in range(24 * 30)]
    
    records = []
    
    for _, vm in df_vms.iterrows():
        vm_id = vm["vm_id"]
        pattern = vm["workload_pattern"]
        
        # Base parameters by pattern
        for ts in timestamps:
            hour = ts.hour
            is_weekend = ts.weekday() >= 5  # Saturday or Sunday
            is_workhour = (9 <= hour <= 19) and not is_weekend
            
            if pattern == "Underutilized":
                # Consistently low CPU and RAM
                cpu = np.clip(np.random.normal(9.0, 2.5), 2.0, 18.0)
                mem = np.clip(np.random.normal(20.0, 3.0), 10.0, 32.0)
                iops = random.randint(10, 80)
                net = round(np.random.uniform(0.5, 4.0), 2)
                
            elif pattern == "Bursty":
                # Typically low (12-25%), but sharp sudden spikes (85-98%)
                is_spike = (random.random() < 0.04) or (hour in [11, 15] and random.random() < 0.3)
                if is_spike:
                    cpu = np.clip(np.random.normal(91.0, 4.0), 82.0, 99.5)
                    mem = np.clip(np.random.normal(78.0, 5.0), 65.0, 92.0)
                    iops = random.randint(1500, 4500)
                    net = round(np.random.uniform(80.0, 350.0), 2)
                else:
                    cpu = np.clip(np.random.normal(16.0, 4.0), 6.0, 32.0)
                    mem = np.clip(np.random.normal(42.0, 4.0), 30.0, 55.0)
                    iops = random.randint(80, 400)
                    net = round(np.random.uniform(5.0, 25.0), 2)
                    
            elif pattern == "Weekend Idle":
                # High/moderate during work hours, almost completely idle at night and weekend
                if is_workhour:
                    cpu = np.clip(np.random.normal(48.0, 8.0), 25.0, 78.0)
                    mem = np.clip(np.random.normal(55.0, 6.0), 38.0, 72.0)
                    iops = random.randint(300, 1100)
                    net = round(np.random.uniform(20.0, 65.0), 2)
                else:
                    # Idle overnight & weekends
                    cpu = np.clip(np.random.normal(2.5, 0.8), 0.5, 6.0)
                    mem = np.clip(np.random.normal(18.0, 2.0), 12.0, 25.0)
                    iops = random.randint(5, 30)
                    net = round(np.random.uniform(0.1, 1.5), 2)
                    
            elif pattern == "Batch":
                # Quiet during the day, massive computation crunch 01:00 - 05:00 AM
                if 1 <= hour <= 5:
                    cpu = np.clip(np.random.normal(88.0, 5.0), 75.0, 98.0)
                    mem = np.clip(np.random.normal(82.0, 4.0), 70.0, 94.0)
                    iops = random.randint(2200, 5000)
                    net = round(np.random.uniform(120.0, 450.0), 2)
                else:
                    cpu = np.clip(np.random.normal(7.0, 2.0), 1.5, 14.0)
                    mem = np.clip(np.random.normal(28.0, 3.0), 18.0, 38.0)
                    iops = random.randint(40, 150)
                    net = round(np.random.uniform(1.0, 8.0), 2)
                    
            elif pattern == "Critical":
                # Steady, moderate to low usage, but must be kept safe (Auth, Payment Standby)
                cpu = np.clip(np.random.normal(22.0, 4.0), 10.0, 42.0)
                mem = np.clip(np.random.normal(38.0, 3.5), 26.0, 52.0)
                iops = random.randint(150, 600)
                net = round(np.random.uniform(10.0, 45.0), 2)
                
            else:  # Normal
                cpu = np.clip(np.random.normal(52.0, 9.0), 28.0, 78.0)
                mem = np.clip(np.random.normal(58.0, 6.0), 40.0, 76.0)
                iops = random.randint(400, 1400)
                net = round(np.random.uniform(30.0, 90.0), 2)
                
            disk_used_pct = round(np.clip(45.0 + (hash(vm_id) % 35), 25.0, 88.0), 1)
            
            records.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "vm_id": vm_id,
                "cpu_pct": round(float(cpu), 2),
                "memory_pct": round(float(mem), 2),
                "disk_used_pct": disk_used_pct,
                "disk_iops": int(iops),
                "network_mbps": float(net)
            })
            
    df_metrics = pd.DataFrame(records)
    path = os.path.join(DATA_DIR, "metrics.csv")
    df_metrics.to_csv(path, index=False)
    print(f"Generated {path} with {len(df_metrics)} metric records.")
    return df_metrics


def generate_resources_csv():
    """
    Generates unattached storage volumes and unassociated Elastic Static IPs.
    Allows testing orphan resource detection.
    """
    resources = []
    
    # 8 Unattached storage volumes
    vol_configs = [
        ("vol-0091", "Block-Storage-SSD", "in-delhi-01", 500, 1800, False, 42, "Marketing-Web", "Old website media backup"),
        ("vol-0142", "Block-Storage-NVMe", "in-mumbai-01", 1000, 4200, False, 65, "Data-Platform", "Legacy Hadoop cluster disk"),
        ("vol-0288", "Block-Storage-SSD", "in-bengaluru-01", 200, 800, False, 18, "DevOps-Infra", "Decommissioned Jenkins workspace"),
        ("vol-0315", "Block-Storage-Standard", "in-noida-01", 800, 1600, False, 90, "Search-Engine", "Historical log archive"),
        ("vol-0402", "Block-Storage-SSD", "in-mumbai-01", 300, 1200, False, 12, "Checkout-Core", "Load test temp disk"),
        ("vol-0511", "Block-Storage-SSD", "in-delhi-01", 150, 600, False, 28, "Analytics-BI", "Staging DB dump"),
        # Attached volumes (normal baseline)
        ("vol-0601", "Block-Storage-NVMe", "in-delhi-01", 500, 2100, True, 0, "Checkout-Core", "Active production DB"),
        ("vol-0602", "Block-Storage-SSD", "in-bengaluru-01", 250, 1000, True, 0, "Search-Engine", "Active search cache"),
    ]
    
    for r_id, r_type, reg, size, cost, attached, days_unatt, team, note in vol_configs:
        resources.append({
            "resource_id": r_id,
            "resource_type": r_type,
            "region": reg,
            "size_gb": size,
            "monthly_cost": cost,
            "attached": attached,
            "days_unattached": days_unatt,
            "critical": False,
            "team": team,
            "notes": note
        })
        
    # 6 Elastic Static IPs (4 unassociated, costing idle fees)
    ip_configs = [
        ("ip-203-0-113-45", "Elastic-Static-IP", "in-delhi-01", 0, 350, False, 34, "Marketing-Web", "Unassigned legacy marketing campaign IP"),
        ("ip-198-51-100-12", "Elastic-Static-IP", "in-mumbai-01", 0, 350, False, 52, "DevOps-Infra", "Former VPN gateway IP"),
        ("ip-203-0-113-98", "Elastic-Static-IP", "in-bengaluru-01", 0, 350, False, 19, "Checkout-Core", "Old checkout microservice IP"),
        ("ip-198-51-100-88", "Elastic-Static-IP", "in-noida-01", 0, 350, False, 75, "Search-Engine", "Deprecated crawlers pool IP"),
        ("ip-203-0-113-04", "Elastic-Static-IP", "in-delhi-01", 0, 350, True, 0, "Checkout-Core", "Active API Gateway IP"),
        ("ip-198-51-100-01", "Elastic-Static-IP", "in-mumbai-01", 0, 350, True, 0, "Search-Engine", "Active Ingress LB IP"),
    ]
    
    for r_id, r_type, reg, size, cost, attached, days_unatt, team, note in ip_configs:
        resources.append({
            "resource_id": r_id,
            "resource_type": r_type,
            "region": reg,
            "size_gb": size,
            "monthly_cost": cost,
            "attached": attached,
            "days_unattached": days_unatt,
            "critical": False,
            "team": team,
            "notes": note
        })
        
    df_res = pd.DataFrame(resources)
    path = os.path.join(DATA_DIR, "resources.csv")
    df_res.to_csv(path, index=False)
    print(f"Generated {path} with {len(df_res)} cloud resources.")
    return df_res


def generate_all():
    print("Generating comprehensive UTHO cloud simulation dataset...")
    df_pricing = generate_pricing_csv()
    df_vms = generate_vms_csv()
    df_metrics = generate_metrics_csv(df_vms)
    df_resources = generate_resources_csv()
    print("Dataset generation complete!")
    return df_pricing, df_vms, df_metrics, df_resources


if __name__ == "__main__":
    generate_all()
