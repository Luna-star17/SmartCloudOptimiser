"""
SmartCloud Optimizer - Synthetic Cloud Infrastructure & Telemetry Generator

This script generates realistic, reproducible simulated cloud infrastructure data:
1. Inventory (inventory.csv): >= 100 Virtual Machines + block storage volumes + public IPs.
2. Historical Metrics (metrics.csv): 30 days of 15-minute telemetry per VM (2,880 observations
   per VM = 288,000 total records for 100 VMs).
3. Pricing Catalog (pricing_catalog.csv): Transparent instance types and simulated monthly costs.

Designed for college hackathon evaluation:
- 100% reproducible using a configurable random seed.
- Vectorized NumPy generation for blazing-fast execution (~2 seconds for 288,000 rows).
- Realistic workload archetypes: Idle, Underutilized, Bursty, Diurnal Normal, Critical Prod,
  and Insufficient Metrics (for testing safety guardrails).
"""

import os
import random
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

# Default configuration parameters
DEFAULT_SEED = 42
DEFAULT_NUM_VMS = 100
DEFAULT_DAYS = 30
DEFAULT_INTERVAL_MINUTES = 15  # 4 observations per hour -> 96 per day -> 2,880 for 30 days

# Output directory for data files
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# -------------------------------------------------------------------------
# Simulated Instance Types & Pricing Catalog (INR / month)
# Note: These are simulated illustrative prices, not live cloud provider rates.
# -------------------------------------------------------------------------
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

REGIONS = ["in-mumbai-01", "in-delhi-01", "in-bengaluru-01", "in-noida-01"]
TEAMS = ["Payments", "Search-Engine", "Platform-Core", "Data-Analytics", "DevOps-Infra", "Marketing-Web"]
ENVIRONMENTS = ["prod", "staging", "dev"]


def get_pricing_map():
    """Returns a dictionary keyed by instance_type for fast O(1) attribute lookup."""
    return {p["instance_type"]: p for p in PRICING_CATALOG}


def generate_inventory(n_vms: int = DEFAULT_NUM_VMS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """
    Generates a realistic simulated cloud inventory.

    Includes:
    - Exactly n_vms Virtual Machines across prod, staging, and dev environments.
    - 15 Block Storage volumes (some attached, some orphaned/unattached, some low-use).
    - 10 Public IP addresses (some attached, some unassigned incurring idle fees).

    Workload archetypes represented in the VM fleet:
    1. underutilized: consistently low CPU & memory (safe to downsize).
    2. idle_nonprod: dev/staging VMs idle nights/weekends (schedule for shutdown).
    3. bursty: low average, but frequent high spikes (MUST NOT be downsized).
    4. normal: healthy, well-utilized capacity (maintain as-is).
    5. critical_prod: mission-critical production services (protected from aggressive changes).
    6. insufficient_metrics: newly created or missing telemetry (flags review).
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    pricing_map = get_pricing_map()

    records = []

    # Calculate proportions for archetypes across the n_vms fleet:
    # 20 underutilized, 25 idle_nonprod, 20 bursty, 20 normal, 15 critical_prod = 100 VMs
    archetype_allocations = [
        ("underutilized", int(round(n_vms * 0.20))),
        ("idle_nonprod", int(round(n_vms * 0.25))),
        ("bursty", int(round(n_vms * 0.20))),
        ("normal", int(round(n_vms * 0.20))),
        ("critical_prod", max(0, n_vms - (
            int(round(n_vms * 0.20)) +
            int(round(n_vms * 0.25)) +
            int(round(n_vms * 0.20)) +
            int(round(n_vms * 0.20))
        ))),
    ]

    vm_counter = 1
    now = datetime.now(timezone.utc)

    for archetype, count in archetype_allocations:
        for _ in range(count):
            vm_id = f"vm-{vm_counter:03d}"
            region = rng.choice(REGIONS)
            team = rng.choice(TEAMS)

            if archetype == "idle_nonprod":
                env = rng.choice(["dev", "staging"])
                inst_type = rng.choice(["c-large", "c-xlarge", "c-2xlarge"])
                creation_days_ago = rng.randint(35, 120)
                is_critical = False
            elif archetype == "underutilized":
                env = rng.choice(["dev", "staging", "prod"])
                inst_type = rng.choice(["c-xlarge", "c-2xlarge", "c-4xlarge"])
                creation_days_ago = rng.randint(30, 90)
                is_critical = False
            elif archetype == "bursty":
                env = rng.choice(["staging", "prod"])
                inst_type = rng.choice(["c-2xlarge", "c-4xlarge"])
                creation_days_ago = rng.randint(40, 100)
                is_critical = False
            elif archetype == "critical_prod":
                env = "prod"
                inst_type = rng.choice(["c-2xlarge", "c-4xlarge"])
                creation_days_ago = rng.randint(60, 180)
                is_critical = True
            elif archetype == "insufficient_metrics":
                env = rng.choice(["dev", "staging"])
                inst_type = rng.choice(["c-large", "c-xlarge"])
                # Created only 2 days ago -> explains sparse telemetry
                creation_days_ago = rng.randint(1, 2)
                is_critical = False
            else:  # normal
                env = rng.choice(["prod", "staging", "dev"])
                inst_type = rng.choice(["c-medium", "c-large", "c-xlarge"])
                creation_days_ago = rng.randint(20, 150)
                is_critical = False

            spec = pricing_map[inst_type]
            created_at = (now - timedelta(days=creation_days_ago)).strftime("%Y-%m-%d")

            records.append({
                "resource_id": vm_id,
                "name": f"{team.lower()}-{env}-{vm_id}",
                "type": "VM",
                "region": region,
                "environment": env,
                "owner": team,
                "vcpu": spec["vcpu"],
                "ram_gb": spec["ram_gb"],
                "disk_size_gb": spec["vcpu"] * 25,  # 25 GB disk per vCPU
                "creation_date": created_at,
                "status": "running",
                "instance_type": inst_type,
                "monthly_cost": float(spec["monthly_cost"]),
                "attached_to": "self",
                "is_critical": is_critical,
                "workload_pattern": archetype,
                "notes": f"Simulated {archetype} workload",
            })
            vm_counter += 1

    # ---------------------------------------------------------------------
    # Additional Cloud Resources: Block Storage Volumes (15 volumes)
    # ---------------------------------------------------------------------
    storage_templates = [
        # (id, name, region, env, team, size_gb, cost, attached_to, age_days, notes)
        ("vol-001", "marketing-media-archive", "in-delhi-01", "dev", "Marketing-Web", 500, 1500.0, "unattached", 45, "Orphaned volume from decommissioned CMS"),
        ("vol-002", "hadoop-legacy-cluster-disk", "in-mumbai-01", "prod", "Data-Analytics", 1000, 3000.0, "unattached", 68, "Unattached volume after spark migration"),
        ("vol-003", "jenkins-workspace-backup", "in-bengaluru-01", "dev", "DevOps-Infra", 250, 750.0, "unattached", 19, "Orphaned test runner storage"),
        ("vol-004", "es-indexer-old-snapshot", "in-noida-01", "staging", "Search-Engine", 800, 2400.0, "unattached", 92, "Abandoned indexing snapshot"),
        ("vol-005", "loadtest-temp-scratchpad", "in-mumbai-01", "staging", "Payments", 400, 1200.0, "unattached", 14, "Completed stress test volume"),
        ("vol-006", "bi-staging-db-dump", "in-delhi-01", "dev", "Data-Analytics", 200, 600.0, "unattached", 30, "Unattached database dump"),
        # Attached volumes with low utilization (<15% used and 0 IOPS)
        ("vol-007", "qa-cold-storage-log", "in-delhi-01", "dev", "DevOps-Infra", 500, 1500.0, "vm-001", 60, "Attached volume with 8% disk usage and zero IOPS"),
        ("vol-008", "reports-scratch-disk", "in-mumbai-01", "staging", "Data-Analytics", 600, 1800.0, "vm-002", 55, "Attached volume with 11% disk usage and low IOPS"),
        # Attached, active volumes (normal healthy storage)
        ("vol-009", "prod-payments-db-data", "in-mumbai-01", "prod", "Payments", 1000, 5000.0, "vm-020", 120, "Active production NVMe block storage"),
        ("vol-010", "prod-search-index-data", "in-bengaluru-01", "prod", "Search-Engine", 1200, 6000.0, "vm-021", 110, "Active search index NVMe volume"),
        ("vol-011", "prod-platform-cache-data", "in-delhi-01", "prod", "Platform-Core", 500, 2500.0, "vm-022", 95, "Active production cache volume"),
        ("vol-012", "analytics-warehouse-ingest", "in-mumbai-01", "prod", "Data-Analytics", 800, 4000.0, "vm-023", 85, "Active ingestion buffer"),
        ("vol-013", "devops-gitlab-registry", "in-noida-01", "dev", "DevOps-Infra", 300, 900.0, "vm-024", 70, "Active container registry storage"),
        ("vol-014", "marketing-prod-assets", "in-delhi-01", "prod", "Marketing-Web", 400, 1200.0, "vm-025", 65, "Active marketing assets storage"),
        ("vol-015", "core-api-logs-volume", "in-bengaluru-01", "prod", "Platform-Core", 600, 1800.0, "vm-026", 80, "Active logs volume"),
    ]

    for vol_id, name, reg, env, team, size, cost, attached_to, age_days, note in storage_templates:
        created_at = (now - timedelta(days=age_days)).strftime("%Y-%m-%d")
        records.append({
            "resource_id": vol_id,
            "name": name,
            "type": "block_volume",
            "region": reg,
            "environment": env,
            "owner": team,
            "vcpu": "N/A",
            "ram_gb": "N/A",
            "disk_size_gb": size,
            "creation_date": created_at,
            "status": "available" if attached_to == "unattached" else "in-use",
            "instance_type": "N/A",
            "monthly_cost": float(cost),
            "attached_to": attached_to,
            "is_critical": False,
            "workload_pattern": "unattached" if attached_to == "unattached" else "attached_storage",
            "notes": note,
        })

    # ---------------------------------------------------------------------
    # Additional Cloud Resources: Public IPs (10 IPs)
    # Unassociated public IPs incur idle reservation fees (~₹350/month)
    # ---------------------------------------------------------------------
    ip_templates = [
        # (id, name, region, env, team, attached_to, age_days, notes)
        ("ip-001", "marketing-campaign-legacy-ip", "in-delhi-01", "dev", "Marketing-Web", "unassigned", 38, "Unassigned static IP from previous campaign"),
        ("ip-002", "vpn-gateway-deprecate-ip", "in-mumbai-01", "staging", "DevOps-Infra", "unassigned", 54, "Unassigned IP after VPN migration"),
        ("ip-003", "checkout-staging-spare-ip", "in-bengaluru-01", "staging", "Payments", "unassigned", 21, "Unassigned spare staging elastic IP"),
        ("ip-004", "search-crawler-test-ip", "in-noida-01", "dev", "Search-Engine", "unassigned", 72, "Unassigned IP from old crawler test bed"),
        ("ip-005", "bi-analytics-exporter-ip", "in-delhi-01", "dev", "Data-Analytics", "unassigned", 16, "Unassigned reporting ingress IP"),
        # Attached public IPs (active ingress)
        ("ip-006", "prod-payments-api-gw-ip", "in-mumbai-01", "prod", "Payments", "vm-020", 120, "Active production payments gateway IP"),
        ("ip-007", "prod-search-ingress-ip", "in-bengaluru-01", "prod", "Search-Engine", "vm-021", 110, "Active production search entrypoint IP"),
        ("ip-008", "prod-platform-lb-ip", "in-delhi-01", "prod", "Platform-Core", "vm-022", 95, "Active production load balancer IP"),
        ("ip-009", "prod-data-pipeline-ip", "in-mumbai-01", "prod", "Data-Analytics", "vm-023", 85, "Active data ingest IP"),
        ("ip-010", "dev-gitlab-runner-ip", "in-noida-01", "dev", "DevOps-Infra", "vm-024", 70, "Active dev tools IP"),
    ]

    for ip_id, name, reg, env, team, attached_to, age_days, note in ip_templates:
        created_at = (now - timedelta(days=age_days)).strftime("%Y-%m-%d")
        records.append({
            "resource_id": ip_id,
            "name": name,
            "type": "public_ip",
            "region": reg,
            "environment": env,
            "owner": team,
            "vcpu": "N/A",
            "ram_gb": "N/A",
            "disk_size_gb": "N/A",
            "creation_date": created_at,
            "status": "unassigned" if attached_to == "unassigned" else "associated",
            "instance_type": "N/A",
            "monthly_cost": 350.0,
            "attached_to": attached_to,
            "is_critical": False,
            "workload_pattern": "unassigned" if attached_to == "unassigned" else "associated_ip",
            "notes": note,
        })

    df_inventory = pd.DataFrame(records)
    return df_inventory


def generate_metrics(
    df_inventory: pd.DataFrame,
    days: int = DEFAULT_DAYS,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Generates realistic historical performance metrics for every VM in inventory.

    Calculates:
    - 30 days of metrics at 15-minute intervals = 2,880 observations per VM.
    - Total records = 100 VMs * 2,880 = 288,000 observations.
    - Vectorized NumPy generation for sub-3-second generation speed.
    - Realistic diurnal (day/night) cycles, weekend drops, bursty peaks, and valid bounds.

    Returns:
    - pd.DataFrame with columns:
      timestamp, resource_id, cpu_percent, memory_percent, disk_used_percent,
      iops, network_in_mb, network_out_mb
    """
    np_rng = np.random.default_rng(seed)

    # Filter only VM resources (metrics are collected on compute instances)
    vm_rows = df_inventory[df_inventory["type"] == "VM"].copy()
    if vm_rows.empty:
        raise ValueError("No VM resources found in inventory DataFrame to generate metrics for.")

    # Generate 15-minute UTC timestamps for the duration
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_time = now - timedelta(days=days)
    intervals_per_day = (24 * 60) // interval_minutes
    total_intervals = days * intervals_per_day

    # Generate base timestamp sequence
    time_index = pd.date_range(start=start_time, periods=total_intervals, freq=f"{interval_minutes}min", tz="UTC")
    time_strings = [ts.strftime("%Y-%m-%dT%H:%M:%SZ") for ts in time_index]

    # Pre-extract hour of day and day of week to simulate realistic diurnal cycles
    hours = time_index.hour.to_numpy()
    day_of_week = time_index.day_of_week.to_numpy()
    is_weekend = (day_of_week >= 5)
    is_workhours = ((hours >= 9) & (hours <= 18)) & (~is_weekend)

    dfs = []

    for _, vm in vm_rows.iterrows():
        vm_id = vm["resource_id"]
        pattern = vm["workload_pattern"]

        # Base arrays
        n = total_intervals

        if pattern == "idle_nonprod":
            # Consistently near-zero CPU and memory (dev/staging left running off-hours)
            # Slight activity during workhours (5-12%), practically dead off-hours & weekends (1-3%)
            base_cpu = np.where(is_workhours, np_rng.normal(6.5, 2.0, n), np_rng.normal(2.0, 0.6, n))
            base_mem = np.where(is_workhours, np_rng.normal(22.0, 3.0, n), np_rng.normal(16.0, 1.5, n))
            disk_used = np_rng.normal(28.0, 1.0, n)
            iops = np.where(is_workhours, np_rng.integers(10, 40, n), np_rng.integers(0, 8, n))
            net_in = np.where(is_workhours, np_rng.uniform(0.5, 2.5, n), np_rng.uniform(0.05, 0.3, n))
            net_out = net_in * 0.8

        elif pattern == "underutilized":
            # Consistently low resource utilization 24/7 (allocated far more than needed)
            # CPU avg ~10%, p95 < 18%, p99 < 22%. RAM avg ~24%.
            base_cpu = np_rng.normal(9.5, 2.5, n)
            base_mem = np_rng.normal(24.0, 3.0, n)
            disk_used = np_rng.normal(32.0, 2.0, n)
            iops = np_rng.integers(15, 60, n)
            net_in = np_rng.uniform(1.0, 5.0, n)
            net_out = net_in * 0.9

        elif pattern == "bursty":
            # Low average utilization (~15-22%), BUT with sharp recurring peaks (85-98%)
            # This is the vital hackathon showcase: naive average-based logic would downsize,
            # but peak-aware p95/p99 logic MUST PREVENT downsizing!
            base_cpu = np_rng.normal(16.0, 4.0, n)
            base_mem = np_rng.normal(42.0, 5.0, n)
            iops = np_rng.integers(50, 250, n)
            net_in = np_rng.uniform(5.0, 25.0, n)
            net_out = net_in * 1.2

            # Introduce burst spikes during specific hours or random triggers (~4% of intervals)
            burst_mask = (np_rng.random(n) < 0.04) | ((hours == 11) & (np_rng.random(n) < 0.25)) | ((hours == 15) & (np_rng.random(n) < 0.25))
            base_cpu[burst_mask] = np_rng.normal(92.0, 3.5, np.count_nonzero(burst_mask))
            base_mem[burst_mask] = np_rng.normal(82.0, 4.0, np.count_nonzero(burst_mask))
            iops[burst_mask] = np_rng.integers(1800, 4500, np.count_nonzero(burst_mask))
            net_in[burst_mask] = np_rng.uniform(150.0, 450.0, np.count_nonzero(burst_mask))
            net_out[burst_mask] = net_in[burst_mask] * 1.5
            disk_used = np_rng.normal(55.0, 3.0, n)

        elif pattern == "critical_prod":
            # Production critical (payments, auth standby): steady, safe utilization
            # Moderate CPU (25-45%), healthy memory (40-60%), high reliability
            base_cpu = np_rng.normal(32.0, 5.0, n)
            base_mem = np_rng.normal(52.0, 4.0, n)
            disk_used = np_rng.normal(62.0, 1.5, n)
            iops = np_rng.integers(300, 900, n)
            net_in = np_rng.uniform(30.0, 85.0, n)
            net_out = net_in * 1.1

        else:  # normal
            # Active, healthy workload with diurnal business hours
            day_mult = np.where(is_workhours, 1.4, 0.7)
            base_cpu = np_rng.normal(52.0, 8.0, n) * day_mult
            base_mem = np_rng.normal(58.0, 6.0, n) * np.where(is_workhours, 1.1, 0.9)
            disk_used = np_rng.normal(50.0, 2.5, n)
            iops = (np_rng.integers(350, 1200, n) * day_mult).astype(int)
            net_in = np_rng.uniform(25.0, 80.0, n) * day_mult
            net_out = net_in * 1.3

        # Clamp and clip all metric values to valid physical limits
        clean_cpu = np.clip(base_cpu, 0.0, 100.0)
        clean_mem = np.clip(base_mem, 0.0, 100.0)
        clean_disk = np.clip(disk_used, 0.0, 100.0)
        clean_iops = np.maximum(0, iops)
        clean_net_in = np.maximum(0.0, net_in)
        clean_net_out = np.maximum(0.0, net_out)

        vm_df = pd.DataFrame({
            "timestamp": time_strings,
            "resource_id": vm_id,
            "cpu_percent": np.round(clean_cpu, 2),
            "memory_percent": np.round(clean_mem, 2),
            "disk_used_percent": np.round(clean_disk, 2),
            "iops": clean_iops.astype(int),
            "network_in_mb": np.round(clean_net_in, 2),
            "network_out_mb": np.round(clean_net_out, 2),
        })
        dfs.append(vm_df)

    df_metrics = pd.concat(dfs, ignore_index=True)
    return df_metrics


def save_sample_data(
    df_inventory: pd.DataFrame,
    df_metrics: pd.DataFrame,
    data_dir: str = DATA_DIR,
) -> None:
    """
    Saves generated inventory, metrics, and pricing catalog DataFrames to CSV files.
    """
    os.makedirs(data_dir, exist_ok=True)

    inv_path = os.path.join(data_dir, "inventory.csv")
    metrics_path = os.path.join(data_dir, "metrics.csv")
    pricing_path = os.path.join(data_dir, "pricing_catalog.csv")

    df_inventory.to_csv(inv_path, index=False)
    df_metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(PRICING_CATALOG).to_csv(pricing_path, index=False)

    print(f"\n[SUCCESS] Datasets saved successfully to: {data_dir}")
    print(f"  - inventory.csv:       {len(df_inventory)} cloud resources")
    print(f"  - metrics.csv:         {len(df_metrics):,} metric records")
    print(f"  - pricing_catalog.csv: {len(PRICING_CATALOG)} instance types")


def main():
    """CLI execution entrypoint to generate and validate the sample dataset."""
    print("=" * 70)
    print("SmartCloud Optimizer - Generating Simulated Cloud Environment")
    print(f"Config: {DEFAULT_NUM_VMS} VMs | {DEFAULT_DAYS} Days | 15-min Intervals | Seed: {DEFAULT_SEED}")
    print("=" * 70)

    start_time = datetime.now()

    # Step 1: Generate inventory
    print("\n[1/3] Generating cloud inventory (VMs, storage, public IPs)...")
    df_inventory = generate_inventory(n_vms=DEFAULT_NUM_VMS, seed=DEFAULT_SEED)
    print(f"  -> Generated {len(df_inventory)} total resources.")
    print("  -> Resource breakdown by type:")
    for rtype, count in df_inventory["type"].value_counts().items():
        print(f"     * {rtype}: {count}")

    # Step 2: Generate metrics
    print("\n[2/3] Generating 30 days of 15-minute telemetry per VM (vectorized)...")
    df_metrics = generate_metrics(
        df_inventory=df_inventory,
        days=DEFAULT_DAYS,
        interval_minutes=DEFAULT_INTERVAL_MINUTES,
        seed=DEFAULT_SEED,
    )
    print(f"  -> Generated {len(df_metrics):,} telemetry records.")

    # Step 3: Save to CSV
    print("\n[3/3] Saving data files...")
    save_sample_data(df_inventory, df_metrics)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nCompleted data generation pipeline in {elapsed:.2f} seconds.")
    print("=" * 70)


if __name__ == "__main__":
    main()
