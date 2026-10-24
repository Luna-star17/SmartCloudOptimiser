# SmartCloud Optimizer — Backend & Optimization Engine

> **A safety-aware, rule-based cloud cost optimization prototype built for college hackathon evaluation.**

SmartCloud Optimizer simulates enterprise cloud infrastructure (100 Virtual Machines, block storage volumes, and public IPs), generates 30 days of realistic 15-minute telemetry (288,000 metrics records), computes rigorous statistical percentiles (p95 and p99), and produces actionable cost-reduction recommendations backed by transparent operational safeguards.

---

## 1. Project Overview

Cloud infrastructure waste typically arises from three primary sources:
1. **Oversized Virtual Machines:** Allocating large VM instances whose compute and memory capacity are rarely needed.
2. **Idle Non-Production Workloads:** Leaving development and staging environments running 24/7 over nights and weekends.
3. **Orphaned Resources:** Retaining unattached block storage volumes and unassigned static public IP addresses.

### The Peak-Aware Problem (Why Averages Are Dangerous)
Naive cloud optimizers look only at **average CPU utilization**. If a VM has an average CPU of 18%, an aggressive tool might recommend downsizing it to a quarter of its size. However, if that VM handles a search index or payment gateway that **spikes to 95% CPU** during peak business hours, downsizing would cause catastrophic service outages.

**SmartCloud Optimizer solves this with dual-percentile capacity safeguards:**
- Evaluates **both average and peak demand** (p95 and p99 percentiles).
- Explicitly blocks downsizing on bursty workloads even when their average utilization is low.
- Protects mission-critical production systems from automated disruption.
- Flags data-quality issues (such as sparse or missing metrics) before acting.

---

## 2. Project File Structure

```text
SmartCloudOptimizer/
├── data_generator.py         # Generates reproducible inventory and 30-day 15-min metrics
├── optimizer.py              # Statistical analysis, decision rules, safeguards, and CSV exports
├── requirements.txt          # Minimal Python dependencies (pandas, numpy, pytest)
├── README.md                 # Complete documentation and build guide
├── data/                     # Generated CSV datasets (created automatically)
│   ├── inventory.csv         # Cloud resources inventory (100 VMs, volumes, IPs)
│   ├── metrics.csv           # 30-day telemetry (288,000 records)
│   ├── pricing_catalog.csv   # Transparent instance pricing reference
│   ├── resource_statistics.csv # Per-resource statistical summary table
│   └── recommendations.csv   # Actionable optimization recommendations
└── tests/
    ├── __init__.py
    └── test_optimizer.py     # 12 unit tests covering all edge cases and safeguards
```

---

## 3. Dataset Schemas

### A. Inventory (`data/inventory.csv`)
Contains at least 100 Virtual Machines plus unattached/attached block storage and public IPs.

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `resource_id` | String | Unique identifier | `vm-001`, `vol-001`, `ip-001` |
| `name` | String | Human-readable resource name | `payments-prod-vm-001` |
| `type` | String | Resource category | `VM`, `block_volume`, `public_ip` |
| `region` | String | Cloud datacenter location | `in-mumbai-01`, `in-delhi-01` |
| `environment` | String | Deployment stage | `prod`, `staging`, `dev` |
| `owner` | String | Responsible engineering team | `Payments`, `DevOps-Infra` |
| `vcpu` | Int / N/A | Allocated virtual CPUs | `4` (or `N/A` for storage/IP) |
| `ram_gb` | Int / N/A | Allocated memory in GB | `16` (or `N/A` for storage/IP) |
| `disk_size_gb`| Int / N/A | Allocated disk capacity | `100` |
| `creation_date` | Date | Creation date in UTC (ISO 8601) | `2026-08-15` |
| `status` | String | Current operational state | `running`, `available`, `unassigned` |
| `instance_type`| String | Instance size specification | `c-xlarge` (or `N/A`) |
| `monthly_cost`| Float | Simulated monthly cost (₹ INR) | `6200.00` |
| `attached_to` | String | Parent resource ID or attachment status | `self`, `vm-020`, `unattached` |
| `is_critical` | Boolean | Production mission-critical flag | `True` / `False` |
| `workload_pattern` | String | Archetype classification | `underutilized`, `bursty`, `idle_nonprod` |

### B. Historical Metrics (`data/metrics.csv`)
Contains 30 complete days of telemetry sampled every 15 minutes (96 intervals/day × 30 days = 2,880 observations per VM). For 100 VMs, this yields **288,000 records**.

| Field | Type | Unit / Range | Description |
| :--- | :--- | :--- | :--- |
| `timestamp` | Timestamp | ISO 8601 UTC | Observation date and time |
| `resource_id` | String | Matches inventory ID | Identifies measured VM |
| `cpu_percent` | Float | 0.0% – 100.0% | CPU utilization during interval |
| `memory_percent` | Float | 0.0% – 100.0% | Memory utilization during interval |
| `disk_used_percent` | Float | 0.0% – 100.0% | Percentage of storage occupied |
| `iops` | Integer | >= 0 | Disk input/output operations per second |
| `network_in_mb` | Float | >= 0.0 MB | Ingress network traffic in interval |
| `network_out_mb` | Float | >= 0.0 MB | Egress network traffic in interval |

---

## 4. Statistical Metrics: Understanding Average, p95, and p99

When `optimizer.py` analyzes telemetry grouped by `resource_id`, it calculates:

1. **Average ($\mu$):** The arithmetic mean of all readings across the 30-day window. While useful for long-term spend estimation, it smooths out all brief bursts.
2. **95th Percentile (p95):** The utilization value below which **95% of all observations fall**. Only 5% of readings are higher. p95 captures sustained heavy business hours while filtering out rare momentary blips.
3. **99th Percentile (p99):** The utilization value below which **99% of all observations fall**. It measures true peak operational capacity (the top 1% of heavy traffic bursts).
4. **Observation Count & Missing Percentage:** Detects whether a VM was newly created or experienced monitoring dropouts. If fewer than 192 observations are present (less than 2 days), the engine blocks automated recommendations and requests additional telemetry.

---

## 5. Optimization Rules and Operational Safeguards

The optimization engine uses transparent, auditable rules configured in `DEFAULT_CONFIG`:

```python
DEFAULT_CONFIG = {
    "min_observations": 192,           # Minimum 2 full days of 15-min intervals
    "max_missing_percent": 15.0,       # Flag data quality issues if missing > 15%
    
    # Downsizing thresholds
    "downsize_cpu_avg_max": 25.0,      # Average CPU must be <= 25%
    "downsize_cpu_p95_max": 45.0,      # p95 CPU must be <= 45%
    "downsize_cpu_p99_max": 60.0,      # p99 CPU must be <= 60%
    "downsize_mem_avg_max": 40.0,      # Average RAM must be <= 40%
    "downsize_mem_p95_max": 50.0,      # p95 RAM must be <= 50%
    
    # Capacity headroom required on the proposed smaller size
    "projected_p95_cpu_limit": 70.0,   # Projected p95 CPU on downsized VM must stay <= 70%
    "projected_p99_cpu_limit": 85.0,   # Projected p99 CPU on downsized VM must stay <= 85%
    
    # Idle non-production detection
    "idle_nonprod_cpu_avg_max": 8.0,
    "idle_nonprod_cpu_p95_max": 15.0,
    "idle_nonprod_iops_max": 50,
    "idle_schedule_savings_ratio": 0.65, # Shutting down nights & weekends saves ~65%
}
```

### Decision Logic Breakdown:
* **Rule A: Safe Rightsizing (Underutilized VMs):** If a VM's average, p95, and p99 CPU and memory are all comfortably low, the engine searches the pricing catalog for the most economical smaller instance where the workload's projected p95 CPU remains below 70% and p99 below 85%.
* **Rule B: Workload Scheduling (Idle Dev/Staging):** Non-production instances with near-zero activity outside office hours are recommended for automated nightly and weekend shutdowns (saving ~65% of monthly runtime cost).
* **Rule C: Bursty Workload Guardrail:** If average CPU is low ($\le 25\%$) but p95 or p99 exceeds safety thresholds ($> 45\%$ or $> 60\%$), **downsizing is strictly prohibited**. The action is marked `Retain current capacity (Do NOT downsize)` to prevent latency and throttling.
* **Rule D: Production Critical Safeguard:** Workloads flagged `is_critical` in production are protected from automated rightsizing; manual architectural verification is required.
* **Rule E: Data Quality Safeguard:** Resources with missing or incomplete metrics are labeled `Data Quality Alert` and marked `Needs review`.
* **Rule F: Orphaned Block Storage:** Unattached volumes are flagged for snapshot and deletion, reclaiming 100% of their monthly cost.
* **Rule G: Underutilized Storage:** Attached volumes with low occupancy (<15%) and zero IOPS are flagged for downsizing or archival investigation.
* **Rule H: Unassigned Public IPs:** Allocated but unassociated static IPs are recommended for immediate release.

---

## 6. Setup and Execution Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.14)
- Git

### Step-by-Step Commands

#### 1. Clone the Repository
```bash
git clone https://github.com/Luna-star17/SmartCloudOptimiser.git
cd SmartCloudOptimiser
```

#### 2. Create and Activate Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell execution policy prevents activation, use Command Prompt: `.\.venv\Scripts\activate.bat`)*

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Generate the Simulated Cloud Environment
```bash
python data_generator.py
```
*Outputs: `data/inventory.csv`, `data/metrics.csv` (288,000 records), and `data/pricing_catalog.csv` in ~2 seconds.*

#### 5. Run the Optimization Engine
```bash
python optimizer.py
```
*Computes metrics, evaluates safeguards, displays a summary table, and exports `data/recommendations.csv` and `data/resource_statistics.csv`.*

#### 6. Run the Automated Test Suite
```bash
python -m unittest tests/test_optimizer.py -v
```
*(Or if pytest is installed: `pytest tests/test_optimizer.py -v`)*

---

## 7. Recommendation Output Schema (`data/recommendations.csv`)

| Field | Type | Description |
| :--- | :--- | :--- |
| `resource_id` | String | Target resource identifier |
| `resource_name` | String | Readable resource name |
| `resource_type` | String | `VM`, `block_volume`, or `public_ip` |
| `environment` | String | `prod`, `staging`, or `dev` |
| `recommendation_type` | String | E.g. `Rightsizing`, `Workload Scheduling`, `Capacity Retention` |
| `suggested_action` | String | Actionable command or instruction |
| `reason` | String | Transparent evidence explaining why action is suggested |
| `average_cpu_percent` | Float | Observed average CPU % |
| `p95_cpu_percent` | Float | Observed 95th percentile CPU % |
| `p99_cpu_percent` | Float | Observed 99th percentile CPU % |
| `average_memory_percent` | Float | Observed average memory % |
| `p95_memory_percent` | Float | Observed 95th percentile memory % |
| `p99_memory_percent` | Float | Observed 99th percentile memory % |
| `risk_level` | String | `Low`, `Medium`, or `High` |
| `current_monthly_cost` | Float | Baseline cost in INR (₹) |
| `estimated_new_monthly_cost` | Float | Cost after recommendation applied |
| `estimated_monthly_savings` | Float | Monthly reduction (`current - new`) |
| `estimated_annual_savings` | Float | Annual reduction (`monthly * 12`) |
| `status` | String | `Pending`, `Accepted`, `Rejected`, `Needs review` |
| `assumptions` | String | Documented assumptions behind calculation |

---

## 8. Integration with the Streamlit Frontend

The backend cleanly exports CSVs and Python functions ready for direct import into the Streamlit dashboard:

```python
# In app.py (Streamlit frontend)
import pandas as pd
from optimizer import load_data, calculate_resource_statistics, generate_recommendations, calculate_savings

# 1. Load baseline data
df_inventory, df_metrics = load_data()

# 2. Run analysis
df_stats = calculate_resource_statistics(df_metrics, df_inventory)
df_recs = generate_recommendations(df_inventory, df_stats)

# 3. What-if simulation when user accepts/rejects recommendations
accepted_ids = st.session_state.get("accepted_recs", set())
accepted_recs = df_recs[df_recs["resource_id"].isin(accepted_ids)]
plan = calculate_savings(accepted_recs)

st.metric("Total Monthly Savings", f"₹{plan['total_monthly_savings']:,.2f}")
st.metric("Projected Monthly Spend", f"₹{plan['projected_monthly_spend']:,.2f}")
```

---

## 9. Assumptions and Limitations

1. **Simulated Telemetry:** The dataset is generated synthetically to reflect realistic workload archetypes (diurnal cycles, bursty traffic, idle staging). It does not capture live hypervisor metrics or hardware CPU cache effects.
2. **Pricing Model:** Prices are illustrative monthly approximations in INR (₹) based on standard cloud instance tiers, not a live billing API feed.
3. **Licensing & Reserved Instances:** The engine evaluates on-demand compute rightsizing. It does not model software licensing agreements (e.g. Windows Server, SQL Server) or multi-year reserved instance commitments.
4. **Human Verification:** All recommendations are designed as decision-support proposals (`Pending` or `Needs review`). Critical production infrastructure should always be validated by the system architect prior to modification.
