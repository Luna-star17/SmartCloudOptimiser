# SmartCloud Optimizer ☁️

A college hackathon cloud FinOps and cost optimization dashboard built with **Python**, **Streamlit**, **Pandas**, and **Plotly**.

**SmartCloud Optimizer** helps organizations identify wasted cloud spend and delivers evidence-based, safety-aware optimization recommendations (rightsizing, workload scheduling, and orphaned resource cleanup) without compromising application performance.

---

## 🌟 Key Features

1. **Executive FinOps Overview**:
   - Real-time summary metric cards: Total Resources, Virtual Machines, Flagged Candidates, Potential Monthly Savings, High-Risk Safeguards, and Baseline Cloud Spend.
   - Interactive breakdown charts showing potential savings across environments, teams, and optimization categories.
   - Active safety guardrails and anomaly feed protecting bursty and critical workloads.

2. **Cloud Resource Inventory**:
   - Filterable and searchable inventory table covering VMs, unattached Block Storage volumes, and unassociated Elastic Static IPs.
   - Dynamic sidebar filtering by **Region**, **Environment** (*Production*, *Staging*, *Development*), **Team/Owner**, and **Resource Type**.
   - Clear display of missing values as `N/A` for non-applicable specs (e.g., vCPU and RAM for block volumes).
   - Direct deep-link button to jump straight from any resource into detailed telemetry analysis.
   - CSV export of filtered inventory records.

3. **Historical Performance Telemetry & Peak-Aware Safeguards**:
   - 30-day time-series telemetry charts for CPU (%), Memory (%), Disk IOPS, Disk Usage, and Network Throughput (Mbps).
   - **Average, p95 (95th percentile), p99 (99th percentile), and Max** calculations with beginner-friendly explanations.
   - **Anti-Blind-Optimization Guardrail**: Highlights bursty or batch workloads where average usage is low but p99 peaks are high, preventing naive downsizing that could cause service outages.
   - Workhour vs. off-hours heatmap showing weekday vs. weekend patterns to validate scheduled shutdowns.

4. **Evidence-Based Optimization Recommendations**:
   - Transparent cards showing Current Spec vs. Proposed Spec, Monthly Savings, and Confidence Level.
   - Detailed audit reasoning and telemetry evidence (e.g., observed p95/p99 vs. projected candidate headroom).
   - Clear risk badges: **Low Risk**, **Medium Risk**, **High Risk**, and **Safety Guardrails**.
   - Functional **Accept** and **Reject** interactive controls:
     - Accepting updates status to *Accepted* immediately in session state.
     - Rejecting prompts for a structured justification (e.g., upcoming seasonal peak, vendor requirement, critical SLA).
     - Full audit trail preserved across dashboard reruns.

5. **Decision History & What-If Financial Simulation**:
   - Non-destructive simulation engine modeling projected monthly cloud spend based on approved actions.
   - **Compounded Savings Logic**: Automatically handles when both Rightsizing and Workload Scheduling are approved for the same VM, downsizes first and then schedules the downsized instance to **prevent double-counting**.
   - Approved actions export to CSV for DevOps/FinOps approval workflows.

---

## 📂 Project Structure

```text
SmartCloudOptimizer/
├── app.py                  # Main Streamlit frontend dashboard
├── requirements.txt        # Python dependency definitions
├── README.md               # Documentation and setup instructions
├── optimizer.py            # Backend recommendation and rightsizing engine
├── simulator.py            # What-if simulation and conflict prevention engine
├── alerts.py               # Anomaly detection and safety guardrails engine
├── data_generator.py       # Deterministic 30-day synthetic telemetry generator
├── test_engine.py          # Backend engine verification script
└── data/
    ├── inventory.csv       # Unified cloud inventory catalog
    ├── metrics.csv         # 30-day historical telemetry time-series
    ├── vms.csv             # Virtual machine configurations and archetypes
    ├── resources.csv       # Unattached block storage and static elastic IPs
    └── pricing.csv         # Cloud instance pricing catalog (UTHO Cloud)
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10+ (Python 3.10, 3.11, or 3.12 recommended)
- Git

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
python -m pip install -r requirements.txt
```

### 4. (Optional) Regenerate Datasets
If the `data/` folder is empty or you wish to generate a fresh dataset:
```bash
python data_generator.py
```

### 5. Launch the Dashboard
```bash
python -m streamlit run app.py
```
Streamlit will launch locally, typically at `http://localhost:8501`.

---

## 📊 Expected Data Formats & Column Schemas

### 1. Cloud Inventory (`data/inventory.csv`)
| Column | Description | Example |
| :--- | :--- | :--- |
| `resource_id` | Unique resource identifier | `VM-014`, `vol-0091` |
| `name` | Resource descriptive name or tag | `checkout-core-dev-vm-014` |
| `type` | Cloud asset category | `VM (u1.large)`, `Block-Storage-SSD` |
| `region` | Cloud region / datacenter | `in-delhi-01`, `in-mumbai-01` |
| `environment` | Deployment lifecycle | `Production`, `Staging`, `Development` |
| `owner` | Team or department owner | `Checkout-Core`, `DevOps-Infra` |
| `vcpu` | Virtual CPU cores (N/A for storage/IP) | `4.0` or empty |
| `ram_gb` | Memory in gigabytes (N/A for storage/IP) | `8.0` or empty |
| `disk_gb` | Primary attached storage (GB) | `100.0`, `500.0` |
| `creation_date` | Provisioning timestamp | `2026-08-15` |
| `monthly_cost` | Estimated monthly expenditure (INR ₹) | `3600.0` |

### 2. Historical Telemetry Metrics (`data/metrics.csv`)
Contains 30 days of hourly readings (720 data points per instance):
| Column | Description | Example |
| :--- | :--- | :--- |
| `timestamp` | Timestamp of telemetry reading | `2026-09-24 09:00:00` |
| `vm_id` | Target compute instance ID | `VM-001` |
| `cpu_pct` | CPU utilization percentage (0 - 100%) | `14.5` |
| `memory_pct` | RAM utilization percentage (0 - 100%) | `32.1` |
| `disk_used_pct` | Disk storage capacity used (%) | `50.0` |
| `disk_iops` | Disk Input/Output Operations per Second | `450` |
| `network_mbps` | Total network throughput in Megabits/sec | `18.4` |

---

## 🤝 Team Roles & Backend Integration

This project is organized across two team members:
- **Backend Engineer**: Implements recommendation logic (`optimizer.py`), safety thresholds, what-if conflict simulation (`simulator.py`), anomaly detection (`alerts.py`), and dataset generation (`data_generator.py`).
- **Frontend Engineer**: Develops the interactive user interface in `app.py` using Streamlit and Plotly, handling state management, dynamic filtering, interactive visualizations, and reporting.

### How `app.py` Integrates with the Backend:
- `calculate_vm_stats(df_metrics, df_vms)`: Aggregates historical metrics into statistically sound `avg`, `p95`, `p99`, `max`, and off-hours metrics.
- `generate_recommendations(df_vms, df_stats, df_resources, df_pricing)`: Evaluates safe rightsizing candidates, schedule opportunities, and orphaned assets.
- `ActionSimulator(df_vms, df_resources, recommendations)`: Simulates in-memory cost impact of accepted actions, correctly resolving overlaps without double-counting.
- `detect_alerts(df_vms, df_stats, df_resources)`: Flags bursty workloads, zombie assets, and capacity exhaustion risks.

---

## 💡 Navigating and Using the Dashboard

1. **Overview Tab**: Review aggregate monthly spend, potential savings, and distribution charts. Use sidebar dropdowns to focus on a particular team, environment, or region.
2. **Resource Inventory Tab**: Search by resource name or ID. Click **Analyze Performance ➔** on any row to automatically load its performance telemetry.
3. **Performance Analysis Tab**: Select any VM to review its 30-day timeline. Notice how the safety ceiling lines (p95 at 70%, p99 at 85%) protect bursty machines from premature downsizing.
4. **Optimization Recommendations Tab**:
   - Filter by strategy (*Safe Rightsizing*, *Workload Scheduling*, or *Orphaned Cleanup*).
   - Click **✓ Accept Action** to approve a recommendation.
   - Click **✕ Reject Action** to supply a rejection reason.
5. **Decision History & What-If Tab**: Inspect the financial summary of approved vs. rejected actions, verify compounded savings, and download the approved plan as a CSV file.

---

## 🛡️ Simulated Data Notice & Limitations

- **Demo Mode — Simulated Data**: All telemetry readings and resources in this project are synthetically generated for hackathon demonstration.
- **Non-Destructive Simulation**: Accepting or rejecting recommendations in this dashboard modifies in-memory session state. It **does not alter, stop, resize, or delete** real cloud resources.
- **Session State Persistence**: Decisions are maintained in Streamlit session state during active browser interaction. To preserve decisions permanently across full server restarts, export the approved decisions via the **Export Approved Plan (CSV)** button.
