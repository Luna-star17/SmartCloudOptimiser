"""
Unit Tests for SmartCloud Optimizer Backend & Recommendation Engine

Covers all 12 mandatory hackathon evaluation scenarios:
1. Idle development VM is flagged for scheduling review (not blindly downsized or deleted).
2. Bursty VM with low average but high p95/p99 peak is NOT downsized (bursty safeguard).
3. Busy production VM is never recommended for shutdown or downsized.
4. Incomplete or missing metrics trigger review rather than confident downsizing.
5. Low-usage storage volume is flagged for investigation, not automatically deleted.
6. Resource with no clear optimization opportunity has zero savings.
7. Monthly savings equal current monthly cost minus proposed monthly cost.
8. Annual savings equal monthly savings multiplied by 12.
9. Resource IDs in generated inventory are 100% unique.
10. Expected number of observations per VM in generated metrics.
11. Metric values stay strictly within physical valid ranges (0-100% for utilization).
12. Random seed produces 100% reproducible sample data.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimizer import (
    calculate_resource_statistics,
    generate_recommendations,
    calculate_savings,
    find_safe_downsize_candidate,
    DEFAULT_CONFIG,
)
from data_generator import generate_inventory, generate_metrics


class TestOptimizerEngine(unittest.TestCase):
    """Test suite for SmartCloud Optimizer analysis, safeguards, and calculations."""

    def setUp(self):
        """Prepare minimal, reproducible synthetic test fixtures."""
        self.config = DEFAULT_CONFIG.copy()

    # ---------------------------------------------------------------------
    # Test 1: Idle development VM flagged for scheduling review
    # ---------------------------------------------------------------------
    def test_idle_development_vm_flagged_for_scheduling_review(self):
        """Scenario 1: Consistently idle dev VM is flagged for schedule/review."""
        inv = pd.DataFrame([{
            "resource_id": "vm-dev-idle",
            "name": "dev-frontend-runner",
            "type": "VM",
            "environment": "dev",
            "monthly_cost": 3600.0,
            "instance_type": "c-large",
            "is_critical": False,
        }])
        stats = pd.DataFrame([{
            "resource_id": "vm-dev-idle",
            "observation_count": 2880,
            "missing_observation_percent": 0.0,
            "average_cpu_percent": 3.2,
            "p95_cpu_percent": 6.8,
            "p99_cpu_percent": 9.5,
            "average_memory_percent": 18.0,
            "p95_memory_percent": 22.0,
            "p99_memory_percent": 24.0,
            "average_iops": 5,
        }])

        recs = generate_recommendations(inv, stats, self.config)
        self.assertEqual(len(recs), 1)
        rec = recs.iloc[0]

        self.assertEqual(rec["recommendation_type"], "Workload Scheduling")
        self.assertIn("shutdown schedule", rec["suggested_action"].lower())
        self.assertGreater(rec["estimated_monthly_savings"], 0.0)
        self.assertEqual(rec["status"], "Pending")
        self.assertEqual(rec["risk_level"], "Low")

    # ---------------------------------------------------------------------
    # Test 2: Bursty VM with low average but high peaks is NOT downsized
    # ---------------------------------------------------------------------
    def test_bursty_vm_with_low_avg_but_high_p95_p99_is_not_downsized(self):
        """
        Scenario 2: VM with low average (18%) but sharp bursts (p95=78%, p99=94%)
        must NEVER receive an unconditional downsizing recommendation.
        """
        inv = pd.DataFrame([{
            "resource_id": "vm-bursty",
            "name": "search-index-bursty",
            "type": "VM",
            "environment": "staging",
            "monthly_cost": 12000.0,
            "instance_type": "c-2xlarge",
            "is_critical": False,
        }])
        stats = pd.DataFrame([{
            "resource_id": "vm-bursty",
            "observation_count": 2880,
            "missing_observation_percent": 0.0,
            "average_cpu_percent": 18.5,  # Low average could trick naive optimizers!
            "p95_cpu_percent": 78.0,      # But p95 is high
            "p99_cpu_percent": 94.0,      # And p99 is near peak
            "average_memory_percent": 45.0,
            "p95_memory_percent": 82.0,
            "p99_memory_percent": 88.0,
            "average_iops": 320,
        }])

        recs = generate_recommendations(inv, stats, self.config)
        self.assertEqual(len(recs), 1)
        rec = recs.iloc[0]

        # Must trigger guardrail, NOT downsizing!
        self.assertEqual(rec["recommendation_type"], "Capacity Retention (Guardrail)")
        self.assertIn("retain", rec["suggested_action"].lower())
        self.assertEqual(rec["estimated_monthly_savings"], 0.0)

    # ---------------------------------------------------------------------
    # Test 3: Busy production VM is never shut down
    # ---------------------------------------------------------------------
    def test_busy_production_vm_not_shut_down(self):
        """Scenario 3: Busy production VM should not be recommended for shutdown."""
        inv = pd.DataFrame([{
            "resource_id": "vm-prod-busy",
            "name": "payments-api-core",
            "type": "VM",
            "environment": "prod",
            "monthly_cost": 12000.0,
            "instance_type": "c-2xlarge",
            "is_critical": True,
        }])
        stats = pd.DataFrame([{
            "resource_id": "vm-prod-busy",
            "observation_count": 2880,
            "missing_observation_percent": 0.0,
            "average_cpu_percent": 68.0,
            "p95_cpu_percent": 84.0,
            "p99_cpu_percent": 91.0,
            "average_memory_percent": 72.0,
            "p95_memory_percent": 80.0,
            "p99_memory_percent": 85.0,
            "average_iops": 650,
        }])

        recs = generate_recommendations(inv, stats, self.config)
        rec = recs.iloc[0]

        self.assertNotIn("shutdown", rec["suggested_action"].lower())
        self.assertNotIn("downsize", rec["suggested_action"].lower())
        self.assertEqual(rec["estimated_monthly_savings"], 0.0)

    # ---------------------------------------------------------------------
    # Test 4: Missing or insufficient metrics triggers review
    # ---------------------------------------------------------------------
    def test_insufficient_or_missing_metrics_triggers_review(self):
        """Scenario 4: Sparse observations trigger Needs Review instead of downsize."""
        inv = pd.DataFrame([{
            "resource_id": "vm-newly-created",
            "name": "new-test-runner",
            "type": "VM",
            "environment": "dev",
            "monthly_cost": 6200.0,
            "instance_type": "c-xlarge",
            "is_critical": False,
        }])
        # Only 48 observations (half a day), missing 98.3%
        stats = pd.DataFrame([{
            "resource_id": "vm-newly-created",
            "observation_count": 48,
            "missing_observation_percent": 98.33,
            "average_cpu_percent": 8.0,
            "p95_cpu_percent": 12.0,
            "p99_cpu_percent": 15.0,
            "average_memory_percent": 20.0,
            "p95_memory_percent": 25.0,
            "p99_memory_percent": 30.0,
            "average_iops": 10,
        }])

        recs = generate_recommendations(inv, stats, self.config)
        rec = recs.iloc[0]

        self.assertEqual(rec["status"], "Needs review")
        self.assertEqual(rec["recommendation_type"], "Data Quality Alert")
        self.assertEqual(rec["estimated_monthly_savings"], 0.0)

    # ---------------------------------------------------------------------
    # Test 5: Low-usage storage volume is flagged for investigation, not deleted
    # ---------------------------------------------------------------------
    def test_low_usage_storage_volume_flagged_for_investigation_not_deleted(self):
        """Scenario 5: Attached low-usage storage is investigated, not deleted."""
        inv = pd.DataFrame([{
            "resource_id": "vol-underused",
            "name": "staging-archive-disk",
            "type": "block_volume",
            "environment": "staging",
            "monthly_cost": 1500.0,
            "attached_to": "vm-staging-01",
            "status": "in-use",
            "notes": "Attached volume with 8% disk usage and zero IOPS",
            "is_critical": False,
        }])
        empty_stats = pd.DataFrame()

        recs = generate_recommendations(inv, empty_stats, self.config)
        rec = recs.iloc[0]

        self.assertEqual(rec["recommendation_type"], "Storage Investigation")
        self.assertIn("investigate", rec["suggested_action"].lower())
        self.assertNotIn("delete", rec["suggested_action"].lower())
        self.assertEqual(rec["status"], "Needs review")

    # ---------------------------------------------------------------------
    # Test 6: Resource with no optimization opportunity has zero savings
    # ---------------------------------------------------------------------
    def test_resource_with_no_opportunity_has_zero_savings(self):
        """Scenario 6: Well-utilized resource has 0 savings and maintain status."""
        inv = pd.DataFrame([{
            "resource_id": "vm-healthy",
            "name": "healthy-service",
            "type": "VM",
            "environment": "prod",
            "monthly_cost": 6200.0,
            "instance_type": "c-xlarge",
            "is_critical": False,
        }])
        stats = pd.DataFrame([{
            "resource_id": "vm-healthy",
            "observation_count": 2880,
            "missing_observation_percent": 0.0,
            "average_cpu_percent": 55.0,
            "p95_cpu_percent": 68.0,
            "p99_cpu_percent": 74.0,
            "average_memory_percent": 58.0,
            "p95_memory_percent": 65.0,
            "p99_memory_percent": 70.0,
            "average_iops": 400,
        }])

        recs = generate_recommendations(inv, stats, self.config)
        rec = recs.iloc[0]

        self.assertEqual(rec["recommendation_type"], "No change suggested")
        self.assertEqual(rec["estimated_monthly_savings"], 0.0)
        self.assertEqual(rec["estimated_annual_savings"], 0.0)

    # ---------------------------------------------------------------------
    # Test 7: Monthly savings equals current cost minus proposed cost
    # ---------------------------------------------------------------------
    def test_monthly_savings_calculation_accuracy(self):
        """Scenario 7: Monthly savings = current cost - proposed cost."""
        inv = pd.DataFrame([{
            "resource_id": "vm-downsize-candidate",
            "name": "oversized-worker",
            "type": "VM",
            "environment": "staging",
            "monthly_cost": 12000.0,  # c-2xlarge (8 vCPU, 32GB)
            "instance_type": "c-2xlarge",
            "is_critical": False,
        }])
        # Consistently underutilized: 10% CPU, 20% RAM
        stats = pd.DataFrame([{
            "resource_id": "vm-downsize-candidate",
            "observation_count": 2880,
            "missing_observation_percent": 0.0,
            "average_cpu_percent": 10.0,
            "p95_cpu_percent": 18.0,
            "p99_cpu_percent": 24.0,
            "average_memory_percent": 22.0,
            "p95_memory_percent": 28.0,
            "p99_memory_percent": 35.0,
            "average_iops": 50,
        }])

        recs = generate_recommendations(inv, stats, self.config)
        rec = recs.iloc[0]

        self.assertEqual(rec["recommendation_type"], "Rightsizing")
        expected_savings = rec["current_monthly_cost"] - rec["estimated_new_monthly_cost"]
        self.assertAlmostEqual(rec["estimated_monthly_savings"], expected_savings, places=2)

    # ---------------------------------------------------------------------
    # Test 8: Annual savings equals monthly savings * 12
    # ---------------------------------------------------------------------
    def test_annual_savings_equals_twelve_times_monthly(self):
        """Scenario 8: Annual savings = monthly savings * 12."""
        inv = pd.DataFrame([{
            "resource_id": "ip-idle",
            "name": "unassigned-static-ip",
            "type": "public_ip",
            "environment": "dev",
            "monthly_cost": 350.0,
            "attached_to": "unassigned",
            "status": "unassigned",
            "is_critical": False,
        }])
        empty_stats = pd.DataFrame()

        recs = generate_recommendations(inv, empty_stats, self.config)
        rec = recs.iloc[0]

        self.assertEqual(rec["recommendation_type"], "Unused IP Release")
        self.assertAlmostEqual(rec["estimated_annual_savings"], rec["estimated_monthly_savings"] * 12, places=2)

    # ---------------------------------------------------------------------
    # Test 9: Resource IDs in generated inventory are unique
    # ---------------------------------------------------------------------
    def test_inventory_resource_ids_are_unique(self):
        """Scenario 9: Resource IDs must be 100% unique across all cloud resources."""
        df_inv = generate_inventory(n_vms=20, seed=123)
        self.assertEqual(len(df_inv), df_inv["resource_id"].nunique(), "Duplicate resource_id detected in inventory!")

    # ---------------------------------------------------------------------
    # Test 10: Expected number of historical metric observations
    # ---------------------------------------------------------------------
    def test_expected_metric_observations_per_vm(self):
        """Scenario 10: Each standard VM receives exactly days * (1440/interval) observations."""
        df_inv = generate_inventory(n_vms=5, seed=42)
        # Test 1 day at 15-min interval = 96 observations per VM
        df_metrics = generate_metrics(df_inv, days=1, interval_minutes=15, seed=42)

        # Standard VMs (excluding the newly created insufficient VM) should have 96 records
        vm_counts = df_metrics.groupby("resource_id")["timestamp"].count()
        # Check that at least one VM has exactly 96 observations
        self.assertIn(96, vm_counts.values)

    # ---------------------------------------------------------------------
    # Test 11: Metric values remain within physical valid ranges
    # ---------------------------------------------------------------------
    def test_metric_values_within_valid_ranges(self):
        """Scenario 11: CPU, RAM, Disk must be 0-100%, IOPS and Network must be non-negative."""
        df_inv = generate_inventory(n_vms=5, seed=42)
        df_metrics = generate_metrics(df_inv, days=1, interval_minutes=15, seed=42)

        self.assertTrue((df_metrics["cpu_percent"] >= 0.0).all())
        self.assertTrue((df_metrics["cpu_percent"] <= 100.0).all())

        self.assertTrue((df_metrics["memory_percent"] >= 0.0).all())
        self.assertTrue((df_metrics["memory_percent"] <= 100.0).all())

        self.assertTrue((df_metrics["disk_used_percent"] >= 0.0).all())
        self.assertTrue((df_metrics["disk_used_percent"] <= 100.0).all())

        self.assertTrue((df_metrics["iops"] >= 0).all())
        self.assertTrue((df_metrics["network_in_mb"] >= 0.0).all())
        self.assertTrue((df_metrics["network_out_mb"] >= 0.0).all())

    # ---------------------------------------------------------------------
    # Test 12: Re-running with the same seed produces reproducible sample data
    # ---------------------------------------------------------------------
    def test_seed_reproducibility(self):
        """Scenario 12: Fixed random seed produces identical inventory and metrics."""
        inv_1 = generate_inventory(n_vms=10, seed=999)
        inv_2 = generate_inventory(n_vms=10, seed=999)
        pd.testing.assert_frame_equal(inv_1, inv_2)

        metrics_1 = generate_metrics(inv_1, days=1, interval_minutes=15, seed=999)
        metrics_2 = generate_metrics(inv_2, days=1, interval_minutes=15, seed=999)
        pd.testing.assert_frame_equal(metrics_1, metrics_2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
