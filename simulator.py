"""
UTHO SmartCloud Optimizer - Safe Action Simulator (What-If Engine)
Simulates adoption of optimization recommendations in-memory without modifying live infrastructure.
Features intelligent conflict detection to prevent double-counting between Rightsizing and Scheduling.
"""

from typing import Iterable
import pandas as pd


class ActionSimulator:
    """
    Manages interactive what-if scenarios, calculating live projected costs,
    preventing overlapping recommendation double-counting, and tracking risk exposure.
    """
    def __init__(
        self,
        df_vms: pd.DataFrame,
        df_resources: pd.DataFrame,
        recommendations: list[dict]
    ):
        self.df_vms = df_vms.copy()
        self.df_resources = df_resources.copy()
        self.recommendations = recommendations
        self.rec_map = {rec["rec_id"]: rec for rec in recommendations}
        
        # Calculate baseline monthly expenditure
        vm_baseline = float(self.df_vms["monthly_cost"].sum())
        resource_baseline = float(self.df_resources["monthly_cost"].sum())
        self.baseline_monthly_spend = vm_baseline + resource_baseline
        self.baseline_vm_spend = vm_baseline
        self.baseline_resource_spend = resource_baseline
        
    def simulate_plan(self, accepted_rec_ids: Iterable[str]) -> dict:
        """
        Calculates projected financial and risk impact of accepted recommendations.
        Accurately models compounded savings when both Rightsizing and Scheduling are selected for a single VM.
        """
        accepted_ids = set(accepted_rec_ids)
        accepted_recs = [self.rec_map[r_id] for r_id in accepted_ids if r_id in self.rec_map]
        
        # Group accepted actions by resource_id to detect and resolve overlaps
        recs_by_resource: dict[str, list[dict]] = {}
        for rec in accepted_recs:
            r_id = rec["resource_id"]
            recs_by_resource.setdefault(r_id, []).append(rec)
            
        applied_actions = []
        compounded_actions = []
        total_monthly_savings = 0.0
        savings_by_category = {
            "Safe Rightsizing": 0.0,
            "Workload Scheduling": 0.0,
            "Orphaned Cleanup": 0.0
        }
        
        for r_id, r_list in recs_by_resource.items():
            categories = [r["category"] for r in r_list]
            
            # Case 1: Compounded Rightsizing AND Scheduling on the same VM
            if "Safe Rightsizing" in categories and "Workload Scheduling" in categories:
                rs_rec = next(r for r in r_list if r["category"] == "Safe Rightsizing")
                sc_rec = next(r for r in r_list if r["category"] == "Workload Scheduling")
                
                curr_cost = rs_rec["current_cost"]
                downsized_cost = rs_rec["projected_cost"]
                
                # First downsize: savings = curr_cost - downsized_cost
                rs_part = curr_cost - downsized_cost
                # Then schedule downsized VM: saves 65% of the downsized cost
                sc_part = downsized_cost * 0.65
                effective_savings = rs_part + sc_part
                
                total_monthly_savings += effective_savings
                savings_by_category["Safe Rightsizing"] += rs_part
                savings_by_category["Workload Scheduling"] += sc_part
                
                compounded_actions.append({
                    "resource_id": r_id,
                    "resource_name": rs_rec["resource_name"],
                    "team": rs_rec["team"],
                    "current_cost": curr_cost,
                    "simulated_cost": curr_cost - effective_savings,
                    "effective_savings": round(effective_savings, 2),
                    "note": (
                        f"Compounded Optimization: VM first downsized to {rs_rec['proposed_spec']}, "
                        f"then scheduled for off-hours idle shutdown (saving 65% of downsized cost). "
                        f"Double-counting avoided."
                    )
                })
                applied_actions.extend(r_list)
                
            else:
                # Case 2: Individual independent recommendations
                for rec in r_list:
                    cat = rec["category"]
                    savings = float(rec["monthly_savings"])
                    total_monthly_savings += savings
                    if cat in savings_by_category:
                        savings_by_category[cat] += savings
                    applied_actions.append(rec)
                    
        projected_monthly_spend = max(0.0, self.baseline_monthly_spend - total_monthly_savings)
        savings_pct = round((total_monthly_savings / max(self.baseline_monthly_spend, 1.0)) * 100.0, 1)
        
        # Risk profiling of accepted decisions
        risk_counts = {"Low": 0, "Medium": 0, "High": 0}
        for rec in accepted_recs:
            lvl = rec.get("risk_level", "Low")
            risk_counts[lvl] = risk_counts.get(lvl, 0) + 1
            
        overall_risk = "Low Risk"
        if risk_counts.get("High", 0) > 0:
            overall_risk = "Elevated Risk (High Risk Actions Present)"
        elif risk_counts.get("Medium", 0) >= 3:
            overall_risk = "Moderate Risk (Multiple Medium Actions)"
            
        return {
            "baseline_spend": round(self.baseline_monthly_spend, 2),
            "projected_spend": round(projected_monthly_spend, 2),
            "total_savings": round(total_monthly_savings, 2),
            "savings_pct": savings_pct,
            "accepted_count": len(accepted_recs),
            "savings_by_category": savings_by_category,
            "compounded_count": len(compounded_actions),
            "compounded_details": compounded_actions,
            "risk_profile": risk_counts,
            "overall_risk_assessment": overall_risk,
            "is_simulation": True
        }
