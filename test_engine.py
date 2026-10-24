import pandas as pd
from optimizer import calculate_vm_stats, generate_recommendations
from simulator import ActionSimulator
from alerts import detect_alerts

def test():
    vms = pd.read_csv("data/vms.csv")
    metrics = pd.read_csv("data/metrics.csv")
    resources = pd.read_csv("data/resources.csv")
    pricing = pd.read_csv("data/pricing.csv")

    stats = calculate_vm_stats(metrics, vms)
    print(f"Stats computed for {len(stats)} VMs.")

    recs = generate_recommendations(vms, stats, resources, pricing)
    print(f"Generated {len(recs)} optimization recommendations.")

    sim = ActionSimulator(vms, resources, recs)
    all_rec_ids = [r["rec_id"] for r in recs]
    sim_result = sim.simulate_plan(all_rec_ids)

    print(f"Baseline Spend: ₹{sim_result['baseline_spend']:,.2f}")
    print(f"Simulated Spend: ₹{sim_result['projected_spend']:,.2f}")
    print(f"Total Monthly Savings: ₹{sim_result['total_savings']:,.2f} ({sim_result['savings_pct']}%)")
    print(f"Compounded Actions handled without double-counting: {sim_result['compounded_count']}")

    alerts = detect_alerts(vms, stats, resources)
    print(f"Detected {len(alerts)} alerts & safety guardrails.")
    
    # Check guardrails on Bursty VMs
    bursty_guards = [a for a in alerts if "Bursty" in a["type"]]
    print(f"Bursty Downsizing Guardrails active: {len(bursty_guards)}")

if __name__ == "__main__":
    test()
