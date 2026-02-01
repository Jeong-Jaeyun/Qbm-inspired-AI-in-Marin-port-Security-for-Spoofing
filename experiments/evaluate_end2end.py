from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt

from policy.engine import load_policy, apply_policy
from blockchain_net.simulator import simulate_raft

def _load(processed_dir: str):
    feats = pd.read_parquet(os.path.join(processed_dir, "features.parquet"))
    windows = pd.read_parquet(os.path.join(processed_dir, "windows.parquet"))
    return feats, windows

def summarize(sim: pd.DataFrame) -> dict:
    return {
        "processed_tps_mean": float(sim["processed_tps"].mean()),
        "latency_ms_mean": float(sim["latency_ms"].mean()),
        "backlog_max": float(sim["backlog"].max()),
        "dropped_sum": float(sim["dropped"].sum()),
        "policy_fired_ratio": float(sim["policy_fired"].mean()),
    }

def main():
    scenarios = {
        "S0": "data/processed/busan/s0",
        "S1": "data/processed/busan/s1",
        "S2": "data/processed/busan/s2",
        "S3": "data/processed/busan/s3",
    }

    policy = load_policy("policy/policy_table.yaml")

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    summary_rows = []

                               
    sims = {}
    for name, pdir in scenarios.items():
        feats, _ = _load(pdir)
        feats_pol = apply_policy(feats, policy)
        sim = simulate_raft(feats_pol, policy)
        sims[name] = sim

        row = {"scenario": name, **summarize(sim)}
        summary_rows.append(row)

                           
        sim.to_csv(f"results/tables/sim_{name}.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv("results/tables/summary_end2end.csv", index=False)
    print("Saved results/tables/summary_end2end.csv")
    print(summary)

                                    
                      
    plt.figure()
    plt.bar(summary["scenario"], summary["processed_tps_mean"])
    plt.xlabel("Scenario")
    plt.ylabel("Mean processed TPS")
    plt.tight_layout()
    plt.savefig("results/figures/bar_tps_mean.png", dpi=200)
    plt.close()

                          
    plt.figure()
    plt.bar(summary["scenario"], summary["latency_ms_mean"])
    plt.xlabel("Scenario")
    plt.ylabel("Mean latency (ms)")
    plt.tight_layout()
    plt.savefig("results/figures/bar_latency_mean.png", dpi=200)
    plt.close()

                                                      
    plt.figure()
    for name in ["S0", "S3"]:
        sim = sims[name].sort_values("window_id")
        plt.plot(sim["window_id"], sim["backlog"], label=name)
    plt.xlabel("window_id")
    plt.ylabel("backlog")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/line_backlog_s0_vs_s3.png", dpi=200)
    plt.close()

    print("Saved figures to results/figures/")

if __name__ == "__main__":
    main()
