"""CLI: build the per-cluster training parquet (truth-free DBSCAN selection)."""
import argparse

from colliderml_electron.cluster_pipeline import build_cluster_table


def main():
    p = argparse.ArgumentParser(
        description="Cluster calo cells with DBSCAN (no truth) and label via truth matching."
    )
    p.add_argument("--channel", default="zee")
    p.add_argument("--pileup", default="pu200")
    p.add_argument("--n-events", type=int, default=None, help="Limit number of events.")

    # DBSCAN / selection knobs (all truth-free)
    p.add_argument("--eps", type=float, default=0.05,
                   help="DBSCAN neighbourhood radius in (eta, phi).")
    p.add_argument("--min-samples", type=int, default=4,
                   help="DBSCAN core-point threshold (cell count, or energy sum "
                        "in GeV if --energy-weighted).")
    p.add_argument("--e-thresh-gev", type=float, default=0.1,
                   help="Drop cells below this calibrated energy before clustering.")
    p.add_argument("--energy-weighted", action="store_true",
                   help="Use calibrated cell energy as DBSCAN sample_weight.")

    # truth matching (labels only)
    p.add_argument("--dR-match", type=float, default=0.1,
                   help="Max dR between cluster centroid and truth electron to label it.")
    p.add_argument("--keep-unmatched", action="store_true",
                   help="Also write background clusters (null targets, is_electron=0).")

    p.add_argument("--out", default="data/clusters/clusters.parquet")
    p.add_argument("--shard-min", type=int, default=None)
    p.add_argument("--shard-max", type=int, default=None)
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--n-tasks", type=int, default=1)
    a = p.parse_args()
    build_cluster_table(
        channel=a.channel,
        pileup=a.pileup,
        max_events=a.n_events,
        eps=a.eps,
        min_samples=a.min_samples,
        e_thresh_gev=a.e_thresh_gev,
        energy_weighted=a.energy_weighted,
        dR_match=a.dR_match,
        keep_unmatched=a.keep_unmatched,
        out_path=a.out,
        shard_min=a.shard_min,
        shard_max=a.shard_max,
        task_id=a.task_id,
        n_tasks=a.n_tasks,
    )


if __name__ == "__main__":
    main()