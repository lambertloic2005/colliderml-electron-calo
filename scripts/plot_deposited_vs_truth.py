"""CLI: plot calibrated deposited energy vs truth energy per electron."""
import argparse
from pathlib import Path
from colliderml_electron import io, plots


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--channel", default="zee")
    p.add_argument("--pileup", default="pu200")
    p.add_argument("--n-events", type=int, default=50)
    p.add_argument("--out", type=Path, default=Path("plots/deposited_vs_truth.png"))
    args = p.parse_args()

    print(f"Loading {args.n_events} events of {args.channel}_{args.pileup}...")
    frames = io.load_frames(channel=args.channel, pileup=args.pileup,
                            max_events=args.n_events)
    fig = plots.deposited_vs_truth(frames, n_events=args.n_events)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
