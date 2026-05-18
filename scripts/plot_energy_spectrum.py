"""CLI: plot cell energy spectrum for a channel."""
import argparse
from pathlib import Path
from colliderml_electron import io, plots


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--channel", default="zee")
    p.add_argument("--pileup", default="pu200")
    p.add_argument("--n-events", type=int, default=20)
    p.add_argument("--out", type=Path, default=Path("plots/cell_energy_spectrum.png"))
    args = p.parse_args()

    print(f"Loading {args.n_events} events of {args.channel}_{args.pileup}...")
    frames = io.load_frames(channel=args.channel, pileup=args.pileup,
                            max_events=args.n_events)
    fig = plots.cell_energy_spectrum(frames, n_events=args.n_events)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
