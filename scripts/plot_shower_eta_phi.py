"""CLI: η–φ lego plot for a single electron's shower."""
import argparse
from pathlib import Path
from colliderml_electron import io, plots


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--channel", default="zee")
    p.add_argument("--pileup", default="pu200")
    p.add_argument("--n-events", type=int, default=10)
    p.add_argument("--event-idx", type=int, default=0)
    p.add_argument("--electron-rank", type=int, default=0)
    p.add_argument("--half-window", type=float, default=0.4)
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--direct-only", action="store_true")
    p.add_argument("--out", type=Path, default=Path("plots/shower_eta_phi_lego.png"))
    args = p.parse_args()

    print(f"Loading {args.n_events} events of {args.channel}_{args.pileup}...")
    frames = io.load_frames(channel=args.channel, pileup=args.pileup,
                            max_events=args.n_events)
    p_row, c_row = io.get_event(frames, args.event_idx)
    electrons = io.prompt_electrons(p_row)
    if not electrons:
        print("No prompt electrons in this event!")
        return
    electrons.sort(key=lambda e: -e["energy"])
    e = electrons[args.electron_rank]
    print(f"Event {args.event_idx}: electron pid={e['particle_id']} "
          f"E={e['energy']:.2f} GeV")
    fig = plots.shower_eta_phi_lego(
        p_row, c_row, e["particle_id"],
        full_shower=not args.direct_only,
        half_window=args.half_window, n_bins=args.bins,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
