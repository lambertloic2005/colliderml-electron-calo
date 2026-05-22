"""CLI: build the per electron training parquet"""
import argparse
from colliderml_electron.pipeline import build_electron_table

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--channel", default="zee")
    p.add_argument("--pileup", default="pu200")
    p.add_argument("--n-events", type=int, default=None, 
                   help="Limit number of events.")
    p.add_argument("--dR-max", type=float, default=0.1,
                   help="Difference in R cut in (eta, phi) space. " \
                   "Cells with difference in R greater than this " \
                   "are dropped")
    p.add_argument("--out", default="data/electrons/electrons.parquet")
    # p.add_argument("--eta-max", type=float, default=2.5,
    #            help="Drop electrons with |truth eta| > eta_max.")
    # p.add_argument("--pt-min", type=float, default=1.0,
    #            help="Drop electrons with truth pT < pt_min GeV.")

    a = p.parse_args()
    build_electron_table(
        channel = a.channel,
        pileup = a.pileup,
        max_events = a.n_events,
        dR_max = a.dR_max,
        out_path = a.out,
        # eta_max = a.eta_max,
        # pt_min = a.pt_min,
    )

if __name__ == "__main__":
    main()
