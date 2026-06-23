"""Quick dim sanity check: does the dataset's high-level width match the config,
and does one forward pass run? Reproduces the training crash on a single batch."""
import argparse
import torch
from colliderml_electron.dataset import make_loader
from colliderml_electron.model import ConvCaloRegressor

p = argparse.ArgumentParser()
p.add_argument("--parquet", default="data/electrons/eta_phi_pt_z0_charge/zee_pu200_z0_charge.parquet")
p.add_argument("--stats",   default="data/electrons/eta_phi_pt_z0_charge/target_stats.json")
p.add_argument("--split",   default="train")
p.add_argument("--high-level-dim", type=int, default=21,  # <-- the value you set in the train config
               help="must equal the cfg['high_level_dim'] you are about to train with")
p.add_argument("--output-dim", type=int, default=5)
args = p.parse_args()

loader = make_loader(
    parquet_path=args.parquet,
    split=args.split,
    target_stats_path=args.stats,
    batch_size=8,
    shuffle=False,
    use_angular_features=True,   # same flags training uses
    use_cluster_features=True,
    max_abs_eta=3.0,
)

batch = next(iter(loader))
emitted = batch["x_high_level"].shape[-1]
print(f"dataset emits high_level width = {emitted}")
print(f"config high_level_dim          = {args.high_level_dim}")

assert emitted == args.high_level_dim, (
    f"MISMATCH: data emits {emitted} but config declares {args.high_level_dim}. "
    f"Set cfg['high_level_dim'] = {emitted}."
)

# Build the model exactly as the high-level path requires and forward one batch.
# Only high_level_dim must match the data; the other dims are arbitrary-but-valid here.
model = ConvCaloRegressor(
    max_cells=256, model_dim=64, n_heads=4, n_layers=2,
    dim_feedforward=128, dropout=0.0,
    output_dim=args.output_dim, high_level_dim=args.high_level_dim,
    conv_dim=64, kernel_size=3,
).eval()

with torch.no_grad():
    out = model(batch["x_sampled"], batch["x_high_level"], batch["mask"])

print(f"forward OK -> output shape {tuple(out.shape)} (expected (*, {args.output_dim}))")
assert out.shape[-1] == args.output_dim
print("\nPASS: dims are consistent; training will not hit the Linear mismatch.")