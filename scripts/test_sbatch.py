import os
import re
from pathlib import Path

script = Path("scripts/test_eta_phi_pt_z0_charge.py")
text = script.read_text()

# Patch by variable name, not by the literal path string.
subs = {
    "checkpoint_path": os.environ["TEST_CKPT"],
    "parquet_path":    os.environ["TEST_PARQUET"],
    "stats_path":      os.environ["TEST_STATS"],
    "output_dir":      os.environ["TEST_OUTPUT_DIR"],
}
for var, newpath in subs.items():
    text, n = re.subn(rf'{var} = Path\("[^"]*"\)',
                      f'{var} = Path("{newpath}")', text)
    if n == 0:
        raise RuntimeError(f"Could not find a Path() assignment to patch for: {var}")

if os.environ["TEST_CKPT"] not in text:
    raise RuntimeError("Test script checkpoint path was not patched.")
if os.environ["TEST_PARQUET"] not in text:
    raise RuntimeError("Test script parquet path was not patched.")
if os.environ["TEST_OUTPUT_DIR"] not in text:
    raise RuntimeError("Test script output directory was not patched.")
if os.environ["TEST_STATS"] not in text:
    raise RuntimeError("Test script stats path was not patched.")

script.write_text(text)
print(f"Patched {script} for this SLURM run.")
