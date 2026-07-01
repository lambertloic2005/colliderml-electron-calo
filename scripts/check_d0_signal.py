"""Does truth d0 have any spread the calorimeter could resolve?"""
import numpy as np
from colliderml_electron.io import load_frames, get_event, prompt_electrons
from colliderml_electron.pipeline import truth_kinematics

frames = load_frames(channel="zee", pileup="pu200", max_events=500)
n_events = frames["particles"].height

d0, z0 = [], []
for i in range(n_events):
    prow, _ = get_event(frames, i)
    for e in prompt_electrons(prow):
        t = truth_kinematics(e)
        d0.append(t["truth_d0"]); z0.append(t["truth_z0"])

d0 = np.array(d0); z0 = np.array(z0)
print(f"n electrons      : {len(d0)}")
print(f"std(truth_z0)    : {z0.std():8.3f} mm   (known recoverable, ~56)")
print(f"std(truth_d0)    : {d0.std():8.4f} mm")
print(f"95% |d0| within  : {np.percentile(np.abs(d0), 95):8.4f} mm")
print(f"max |d0|         : {np.abs(d0).max():8.4f} mm")
# Calorimeter transverse miss-distance resolution ~ sigma_phi * lever_arm
# ~ 0.012 rad * ~1000 mm ~ 10 mm. Decision:
print("\nDecision rule: a 'predict 0' prior already achieves RMSE = std(truth_d0).")
print("If std(truth_d0) << ~5-10 mm, the calorimeter cannot beat it -> STOP.")