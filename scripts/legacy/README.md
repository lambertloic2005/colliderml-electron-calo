# Legacy scripts

Superseded training and evaluation entry points from earlier stages of the
project, kept for reference. These predate the current five-target model and
the AttnPool architecture.

They target intermediate label sets (eta/phi, then + angular features, then
+ pT) and do not carry z0 or charge. They are not maintained and are not
expected to run against the current dataset code.

The current pipeline is:

    scripts/train_eta_phi_pt_z0_charge.py
    scripts/test_eta_phi_pt_z0_charge.py
