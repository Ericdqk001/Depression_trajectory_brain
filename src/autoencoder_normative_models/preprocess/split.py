from pathlib import Path

import pandas as pd

psydx_path = Path(
    "data",
    "processed_data",
    "all_psych_dx_r5.csv",
)

psydx = pd.read_csv(
    psydx_path,
    index_col=0,
    low_memory=False,
)

cbcl_path = Path(
    "data",
    "LCA",
    "cbcl_t_no_mis_dummy.csv",
)

cbcl = pd.read_csv(
    cbcl_path,
    index_col=0,
    low_memory=False,
)

cbcl_scales = cbcl.columns

t1w_ct_bl_pass_path = Path(
    "data",
    "nm_processed_data",
    "t1w_cortical_thickness_bl_pass.csv",
)

t1w_ct_bl_pass = pd.read_csv(
    t1w_ct_bl_pass_path,
    index_col=0,
    low_memory=False,
)

cohort_no_psydx = psydx[psydx["psych_dx"] == "control"]

# No missing data
normative_cohort = t1w_ct_bl_pass.join(
    cbcl,
    how="inner",
).join(
    cohort_no_psydx,
    how="inner",
)

# Remove samples with 2 for any of the CBCL scales

normative_cohort = normative_cohort[
    normative_cohort[cbcl_scales].apply(
        lambda x: x.isin([0, 1]).all(),
        axis=1,
    )
]

# Read the depressive trajectory data

dep_traj_path = Path(
    "data",
    "trajectories",
    "ABCD_BPM_4Trajectories_long.txt",
)

dep_traj = pd.read_csv(
    dep_traj_path,
    sep="\t",
)

dep_traj_path = Path(
    "data",
    "trajectories",
    "ABCD_BPM_4Trajectories_long.txt",
)

dep_traj = pd.read_csv(
    dep_traj_path,
    sep="\t",
)

# Ensure src_subject_id is unique with information on trajectory and class
unique_dep_traj = dep_traj.groupby("src_subject_id").agg(
    {"trajectory": "first", "class": "first"}
)
