import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# TODO: Check for outliers of the features (+- 5SD)

# Standardize the imaging features and age (done)


# %%
poppy_data_path = Path(
    "data",
    "poppy",
)

# Create the processed_data folder
if not poppy_data_path.exists():
    poppy_data_path.mkdir(parents=True)

core_data_path = Path(
    "data",
    "raw_data",
    "core",
)

imaging_path = Path(
    core_data_path,
    "imaging",
)

general_info_path = Path(
    core_data_path,
    "abcd-general",
)

# For biological sex (demo_sex_v2)
demographics_path = Path(
    general_info_path,
    "abcd_p_demo.csv",
)

demographics = pd.read_csv(
    demographics_path,
    index_col=0,
    low_memory=False,
)

demographics_bl = demographics[demographics.eventname == "baseline_year_1_arm_1"]

demographics_bl.demo_sex_v2.value_counts()

inter_sex_subs = demographics_bl[demographics_bl.demo_sex_v2 == 3].index

# Recommended image inclusion (NDA 4.0 abcd_imgincl01)
mri_y_qc_incl_path = Path(
    imaging_path,
    "mri_y_qc_incl.csv",
)

mri_y_qc_incl = pd.read_csv(
    mri_y_qc_incl_path,
    index_col=0,
    low_memory=False,
)

mri_y_qc_incl_bl = mri_y_qc_incl[mri_y_qc_incl.eventname == "baseline_year_1_arm_1"]

# Remove subjects with intersex from the imaging data
mri_y_qc_incl_bl = mri_y_qc_incl_bl[~mri_y_qc_incl_bl.index.isin(inter_sex_subs)]

# %%
### Remove imaging data with data quality issues, overall MRI clinical report is used
# here as well.

# First, we apply quality control to T1 weighted images (for structural features).
# Conditions for inclusion:
# 1. T1w data recommended for inclusion (YES)
# 2. dmri data recommended for inclusion (YES)
# 3. Overall MRI clinical report score < 3, which excludes subjects with neurological issues.

mri_clin_report_path = Path(
    imaging_path,
    "mri_y_qc_clfind.csv",
)

mri_clin_report = pd.read_csv(
    mri_clin_report_path,
    index_col=0,
    low_memory=False,
)

mri_clin_report_bl = mri_clin_report[
    mri_clin_report.eventname == "baseline_year_1_arm_1"
]

qc_passed_indices = list(
    mri_y_qc_incl_bl[
        (mri_y_qc_incl_bl.imgincl_t1w_include == 1)
        & (mri_y_qc_incl_bl.imgincl_dmri_include == 1)
    ].index
)

qc_passed_mask = mri_clin_report_bl.index.isin(qc_passed_indices)

score_mask = mri_clin_report_bl.mrif_score < 3

# No missing values here
subs_pass = mri_clin_report_bl[qc_passed_mask & score_mask]

###

# %%
### Now prepare the smri data

mri_y_smr_thk_dst_path = Path(
    imaging_path,
    "mri_y_smr_thk_dsk.csv",
)

mri_y_smr_thk_dst = pd.read_csv(
    mri_y_smr_thk_dst_path,
    index_col=0,
    low_memory=False,
)

mri_y_smr_vol_dst_path = Path(
    imaging_path,
    "mri_y_smr_vol_dsk.csv",
)

mri_y_smr_vol_dst = pd.read_csv(
    mri_y_smr_vol_dst_path,
    index_col=0,
    low_memory=False,
)

mri_y_smr_area_dst_path = Path(
    imaging_path,
    "mri_y_smr_area_dsk.csv",
)

mri_y_smr_area_dst = pd.read_csv(
    mri_y_smr_area_dst_path,
    index_col=0,
    low_memory=False,
)

mir_y_smr_vol_aseg_path = Path(
    imaging_path,
    "mri_y_smr_vol_aseg.csv",
)

mri_y_smr_vol_aseg = pd.read_csv(
    mir_y_smr_vol_aseg_path,
    index_col=0,
    low_memory=False,
)

mri_y_dti_fa_fs_at_path = Path(
    imaging_path,
    "mri_y_dti_fa_fs_at.csv",
)

mri_y_dti_fa_fs_at = pd.read_csv(
    mri_y_dti_fa_fs_at_path,
    index_col=0,
    low_memory=False,
)

mri_y_dti_md_fs_at_path = Path(
    imaging_path,
    "mri_y_dti_md_fs_at.csv",
)

mri_y_dti_md_fs_at = pd.read_csv(
    mri_y_dti_md_fs_at_path,
    index_col=0,
    low_memory=False,
)

# Select the baseline data for the subjects who passed the quality control and drop
# subjects with missing data and save the data in csv files

# Cortical thickness data
mri_y_smr_thk_dst_bl = mri_y_smr_thk_dst[
    mri_y_smr_thk_dst.eventname == "baseline_year_1_arm_1"
]

t1w_cortical_thickness_bl_pass = mri_y_smr_thk_dst_bl[
    mri_y_smr_thk_dst_bl.index.isin(subs_pass.index)
].dropna()

t1w_cortical_thickness_bl_pass.to_csv(
    Path(
        poppy_data_path,
        "t1w_cortical_thickness_bl_pass.csv",
    ),
    index=True,
)

# Cortical volume data
mri_y_smr_vol_dst_bl = mri_y_smr_vol_dst[
    mri_y_smr_vol_dst.eventname == "baseline_year_1_arm_1"
]

t1w_cortical_volume_bl_pass = mri_y_smr_vol_dst_bl[
    mri_y_smr_vol_dst_bl.index.isin(subs_pass.index)
].dropna()

t1w_cortical_volume_bl_pass.to_csv(
    Path(
        poppy_data_path,
        "t1w_cortical_volume_bl_pass.csv",
    ),
    index=True,
)

# Cortical surface area data

mri_y_smr_area_dst_bl = mri_y_smr_area_dst[
    mri_y_smr_area_dst.eventname == "baseline_year_1_arm_1"
]

t1w_cortical_surface_area_bl_pass = mri_y_smr_area_dst_bl[
    mri_y_smr_area_dst_bl.index.isin(subs_pass.index)
].dropna()

# Save data
t1w_cortical_surface_area_bl_pass.to_csv(
    Path(
        poppy_data_path,
        "t1w_cortical_surface_area_bl_pass.csv",
    ),
    index=True,
)

# Subcortical volume

t1w_subcortical_volume_bl = mri_y_smr_vol_aseg[
    mri_y_smr_vol_aseg.eventname == "baseline_year_1_arm_1"
]

# NOTE: Everyone is missing values for smri_vol_scs_lesionlh and
# smri_vol_scs_lesionrh so I drop them here

t1w_subcortical_volume_bl_pass = t1w_subcortical_volume_bl[
    t1w_subcortical_volume_bl.index.isin(subs_pass.index)
]

t1w_subcortical_volume_bl_pass = t1w_subcortical_volume_bl_pass.drop(
    columns=[
        "smri_vol_scs_lesionlh",
        "smri_vol_scs_lesionrh",
        "smri_vol_scs_wmhintlh",
        "smri_vol_scs_wmhintrh",
    ]
).dropna()

t1w_subcortical_volume_bl_pass.to_csv(
    Path(
        poppy_data_path,
        "t1w_subcortical_volume_bl_pass.csv",
    ),
    index=True,
)

# # Add tracts data (mri_y_dti_fa_fs_at (FA), mri_y_dti_md_fs_at(MD))

dmir_fractional_anisotropy_bl = mri_y_dti_fa_fs_at[
    mri_y_dti_fa_fs_at.eventname == "baseline_year_1_arm_1"
]

dmir_mean_diffusivity_bl = mri_y_dti_md_fs_at[
    mri_y_dti_md_fs_at.eventname == "baseline_year_1_arm_1"
]

dmir_fractional_anisotropy_bl_pass = dmir_fractional_anisotropy_bl[
    dmir_fractional_anisotropy_bl.index.isin(subs_pass.index)
].dropna()

dmir_mean_diffusivity_bl_pass = dmir_mean_diffusivity_bl[
    dmir_mean_diffusivity_bl.index.isin(subs_pass.index)
].dropna()


# Rename the FA and DM features to have "lh" or "rh" suffixes

dmri_data_dict_path = Path(
    "data",
    "poppy",
    "dmri_data_dict.txt",
)

with open(dmri_data_dict_path, "r") as f:
    dmri_data_dict = f.read()


def parse_dti_features_pretty(raw_text: str) -> dict:
    mapping = {}
    for line in raw_text.strip().split("\n"):
        parts = re.split(r"\t+", line.strip())
        if len(parts) < 4:
            continue

        original_feature = parts[0]
        description = parts[3].strip()

        # Determine modality: FA or MD
        if "fractional anisotropy" in description.lower():
            prefix = "FA"
        elif "mean diffusivity" in description.lower():
            prefix = "MD"
        else:
            continue  # skip if neither FA nor MD

        # Determine hemisphere
        if "right" in description.lower():
            suffix = "rh"
        elif "left" in description.lower():
            suffix = "lh"
        else:
            suffix = ""

        # Clean region name
        region = description.lower()
        region = re.sub(
            r"(average|mean)\s+(fractional anisotropy|diffusivity)\s+within",
            "",
            region,
        )
        region = region.strip().strip(".,")
        region = region.replace(",", "")
        region = re.sub(r"\bright\b|\bleft\b", "", region)  # remove 'right' or 'left'
        region = re.sub(
            r"[-/]", "_", region
        )  # replace dashes and slashes with underscores
        region = re.sub(r"\s+", "_", region)  # replace spaces with underscores
        region = region.strip("_")  # remove leading/trailing underscores if any

        cleaned_name = f"{prefix}_{region}{suffix}"
        mapping[original_feature] = cleaned_name

    return mapping


# Sort the columns of the DTI dataframes because they can cause issues with later
# concatenation to create long form data
def sort_dmri_columns(df):
    # Separate 'eventname' from the rest
    columns = list(df.columns)
    first_col = [columns[0]]  # keep 'eventname' first
    rest_cols = columns[1:]

    # Sort by the number at the end of each column name
    sorted_cols = sorted(
        rest_cols,
        key=lambda x: int(re.search(r"_(\d+)$", x).group(1))
        if re.search(r"_(\d+)$", x)
        else float("inf"),
    )

    # Reorder the DataFrame
    return df[first_col + sorted_cols]


dmir_fractional_anisotropy_bl_pass = sort_dmri_columns(
    dmir_fractional_anisotropy_bl_pass
)

dmir_mean_diffusivity_bl_pass = sort_dmri_columns(dmir_mean_diffusivity_bl_pass)


# Parse the DTI features
dti_features_mapping = parse_dti_features_pretty(dmri_data_dict)

# Rename the columns in dmri data

dmir_fractional_anisotropy_bl_pass.rename(
    columns=dti_features_mapping,
    inplace=True,
)
dmir_mean_diffusivity_bl_pass.rename(
    columns=dti_features_mapping,
    inplace=True,
)

# Combine the all the modalities

mri_all_features = pd.concat(
    [
        t1w_cortical_thickness_bl_pass,
        t1w_cortical_volume_bl_pass,
        t1w_cortical_surface_area_bl_pass,
        t1w_subcortical_volume_bl_pass,
        dmir_fractional_anisotropy_bl_pass,
        dmir_mean_diffusivity_bl_pass,
    ],
    axis=1,
)

# Drop eventname column
mri_all_features = mri_all_features.drop(columns="eventname")

### Add covariates to be considered in the analysis (Covariates included age, age2,
# sex, ethnicity, study site, recent social deprivation and additional imaging
# covariates: head motion)

# For site information (imaging device ID)
mri_y_adm_info_path = Path(
    imaging_path,
    "mri_y_adm_info.csv",
)

mri_y_adm_info = pd.read_csv(
    mri_y_adm_info_path,
    index_col=0,
    low_memory=False,
)

mri_y_adm_info_bl = mri_y_adm_info[mri_y_adm_info.eventname == "baseline_year_1_arm_1"]

le = LabelEncoder()

# Using .fit_transform function to fit label
# encoder and return encoded label
label = le.fit_transform(mri_y_adm_info_bl["mri_info_deviceserialnumber"])
mri_y_adm_info_bl["label_site"] = label

# For interview_age (in months)
abcd_y_lt_path = Path(
    general_info_path,
    "abcd_y_lt.csv",
)

abcd_y_lt = pd.read_csv(
    abcd_y_lt_path,
    index_col=0,
    low_memory=False,
)

abcd_y_lt_bl = abcd_y_lt[abcd_y_lt.eventname == "baseline_year_1_arm_1"]

# Add an age squared term

abcd_y_lt_bl["age2"] = abcd_y_lt_bl.interview_age**2

# Add family ID

genetics_path = Path(
    core_data_path,
    "genetics",
)

genetics_relatedness_path = Path(
    genetics_path,
    "gen_y_pihat.csv",
)

genetics_relatedness = pd.read_csv(
    genetics_relatedness_path,
    index_col=0,
    low_memory=False,
)

family_id = genetics_relatedness["rel_family_id"]

# Add household income

household_income = demographics_bl["demo_comb_income_v2"].copy()

# Not available category (777:refused to answer, 999: don't know, missing values)

household_income = household_income.replace([777, np.nan], 999)

# 6 principle components were added here to control for genetic ancestry

pca_path = Path("data", "poppy", "abcd_pca_from_randomforest.tsv")

pca_data = pd.read_csv(pca_path, sep="\t", index_col=0)

pca_data = pca_data.set_index("IID")

pca_data = pca_data[["pc1", "pc2", "pc3", "pc4", "pc5", "pc6"]]

series_list = [
    demographics_bl.demo_sex_v2,
    mri_y_adm_info_bl.label_site,
    # mri_y_smr_vol_aseg_bl.smri_vol_scs_intracranialv,s
    abcd_y_lt_bl.interview_age,
    abcd_y_lt_bl.age2,
    family_id,
    # household_income,
    # ethnicity,
    # prnt_education,
    # height_avg,
    # weight_avg,
]

covariates = pd.concat(series_list, axis=1).dropna()

covariates = covariates.join(pca_data, how="inner")

# Calculate BMI z-score referencing to WHO growth standards

# who_bmi_std_path = Path(
#     "data",
#     "raw_data",
#     "who_bmi",
# )

# boys_std_path = Path(
#     who_bmi_std_path,
#     "bmi-boys-z-who-2007-exp.xlsx",
# )

# girls_std_path = Path(
#     who_bmi_std_path,
#     "bmi-girls-z-who-2007-exp.xlsx",
# )

# boys_lms = pd.read_excel(boys_std_path)
# girls_lms = pd.read_excel(girls_std_path)

# def convert_units(weight_lbs, height_inches):
#     weight_kg = weight_lbs * 0.453592  # Convert pounds to kg
#     height_m = height_inches * 0.0254  # Convert inches to meters
#     return weight_kg, height_m

# def calculate_bmi(weight_kg, height_m):
#     return weight_kg / (height_m**2)

# def get_lms_values(age_months, sex):
#     """Retrieves L, M, S values for a given age (months) and sex.
#     Sex: 1 = Male (boys_lms), 2 = Female (girls_lms).
#     """
#     lms_table = boys_lms if sex == 1 else girls_lms
#     row = lms_table[lms_table["Month"] == age_months]

#     if row.empty:
#         raise ValueError(
#             f"LMS values not found for age {age_months} months and sex {sex}"
#         )

#     return row["L"].values[0], row["M"].values[0], row["S"].values[0]

# def calculate_bmi_zscore(bmi, age_months, sex):
#     L, M, S = get_lms_values(age_months, sex)

#     if L == 0:
#         z = np.log(bmi / M) / S  # Special case when L = 0
#     else:
#         z = ((bmi / M) ** L - 1) / (L * S)

#     return z

# # Convert weight and height to metric units
# covariates[["weight_kg", "height_m"]] = covariates.apply(
#     lambda row: convert_units(row["anthroweightcalc"], row["anthroheightcalc"]),
#     axis=1,
#     result_type="expand",
# )

# # Calculate BMI
# covariates["BMI"] = covariates.apply(
#     lambda row: calculate_bmi(row["weight_kg"], row["height_m"]), axis=1
# )

# Calculate BMI z-scores
# covariates["BMI_zscore"] = covariates.apply(
#     lambda row: calculate_bmi_zscore(
#         row["BMI"],
#         row["interview_age"],
#         row["demo_sex_v2"],
#     ),
#     axis=1,
# )

# Join the covariates to the brain features ()

mri_all_features_cov = mri_all_features.join(
    covariates,
    how="left",
).dropna()

mri_all_features_cov.to_csv(
    Path(
        poppy_data_path,
        "mri_all_features_cov.csv",
    ),
    index=True,
)

# %% TODO: This section joins the rick factors

PRS_path = Path(
    "data",
    "poppy",
    "abcd_plink1-aoDEP-EUR.profiles",
)

# Read the PRS file as space-delimited
prs_df = pd.read_csv(PRS_path, delim_whitespace=True)

prs_df = prs_df.set_index("IID").drop(columns=["FID"])

# Rename the index name here for later long data concatenation
prs_df.index.name = "src_subject_id"

# 4313 removed (NOTE: you might wanna ask if this is expected)
mri_all_features_with_prs = mri_all_features_cov.join(prs_df, how="inner")

mri_all_features_with_prs.to_csv(
    Path(
        poppy_data_path,
        "mri_all_features_with_prs.csv",
    ),
    index=True,
)

# %% Keep unrelated subjects

seed = 42

mri_all_features_with_prs = mri_all_features_with_prs.loc[
    mri_all_features_with_prs.groupby(["rel_family_id"]).apply(
        lambda x: x.sample(n=1, random_state=seed).index[0]
    ),
]

# Standardize the continuous variables

# Columns to exclude from scaling
exclude_cols = [
    "demo_sex_v2",
    "label_site",
    "rel_family_id",
]

# Get columns to scale (everything else)
cols_to_scale = [
    col for col in mri_all_features_with_prs.columns if col not in exclude_cols
]

# Standardize selected columns
scaler = StandardScaler()

mri_all_features_with_prs[cols_to_scale] = scaler.fit_transform(
    mri_all_features_with_prs[cols_to_scale]
)

rescaled_mri_all_features_with_prs = mri_all_features_with_prs.copy()

# This is for performing GLM (for unilateral features)
rescaled_mri_all_features_with_prs.to_csv(
    Path(
        poppy_data_path,
        "mri_all_features_with_prs_rescaled.csv",
    ),
    index=True,
)

# %%
# Identify left/right hemisphere columns
lh_columns = [
    col for col in rescaled_mri_all_features_with_prs.columns if col.endswith("lh")
]
rh_columns = [
    col for col in rescaled_mri_all_features_with_prs.columns if col.endswith("rh")
]

# Identify all non-imaging columns
other_columns = [
    col
    for col in rescaled_mri_all_features_with_prs.columns
    if col not in lh_columns + rh_columns
]

# Create left and right hemisphere datasets
lh_data = rescaled_mri_all_features_with_prs[other_columns + lh_columns].copy()
rh_data = rescaled_mri_all_features_with_prs[other_columns + rh_columns].copy()

# Add a prefix to the imaging columns to avoid duplicates
# Rename feature columns to remove hemisphere suffixes
lh_data = lh_data.rename(
    columns={col: f"img_{col[:-2]}" for col in lh_columns if col.endswith("lh")}
)

rh_data = rh_data.rename(
    columns={col: f"img_{col[:-2]}" for col in rh_columns if col.endswith("rh")}
)

# Add hemisphere and subject ID
lh_data["hemisphere"] = "Left"
rh_data["hemisphere"] = "Right"

# Concatenate into long-form
long_data = pd.concat(
    [lh_data, rh_data],
    axis=0,
)
# %%

# Save (index already captured in column)
long_data.to_csv(
    Path(poppy_data_path, "mri_all_features_with_prs_long_rescaled.csv"), index=True
)
# %%
### Now select the columns that are the phenotypes of interest for each modality

### Remove global features for all modality
t1w_cortical_thickness_rois = list(t1w_cortical_thickness_bl_pass.columns[1:-3])

# For cortical volume
t1w_cortical_volume_rois = list(t1w_cortical_volume_bl_pass.columns[1:-3])

# For surface area
t1w_cortical_surface_area_rois = list(t1w_cortical_surface_area_bl_pass.columns[1:-3])

### For subcortical volume

# NOTE: A list of global features selected by GPT, this might need to be updated
global_subcortical_features = [
    "smri_vol_scs_csf",
    "smri_vol_scs_wholeb",
    "smri_vol_scs_intracranialv",
    "smri_vol_scs_latventricles",
    "smri_vol_scs_allventricles",
    "smri_vol_scs_subcorticalgv",
    "smri_vol_scs_suprateialv",
    "smri_vol_scs_wmhint",
]

# FA global features
global_FA_features = [
    "FA_all_dti_atlas_tract_fibers",
    "FA_hemisphere_dti_atlas_tract_fibers_without_corpus_callosumrh",
    "FA_hemisphere_dti_atlas_tract_fibers_without_corpus_callosumlh",
    "FA_hemisphere_dti_atlas_tract_fibersrh",
    "FA_hemisphere_dti_atlas_tract_fiberslh",
]

# MD global features
global_MD_features = [
    "MD_all_dti_atlas_tract_fibers",
    "MD_hemisphere_dti_atlas_tract_fibers_without_corpus_callosumrh",
    "MD_hemisphere_dti_atlas_tract_fibers_without_corpus_callosumlh",
    "MD_hemisphere_dti_atlas_tract_fibersrh",
    "MD_hemisphere_dti_atlas_tract_fiberslh",
]

# Step 2: Select subcortical ROIs
t1w_subcortical_volume_rois = [
    col
    for col in t1w_subcortical_volume_bl_pass.columns
    if col not in global_subcortical_features and col != "eventname"
]

# For tract features

FA_rois = [
    col
    for col in dmir_fractional_anisotropy_bl_pass.columns
    if col not in global_FA_features and col != "eventname"
]

MD_rois = [
    col
    for col in dmir_mean_diffusivity_bl_pass.columns
    if col not in global_MD_features and col != "eventname"
]

# Save features of interest for mixed effects models for each modalities


def get_bilateral_and_unilateral_features(feature_list):
    """Returns bilateral and unilateral features from a list of features."""
    lh_roots = {f[:-2] for f in feature_list if f.endswith("lh")}
    rh_roots = {f[:-2] for f in feature_list if f.endswith("rh")}
    bilateral_roots = sorted(lh_roots & rh_roots)

    # Unilateral = present in only one hemisphere or has no suffix
    unilateral_features = [
        f
        for f in feature_list
        if (f.endswith("lh") and f[:-2] not in bilateral_roots)
        or (f.endswith("rh") and f[:-2] not in bilateral_roots)
        or (not f.endswith("lh") and not f.endswith("rh"))
    ]

    # Add prefix (img_) to the bilateral features

    bilateral_roots = [f"img_{f}" for f in bilateral_roots]

    return bilateral_roots, unilateral_features


# Assemble all features for repeated effects modeling
features_of_interest = {
    "bilateral_cortical_thickness": get_bilateral_and_unilateral_features(
        t1w_cortical_thickness_rois
    )[0],
    "bilateral_cortical_volume": get_bilateral_and_unilateral_features(
        t1w_cortical_volume_rois
    )[0],
    "bilateral_cortical_surface_area": get_bilateral_and_unilateral_features(
        t1w_cortical_surface_area_rois
    )[0],
    "bilateral_subcortical_volume": get_bilateral_and_unilateral_features(
        t1w_subcortical_volume_rois
    )[0],
    # Unilateral features are for performing GLM
    "unilateral_subcortical_features": get_bilateral_and_unilateral_features(
        t1w_subcortical_volume_rois
    )[1],
    "bilateral_tract_FA": get_bilateral_and_unilateral_features(FA_rois)[0],
    "bilateral_tract_MD": get_bilateral_and_unilateral_features(MD_rois)[0],
    # Unilateral features are for performing GLM
    "unilateral_tract_FA": get_bilateral_and_unilateral_features(FA_rois)[1],
    "unilateral_tract_MD": get_bilateral_and_unilateral_features(MD_rois)[1],
}

features_for_repeated_effects_path = Path(
    poppy_data_path,
    "bilateral_features.json",
)

with open(features_for_repeated_effects_path, "w") as f:
    json.dump(features_of_interest, f)
