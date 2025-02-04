import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# TODO: Check for outliers of the features (+- 5SD)
# TODO: Add subcortical measures
# TODO: Prepare long form data for mixed effects models (done)


def prepare_image():
    """Prepare the imaging data for the analysis.

    Subjects with intersex were removed. Subjects with data quality issues were removed.
    The data was saved in csv files. Brain features of interest were selected and saved
    in a JSON file. The imaging data was joined with the trajectory data and covariates
    and was saved in a csv file.

    remapped trajectory to class (low - 0, increasing - 1, decreasing - 2, high - 3)

    """
    # %%
    processed_data_path = Path(
        "data",
        "processed_data",
    )

    # Create the processed_data folder
    if not processed_data_path.exists():
        processed_data_path.mkdir(parents=True)

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
    # 2. Overall MRI clinical report score < 3, which excludes subjects with neurological issues.

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

    t1w_qc_passed_indices = list(
        mri_y_qc_incl_bl[(mri_y_qc_incl_bl.imgincl_t1w_include == 1)].index
    )

    t1w_qc_passed_mask = mri_clin_report_bl.index.isin(t1w_qc_passed_indices)

    score_mask = mri_clin_report_bl.mrif_score < 3

    # No missing values here
    subs_t1w_pass = mri_clin_report_bl[t1w_qc_passed_mask & score_mask]

    ###

    # %%
    ### Now prepare the smri (cortical features) data

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

    # Select the baseline data for the subjects who passed the quality control and drop
    # subjects with missing data and save the data in csv files

    # Cortical thickness data
    mri_y_smr_thk_dst_bl = mri_y_smr_thk_dst[
        mri_y_smr_thk_dst.eventname == "baseline_year_1_arm_1"
    ]

    t1w_cortical_thickness_bl_pass = mri_y_smr_thk_dst_bl[
        mri_y_smr_thk_dst_bl.index.isin(subs_t1w_pass.index)
    ].dropna()

    t1w_cortical_thickness_bl_pass.to_csv(
        Path(
            processed_data_path,
            "t1w_cortical_thickness_bl_pass.csv",
        ),
        index=True,
    )

    # Cortical volume data
    mri_y_smr_vol_dst_bl = mri_y_smr_vol_dst[
        mri_y_smr_vol_dst.eventname == "baseline_year_1_arm_1"
    ]

    t1w_cortical_volume_bl_pass = mri_y_smr_vol_dst_bl[
        mri_y_smr_vol_dst_bl.index.isin(subs_t1w_pass.index)
    ].dropna()

    t1w_cortical_volume_bl_pass.to_csv(
        Path(
            processed_data_path,
            "t1w_cortical_volume_bl_pass.csv",
        ),
        index=True,
    )

    # Cortical surface area data

    mri_y_smr_area_dst_bl = mri_y_smr_area_dst[
        mri_y_smr_area_dst.eventname == "baseline_year_1_arm_1"
    ]

    t1w_cortical_surface_area_bl_pass = mri_y_smr_area_dst_bl[
        mri_y_smr_area_dst_bl.index.isin(subs_t1w_pass.index)
    ].dropna()

    # Save data
    t1w_cortical_surface_area_bl_pass.to_csv(
        Path(
            processed_data_path,
            "t1w_cortical_surface_area_bl_pass.csv",
        ),
        index=True,
    )

    # Combine the three cortical modalities

    t1w_all_cortical_features = pd.concat(
        [
            t1w_cortical_thickness_bl_pass,
            t1w_cortical_volume_bl_pass,
            t1w_cortical_surface_area_bl_pass,
        ],
        axis=1,
    )

    # Drop eventname column
    t1w_all_cortical_features = t1w_all_cortical_features.drop(columns="eventname")

    ### Add covariates to be considered in the analysis (Covariates included age, age2,
    # sex, ethnicity, study site, recent social deprivation and additional imaging
    # covariates: head motion)

    # For site information
    mri_y_adm_info_path = Path(
        imaging_path,
        "mri_y_adm_info.csv",
    )

    mri_y_adm_info = pd.read_csv(
        mri_y_adm_info_path,
        index_col=0,
        low_memory=False,
    )

    mri_y_adm_info_bl = mri_y_adm_info[
        mri_y_adm_info.eventname == "baseline_year_1_arm_1"
    ]

    le = LabelEncoder()

    # Using .fit_transform function to fit label
    # encoder and return encoded label
    label = le.fit_transform(mri_y_adm_info_bl["mri_info_deviceserialnumber"])
    mri_y_adm_info_bl["label_site"] = label

    # For smri_vol_scs_intracranialv (intracranial volume)
    mri_y_smr_vol_aseg_path = Path(
        imaging_path,
        "mri_y_smr_vol_aseg.csv",
    )

    mri_y_smr_vol_aseg = pd.read_csv(
        mri_y_smr_vol_aseg_path,
        index_col=0,
        low_memory=False,
    )

    mri_y_smr_vol_aseg_bl = mri_y_smr_vol_aseg[
        mri_y_smr_vol_aseg.eventname == "baseline_year_1_arm_1"
    ]

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

    # Social deprivation (tried with variables of social deprivation as documented in
    # Shen's work but these variables had full missing values)

    # social_dep_var_dict_path = Path(
    #     "data",
    #     "var_dict",
    #     "social_deprivation_dict.csv",
    # )

    # social_dep_var_dict = pd.read_csv(
    #     social_dep_var_dict_path,
    #     index_col=0,
    #     low_memory=False,
    # )

    # social_dep_var_names = social_dep_var_dict["var_name"].values

    # # Remove samples who refused to answer any of the social deprivation questions (No one removed)
    # demographics_bl = demographics_bl[
    #     ~demographics_bl[social_dep_var_names].isin([777]).any(axis=1)
    # ]

    # demographics_bl["social_deprivation"] = demographics_bl[social_dep_var_names].sum(
    #     axis=1
    # )

    # Add Education levels for now

    # Put the covariates together

    series_list = [
        demographics_bl.demo_sex_v2,
        mri_y_adm_info_bl.label_site,
        mri_y_smr_vol_aseg_bl.smri_vol_scs_intracranialv,
        abcd_y_lt_bl.interview_age,
        abcd_y_lt_bl.age2,
    ]

    covariates = pd.concat(series_list, axis=1)

    # Join the covariates to the brain features (no missing values)

    t1w_all_cortical_features_cov = t1w_all_cortical_features.join(
        covariates,
        how="left",
    )

    t1w_all_cortical_features_cov.to_csv(
        Path(
            processed_data_path,
            "t1w_all_cortical_features_cov.csv",
        ),
        index=True,
    )

    # Join the imaging features and trajectories here

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

    unique_dep_traj = unique_dep_traj.rename(columns={"class": "class_label"})

    # Remap tjrajectory to numeric values
    # old: (low - 2, increasing - 1, decreasing - 3, high - 4)
    # new: (low - 0, increasing - 1, decreasing - 2, high - 3)

    class_label_mapping = {
        2: 0,
        1: 1,
        3: 2,
        4: 3,
    }

    # Apply the mapping to the class_label column
    unique_dep_traj["class_label"] = unique_dep_traj["class_label"].replace(
        class_label_mapping
    )

    # Join the imaging data with the trajectory data
    # 124 missing values for now

    t1w_all_cortical_features_cov_traj = t1w_all_cortical_features_cov.join(
        unique_dep_traj,
        how="left",
    )

    # Remove missing values
    t1w_all_cortical_features_cov_traj = t1w_all_cortical_features_cov_traj.dropna()

    t1w_all_cortical_features_cov_traj.to_csv(
        Path(
            processed_data_path,
            "t1w_all_cortical_features_cov_traj.csv",
        ),
        index=True,
    )

    # %%
    # Create a long form data for mixed effects models by hemispheres

    lh_columns = [
        col for col in t1w_all_cortical_features_cov_traj.columns if col.endswith("lh")
    ]
    rh_columns = [
        col for col in t1w_all_cortical_features_cov_traj.columns if col.endswith("rh")
    ]

    # Identify all non-imaging columns
    non_imaging_columns = [
        col
        for col in t1w_all_cortical_features_cov_traj.columns
        if col not in lh_columns + rh_columns
    ]

    # Create new DataFrames for left and right hemispheres
    lh_data = t1w_all_cortical_features_cov_traj[
        non_imaging_columns + lh_columns
    ].copy()

    rh_data = t1w_all_cortical_features_cov_traj[
        non_imaging_columns + rh_columns
    ].copy()

    # Rename columns to match (remove "lh" and "rh" suffixes)
    lh_data = lh_data.rename(
        columns=lambda col: col.replace("lh", "") if col in lh_columns else col
    )
    rh_data = rh_data.rename(
        columns=lambda col: col.replace("rh", "") if col in rh_columns else col
    )

    lh_data["hemisphere"] = "Left"
    rh_data["hemisphere"] = "Right"

    # Concatenate left and right hemisphere data
    long_data = pd.concat([lh_data, rh_data], axis=0)

    long_data.to_csv(
        Path(
            processed_data_path,
            "t1w_all_cortical_features_cov_traj_long.csv",
        ),
        index=True,
    )

    # %%
    ### Now select the columns that are the phenotypes of interest for each modality

    # For cortical thickness ('mrisdp_1' to 'mrisdp_148')
    t1w_cortical_thickness_rois = list(t1w_cortical_thickness_bl_pass.columns[1:-3])

    # For cortical volume
    t1w_cortical_volume_rois = list(t1w_cortical_volume_bl_pass.columns[1:-3])

    # For surface area
    t1w_cortical_surface_area_rois = list(
        t1w_cortical_surface_area_bl_pass.columns[1:-3]
    )

    # Save the selected features to a dictionary as a JSON file
    brain_features_of_interest = {
        "cortical_thickness": t1w_cortical_thickness_rois,
        "cortical_volume": t1w_cortical_volume_rois,
        "cortical_surface_area": t1w_cortical_surface_area_rois,
    }

    brain_features_of_interest_path = Path(
        processed_data_path,
        "brain_features_of_interest.json",
    )

    with open(brain_features_of_interest_path, "w") as f:
        json.dump(brain_features_of_interest, f)

    # Save unilateral features of interest for mixed effects models for each modalities

    def get_unilateral_features(feature_list):
        seen = set()  # Track unique names
        unilateral_features = []

        for feature in feature_list:
            # Remove "lh" or "rh" at the end of the feature name
            if feature.endswith("lh"):
                feature_name = feature[:-2]
            elif feature.endswith("rh"):
                feature_name = feature[:-2]
            else:
                feature_name = feature

            # Preserve order while ensuring uniqueness
            if feature_name not in seen:
                seen.add(feature_name)
                unilateral_features.append(feature_name)

        return unilateral_features

    unilateral_brain_features = {
        "cortical_thickness": get_unilateral_features(t1w_cortical_thickness_rois),
        "cortical_volume": get_unilateral_features(t1w_cortical_volume_rois),
        "cortical_surface_area": get_unilateral_features(
            t1w_cortical_surface_area_rois
        ),
    }

    unilateral_brain_features_path = Path(
        processed_data_path,
        "unilateral_brain_features.json",
    )

    with open(unilateral_brain_features_path, "w") as f:
        json.dump(unilateral_brain_features, f)


if __name__ == "__main__":
    prepare_image()
