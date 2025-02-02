import json
from pathlib import Path

import pandas as pd


def prepare_image():
    """Prepare the imaging data for the analysis.

    Familial members were removed (randomly kept one). Subjects with intersex were
    removed. Subjects with data quality issues were removed. The data was saved in csv
    files. Brain features of interest were selected and saved in a JSON file.

    """
    processed_data_path = Path(
        "data",
        "nm_processed_data",
    )

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

    genetics_path = Path(
        core_data_path,
        "genetics",
    )

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

    # TODO Replace cbcl_LCA_path with raw path

    # Remove subjects with no cbcl data here

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

    # 6 removed
    subs_t1w_pass = subs_t1w_pass[subs_t1w_pass.index.isin(cbcl.index)]

    # Remove subjects without diagnosis

    psych_dx_path = Path(
        processed_data_path,
        "all_psych_dx_r5.csv",
    )

    psych_dx = pd.read_csv(
        psych_dx_path,
        index_col=0,
        low_memory=False,
    )["psych_dx"]

    subs_t1w_pass = subs_t1w_pass[subs_t1w_pass.index.isin(psych_dx.index)]

    # Genetics and relatedness (NDA 4.0 acspsw03), used to remove familial members.
    # Randomly select one subject from each family to include in the analysis.
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

    # No missing value of family_id after joining
    subs_t1w_pass_fam_id = subs_t1w_pass.join(
        family_id,
        how="inner",
    )

    seed = 42

    # Before removing familial members, a total of 11771 - 10733 = 1038 subjects were removed
    # 10733 - 9031 = 1702
    unrelated_subs_t1w = subs_t1w_pass_fam_id.loc[
        subs_t1w_pass_fam_id.groupby(["rel_family_id"]).apply(
            lambda x: x.sample(n=1, random_state=seed).index[0]
        ),
    ]

    ###

    # %%
    ### Now prepare the smri (cortical features) data

    mri_y_smr_thk_dst_path = Path(
        imaging_path,
        "mri_y_smr_thk_dst.csv",
    )

    mri_y_smr_thk_dst = pd.read_csv(
        mri_y_smr_thk_dst_path,
        index_col=0,
        low_memory=False,
    )

    mri_y_smr_vol_dst_path = Path(
        imaging_path,
        "mri_y_smr_vol_dst.csv",
    )

    mri_y_smr_vol_dst = pd.read_csv(
        mri_y_smr_vol_dst_path,
        index_col=0,
        low_memory=False,
    )

    mri_y_smr_area_dst_path = Path(
        imaging_path,
        "mri_y_smr_area_dst.csv",
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

    # 10 with missing values are dropped here for t1w
    t1w_cortical_thickness_bl_pass = mri_y_smr_thk_dst_bl[
        mri_y_smr_thk_dst_bl.index.isin(unrelated_subs_t1w.index)
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
        mri_y_smr_vol_dst_bl.index.isin(unrelated_subs_t1w.index)
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
        mri_y_smr_area_dst_bl.index.isin(unrelated_subs_t1w.index)
    ].dropna()

    t1w_cortical_surface_area_bl_pass.to_csv(
        Path(
            processed_data_path,
            "t1w_cortical_surface_area_bl_pass.csv",
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
        "t1w_cortical_thickness_rois": t1w_cortical_thickness_rois,
        "t1w_cortical_volume_rois": t1w_cortical_volume_rois,
        "t1w_cortical_surface_area_rois": t1w_cortical_surface_area_rois,
    }

    brain_features_of_interest_path = Path(
        processed_data_path,
        "brain_features_of_interest.json",
    )

    with open(brain_features_of_interest_path, "w") as f:
        json.dump(brain_features_of_interest, f)


if __name__ == "__main__":
    prepare_image()
