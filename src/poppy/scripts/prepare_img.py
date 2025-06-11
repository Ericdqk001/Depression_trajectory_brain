import json
import re
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# TODO: Check for outliers of the features (+- 5SD)


def preprocess(
    wave: str = "baseline_year_1_arm_1",
    version_name: str = "abcd_pgcmdd3",
):
    print("-----------------------")
    print("Processing wave: ", wave)
    # %%

    data_store_path = Path(
        "/",
        "Volumes",
        "GenScotDepression",
    )

    if data_store_path.exists():
        print("Mounted data store path: ", data_store_path)

    analysis_root_path = Path(
        data_store_path,
        "users",
        "Eric",
        "poppy_neuroimaging",
    )

    analysis_data_path = Path(
        analysis_root_path,
        "data",
    )

    processed_data_path = Path(
        analysis_root_path,
        version_name,
        "processed_data",
    )

    if not processed_data_path.exists():
        processed_data_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    core_data_path = Path(
        data_store_path,
        "data",
        "abcd",
        "release5.1",
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

    # Select the baseline year 1 demographics data
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

    mri_y_qc_incl = mri_y_qc_incl[mri_y_qc_incl.eventname == wave]

    print("Sample size with MRI recommended inclusion", mri_y_qc_incl.shape[0])

    # Remove subjects with intersex from the imaging data
    mri_y_qc_incl = mri_y_qc_incl[~mri_y_qc_incl.index.isin(inter_sex_subs)]

    print(
        "Remove intersex subjects from the imaging data, number = ", len(inter_sex_subs)
    )

    # %%
    ### Remove imaging data with data quality issues, overall MRI clinical report is used
    # here as well.

    # First, we apply quality control to T1 weighted images (for structural features).
    # Conditions for inclusion:
    # 1. T1w data recommended for inclusion (YES)
    # 2. dmri data recommended for inclusion (YES)
    # 3. Overall MRI clinical report score < 3, which excludes subjects with neurological issues.

    print("Quality Control Criteria:")
    print("T1 data recommended for inclusion = 1")
    print("dMRI data recommended for inclusion = 1")
    print("Overall MRI clinical report score < 3")

    mri_clin_report_path = Path(
        imaging_path,
        "mri_y_qc_clfind.csv",
    )

    mri_clin_report = pd.read_csv(
        mri_clin_report_path,
        index_col=0,
        low_memory=False,
    )

    mri_clin_report_bl = mri_clin_report[mri_clin_report.eventname == wave]

    qc_passed_indices = list(
        mri_y_qc_incl[
            (mri_y_qc_incl.imgincl_t1w_include == 1)
            & (mri_y_qc_incl.imgincl_dmri_include == 1)
        ].index
    )

    qc_passed_mask = mri_clin_report_bl.index.isin(qc_passed_indices)

    print(
        "Sample size after QC passed, number = ",
        mri_clin_report_bl[qc_passed_mask].shape[0],
    )

    score_mask = mri_clin_report_bl.mrif_score < 3

    subs_pass = mri_clin_report_bl[qc_passed_mask & score_mask]

    print(
        "sample size after QC passed and clinical report (score < 3), number = ",
        subs_pass.shape[0],
    )

    ###

    # %%
    ### Now prepare the imaging data

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

    # Select the data for the subjects who passed the quality control and drop
    # subjects with missing data

    # Cortical thickness data
    mri_y_smr_thk_dst = mri_y_smr_thk_dst[mri_y_smr_thk_dst.eventname == wave]

    print(
        "Sample size with T1w cortical thickness data, number =",
        mri_y_smr_thk_dst.shape[0],
    )

    t1w_cortical_thickness_pass = mri_y_smr_thk_dst[
        mri_y_smr_thk_dst.index.isin(subs_pass.index)
    ].dropna()

    print(
        "Sample size with complete CT data after QC, number =",
        t1w_cortical_thickness_pass.shape[0],
    )

    # Cortical volume data
    mri_y_smr_vol_dst = mri_y_smr_vol_dst[mri_y_smr_vol_dst.eventname == wave]

    print(
        "Sample size with T1w cortical volume data, number =",
        mri_y_smr_vol_dst.shape[0],
    )

    t1w_cortical_volume_pass = mri_y_smr_vol_dst[
        mri_y_smr_vol_dst.index.isin(subs_pass.index)
    ].dropna()

    print(
        "Sample size with complete CV data after QC, number =",
        t1w_cortical_volume_pass.shape[0],
    )

    # Cortical surface area data

    mri_y_smr_area_dst = mri_y_smr_area_dst[mri_y_smr_area_dst.eventname == wave]

    print(
        "Sample size with T1w cortical surface area data, number =",
        mri_y_smr_area_dst.shape[0],
    )

    t1w_cortical_surface_area_pass = mri_y_smr_area_dst[
        mri_y_smr_area_dst.index.isin(subs_pass.index)
    ].dropna()

    print(
        "Sample size with complete SA data after QC, number =",
        t1w_cortical_surface_area_pass.shape[0],
    )

    # Subcortical volume

    t1w_subcortical_volume = mri_y_smr_vol_aseg[mri_y_smr_vol_aseg.eventname == wave]

    print(
        "Sample size with T1w subcortical volume data, number =",
        t1w_subcortical_volume.shape[0],
    )

    t1w_subcortical_volume_pass = t1w_subcortical_volume[
        t1w_subcortical_volume.index.isin(subs_pass.index)
    ]

    # NOTE: These columns were dropped because they had all missing values or all zeros

    subcortical_all_zeros_cols = [
        "smri_vol_scs_lesionlh",
        "smri_vol_scs_lesionrh",
        "smri_vol_scs_wmhintlh",
        "smri_vol_scs_wmhintrh",
    ]

    t1w_subcortical_volume_pass = t1w_subcortical_volume_pass.drop(
        columns=subcortical_all_zeros_cols
    ).dropna()

    print("Subcortical all zeros columns dropped")
    print("Column names: ", subcortical_all_zeros_cols)

    print(
        "Sample size with complete subcortical volume data after QC, number =",
        t1w_subcortical_volume_pass.shape[0],
    )

    # # Add tracts data (mri_y_dti_fa_fs_at (FA), mri_y_dti_md_fs_at(MD))

    dmir_fractional_anisotropy = mri_y_dti_fa_fs_at[
        mri_y_dti_fa_fs_at.eventname == wave
    ]

    print(
        "Sample size with dMRI fractional anisotropy data, number =",
        dmir_fractional_anisotropy.shape[0],
    )

    # Dropna later because some columns will be removed
    dmir_fractional_anisotropy_pass = dmir_fractional_anisotropy[
        dmir_fractional_anisotropy.index.isin(subs_pass.index)
    ]

    print(
        "Sample size with complete FA data after QC, number =",
        dmir_fractional_anisotropy_pass.shape[0],
    )

    dmir_mean_diffusivity = mri_y_dti_md_fs_at[mri_y_dti_md_fs_at.eventname == wave]

    print(
        "Sample size with dMRI mean diffusivity data, number =",
        dmir_mean_diffusivity.shape[0],
    )

    dmir_mean_diffusivity_pass = dmir_mean_diffusivity[
        dmir_mean_diffusivity.index.isin(subs_pass.index)
    ]

    print(
        "Sample size with complete MD data after QC, number =",
        dmir_mean_diffusivity_pass.shape[0],
    )

    # Rename the FA and DM features to have "lh" or "rh" suffixes

    dmri_data_dict_path = Path(
        analysis_data_path,
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
            region = re.sub(
                r"\bright\b|\bleft\b", "", region
            )  # remove 'right' or 'left'
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

    dmir_fractional_anisotropy_pass = sort_dmri_columns(dmir_fractional_anisotropy_pass)

    dmir_mean_diffusivity_pass = sort_dmri_columns(dmir_mean_diffusivity_pass)

    print("Sort dMRI FA/MD columns by number at the end of each column name")
    print("For example: 'dmdtifp1_43', ''dmdtifp1_44', 'dmdtifp1_45'")
    print("Sorting is error-free, checked")

    # Parse the DTI features
    dti_features_mapping = parse_dti_features_pretty(dmri_data_dict)

    print(
        "Parsing FA/MD feature descriptions to new feature names is error-free, Checked"
    )

    # Rename the columns in dmri data
    dmir_fractional_anisotropy_pass.rename(
        columns=dti_features_mapping,
        inplace=True,
    )

    print("Renaming FA features to new feature names is error-free, Checked")

    # Drop these columns because they are duplicates with a slightly different regional focus
    FA_cols_to_drop = [
        "FA_dti_atlas_tract_fornix_excluding_fimbrialh",
        "FA_dti_atlas_tract_fornix_excluding_fimbriarh",
        "FA_dti_atlas_tract_superior_corticostriate_frontal_cortex_onlylh",
        "FA_dti_atlas_tract_superior_corticostriate_frontal_cortex_onlyrh",
        "FA_dti_atlas_tract_superior_corticostriate_parietal_cortex_onlylh",
        "FA_dti_atlas_tract_superior_corticostriate_parietal_cortex_onlyrh",
    ]

    print(
        "Drop the following FA columns because they are duplicates with a slightly different regional focus:"
    )
    print(FA_cols_to_drop)

    print(
        "FA number of features before dropping columns: ",
        dmir_fractional_anisotropy_pass.shape[1],
    )

    dmir_fractional_anisotropy_pass = dmir_fractional_anisotropy_pass.drop(
        columns=FA_cols_to_drop
    )

    print(
        "FA number of features after dropping columns: ",
        dmir_fractional_anisotropy_pass.shape[1],
    )

    dmir_fractional_anisotropy_pass = dmir_fractional_anisotropy_pass.dropna()

    print(
        "Sample size with complete FA data after QC, number =",
        dmir_fractional_anisotropy_pass.shape[0],
    )

    dmir_mean_diffusivity_pass.rename(
        columns=dti_features_mapping,
        inplace=True,
    )

    print("Renaming MD features to new feature names is error-free, Checked")

    MD_cols_to_drop = [
        "MD_dti_atlas_tract_fornix_excluding_fimbrialh",
        "MD_dti_atlas_tract_fornix_excluding_fimbriarh",
        "MD_dti_atlas_tract_superior_corticostriate_frontal_cortex_onlylh",
        "MD_dti_atlas_tract_superior_corticostriate_frontal_cortex_onlyrh",
        "MD_dti_atlas_tract_superior_corticostriate_parietal_cortex_onlylh",
        "MD_dti_atlas_tract_superior_corticostriate_parietal_cortex_onlyrh",
    ]
    print(
        "Drop the following MD columns because they are duplicates with a slightly different regional focus:"
    )
    print(MD_cols_to_drop)

    print(
        "MD number of features before dropping columns: ",
        dmir_mean_diffusivity_pass.shape[1],
    )

    dmir_mean_diffusivity_pass = dmir_mean_diffusivity_pass.drop(
        columns=MD_cols_to_drop
    )

    print(
        "MD number of features after dropping columns: ",
        dmir_mean_diffusivity_pass.shape[1],
    )

    dmir_mean_diffusivity_pass = dmir_mean_diffusivity_pass.dropna()

    print(
        "Sample size with complete MD data after QC, number =",
        dmir_mean_diffusivity_pass.shape[0],
    )

    # Combine all the modalities

    mri_all_features = pd.concat(
        [
            t1w_cortical_thickness_pass,
            t1w_cortical_volume_pass,
            t1w_cortical_surface_area_pass,
            t1w_subcortical_volume_pass,
            dmir_fractional_anisotropy_pass,
            dmir_mean_diffusivity_pass,
        ],
        axis=1,
    )

    print("Sample size with all imaging features, number = ", mri_all_features.shape[0])

    # Drop eventname column
    mri_all_features = mri_all_features.drop(columns="eventname")

    ### Add covariates to be considered in the analysis

    print("Adding covariates to the imaging features")

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

    mri_y_adm_info = mri_y_adm_info[mri_y_adm_info.eventname == wave]

    le = LabelEncoder()

    # Using .fit_transform function to fit label
    # encoder and return encoded label
    label = le.fit_transform(mri_y_adm_info["mri_info_deviceserialnumber"])

    print("Add covariate: mri_info_deviceserialnumber")

    mri_y_adm_info["img_device_label"] = label

    print("Using LabelEncoder to encode the imaging device ID is error-free, Checked")

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

    abcd_y_lt = abcd_y_lt[abcd_y_lt.eventname == wave]

    # Add an age squared term

    abcd_y_lt["age2"] = abcd_y_lt.interview_age**2

    print("Add covariate: interview_age and age2 (sqaured interview_age)")

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

    # household_income = demographics_bl["demo_comb_income_v2"].copy()

    # Not available category (777:refused to answer, 999: don't know, missing values)

    # household_income = household_income.replace(
    #     [777, 999],
    #     np.nan,
    # )

    # print(
    #     "Subjects who either refused to answer or don't know their income are set to NA"
    # )

    # 6 principle components were added here to control for genetic ancestry

    # pca_path = Path(
    #     analysis_data_path,
    #     "abcd_pca_from_randomforest.tsv",
    # )

    # pca_data = pd.read_csv(
    #     pca_path,
    #     sep="\t",
    #     index_col=0,
    # )

    # pca_data = pca_data.set_index("IID")

    # pca_data = pca_data[
    #     [
    #         "pc1",
    #         "pc2",
    #         "pc3",
    #         "pc4",
    #         "pc5",
    #         "pc6",
    #     ]
    # ]

    # print("Add covariates: 6 principle components from abcd_pca_from_randomforest.tsv")

    series_list = [
        demographics_bl.demo_sex_v2,
        mri_y_adm_info.img_device_label,
        abcd_y_lt.interview_age,
        abcd_y_lt.age2,
        family_id,
        # household_income,
    ]

    covariates = pd.concat(series_list, axis=1).dropna()

    # covariates = covariates.join(pca_data, how="inner")

    # Join the covariates to the brain features

    mri_all_features_cov = mri_all_features.join(
        covariates,
        how="left",
    ).dropna()

    print(
        "Sample size with all imaging features and covariates, number = ",
        mri_all_features_cov.shape[0],
    )

    # %% TODO: This section joins the rick factors

    # PRS_path = Path(
    #     "data",
    #     "poppy",
    #     "adoldep_noABCD_sbayesrc.profile",
    # )

    PRS_path = Path(
        analysis_data_path,
        "ABCD_CBCL_quant_pheno.txt",
    )

    print(
        "PRS data file name: ",
        PRS_path.name,
    )

    # Read the PRS file as space-delimited
    prs_df = pd.read_csv(
        PRS_path,
        delim_whitespace=True,
    )

    prs_df = prs_df.set_index("IID")

    # Select wave for prs_df
    wave_number_map = {
        "baseline_year_1_arm_1": 0,
        "2_year_follow_up_y_arm_1": 2,
        # "4_year_follow_up_y_arm_1": 3,
    }

    wave_number = wave_number_map.get(wave)

    prs_df = prs_df[prs_df["time"] == wave_number]

    # Rename the index name here for later long data concatenation
    prs_df.index.name = "src_subject_id"

    # Drop not needed columns
    not_needed_cols = [
        "age",
        "sex",
        "time",
    ]

    prs_df = prs_df.drop(columns=not_needed_cols)

    # A lot removed (NOTE: you might wanna ask if this is expected)
    mri_all_features_with_prs = mri_all_features_cov.join(prs_df, how="inner")

    mri_all_features_with_prs = mri_all_features_with_prs.dropna()

    print(
        "Sample size with all imaging features and covariates and PRS, number = ",
        mri_all_features_with_prs.shape[0],
    )

    # %% Keep unrelated subjects

    seed = 42

    print("Keeping unrelated subjects, random seed = ", seed)

    mri_all_features_with_prs = mri_all_features_with_prs.loc[
        mri_all_features_with_prs.groupby(["rel_family_id"]).apply(
            lambda x: x.sample(n=1, random_state=seed).index[0]
        ),
    ]

    print("Keeping unrelated subjects is error-free, Checked")

    print(
        "Sample size after keeping unrelated subjects, number = ",
        mri_all_features_with_prs.shape[0],
    )

    # Standardize the continuous variables

    print("Standardizing the continuous variables")

    categorical_variables = [
        "demo_sex_v2",
        "img_device_label",
        "rel_family_id",
        # "demo_comb_income_v2",
    ]

    for col in categorical_variables:
        if col in mri_all_features_with_prs.columns:
            mri_all_features_with_prs[col] = mri_all_features_with_prs[col].astype(
                "category"
            )

    print(
        "Make sure the following columns are categorical: ",
    )
    print(", ".join(categorical_variables))

    # Columns to exclude from standardization
    exclude_cols = [
        "demo_sex_v2",
        "img_device_label",
        "rel_family_id",
        # "demo_comb_income_v2",
    ]

    print(
        "Excluding the following columns from standardisation: ",
        ", ".join(exclude_cols),
    )

    # Get columns to scale (everything else)
    cols_to_scale = [
        col for col in mri_all_features_with_prs.columns if col not in exclude_cols
    ]

    # Standardize selected columns
    scaler = StandardScaler()

    mri_all_features_with_prs[cols_to_scale] = scaler.fit_transform(
        mri_all_features_with_prs[cols_to_scale]
    )

    print("Standardization of continuous variables is error-free, Checked")

    rescaled_mri_all_features_with_prs = mri_all_features_with_prs.copy()

    # This is for performing GLM (for unilateral features)
    rescaled_mri_all_features_with_prs.to_csv(
        Path(
            processed_data_path,
            f"mri_all_features_with_prs_rescaled-{wave}.csv",
        ),
        index=True,
    )

    print("Rescaled imaging features with PRS saved to CSV")

    print(
        f"Final Sample size for wave:{wave}",
        rescaled_mri_all_features_with_prs.shape[0],
    )

    ### Create long-form data for left and right hemisphere features
    # Identify left/right hemisphere columns

    print("Creating long-form data for left and right hemisphere features")

    lh_columns = [
        col for col in rescaled_mri_all_features_with_prs.columns if col.endswith("lh")
    ]

    print("Number of left hemisphere features: ", len(lh_columns))

    rh_columns = [
        col for col in rescaled_mri_all_features_with_prs.columns if col.endswith("rh")
    ]

    print("Number of right hemisphere features: ", len(rh_columns))

    # Identify all other columns (covariates, unilateral features, PRS.)
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

    print("Creating long-form data is error-free, Checked")

    # %%

    # Save (index already captured in column)
    processed_data_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    long_data.to_csv(
        Path(
            processed_data_path,
            f"mri_all_features_with_prs_long_rescaled-{wave}.csv",
        ),
        index=True,
    )

    print("Long-form imaging features with PRS saved to CSV")
    # %%
    ### Now select the columns that are the phenotypes of interest for each modality

    print("Selecting features of interest for each modality")

    ### Remove global features for all modality
    print("Removing global features for each modality")

    print("Cortical thickness global features:")
    print(list(t1w_cortical_thickness_pass.columns[-3:]))

    t1w_cortical_thickness_rois = list(t1w_cortical_thickness_pass.columns[1:-3])

    # For cortical volume

    print("Cortical volume global features:")
    print(list(t1w_cortical_volume_pass.columns[-3:]))
    t1w_cortical_volume_rois = list(t1w_cortical_volume_pass.columns[1:-3])

    # For surface area

    print("Cortical surface area global features:")
    print(list(t1w_cortical_surface_area_pass.columns[-3:]))
    t1w_cortical_surface_area_rois = list(t1w_cortical_surface_area_pass.columns[1:-3])

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

    print("Subcortical volume global features:")
    print(global_subcortical_features)

    # FA global features
    global_FA_features = [
        "FA_all_dti_atlas_tract_fibers",
        "FA_hemisphere_dti_atlas_tract_fibers_without_corpus_callosumrh",
        "FA_hemisphere_dti_atlas_tract_fibers_without_corpus_callosumlh",
        "FA_hemisphere_dti_atlas_tract_fibersrh",
        "FA_hemisphere_dti_atlas_tract_fiberslh",
    ]

    print("FA global features:")
    print(global_FA_features)

    # MD global features
    global_MD_features = [
        "MD_all_dti_atlas_tract_fibers",
        "MD_hemisphere_dti_atlas_tract_fibers_without_corpus_callosumrh",
        "MD_hemisphere_dti_atlas_tract_fibers_without_corpus_callosumlh",
        "MD_hemisphere_dti_atlas_tract_fibersrh",
        "MD_hemisphere_dti_atlas_tract_fiberslh",
    ]

    print("MD global features:")
    print(global_MD_features)

    # Step 2: Select subcortical ROIs
    t1w_subcortical_volume_rois = [
        col
        for col in t1w_subcortical_volume_pass.columns
        if col not in global_subcortical_features and col != "eventname"
    ]

    # For tract features

    FA_rois = [
        col
        for col in dmir_fractional_anisotropy_pass.columns
        if col not in global_FA_features and col != "eventname"
    ]

    MD_rois = [
        col
        for col in dmir_mean_diffusivity_pass.columns
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
            f for f in feature_list if (not f.endswith("lh") and not f.endswith("rh"))
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

    print(
        "Creating features of interest for repeated effects modeling is error-free, Checked"
    )

    print("Number of features for each modality:")
    for modality, features in features_of_interest.items():
        print(f"{modality}: {len(features)} features")

    features_for_repeated_effects_path = Path(
        processed_data_path,
        "features_of_interest.json",
    )

    with open(features_for_repeated_effects_path, "w") as f:
        json.dump(features_of_interest, f)


if __name__ == "__main__":
    all_img_waves = [
        "baseline_year_1_arm_1",
        "2_year_follow_up_y_arm_1",
        "4_year_follow_up_y_arm_1",
    ]

    # Process all waves
    for wave in all_img_waves:
        preprocess(wave=wave)
