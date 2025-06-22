import json
import logging
import warnings
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def perform_repeated_measures_analysis(
    wave: str = "baseline_year_1_arm_1",
    experiment_number: int = 1,
    version_name: str = "CBCL_replication_test",
    predictor: str = "CBCL_quant",
):
    # Define the brain modalities
    modalities = [
        "bilateral_cortical_thickness",
        "bilateral_cortical_volume",
        "bilateral_cortical_surface_area",
        "bilateral_subcortical_volume",
        "bilateral_tract_FA",
        "bilateral_tract_MD",
    ]

    data_store_path = Path(
        "/",
        "Volumes",
        "GenScotDepression",
    )

    if data_store_path.exists():
        print("Mounted data store path: ", data_store_path)
        logging.info("Mounted data store path: %s", data_store_path)

    analysis_root_path = Path(
        data_store_path,
        "users",
        "Eric",
        "poppy_neuroimaging",
    )

    processed_data_path = Path(
        analysis_root_path,
        version_name,
        "processed_data",
    )

    # File paths
    features_path = Path(
        processed_data_path,
        f"mri_all_features_with_prs_long_rescaled-{wave}.csv",
    )

    feature_sets_path = Path(
        processed_data_path,
        "features_of_interest.json",
    )

    # Load imaging and covariate data
    features_df = pd.read_csv(
        features_path,
        low_memory=False,
    )

    features_df = features_df.reset_index(drop=True)

    # Load feature sets for each modality
    with open(feature_sets_path, "r") as f:
        feature_sets = json.load(f)

    # Set categorical variables
    features_df["demo_sex_v2"] = features_df["demo_sex_v2"].astype("category")
    features_df["img_device_label"] = features_df["img_device_label"].astype("category")
    features_df["hemisphere"] = features_df["hemisphere"].astype("category")
    features_df["src_subject_id"] = features_df["src_subject_id"].astype("category")
    # features_df["demo_comb_income_v2"] = features_df["demo_comb_income_v2"].astype(
    #     "category"
    # )

    # PRS variable of interest

    print("The PRS variable of interest is:", predictor)
    logging.info("The PRS variable of interest is: %s", predictor)

    # Store results here
    results_list = []

    # Loop over each modality
    for modality in modalities:
        print(f"Processing {modality}")
        logging.info("Processing %s", modality)
        roi_list = feature_sets[modality]

        # Fixed effects to include
        fixed_effects = [
            "interview_age",
            "age2",
            "C(demo_sex_v2)",
            "C(img_device_label)",
            "pc1",
            "pc2",
            "pc3",
            "pc4",
            "pc5",
            "pc6",
        ]

        # if modality == "bilateral_cortical_thickness":
        #     fixed_effects.append("smri_thick_cdk_mean")

        # elif modality == "bilateral_cortical_surface_area":
        #     fixed_effects.append("smri_area_cdk_total")

        # elif modality == "bilateral_cortical_volume":
        #     fixed_effects.append("smri_vol_scs_intracranialv")

        # elif modality == "bilateral_subcortical_volume":
        #     fixed_effects.append("smri_vol_scs_intracranialv")

        # elif modality == "bilateral_tract_FA":
        #     fixed_effects.append("FA_all_dti_atlas_tract_fibers")

        # elif modality == "bilateral_tract_MD":
        #     fixed_effects.append("MD_all_dti_atlas_tract_fibers")

        for feature in roi_list:
            print(f"Fitting model for: {feature}")
            logging.info("Fitting model for: %s", feature)

            # Mixed-effects model formula with PRS x hemisphere interaction
            formula = (
                f"{feature} ~ C(hemisphere) * {predictor} + {' + '.join(fixed_effects)}"
            )

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)

                try:
                    model = smf.mixedlm(
                        formula=formula,
                        data=features_df,
                        groups=features_df["src_subject_id"],
                    ).fit()

                    # Log any convergence warnings
                    for warning in w:
                        if issubclass(warning.category, ConvergenceWarning):
                            logging.warning(
                                f"Convergence warning for {feature} in {modality} for {wave}: {warning.message}"
                            )
                            print(
                                f"Convergence warning for {feature} in {modality} for {wave}: {warning.message}"
                            )

                except Exception as e:
                    logging.error(
                        f"Model failed for {feature} in {modality} for {wave}: {e}"
                    )
                    print(f"Model failed for {feature} in {modality} for {wave}: {e}")
                    continue

            # Save both main PRS effect and hemisphere interaction
            for effect in [predictor, f"C(hemisphere)[T.Right]:{predictor}"]:
                if effect in model.params.index:
                    coef = model.params[effect]
                    pval = model.pvalues[effect]
                    ci_low, ci_high = model.conf_int().loc[effect].values

                    results_list.append(
                        {
                            "modality": modality,
                            "feature": feature,
                            "predictor": effect,
                            "coefficient": coef,
                            "p_value": pval,
                            "CI_lower": ci_low,
                            "CI_upper": ci_high,
                        }
                    )

    # Create results directory
    results_path = Path(
        analysis_root_path,
        version_name,
        "experiments",
        f"exp_{experiment_number}",
    )

    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)

    # Save results as CSV
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(
        Path(
            results_path,
            f"repeated_bilateral_prs_results-{wave}.csv",
        ),
        index=False,
    )

    print(f"Repeated analysis complete for {wave}. Results saved to:")
    logging.info(f"Repeated analysis complete for {wave}. Results saved to:")
    print(results_path / f"repeated_bilateral_prs_results-{wave}.csv")
    logging.info(results_path / f"repeated_bilateral_prs_results-{wave}.csv")


if __name__ == "__main__":
    # Run the analysis for the baseline year 1 arm 1
    perform_repeated_measures_analysis(wave="baseline_year_1_arm_1")
