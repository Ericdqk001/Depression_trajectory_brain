import json
import logging
import warnings
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def perform_glm(
    wave: str = "baseline_year_1_arm_1",
    version_name: str = "",
    experiment_number: int = 1,
    predictor="score",
):
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

    processed_data_path = Path(
        analysis_root_path,
        version_name,
        "processed_data",
    )

    data_path = Path(
        processed_data_path,
        f"mri_all_features_with_prs_rescaled-{wave}.csv",
    )

    features_json_path = Path(
        processed_data_path,
        "features_of_interest.json",
    )

    experiments_path = Path(
        analysis_root_path,
        version_name,
        "experiments",
    )

    if not experiments_path.exists():
        experiments_path.mkdir(parents=True, exist_ok=True)

    results_path = Path(
        experiments_path,
        f"exp_{experiment_number}",
    )

    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)

    # === Load data ===
    features_df = pd.read_csv(
        data_path,
        low_memory=False,
    )

    with open(features_json_path, "r") as f:
        feature_dict = json.load(f)

    # Set categorical variables
    features_df["demo_sex_v2"] = features_df["demo_sex_v2"].astype("category")
    features_df["img_device_label"] = features_df["img_device_label"].astype("category")
    features_df["src_subject_id"] = features_df["src_subject_id"].astype("category")
    # features_df["demo_comb_income_v2"] = features_df["demo_comb_income_v2"].astype(
    #     "category"
    # )

    modalities = [
        "unilateral_subcortical_features",
        "unilateral_tract_FA",
        "unilateral_tract_MD",
    ]

    # === Run GLMs ===
    glm_results = []

    for modality in modalities:
        logging.info("Performing unilateral GLM for modality: %s", modality)

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
            # "C(demo_comb_income_v2)",
        ]

        # if modality == "unilateral_subcortical_features":
        #     fixed_effects.append("smri_vol_scs_intracranialv")

        # elif modality == "unilateral_tract_FA":
        #     fixed_effects.append("FA_all_dti_atlas_tract_fibers")

        # elif modality == "unilateral_tract_MD":
        #     fixed_effects.append("MD_all_dti_atlas_tract_fibers")

        features = feature_dict[modality]

        for feature in features:
            logging.info("Processing feature: %s", feature)

            formula = f"{feature} ~ {predictor} + {' + '.join(fixed_effects)}"
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                try:
                    model = smf.ols(formula=formula, data=features_df).fit()
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

            for effect in [predictor]:
                coef = model.params[effect]
                pval = model.pvalues[effect]
                ci_low, ci_high = model.conf_int().loc[effect].values

                glm_results.append(
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

    # === Save results ===
    glm_results_df = pd.DataFrame(glm_results)
    glm_results_df.to_csv(
        Path(
            results_path,
            f"unilateral_features_glm_results-{wave}.csv",
        ),
        index=False,
    )

    logging.info("GLM results saved to:")
    logging.info(results_path / f"unilateral_features_glm_results-{wave}.csv")

    # === For bilateral features with significant hemi interaction terms ===

    logging.info(
        "Performing GLM for bilateral features with significant hemi interaction terms..."
    )

    # Load the features
    sig_interaction_terms_path = Path(
        results_path,
        f"sig_interaction_terms-{wave}.json",
    )

    with open(sig_interaction_terms_path, "r") as f:
        sig_interaction_terms = json.load(f)

    sig_hemi_glm_results = []

    for modality in sig_interaction_terms.keys():
        features = sig_interaction_terms[modality]

        logging.info(f"Processing modality: {modality}")

        if not features:
            logging.info("No significant hemi interaction features found")
            continue

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
            # "C(demo_comb_income_v2)",
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

        if features:
            hemi_suffix = [
                "lh",
                "rh",
            ]

            for feature in features:
                logging.info(f"Processing feature: {feature}")
                for hemi in hemi_suffix:
                    feature_with_hemi = f"{feature}{hemi}"
                    feature_with_hemi = feature_with_hemi.replace("img_", "")
                    formula = f"{feature_with_hemi} ~ {predictor} + {' + '.join(fixed_effects)}"
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always", ConvergenceWarning)
                        try:
                            model = smf.ols(formula=formula, data=features_df).fit()
                            for warning in w:
                                if issubclass(warning.category, ConvergenceWarning):
                                    logging.warning(
                                        f"Convergence warning for {feature_with_hemi} in {modality} for {wave}: {warning.message}"
                                    )
                                    print(
                                        f"Convergence warning for {feature_with_hemi} in {modality} for {wave}: {warning.message}"
                                    )
                        except Exception as e:
                            logging.error(
                                f"Model failed for {feature_with_hemi} in {modality} for {wave}: {e}"
                            )
                            print(
                                f"Model failed for {feature_with_hemi} in {modality} for {wave}: {e}"
                            )
                            continue
    # === Save results ===

    columns = [
        "modality",
        "feature",
        "predictor",
        "coefficient",
        "p_value",
        "CI_lower",
        "CI_upper",
    ]

    sig_hemi_glm_results_df = pd.DataFrame(sig_hemi_glm_results, columns=columns)
    sig_hemi_glm_results_df.to_csv(
        Path(
            results_path,
            f"sig_hemi_features_glm_results-{wave}.csv",
        ),
        index=False,
    )
    logging.info("Significant hemisphere features GLM results saved to:")
    logging.info(results_path / f"sig_hemi_features_glm_results-{wave}.csv")
    logging.info("GLM analysis completed.")


if __name__ == "__main__":
    pass
