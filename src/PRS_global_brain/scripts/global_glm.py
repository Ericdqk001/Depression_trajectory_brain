import logging
import warnings
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def perform_glm(
    wave: str = "baseline_year_1_arm_1",
    version_name: str = "abcd_pgcmdd3",
    experiment_number: int = 1,
    predictor="CBCL_quant",
):
    data_store_path = Path(
        "/",
        "Volumes",
        "GenScotDepression",
    )

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

    print("Performing GLM analysis for wave:", wave)
    logging.info("Performing GLM analysis for wave: %s", wave)

    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)

    # === Load data ===
    features_df = pd.read_csv(
        data_path,
        low_memory=False,
    )

    # Set categorical variables
    features_df["demo_sex_v2"] = features_df["demo_sex_v2"].astype("category")
    features_df["img_device_label"] = features_df["img_device_label"].astype("category")
    features_df["src_subject_id"] = features_df["src_subject_id"].astype("category")
    # features_df["demo_comb_income_v2"] = features_df["demo_comb_income_v2"].astype(
    #     "category"
    # )

    global_brain_feature_map = {
        "cortical_thickness": "smri_thick_cdk_mean",
        "cortical_surface_area": "smri_area_cdk_total",
        "cortical_volume": "smri_vol_scs_intracranialv",
        "tract_FA": "FA_all_dti_atlas_tract_fibers",
        "tract_MD": "MD_all_dti_atlas_tract_fibers",
    }

    # === Run GLMs ===
    glm_results = []

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

    for modality, feature in global_brain_feature_map.items():
        print(f"Performing GLM for global feature: {feature}")
        logging.info(f"Performing GLM for global feature: {feature}")

        formula = f"{feature} ~ {predictor} + {' + '.join(fixed_effects)}"

        print(f"Formula for GLM: {formula}")
        logging.info(f"Formula for GLM: {formula}")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            try:
                model = smf.ols(formula=formula, data=features_df).fit()
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

    glm_results_path = Path(
        results_path,
        f"global_features_glm_results-{wave}.csv",
    )

    glm_results_df.to_csv(
        glm_results_path,
        index=False,
    )

    print(f"GLM results saved to: {results_path / glm_results_path}")
    logging.info(f"GLM results saved to: {results_path / glm_results_path}")


if __name__ == "__main__":
    pass
