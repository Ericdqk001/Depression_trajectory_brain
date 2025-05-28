import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def perform_glm(
    wave: str = "baseline_year_1_arm_1",
    experiment_number: int = 1,
):
    data_path = Path(
        "data",
        "poppy",
        f"mri_all_features_with_prs_rescaled-{wave}.csv",
    )

    features_json_path = Path(
        "data",
        "poppy",
        "features_of_interest.json",
    )

    results_path = Path(
        "src",
        "poppy",
        "experiments",
        f"exp_{experiment_number}",
    )

    # === Load data ===
    df = pd.read_csv(
        data_path,
        low_memory=False,
    )

    with open(features_json_path, "r") as f:
        feature_dict = json.load(f)

    df["demo_sex_v2"] = df["demo_sex_v2"].astype("category")
    df["img_device_label"] = df["img_device_label"].astype("category")

    prs_variable = "aoDEP_SBayesR"

    modalities = [
        "unilateral_subcortical_features",
        "unilateral_tract_FA",
        "unilateral_tract_MD",
    ]

    # === Run GLMs ===
    glm_results = []

    for modality in modalities:
        fixed_effects = [
            "interview_age",
            "age2",
            "demo_sex_v2",
            "img_device_label",
            "pc1",
            "pc2",
            "pc3",
            "pc4",
            "pc5",
            "pc6",
        ]

        if modality == "unilateral_subcortical_features":
            fixed_effects.append("smri_vol_scs_intracranialv")

        elif modality == "unilateral_tract_FA":
            fixed_effects.append("FA_all_dti_atlas_tract_fibers")

        elif modality == "unilateral_tract_MD":
            fixed_effects.append("MD_all_dti_atlas_tract_fibers")

        features = feature_dict[modality]

        for feature in features:
            if feature in df.columns:
                formula = f"{feature} ~ {prs_variable} + {' + '.join(fixed_effects)}"
                model = smf.ols(formula=formula, data=df).fit()

                for effect in [prs_variable]:
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

    print("GLM results saved to:")
    print(results_path / f"unilateral_features_glm_results-{wave}.csv")

    # === For bilateral features with significant hemi interaction terms ===

    # Load the features
    sig_interaction_terms_path = Path(
        results_path,
        f"sig_interaction_terms-{wave}.json",
    )

    with open(sig_interaction_terms_path, "r") as f:
        sig_interaction_terms = json.load(f)

    sig_hemi_glm_results = []

    for modality in sig_interaction_terms.keys():
        fixed_effects = [
            "interview_age",
            "age2",
            "demo_sex_v2",
            "img_device_label",
            "pc1",
            "pc2",
            "pc3",
            "pc4",
            "pc5",
            "pc6",
        ]

        if modality == "bilateral_cortical_thickness":
            fixed_effects.append("smri_thick_cdk_mean")

        elif modality == "bilateral_cortical_surface_area":
            fixed_effects.append("smri_area_cdk_total")

        elif modality == "bilateral_cortical_volume":
            fixed_effects.append("smri_vol_scs_intracranialv")

        elif modality == "bilateral_subcortical_volume":
            fixed_effects.append("smri_vol_scs_intracranialv")

        elif modality == "bilateral_tract_FA":
            fixed_effects.append("FA_all_dti_atlas_tract_fibers")

        elif modality == "bilateral_tract_MD":
            fixed_effects.append("MD_all_dti_atlas_tract_fibers")

        features = sig_interaction_terms[modality]

        if features:
            hemi_suffix = [
                "lh",
                "rh",
            ]

            for feature in features:
                for hemi in hemi_suffix:
                    feature_with_hemi = f"{feature}{hemi}"

                    # Remove prefix from bilateral features
                    feature_with_hemi = feature_with_hemi.replace("img_", "")

                    if feature_with_hemi in df.columns:
                        formula = f"{feature_with_hemi} ~ {prs_variable} + {' + '.join(fixed_effects)}"

                        model = smf.ols(formula=formula, data=df).fit()

                        for effect in [prs_variable]:
                            coef = model.params[effect]
                            pval = model.pvalues[effect]
                            ci_low, ci_high = model.conf_int().loc[effect].values

                            sig_hemi_glm_results.append(
                                {
                                    "modality": modality,
                                    "feature": feature_with_hemi,
                                    "predictor": effect,
                                    "coefficient": coef,
                                    "p_value": pval,
                                    "CI_lower": ci_low,
                                    "CI_upper": ci_high,
                                }
                            )
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
    print("Significant hemisphere features GLM results saved to:")
    print(results_path / f"sig_hemi_features_glm_results-{wave}.csv")
    print("GLM analysis completed.")
