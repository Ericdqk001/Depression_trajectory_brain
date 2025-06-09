import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def perform_glm(
    wave: str = "baseline_year_1_arm_1",
    experiment_number: int = 3,
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

    prs_variable = "score"

    modalities = [
        "unilateral_subcortical_features",
        "unilateral_tract_FA",
        "unilateral_tract_MD",
    ]

    # === Run GLMs ===
    glm_results = []

    for modality in modalities:
        print("Performing unilateral GLM for modality:", modality)

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

        if modality == "unilateral_subcortical_features":
            fixed_effects.append("smri_vol_scs_intracranialv")

        elif modality == "unilateral_tract_FA":
            fixed_effects.append("FA_all_dti_atlas_tract_fibers")

        elif modality == "unilateral_tract_MD":
            fixed_effects.append("MD_all_dti_atlas_tract_fibers")

        features = feature_dict[modality]

        for feature in features:
            print(f"Processing feature: {feature}")

            formula = f"{feature} ~ {prs_variable} + {' + '.join(fixed_effects)}"
            model = smf.ols(formula=formula, data=features_df).fit()

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

    print(
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

        print(f"Processing modality: {modality}")

        if not features:
            print("No significant hemi interaction features found")
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

        if features:
            hemi_suffix = [
                "lh",
                "rh",
            ]

            for feature in features:
                print(f"Processing feature: {feature}")

                for hemi in hemi_suffix:
                    feature_with_hemi = f"{feature}{hemi}"

                    # Remove prefix from bilateral features
                    feature_with_hemi = feature_with_hemi.replace("img_", "")

                    formula = f"{feature_with_hemi} ~ {prs_variable} + {' + '.join(fixed_effects)}"

                    model = smf.ols(formula=formula, data=features_df).fit()

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


if __name__ == "__main__":
    wave = "2_year_follow_up_y_arm_1"

    experiment_number = 4
    perform_glm(
        wave=wave,
        experiment_number=experiment_number,
    )
