import json
from pathlib import Path

import pandas as pd
from pymer4.models import Lmer


def perform_unilateral(wave: str = "baseline_year_1_arm_1"):
    data_path = Path(
        "data",
        "processed_data",
        f"mri_all_features_with_traj_rescaled-{wave}.csv",
    )
    features_json_path = Path(
        "data",
        "processed_data",
        "features_of_interest.json",
    )
    results_path = Path(
        "src",
        "img_analysis",
        "analysis_results",
    )
    # === Load data ===
    features_df = pd.read_csv(
        data_path,
        low_memory=False,
    )

    with open(features_json_path, "r") as f:
        feature_dict = json.load(f)

    # Convert categorical variables
    for col in [
        "class_label",
        "demo_sex_v2",
        "img_device_label",
        "site_id_l",
        "rel_family_id",
    ]:
        features_df[col] = features_df[col].astype(str).astype("category")

    target_feature = "class_label"

    modalities = [
        "unilateral_subcortical_features",
        "unilateral_tract_FA",
        "unilateral_tract_MD",
    ]

    # === Run Mixed Effects Models for Unilateral Features ===
    unilateral_results = []
    for modality in modalities:
        fixed_effects = [
            "interview_age",
            "age2",
            "demo_sex_v2",
            "img_device_label",
            "demo_comb_income_v2",
            "BMI_zscore",
        ]
        if modality == "unilateral_subcortical_features":
            fixed_effects.append("smri_vol_scs_intracranialv")
        elif modality == "unilateral_tract_FA":
            fixed_effects.append("FA_all_dti_atlas_tract_fibers")
        elif modality == "unilateral_tract_MD":
            fixed_effects.append("MD_all_dti_atlas_tract_fibers")
        features = feature_dict[modality]

        for feature in features[:5]:
            if feature in features_df.columns:
                formula = f"{feature} ~ {target_feature} + {' + '.join(fixed_effects)} + (1|site_id_l:rel_family_id)"
                print(f"\nFitting Lmer for {feature} in {modality}")
                model = Lmer(formula, data=features_df)
                model.fit(summarize=False)

                coefs = model.coefs.reset_index().rename(
                    columns={"index": "effect_name"}
                )
                coefs["modality"] = modality
                coefs["feature"] = feature
                unilateral_results.append(coefs)

    if unilateral_results:
        glm_results_df = pd.concat(unilateral_results, ignore_index=True)
        glm_results_df.to_csv(
            Path(
                results_path,
                f"unilateral_features_results-{wave}.csv",
            ),
            index=False,
        )
        print("Unilateral features results saved to:")
        print(results_path / f"unilateral_features_results-{wave}.csv")

    # === For bilateral features with significant hemi interaction terms ===
    sig_interaction_terms_path = Path(
        results_path,
        f"sig_interaction_terms-{wave}.json",
    )
    with open(sig_interaction_terms_path, "r") as f:
        sig_interaction_terms = json.load(f)

    sig_hemi_results = []
    for modality in sig_interaction_terms.keys():
        fixed_effects = [
            "interview_age",
            "age2",
            "demo_sex_v2",
            "img_device_label",
            "demo_comb_income_v2",
            "BMI_zscore",
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
            hemi_suffix = ["lh", "rh"]
            for feature in features:
                for hemi in hemi_suffix:
                    feature_with_hemi = f"{feature}{hemi}"
                    feature_with_hemi = feature_with_hemi.replace("img_", "")
                    if feature_with_hemi in features_df.columns:
                        formula = f"{feature_with_hemi} ~ {target_feature} + {' + '.join(fixed_effects)} + (1|site_id_l:rel_family_id)"
                        print(f"\nFitting model for {feature_with_hemi} in {modality}")
                        model = Lmer(formula, data=features_df)
                        model.fit(summarize=False)

                        coefs = model.coefs.reset_index().rename(
                            columns={"index": "effect_name"}
                        )

                        coefs["modality"] = modality
                        coefs["feature"] = feature_with_hemi
                        sig_hemi_results.append(coefs)

    if sig_hemi_results:
        glm_results_df = pd.concat(sig_hemi_results, ignore_index=True)
        glm_results_df.to_csv(
            Path(
                results_path,
                f"sig_hemi_features_results-{wave}.csv",
            ),
            index=False,
        )
        print("sig_hemi_features results saved to:")
        print(results_path / f"sig_hemi_features_results-{modality}-{wave}.csv")


if __name__ == "__main__":
    perform_unilateral()
