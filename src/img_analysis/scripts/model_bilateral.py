import json
import logging
from pathlib import Path

import pandas as pd
from pymer4.models import Lmer

logging.basicConfig(
    filename="model_fitting.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def perform_repeated_measures_analysis(wave: str = "baseline_year_1_arm_1"):
    # Define the brain modalities
    modalities = [
        "bilateral_cortical_thickness",
        "bilateral_cortical_volume",
        "bilateral_cortical_surface_area",
        "bilateral_subcortical_volume",
        "bilateral_tract_FA",
        "bilateral_tract_MD",
    ]

    # File paths
    features_path = Path(
        "data",
        "processed_data",
        f"mri_all_features_with_traj_long_rescaled-{wave}.csv",
    )

    feature_sets_path = Path(
        "data",
        "processed_data",
        "features_of_interest.json",
    )

    # Load imaging and covariate data
    features_df = pd.read_csv(
        features_path,
        low_memory=False,
    )

    # Load feature sets for each modality
    with open(feature_sets_path, "r") as f:
        feature_sets = json.load(f)

    # Convert categorical variables
    for col in [
        "class_label",
        "demo_sex_v2",
        "img_device_label",
        "hemisphere",
        "site_id_l",
        "rel_family_id",
    ]:
        features_df[col] = features_df[col].astype(str).astype("category")

    # Effects of interest
    effects_of_interest = [
        "class_label1",
        "class_label2",
        "class_label3",
        "hemisphereRight:class_label1",
        "hemisphereRight:class_label2",
        "hemisphereRight:class_label3",
    ]
    # Store results here
    results_list = []

    # Loop over each modality
    for modality in modalities:
        print(f"🔍 Processing modality: {modality}")
        roi_list = feature_sets[modality]

        # Fixed effects to include
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

        for feature in roi_list[:5]:
            if feature not in features_df.columns:
                print(f"Feature {feature} not found in columns for {wave}")
                continue
            print(f"🧠 Fitting model for feature: {feature}")
            formula = f"{feature} ~ hemisphere * class_label + {' + '.join(fixed_effects)} + (1|src_subject_id) + (1|site_id_l:rel_family_id)"
            try:
                model = Lmer(formula, data=features_df)
                model.fit(summarize=False)
            except Exception as e:
                print(f"Model failed for {feature} in {modality} for {wave}: {e}")
                logging.error(
                    f"Model failed for {feature} in {modality} for {wave}: {e}"
                )
                continue

            # Save both main trajectory class effect and hemisphere interaction
            coefs = model.coefs

            # print(coefs)
            # Filter by index (effect names)
            coefs = coefs.loc[coefs.index.isin(effects_of_interest)].copy()
            coefs = coefs.reset_index().rename(columns={"index": "effect_name"})
            coefs["modality"] = modality
            coefs["feature"] = feature

            results_list.append(coefs)
            print(results_list)

    # Create results directory
    results_path = Path(
        "src",
        "img_analysis",
        "analysis_results",
    )

    results_path.mkdir(parents=True, exist_ok=True)

    # Save results as CSV
    results_df = pd.concat(results_list, ignore_index=True)
    results_df.to_csv(
        Path(
            results_path,
            f"repeated_bilateral_traj_results-{wave}.csv",
        ),
        index=False,
    )

    print(f"Repeated analysis complete for {wave}. Results saved to:")
    print(results_path / f"repeated_bilateral_traj_results-{wave}.csv")


if __name__ == "__main__":
    wave = "2_year_follow_up_y_arm_1"

    perform_repeated_measures_analysis(wave=wave)
