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


def perform_repeated_measures_analysis(
    wave: str = "baseline_year_1_arm_1",
    experiment_number: int = 1,
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

    # File paths
    features_path = Path(
        "data",
        "poppy",
        f"mri_all_features_with_prs_long_rescaled-{wave}.csv",
    )

    feature_sets_path = Path(
        "data",
        "poppy",
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

    # Set categorical variables
    features_df["demo_sex_v2"] = features_df["demo_sex_v2"].astype("category")
    features_df["img_device_label"] = features_df["img_device_label"].astype("category")

    # PRS variable of interest

    prs_variable = "SCORESUM"

    print("The PRS variable of interest is:", prs_variable)

    # Store results here
    results_list = []

    # Loop over each modality
    for modality in modalities:
        print(f"Processing {modality}")
        roi_list = feature_sets[modality]

        # Fixed effects to include
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
            "demo_comb_income_v2",
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

        for feature in roi_list:
            print(f"Fitting model for: {feature}")

            # Mixed-effects model formula with PRS x hemisphere interaction
            formula = f"{feature} ~ hemisphere * {prs_variable} + {' + '.join(fixed_effects)} + (1|src_subject_id)"

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
            # print(results_list)

            # Save both main PRS effect and hemisphere interaction
            for effect in [prs_variable, f"hemisphere[T.Right]:{prs_variable}"]:
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
        "src",
        "poppy",
        "experiments",
        f"exp_{experiment_number}",
    )

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
    print(results_path / f"repeated_bilateral_prs_results-{wave}.csv")


if __name__ == "__main__":
    # Run the analysis for the baseline year 1 arm 1
    perform_repeated_measures_analysis(wave="baseline_year_1_arm_1")
