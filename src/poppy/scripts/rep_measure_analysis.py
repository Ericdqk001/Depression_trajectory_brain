import json
import logging
import warnings
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

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
            formula = (
                f"{feature} ~ hemisphere * {prs_variable} + {' + '.join(fixed_effects)}"
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

                except Exception as e:
                    logging.error(
                        f"Model failed for {feature} in {modality} for {wave}: {e}"
                    )
                    continue

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
