import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def run_multinomial():
    """Run multinomial logistic regression on imaging features.

    For now, structural cortical modalities (thickness, volume, surface area) are
    included. The DV is the trajectory class (low - 0, increasing - 1, decreasing - 2,
    high - 3), so the reference class is the low class. The IVs are the imaging features
    and the covariates specified below.

    """
    # File paths
    cortical_features_path = Path(
        "data",
        "processed_data",
        "t1w_all_cortical_features_cov_traj.csv",
    )

    cortical_feature_sets_path = Path(
        "data",
        "processed_data",
        "brain_features_of_interest.json",
    )

    # Load data
    cortical_features = pd.read_csv(
        cortical_features_path,
        index_col=0,
    )

    with open(cortical_feature_sets_path, "r") as f:
        cortical_feature_sets = json.load(f)

    # Parameters
    modalities = [
        "cortical_thickness",
        "cortical_volume",
        "cortical_surface_area",
    ]

    results_path = Path(
        "src",
        "img_analysis",
        "results",
    )

    dependent_var = "class_label"

    covariates = [
        "demo_sex_v2",
        "label_site",
        "smri_vol_scs_intracranialv",
        "interview_age",
        "age2",
    ]

    # Process each modality
    for modality in modalities:
        print(f"Processing modality: {modality}")
        imaging_features = cortical_feature_sets[modality]
        modality_results = []

        for feature in imaging_features:
            # Build and fit the model
            formula = f"{dependent_var} ~ {feature} + {' + '.join(covariates)}"
            model = sm.MNLogit.from_formula(formula, data=cortical_features).fit(
                disp=False
            )

            # Extract parameters, p-values, and confidence intervals
            params = model.params.loc[feature].values
            pvalues = model.pvalues.loc[feature].values
            conf_intervals = model.conf_int()

            # Combine results into a list of dictionaries
            for class_index, (coef, p_value) in enumerate(
                zip(params, pvalues), start=1
            ):
                # Extract confidence intervals for the specific class and feature
                ci_lower, ci_upper = conf_intervals.loc[
                    (
                        str(class_index),
                        feature,
                    ),
                    :,
                ]
                modality_results.append(
                    {
                        "Modality": modality,
                        "Feature": feature,
                        "Class": class_index,
                        "Odds Ratio": round(np.exp(coef), 4),
                        "P-Value": round(p_value, 4),
                        "Confidence Interval (0.025)": round(np.exp(ci_lower), 4),
                        "Confidence Interval (0.975)": round(np.exp(ci_upper), 4),
                        "Significance": p_value < 0.05,
                    }
                )

            # Convert results to a DataFrame
            modality_results_df = pd.DataFrame(modality_results)

            # Save results to a CSV file
            multinomial_output_path = Path(
                results_path,
                "multinomial",
            )

            if not multinomial_output_path.exists():
                multinomial_output_path.mkdir(parents=True)

            output_path = Path(
                multinomial_output_path,
                f"{modality}_results.csv",
            )

            modality_results_df.to_csv(
                output_path,
                index=False,
            )

            print(f"Results for {modality} saved to {output_path}")


if __name__ == "__main__":
    run_multinomial()

# TODO: Figure out the interaction terms