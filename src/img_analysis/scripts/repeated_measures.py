import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

# TODO Rescale variables (done in the preprocessing)
# TODO Add BMI

# Define modalities
modalities = [
    "cortical_thickness",
    "cortical_volume",
    "cortical_surface_area",
]

# File paths
cortical_features_path = Path(
    "data",
    "processed_data",
    "t1w_all_cortical_features_cov_traj_long.csv",
)
cortical_feature_sets_path = Path(
    "data",
    "processed_data",
    "unilateral_brain_features.json",
)

# Load data
cortical_features = pd.read_csv(cortical_features_path)

with open(cortical_feature_sets_path, "r") as f:
    cortical_feature_sets = json.load(f)


# Set as categorical
cortical_features["class_label"] = cortical_features["class_label"].astype("category")
cortical_features["label_site"] = cortical_features["label_site"].astype("category")
cortical_features["rel_family_id"] = cortical_features["rel_family_id"].astype(
    "category"
)
cortical_features["demo_sex_v2"] = cortical_features["demo_sex_v2"].astype("category")

# Effects of interest
effects_to_save = [
    "C(class_label)[T.1.0]",
    "C(class_label)[T.2.0]",
    "C(class_label)[T.3.0]",
    "hemisphere[T.Right]:C(class_label)[T.1.0]",
    "hemisphere[T.Right]:C(class_label)[T.2.0]",
    "hemisphere[T.Right]:C(class_label)[T.3.0]",
]

results_list = []

# Iterate through each modality
for modality in modalities:
    print(f"Processing modality: {modality}")

    # Get the list of imaging features for the current modality
    imaging_features = cortical_feature_sets[modality]

    # Store results for the modality
    modality_results = []

    # Define covariates
    fixed_effect = [
        "demo_sex_v2",
        "interview_age",
        "age2",
        "race_ethnicity",
        "demo_comb_income_v2",
        "smri_vol_scs_intracranialv",
    ]

    if modality == "cortical_thickness":

        fixed_effect.remove("smri_vol_scs_intracranialv")
        fixed_effect.append("smri_thick_cdk_mean")

    if modality == "cortical_surface_area":

        fixed_effect.remove("smri_vol_scs_intracranialv")
        fixed_effect.append("smri_area_cdk_total")

    # Loop through each feature in the modality
    for feature in imaging_features:
        print(f"Fitting model for {feature}")

        # Define the model formula dynamically
        formula = (
            f"{feature} ~ hemisphere * C(class_label) + {' + '.join(fixed_effect)}"
        )

        # Fit the mixed-effects model
        model = smf.mixedlm(
            formula=formula,
            data=cortical_features,
            groups=cortical_features["src_subject_id"],
            # Random intercept for family nested within sites
            vc_formula={"label_site:rel_family_id": "1"},
        ).fit()

        # Extract required statistics
        for effect in effects_to_save:
            if effect in model.params.index:
                coef = model.params[effect]
                p_value = model.pvalues[effect]
                conf_int = model.conf_int().loc[effect].values  # 95% CI

                # Append results
                results_list.append(
                    {
                        "modality": modality,
                        "feature": feature,
                        "effect": effect,
                        "coefficient": coef,
                        "p_value": p_value,
                        "CI_lower": conf_int[0],
                        "CI_upper": conf_int[1],
                    }
                )

results_path = Path(
    "img_analysis_results",
    "repeated_measures",
)

results_path.mkdir(
    parents=True,
    exist_ok=True,
)

results_df = pd.DataFrame(results_list)
results_df.to_csv(
    Path(
        results_path,
        "repeated_regional_results.csv",
    ),
    index=False,
)
