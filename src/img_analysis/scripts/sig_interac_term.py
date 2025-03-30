# Load the data
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

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


# Load the model results

results_path = Path(
    "img_analysis_results",
    "repeated_measures",
)

results_df = pd.read_csv(
    Path(
        results_path,
        "repeated_regional_results.csv",
    )
)

# Identify significant interaction terms (p < 0.05)
interaction_terms = [
    "hemisphere[T.Right]:C(class_label)[T.1.0]",
    "hemisphere[T.Right]:C(class_label)[T.2.0]",
    "hemisphere[T.Right]:C(class_label)[T.3.0]",
]

significant_features = results_df[
    (results_df["effect"].isin(interaction_terms)) & (results_df["p_value"] < 0.05)
]["feature"].unique()

# Save the list of significant features

significant_features_df = pd.DataFrame(significant_features, columns=["feature"])

significant_features_df.to_csv(
    Path(
        results_path,
        "significant_interaction_features.csv",
    ),
    index=False,
)

# Initialize a list to store results
glm_results_list = []

# If significant interactions exist, run separate GLMs for left and right hemispheres
if len(significant_features) > 0:
    modalities = ["cortical_thickness", "cortical_volume", "cortical_surface_area"]

    fixed_effect = [
        "demo_sex_v2",
        "interview_age",
        "age2",
        "race_ethnicity",
        "demo_comb_income_v2",
        "smri_vol_scs_intracranialv",
    ]

    for modality in modalities:
        imaging_features = [
            feat
            for feat in significant_features
            if feat in cortical_feature_sets[modality]
        ]

        fixed_effects_mod = fixed_effect.copy()

        if modality == "cortical_thickness":
            fixed_effects_mod.remove("smri_vol_scs_intracranialv")
            fixed_effects_mod.append("smri_thick_cdk_mean")
        elif modality == "cortical_surface_area":
            fixed_effects_mod.remove("smri_vol_scs_intracranialv")
            fixed_effects_mod.append("smri_area_cdk_total")

        for feature in imaging_features:
            for hemisphere in ["Left", "Right"]:

                hemisphere_data = cortical_features[
                    cortical_features["hemisphere"] == hemisphere
                ].copy()

                formula = (
                    f"{feature} ~ C(class_label) + {' + '.join(fixed_effects_mod)}"
                )

                model = smf.mixedlm(
                    formula=formula,
                    data=hemisphere_data,
                    groups=hemisphere_data["src_subject_id"],
                    vc_formula={"label_site:rel_family_id": "1"},
                ).fit()

                for effect in model.params.index:
                    coef = model.params[effect]
                    p_value = model.pvalues[effect]
                    conf_int = model.conf_int().loc[effect].values

                    glm_results_list.append(
                        {
                            "modality": modality,
                            "hemisphere": hemisphere,
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

# Convert results to DataFrame and save as CSV
glm_results_df = pd.DataFrame(glm_results_list)
glm_results_df.to_csv(
    Path(
        results_path,
        "unilateral_regional_results.csv",
    ),
    index=False,
)
