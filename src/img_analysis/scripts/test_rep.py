import json
from pathlib import Path

import pandas as pd
from pymer4.models import Lmer

wave = "baseline_year_1_arm_1"
modalities = [
    "bilateral_cortical_thickness",
    "bilateral_cortical_volume",
    "bilateral_cortical_surface_area",
    "bilateral_subcortical_volume",
    "bilateral_tract_FA",
    "bilateral_tract_MD",
]

features_path = Path(
    "data", "processed_data", f"mri_all_features_with_traj_long_rescaled-{wave}.csv"
)
feature_sets_path = Path("data", "processed_data", "features_of_interest.json")

features_df = pd.read_csv(features_path, low_memory=False)

with open(feature_sets_path, "r") as f:
    feature_sets = json.load(f)

for col in [
    "class_label",
    "demo_sex_v2",
    "img_device_label",
    "hemisphere",
    "site_id_l",
    "rel_family_id",
]:
    features_df[col] = features_df[col].astype(str).astype("category")

results_list = []
effects_of_interest = [
    "class_label1.0",
    "class_label2.0",
    "class_label3.0",
    "hemisphereRight:class_label1.0",
    "hemisphereRight:class_label2.0",
    "hemisphereRight:class_label3.0",
]

for modality in modalities:
    print(f"🔍 Processing modality: {modality}")
    roi_list = feature_sets[modality]
    feature = roi_list[0]  # only first feature

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

    print(f"🧠 Fitting model for feature: {feature}")
    formula = f"{feature} ~ hemisphere * class_label + {' + '.join(fixed_effects)} + (1|src_subject_id) + (1|site_id_l:rel_family_id)"
    model = Lmer(formula, data=features_df)
    model.fit(summarize=False)

    coefs = model.coefs
    coefs = coefs.loc[coefs.index.isin(effects_of_interest)].copy()
    coefs = coefs.reset_index().rename(columns={"index": "effect_name"})
    coefs["modality"] = modality
    coefs["feature"] = feature

    print(f"\n--- Coefs for feature: {feature} ---\n{coefs}\n")
    results_list.append(coefs)

print("\n=== All results_list (list of DataFrames) ===\n")
for df in results_list:
    print(df)
    print("\n" + "-" * 80 + "\n")


# Save the results to a CSV file
results_df = pd.concat(results_list, ignore_index=False)
results_df.to_csv("results.csv", index=False)
print("\nResults saved to results.csv")
