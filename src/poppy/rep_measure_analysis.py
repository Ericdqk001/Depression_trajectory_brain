import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

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
    "mri_all_features_with_prs_long_rescaled.csv",
)

feature_sets_path = Path(
    "data",
    "poppy",
    "bilateral_features.json",
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
features_df["label_site"] = features_df["label_site"].astype("category")

# PRS variable of interest
prs_variable = "aoDEP_SBayesR"

# Fixed effects to include
fixed_effects = [
    "interview_age",
    "age2",
    "demo_sex_v2",
    "label_site",
    "pc1",
    "pc2",
    "pc3",
    "pc4",
    "pc5",
    "pc6",
]

# Store results here
results_list = []

# Loop over each modality
for modality in modalities:
    print(f"Processing {modality}")
    roi_list = feature_sets[modality]

    for feature in roi_list:
        print(f"Fitting model for: {feature}")

        # Mixed-effects model formula with PRS x hemisphere interaction
        formula = (
            f"{feature} ~ hemisphere * {prs_variable} + {' + '.join(fixed_effects)}"
        )

        model = smf.mixedlm(
            formula=formula,
            data=features_df,
            groups=features_df["src_subject_id"],
            # vc_formula={"label_site:rel_family_id": "1"},
        ).fit(method="powell")

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
    "analysis_results",
)
results_path.mkdir(parents=True, exist_ok=True)

# Save results as CSV
results_df = pd.DataFrame(results_list)
results_df.to_csv(Path(results_path, "repeated_bilateral_prs_results.csv"), index=False)

print("PRS-based repeated measures modeling complete. Results saved to:")
print(results_path / "repeated_bilateral_prs_results.csv")
