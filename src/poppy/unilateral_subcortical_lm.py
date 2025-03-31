import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

# === Paths ===
data_path = Path(
    "data",
    "poppy",
    "mri_all_features_with_prs_rescaled.csv",
)

features_json_path = Path(
    "data",
    "poppy",
    "bilateral_features.json",
)

results_path = Path(
    "src",
    "poppy",
    "analysis_results",
)

results_path.mkdir(parents=True, exist_ok=True)

# === Load data ===
df = pd.read_csv(data_path, low_memory=False)

with open(features_json_path, "r") as f:
    feature_dict = json.load(f)

# === Set categorical variables ===
df["demo_sex_v2"] = df["demo_sex_v2"].astype("category")
df["label_site"] = df["label_site"].astype("category")

# === Define variables ===
prs_variable = "aoDEP_SBayesR"
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

# === List of modalities to loop through ===
modalities = [
    "unilateral_subcortical_features",
    "unilateral_tract_FA",
    "unilateral_tract_MD",
]

# === Run GLMs ===
glm_results = []

for modality in modalities:
    features = feature_dict.get(modality, [])
    for feature in features:
        if feature in df.columns:
            formula = f"{feature} ~ {prs_variable} + {' + '.join(fixed_effects)}"
            model = smf.ols(formula=formula, data=df).fit()

            for effect in [prs_variable]:
                coef = model.params[effect]
                pval = model.pvalues[effect]
                ci_low, ci_high = model.conf_int().loc[effect].values

                glm_results.append(
                    {
                        "modality": modality,
                        "feature": feature,
                        "effect": effect,
                        "coefficient": coef,
                        "p_value": pval,
                        "CI_lower": ci_low,
                        "CI_upper": ci_high,
                    }
                )

# === Save results ===
glm_results_df = pd.DataFrame(glm_results)
glm_results_df.to_csv(
    results_path / "unilateral_features_glm_results.csv",
    index=False,
)

print("GLM results saved to:")
print(results_path / "unilateral_features_glm_results.csv")
