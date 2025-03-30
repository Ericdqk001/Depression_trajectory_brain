import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

# Paths
data_path = Path(
    "data",
    "poppy",
    "mri_all_features_with_prs_rescaled.csv",
)

features_json_path = Path(
    "data",
    "poppy",
    "features_for_repeated_effects.json",
)

results_path = Path(
    "src",
    "poppy",
    "analysis_results",
)

results_path.mkdir(
    parents=True,
    exist_ok=True,
)

# Load data
df = pd.read_csv(data_path, low_memory=False)

with open(features_json_path, "r") as f:
    feature_dict = json.load(f)

# Get list of unilateral subcortical features
unilateral_features = feature_dict.get("unilateral_subcortical_features", [])

# Set categorical variables
df["demo_sex_v2"] = df["demo_sex_v2"].astype("category")
df["label_site"] = df["label_site"].astype("category")

# Predictor
prs_variable = "aoDEP_SBayesR"

# Fixed effects
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

# Run GLMs
glm_results = []

for feature in unilateral_features:
    if feature in df.columns:
        formula = f"{feature} ~ {prs_variable} + {' + '.join(fixed_effects)}"
        model = smf.ols(formula=formula, data=df).fit()

        for effect in model.params.index:
            coef = model.params[effect]
            pval = model.pvalues[effect]
            ci_low, ci_high = model.conf_int().loc[effect].values

            glm_results.append(
                {
                    "modality": "unilateral_subcortical_volume",
                    "feature": feature,
                    "effect": effect,
                    "coefficient": coef,
                    "p_value": pval,
                    "CI_lower": ci_low,
                    "CI_upper": ci_high,
                }
            )

# Save results
glm_results_df = pd.DataFrame(glm_results)
glm_results_df.to_csv(
    Path(
        results_path,
        "unilateral_subcortical_glm_results.csv",
    ),
    index=False,
)

print("Unilateral subcortical GLM results saved to:")
print(results_path / "unilateral_subcortical_glm_results.csv")
