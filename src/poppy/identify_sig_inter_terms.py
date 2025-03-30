import json
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

modalities = [
    "cortical_thickness",
    "cortical_volume",
    "cortical_surface_area",
    "subcortical_volume",
]

interaction_term = "hemisphere[T.Right]:aoDEP_SBayesR"
prs_variable = "aoDEP_SBayesR"

# File paths
features_path = Path(
    "data",
    "poppy",
    "mri_all_features_with_prs_long_rescaled.csv",
)


feature_sets_path = Path(
    "data",
    "poppy",
    "features_for_repeated_effects.json",
)

results_path = Path(
    "src",
    "poppy",
    "analysis_results",
)

results_path.mkdir(parents=True, exist_ok=True)

# Load data
features_df = pd.read_csv(features_path, low_memory=False)
with open(feature_sets_path, "r") as f:
    feature_sets = json.load(f)

# Categorical columns
features_df["demo_sex_v2"] = features_df["demo_sex_v2"].astype("category")
features_df["label_site"] = features_df["label_site"].astype("category")
features_df["rel_family_id"] = features_df["rel_family_id"].astype("category")

# Load results
repeated_results = pd.read_csv(
    Path(
        results_path,
        "repeated_regional_prs_results.csv",
    )
)


significant_features_by_modality = {}

for modality in modalities:
    modality_df = repeated_results[
        (repeated_results["modality"] == modality)
        & (repeated_results["predictor"] == interaction_term)
    ]

    if not modality_df.empty:
        # Apply FDR correction
        pvals = modality_df["p_value"].values
        rejected, qvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

        modality_df = modality_df.copy()
        modality_df["q_value"] = qvals
        modality_df["significant"] = rejected

        # Filter significant
        significant = modality_df[modality_df["significant"]]
        significant_features_by_modality[modality] = significant["feature"].tolist()

        # # Save q-values
        # modality_df.to_csv(
        #     Path(
        #         results_path,
        #         f"{modality}_interaction_qvals.csv",
        #     ),
        #     index=False,
        # )


print(significant_features_by_modality)
