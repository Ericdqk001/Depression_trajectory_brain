import json
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests


def identify_sig_inter_terms(wave: str = "baseline_year_1_arm_1"):
    modalities = [
        "bilateral_cortical_thickness",
        "bilateral_cortical_volume",
        "bilateral_cortical_surface_area",
        "bilateral_subcortical_volume",
        "bilateral_tract_FA",
        "bilateral_tract_MD",
    ]

    interaction_term = "hemisphere[T.Right]:aoDEP_SBayesR"

    # File paths
    features_path = Path(
        "data",
        "poppy",
        f"mri_all_features_with_prs_long_rescaled-{wave}.csv",
    )

    results_path = Path(
        "src",
        "poppy",
        "analysis_results",
    )

    # Load data
    features_df = pd.read_csv(features_path, low_memory=False)

    # Categorical columns
    features_df["demo_sex_v2"] = features_df["demo_sex_v2"].astype("category")
    features_df["img_device_label"] = features_df["img_device_label"].astype("category")
    features_df["rel_family_id"] = features_df["rel_family_id"].astype("category")

    # Load results
    repeated_results = pd.read_csv(
        Path(
            results_path,
            f"repeated_bilateral_prs_results-{wave}.csv",
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

    # Print results
    for modality, features in significant_features_by_modality.items():
        print(f"{modality}: {features}")

    # Save results to json
    with open(
        Path(
            results_path,
            f"sig_interaction_terms-{wave}.json",
        ),
        "w",
    ) as f:
        json.dump(significant_features_by_modality, f, indent=4)
