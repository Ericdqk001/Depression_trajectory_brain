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

    interaction_terms = [
        "hemisphere[T.Right]:C(class_label)[T.1.0]",
        "hemisphere[T.Right]:C(class_label)[T.2.0]",
        "hemisphere[T.Right]:C(class_label)[T.3.0]",
    ]

    # File paths
    features_path = Path(
        "data",
        "processed_data",
        f"mri_all_features_with_traj_long_rescaled-{wave}.csv",
    )

    results_path = Path(
        "src",
        "processed_data",
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
            f"repeated_bilateral_traj_results-{wave}.csv",
        )
    )

    significant_features_by_modality = {}

    for modality in modalities:
        # Filter results by modality
        modality_df = repeated_results[repeated_results["modality"] == modality]

        modality_significant = []

        for interaction_term in interaction_terms:
            inter_df = modality_df[modality_df["predictor"] == interaction_term]

            if not inter_df.empty:
                # Apply FDR correction
                pvals = inter_df["p_value"].values
                rejected, qvals, _, _ = multipletests(
                    pvals, alpha=0.05, method="fdr_bh"
                )

                inter_df = inter_df.copy()
                inter_df["q_value"] = qvals
                inter_df["significant"] = rejected

                # Get significant features
                sig_df = inter_df[inter_df["significant"]]

                for _, row in sig_df.iterrows():
                    modality_significant.append(
                        {
                            "feature": row["feature"],
                            "interaction_term": interaction_term,
                        }
                    )

        if modality_significant:
            significant_features_by_modality[modality] = modality_significant

    # Print results
    for modality, features_info in significant_features_by_modality.items():
        print(f"\n{modality}:")
        for info in features_info:
            print(
                f"  Feature: {info['feature']}, Interaction: {info['interaction_term']}"
            )

    # Save results to JSON
    with open(
        Path(
            results_path,
            f"sig_interaction_terms-{wave}.json",
        ),
        "w",
    ) as f:
        json.dump(significant_features_by_modality, f, indent=4)
