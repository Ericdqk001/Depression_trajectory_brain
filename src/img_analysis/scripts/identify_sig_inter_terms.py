import json
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests


def identify_sig_inter_terms(
    wave: str = "baseline_year_1_arm_1", results_number: int = 1
):
    modalities = [
        "bilateral_cortical_thickness",
        "bilateral_cortical_volume",
        "bilateral_cortical_surface_area",
        "bilateral_subcortical_volume",
        "bilateral_tract_FA",
        "bilateral_tract_MD",
    ]

    results_path = Path(
        "src",
        "img_analysis",
        "results",
        f"exp_{results_number}",
    )

    # Load results
    repeated_results = pd.read_csv(
        Path(
            results_path,
            f"repeated_bilateral_traj_results-{wave}.csv",
        )
    )

    significant_features_by_modality = {}

    # The interaction terms of interest (as in rep_measure_analysis.py)
    interaction_terms = [
        "hemisphereRight:class_label1",
        "hemisphereRight:class_label2",
        "hemisphereRight:class_label3",
    ]

    for modality in modalities:
        modality_df = repeated_results[
            (repeated_results["modality"] == modality)
            & (repeated_results["effect_name"].isin(interaction_terms))
        ]

        if not modality_df.empty:
            # Apply FDR correction
            pvals = modality_df["P-val"].values

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


if __name__ == "__main__":
    identify_sig_inter_terms()
