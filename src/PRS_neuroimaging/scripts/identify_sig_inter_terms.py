import json
import logging
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests


def identify_sig_inter_terms(
    wave: str = "baseline_year_1_arm_1",
    version_name: str = "",
    experiment_number: int = 3,
    predictor="score",
):
    data_store_path = Path(
        "/",
        "Volumes",
        "GenScotDepression",
    )

    if data_store_path.exists():
        logging.info("Mounted data store path: %s", data_store_path)

    analysis_root_path = Path(
        data_store_path,
        "users",
        "Eric",
        "poppy_neuroimaging",
    )

    modalities = [
        "bilateral_cortical_thickness",
        "bilateral_cortical_volume",
        "bilateral_cortical_surface_area",
        "bilateral_subcortical_volume",
        "bilateral_tract_FA",
        "bilateral_tract_MD",
    ]

    # TODO: Uncomment this
    interaction_term = f"C(hemisphere)[T.Right]:{predictor}"

    # interaction_term = f"hemisphere[T.Right]:{prs_variable}"

    results_path = Path(
        analysis_root_path,
        version_name,
        "experiments",
        f"exp_{experiment_number}",
    )

    # Load results
    repeated_results = pd.read_csv(
        Path(
            results_path,
            f"repeated_bilateral_prs_results-{wave}.csv",
        )
    )

    significant_features_by_modality = {}

    for modality in modalities:
        logging.info(
            "Running `fdr_bh` : Benjamini/Hochberg  (non-negative) correction for modality for the interaction term: %s",
            modality,
        )

        modality_df = repeated_results[
            (repeated_results["modality"] == modality)
            & (repeated_results["predictor"] == interaction_term)
        ]

        pvals = modality_df["p_value"].values

        rejected, qvals, _, _ = multipletests(
            pvals,
            alpha=0.05,
            method="fdr_bh",
        )

        modality_df = modality_df.copy()
        modality_df["q_value"] = qvals
        modality_df["significant"] = rejected

        # Filter significant
        significant = modality_df[modality_df["significant"]]

        significant_features_by_modality[modality] = significant["feature"].tolist()

    # Print results
    for modality, features in significant_features_by_modality.items():
        logging.info(f"{modality}: {features}")

    # Save results to json
    with open(
        Path(
            results_path,
            f"sig_interaction_terms-{wave}.json",
        ),
        "w",
    ) as f:
        json.dump(significant_features_by_modality, f, indent=4)
