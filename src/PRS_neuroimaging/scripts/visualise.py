import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.multitest import multipletests


def visualise_effect_size(
    wave: str = "baseline_year_1_arm_1",
    experiment_number: int = 4,
    version_name: str = "",
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

    # File paths
    results_path = Path(
        analysis_root_path,
        version_name,
        "experiments",
        f"exp_{experiment_number}",
    )

    repeated_bilateral_prs_results_path = Path(
        results_path,
        f"repeated_bilateral_prs_results-{wave}.csv",
    )
    unilateral_features_glm_results_path = Path(
        results_path,
        f"unilateral_features_glm_results-{wave}.csv",
    )
    sig_hemi_features_glm_results_path = Path(
        results_path,
        f"sig_hemi_features_glm_results-{wave}.csv",
    )

    sig_interaction_terms_path = Path(
        results_path,
        f"sig_interaction_terms-{wave}.json",
    )

    # Read in results
    bilateral_df = pd.read_csv(repeated_bilateral_prs_results_path)
    unilateral_df = pd.read_csv(unilateral_features_glm_results_path)

    with open(sig_interaction_terms_path, "r") as f:
        sig_interaction_terms = json.load(f)

    # Filter out the interaction terms from bilateral_df
    interaction_term = f"C(hemisphere)[T.Right]:{predictor}"

    bilateral_df = bilateral_df[bilateral_df["predictor"] != interaction_term].copy()

    # Take out the bilateral features which had significant interaction terms
    to_remove = set()
    for modality, features in sig_interaction_terms.items():
        for feature in features:
            logging.info(
                f"Removing {modality} - {feature} due to significant interaction term."
            )
            to_remove.add((modality, feature))

    # Filter out rows where (modality, feature) is in to_remove
    mask = bilateral_df.apply(
        lambda row: (row["modality"], row["feature"]) not in to_remove, axis=1
    )

    bilateral_df = bilateral_df[mask].reset_index(drop=True).copy()

    logging.info("Removing features is error-free, checked.")

    # Optional: Load sig_hemi_df if not empty
    sig_hemi_df = pd.read_csv(sig_hemi_features_glm_results_path)

    if sig_hemi_df.empty:
        logging.info("No significant hemispheric features to visualise.")
        combined_df = pd.concat(
            [bilateral_df, unilateral_df],
            ignore_index=True,
        )
    else:
        combined_df = pd.concat(
            [bilateral_df, unilateral_df, sig_hemi_df],
            ignore_index=True,
        )

    # Clean and validate numeric columns
    key_cols = [
        "coefficient",
        "CI_lower",
        "CI_upper",
        "p_value",
    ]

    # Try to convert key columns to numeric
    # Ensure the data is valid and clean

    try:
        for col in key_cols:
            combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")
        if combined_df[key_cols].isnull().any().any():
            raise ValueError("Missing or invalid values found in numerical columns.")
    except Exception as e:
        warnings.warn(f"Data validation failed: {e}")
        logging.error(f"Data validation failed: {e}")
        sys.exit("Visualisation aborted due to data issues. Please check your input.")

    # Collapse modality labels
    modality_map = {
        "bilateral_cortical_thickness": "cortical_thickness",
        "bilateral_cortical_volume": "cortical_volume",
        "bilateral_cortical_surface_area": "cortical_surface_area",
        "bilateral_subcortical_volume": "subcortical_volume",
        "unilateral_subcortical_features": "subcortical_volume",
        "bilateral_tract_FA": "tract_FA",
        "unilateral_tract_FA": "tract_FA",
        "bilateral_tract_MD": "tract_MD",
        "unilateral_tract_MD": "tract_MD",
    }

    combined_df["modality"] = combined_df["modality"].map(modality_map)

    # Validate modality mapping
    if combined_df["modality"].isnull().any():
        unknown_modalities = combined_df.loc[
            combined_df["modality"].isnull(), "modality"
        ].unique()
        warnings.warn(f"Unrecognized modality values found: {unknown_modalities}")
        logging.error(f"Unrecognized modality values found: {unknown_modalities}")
        sys.exit("Visualisation aborted due to unknown modalities.")

    logging.info("Mapping modality names is error-free, checked.")

    # Assign colors
    modality_palette = {
        "cortical_thickness": "#1f77b4",
        "cortical_volume": "#aec7e8",
        "cortical_surface_area": "#ffbb78",
        "subcortical_volume": "#ff7f0e",
        "tract_FA": "#2ca02c",
        "tract_MD": "#98df8a",
    }

    combined_df["modality_color"] = combined_df["modality"].map(modality_palette)

    # Annotation logic
    def get_label(modality, feature):
        # Remove the prefix "img_" from the feature name

        feature = feature.replace("img_", "") if feature.startswith("img_") else feature

        if modality == "cortical_thickness":
            return feature.replace("smri_thick_cdk_", "")
        elif modality == "cortical_volume":
            return feature.replace("smri_vol_cdk_", "")
        elif modality == "cortical_surface_area":
            return feature.replace("smri_area_cdk_", "")
        elif modality == "subcortical_volume":
            return feature.replace("smri_vol_scs_", "")
        elif modality == "tract_FA":
            return feature.replace("FA_dti_atlas_tract_", "")
        elif modality == "tract_MD":
            return feature.replace("MD_dti_atlas_tract_", "")
        else:
            return feature

    # FDR correction and labels
    combined_df["significant"] = False
    combined_df["label"] = ""
    for modality in combined_df["modality"].unique():
        modality_mask = combined_df["modality"] == modality
        modality_df = combined_df.loc[modality_mask]

        logging.info(
            "Running `fdr_bh` : Benjamini/Hochberg  (non-negative) correction for modality %s",
            modality,
        )

        pvals = modality_df["p_value"].values

        rejected, pvals_corrected, _, _ = multipletests(
            pvals,
            alpha=0.05,
            method="fdr_bh",
        )

        combined_df.loc[modality_df.index, "significant"] = rejected

        # Save the corrected p-values
        combined_df.loc[modality_df.index, "p_value_corrected"] = pvals_corrected

        logging.info("Applying FDR correction is error-free, checked.")

        if rejected.any():
            combined_df.loc[modality_df.index[rejected], "label"] = (
                modality_df.loc[rejected]
                .apply(lambda row: get_label(row["modality"], row["feature"]), axis=1)
                .values
            )

        logging.info("Assigning labels is error-free, checked.")

    # Save the combined DataFrame
    combined_df.to_csv(
        results_path / f"combined_fdr_corrected_prs_results-{wave}.csv",
        index=False,
    )

    # Sort by modality group
    modality_order = list(modality_palette.keys())

    ordered_df = pd.concat(
        [combined_df[combined_df["modality"] == mod] for mod in modality_order]
    )
    ordered_df = ordered_df.reset_index(drop=True)

    # Plot
    plt.figure(figsize=(26, 6))
    offset = (ordered_df["CI_upper"].max() - ordered_df["CI_lower"].min()) * 0.05

    for i, row in ordered_df.iterrows():
        plt.errorbar(
            i,
            row["coefficient"],
            yerr=[
                [row["coefficient"] - row["CI_lower"]],
                [row["CI_upper"] - row["coefficient"]],
            ],
            fmt="o",
            color=row["modality_color"],
            ecolor="gray",
            capsize=2,
        )
        if row["significant"] and row["label"]:
            y_text = (
                row["coefficient"] + offset
                if row["coefficient"] >= 0
                else row["coefficient"] - offset
            )
            va = "bottom" if row["coefficient"] >= 0 else "top"

            plt.annotate(
                row["label"],
                xy=(i, row["coefficient"]),
                xytext=(i + 1.5, y_text),
                textcoords="data",
                fontsize=8,
                ha="left",
                va=va,
                arrowprops=dict(
                    arrowstyle="-",
                    connectionstyle="angle,angleA=0,angleB=60,rad=3",
                    lw=0.7,
                    color="black",
                ),
                rotation=45,
            )

    plt.axhline(0, color="black", linewidth=0.5)
    plt.xticks([])
    plt.xlabel("Brain Features (grouped by modality)")
    plt.ylabel("Beta coefficient (PRS effect)")
    plt.title("PRS Effects on Imaging Features (FDR-corrected, per modality)")

    # Legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=mod,
            markerfacecolor=color,
            markersize=8,
        )
        for mod, color in modality_palette.items()
    ]
    plt.legend(
        handles=handles, title="Modality", bbox_to_anchor=(1.01, 1), loc="upper left"
    )

    plt.tight_layout()

    # Save
    output_dir = Path(
        analysis_root_path,
        version_name,
        "experiments",
        f"exp_{experiment_number}",
        "images",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"PRS_plot-{wave}.png", dpi=300)

    plt.show()
