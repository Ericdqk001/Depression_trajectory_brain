import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def visualise_effect_size(
    wave: str = "baseline_year_1_arm_1",
    experiment_number: int = 1,
    version_name: str = "CBCL_replication_test",
):
    data_store_path = Path(
        "/",
        "Volumes",
        "GenScotDepression",
    )

    if data_store_path.exists():
        logging.info(f"Mounted data store path: {data_store_path}")

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

    global_feature_prs_results_path = Path(
        results_path,
        f"global_features_glm_results-{wave}.csv",
    )

    # Read in results
    global_feature_df = pd.read_csv(global_feature_prs_results_path)

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
            global_feature_df[col] = pd.to_numeric(
                global_feature_df[col], errors="coerce"
            )
        # Check for missing values after conversion
        if global_feature_df[key_cols].isnull().any().any():
            raise ValueError("Missing or invalid values found in numerical columns.")
        logging.info("Numeric columns validated and converted: %s", key_cols)
    except Exception as e:
        warnings.warn(f"Data validation failed: {e}")
        logging.error(f"Data validation failed: {e}")
        sys.exit("Visualisation aborted due to data issues. Please check your input.")

    # Validate modality mapping
    if global_feature_df["modality"].isnull().any():
        unknown_modalities = global_feature_df.loc[
            global_feature_df["modality"].isnull(), "modality"
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

    global_feature_df["modality_color"] = global_feature_df["modality"].map(
        modality_palette
    )

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

    # Mark features as significant if p_value < 0.05 (no FDR correction)
    global_feature_df["significant"] = global_feature_df["p_value"] < 0.05

    # Assign plot color: red for significant, otherwise modality color
    sig_color = "#d62728"  # red for significant
    global_feature_df["plot_color"] = global_feature_df.apply(
        lambda row: sig_color if row["significant"] else row["modality_color"], axis=1
    )

    # Annotate all features regardless of significance
    global_feature_df["label"] = global_feature_df.apply(
        lambda row: get_label(row["modality"], row["feature"]), axis=1
    )
    logging.info("All features annotated, significance based on raw p-value < 0.05.")

    # Save the combined DataFrame
    global_feature_df.to_csv(
        results_path / f"combined_fdr_corrected_prs_results-{wave}.csv",
        index=False,
    )
    logging.info(
        f"Combined results saved to: {results_path / f'combined_fdr_corrected_prs_results-{wave}.csv'}"
    )

    # Sort by modality group
    modality_order = list(modality_palette.keys())

    ordered_df = pd.concat(
        [
            global_feature_df[global_feature_df["modality"] == mod]
            for mod in modality_order
        ]
    )
    ordered_df = ordered_df.reset_index(drop=True)

    # Plot
    plt.figure(figsize=(26, 6))
    offset = (ordered_df["CI_upper"].max() - ordered_df["CI_lower"].min()) * 0.05

    for i, row in ordered_df.iterrows():
        # Use red for significant features, modality color otherwise
        point_color = "red" if row["significant"] else row["modality_color"]
        plt.errorbar(
            i,
            row["coefficient"],
            yerr=[
                [row["coefficient"] - row["CI_lower"]],
                [row["CI_upper"] - row["coefficient"]],
            ],
            fmt="o",
            color=point_color,
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
    # Add a legend entry for significant features
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Significant (p < 0.05)",
            markerfacecolor="red",
            markersize=8,
        )
    )
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
    logging.info(f"Plot saved to: {output_dir / f'PRS_plot-{wave}.png'}")

    plt.show()
