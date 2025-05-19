import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection


def visualise_effect_size(
    wave: str = "baseline_year_1_arm_1",
    results_number: int = 1,
):
    # File paths

    results_path = Path(
        "src",
        "img_analysis",
        "analysis_results",
        f"exp_{results_number}",
    )

    repeated_bilateral_results_path = Path(
        results_path,
        f"repeated_bilateral_traj_results-{wave}.csv",
    )
    unilateral_features_results_path = Path(
        results_path,
        f"unilateral_features_results-{wave}.csv",
    )
    sig_hemi_features_glm_results_path = Path(
        results_path,
        f"sig_hemi_features_results-{wave}.csv",
    )

    # Read in results
    bilateral_df = pd.read_csv(repeated_bilateral_results_path)
    unilateral_df = pd.read_csv(unilateral_features_results_path)

    if sig_hemi_features_glm_results_path.exists():
        sig_hemi_df = pd.read_csv(sig_hemi_features_glm_results_path)
    else:
        sig_hemi_df = pd.DataFrame()

    if sig_hemi_df.empty:
        print("No significant hemispheric features found.")
        combined_df = pd.concat([bilateral_df, unilateral_df], ignore_index=True)
    else:
        combined_df = pd.concat(
            [bilateral_df, unilateral_df, sig_hemi_df], ignore_index=True
        )

    # Filter to class_label effects only (assume effect_name column contains class_labelX.X)
    class_label_effects = [
        "class_label1",
        "class_label2",
        "class_label3",
    ]
    combined_df = combined_df[
        combined_df["effect_name"].isin(class_label_effects)
    ].copy()

    # Clean and validate numeric columns
    key_cols = ["Estimate", "2.5_ci", "97.5_ci", "P-val"]

    try:
        for col in key_cols:
            combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

        # Check for missing values after conversion
        if combined_df[key_cols].isnull().any().any():
            raise ValueError("Missing or invalid values found in numerical columns.")

    except Exception as e:
        warnings.warn(f"Data validation failed: {e}")
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

    # Assign colors
    modality_palette = {
        "cortical_thickness": "#1f77b4",
        "cortical_volume": "#aec7e8",
        "cortical_surface_area": "#ffbb78",
        "subcortical_volume": "#ff7f0e",
        "tract_FA": "#2ca02c",
        "tract_MD": "#98df8a",
    }
    combined_df["modality_color"] = (
        combined_df["modality"].map(modality_palette).fillna("gray")
    )

    # Annotation logic
    def get_label(modality, feature):
        if modality == "cortical_thickness":
            return feature.replace("img_smri_thick_cdk_", "")
        elif modality == "cortical_volume":
            return feature.replace("img_smri_vol_cdk_", "")
        elif modality == "cortical_surface_area":
            return feature.replace("img_smri_area_cdk_", "")
        elif modality == "subcortical_volume":
            return feature.replace("img_smri_vol_scs_", "")
        elif modality == "tract_FA":
            return feature.replace("img_FA_dti_atlas_tract_", "")
        elif modality == "tract_MD":
            return feature.replace("img_MD_dti_atlas_tract_", "")
        else:
            return feature

    # For each class_label, plot effect sizes
    for class_label in class_label_effects:
        df = combined_df[combined_df["effect_name"] == class_label].copy()
        if df.empty:
            print(f"No results for {class_label} in {wave}")
            continue

        # Clean and validate numeric columns
        key_cols = ["Estimate", "2.5_ci", "97.5_ci", "P-val"]
        for col in key_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[key_cols].isnull().any().any():
            warnings.warn(
                f"Missing or invalid values found in numerical columns for {class_label}."
            )
            continue

        # FDR correction and labels
        df["significant"] = False
        df["label"] = ""
        for modality in df["modality"].unique():
            modality_mask = df["modality"] == modality
            modality_df = df.loc[modality_mask]
            pvals = modality_df["P-val"].values
            reject, _ = fdrcorrection(pvals, alpha=0.05)
            df.loc[modality_df.index, "significant"] = reject
            df.loc[modality_df.index[reject], "label"] = (
                modality_df.loc[reject]
                .apply(lambda row: get_label(row["modality"], row["feature"]), axis=1)
                .values
            )

        # Sort by modality group
        modality_order = list(modality_palette.keys())
        ordered_df = pd.concat(
            [
                df[df["modality"] == mod]
                for mod in modality_order
                if mod in df["modality"].unique()
            ]
        )
        ordered_df = ordered_df.reset_index(drop=True)

        # Plot
        plt.figure(figsize=(26, 6))
        offset = (ordered_df["97.5_ci"].max() - ordered_df["2.5_ci"].min()) * 0.05
        for i, row in ordered_df.iterrows():
            plt.errorbar(
                i,
                row["Estimate"],
                yerr=[
                    [row["Estimate"] - row["2.5_ci"]],
                    [row["97.5_ci"] - row["Estimate"]],
                ],
                fmt="o",
                color=row["modality_color"],
                ecolor="gray",
                capsize=2,
            )
            if row["significant"] and row["label"]:
                y_text = (
                    row["Estimate"] + offset
                    if row["Estimate"] >= 0
                    else row["Estimate"] - offset
                )
                va = "bottom" if row["Estimate"] >= 0 else "top"
                plt.annotate(
                    row["label"],
                    xy=(i, row["Estimate"]),
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
        plt.ylabel(f"Beta coefficient ({class_label} effect)")
        plt.title(
            f"{class_label} Effects on Imaging Features (FDR-corrected, per modality)"
        )
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
            handles=handles,
            title="Modality",
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
        )
        plt.tight_layout()

        output_dir = Path(
            results_path,
            "images",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"class_label_{class_label}_plot-{wave}.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    visualise_effect_size()
