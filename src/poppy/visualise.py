from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection

# Load data
bilateral_results_path = Path(
    "src", "poppy", "analysis_results", "repeated_bilateral_prs_results.csv"
)
unilateral_results_path = Path(
    "src", "poppy", "analysis_results", "unilateral_features_glm_results.csv"
)
bilateral_df = pd.read_csv(bilateral_results_path)
unilateral_df = pd.read_csv(unilateral_results_path)

# Filter to main effect
bilateral_df = bilateral_df[bilateral_df["predictor"] == "aoDEP_SBayesR"].copy()
bilateral_df = bilateral_df.rename(columns={"predictor": "effect"})
unilateral_df = unilateral_df[unilateral_df["effect"] == "aoDEP_SBayesR"].copy()
df = pd.concat([bilateral_df, unilateral_df], ignore_index=True)

# Clean numeric columns
for col in ["coefficient", "CI_lower", "CI_upper", "p_value"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["coefficient", "CI_lower", "CI_upper", "p_value"])

# Modality colors
modality_palette = {
    "bilateral_cortical_thickness": "#1f77b4",
    "bilateral_cortical_volume": "#aec7e8",
    "bilateral_cortical_surface_area": "#ffbb78",
    "bilateral_subcortical_volume": "#ff7f0e",
    "unilateral_subcortical_features": "#d62728",
    "bilateral_tract_FA": "#2ca02c",
    "bilateral_tract_MD": "#98df8a",
    "unilateral_tract_FA": "#9467bd",
    "unilateral_tract_MD": "#c5b0d5",
}
df["modality_color"] = df["modality"].map(modality_palette).fillna("gray")


# Annotation logic
def get_label(modality, feature):
    if modality == "bilateral_cortical_thickness":
        return feature.replace("img_smri_thick_cdk_", "")
    elif modality == "bilateral_cortical_volume":
        return feature.replace("img_smri_vol_cdk_", "")
    elif modality == "bilateral_cortical_surface_area":
        return feature.replace("img_smri_area_cdk_", "")
    elif modality in [
        "bilateral_subcortical_volume",
        "unilateral_subcortical_features",
    ]:
        return feature.replace("img_smri_vol_scs_", "")
    elif modality in ["bilateral_tract_FA", "unilateral_tract_FA"]:
        return feature.replace("img_FA_dti_atlas_tract_", "")
    elif modality in ["bilateral_tract_MD", "unilateral_tract_MD"]:
        return feature.replace("img_MD_dti_atlas_tract_", "")
    else:
        return feature


# FDR correction and label assignment
df["significant"] = False
df["label"] = ""
for modality in df["modality"].unique():
    modality_mask = df["modality"] == modality
    modality_df = df.loc[modality_mask]

    pvals = modality_df["p_value"].values
    reject, _ = fdrcorrection(pvals, alpha=0.05)

    df.loc[modality_df.index, "significant"] = reject
    df.loc[modality_df.index[reject], "label"] = (
        modality_df.loc[reject]
        .apply(lambda row: get_label(row["modality"], row["feature"]), axis=1)
        .values
    )

# Maintain modality group order
modality_order = list(modality_palette.keys())
ordered_df = pd.DataFrame()
for mod in modality_order:
    ordered_df = pd.concat([ordered_df, df[df["modality"] == mod]])
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
        # Choose label position
        y_text = (
            row["coefficient"] + offset
            if row["coefficient"] >= 0
            else row["coefficient"] - offset
        )
        va = "bottom" if row["coefficient"] >= 0 else "top"

        # Diagonal arrow annotation
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

# Formatting
plt.axhline(0, color="black", linewidth=0.5)
plt.xticks([])
plt.xlabel("Brain Features (grouped by modality)")
plt.ylabel("Beta coefficient (PRS effect)")
plt.title("aoDEP_SBayesR Effects on Imaging Features (FDR-corrected, per modality)")

# Legend
handles = [
    plt.Line2D(
        [0], [0], marker="o", color="w", label=mod, markerfacecolor=color, markersize=8
    )
    for mod, color in modality_palette.items()
]
plt.legend(
    handles=handles, title="Modality", bbox_to_anchor=(1.01, 1), loc="upper left"
)

plt.tight_layout()

# Save plot
output_dir = Path("src", "poppy", "images")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "aoDEP_SBayesR_plot.png", dpi=300)

plt.show()
