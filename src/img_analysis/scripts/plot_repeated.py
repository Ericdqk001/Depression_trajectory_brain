from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define file paths
data_path = Path("img_analysis_results", "repeated_measures")
plots_save_path = Path("img_analysis_results", "images", "mixed_effect_plots")

# Ensure save directory exists
plots_save_path.mkdir(parents=True, exist_ok=True)

# Load the results files
repeated_results = pd.read_csv(Path(data_path, "repeated_regional_results.csv"))
significant_interactions = pd.read_csv(
    Path(data_path, "significant_interaction_features.csv")
)
unilateral_results = pd.read_csv(Path(data_path, "unilateral_regional_results.csv"))

# Filter for trajectory class effects
effects_of_interest = {
    "C(class_label)[T.1.0]": "Increasing",
    "C(class_label)[T.2.0]": "Decreasing",
    "C(class_label)[T.3.0]": "High",
}

repeated_results = repeated_results[
    repeated_results["effect"].isin(effects_of_interest.keys())
]
unilateral_results = unilateral_results[
    unilateral_results["effect"].isin(effects_of_interest.keys())
]

# Get significant interaction features
sig_features = set(significant_interactions["feature"].unique())

# Define modalities and rename for titles
modality_names = {
    "cortical_thickness": "Thickness",
    "cortical_volume": "Volume",
    "cortical_surface_area": "Surface Area",
}

# Iterate over trajectory class effects and modalities
for effect, effect_label in effects_of_interest.items():
    for modality, modality_label in modality_names.items():
        fig, axes = plt.subplots(
            ncols=2, figsize=(12, 6), sharey=True, gridspec_kw={"width_ratios": [1, 1]}
        )

        # Subset repeated measures results for the modality
        repeated_modality = repeated_results[
            (repeated_results["modality"] == modality)
            & (repeated_results["effect"] == effect)
        ]
        unilateral_modality = unilateral_results[
            (unilateral_results["modality"] == modality)
            & (unilateral_results["effect"] == effect)
        ]

        # Separate features into left and right panels
        non_sig_features = repeated_modality[
            ~repeated_modality["feature"].isin(sig_features)
        ]
        sig_unilateral_features = unilateral_modality[
            unilateral_modality["feature"].isin(sig_features)
        ]

        # Left Panel: Features without significant hemisphere interaction
        ax_left = axes[0]
        ax_left.errorbar(
            non_sig_features["feature"],
            non_sig_features["coefficient"],
            yerr=[
                non_sig_features["coefficient"] - non_sig_features["CI_lower"],
                non_sig_features["CI_upper"] - non_sig_features["coefficient"],
            ],
            fmt="o",
            color="blue",
            label="Repeated Measures",
        )

        ax_left.axhline(0, color="black", linestyle="--", linewidth=1)
        ax_left.set_xticklabels(non_sig_features["feature"], rotation=90, fontsize=8)
        ax_left.set_title(
            f"{modality_label} - {effect_label} (No Significant Interaction)"
        )
        ax_left.legend()

        # Right Panel: Features with significant hemisphere interaction
        ax_right = axes[1]
        unique_features = sig_unilateral_features["feature"].unique()
        x_positions = np.arange(len(unique_features))  # Base x positions

        for i, hemisphere in enumerate(["Left", "Right"]):
            hemi_data = sig_unilateral_features[
                sig_unilateral_features["hemisphere"] == hemisphere
            ]
            shift = -0.1 if hemisphere == "Left" else 0.1  # Offset left and right

            ax_right.errorbar(
                x_positions + shift,
                hemi_data["coefficient"],
                yerr=[
                    hemi_data["coefficient"] - hemi_data["CI_lower"],
                    hemi_data["CI_upper"] - hemi_data["coefficient"],
                ],
                fmt="o",
                label=f"Unilateral {hemisphere}",
            )

        ax_right.axhline(0, color="black", linestyle="--", linewidth=1)
        ax_right.set_xticks(x_positions)
        ax_right.set_xticklabels(unique_features, rotation=90, fontsize=8)
        ax_right.set_title(
            f"{modality_label} - {effect_label} (Significant Hemisphere Interaction)"
        )
        ax_right.legend()

        plt.tight_layout()

        # Save plot
        save_path = Path(
            plots_save_path, f"{modality}_{effect_label.replace(' ', '_')}_CI_plot.png"
        )
        plt.savefig(save_path)
        plt.show()
