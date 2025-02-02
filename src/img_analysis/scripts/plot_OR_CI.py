from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Path to the results directory
multinomial_results_path = Path(
    "src",
    "img_analysis",
    "results",
    "multinomial",
)

class_label_name_map = {
    0: "Low",
    1: "Increasing",
    2: "Decreasing",
    3: "High",
}

# Iterate through each results CSV file
for modality_result in multinomial_results_path.glob("*.csv"):
    print(f"Processing results for {modality_result.stem}")

    # Load the results CSV
    result_df = pd.read_csv(modality_result)

    # Iterate over classes (1, 2, 3)
    for class_label in result_df["Class"].unique():
        print(f"Plotting for Class {class_label}")

        # Filter data for the current class
        class_df = result_df[result_df["Class"] == class_label]

        # Determine significance: confidence intervals do not include 1
        class_df["Significant"] = (class_df["Confidence Interval (0.025)"] > 1) | (
            class_df["Confidence Interval (0.975)"] < 1
        )

        # Plot confidence intervals
        plt.figure(figsize=(10, 6))

        for _, row in class_df.iterrows():
            color = (
                "red" if row["Significant"] else "blue"
            )  # Red for significant, blue otherwise
            plt.errorbar(
                row["Feature"],
                row["Odds Ratio"],
                yerr=[
                    [row["Odds Ratio"] - row["Confidence Interval (0.025)"]],
                    [row["Confidence Interval (0.975)"] - row["Odds Ratio"]],
                ],
                fmt="o",
                color=color,
                capsize=5,
                label="Significant" if row["Significant"] else "Non-significant",
            )

        plt.axhline(
            y=1, color="gray", linestyle="--", linewidth=1
        )  # Reference line for Odds Ratio = 1
        plt.title(f"Confidence Intervals for Class {class_label_name_map[class_label]}")
        plt.xlabel("Feature")
        plt.ylabel("Odds Ratio")
        plt.xticks(rotation=90)
        plt.tight_layout()

        # Save the plot
        images_path = Path(
            multinomial_results_path.parent,
            "images",
        )

        if not images_path.exists():
            images_path.mkdir(parents=True)

        output_image_path = Path(
            images_path,
            f"{modality_result.stem}_class_{class_label}_odds_ratio.png",
        )

        plt.savefig(output_image_path)
        print(f"Saved plot to {output_image_path}")
        plt.close()
