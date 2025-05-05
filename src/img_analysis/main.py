import subprocess

from src.poppy.scripts.prepare_img import preprocess


def run_r_model(wave: str):
    try:
        # Run the R script with the wave as an argument
        subprocess.run(
            ["Rscript", "src/img_analysis/scripts/rep_measure_analysis.R", wave],
            check=True,
        )

        print(f"R modeling completed for {wave}")
    except subprocess.CalledProcessError as e:
        print(f"R modeling failed for {wave}: {e}")


def main(wave):
    # Step 1: Preprocessing
    preprocess(wave=wave)

    # Step 2: R modeling
    run_r_model(wave=wave)


if __name__ == "__main__":
    all_img_waves = [
        "baseline_year_1_arm_1",
        "2_year_follow_up_y_arm_1",
        "4_year_follow_up_y_arm_1",
    ]
    for wave in all_img_waves:
        print(f"Running full pipeline for {wave}...")
        main(wave=wave)
        print(f"Analysis completed for {wave}.\n")
