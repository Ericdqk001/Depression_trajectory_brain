from src.img_analysis.scripts.identify_sig_inter_terms import identify_sig_inter_terms
from src.img_analysis.scripts.prepare_img import preprocess
from src.img_analysis.scripts.rep_measure_analysis import (
    perform_repeated_measures_analysis,
)


def main(wave):
    # Step 1: Preprocessing
    preprocess(wave=wave)

    # Step 2: modeling
    perform_repeated_measures_analysis(wave=wave)

    # Step 3: check significant interactions (hemisphere x class)
    identify_sig_inter_terms(wave=wave)


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
