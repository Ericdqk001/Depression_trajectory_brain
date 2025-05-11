from src.img_analysis.scripts.identify_sig_inter_terms import identify_sig_inter_terms
from src.img_analysis.scripts.model_unilateral import (
    perform_unilateral,
)
from src.img_analysis.scripts.prepare_img import preprocess
from src.img_analysis.scripts.rep_measure_analysis import (
    perform_repeated_measures_analysis,
)
from src.img_analysis.scripts.visualise import visualise_effect_size


def main(wave):
    # Step 1: Preprocessing
    print(f"Preprocessing data for {wave}...")
    preprocess(wave=wave)

    # Step 2: Modeling for bilateral features
    print(f"Performing repeated measures analysis for {wave}...")
    perform_repeated_measures_analysis(wave=wave)

    # Step 3: Check significant interactions (hemisphere x class)
    print(f"Identifying significant interaction terms for {wave}...")
    identify_sig_inter_terms(wave=wave)

    # Step 4:Modeling for unilateral features
    print(f"Performing unilateral analysis for {wave}...")
    perform_unilateral(wave=wave)

    # Step 5: Visualize effect sizes
    print(f"Visualizing effect sizes for {wave}...")
    visualise_effect_size(wave=wave)


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
