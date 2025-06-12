from src.PRS_global_brain.scripts.global_glm import perform_glm
from src.PRS_global_brain.scripts.global_visualise import visualise_effect_size
from src.PRS_global_brain.scripts.prepare_img import preprocess


def main(
    wave: str = "baseline_year_1_arm_1",
    version_name: str = "abcd_adoldep_global_brain_analysis",
    experiment_number: int = 1,
    predictor: str = "score",
):
    # Call the preprocess function from the prepare_img module
    preprocess(
        wave=wave,
        version_name=version_name,
    )

    # # Call the glm_analysis function from the glm module
    perform_glm(
        wave=wave,
        experiment_number=experiment_number,
        version_name=version_name,
        predictor=predictor,
    )
    # # Call the visualise_effect_size function from the visualise module
    visualise_effect_size(
        wave=wave,
        experiment_number=experiment_number,
        version_name=version_name,
        predictor=predictor,
    )


if __name__ == "__main__":
    # Run the main function with the default wave
    all_img_waves = [
        "baseline_year_1_arm_1",
        "2_year_follow_up_y_arm_1",
        "4_year_follow_up_y_arm_1",
    ]

    version_name = "abcd_adoldep_global_brain_analysis"

    predictor = "score"

    experiment_number = 1

    for wave in all_img_waves:
        print(f"Running analysis for {wave}...")
        main(
            wave=wave,
            version_name=version_name,
            experiment_number=experiment_number,
            predictor=predictor,
        )
        print(f"Analysis for {wave} completed.\n")
