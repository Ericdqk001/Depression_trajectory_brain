import logging
from pathlib import Path

from src.PRS_global_brain.scripts.global_glm import perform_glm
from src.PRS_global_brain.scripts.global_visualise import visualise_effect_size
from src.PRS_global_brain.scripts.prepare_img import preprocess


def main(
    wave: str = "baseline_year_1_arm_1",
    version_name: str = "abcd_adoldep_global_brain_analysis",
    experiment_number: int = 1,
    predictor: str = "score",
):
    data_store_path = Path(
        "/",
        "Volumes",
        "GenScotDepression",
    )

    analysis_root_path = Path(
        data_store_path,
        "users",
        "Eric",
        "poppy_neuroimaging",
    )

    experiments_path = Path(
        analysis_root_path,
        version_name,
        "experiments",
    )

    if not experiments_path.exists():
        experiments_path.mkdir(parents=True, exist_ok=True)

    results_path = Path(
        experiments_path,
        f"exp_{experiment_number}",
    )
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)

    log_file = results_path / "experiment.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

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
    )


if __name__ == "__main__":
    # Run the main function with the default wave
    all_img_waves = [
        "baseline_year_1_arm_1",
        "2_year_follow_up_y_arm_1",
        "4_year_follow_up_y_arm_1",
    ]

    version_name = "abcd_adoldep_sbayesrc_eur_amr_global_brain_analysis"

    predictor = "score"

    experiment_number = 1

    for wave in all_img_waves:
        print(f"Running analysis for wave: {wave}")
        main(
            wave=wave,
            version_name=version_name,
            experiment_number=experiment_number,
            predictor=predictor,
        )
