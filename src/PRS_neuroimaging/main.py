from src.PRS_neuroimaging.scripts.glm import perform_glm
from src.PRS_neuroimaging.scripts.identify_sig_inter_terms import (
    identify_sig_inter_terms,
)
from src.PRS_neuroimaging.scripts.prepare_img import preprocess

# from src.poppy.scripts.prepare_img import preprocess
from src.PRS_neuroimaging.scripts.rep_measure_analysis import (
    perform_repeated_measures_analysis,
)
from src.PRS_neuroimaging.scripts.visualise import visualise_effect_size


def main(
    wave: str = "baseline_year_1_arm_1",
    version_name: str = "",
    experiment_number: int = 1,
    predictor: str = "CBCL_quant",
):
    # Call the preprocess function from the prepare_img module
    preprocess(
        wave=wave,
        version_name=version_name,
    )
    # Call the rep_measure_analysis function from the rep_measure_analysis module
    perform_repeated_measures_analysis(
        wave=wave,
        experiment_number=experiment_number,
        version_name=version_name,
        predictor=predictor,
    )
    # Call the identify_sig_inter_terms function from the identify_sig_inter_terms module
    identify_sig_inter_terms(
        wave=wave,
        experiment_number=experiment_number,
        version_name=version_name,
        predictor=predictor,
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
        # "4_year_follow_up_y_arm_1",
    ]

    version_name = "CBCL_replication_test"

    predictor = "CBCL_quant"

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
