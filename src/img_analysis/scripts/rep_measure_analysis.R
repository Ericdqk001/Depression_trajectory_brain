#!/usr/bin/env Rscript

# -------------------------------
# R Script: repeated_measures_analysis.R
# Description: Mixed-effects modeling with repeated hemisphere measures and nested random effects.
# Author: [Your Name]
# -------------------------------

# Load required libraries
suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(broom.mixed)
  library(readr)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(purrr)
})

# -------------------------------
# Read optional command-line argument for wave
# -------------------------------
args <- commandArgs(trailingOnly = TRUE)
wave <- ifelse(length(args) >= 1, args[1], "baseline_year_1_arm_1")

# File paths
features_path <- file.path("data", "processed_data", paste0("mri_all_features_with_traj_long_rescaled-", wave, ".csv"))
feature_sets_path <- file.path("data", "processed_data", "features_of_interest.json")
output_path <- file.path("src", "img_analysis", "analysis_results", paste("repeated_bilateral_traj_results-", wave, ".csv"))

# -------------------------------
# Load data
# -------------------------------
features_df <- read_csv(features_path, show_col_types = FALSE)

# Set categorical variables
features_df <- features_df %>%
  mutate(
    demo_sex_v2 = as.factor(demo_sex_v2),
    img_device_label = as.factor(img_device_label),
    class_label = as.factor(class_label),
    hemisphere = as.factor(hemisphere),
    rel_family_id = as.factor(rel_family_id),
    src_subject_id = as.factor(src_subject_id),
    site_id_l = as.factor(site_id_l),
  )

# Load list of features by modality
feature_sets <- fromJSON(feature_sets_path)

# Define modality-specific global covariates
modality_globals <- list(
  bilateral_cortical_thickness = "smri_thick_cdk_mean",
  bilateral_cortical_surface_area = "smri_area_cdk_total",
  bilateral_cortical_volume = "smri_vol_scs_intracranialv",
  bilateral_subcortical_volume = "smri_vol_scs_intracranialv",
  bilateral_tract_FA = "FA_all_dti_atlas_tract_fibers",
  bilateral_tract_MD = "MD_all_dti_atlas_tract_fibers"
)

# Common covariates
base_covariates <- c("interview_age", "age2", "demo_sex_v2", "demo_comb_income_v2", "BMI_zscore", "img_device_label")

# -------------------------------
# Fit model for one feature
# -------------------------------
fit_feature_model <- function(data, feature_name, fixed_effects, modality) {
  formula_str <- paste0(
    feature_name, " ~ hemisphere * class_label + ",
    paste(fixed_effects, collapse = " + "),
    " + (1 | src_subject_id)"
  )

  message(paste("Fitting model:", feature_name, "in", modality))

  model <- lmer(as.formula(formula_str), data = data)
  tidy_out <- tidy(model, conf.int = TRUE)

  print(tidy_out)

  # Keep all terms — add metadata
  tidy_out <- tidy_out %>%
    mutate(
      modality = modality,
      feature = feature_name
    )

  return(tidy_out)
}

# -------------------------------
# Loop through modalities and features
# -------------------------------
results_all <- purrr::map_dfr(
  names(feature_sets),
  function(modality) {
    message("Processing modality: ", modality)

    roi_features <- feature_sets[[modality]]
    fixed_effects <- c(base_covariates, modality_globals[[modality]])

    purrr::map_dfr(
      head(roi_features, 3),  # Only take the first 3 features
      ~ fit_feature_model(features_df, .x, fixed_effects, modality)
    )
  }
)

# -------------------------------
# Save results
# -------------------------------
write_csv(results_all, output_path)
message("Analysis complete. Results saved to: ", output_path)
