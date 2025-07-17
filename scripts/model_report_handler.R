library(reticulate)

# Use reticulate to import our Python module
py_model_report_handler <- import("model_report_handler")

read_model_report <- function(individual_dir) {
  py_model_report_handler$read_model_report(individual_dir)
}

update_model_report <- function(individual_dir, updates) {
  py_model_report_handler$update_model_report(individual_dir, updates)
}

get_model_status <- function(individual_dir) {
  py_model_report_handler$get_model_status(individual_dir)
}

get_objective_value <- function(individual_dir) {
  py_model_report_handler$get_objective_value(individual_dir)
}
