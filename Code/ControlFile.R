require(TMB)
require(jsonlite)
library(here)
library(ggplot2)

# Function to read existing report or create new structure
read_or_create_report <- function(individual_dir) {
  report_file <- file.path(individual_dir, 'model_report.json')
  if (file.exists(report_file)) {
    tryCatch({
      existing_report <- fromJSON(report_file)
      if (!"iterations" %in% names(existing_report)) {
        list(iterations = list())
      } else {
        existing_report
      }
    }, error = function(e) {
      list(iterations = list())
    })
  } else {
    list(iterations = list())
  }
}

# Function to get next iteration number
get_next_iteration <- function(report_data) {
  if (length(report_data$iterations) == 0) {
    return("1")
  }
  as.character(max(as.numeric(names(report_data$iterations))) + 1)
}

# Function to write report directly to file
write_report_direct <- function(individual_dir, report_data) {
  report_file <- file.path(individual_dir, 'model_report.json')
  existing_report <- read_or_create_report(individual_dir)
  next_iteration <- get_next_iteration(existing_report)
  existing_report$iterations[[next_iteration]] <- report_data
  write(toJSON(existing_report, auto_unbox = TRUE, pretty = TRUE), report_file)
}

# Function to write error JSON
write_error_json <- function(error_message, individual_dir) {
  error_data <- list(
    status = "ERROR",
    message = error_message
  )
  write_report_direct(individual_dir, error_data)
}

# Source validation function
source("Code/validate_model.R")

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
individual_dir <- tryCatch({
  if (length(args) >= 2 && args[1] == "--model_location") {
    args[2]
  } else {
    here('POPULATIONS/POPULATION_0001/INDIVIDUAL_00000001')
  }
}, error = function(e) {
  here('POPULATIONS/POPULATION_0001/INDIVIDUAL_00000001')
})

# Read population metadata to get data files
population_dir <- gsub("/BROKEN", "", dirname(individual_dir))
metadata_file <- file.path(population_dir, "population_metadata.json")
metadata <- fromJSON(metadata_file, simplifyVector = TRUE)
response_file <- metadata$response_file
forcing_file <- metadata$forcing_file

model_file <- 'model.cpp'
# Check OS and set appropriate flags
if (.Platform$OS.type == "windows") {
  flags <- "-O2 -Wa,-mbig-obj"
} else {
  # For Linux/Unix/macOS
  flags <- "-O2"
}
compilation_result <- try(TMB::compile(file.path(individual_dir, model_file),
                                       flags = flags,
                                       safebounds = FALSE,
                                       safeunload = FALSE,
                                       openmp = FALSE,
                                       debug = FALSE,
                                       split = TRUE,
                                       verbose = TRUE))

if (inherits(compilation_result, "try-error")) {
  stop("Compilation failed")
}

model_path <- file.path(individual_dir, 'model')
dyn.load(dynlib(model_path))

# Load and process parameters from JSON
params <- tryCatch({
  json_data <- fromJSON(file.path(individual_dir, 'parameters.json'))
  
  # Check if the JSON is in the new format (with "parameters" array)
  if (!is.null(json_data$parameters)) {
    # New format - array of parameter objects
    json_data$parameters
  } else {
    # Old format - direct key-value pairs
    param_df <- data.frame(
      parameter = names(json_data),
      value = unlist(json_data),
      stringsAsFactors = FALSE
    )
    param_df$source <- "initial estimate"
    param_df$import_type <- "PARAMETER"
    param_df$priority <- 1
    param_df
  }
}, error = function(e) {
  stop("Failed to load parameters")
})

# Load and process time series data
time_series_data <- tryCatch({
  # Read response data
  response_data <- read.csv(response_file)
  
  # Modify response column names by splitting at '..' and keeping the first part
  colnames(response_data) <- sapply(colnames(response_data), function(x) {
    strsplit(x, "..", fixed = TRUE)[[1]][1]
  })
  
  # Initialize merged_data with response_data
  merged_data <- response_data
  
  # If forcing file is provided, merge with forcing data
  if (!is.null(forcing_file) && forcing_file != "") {
    forcing_data <- read.csv(forcing_file)
    
    # Modify forcing column names
    colnames(forcing_data) <- sapply(colnames(forcing_data), function(x) {
      strsplit(x, "..", fixed = TRUE)[[1]][1]
    })
    
    # Merge on first column
    merged_data <- merge(response_data, forcing_data, by = names(response_data)[1], all = TRUE)
    
    if (nrow(merged_data) != nrow(response_data) || nrow(merged_data) != nrow(forcing_data)) {
      warning("Possible data loss during merge. Check time columns for mismatches.")
    }
  }
  
  merged_data
}, error = function(e) {
  warning(paste("Failed to load time series data:", e))
  warning("Response file:", response_file)
  warning("Forcing file:", forcing_file)
  warning("Current working directory:", getwd())
  stop("Failed to load time series data")
})

# # Modify column names by splitting at '..' and keeping the first part
# colnames(time_series_data) <- sapply(colnames(time_series_data), function(x) {
#   strsplit(x, "..", fixed = TRUE)[[1]][1]
# })

NumRow <- nrow(time_series_data)
NumCol <- ncol(time_series_data)
other_data <- as.list(time_series_data)

# Prepare data for TMB
data_in <- list()
parameters <- list()

for (i in 1:nrow(params)) {
  param_name <- params$parameter[i]
  param_source <- params$source[i]
  # Preferentially use found_value if it exists
  param_value <- if (!is.null(params$found_value[[i]])) {
    params$found_value[[i]]
  } else {
    params$value[[i]]
  }
  param_import_type <- params$import_type[i]
  
  if (param_import_type == "PARAMETER") {
    if (!is.null(param_value)) {
      parameters[[param_name]] <- param_value
    }
  } else {
    # For all other import types (DATA_SCALAR, DATA_VECTOR, etc.)
    if (param_source == "data vector" && is.null(param_value) && param_name %in% names(time_series_data)) {
      data_in[[param_name]] <- time_series_data[[param_name]]
    } else if (!is.null(param_value)) {
      data_in[[param_name]] <- param_value
    }
  }
}

# Add time variable if needed
if ("time" %in% names(data_in)) {
  # time already exists in data_in, keep it
  data_in <- c(data_in, other_data)
  time_col = 'time'
} else {
  # Add time from first column of time_series_data
  data_in$time <- time_series_data[[1]]
  time_col <- names(time_series_data)[1]
  data_in <- c(data_in, other_data)
}

# Define map function
generate_map <- function(names_list) {
  all_param <- params[params$import_type == "PARAMETER", ]
  all_param_names <- all_param$parameter
  map <- list()
  leftovers <- all_param_names[!(all_param_names %in% names_list)]
  for (name in leftovers) {
    row <- all_param[all_param$parameter == name, ]
    map[[name]] <- factor(rep(NA, ifelse(is.null(row$Dimension), 1, row$Dimension)))
  }
  return(map)
}

# Run model
n_phases <- max(params$priority, na.rm = TRUE)

cat("Starting model phases\n")
cat("Number of phases:", n_phases, "\n")

tryCatch({
  for (m in seq(1, n_phases)) {
    cat("Phase", m, "\n")
    to_fit <- params[params$priority == m & !is.na(params$priority) & params$import_type == "PARAMETER", ]$parameter
    map <- generate_map(to_fit)
    if (exists('model')) {
      parameters <- model$env$parList(fit$par)
    }
    
    model <- MakeADFun(data_in, parameters, DLL = 'model', silent = TRUE, map = map)
    
    if (is.null(model)) {
      stop("Failed to create model")
    }
    
    cat("Initial parameter values for phase", m, ":\n")
    print(model$par)
    
    fit <- nlminb(model$par, model$fn, model$gr)
    
    if (is.null(fit)) {
      stop("Failed to fit model")
    }
    
    cat("Final parameter values for phase", m, ":\n")
    print(fit$par)
    cat("Convergence message:", fit$message, "\n")
    cat("Number of iterations:", fit$iterations, "\n")
    cat("Objective function value:", fit$objective, "\n")
    
    if (any(is.nan(fit$par)) || any(is.infinite(fit$par))) {
      cat("WARNING: NaN or Inf values detected in parameters at phase", m, "\n")
    }
    
    cat("Gradient at solution for phase", m, ":\n")
    grad <- model$gr(fit$par)
    print(grad)
    
    if (any(is.nan(grad)) || any(is.infinite(grad))) {
      cat("WARNING: NaN or Inf values detected in gradient at phase", m, "\n")
    }
    
    best <- model$env$last.par.best
    model$report()
  }
  
  # FINAL PHASE
  cat("Final Phase\n")
  to_fit <- params[!is.na(params$priority) & params$import_type == "PARAMETER", ]$parameter
  map <- generate_map(to_fit)
  parameters <- model$env$parList(fit$par)
  model <- MakeADFun(data_in, parameters, DLL = 'model', silent = TRUE, map = map)
  
  cat("Initial parameter values for final phase:\n")
  print(model$par)
  
  fit <- nlminb(model$par, model$fn, model$gr)
  
  cat("Final parameter values for final phase:\n")
  print(fit$par)
  cat("Convergence message:", fit$message, "\n")
  cat("Number of iterations:", fit$iterations, "\n")
  cat("Objective function value:", fit$objective, "\n")
  
  if (any(is.nan(fit$par)) || any(is.infinite(fit$par))) {
    cat("WARNING: NaN or Inf values detected in parameters at final phase\n")
  }
  
  cat("Gradient at solution for final phase:\n")
  grad <- model$gr(fit$par)
  print(grad)
  
  if (any(is.nan(grad)) || any(is.infinite(grad))) {
    cat("WARNING: NaN or Inf values detected in gradient at final phase\n")
  }
  
  best <- model$env$last.par.best
  
}, error = function(e) {
  error_message <- paste("Error in model phases:", conditionMessage(e))
  cat(error_message, "\n")
  stop(error_message)
})

cat("All phases completed\n")

# Get and save optimized parameters
final_params <- model$env$parList(fit$par)
optimized_params <- params  # Keep original structure
for (param_name in names(final_params)) {
  idx <- which(optimized_params$parameter == param_name)
  if (length(idx) > 0) {
    optimized_params$found_value[idx] <- final_params[[param_name]]
  }
}

# Save optimized parameters
write(toJSON(optimized_params, auto_unbox = TRUE, pretty = TRUE), 
      file.path(individual_dir, 'optimized_parameters.json'))

# Extract results
report <- model$report()

# Run validation using train_test_split from metadata
train_test_split <- metadata$train_test_split
if (is.null(train_test_split)) {
  train_test_split <- 1.0  # Default to using all data if not specified
}
print("Train test split:")
print(train_test_split)
cat("\nRunning model validation...\n")
tryCatch({
  validate_model(model, fit, response_file, if (!is.null(forcing_file) && forcing_file != "") forcing_file else NULL, individual_dir, time_col, train_test_split)
}, error = function(e) {
  cat("\nError in model validation:", conditionMessage(e), "\n")
})

# Get prediction variables from response data for plotting
input_vars <- names(response_data)[grepl("_dat$", names(response_data))]

# Create mapping between input and output variables
var_mapping <- sapply(input_vars, function(x) {
  # Remove "_dat" suffix to get corresponding model output name
  sub("_dat$", "_pred", x)
})

# Calculate residuals and store them
residuals_list <- list()
plot_data_list <- list()
# time_col <- names(time_series_data)[1]  # Get the name of the first column
time_values <- time_series_data[[time_col]]  # Get the time values

# Calculate residuals for each prediction variable
input_vars <- names(response_data)[grepl("_dat$", names(response_data))]
var_mapping <- sapply(input_vars, function(x) sub("_dat$", "_pred", x))

for (input_var in input_vars) {
  output_var <- var_mapping[input_var]
  if (output_var %in% names(report)) {
    observed <- response_data[[input_var]]
    predicted <- report[[output_var]]
    residuals <- observed - predicted
    
    # Store residuals with time values
    residuals_list[[output_var]] <- list(
      time = time_values,
      residuals = residuals,
      observed = observed,
      predicted = predicted
    )
  }
}

# Save residuals to JSON file
residuals_file <- file.path(individual_dir, 'model_residuals.json')
write(toJSON(residuals_list, auto_unbox = TRUE, pretty = TRUE), residuals_file)


for (input_var in input_vars) {
  output_var <- var_mapping[input_var]
  
  # Check if the model output exists
  if (output_var %in% names(report)) {
    # Create plot data using the original time column name
    plot_data <- data.frame(
      time_series_data[time_col],  # Use original column name
      Modeled = report[[output_var]],
      Observed = response_data[[input_var]]
    )
    names(plot_data)[1] <- time_col  # Ensure first column has original name
    plot_data_list[[output_var]] <- as.list(plot_data)  # Convert to list
    
    # Get variable description from column name
    var_desc <- ""
    if (grepl("\\(.*\\)", colnames(response_data)[which(names(response_data) == input_var)])) {
      var_desc <- sub(".*\\((.*)\\).*", "\\1", colnames(response_data)[which(names(response_data) == input_var)])
    }
    
    # Create plot
    p <- ggplot(plot_data, aes_string(x = time_col)) +
      geom_line(aes(y = Modeled, color = "Modeled"), linewidth = 1, alpha = 0.7) +
      geom_point(aes(y = Observed), color = "grey50", size = 3) +
      labs(title = paste("Modeled vs Observed", output_var),
           subtitle = var_desc,
           y = var_desc,
           color = "Type") +
      scale_color_manual(values = c("Modeled" = "#1f77b4")) +  # Blue from tab10
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5, size = 14),
        plot.subtitle = element_text(size = 12),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 10),
        panel.grid = element_blank(),  # Remove grid lines
        legend.position = "right",
        legend.text = element_text(size = 10),
        legend.title = element_text(size = 12)
      )
    
    # Save plots
    plot_filename <- file.path(individual_dir, paste0(output_var, "_comparison"))
    ggsave(paste0(plot_filename, ".png"), plot = p, width = 10, height = 6, dpi = 300)
    ggsave(paste0(plot_filename, ".pdf"), plot = p, width = 10, height = 6)
  }
}

# Print report contents for debugging
cat("\nDEBUG: Report contents:\n")
print(str(report))
cat("\nDEBUG: Available variables in report:\n")
print(names(report))

# Check predictions and calculate objective function
cat("\nChecking predictions and calculating objective function...\n")

# Function to check if predictions are valid
check_predictions <- function() {
  # Get prediction variables only from response data
  prediction_vars <- names(response_data)[grepl("_dat$", names(response_data))]
  prediction_mapping <- sapply(prediction_vars, function(x) sub("_dat$", "_pred", x))
  
  for (input_var in prediction_vars) {
    output_var <- prediction_mapping[input_var]
    cat("\nDEBUG: Processing", output_var, "\n")
    if (output_var %in% names(report)) {
      pred_values <- report[[output_var]]
      cat("DEBUG: First few predictions:", head(pred_values), "\n")
      cat("DEBUG: Summary statistics:\n")
      print(summary(pred_values))
      cat("DEBUG: All zeros?", all(pred_values == 0), "\n")
      cat("DEBUG: Any infinities?", any(is.infinite(pred_values)), "\n")
      if (length(unique(pred_values)) == 1 || any(is.infinite(pred_values)) || any(is.nan(pred_values))) {
        cat("WARNING: ", output_var, " contains all zeros, infinities, or NaN values\n")
        return(FALSE)
      }
    } else {
      cat("DEBUG:", output_var, "not found in report\n")
      cat("DEBUG: Available variables:", paste(names(report), collapse=", "), "\n")
      return(TRUE) # temporary fix until a better data provenance solution is developed
    }
  }
  return(TRUE)
}

objective_fn <- tryCatch({
  # First check if predictions are valid
  if (!check_predictions()) {
    cat("Invalid predictions detected - applying penalty\n")
    NA  # Return NA
  } else {
    # Get prediction variables only from response data
    prediction_vars <- names(response_data)[grepl("_dat$", names(response_data))]
    prediction_mapping <- sapply(prediction_vars, function(x) sub("_dat$", "_pred", x))
    
    cat("\nDEBUG: Using the following variables for objective function:\n")
    for(i in seq_along(prediction_vars)) {
      cat(sprintf("%s -> %s\n", prediction_vars[i], prediction_mapping[i]))
    }
    
    # If predictions look valid, calculate normal objective function
    cat("\nDEBUG: Calculating MSE for prediction variables only:\n")
    mse_values <- sapply(prediction_vars, function(input_var) {
      output_var <- prediction_mapping[input_var]
      if (output_var %in% names(report)) {
        obs_values <- response_data[[input_var]]
        pred_values <- report[[output_var]]
        cat("\nDEBUG:", output_var, "\n")
        cat("Observed:", head(obs_values), "...\n")
        cat("Predicted:", head(pred_values), "...\n")
        
        # Calculate standard deviation of observed values
        obs_sd <- sd(obs_values, na.rm = TRUE)
        if (obs_sd == 0) {
          # If sd is 0, use raw values to avoid division by zero
          mse <- mean((obs_values - pred_values)^2, na.rm = TRUE)
        } else {
          # Normalize both observed and predicted by observed sd before calculating MSE
          mse <- mean(((obs_values/obs_sd) - (pred_values/obs_sd))^2, na.rm = TRUE)
        }
        cat("MSE (normalized):", mse, "\n")
        mse
        
        # Original implementation without normalization
        # mse <- mean((obs_values - pred_values)^2, na.rm = TRUE)
        # cat("MSE:", mse, "\n")
        # mse
      } else {
        NA
      }
    })
    
    cat("\nDEBUG: MSE values:\n")
    print(mse_values)
    
    final_obj <- mean(mse_values, na.rm = TRUE)
    cat("\nDEBUG: Final objective value (mean of MSEs):", final_obj, "\n")
    final_obj
  }
}, error = function(e) {
  cat("Error in objective function calculation:", conditionMessage(e), "\n")
  NA  # Return penalty value on error
})

cat("\nFinal objective function value:", objective_fn, "\n")

# Prepare JSON report
json_report <- list(
  status = "SUCCESS",
  objective_value = objective_fn,
  model_summary = capture.output(summary(fit)),
  model_report = as.list(report),  # Convert to list
  plot_data = plot_data_list
)

# Write report to stdout for run_model.py to capture
cat("\nJSON_REPORT_START\n")
cat(toJSON(json_report, auto_unbox = TRUE, pretty = TRUE))
cat("\nJSON_REPORT_END\n")
