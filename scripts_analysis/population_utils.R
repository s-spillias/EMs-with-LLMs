# Load required libraries
library(jsonlite)
library(dplyr)
library(fs)

# Function to get objective value from population metadata
get_objective_value_pop <- function(population_data, individual_id) {
  # Check current best performers first
  best_performers <- population_data$current_best_performers
  if (individual_id %in% best_performers$individual) {
    return(best_performers$objective_value[best_performers$individual == individual_id])
  }
  
  # Search through generations
  generations <- population_data$generations
  for (i in seq_along(generations$generation_number)) {
    best_individuals <- generations$best_individuals[[i]]
    if (individual_id %in% best_individuals$individual) {
      return(best_individuals$objective_value[best_individuals$individual == individual_id])
    }
  }
  
  return(NA)
}

# Function to read population metadata with enhanced error handling
read_population_metadata <- function(population_number) {
  file_path <- sprintf("POPULATIONS/POPULATION_%04d/population_metadata.json", population_number)
  
  tryCatch({
    # Read the JSON file
    data <- fromJSON(file_path)
    
    # Ensure required fields exist
    if (is.null(data$generations)) {
      data$generations <- data.frame(
        generation_number = integer(0),
        best_individuals = list()
      )
    }
    
    # Convert generations to data frame if it's a list
    if (is.list(data$generations) && !is.data.frame(data$generations)) {
      # Extract generation information
      gen_data <- lapply(seq_along(data$generations), function(i) {
        gen <- data$generations[[i]]
        if (is.null(gen$best_individuals)) {
          gen$best_individuals <- list(data.frame(
            individual = character(0),
            objective_value = numeric(0)
          ))
        }
        data.frame(
          generation_number = i,
          best_individuals = I(list(gen$best_individuals))
        )
      })
      data$generations <- do.call(rbind, gen_data)
    }
    
    # Ensure current_best_performers exists
    if (is.null(data$current_best_performers)) {
      data$current_best_performers <- data.frame(
        individual = character(0),
        objective_value = numeric(0)
      )
    }
    
    # Add other required fields if missing
    if (is.null(data$start_time)) data$start_time <- NA
    if (is.null(data$end_time)) data$end_time <- NA
    if (is.null(data$total_runtime)) data$total_runtime <- 0
    if (is.null(data$converged)) data$converged <- FALSE
    
    return(data)
    
  }, error = function(e) {
    warning(sprintf("Error reading population %04d metadata: %s", population_number, e$message))
    # Return minimal valid structure
    list(
      generations = data.frame(
        generation_number = integer(0),
        best_individuals = I(list())
      ),
      current_best_performers = data.frame(
        individual = character(0),
        objective_value = numeric(0)
      ),
      start_time = NA,
      end_time = NA,
      total_runtime = 0,
      converged = FALSE
    )
  })
}

# Function to read individual metadata
read_individual_metadata <- function(file_path) {
  tryCatch({
    data <- fromJSON(file_path)
    # Ensure all required fields are present
    required_fields <- c("lineage")
    missing_fields <- setdiff(required_fields, names(data))
    if (length(missing_fields) > 0) {
      cat("Warning: Missing fields in", file_path, ":", paste(missing_fields, collapse = ", "), "\n")
      # Add missing fields with NA values
      for (field in missing_fields) {
        data[[field]] <- NA
      }
    }
    data$file_path <- file_path
    return(data)
  }, error = function(e) {
    cat("Error reading", file_path, ":", e$message, "\n")
    return(NULL)
  })
}

# Function to ensure results directory exists
ensure_results_dir <- function() {
  results_dir <- "Results"
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
  return(results_dir)
}

# Function to get population directories
get_population_dirs <- function() {
  population_dirs <- dir_ls("POPULATIONS", regexp = "POPULATION_\\d{4}")
  population_numbers <- as.numeric(gsub(".*POPULATION_(\\d{4}).*", "\\1", population_dirs))
  list(
    dirs = population_dirs,
    numbers = population_numbers
  )
}
