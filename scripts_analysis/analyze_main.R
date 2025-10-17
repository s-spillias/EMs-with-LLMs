# Load required libraries
library(readr)
library(jsonlite)
library(dplyr)
library(purrr)
library(tibble)
library(fs)

source("scripts_analysis/population_utils.R")
source("scripts_analysis/analyze.R")
# source("scripts_analysis/terminal_figure.r")
source("scripts_analysis/figure_status.R")
source("scripts_analysis/iteration_figures.R")
source("scripts_analysis/combine_validation_plots.R")
source("scripts_analysis/analyze_NPZ_pops.r")


# ---------------- Helpers ----------------

# Determine NPZ vs COTS from response_file
get_population_category <- function(population_data) {
  resp <- population_data$response_file
  if (is.null(resp)) {
    return("Unknown")
  }
  if (grepl("NPZ", resp, ignore.case = TRUE)) "NPZ" else "COTS"
}

# Extract the best individual (and objective) from current_best_performers
safe_best_performer <- function(population_data) {
  performers <- population_data$current_best_performers
  if (is.null(performers)) {
    return(list(individual = NA_character_, objective_value = NA_real_))
  }
  # Handle data.frame and list-of-lists
  if (is.data.frame(performers)) {
    if (!all(c("objective_value", "individual") %in% names(performers))) {
      return(list(individual = NA_character_, objective_value = NA_real_))
    }
    obj_vals <- suppressWarnings(as.numeric(performers$objective_value))
    idx <- which.min(obj_vals)
    indiv <- performers$individual[idx]
    return(list(
      individual = gsub("^INDIVIDUAL_", "", indiv),
      objective_value = obj_vals[idx]
    ))
  } else if (is.list(performers)) {
    if (length(performers) == 0) {
      return(list(individual = NA_character_, objective_value = NA_real_))
    }
    obj_vals <- suppressWarnings(sapply(performers, function(p) as.numeric(p$objective_value)))
    idx <- which.min(obj_vals)
    indiv <- performers[[idx]]$individual
    return(list(
      individual = gsub("^INDIVIDUAL_", "", indiv),
      objective_value = obj_vals[idx]
    ))
  }
  list(individual = NA_character_, objective_value = NA_real_)
}

# ---------------- Inputs & Setup ----------------

# Get population directories
pop_info <- get_population_dirs()
population_dirs <- pop_info$dirs
population_numbers <- pop_info$numbers

cat(sprintf(
  "Found %d populations: %s\n",
  length(population_numbers),
  paste(population_numbers, collapse = ", ")
))

# Ensure results directory exists
results_dir <- ensure_results_dir()

# ---------------- Main analysis workflow ----------------
main <- function() {
  # First analyze all populations
  results <- analyze_populations(population_numbers)

  # Store iteration statistics for all populations
  iteration_stats <- list()
  # Per-population summary (category & out-of-sample context)
  per_population_summary <- list()
  # For status plot
  all_population_data <- list()

  # Process each population
  for (i in seq_along(population_numbers)) {
    pop_num <- population_numbers[i]
    cat(sprintf(
      "\nGenerating figures for population %04d (%d/%d)...\n",
      pop_num, i, length(population_numbers)
    ))

    population_data <- read_population_metadata(pop_num)

    # Determine category and train_test_split
    category <- get_population_category(population_data)
    tts <- suppressWarnings(as.numeric(population_data$train_test_split))
    if (is.na(tts)) tts <- NA_real_

    # Best performer for this population (from metadata)
    best_perf <- safe_best_performer(population_data)

    # Capture per-population summary
    llm_value <- if (!is.null(population_data$llm_choice)) {
      as.character(population_data$llm_choice)
    } else {
      "Unknown"
    }
    
    per_population_summary[[as.character(pop_num)]] <- list(
      population = pop_num,
      category = category,
      train_test_split = tts,
      best_individual = best_perf$individual,
      best_objective = best_perf$objective_value,
      llm = llm_value
    )

    # Get model report files and extract iterations
    model_report_files <- dir_ls(population_dirs[i], regexp = "/model_report.json", recurse = TRUE)
    model_iterations <- map_dfr(model_report_files, function(path) {
      tryCatch(
        {
          report <- fromJSON(path)
          individual <- gsub("INDIVIDUAL_", "", basename(dirname(path)))
          tibble(
            individual = individual,
            num_iterations = length(report$iterations)
          )
        },
        error = function(e) {
          cat("Error reading", path, ":", e$message, "\n")
          return(NULL)
        }
      )
    })

    metadata_files <- dir_ls(population_dirs[i], regexp = "/metadata.json", recurse = TRUE)

    # Read all individual metadata and join with iterations data
    all_metadata <- sapply(metadata_files, FUN = read_individual_metadata) %>%
      imap_dfr(~ tibble(
        file_path = .x$file_path,
        lineage = list(.x$lineage) %>% lapply(function(x) gsub("INDIVIDUAL_", "", x)),
        individual = gsub("INDIVIDUAL_", "", tail(strsplit(.x$file_path, "/")[[1]], 2)[1])
      )) %>%
      rowwise() %>%
      mutate(
        lineage_vec = list(if (length(lineage) == 0 || is.null(lineage[[1]])) character(0) else lineage[[1]]),
        generation = length(lineage_vec) + 1, # Add 1 since generation starts at 1
        founder = ifelse(length(lineage_vec) == 0, individual, lineage_vec[1]),
        parent = ifelse(length(lineage_vec) > 0, tail(lineage_vec, 1), NA),
        objective_value = get_objective_value_pop(population_data, individual)
      ) %>%
      ungroup() %>%
      left_join(model_iterations, by = "individual")

    # Calculate summary statistics for this population
    summary_stats <- all_metadata %>%
      group_by(generation) %>%
      summarize(
        mean_objective_value = mean(objective_value, na.rm = TRUE),
        min_objective_value = min(objective_value, na.rm = TRUE),
        max_objective_value = max(objective_value, na.rm = TRUE),
        mean_iterations = mean(num_iterations, na.rm = TRUE),
        min_iterations = min(num_iterations, na.rm = TRUE),
        max_iterations = max(num_iterations, na.rm = TRUE),
        .groups = "drop"
      )

    # Store population data for status plot
    all_population_data[[as.character(pop_num)]] <- list(
      data = population_data,
      number = pop_num
    )

    # Store iteration statistics
    iteration_stats[[as.character(pop_num)]] <- list(
      mean_iterations = mean(all_metadata$num_iterations, na.rm = TRUE),
      min_iterations = suppressWarnings(min(all_metadata$num_iterations, na.rm = TRUE)),
      max_iterations = suppressWarnings(max(all_metadata$num_iterations, na.rm = TRUE)),
      by_generation = summary_stats %>%
        dplyr::select(generation, mean_iterations, min_iterations, max_iterations) %>%
        as.list()
    )
  }

  # Add iteration statistics to results
  results$iteration_statistics <- iteration_stats

  # ---------------- Category Segregation & Out-of-Sample Summary ----------------
  pop_summary_df <- bind_rows(lapply(per_population_summary, as_tibble))

  npz_df <- pop_summary_df %>% filter(category == "NPZ")
  cots_df <- pop_summary_df %>% filter(category == "COTS")

  # Compute best per category (by best_objective at population level)
  best_npz <- if (nrow(npz_df) > 0 && any(is.finite(npz_df$best_objective))) {
    npz_df %>%
      filter(best_objective == min(best_objective, na.rm = TRUE)) %>%
      slice(1)
  } else {
    NULL
  }

  best_cots <- if (nrow(cots_df) > 0 && any(is.finite(cots_df$best_objective))) {
    cots_df %>%
      filter(best_objective == min(best_objective, na.rm = TRUE)) %>%
      slice(1)
  } else {
    NULL
  }

  # Out-of-sample: train_test_split < 1.0
  oos_df <- pop_summary_df %>% filter(is.finite(train_test_split), train_test_split < 1.0)
  best_oos <- if (nrow(oos_df) > 0 && any(is.finite(oos_df$best_objective))) {
    oos_df %>%
      filter(best_objective == min(best_objective, na.rm = TRUE)) %>%
      slice(1)
  } else {
    NULL
  }

  # Build/augment stats structure
  if (is.null(results$stats)) results$stats <- list()
  results$stats$category_summary <- list(
    NPZ = list(
      count = nrow(npz_df),
      populations = npz_df$population,
      best = if (!is.null(best_npz)) {
        list(
          population = best_npz$population[[1]],
          individual = best_npz$best_individual[[1]],
          objective_value = best_npz$best_objective[[1]],
          train_test_split = best_npz$train_test_split[[1]]
        )
      } else {
        NULL
      }
    ),
    COTS = list(
      count = nrow(cots_df),
      populations = cots_df$population,
      best = if (!is.null(best_cots)) {
        list(
          population = best_cots$population[[1]],
          individual = best_cots$best_individual[[1]],
          objective_value = best_cots$best_objective[[1]],
          train_test_split = best_cots$train_test_split[[1]]
        )
      } else {
        NULL
      }
    )
  )

  results$stats$best_out_of_sample <- if (!is.null(best_oos)) {
    list(
      population = best_oos$population[[1]],
      category = best_oos$category[[1]],
      individual = best_oos$best_individual[[1]],
      objective_value = best_oos$best_objective[[1]],
      train_test_split = best_oos$train_test_split[[1]]
    )
  } else {
    NULL
  }

  # Store per-population overview for downstream use
  results$populations_overview <- per_population_summary

  # ---------------- Create Combined Validation Plot for Best OOS ----------------
  if (!is.null(best_oos)) {
    cat(sprintf("\nCreating combined validation plot for best out-of-sample performer...\n"))
    cat(sprintf("Population %04d (%s) | Individual %s | Objective %.6f | train_test_split = %.3f\n",
                best_oos$population[[1]], best_oos$category[[1]], best_oos$best_individual[[1]],
                best_oos$best_objective[[1]], best_oos$train_test_split[[1]]))
    
    # Find the population directory
    pop_idx <- which(population_numbers == best_oos$population[[1]])
    if (length(pop_idx) > 0) {
      best_individual_dir <- file.path(population_dirs[pop_idx], paste0("INDIVIDUAL_", best_oos$best_individual[[1]]))
      validation_file <- file.path(best_individual_dir, "validation_report.json")
      
      if (file.exists(validation_file)) {
        tryCatch({
          create_combined_validation_plot(validation_file, "Figures")
          cat("Combined validation plot saved to Figures/combined_validation.*\n")
        }, error = function(e) {
          cat("Error creating combined validation plot:", e$message, "\n")
        })
      } else {
        cat(sprintf("Warning: Validation file not found at %s\n", validation_file))
      }
    }
  } else {
    cat("\nNo out-of-sample populations found - skipping combined validation plot.\n")
  }

  # ---------------- Experimental Design Completion Analysis ----------------
  # Goal: 3 replicates each for LLM × Category × train_test_split=1
  # LLMs: gpt-5, claude, gemini
  # Categories: NPZ, COTS
  # Each should have 10 generations unless converged
  
  llm_mapping <- list(
    "openrouter:openai/gpt-5" = "GPT-5",
    "openrouter:anthropic/claude-sonnet-4.5" = "Claude",
    "openrouter:google/gemini-2.5-pro" = "Gemini"
  )
  
  # Filter to train_test_split = 1 populations only
  design_df <- pop_summary_df %>%
    filter(train_test_split == 1) %>%
    mutate(llm_short = sapply(llm, function(x) {
      if (x %in% names(llm_mapping)) llm_mapping[[x]] else x
    }))
  
  # Count by LLM and Category
  design_counts <- design_df %>%
    group_by(llm_short, category) %>%
    summarise(
      current_count = n(),
      populations = list(population),
      .groups = "drop"
    ) %>%
    mutate(
      target_count = 3,
      needed = pmax(0, target_count - current_count),
      status = ifelse(current_count >= target_count, "Complete", sprintf("Need %d more", needed))
    )
  
  # Calculate total missing runs
  total_needed <- sum(design_counts$needed)
  total_complete <- sum(design_counts$current_count >= design_counts$target_count)
  total_cells <- nrow(design_counts)
  
  # Build experimental design summary
  results$experimental_design <- list(
    target_design = "3 replicates × 3 LLMs × 2 Categories × train_test_split=1",
    target_populations = 18,
    current_populations = nrow(design_df),
    populations_needed = total_needed,
    cells_complete = sprintf("%d/%d", total_complete, total_cells),
    completion_status = design_counts %>%
      rowwise() %>%
      mutate(populations_list = paste(sprintf("POP_%04d", populations[[1]]), collapse = ", ")) %>%
      ungroup() %>%
      dplyr::select(llm_short, category, current_count, target_count, needed, status, populations_list)
  )
  
  # ---------------- Save updated results to JSON ----------------
  write_json(results, file.path(results_dir, "populations_analysis.json"), pretty = TRUE, auto_unbox = TRUE)

  # ---------------- Human-readable report ----------------
  report_lines <- c(
    "Population Analysis Report\n",
    "=======================\n\n",
    "Iteration Statistics:\n",
    "-------------------\n"
  )

  for (pop_num in population_numbers) {
    stats <- iteration_stats[[as.character(pop_num)]]
    report_lines <- c(report_lines, sprintf(
      "\nPopulation %04d:\n  Mean iterations: %.2f\n  Min iterations: %s\n  Max iterations: %s\n",
      pop_num,
      stats$mean_iterations,
      ifelse(is.infinite(stats$min_iterations), "NA", as.character(stats$min_iterations)),
      ifelse(is.infinite(stats$max_iterations), "NA", as.character(stats$max_iterations))
    ))
  }

  # Append category summary
  report_lines <- c(
    report_lines,
    "\nCategory Summary:\n",
    "-----------------\n",
    sprintf(
      "NPZ populations: %d (%s)\n",
      nrow(npz_df),
      ifelse(nrow(npz_df) > 0, paste(sprintf("%04d", npz_df$population), collapse = ", "), "none")
    ),
    sprintf(
      "COTS populations: %d (%s)\n",
      nrow(cots_df),
      ifelse(nrow(cots_df) > 0, paste(sprintf("%04d", cots_df$population), collapse = ", "), "none")
    )
  )

  if (!is.null(best_npz)) {
    report_lines <- c(
      report_lines,
      sprintf(
        "\nBest NPZ: Population %04d | Individual %s | Objective %.6f | train_test_split = %s\n",
        best_npz$population[[1]], best_npz$best_individual[[1]],
        best_npz$best_objective[[1]], as.character(best_npz$train_test_split[[1]])
      )
    )
  }
  if (!is.null(best_cots)) {
    report_lines <- c(
      report_lines,
      sprintf(
        "Best COTS: Population %04d | Individual %s | Objective %.6f | train_test_split = %s\n",
        best_cots$population[[1]], best_cots$best_individual[[1]],
        best_cots$best_objective[[1]], as.character(best_cots$train_test_split[[1]])
      )
    )
  }

  # Append out-of-sample summary
  report_lines <- c(
    report_lines,
    "\nOut-of-Sample Summary (train_test_split < 1.0):\n",
    "----------------------------------------------\n"
  )
  if (!is.null(best_oos)) {
    report_lines <- c(
      report_lines,
      sprintf(
        "Best OOS: Population %04d (%s) | Individual %s | Objective %.6f | train_test_split = %.3f\n",
        best_oos$population[[1]], best_oos$category[[1]], best_oos$best_individual[[1]],
        best_oos$best_objective[[1]], best_oos$train_test_split[[1]]
      )
    )
  } else {
    report_lines <- c(report_lines, "No out-of-sample populations found.\n")
  }

  # Append experimental design completion
  report_lines <- c(
    report_lines,
    "\nExperimental Design Completion (train_test_split=1 only):\n",
    "--------------------------------------------------------\n",
    sprintf("Target: %s\n", results$experimental_design$target_design),
    sprintf("Progress: %d/%d populations (%d needed)\n",
            results$experimental_design$current_populations,
            results$experimental_design$target_populations,
            results$experimental_design$populations_needed),
    sprintf("Cells complete: %s\n\n", results$experimental_design$cells_complete)
  )
  
  # Add detailed breakdown by LLM and Category
  for (i in seq_along(design_counts$llm_short)) {
    report_lines <- c(
      report_lines,
      sprintf(
        "%s × %s: %d/%d replicate%s %s\n  Populations: %s\n",
        design_counts$llm_short[i],
        design_counts$category[i],
        design_counts$current_count[i],
        design_counts$target_count[i],
        ifelse(design_counts$target_count[i] > 1, "s", ""),
        design_counts$status[i],
        paste(sprintf("%04d", design_counts$populations[[i]]), collapse = ", ")
      )
    )
  }

  report_lines <- c(
    report_lines,
    "\nAnalysis Files:\n",
    "---------------\n",
    " - populations_analysis.json (Detailed analysis data)\n",
    " - Figures/ (Visualizations for each population)\n"
  )

  writeLines(report_lines, file.path(results_dir, "populations_analysis_report.txt"))

  cat("\nAnalysis completed. Please check Manuscript/Results directory for:\n")
  cat(" - populations_analysis_report.txt (Main statistical report)\n")
  cat(" - populations_analysis.json (Detailed analysis data with NPZ/COTS and OOS summaries)\n")
  cat(" - Figures/ (Visualizations for each population)\n")

  # Create LLM comparison plot
  cat("\nGenerating LLM comparison plot...\n")
  result <- system2("python3",
    args = "scripts_analysis/plot_best_predictions.py",
    stdout = TRUE,
    stderr = TRUE
  )
  cat(paste(result, collapse = "\n"), "\n")

  # Run citation analysis
  cat("\nRunning citation analysis...\n")
  citation_result <- system2("python3",
    args = "scripts_analysis/analyze_citations.py",
    stdout = TRUE,
    stderr = TRUE
  )
  cat(paste(citation_result, collapse = "\n"), "\n")

  # Create combined status plot for all populations
  if (length(all_population_data) > 0) {
    create_population_status_plot(all_population_data)
  }

  # Create iteration figures
  cat("\nGenerating iteration figures...\n")
  create_iteration_figures(population_dirs)

  # Extract best models
  cat("\nExtracting best performing models...\n")
  source("scripts_analysis/extract_best_models.R")
}

# Run main function if script is run directly
if (sys.nframe() == 0) {
  main()
}
