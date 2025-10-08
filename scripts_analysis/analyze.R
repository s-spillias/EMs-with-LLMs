library(lubridate)
source("scripts_analysis/population_utils.R")

analyze_populations <- function(population_numbers) {
  # Get population directories
  pop_info <- get_population_dirs()
  population_dirs <- sprintf("POPULATIONS/POPULATION_%04d", population_numbers)
  
  # Load population metadata for each population
  populations_data <- lapply(population_numbers, function(num) {
    read_population_metadata(num)
  })
  names(populations_data) <- basename(population_dirs)
  
  calculate_population_statistics <- function(metadata) {
    # Convert generations to a list if it's a data frame
    generations <- if(is.data.frame(metadata$generations)) {
      split(metadata$generations, seq_len(nrow(metadata$generations)))
    } else {
      metadata$generations
    }
    
    # Extract best objectives with enhanced error handling
    best_objectives <- numeric(length(generations))
    for (i in seq_along(generations)) {
      gen <- generations[[i]]
      tryCatch({
        if (!is.null(gen$best_individuals)) {
          # Handle different data structures
          if (is.data.frame(gen$best_individuals)) {
            if (nrow(gen$best_individuals) > 0 && "objective_value" %in% names(gen$best_individuals)) {
              obj_value <- gen$best_individuals$objective_value[1]
            } else {
              obj_value <- NA
            }
          } else if (is.list(gen$best_individuals) && length(gen$best_individuals) > 0) {
            if (is.data.frame(gen$best_individuals[[1]]) && "objective_value" %in% names(gen$best_individuals[[1]])) {
              obj_value <- gen$best_individuals[[1]]$objective_value[1]
            } else if (is.numeric(gen$best_individuals[[1]])) {
              obj_value <- gen$best_individuals[[1]]
            } else {
              obj_value <- NA
            }
          } else {
            obj_value <- NA
          }
          # Ensure numeric conversion
          best_objectives[i] <- suppressWarnings(as.numeric(obj_value))
          if (is.na(best_objectives[i])) best_objectives[i] <- Inf
        } else {
          best_objectives[i] <- Inf
        }
      }, error = function(e) {
        cat("\nError processing generation", i, ":", conditionMessage(e), "\n")
        best_objectives[i] <- Inf
      })
    }
    
    # Calculate time per generation
    start_time <- ymd_hms(metadata$start_time)
    end_time <- ymd_hms(metadata$end_time)
    total_runtime_minutes <- metadata$total_runtime / 60
    time_per_generation <- total_runtime_minutes / length(generations)
    
    # Safely handle culled and broken individuals
    total_culled <- tryCatch({
      sum(sapply(generations, function(gen) {
        if (is.null(gen$culled_individuals)) 0 else length(gen$culled_individuals)
      }))
    }, error = function(e) {
      cat("Warning: Error counting culled individuals:", conditionMessage(e), "\n")
      0
    })
    
    total_broken <- tryCatch({
      sum(sapply(generations, function(gen) {
        if (is.null(gen$broken_individuals)) 0 else length(gen$broken_individuals)
      }))
    }, error = function(e) {
      cat("Warning: Error counting broken individuals:", conditionMessage(e), "\n")
      0
    })
    
    # Ensure objective values are numeric
    best_objectives <- as.numeric(best_objectives)
    best_objectives[is.na(best_objectives)] <- Inf
    
    list(
      total_generations = length(generations),
      converged = metadata$converged %||% FALSE,
      final_best_objective = tail(best_objectives, 1),
      # Calculate generation-by-generation improvements
      improvements_per_gen = diff(best_objectives),
      # Average improvement per generation (negative means getting better)
      mean_improvement_rate = mean(diff(best_objectives), na.rm = TRUE),
      # Number of generations until best objective is reached
      generations_to_best = which.min(best_objectives),
      # Percentage of generations that showed improvement
      improvement_frequency = mean(diff(best_objectives) < 0, na.rm = TRUE) * 100,
      best_objectives_progression = best_objectives,
      total_culled = total_culled,
      total_broken = total_broken,
      runtime_minutes = total_runtime_minutes,
      time_per_generation = time_per_generation,
      llm_choice = metadata$llm_choice %||% "unknown",
      rag_choice = metadata$rag_choice %||% "unknown",
      embed_choice = metadata$embed_choice %||% "unknown"
    )
  }
  
    # Calculate statistics for each population
    all_stats <- list()
    best_objectives_all <- list()
    final_objectives <- numeric()
    llm_stats <- list()
  
    for (pop_id in names(populations_data)) {
      tryCatch({
        stats <- calculate_population_statistics(populations_data[[pop_id]])
        all_stats[[pop_id]] <- stats
        best_objectives_all[[pop_id]] <- stats$best_objectives_progression
        # Ensure final objective is numeric
        final_obj <- as.numeric(stats$final_best_objective)
        if (is.na(final_obj)) final_obj <- Inf
        final_objectives <- c(final_objectives, final_obj)
      }, error = function(e) {
        cat(sprintf("Warning: Error processing population %s: %s\n", 
                   pop_id, conditionMessage(e)))
        # Add placeholder stats for failed population
        all_stats[[pop_id]] <- list(
          total_generations = 0,
          converged = FALSE,
          final_best_objective = Inf,
          improvements_per_gen = numeric(0),
          mean_improvement_rate = NA,
          generations_to_best = NA,
          improvement_frequency = 0,
          best_objectives_progression = numeric(0),
          total_culled = 0,
          total_broken = 0,
          runtime_minutes = 0,
          time_per_generation = 0,
          llm_choice = "unknown",
          rag_choice = "unknown",
          embed_choice = "unknown"
        )
        final_objectives <- c(final_objectives, Inf)
      })
    
    # Group statistics by LLM
    llm <- stats$llm_choice
    if (is.null(llm_stats[[llm]])) {
      llm_stats[[llm]] <- list(
        populations = character(),
        final_objectives = numeric(),
        improvements = numeric(),
        runtimes = numeric(),
        times_per_generation = numeric()
      )
    }
    
    llm_stats[[llm]]$populations <- c(llm_stats[[llm]]$populations, pop_id)
    llm_stats[[llm]]$final_objectives <- c(llm_stats[[llm]]$final_objectives, stats$final_best_objective)
    llm_stats[[llm]]$improvements <- c(llm_stats[[llm]]$improvements, stats$mean_improvement_rate)
    llm_stats[[llm]]$runtimes <- c(llm_stats[[llm]]$runtimes, stats$runtime_minutes)
    llm_stats[[llm]]$times_per_generation <- c(llm_stats[[llm]]$times_per_generation, stats$time_per_generation)
  }
  
    # Find best performing population, handling NA/Inf values
    best_pop_id <- names(all_stats)[which.min(sapply(all_stats, function(x) {
      obj <- as.numeric(x$final_best_objective)
      if (is.na(obj)) Inf else obj
    }))]
  
  # Calculate LLM-specific statistics
  for (llm in names(llm_stats)) {
    llm_stats[[llm]] <- c(llm_stats[[llm]], list(
      count = length(llm_stats[[llm]]$populations),
      mean_final_objective = mean(llm_stats[[llm]]$final_objectives),
      std_final_objective = sd(llm_stats[[llm]]$final_objectives),
      mean_improvement = mean(llm_stats[[llm]]$improvements),
      mean_runtime = mean(llm_stats[[llm]]$runtimes),
      mean_time_per_generation = mean(llm_stats[[llm]]$times_per_generation),
      std_time_per_generation = sd(llm_stats[[llm]]$times_per_generation)
    ))
  }
  
  # Get best individual info from the last generation
  best_gen <- populations_data[[best_pop_id]]$generations
  best_gen_df <- best_gen[best_gen$generation_number == max(best_gen$generation_number), ]
  best_individual <- best_gen_df$best_individuals[[1]][1, ]
  
    # Calculate aggregate statistics with error handling
    aggregate_stats <- list(
      total_populations = length(populations_data),
      mean_final_objective = mean(final_objectives, na.rm = TRUE),
      std_final_objective = tryCatch({
        valid_objectives <- final_objectives[!is.na(final_objectives) & is.finite(final_objectives)]
        if (length(valid_objectives) >= 2) {
          sd(valid_objectives)
        } else {
          NA
        }
      }, error = function(e) {
        cat("Warning: Error calculating std_final_objective:", conditionMessage(e), "\n")
        NA
      }),
    best_performing_population = list(
      population_id = best_pop_id,
      final_objective = all_stats[[best_pop_id]]$final_best_objective,
      individual_id = best_individual$individual
    ),
    mean_improvement_rate = mean(sapply(all_stats, function(x) x$mean_improvement_rate)),
    mean_generations_to_best = mean(sapply(all_stats, function(x) x$generations_to_best)),
    mean_improvement_frequency = mean(sapply(all_stats, function(x) x$improvement_frequency)),
    mean_runtime_minutes = mean(sapply(all_stats, function(x) x$runtime_minutes)),
    mean_time_per_generation = mean(sapply(all_stats, function(x) x$time_per_generation)),
    std_time_per_generation = sd(sapply(all_stats, function(x) x$time_per_generation)),
    convergence_rate = mean(sapply(all_stats, function(x) x$converged)),
    llm_statistics = llm_stats,
    population_statistics = all_stats
  )
  
  # Ensure results directory exists
  results_dir <- ensure_results_dir()
  
  # Save analysis results
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  analysis_file <- file.path(results_dir, "populations_analysis.json")
  write_json(aggregate_stats, analysis_file, pretty = TRUE, auto_unbox = TRUE)
  
  # Generate summary report
  report_file <- file.path(results_dir, "populations_analysis_report.txt")
  
  # Write report
  report_lines <- c(
    "Population Analysis Summary\n",
    "=======================\n\n",
    sprintf("Analysis Date: %s\n\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
    
    "Overall Statistics\n",
    "-----------------\n",
    sprintf("Total Populations Analyzed: %d", aggregate_stats$total_populations),
    sprintf("Mean Final Objective Value: %.4f", aggregate_stats$mean_final_objective),
    sprintf("Standard Deviation: %.4f", aggregate_stats$std_final_objective),
    sprintf("Best Performing Population: %s", aggregate_stats$best_performing_population$population_id),
    sprintf("Best Final Objective Value: %.4f", aggregate_stats$best_performing_population$final_objective),
    sprintf("Mean Improvement Rate per Generation: %.4f", aggregate_stats$mean_improvement_rate),
    sprintf("Mean Generations to Best: %.1f", aggregate_stats$mean_generations_to_best),
    sprintf("Mean Improvement Frequency: %.1f%%", aggregate_stats$mean_improvement_frequency),
    sprintf("Mean Runtime (minutes): %.2f", aggregate_stats$mean_runtime_minutes),
    sprintf("Convergence Rate: %.1f%%", aggregate_stats$convergence_rate * 100),
    "\n",
    
    "LLM Performance Analysis\n",
    "----------------------\n"
  )
  
  # Add LLM statistics with more detailed analysis
  for (llm in names(llm_stats)) {
    stats <- llm_stats[[llm]]
    llm_lines <- c(
      sprintf("\n%s:", llm),
      sprintf("  Populations: %s", paste(stats$populations, collapse = ", ")),
      sprintf("  Performance Metrics:"),
      sprintf("    - Mean Final Objective: %.4f (±%.4f)", stats$mean_final_objective, stats$std_final_objective),
      sprintf("    - Mean Improvement Rate: %.4f", stats$mean_improvement),
      sprintf("    - Mean Runtime: %.2f minutes", stats$mean_runtime),
      sprintf("    - Mean Time per Generation: %.2f minutes (±%.2f)", 
              stats$mean_time_per_generation, stats$std_time_per_generation)
    )
    report_lines <- c(report_lines, llm_lines)
  }
  
  report_lines <- c(report_lines, "\nDetailed Population Analysis\n", 
                   "-------------------------\n")
  
  # Add detailed population summaries with comparative analysis
  for (pop_id in names(all_stats)) {
    stats <- all_stats[[pop_id]]
    relative_performance <- (aggregate_stats$mean_final_objective - stats$final_best_objective) / 
                          aggregate_stats$mean_final_objective * 100
    
    pop_lines <- c(
      sprintf("\n%s:", pop_id),
      sprintf("  Performance:"),
      sprintf("    - Final Objective: %.4f", stats$final_best_objective),
      sprintf("    - Relative to Mean: %s%.2f%%", 
              ifelse(relative_performance > 0, "+", ""), 
              relative_performance),
      sprintf("    - Mean Improvement Rate: %.4f", stats$mean_improvement_rate),
      sprintf("    - Generations to Best: %d", stats$generations_to_best),
      sprintf("    - Improvement Frequency: %.1f%%", stats$improvement_frequency),
      sprintf("  Execution:"),
      sprintf("    - Generations: %d", stats$total_generations),
      sprintf("    - Runtime: %.2f minutes", stats$runtime_minutes),
      sprintf("    - Time per Generation: %.2f minutes", stats$time_per_generation),
      sprintf("  Configuration:"),
      sprintf("    - LLM: %s", stats$llm_choice),
      sprintf("    - RAG: %s", stats$rag_choice),
      sprintf("    - Embedding: %s", stats$embed_choice)
    )
    report_lines <- c(report_lines, pop_lines)
  }
  
  writeLines(report_lines, report_file)
  
  cat(sprintf("\nAnalysis completed and saved to %s\n", analysis_file))
  cat(sprintf("Report saved to %s\n", report_file))
  
  # Return analysis results
  list(
    stats = aggregate_stats,
    timestamp = timestamp,
    analysis_file = analysis_file,
    report_file = report_file
  )
}

# Only run if script is run directly (not sourced)
if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 0) {
    stop("Please provide population numbers as arguments")
  }
  population_numbers <- as.numeric(args)
  analyze_populations(population_numbers)
}
