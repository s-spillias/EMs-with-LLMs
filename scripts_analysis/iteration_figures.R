# This script is sourced by analyze_populations.R
# Do not include library loads or source other scripts here

# Function to process a single model_report.json file
process_model_report <- function(file_path) {
  if (!file.exists(file_path)) return(NULL)
  
  report <- fromJSON(file_path)
  
  # Get all iterations
  iterations <- report$iterations
  if (is.null(iterations)) return(NULL)
  
  # Convert iteration numbers to numeric and sort
  iter_nums <- as.numeric(names(iterations))
  max_iter <- max(iter_nums)
  
  # Get the final objective value
  final_value <- NULL
  for (i in sort(iter_nums, decreasing = TRUE)) {
    if (!is.null(iterations[[as.character(i)]]$objective_value)) {
      final_value <- iterations[[as.character(i)]]$objective_value
      break
    }
  }
  
  if (is.null(final_value)) return(NULL)
  
  return(data.frame(
    total_iterations = max_iter,
    objective_value = final_value
  ))
}

# Function to process a population directory
process_population <- function(pop_dir) {
  # Extract population number from directory name
  pop_num <- as.numeric(gsub(".*POPULATION_([0-9]+).*", "\\1", pop_dir))
  if (is.na(pop_num)) {
    warning(sprintf("Could not extract population number from %s", pop_dir))
    return(data.frame())
  }
  
  # Read population metadata to get LLM choice
  metadata_path <- file.path(pop_dir, "population_metadata.json")
  if (!file.exists(metadata_path)) {
    warning(sprintf("No population_metadata.json found in %s", pop_dir))
    return(data.frame())
  }
  
  metadata <- fromJSON(metadata_path)
  llm_choice <- metadata$llm_choice
  if (is.null(llm_choice)) llm_choice <- "unknown"
  
  # Extract and sanitize topic
  topic <- metadata$project_topic
  if (!is.null(topic)) {
    # Simplify topic to key terms
    if (grepl("Crown of Thorns", topic, ignore.case=TRUE)) {
      topic <- "COTS"
    } else if (grepl("NPZ", topic, ignore.case=TRUE)) {
      topic <- "NPZ"
    } else {
      # Extract first few meaningful words
      topic <- gsub("\\s+", " ", topic)  # normalize whitespace
      topic_words <- strsplit(topic, " ")[[1]]
      topic <- paste(topic_words[1:min(3, length(topic_words))], collapse=" ")
    }
  } else {
    topic <- "unknown"
  }
  
  # Get all individual directories, excluding BROKEN and CULLED
  all_dirs <- list.dirs(pop_dir, full.names = TRUE, recursive = FALSE)
  indiv_dirs <- all_dirs[!grepl("/(BROKEN|CULLED)$", all_dirs)]
  
  results <- data.frame()
  
  for (indiv_dir in indiv_dirs) {
    report_path <- file.path(indiv_dir, "model_report.json")
    if (!file.exists(report_path)) next
    
    # Process model report
    report_data <- process_model_report(report_path)
    if (is.null(report_data)) next
    
    results <- rbind(results, data.frame(
      individual = basename(indiv_dir),
      population = sprintf("%04d", pop_num),
      total_iterations = report_data$total_iterations,
      objective_value = report_data$objective_value,
      llm_choice = llm_choice,
      topic = topic
    ))
  }
  
  return(results)
}

# Function to create iteration figures
create_iteration_figures <- function(population_dirs) {
  all_results <- data.frame()
  
  for (pop_dir in population_dirs) {
    results <- process_population(pop_dir)
    all_results <- rbind(all_results, results)
  }
  
  # Rename LLM choices for better display
  all_results <- all_results %>%
    mutate(llm_choice = case_when(
      llm_choice == "anthropic_sonnet" ~ "Claude 3.6 Sonnet",
      llm_choice == "claude_3_7_sonnet" ~ "Claude 3.7 Sonnet",
      llm_choice == 'openrouter:openai/gpt-5' ~ "GPT-5",
      llm_choice == 'openrouter:anthropic/claude-sonnet-4.5' ~ "Sonnet-4.5",
      llm_choice == 'openrouter:google/gemini-2.5-pro' ~ "Gemini-2.5",
      TRUE ~ llm_choice
    ))
  # Debug: show unique labels with lengths and raw bytes
  u <- sort(unique(all_results$llm_choice))
  cat("Unique llm_choice labels and lengths:\n")
  print(data.frame(label = u, nchar = nchar(u)))

  # Calculate summary statistics
  summary_stats <- all_results %>%
    group_by(population) %>%
    summarise(
      n_individuals = n(),
      mean_iterations = mean(total_iterations),
      median_iterations = median(total_iterations),
      sd_iterations = sd(total_iterations)
    )
  
  # Create boxplot by population
  p1 <- ggplot(all_results, aes(x = population, y = total_iterations)) +
    geom_boxplot(outlier.shape = NA) +  # Hide boxplot outliers since we'll show actual points
    geom_jitter(width = 0.2, alpha = 0.6, size = 2) +  # Add individual points
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(hjust = 0.5)
    ) +
    labs(
      x = "Population",
      y = "Number of Iterations"
    ) +
    scale_y_continuous(breaks = seq(0, max(all_results$total_iterations), by = 1))  # Integer y-axis breaks
  
  # Save population plot
  ggsave("Figures/iterations_by_population.png", p1, width = 10, height = 6)
  
  # Create faceted boxplot by LLM choice and topic
  p2 <- ggplot(all_results, aes(x = llm_choice, y = total_iterations)) +
    geom_boxplot(outlier.shape = NA) +
    geom_jitter(width = 0.2, alpha = 0.6, size = 2) +
    facet_wrap(~topic) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(hjust = 0.5),
      strip.text = element_text(size = 10)
    ) +
    labs(
      x = "LLM Choice",
      y = "Number of Iterations"
    ) +
    scale_y_continuous(breaks = seq(0, max(all_results$total_iterations), by = 1))
  
  # Save LLM choice plot
  ggsave("Figures/iterations_by_llm.png", p2, width = 10, height = 6)
  
  # Calculate and print summary statistics by LLM and topic
  llm_stats <- all_results %>%
    group_by(llm_choice, topic) %>%
    summarise(
      n_individuals = n(),
      mean_iterations = mean(total_iterations),
      median_iterations = median(total_iterations),
      sd_iterations = sd(total_iterations)
    )
  
  cat("\nIteration Statistics by Population:\n")
  print(summary_stats)
  
  cat("\nIteration Statistics by LLM Choice:\n")
  print(llm_stats)
  
  return(invisible(all_results))
}
