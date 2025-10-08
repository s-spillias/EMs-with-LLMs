# Load required libraries
library(ggplot2)
library(tidyr)
library(purrr)
library(patchwork)
source("scripts_analysis/population_utils.R")

create_population_figures <- function(population_data, all_metadata) {
  # Identify terminal children
  all_individuals <- unique(all_metadata$individual)
  all_parents <- unique(unlist(all_metadata$lineage)) 
  terminal_children <- setdiff(all_individuals, all_parents)
  
  # Create a function to get the lineage path for a terminal child
  get_lineage_path <- function(terminal_child, metadata) {
    path <- terminal_child
    current <- terminal_child
    while(!is.na(metadata$parent[metadata$individual == current])) {
      current <- metadata$parent[metadata$individual == current]
      path <- c(current, path)
    }
    return(path)
  }
  
  # First get all objective values
  all_objectives <- tibble(
    individual = character(),
    objective_value = numeric()
  )
  
  # Add best performers
  all_objectives <- bind_rows(
    all_objectives,
    select(as_tibble(population_data$current_best_performers), individual, objective_value)
  )
  
  # Add from generations
  for (i in seq_along(population_data$generations$generation_number)) {
    all_objectives <- bind_rows(
      all_objectives,
      select(population_data$generations$best_individuals[[i]], individual, objective_value)
    )
  }
  
  # Remove duplicates (keep first occurrence)
  all_objectives <- distinct(all_objectives, individual, .keep_all = TRUE)
  
  # Create the new dataframe for plotting
  plot_data <- lapply(terminal_children, function(child) {
    lineage_path <- get_lineage_path(child, all_metadata)
    tibble(
      terminal_child = child,
      generation = seq_along(lineage_path),
      individual = lineage_path,
      objective_value = all_metadata$objective_value[match(lineage_path, all_metadata$individual)],
      founder = all_metadata$founder[all_metadata$individual == child]
    )
  }) %>% bind_rows()

  # Create figures directory
  figures_dir <- file.path(ensure_results_dir(), "Figures")
  dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
  # 1. Founder to Terminal Children Evolution
  ggplot(plot_data, aes(x = generation, y = objective_value, group = terminal_child, color = founder)) +
    geom_line() +
    geom_point(data = plot_data %>% filter(generation == max(generation)), aes(shape = "Terminal"), size = 3) +
    geom_point(data = plot_data %>% filter(generation == 1), aes(shape = "Founder"), size = 3) +
    labs(title = "Objective Values from Founders to Terminal Children",
         x = "Generation",
         y = "Objective Value",
         color = "Founder",
         shape = "Type") +
    theme_minimal() +
    theme(legend.position = "right")
  
  ggsave(file.path(figures_dir, "founder_to_terminal_evolution.png"), width = 12, height = 8)

  
  # 2. Evolution of Best Objective Values
  best_objectives <- lapply(seq_along(population_data$generations$generation_number), function(i) {
    gen <- population_data$generations$best_individuals[[i]]
    tibble(
      generation = i,
      individual = gen$individual,
      objective_value = gen$objective_value
    )
  }) %>% bind_rows()
  
  ggplot(best_objectives, aes(x = generation, y = objective_value)) +
    geom_line() +
    geom_point() +
    labs(title = "Evolution of Best Objective Values",
         x = "Generation",
         y = "Objective Value") +
    theme_minimal()
  
  # Create training plot
  training_plot <- ggplot(best_objectives, aes(x = generation, y = objective_value)) +
    geom_line() +
    geom_point() +
    labs(title = "Training Progress",
         x = "Generation",
         y = "Objective Value") +
    theme_minimal()
  
  # Save training plot
  ggsave(file.path(figures_dir, "objective_values_evolution.png"), width = 10, height = 6)
  
  # 3. Best Performers
  best_performers_plot <- 
  current_best <- population_data$current_best_performers %>%
    as_tibble()
  
  best_performers_plot <- ggplot(current_best, aes(x = reorder(individual, -objective_value), y = objective_value)) +
    geom_bar(stat = "identity") +
    labs(title = "Current Best Performers",
         x = "Individual",
         y = "Objective Value") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave(file.path(figures_dir, "best_performers.png"), width = 10, height = 6)
  
  # Create combined plot using patchwork
  combined_plot <- training_plot + best_performers_plot +
    plot_layout(widths = c(1, 1))
  
  # Save combined plot
  ggsave(file.path(figures_dir, "training_and_best_performers.png"), 
         combined_plot, 
         width = 15, 
         height = 6)
  
  # 4. Culled and Broken Individuals
  culled_broken_data <- tibble(
    generation = 1:length(population_data$generations$generation_number),
    culled = map_int(population_data$generations$culled_individuals, length),
    broken = map_int(population_data$generations$broken_individuals, length)
  )
  
  culled_broken_long <- culled_broken_data %>%
    pivot_longer(cols = c(culled, broken), names_to = "type", values_to = "count")
  
  ggplot(culled_broken_long, aes(x = generation, y = count, fill = type)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(title = "Culled and Broken Individuals per Generation",
         x = "Generation",
         y = "Count",
         fill = "Type") +
    theme_minimal()
  
  ggsave(file.path(figures_dir, "culled_broken_individuals.png"), width = 10, height = 6)
  
  # Return data that might be useful for statistics
  list(
    culled_broken_data = culled_broken_data,
    best_performers = current_best,
    plot_data = plot_data
  )
}
