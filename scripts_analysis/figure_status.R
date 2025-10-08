# Load required libraries
library(ggplot2)
library(tidyr)
library(dplyr)
library(purrr)
library(stringr)
source("scripts_analysis/population_utils.R")

create_population_status_plot <- function(all_population_data, plot_name = "individual_status_distribution") {
  # Combine data from all populations
  all_status_data <- map_dfr(names(all_population_data), function(pop_name) {
    pop_info <- all_population_data[[pop_name]]
    print(paste("Processing population:", pop_name))
    print(str(pop_info))  # Debug print
    generations <- pop_info$data$generations
    print(paste("Number of generations:", length(generations$best_individuals)))  # Debug print
    print(paste("LLM choice:", pop_info$data$llm_choice))  # Debug print
    print(paste("Converged:", pop_info$data$converged))  # Debug print
    num_generations <- length(generations$best_individuals)
    
    # Get convergence status
    converged <- if (!is.null(pop_info$data$converged)) pop_info$data$converged else FALSE
    
    # Track all seen individual IDs
    seen_ids <- character(0)
    
    # Process each generation
    generation_data <- map_dfr(seq_len(num_generations), function(gen) {
      print(paste("Processing generation:", gen))  # Debug print
      
      # Get all individuals in current generation
      best_ids <- tryCatch({
        gen_data <- generations$best_individuals[[gen]]
        print(paste("Generation data class:", class(gen_data)))  # Debug print
        
        if (is.data.frame(gen_data) && nrow(gen_data) > 0) {
          ids <- gen_data$individual
          print(paste("Best IDs:", paste(ids, collapse=", ")))  # Debug print
          ids
        } else {
          print("No best individuals")  # Debug print
          character(0)
        }
      }, error = function(e) {
        print(paste("Error getting best IDs:", e$message))  # Debug print
        character(0)
      })
      
      culled_ids <- tryCatch({
        ids <- generations$culled_individuals[[gen]]
        if (!is.null(ids) && length(ids) > 0) {
          print(paste("Culled IDs:", paste(ids, collapse=", ")))  # Debug print
          ids
        } else {
          print("No culled individuals")  # Debug print
          character(0)
        }
      }, error = function(e) {
        print(paste("Error getting culled IDs:", e$message))  # Debug print
        character(0)
      })
      
      broken_ids <- tryCatch({
        ids <- generations$broken_individuals[[gen]]
        if (!is.null(ids) && length(ids) > 0) {
          print(paste("Broken IDs:", paste(ids, collapse=", ")))  # Debug print
          ids
        } else {
          print("No broken individuals")  # Debug print
          character(0)
        }
      }, error = function(e) {
        print(paste("Error getting broken IDs:", e$message))  # Debug print
        character(0)
      })
      
      # All IDs in current generation
      current_gen_ids <- c(best_ids, culled_ids, broken_ids)
      
      # Find new individuals in this generation
      new_ids <- setdiff(current_gen_ids, seen_ids)
      
      # Update seen IDs for next generation
      seen_ids <<- union(seen_ids, current_gen_ids)
      
      # Count where new individuals ended up
      new_kept <- sum(best_ids %in% new_ids)
      new_culled <- sum(culled_ids %in% new_ids)
      
      tibble(
        generation = gen,
        culled = new_culled,  # Only count new individuals that were culled
        broken = length(broken_ids),  # All broken are from current generation
        kept = new_kept,  # Only count new individuals that were kept
        llm_choice = pop_info$data$llm_choice,
        population_name = pop_name,
        converged = converged
      )
    })
    
    generation_data
  })
  
  # Extract population number and group by LLM choice and add replicate numbers
  all_status_data <- all_status_data %>%
    mutate(population = as.numeric(str_extract(population_name, "\\d+"))) %>%
    # Rename LLM choices for better display
    mutate(llm_choice = case_when(
      llm_choice == "anthropic_sonnet" ~ "Claude 3.6 Sonnet",
      llm_choice == "claude_3_7_sonnet" ~ "Claude 3.7 Sonnet",
      TRUE ~ llm_choice
    )) %>%
    group_by(llm_choice, population) %>%
  mutate(replicate = cur_group_id()) %>%
  ungroup() %>%
  group_by(llm_choice) %>%
  mutate(replicate = dense_rank(replicate)) %>%
  ungroup()
  
  # Convert to long format for stacked plot
  status_long <- all_status_data %>%
    pivot_longer(
      cols = c(kept, culled, broken),
      names_to = "status",
      values_to = "count"
    ) %>%
    filter(generation <= 10,
    replicate == 1) %>%
    mutate(
      # Capitalize status for better presentation
      status = factor(
        str_to_title(status),
        levels = c("Kept", "Culled", "Broken")
      )
    )
  
  # Create stacked bar plot
  # Calculate total height for each generation to place asterisk
  status_totals <- status_long %>%
    group_by(llm_choice, replicate, generation) %>%
    summarise(total = sum(count), converged = first(converged), .groups = 'drop')
  
  # Create base plot
  plot <- ggplot() +
    # Add stacked bars
    geom_bar(data = status_long,
             aes(x = generation, y = count, fill = status),
             stat = "identity", alpha = 0.7) +
    scale_fill_manual(
      values = c(
        "Kept" = "#2ca02c",   # Green from tab10
        "Culled" = "#1f77b4",    # Blue from tab10
        "Broken" = "#ff7f0e"  # Orange from tab10
      )
    ) +
    facet_grid(replicate ~ llm_choice) +
    labs(
      x = "Generation",
      y = "Number of Individuals",
      fill = "Status"
    ) +
    scale_x_continuous(breaks = function(x) seq(floor(min(x)), ceiling(max(x)), by = 1)) +
    theme_minimal() +
    # Add asterisk for converged populations
    # geom_text(data = status_long %>% group_by(llm_choice,population_name) %>%
    #           filter(generation == max(generation), converged == TRUE),
    #           aes(x = generation, y = 3.3),
    #           label = "*",
    #           vjust = 0.5, size = 10) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      panel.grid = element_blank(),  # Remove grid lines
      strip.text = element_text(size = 11),
      legend.position = "right",
      legend.text = element_text(size = 10),
      legend.title = element_text(size = 12)
    )
  
  # Save to manuscript figures directory
  figures_dir <- "Figures"
  dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Calculate dimensions based on number of facets
  n_replicates <- length(unique(all_status_data$replicate))
  n_llm_choices <- length(unique(all_status_data$llm_choice))
  
  width <- max(12, 6 + 2.5 * n_llm_choices)
  height <- max(8, 4 + 1.5 * n_replicates)
  
  ggsave(
    file.path(figures_dir, "success_frequency.png"),
    plot,
    width = width,
    height = height/5,
    dpi = 300
  )
  
  # Return the data for potential further analysis
  return(all_status_data)
}
