# Load required libraries
library(jsonlite)
library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)
library(purrr)
library(fs)

source("scripts_analysis/population_utils.R")

# Function to read population metadata
read_population_metadata <- function(population_number) {
  file_path <- sprintf("POPULATIONS/POPULATION_%04d/population_metadata.json", population_number)
  fromJSON(file_path)
}

# Function to read individual metadata
read_individual_metadata <- function(file_path) {
  tryCatch({
    data <- fromJSON(file_path)
    # Ensure all required fields are present
    required_fields <- c("objective_value", "lineage")
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

# Get population directories
pop_info <- get_population_dirs()
population_dirs <- pop_info$dirs
population_numbers <- pop_info$numbers

cat(sprintf("Found %d populations: %s\n", 
           length(population_numbers), 
           paste(population_numbers, collapse = ", ")))

# Process each population and combine data
all_populations_data <- map_dfr(seq_along(population_numbers), function(i) {
  pop_num <- population_numbers[i]
  cat(sprintf("\nProcessing population %04d (%d/%d)...\n", 
              pop_num, i, length(population_numbers)))
  
  tryCatch({
    # Get model report files for this population
    model_report_files <- dir_ls(population_dirs[i], regexp = "/model_report.json", recurse = TRUE)
    model_objectives <- map_dfr(model_report_files, function(path) {
    tryCatch({
      report <- fromJSON(path)
      iteration_numbers <- as.numeric(names(report$iterations))
      last_iteration <- max(iteration_numbers)
      individual <- gsub("INDIVIDUAL_", "", basename(dirname(path)))
      obj_value <- report$iterations[[as.character(last_iteration)]]$objective_value
      num_iterations <- length(report$iterations)
      # Ensure objective value is numeric
      obj_value <- tryCatch({
        as.numeric(obj_value)
      }, warning = function(w) {
        cat(sprintf("Warning converting objective value for %s: %s\n", 
                   individual, conditionMessage(w)))
        NA
      }, error = function(e) {
        cat(sprintf("Error converting objective value for %s: %s\n", 
                   individual, conditionMessage(e)))
        NA
      })
      tibble(
        individual = individual,
        objective_value = obj_value,
        num_iterations = num_iterations
      )
    }, error = function(e) {
      cat("Error reading", path, ":", e$message, "\n")
      return(NULL)
    })
  })
  
  # Get metadata files for this population
  metadata_files <- dir_ls(population_dirs[i], regexp = "/metadata.json", recurse = TRUE)
  
  # Process metadata and join with objectives
  population_metadata <- sapply(metadata_files, FUN=read_individual_metadata) %>%
    imap_dfr(~ tibble(
      file_path = .x$file_path,
      lineage = list(.x$lineage) %>% lapply(function(x) gsub("INDIVIDUAL_","",x))
    )) %>% 
    rowwise() %>% 
    mutate(
      generation = length(lineage),
      individual = gsub("INDIVIDUAL_", "", tail(strsplit(file_path, "/")[[1]], 2)[1]),
      founder = ifelse(length(lineage) == 0, individual, lineage[[1]]),
      parent = ifelse(length(lineage) > 0, tail(lineage, 1), NA),
      population = pop_num
    ) %>%
    ungroup() %>%
    left_join(model_objectives, by = "individual") %>%
    ungroup()
  
  # Read population metadata to get llm_choice
  pop_metadata <- read_population_metadata(pop_num)
  # Add debug print to check structure
  # print("Population metadata structure:")
  # print(str(pop_metadata))
  
  # Extract llm_choice, defaulting to "unknown" if not found
  llm_choice <- tryCatch({
    if ("data" %in% names(pop_metadata) && "llm_choice" %in% names(pop_metadata$data)) {
      pop_metadata$data$llm_choice
    } else if ("llm_choice" %in% names(pop_metadata)) {
      pop_metadata$llm_choice
    } else {
      "unknown"
    }
  }, error = function(e) {
    print(paste("Error extracting llm_choice:", e$message))
    "unknown"
  })
  
    population_metadata$llm_choice <- llm_choice
    
    return(population_metadata)
  }, error = function(e) {
    cat(sprintf("Skipping population %04d due to error: %s\n", 
                pop_num, conditionMessage(e)))
    return(NULL)
  })
})

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

# Process terminal children for all populations
all_populations_plot_data <- map_dfr(unique(all_populations_data$population), function(pop_num) {
  pop_data <- all_populations_data %>% filter(population == pop_num)
  
  # Identify terminal children for this population
  all_individuals <- unique(pop_data$individual)
  all_parents <- unique(unlist(pop_data$lineage))
  terminal_children <- setdiff(all_individuals, all_parents)
  
  # Create lineage paths for terminal children
  map_dfr(terminal_children, function(child) {
    lineage_path <- get_lineage_path(child, pop_data)
    tibble(
      terminal_child = child,
      generation = seq_along(lineage_path),
      individual = lineage_path,
      objective_value = pop_data$objective_value[match(lineage_path, pop_data$individual)],
      founder = pop_data$founder[pop_data$individual == child],
      population = pop_num,
      llm_choice = pop_data$llm_choice[1]
    )
  })
})
all_populations_plot_data_edit <- all_populations_plot_data %>%  group_by(population) %>%
filter(!is.na(objective_value)) %>%
  mutate(founder_id = as.factor(as.integer(factor(founder)))) %>%
   group_by(llm_choice, population) %>%
  mutate(replicate = cur_group_id()) %>%
  ungroup() %>%
  group_by(llm_choice) %>%
  mutate(replicate = dense_rank(replicate)) %>%
  ungroup()

# Create faceted plots
# 1. Founder to terminal evolution
ggplot(all_populations_plot_data_edit, 
       aes(x = generation, y = objective_value, group = terminal_child, color = founder_id)) +
  geom_line() +
  # geom_point(data = all_populations_plot_data_edit %>% filter(generation == max(generation)), 
  #            aes(shape = "Terminal"), size = 3) +
  geom_point(data = all_populations_plot_data_edit %>% filter(generation == 1), 
             aes(shape = "Founder"), size = 3) +
  facet_grid(replicate~llm_choice, scales = "free_y") +
  labs(title = "Objective Values from Founders to Terminal Children",
       x = "Generation",
       y = "Objective Value",
       color = "Founder",
       shape = "Type") +
  theme_classic() +
  theme(legend.position = "right") +
  scale_y_log10()

ggsave("Figures/founder_to_terminal_evolution.png", width = 15, height = 10)

# # 2. Evolution of objective values
# ggplot(all_populations_data, 
#        aes(x = generation, y = objective_value, color = founder, group = founder)) +
#   geom_line() +
#   geom_point() +
#   facet_wrap(~population, scales = "free_y") +
#   labs(title = "Evolution of Objective Values Across Generations",
#        x = "Generation",
#        y = "Objective Value") +
#   theme_minimal() +
#   scale_y_log10()

# ggsave("Figures/objective_values_evolution.png", width = 15, height = 10)

# # 3. Best performers for each population
# map(population_numbers, function(pop_num) {
#   population_data <- read_population_metadata(pop_num)
#   best_performers <- population_data$current_best_performers %>%
#     as_tibble()
  
#   ggplot(best_performers, aes(x = reorder(individual, -objective_value), y = objective_value)) +
#     geom_bar(stat = "identity") +
#     labs(title = sprintf("Current Best Performers - Population %04d", pop_num),
#          x = "Individual",
#          y = "Objective Value") +
#     theme_minimal() +
#     theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
#   ggsave(sprintf("Figures/best_performers_pop%04d.png", pop_num), 
#          width = 10, height = 6)
# })

# Print summary statistics for each population
all_populations_data %>%
  group_by(population, generation) %>%
  summarize(
    mean_objective_value = mean(objective_value, na.rm = TRUE),
    min_objective_value = min(objective_value, na.rm = TRUE),
    max_objective_value = max(objective_value, na.rm = TRUE)
  ) %>%
  print()

cat("Analysis script executed. Please check the Figures directory for the generated files.\n")
