library(ggplot2)
library(jsonlite)
library(patchwork)
library(png)
library(grid)
library(gridExtra)

create_combined_validation_plot <- function(validation_file, output_dir) {
  # Read the validation report
  validation_data <- fromJSON(validation_file)
  
  # Extract the individual directory from the validation_file path
  individual_dir <- dirname(validation_file)
  population_dir <- dirname(individual_dir)
  
  # Read the population metadata to get train_test_split
  metadata_file <- file.path(population_dir, "population_metadata.json")
  metadata <- fromJSON(metadata_file)
  train_test_split <- metadata$train_test_split
  
  # Get time data from the first variable in the validation report
  time_data <- validation_data$plot_data[[1]]$Time
  
  # Calculate training_end based on train_test_split
  n_points <- length(time_data)
  n_train_points <- floor(n_points * train_test_split)
  training_end <- time_data[n_train_points]
  
  # Define label mapping for titles
  title_mapping <- c(
    'cots' = 'Crown-of-Thorns Starfish Abundance',
    'fast' = 'Fast-Growing Coral Cover',
    'slow' = 'Slow-Growing Coral Cover'
  )
  
  # Define y-axis label mapping
  ylabel_mapping <- c(
    'cots' = 'Abundance (individuals/m2)',
    'fast' = 'Cover (%)',
    'slow' = 'Cover (%)'
  )
  
  # Create a list to store the plots
  plots <- list()
  
  # Variables to plot
  variables <- c("cots", "fast", "slow")
  
  # Load PNG icons
  cots_icon <- readPNG("Figures/cots.png")
  fast_coral_icon <- readPNG("Figures/fast_coral.drawio.png")
  slow_coral_icon <- readPNG("Figures/slow_coral.drawio.png")
  
  # Create raster objects from the icons (larger size)
  cots_raster <- rasterGrob(cots_icon, interpolate = TRUE, width = unit(1.5, "npc"))
  fast_coral_raster <- rasterGrob(fast_coral_icon, interpolate = TRUE, width = unit(1.5, "npc"))
  slow_coral_raster <- rasterGrob(slow_coral_icon, interpolate = TRUE, width = unit(1.5, "npc"))
  
  # Create a list of icons
  icons <- list(cots_raster, fast_coral_raster, slow_coral_raster)
  
  # Create individual plots
  for (i in seq_along(variables)) {
    var <- variables[i]
    
    # Extract plot data
    plot_data_key <- paste0(var, "_pred")
    plot_data <- as.data.frame(validation_data$plot_data[[plot_data_key]])
    
    # Extract metrics
    rmse <- validation_data$metrics[[var]]$RMSE
    mae <- validation_data$metrics[[var]]$MAE
    r2 <- validation_data$metrics[[var]]$R2
    
    # Create plot
    p <- ggplot(plot_data, aes(x = Time)) +
      # Add shaded test region
      annotate("rect", xmin =  min(plot_data$Time), xmax = training_end,
               ymin = -Inf, ymax = Inf,
               fill = "mistyrose", alpha = 0.3) +
      # Add vertical line at end of training
      geom_vline(xintercept = training_end,
                 linetype = "dashed", color = "grey50") +
      geom_line(aes(y = Predicted, color = "Predicted"), linewidth = 2.5, alpha = 0.8) +
      geom_point(aes(y = Observed, color = "Observed"), size = 3, alpha = 0.8) +
      labs(title = title_mapping[var],
           subtitle = paste("RMSE =", round(rmse, 3),
                          "MAE =", round(mae, 3),
                          "R² =", round(r2, 3)),
           x = if(i == 3) "Year" else "",  # Only show x-axis label on bottom plot
           y = ylabel_mapping[var],
           color = "Type") +
      scale_color_manual(values = c("Predicted" = "#D55E00", "Observed" = "#333333")) +
      theme_classic() +
      theme(
        plot.title = element_text(size = 16, margin = margin(b = 20)),
        plot.subtitle = element_text(size = 12),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 14),
        legend.position = if(i == 3) "bottom" else "none",  # Only show legend on bottom plot
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 12),
        # Set spine colors and width
        axis.line = element_line(linewidth = 1.2, color = "#333333")
      )
    
    # Add icon to the plot (positioned in top right corner)
    p <- p + annotation_custom(
      icons[[i]],
      xmin = max(plot_data$Time) - 3,  # Position near the right edge
      xmax = max(plot_data$Time),
      ymin = max(plot_data$Observed, na.rm = TRUE) * 0.6,  # Position near the top
      ymax = max(plot_data$Observed, na.rm = TRUE) * 1.0
    )
    
    plots[[i]] <- p
  }
  
  # Combine plots using patchwork
  combined_plot <- plots[[1]] / plots[[2]] / plots[[3]] +
    plot_layout(heights = c(1, 1, 1.2))  # Make bottom plot slightly taller to accommodate legend
  
  # Add a title to the combined plot
  combined_plot <- combined_plot +
    plot_annotation(
      # title = "Validation Results for Marine Ecosystem Model",
      theme = theme(
        plot.title = element_text(size = 18, face = "bold", hjust = 0.5)
      )
    ) &
    # Apply common theme elements to all plots
    theme(
      plot.margin = margin(t = 20, r = 20, b = 20, l = 20)
    )
  
  # Save the combined plot
  plot_filename <- file.path(output_dir, "combined_validation")
  ggsave(paste0(plot_filename, ".png"), plot = combined_plot, width = 12, height = 16, dpi = 300)
  ggsave(paste0(plot_filename, ".svg"), plot = combined_plot, width = 12, height = 16, dpi = 300)
  ggsave(paste0(plot_filename, ".pdf"), plot = combined_plot, width = 12, height = 16)
  
  return(combined_plot)
}




# cat("Combined validation plot created and saved to:", output_dir, "\n")
# cat("Files created: combined_validation.png, combined_validation.svg, combined_validation.pdf\n")