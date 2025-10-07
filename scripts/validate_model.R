library(TMB)
library(jsonlite)
library(ggplot2)

validate_model <- function(model, fit, response_file, forcing_file, individual_dir, time_col, train_test_split = 1.0) {
  # Check if we have any test data
  if (train_test_split >= 1.0) {
    cat("No test data available (train_test_split >= 1.0)\n")
    return()
  }
  # Read data
  response_data <- read.csv(response_file)
  forcing_data <- read.csv(forcing_file)
  print("look here")
  print(response_data)
  print('time col')
  print(time_col)
  # Get time column (first column) and sanitize it
  # time_col <- strsplit(colnames(response_data)[1], " ")[[1]][1]
  # Update column name in data

  colnames(response_data)[1] <- time_col
  
  # Calculate training end point based on train_test_split
  if (train_test_split < 1.0) {
    n_points <- nrow(response_data)
    n_train_points <- floor(n_points * train_test_split)
    training_end <- response_data[[time_col]][n_train_points]
  } else {
    training_end <- max(response_data[[time_col]])
  }
  
  # Modify column names by splitting at spaces and '..' and keeping the first part
  colnames(response_data) <- sapply(colnames(response_data), function(x) {
    # First split by '..' if it exists
    parts <- strsplit(x, "..", fixed = TRUE)[[1]][1]
    # Then split by space and take first part
    strsplit(parts, " ")[[1]][1]
  })
  colnames(forcing_data) <- sapply(colnames(forcing_data), function(x) {
    # First split by '..' if it exists
    parts <- strsplit(x, "..", fixed = TRUE)[[1]][1]
    # Then split by space and take first part
    strsplit(parts, " ")[[1]][1]
  })
  
  # Merge data
  data <- merge(response_data, forcing_data, by = time_col, all = TRUE)
  print("look here")
  print(data)
  # Get optimized parameters
  parameters <- model$env$parList(fit$par)
  
  # Prepare data for TMB using full time series
  data_in <- as.list(data)
  
  # Add time variable if needed (similar to ControlFile.R)
  if (!"time" %in% names(data_in)) {
    data_in$time <- data[[time_col]]
  }
  
  # Create model with full data and optimized parameters
  full_model <- MakeADFun(data_in, parameters, DLL = 'model', silent = TRUE, map = model$env$map)
  report <- full_model$report()
  
  # Extract observed variables from response data (excluding Year column)
  observed_vars <- colnames(response_data)[grep("_dat$", colnames(response_data))]
  
  # Create prediction variables by replacing _dat with _pred
  prediction_vars <- gsub("_dat$", "_pred", observed_vars)
  
  # Create variable labels by removing _dat suffix
  var_labels <- gsub("_dat$", "", observed_vars)
  # Update your var_labels vector
  label_mapping <- c('fast' = 'Fast-growing Coral', 
                  'slow' = 'Slow-growing Coral', 
                  'cots' = 'Crown of Thorns')
  # Initialize validation results
  validation_results <- list(
    status = "SUCCESS",
    metrics = list(),
    plot_data = list()
  )
  
  # Create plots and calculate metrics
  for(i in 1:length(prediction_vars)) {
    pred_var <- prediction_vars[i]
    obs_var <- observed_vars[i]
    
    # Extract predictions for test period
    test_predictions <- report[[pred_var]][data[[time_col]] > training_end]
    test_observed <- data[[obs_var]][data[[time_col]] > training_end]
    
    # Calculate error metrics on test period
    rmse <- sqrt(mean((test_predictions - test_observed)^2, na.rm = TRUE))
    mae <- mean(abs(test_predictions - test_observed), na.rm = TRUE)
    r2 <- cor(test_predictions, test_observed, use = "complete.obs")^2
    
    # Store metrics
    validation_results$metrics[[var_labels[i]]] <- list(
      RMSE = rmse,
      MAE = mae,
      R2 = r2
    )
    
    # Create plot data using full time series
    plot_data <- data.frame(
      Time = data[[time_col]],
      Predicted = report[[pred_var]],  # Full predictions from the full model run
      Observed = data[[obs_var]]       # Full observations
    )
    
    validation_results$plot_data[[pred_var]] <- as.list(plot_data)
    
    # Create plot
    p <- ggplot(plot_data, aes(x = Time)) +
      # Add shaded test region
      annotate("rect", xmin = training_end, xmax = max(plot_data$Time),
               ymin = -Inf, ymax = Inf,
               fill = "mistyrose", alpha = 0.3) +
      # Add vertical line at end of training
      geom_vline(xintercept = training_end, 
                 linetype = "dashed", color = "grey50") +
      geom_line(aes(y = Predicted, color = "Predicted"), linewidth = 1) +
      geom_point(aes(y = Observed, color = "Observed"), size = 1) +
      labs(title = label_mapping[var_labels[i]],
           subtitle = paste("RMSE =", round(rmse, 3), 
                          "MAE =", round(mae, 3),
                          "R² =", round(r2, 3)),
           x = time_col,
           y = var_labels[i],
           color = "Type") +
      scale_color_manual(values = c("Predicted" = "#D55E00", "Observed" = "#0072B2")) +
      theme_classic() +
      theme(
        plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 10),
        legend.position = "bottom"
      )
    
    # Save plot
    plot_filename <- file.path(individual_dir, paste0(pred_var, "_validation"))
    ggsave(paste0(plot_filename, ".png"), plot = p, width = 10, height = 6, dpi = 300)
    ggsave(paste0(plot_filename, ".svg"), plot = p, width = 10, height = 6, dpi = 300)
    ggsave(paste0(plot_filename, ".pdf"), plot = p, width = 10, height = 6)
  }
  
  # Save validation results
  validation_file <- file.path(individual_dir, "validation_report.json")
  write(toJSON(validation_results, auto_unbox = TRUE, pretty = TRUE), validation_file)
  
  # Print summary to console
  cat("\nValidation Results Summary:\n")
  for(var in names(validation_results$metrics)) {
    cat("\n", var, ":\n")
    metrics <- validation_results$metrics[[var]]
    cat("  RMSE:", round(metrics$RMSE, 3), "\n")
    cat("  MAE:", round(metrics$MAE, 3), "\n")
    cat("  R²:", round(metrics$R2, 3), "\n")
  }
}
