# Script to extract best models segregated by NPZ and CoTS, and best out-of-sample model
library(jsonlite)

# ---------- Helpers ----------

# Safely read lines from a file (without warnings)
safe_read_lines <- function(path) {
  if (!file.exists(path)) {
    return(character())
  }
  readLines(path, warn = FALSE)
}

# Function to extract model files for a given population and individual
extract_model_files <- function(pop_num, individual_id) {
  pop_dir <- sprintf("POPULATIONS/POPULATION_%04d", as.integer(pop_num))
  indiv_dir <- file.path(pop_dir, paste0("INDIVIDUAL_", individual_id))

  model_cpp <- safe_read_lines(file.path(indiv_dir, "model.cpp"))
  params_json <- safe_read_lines(file.path(indiv_dir, "parameters.json"))
  intention_fn <- if (file.exists(file.path(indiv_dir, "intention.tex"))) {
    file.path(indiv_dir, "intention.tex")
  } else {
    file.path(indiv_dir, "intention.txt")
  }
  intention_txt <- safe_read_lines(intention_fn)

  list(
    model_cpp = model_cpp,
    params_json = params_json,
    intention_txt = intention_txt
  )
}

# Read population metadata for a given population number
get_population_metadata <- function(pop_num) {
  meta_path <- sprintf("POPULATIONS/POPULATION_%04d/population_metadata.json", as.integer(pop_num))
  if (!file.exists(meta_path)) {
    return(NULL)
  }
  # Use simplifyVector=FALSE to keep lists as lists (more robust for nested content)
  fromJSON(meta_path, simplifyVector = FALSE)
}

# Determine NPZ vs COTS from response_file.
# Per your logic: if 'NPZ' is in response_file -> NPZ, else -> COTS
get_population_category <- function(metadata) {
  resp <- metadata$response_file
  if (is.null(resp)) {
    return("Unknown")
  }
  if (grepl("NPZ", resp, ignore.case = TRUE)) "NPZ" else "COTS"
}

# Extract the best individual from metadata$current_best_performers by minimum objective_value
get_best_individual_from_metadata <- function(metadata) {
  performers <- metadata$current_best_performers
  if (is.null(performers)) {
    return(NULL)
  }

  # current_best_performers can be either a list of lists or a data.frame depending on JSON structure
  if (is.data.frame(performers)) {
    if (!("objective_value" %in% names(performers)) || !("individual" %in% names(performers))) {
      return(NULL)
    }
    idx <- which.min(as.numeric(performers$objective_value))
    indiv <- performers$individual[idx]
  } else if (is.list(performers)) {
    obj_vals <- sapply(performers, function(p) as.numeric(p$objective_value))
    idx <- which.min(obj_vals)
    indiv <- performers[[idx]]$individual
  } else {
    return(NULL)
  }
  gsub("^INDIVIDUAL_", "", indiv)
}

# Map internal LLM key to display name
format_llm_name <- function(llm) {
  if (llm == "anthropic_sonnet") {
    "Claude 3.6 Sonnet"
  } else if (llm == "claude_3_7_sonnet") {
    "Claude 3.7 Sonnet"
  } else {
    gsub("_", " ", llm)
  }
}

# ---------- Load analysis results ----------
results <- fromJSON("Results/populations_analysis.json", simplifyVector = TRUE)
llm_stats <- results$stats$llm_statistics

# ---------- Create output directory ----------
output_dir <- "Manuscript/best_models"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# We will count how many LLMs have valid COTS and NPZ picks
cots_llms_with_results <- 0
npz_llms_with_results <- 0

# ---------- Process each LLM configuration and segregate by category ----------
for (llm in names(llm_stats)) {
  stats <- llm_stats[[llm]]
  pops_vec <- stats$populations
  final_objs <- stats$final_objectives

  # Extract numeric pop numbers (assuming like "POPULATION_0016")
  pop_nums <- as.integer(gsub("POPULATION_", "", pops_vec))

  # Classify each population as NPZ or COTS
  categories <- vapply(pop_nums, function(p) {
    meta <- get_population_metadata(p)
    if (is.null(meta)) "Unknown" else get_population_category(meta)
  }, FUN.VALUE = character(1))

  # Best COTS model for this LLM
  cots_mask <- categories == "COTS"
  if (any(cots_mask, na.rm = TRUE)) {
    idx <- which(cots_mask)[which.min(final_objs[cots_mask])]
    best_obj_cots <- final_objs[idx]
    best_pop_cots <- pop_nums[idx]
    pop_meta_cots <- get_population_metadata(best_pop_cots)
    best_indiv_cots <- get_best_individual_from_metadata(pop_meta_cots)

    if (!is.null(best_indiv_cots)) {
      files <- extract_model_files(best_pop_cots, best_indiv_cots)
      llm_name <- format_llm_name(llm)
      
      # Create filename from LLM name (sanitize for filesystem)
      file_safe_name <- tolower(gsub("[^a-z0-9_]", "_", llm))
      output_file <- file.path(output_dir, sprintf("cots_%s.tex", file_safe_name))
      output_text_file <- file.path(output_dir, sprintf("cots_%s.txt", file_safe_name))
      model_content <- c(
        sprintf("\\subsection{%s Model (CoTS)}", llm_name),
        sprintf("This model achieved an objective value of %.4f (Population %d).", best_obj_cots, best_pop_cots),
        "",
        "\\subsubsection{Model Intention}",
        "\\begin{lstlisting}",
        files$intention_txt,
        "\\end{lstlisting}",
        "",
        "\\subsubsection{Model Implementation}",
        "\\begin{lstlisting}",
        files$model_cpp,
        "\\end{lstlisting}",
        "",
        "\\subsubsection{Model Parameters}",
        "\\begin{lstlisting}",
        files$params_json,
        "\\end{lstlisting}"
      )
      
      writeLines(model_content, output_file)
      system2("python3", args = c("scripts/format_latex_chars.py", output_file))
      cat(sprintf("Written CoTS model for %s to %s\n", llm_name, output_file))
      writeLines(model_content, output_text_file)
      system2("python3", args = c("scripts/format_latex_chars.py", output_text_file))
      cat(sprintf("Written CoTS model for %s to %s\n", llm_name, output_text_file))
      cots_llms_with_results <- cots_llms_with_results + 1
    }
  }
}

# ---------- NPZ Section ----------
for (llm in names(llm_stats)) {
  stats <- llm_stats[[llm]]
  pops_vec <- stats$populations
  final_objs <- stats$final_objectives
  pop_nums <- as.integer(gsub("POPULATION_", "", pops_vec))

  categories <- vapply(pop_nums, function(p) {
    meta <- get_population_metadata(p)
    if (is.null(meta)) "Unknown" else get_population_category(meta)
  }, FUN.VALUE = character(1))

  # Best NPZ model for this LLM
  npz_mask <- categories == "NPZ"
  if (any(npz_mask, na.rm = TRUE)) {
    idx <- which(npz_mask)[which.min(final_objs[npz_mask])]
    best_obj_npz <- final_objs[idx]
    best_pop_npz <- pop_nums[idx]
    pop_meta_npz <- get_population_metadata(best_pop_npz)
    best_indiv_npz <- get_best_individual_from_metadata(pop_meta_npz)

    if (!is.null(best_indiv_npz)) {
      files <- extract_model_files(best_pop_npz, best_indiv_npz)
      llm_name <- format_llm_name(llm)
      
      # Create filename from LLM name (sanitize for filesystem)
      file_safe_name <- tolower(gsub("[^a-z0-9_]", "_", llm))
      output_file <- file.path(output_dir, sprintf("npz_%s.tex", file_safe_name))
      output_text_file <- file.path(output_dir, sprintf("npz_%s.txt", file_safe_name))
      model_content <- c(
        sprintf("\\subsection{%s Model (NPZ)}", llm_name),
        sprintf("This model achieved an objective value of %.4f (Population %d).", best_obj_npz, best_pop_npz),
        "",
        "\\subsubsection{Model Intention}",
        "\\begin{lstlisting}",
        files$intention_txt,
        "\\end{lstlisting}",
        "",
        "\\subsubsection{Model Implementation}",
        "\\begin{lstlisting}",
        files$model_cpp,
        "\\end{lstlisting}",
        "",
        "\\subsubsection{Model Parameters}",
        "\\begin{lstlisting}",
        files$params_json,
        "\\end{lstlisting}"
      )
      
      writeLines(model_content, output_file)
      system2("python3", args = c("scripts/format_latex_chars.py", output_file))
      cat(sprintf("Written NPZ model for %s to %s\n", llm_name, output_file))
      writeLines(model_content, output_text_file)
      system2("python3", args = c("scripts/format_latex_chars.py", output_text_file))
      cat(sprintf("Written NPZ model for %s to %s\n", llm_name, output_text_file))
      npz_llms_with_results <- npz_llms_with_results + 1
    }
  }
}

# ---------- Best Out-of-Sample Test Model ----------
# Scan all populations and find best performer where train_test_split < 1.0
pop_dirs <- list.files("POPULATIONS", pattern = "^POPULATION_\\d+$", full.names = TRUE)
oos_candidates <- list()

for (pop_dir in pop_dirs) {
  meta_path <- file.path(pop_dir, "population_metadata.json")
  if (!file.exists(meta_path)) next
  meta <- fromJSON(meta_path, simplifyVector = FALSE)

  tts <- meta$train_test_split
  if (is.null(tts)) next
  # Out-of-sample condition
  if (is.numeric(tts) && tts < 1.0) {
    performers <- meta$current_best_performers
    if (is.null(performers)) next

    if (is.data.frame(performers)) {
      obj_vals <- as.numeric(performers$objective_value)
      idx <- which.min(obj_vals)
      indiv <- performers$individual[idx]
      best_obj <- obj_vals[idx]
    } else if (is.list(performers)) {
      obj_vals <- sapply(performers, function(p) as.numeric(p$objective_value))
      idx <- which.min(obj_vals)
      indiv <- performers[[idx]]$individual
      best_obj <- obj_vals[idx]
    } else {
      next
    }

    pop_num <- as.integer(gsub("POPULATION_", "", basename(pop_dir)))
    indiv_id <- gsub("^INDIVIDUAL_", "", indiv)
    category <- get_population_category(meta)

    oos_candidates[[length(oos_candidates) + 1]] <- list(
      pop_num = pop_num,
      individual = indiv_id,
      objective = best_obj,
      category = category,
      tts = tts
    )
  }
}

if (length(oos_candidates) > 0) {
  idx_global <- which.min(vapply(oos_candidates, function(x) x$objective, numeric(1)))
  oos_best <- oos_candidates[[idx_global]]
  oos_files <- extract_model_files(oos_best$pop_num, oos_best$individual)
  
  output_file <- file.path(output_dir, "oos_best.tex")
  output_text_file <- file.path(output_dir, "oos_best.txt")
  model_content <- c(
    "\\subsection{Best Out-of-Sample Test Model}",
    sprintf(
      "The best out-of-sample model is from Population %d (%s), with objective value %.4f and \\texttt{train\\_test\\_split} = %.3f.",
      oos_best$pop_num, oos_best$category, oos_best$objective, oos_best$tts
    ),
    "",
    "\\subsubsection{Model Intention}",
    "\\begin{lstlisting}",
    oos_files$intention_txt,
    "\\end{lstlisting}",
    "",
    "\\subsubsection{Model Implementation}",
    "\\begin{lstlisting}",
    oos_files$model_cpp,
    "\\end{lstlisting}",
    "",
    "\\subsubsection{Model Parameters}",
    "\\begin{lstlisting}",
    oos_files$params_json,
    "\\end{lstlisting}"
  )
  
  writeLines(model_content, output_file)
  system2("python3", args = c("scripts/format_latex_chars.py", output_file))
  cat(sprintf("Written out-of-sample model to %s\n", output_file))
  writeLines(model_content, output_text_file)
  system2("python3", args = c("scripts/format_latex_chars.py", output_text_file))
  cat(sprintf("Written out-of-sample model to %s\n", output_text_file))
} else {
  cat("No out-of-sample populations (with train_test_split < 1.0) were found.\n")
}

# ---------- Summary ----------
cat(sprintf("\nBest models extracted to %s/\n", output_dir))
cat(sprintf("  - %d CoTS models\n", cots_llms_with_results))
cat(sprintf("  - %d NPZ models\n", npz_llms_with_results))
if (length(oos_candidates) > 0) {
  cat("  - 1 out-of-sample model\n")
}
