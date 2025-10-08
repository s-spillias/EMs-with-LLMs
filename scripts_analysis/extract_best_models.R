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
results <- fromJSON("Manuscript/Results/populations_analysis.json", simplifyVector = TRUE)
llm_stats <- results$stats$llm_statistics

# ---------- Build LaTeX content ----------
supplement_content <- c(
  "\\section{Best Performing Models for CoTS Case Study}",
  "\\label{sec:best_models_cots}",
  "This section presents the best performing models from different LLM configurations for the Crown of Thorns Starfish (CoTS) case study.",
  ""
)

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

      supplement_content <- c(
        supplement_content,
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
        "\\end{lstlisting}",
        "\\clearpage"
      )
      cots_llms_with_results <- cots_llms_with_results + 1
    }
  }
}

# ---------- NPZ Section ----------
supplement_content <- c(
  supplement_content,
  "\\section{Best Performing Models for NPZ Case Study}",
  "\\label{sec:best_models_npz}",
  "This section presents the best performing models from different LLM configurations for the Nutrient-Phytoplankton-Zooplankton (NPZ) case study.",
  ""
)

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

      supplement_content <- c(
        supplement_content,
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
        "\\end{lstlisting}",
        "\\clearpage"
      )
      npz_llms_with_results <- npz_llms_with_results + 1
    }
  }
}

# ---------- Best Out-of-Sample Test Model ----------
supplement_content <- c(
  supplement_content,
  "\\section{Best Out-of-Sample Test Model}",
  "\\label{sec:best_out_of_sample}",
  "This section presents the best performing out-of-sample test model (populations with \\texttt{train\\_test\\_split} < 1.0).",
  ""
)

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

  supplement_content <- c(
    supplement_content,
    sprintf(
      "The best out-of-sample model is from Population %d (%s), with objective value %.4f and \\texttt{train\\_test\\_split} = %.3f.",
      oos_best$pop_num, oos_best$category, oos_best$objective, oos_best$tts
    ),
    "",
    "\\subsection{Model Intention}",
    "\\begin{lstlisting}",
    oos_files$intention_txt,
    "\\end{lstlisting}",
    "",
    "\\subsection{Model Implementation}",
    "\\begin{lstlisting}",
    oos_files$model_cpp,
    "\\end{lstlisting}",
    "",
    "\\subsection{Model Parameters}",
    "\\begin{lstlisting}",
    oos_files$params_json,
    "\\end{lstlisting}",
    "\\clearpage"
  )
} else {
  supplement_content <- c(
    supplement_content,
    "No out-of-sample populations (with \\texttt{train\\_test\\_split} < 1.0) were found.",
    "\\clearpage"
  )
}

# ---------- Write and format ----------
writeLines(supplement_content, "Manuscript/best_models.tex")

# Run the LaTeX character formatting script
system2("python3", args = "scripts/format_latex_chars.py Manuscript/best_models.tex")

cat("Best models extracted and formatted, saved to Manuscript/best_models.tex\n")