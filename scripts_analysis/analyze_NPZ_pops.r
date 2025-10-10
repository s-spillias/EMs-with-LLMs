# --- Libraries ---
library(jsonlite)
library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(grid)
library(gtable)
library(png)
library(cowplot) # replaces patchwork for mixed objects

library(ggbreak)
# --- Configuration ---
population_roots <- c("POPULATIONS", "Manuscript")

# --- Utilities (same as before) ---
safe_fromJSON <- function(path) tryCatch(fromJSON(path), error = function(e) NULL)
read_model_report <- function(individual_dir) {
  rp <- file.path(individual_dir, "model_report.json")
  if (!file.exists(rp)) {
    return(NULL)
  }
  safe_fromJSON(rp)
}
get_objective_value <- function(individual_dir) {
  rpt <- read_model_report(individual_dir)
  if (is.null(rpt) || is.null(rpt$iterations)) {
    return(NA_real_)
  }
  its <- suppressWarnings(as.numeric(names(rpt$iterations)))
  if (length(its) == 0 || all(is.na(its))) {
    return(NA_real_)
  }
  max_iter <- max(its, na.rm = TRUE)
  obj <- rpt$iterations[[as.character(max_iter)]]$objective_value
  if (is.null(obj)) NA_real_ else obj
}
get_best_individual_dir <- function(pop_dir, pop_metadata = NULL) {
  if (is.null(pop_metadata)) {
    meta_path <- file.path(pop_dir, "population_metadata.json")
    pop_metadata <- safe_fromJSON(meta_path)
  }
  if (!is.null(pop_metadata$current_best_performers$individual)) {
    best_id <- pop_metadata$current_best_performers$individual[1]
    best_dir <- file.path(pop_dir, best_id)
    if (dir.exists(best_dir)) {
      return(best_dir)
    }
  }
  indiv_dirs <- list.dirs(pop_dir, full.names = TRUE, recursive = FALSE)
  indiv_dirs <- indiv_dirs[grepl("INDIVIDUAL_", basename(indiv_dirs))]
  if (length(indiv_dirs) == 0) {
    return(NULL)
  }
  objs <- sapply(indiv_dirs, get_objective_value)
  if (all(is.na(objs))) {
    return(NULL)
  }
  indiv_dirs[which.min(objs)]
}
extract_timeseries_second_last <- function(individual_dir) {
  rpt <- read_model_report(individual_dir)
  if (is.null(rpt) || is.null(rpt$iterations)) {
    return(NULL)
  }
  its <- sort(suppressWarnings(as.numeric(names(rpt$iterations))))
  if (length(its) < 2) {
    return(NULL)
  }
  second_last <- its[length(its) - 1]
  iter_data <- rpt$iterations[[as.character(second_last)]]$plot_data
  if (is.null(iter_data)) {
    return(NULL)
  }
  time <- iter_data$Z_pred$Time
  df_list <- list(
    Nutrients     = list(obs = iter_data$N_pred$Observed, mod = iter_data$N_pred$Modeled),
    Phytoplankton = list(obs = iter_data$P_pred$Observed, mod = iter_data$P_pred$Modeled),
    Zooplankton   = list(obs = iter_data$Z_pred$Observed, mod = iter_data$Z_pred$Modeled)
  )
  out <- lapply(names(df_list), function(v) {
    data.frame(
      Time = time, Variable = v,
      Observed = unlist(df_list[[v]]$obs),
      Modeled = unlist(df_list[[v]]$mod)
    )
  })
  do.call(rbind, out)
}
get_training_series <- function(meta) {
  if (is.null(meta$generations$best_individual)) {
    return(NULL)
  }
  gens <- meta$generations$best_individual
  obj <- unlist(lapply(gens, function(x) tryCatch(x[1, 2], error = function(e) NA_real_)))
  data.frame(generation = seq_along(obj), objective_value = obj)
}
find_npz_populations <- function(roots = population_roots) {
  meta_paths <- unlist(lapply(roots, function(root) {
    if (!dir.exists(root)) {
      return(character(0))
    }
    list.files(root, pattern = "population_metadata\\.json$", recursive = TRUE, full.names = TRUE)
  }))
  meta_paths <- meta_paths[grepl("POPULATION_", meta_paths)]
  npz <- list()
  for (mp in meta_paths) {
    meta <- safe_fromJSON(mp)
    if (is.null(meta)) next
    resp <- meta$response_file
    llm <- meta$llm_choice
    if ((!is.null(resp) && grepl("NPZ", resp, ignore.case = TRUE)) ||
      (!is.null(llm) && grepl("NPZ", llm, ignore.case = TRUE))) {
      npz[[length(npz) + 1]] <- list(pop_dir = dirname(mp), pop_id = basename(dirname(mp)), meta = meta)
    }
  }
  npz
}

# --- Icon helpers ---
load_icons <- function() {
  nutrient_icon <- tryCatch(readPNG("Figures/nutrient_trans.png"), error = function(e) NULL)
  phyto_icon <- tryCatch(readPNG("Figures/phyto_trans.png"), error = function(e) NULL)
  copepod_icon <- tryCatch(readPNG("Figures/copepod_trans.png"), error = function(e) NULL)
  list(
    Nutrients     = if (!is.null(nutrient_icon)) rasterGrob(nutrient_icon, interpolate = TRUE, width = unit(0.15, "npc"), height = unit(0.35, "npc"), x = unit(0.9, "npc"), y = unit(0.85, "npc")) else NULL,
    Phytoplankton = if (!is.null(phyto_icon)) rasterGrob(phyto_icon, interpolate = TRUE, width = unit(0.15, "npc"), height = unit(0.40, "npc"), x = unit(0.9, "npc"), y = unit(0.80, "npc")) else NULL,
    Zooplankton   = if (!is.null(copepod_icon)) rasterGrob(copepod_icon, interpolate = TRUE, width = unit(0.15, "npc"), height = unit(0.40, "npc"), x = unit(0.9, "npc"), y = unit(0.80, "npc")) else NULL
  )
}
add_facet_icons <- function(p, plot_df, icons) {
  gt <- ggplotGrob(p)
  panels <- grep("panel", gt$layout$name)
  vars <- unique(plot_df$Variable)
  for (i in seq_along(panels)) {
    panel_pos <- gt$layout[panels[i], c("t", "l", "b", "r")]
    var_name <- vars[i]
    icon <- icons[[var_name]]
    if (!is.null(icon)) {
      gt <- gtable_add_grob(gt, icon,
        t = panel_pos$t, l = panel_pos$l, b = panel_pos$b, r = panel_pos$r,
        name = paste0("icon-", var_name)
      )
    }
  }
  gt
}

# --- Main ---
npz_pops <- find_npz_populations()
if (length(npz_pops) == 0) stop("No NPZ populations found.")

all_modeled <- list()
ground_truth <- NULL
training_all <- list()
for (p in npz_pops) {
  best_dir <- get_best_individual_dir(p$pop_dir, p$meta)
  if (is.null(best_dir)) next
  ts <- extract_timeseries_second_last(best_dir)
  if (is.null(ts)) next
  modeled_df <- ts %>%
    select(Time, Variable, Value = Modeled) %>%
    mutate(Population = p$pop_id)
  all_modeled[[length(all_modeled) + 1]] <- modeled_df
  if (is.null(ground_truth)) {
    ground_truth <- ts %>% select(Time, Variable, Value = Observed)
  }
  tr <- get_training_series(p$meta)
  if (!is.null(tr)) training_all[[length(training_all) + 1]] <- tr %>% mutate(Population = p$pop_id)
}

modeled_all_df <- bind_rows(all_modeled)
rep_map <- setNames(
  paste0("Replicate ", seq_along(unique(modeled_all_df$Population))),
  sort(unique(modeled_all_df$Population))
)
modeled_all_df <- modeled_all_df %>% mutate(Replicate = rep_map[Population], Series = Replicate)
ground_truth <- ground_truth %>% mutate(Series = "Ground Truth")
plot_df <- bind_rows(modeled_all_df, ground_truth)

# --- Timeseries plot ---
n_rep <- length(unique(modeled_all_df$Replicate))
rep_cols <- setNames(hue_pal()(n_rep), sort(unique(modeled_all_df$Replicate)))
p_timeseries <- ggplot(plot_df, aes(x = Time, y = Value, color = Series, linetype = Series)) +
  geom_line(linewidth = 1) +
  facet_wrap(~Variable, scales = "free_y", ncol = 1) +
  scale_color_manual(values = c("Ground Truth" = "#0072B2", rep_cols)) +
  scale_linetype_manual(values = c("Ground Truth" = "solid", setNames(rep("dashed", n_rep), sort(unique(modeled_all_df$Replicate))))) +
  labs(x = "Time (days)", y = "Concentration (g C m^-3)") +
  theme_classic() +
  theme(legend.position = "bottom")

# Add icons
icons <- load_icons()
p_timeseries_icons <- add_facet_icons(p_timeseries, plot_df, icons)

# --- Training plot ---
training_all_df <- if (length(training_all) > 0) bind_rows(training_all) else NULL
p_training <- NULL
if (!is.null(training_all_df)) {
  training_all_df <- training_all_df %>% mutate(Replicate = rep_map[Population])
  p_training <- ggplot(training_all_df, aes(x = generation, y = objective_value, color = Replicate)) +
    scale_y_log10() +
    geom_line(linewidth = 1) +
    geom_point(size = 1.5) +
    theme_classic() +
    labs(title = "Training Progress", x = "Generation", y = "Objective Value (log)")
}

# --- Combine using cowplot ---
combined <- plot_grid(
  if (!is.null(p_training)) p_training else NULL,
  p_timeseries_icons,
  ncol = 2,
  rel_widths = c(1, 1)
)

# --- Save ---
figures_dir <- "Figures"
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
png(file.path(figures_dir, "NPZ_combined_with_icons.png"), width = 8, height = 10, units = "in", res = 300)
grid.draw(combined)
dev.off()
svg(file.path(figures_dir, "NPZ_combined_with_icons.svg"), width = 8, height = 10)
grid.draw(combined)
dev.off()

extract_ecological_scores <- function(metadata_path, individual_dir) {
  # Strict: read ONLY scores.json written by evaluate_ecological_characteristics.py
  scores_path <- file.path(individual_dir, "scores.json")
  if (!file.exists(scores_path)) {
    return(NULL)
  }

  sj <- safe_fromJSON(scores_path)
  if (is.null(sj) || is.null(sj$characteristic_scores) || is.null(sj$aggregate_scores)) {
    return(NULL)
  }

  cs <- sj$characteristic_scores
  agg <- sj$aggregate_scores

  # Helper to pull numeric scores for each named characteristic
  get_score <- function(name) {
    v <- tryCatch(cs[[name]][["score"]], error = function(e) NA_real_)
    if (is.null(v)) NA_real_ else suppressWarnings(as.numeric(v))
  }

  # --- Nine granular components (exact keys from the Python scorer) ---
  nutrient_equation_uptake <- get_score("nutrient_equation_uptake")
  nutrient_equation_recycling <- get_score("nutrient_equation_recycling")
  nutrient_equation_mixing <- get_score("nutrient_equation_mixing")
  phytoplankton_equation_growth <- get_score("phytoplankton_equation_growth")
  phytoplankton_equation_grazing_loss <- get_score("phytoplankton_equation_grazing_loss")
  phytoplankton_equation_mortality <- get_score("phytoplankton_equation_mortality")
  phytoplankton_equation_mixing <- get_score("phytoplankton_equation_mixing")
  zooplankton_equation_growth <- get_score("zooplankton_equation_growth")
  zooplankton_equation_mortality <- get_score("zooplankton_equation_mortality")

  # --- Extras (count + description + optional list) ---
  extras_count <- suppressWarnings(as.numeric(sj$extra_components_count))
  if (is.na(extras_count)) extras_count <- NA_real_
  extras_description <- if (!is.null(sj$extra_components_description)) {
    as.character(sj$extra_components_description)
  } else {
    NA_character_
  }
  extras_list_json <- if (!is.null(sj$extra_components)) {
    jsonlite::toJSON(sj$extra_components, auto_unbox = TRUE, null = "null")
  } else {
    NA_character_
  }

  # Build a single-row data.frame for this individual
  data.frame(
    individual = basename(individual_dir),

    # Aggregates from scores.json (no fallbacks)
    total_score = suppressWarnings(as.numeric(agg$raw_total)), # unnormalized (max ≈ 9)
    normalized_total = suppressWarnings(as.numeric(agg$normalized_total)), # 0–1
    final_score = suppressWarnings(as.numeric(agg$final_score)), # may be NA if absent

    # Granular components (9)
    nutrient_equation_uptake = nutrient_equation_uptake,
    nutrient_equation_recycling = nutrient_equation_recycling,
    nutrient_equation_mixing = nutrient_equation_mixing,
    phytoplankton_equation_growth = phytoplankton_equation_growth,
    phytoplankton_equation_grazing_loss = phytoplankton_equation_grazing_loss,
    phytoplankton_equation_mortality = phytoplankton_equation_mortality,
    phytoplankton_equation_mixing = phytoplankton_equation_mixing,
    zooplankton_equation_growth = zooplankton_equation_growth,
    zooplankton_equation_mortality = zooplankton_equation_mortality,

    # Extras
    extras_count = extras_count,
    extras_description = extras_description,
    extras_list_json = extras_list_json,
    stringsAsFactors = FALSE
  )
}

# Helper: find the directory that contains scores.json for a given individual ID
find_scores_dir <- function(pop_dir, individual_id) {
  candidates <- c(
    file.path(pop_dir, "CULLED", individual_id),
    file.path(pop_dir, individual_id)
  )
  hit <- candidates[file.exists(file.path(candidates, "scores.json"))]
  if (length(hit) == 0) NULL else hit[1]
}

ecology_results <- list()
for (p in npz_pops) {
  pop_dir <- p$pop_dir
  meta_path <- file.path(pop_dir, "population_metadata.json")
  pop_meta <- safe_fromJSON(meta_path)
  if (is.null(pop_meta)) next

  # Kept individuals from population metadata
  kept_df <- tryCatch(bind_rows(pop_meta$generations$best_individuals), error = function(e) NULL)
  kept_ids <- if (!is.null(kept_df) && "individual" %in% names(kept_df)) unique(kept_df$individual) else character(0)
  if (length(kept_ids) == 0) next

  # Only consider individuals that HAVE scores.json (in CULLED or top-level)
  indiv_dirs <- lapply(kept_ids, function(id) find_scores_dir(pop_dir, id))
  indiv_dirs <- indiv_dirs[!vapply(indiv_dirs, is.null, logical(1))]
  if (length(indiv_dirs) == 0) next

  for (indiv_dir in indiv_dirs) {
    eco <- extract_ecological_scores(NA, indiv_dir) # metadata_path ignored (strict scores.json)
    if (!is.null(eco)) {
      eco$objective_value <- get_objective_value(indiv_dir)
      eco$Population <- basename(pop_dir)
      ecology_results[[length(ecology_results) + 1]] <- eco
    }
  }
}
ecology_all <- if (length(ecology_results) > 0) bind_rows(ecology_results) else NULL
# Collect ecological scores from "kept" individuals across all NPZ populations (mirrors your approach). [1](https://csiroau-my.sharepoint.com/personal/spi085_csiro_au/Documents/Microsoft%20Copilot%20Chat%20Files/analyze_NPZ.txt)
ecology_results <- list()

for (p in npz_pops) {
  pop_dir <- p$pop_dir
  meta_path <- file.path(pop_dir, "population_metadata.json")
  pop_meta <- safe_fromJSON(meta_path)
  if (is.null(pop_meta)) next

  # Kept individuals from metadata (same field you used): generations$best_individuals → individual column. [1](https://csiroau-my.sharepoint.com/personal/spi085_csiro_au/Documents/Microsoft%20Copilot%20Chat%20Files/analyze_NPZ.txt)
  kept_df <- tryCatch(bind_rows(pop_meta$generations$best_individuals), error = function(e) NULL)
  kept_ids <- if (!is.null(kept_df) && "individual" %in% names(kept_df)) unique(kept_df$individual) else character(0)

  # Helper to find metadata.json for an individual (culled vs toplevel), same logic you wrote. [1](https://csiroau-my.sharepoint.com/personal/spi085_csiro_au/Documents/Microsoft%20Copilot%20Chat%20Files/analyze_NPZ.txt)
  find_metadata_file <- function(individual_id) {
    culled_path   <- file.path(pop_dir, "CULLED", individual_id, "metadata.json")
    toplevel_path <- file.path(pop_dir, individual_id, "metadata.json")
    if (file.exists(culled_path)) return(culled_path)
    if (file.exists(toplevel_path)) return(toplevel_path)
    NULL
  }

  files <- sapply(kept_ids, find_metadata_file)
  files <- files[!sapply(files, is.null)]
  for (fp in files) {
    indiv_dir <- dirname(fp)
    eco <- extract_ecological_scores(fp, indiv_dir)
    if (!is.null(eco)) {
      eco$objective_value <- get_objective_value(indiv_dir)
      eco$Population <- basename(pop_dir)
      ecology_results[[length(ecology_results) + 1]] <- eco
    }
  }
}

ecology_all <- if (length(ecology_results) > 0) bind_rows(ecology_results) else NULL

# --- TOTAL vs OBJECTIVE with axis break + bottom axis only ---

library(ggbreak)

if (!is.null(ecology_all) && nrow(ecology_all) > 0) {
  results_base <- ecology_all %>%
    mutate(
      objective_value = suppressWarnings(as.numeric(objective_value)),
      total_score     = suppressWarnings(as.numeric(total_score))
    )

  results_clean <- results_base %>%
    filter(
      !is.na(objective_value), !is.na(total_score),
      is.finite(objective_value), is.finite(total_score),
      objective_value > 0
    )

  eco_threshold <- 6.0
  obj_threshold <- 0.1

  low_eco_scores <- results_clean %>% filter(total_score < eco_threshold)
  good_performers <- results_clean %>% filter(objective_value < obj_threshold)

  low_eco_model <- if (nrow(low_eco_scores) >= 2) tryCatch(lm(total_score ~ objective_value, data = low_eco_scores), error = function(e) NULL) else NULL
  good_perf_model <- if (nrow(good_performers) >= 2) tryCatch(lm(total_score ~ objective_value, data = good_performers), error = function(e) NULL) else NULL

  low_eco_r2 <- if (!is.null(low_eco_model)) summary(low_eco_model)$r.squared else NA_real_
  good_perf_r2 <- if (!is.null(good_perf_model)) summary(good_perf_model)$r.squared else NA_real_

  model_comparison_label <- sprintf(
    "Low eco (<%.1f): %s\nGood performers (<%.1f obj): %s",
    eco_threshold,
    if (!is.na(low_eco_r2)) sprintf("R² = %.3f", low_eco_r2) else "not enough data",
    obj_threshold,
    if (!is.na(good_perf_r2)) sprintf("R² = %.3f", good_perf_r2) else "not enough data"
  )

  # IQR-based outlier cutoff for the X axis (objective_value)
  x_q <- quantile(results_clean$objective_value, probs = c(0.25, 0.75), na.rm = TRUE)
  x_iqr <- diff(x_q)
  x_thr <- x_q[2] + 1.5 * x_iqr
  x_max <- suppressWarnings(max(results_clean$objective_value, na.rm = TRUE))

  p_ecology_total <- ggplot(results_clean, aes(x = objective_value, y = total_score)) +
    geom_point(aes(color = case_when(
      (total_score < eco_threshold) & (objective_value < obj_threshold) ~ "Both",
      (total_score < eco_threshold) ~ "Low Eco Only",
      (objective_value < obj_threshold) ~ "Good Performers Only",
      TRUE ~ "Neither"
    )), alpha = 0.6, size = 1.5) +
    # {
    #   if (!is.null(low_eco_model)) geom_smooth(data = low_eco_scores, method = "lm", se = FALSE, color = "#0173B2", linewidth = 1.2)
    # } +
    # {
    #   if (!is.null(good_perf_model)) geom_smooth(data = good_performers, method = "lm", se = FALSE, color = "#DE8F05", linewidth = 1.2)
    # } +
    # geom_hline(yintercept = eco_threshold, linetype = "dashed", color = "gray50", alpha = 0.7) +
    # geom_vline(xintercept = obj_threshold, linetype = "dashed", color = "gray50", alpha = 0.7) +
    # scale_color_manual(
    #   values = c("Low Eco Only" = "#0173B2", "Good Performers Only" = "#DE8F05", "Both" = "#CC78BC", "Neither" = "gray70"),
    #   name = "Point Category",
    #   labels = c("Low Eco Only" = "Low Eco Scores", "Good Performers Only" = "Good Performers", "Both" = "Both Groups", "Neither" = "Neither Group")
    # ) +
    scale_x_log10() + # compatible with ggbreak [4](https://cran.r-project.org/web//packages/ggbreak/vignettes/ggbreak.html)
    theme_classic() +
    labs(
      x = "Objective Value (log scale)",
      y = "Total Ecological Score",
      caption = sprintf("n = %d individuals across %d NPZ populations", nrow(results_clean), length(unique(results_clean$Population)))
    ) +
    geom_text(
      x = min(results_clean$objective_value, na.rm = TRUE),
      y = max(results_clean$total_score, na.rm = TRUE) - 0.5,
      label = model_comparison_label,
      hjust = 0, vjust = 1, size = 3.5
    )

  # Insert the axis break and hide the TOP axis (text, ticks, line)
  if (is.finite(x_thr) && is.finite(x_max) && x_max > x_thr * 1.05) {
    p_ecology_total <- p_ecology_total +
      scale_x_break(c(x_thr, x_max), space = 0.18) + # break creates two subpanels (top & bottom) [4](https://cran.r-project.org/web//packages/ggbreak/vignettes/ggbreak.html)
      theme(
        axis.text.x.top  = element_blank(),
        axis.ticks.x.top = element_blank(),
        axis.line.x.top  = element_blank()
      ) # hide the top axis elements; keep bottom only [1](https://ggplot2.tidyverse.org/reference/theme.html)[3](https://stackoverflow.com/questions/71992148/removing-duplicate-top-x-axis-in-ggplot-plus-add-axis-break)
  }

  figures_dir <- "Figures"
  dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
  ggsave(file.path(figures_dir, "ecological_vs_objective_total_ALL.png"), p_ecology_total, width = 8, height = 6)
  ggsave(file.path(figures_dir, "ecological_vs_objective_total_ALL.svg"), p_ecology_total, width = 8, height = 6)
}
# --- COMPONENTS vs OBJECTIVE with axis break + bottom axis only ---

library(ggbreak)

if (!is.null(ecology_all) && nrow(ecology_all) > 0) {
  component_scores <- c(
    "nutrient_equation_uptake",
    "nutrient_equation_recycling",
    "nutrient_equation_mixing",
    "phytoplankton_equation_growth",
    "phytoplankton_equation_grazing_loss",
    "phytoplankton_equation_mortality",
    "phytoplankton_equation_mixing",
    "zooplankton_equation_growth",
    "zooplankton_equation_mortality"
  )

  results_long <- ecology_all %>%
    mutate(objective_value = suppressWarnings(as.numeric(objective_value))) %>%
    dplyr::select(Population, individual, objective_value, all_of(component_scores)) %>%
    tidyr::pivot_longer(cols = all_of(component_scores), names_to = "characteristic", values_to = "score") %>%
    mutate(
      score = suppressWarnings(as.numeric(score)),
      characteristic_clean = gsub("_", " ", tools::toTitleCase(characteristic))
    ) %>%
    filter(
      !is.na(objective_value), !is.na(score),
      is.finite(objective_value), is.finite(score),
      objective_value > 0
    )

  # Shared outlier threshold for X across facets (IQR-based)
  x_q <- quantile(results_long$objective_value, probs = c(0.25, 0.75), na.rm = TRUE)
  x_iqr <- diff(x_q)
  x_thr <- x_q[2] + 1.5 * x_iqr
  x_max <- suppressWarnings(max(results_long$objective_value, na.rm = TRUE))

  # Correlation label per facet (unchanged)
  correlations <- results_long %>%
    group_by(characteristic_clean) %>%
    summarise(
      n = n(),
      var_score = var(score, na.rm = TRUE),
      var_obj = var(objective_value, na.rm = TRUE),
      r = if (!is.na(var_score) && var_score > 0 && !is.na(var_obj) && var_obj > 0 && n >= 3) cor(score, objective_value) else NA_real_,
      p = if (!is.na(r)) tryCatch(cor.test(score, objective_value)$p.value, error = function(e) NA_real_) else NA_real_
    ) %>%
    mutate(label = ifelse(!is.na(r) & !is.na(p), sprintf("r = %.2f\np = %.2e", r, p), "not enough data"))

  p_ecology_components <- ggplot(results_long, aes(x = objective_value, y = score)) +
    geom_point(alpha = 0.6, color = "#0072B2") +
    geom_smooth(method = "lm", se = FALSE, color = "#D55E00", linetype = "dashed") +
    scale_x_log10() + # compatible with ggbreak’s break scales [4](https://cran.r-project.org/web//packages/ggbreak/vignettes/ggbreak.html)
    facet_wrap(~characteristic_clean, ncol = 2) +
    geom_text(
      data = correlations,
      x = min(results_long$objective_value, na.rm = TRUE),
      y = max(results_long$score, na.rm = TRUE),
      label = correlations$label,
      hjust = 0, vjust = 1, size = 3
    ) +
    theme_classic() +
    labs(
      x = "Objective Value (log scale)",
      y = "Score",
      caption = sprintf("n = %d individuals across %d NPZ populations", length(unique(results_long$individual)), length(unique(results_long$Population)))
    )

  # Insert the same X-axis break across all facets; hide the top axis elements
  if (is.finite(x_thr) && is.finite(x_max) && x_max > x_thr * 1.05) {
    p_ecology_components <- p_ecology_components +
      scale_x_break(c(x_thr, x_max), space = 0.18) + # break produces top & bottom subpanels per facet [4](https://cran.r-project.org/web//packages/ggbreak/vignettes/ggbreak.html)
      theme(
        axis.text.x.top  = element_blank(),
        axis.ticks.x.top = element_blank(),
        axis.line.x.top  = element_blank()
      ) # keep bottom axis only in all facets [1](https://ggplot2.tidyverse.org/reference/theme.html)
  }

  figures_dir <- "Figures"
  dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
  ggsave(file.path(figures_dir, "ecological_characteristics_vs_objective_ALL.png"), p_ecology_components, width = 12, height = 8)
  ggsave(file.path(figures_dir, "ecological_characteristics_vs_objective_ALL.svg"), p_ecology_components, width = 12, height = 8)
}
# --- EXTRAS summary across populations ---

if (!is.null(ecology_all) && nrow(ecology_all) > 0) {
  extras_df <- ecology_all %>%
    dplyr::select(Population, individual, extras_count, extras_description, extras_list_json) %>%
    mutate(
      extras_count = suppressWarnings(as.numeric(extras_count)),
      extras_description = as.character(extras_description),
      extras_list_json = as.character(extras_list_json)
    )

  # Save a tidy table
  results_dir <- "Manuscript/Results/ecology_analysis"
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(extras_df, file.path(results_dir, "extras_summary_ALL.csv"), row.names = FALSE)

  # Bar chart of extras_count distribution per population
  extras_counts <- extras_df %>%
    filter(!is.na(extras_count)) %>%
    group_by(Population) %>%
    summarise(mean_extras = mean(extras_count, na.rm = TRUE), n = n())

  p_extras <- ggplot(extras_counts, aes(x = Population, y = mean_extras)) +
    geom_col(fill = "#3EBCD2") +
    theme_classic() +
    labs(title = "Mean Extras Count by Population", x = "Population", y = "Mean Extras Count")

  figures_dir <- "Figures"
  dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
  ggsave(file.path(figures_dir, "extras_mean_by_population.png"), p_extras, width = 8, height = 4)
  ggsave(file.path(figures_dir, "extras_mean_by_population.svg"), p_extras, width = 8, height = 4)
}

# Save combined ecology results
results_dir <- "Manuscript/Results/ecology_analysis"
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(results_clean, file.path(results_dir, "ecological_scores_ALL.csv"), row.names = FALSE)

# Text summary (generalized)
sink(file.path(results_dir, "analysis_summary_ALL.txt"))
cat("Analysis Summary (ALL NPZ populations)\n")
cat("================\n\n")
cat("Number of individuals analyzed:", nrow(results_clean), "\n")
cat("Number of NPZ populations:", length(unique(results_clean$Population)), "\n\n")
cat("Summary Statistics:\n")
print(summary(results_clean))
cat("\nCorrelations with objective value:\n")
if (nrow(results_clean) >= 3) {
  total_cor <- cor.test(results_clean$total_score, results_clean$objective_value)
  cat(sprintf("Total Score: r = %.3f (p = %.3e)\n", total_cor$estimate, total_cor$p.value))
  for (sc in component_scores) {
    ct <- cor.test(results_clean[[sc]], results_clean$objective_value)
    cat(sprintf("%s: r = %.3f (p = %.3e)\n",
                gsub("_", " ", tools::toTitleCase(sc)),
                ct$estimate, ct$p.value))
  }
} else {
  cat("Not enough complete cases to calculate correlations.\n")
}
sink()


message("Finished: NPZ_timeseries_all_populations.(png|svg); NPZ_training_all_populations.(png|svg) and (optional) ecology plots saved in Figures/")