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

# --- Helper: shorten LLM names (everything after the last '/')
short_llm <- function(x) {
  ifelse(grepl("/", x), sub(".*/", "", x), x)
}


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
figures_dir <- "Figures"
npz_pops <- find_npz_populations()
if (length(npz_pops) == 0) stop("No NPZ populations found.")
# --- Collect modeled series, training, and per-pop best objective ---
all_modeled <- list()
ground_truth <- NULL
training_all <- list()
llm_map <- list()
best_candidates <- list() # NEW: track (Population, individual, LLM_short, objective_value) for global best

for (p in npz_pops) {
  best_dir <- get_best_individual_dir(p$pop_dir, p$meta)
  if (is.null(best_dir)) next

  ts <- extract_timeseries_second_last(best_dir)
  if (is.null(ts)) next

  # Use short LLM name for plotting/legend
  llm_name <- if (!is.null(p$meta$llm_choice)) p$meta$llm_choice else "Unknown"
  llm_short <- short_llm(llm_name)
  llm_map[[p$pop_id]] <- llm_short

  # Modeled series for the best individual in this population
  modeled_df <- ts %>%
    dplyr::select(Time, Variable, Value = Modeled) %>%
    dplyr::mutate(Population = p$pop_id, LLM = llm_short)
  all_modeled[[length(all_modeled) + 1]] <- modeled_df

  # Keep first ground truth
  if (is.null(ground_truth)) {
    ground_truth <- ts %>% dplyr::select(Time, Variable, Value = Observed)
  }

  # Training trajectory (use short LLM name)
  tr <- get_training_series(p$meta)
  if (!is.null(tr)) {
    training_all[[length(training_all) + 1]] <- tr %>% dplyr::mutate(Population = p$pop_id, LLM = llm_short)
  }

  # NEW: capture objective value of this best individual
  cur_obj <- get_objective_value(best_dir)
  best_candidates[[length(best_candidates) + 1]] <- data.frame(
    Population = p$pop_id,
    individual = basename(best_dir),
    LLM = llm_short,
    objective_value = cur_obj,
    stringsAsFactors = FALSE
  )
}

modeled_all_df <- dplyr::bind_rows(all_modeled)

# Replicate mapping (unchanged)
rep_map <- setNames(
  paste0("Replicate ", seq_along(unique(modeled_all_df$Population))),
  sort(unique(modeled_all_df$Population))
)
modeled_all_df <- modeled_all_df %>% dplyr::mutate(Replicate = rep_map[Population])

# Ground truth row additions (unchanged)
ground_truth <- ground_truth %>% dplyr::mutate(LLM = "Ground Truth", Replicate = "Ground Truth", Population = "Ground Truth")
plot_df <- dplyr::bind_rows(modeled_all_df, ground_truth)

# --- Determine global best by objective value across populations ---
best_candidates_df <- dplyr::bind_rows(best_candidates)
best_candidates_df <- best_candidates_df %>% dplyr::filter(is.finite(objective_value))
if (nrow(best_candidates_df) > 0) {
  best_global <- best_candidates_df %>%
    dplyr::arrange(objective_value) %>%
    dplyr::slice(1)
  best_pop_id <- best_global$Population
  # Console identification
  cat(sprintf(
    "\nBest NPZ model by objective value:\n  Population: %s\n  Individual: %s\n  LLM: %s\n  Objective: %.6f\n\n",
    best_global$Population, best_global$individual, best_global$LLM, best_global$objective_value
  ))
} else {
  best_global <- NULL
  best_pop_id <- NA_character_
  cat("\nBest NPZ model by objective value: not found (no finite objective values).\n\n")
}

modeled_all_df <- bind_rows(all_modeled)
rep_map <- setNames(
  paste0("Replicate ", seq_along(unique(modeled_all_df$Population))),
  sort(unique(modeled_all_df$Population))
)
modeled_all_df <- modeled_all_df %>% mutate(Replicate = rep_map[Population])
ground_truth <- ground_truth %>% mutate(LLM = "Ground Truth", Replicate = "Ground Truth", Population = "Ground Truth")
plot_df <- bind_rows(modeled_all_df, ground_truth)

# --- Timeseries plot ---
## --- Colors/legend for short LLM names ---
llm_names <- unique(modeled_all_df$LLM) # these are short now
n_llm <- length(llm_names)
llm_cols <- setNames(scales::hue_pal()(n_llm), sort(llm_names))

# Timeseries plot (base)
p_timeseries <- ggplot(
  plot_df,
  aes(
    x = Time,
    y = Value,
    color = LLM,
    group = Replicate,
    linewidth = ifelse(Replicate == "Ground Truth", 3, 0.8),
    linetype = ifelse(Replicate == "Ground Truth", "solid", "dashed"),
    alpha = ifelse(Replicate == "Ground Truth", 1.0, 0.5)
  )
) +
  geom_line() +
  facet_wrap(~Variable, scales = "free_y", ncol = 1) +
  scale_color_manual(values = c("Ground Truth" = "black", llm_cols)) +
  scale_linewidth_identity() +
  scale_linetype_identity() +
  scale_alpha_identity() +
  labs(x = "Time (days)", y = "Concentration (g C m^-3)", color = "Model") +
  theme_classic() +
  theme(legend.position = "none")

# Add icons
icons <- load_icons()
p_timeseries_icons <- add_facet_icons(p_timeseries, plot_df, icons)

# --- Red crosses on the model dynamics (timeseries) for the global best ---
if (!is.na(best_pop_id)) {
  best_plot_df <- plot_df %>%
    dplyr::filter(Population == best_pop_id, Replicate != "Ground Truth", LLM != "Ground Truth")
  p_timeseries <- p_timeseries +
    ggplot2::geom_point(
      data = best_plot_df,
      aes(x = Time, y = Value),
      inherit.aes = FALSE,
      color = "red",
      size = 0.5,
      shape = 4
    )
}

# Add icons (unchanged)
icons <- load_icons()
p_timeseries_icons <- add_facet_icons(p_timeseries, plot_df, icons)

# --- Training plot (existing creation) ---
training_all_df <- if (length(training_all) > 0) dplyr::bind_rows(training_all) else NULL
p_training <- NULL
if (!is.null(training_all_df)) {
  training_all_df <- training_all_df %>% dplyr::mutate(Replicate = rep_map[Population])
  p_training <- ggplot(training_all_df, aes(x = generation, y = objective_value, color = LLM, group = Replicate)) +
    scale_y_log10() +
    geom_line(linewidth = 1) +
    geom_point(size = 1.5) +
    scale_color_manual(values = llm_cols) +
    theme_classic() +
    labs(title = "Training Progress", x = "Generation", y = "Objective Value (log)", color = "Model") +
    theme(legend.position = "none")

  # --- Red crosses on the best population's training curve ---
  if (!is.na(best_pop_id)) {
    best_training_df <- training_all_df %>% dplyr::filter(Population == best_pop_id)
    p_training <- p_training +
      ggplot2::geom_point(
        data = best_training_df,
        aes(x = generation, y = objective_value),
        color = "red",
        size = 0.5,
        shape = 4
      )
  }
}

# --- Training plot ---
# --- Training plot (existing creation) ---
training_all_df <- if (length(training_all) > 0) dplyr::bind_rows(training_all) else NULL
p_training <- NULL
if (!is.null(training_all_df)) {
  training_all_df <- training_all_df %>% dplyr::mutate(Replicate = rep_map[Population])
  p_training <- ggplot(training_all_df, aes(x = generation, y = objective_value, color = LLM, group = Replicate)) +
    scale_y_log10() +
    geom_line(linewidth = 1) +
    geom_point(size = 1.5) +
    scale_color_manual(values = llm_cols) +
    theme_classic() +
    labs(title = "Training Progress", x = "Generation", y = "Objective Value (log)", color = "Model") +
    theme(legend.position = "none")

  # --- Red dots on the best population's training curve ---
  if (!is.na(best_pop_id)) {
    best_training_df <- training_all_df %>% dplyr::filter(Population == best_pop_id)
    p_training <- p_training +
      ggplot2::geom_point(
        data = best_training_df,
        aes(x = generation, y = objective_value),
        color = "red",
        size = 0.5
      )
  }
}

# --- Combine using cowplot with shared legend (robust & finite) ---

# Ensure output dir exists
figures_dir <- if (exists("figures_dir")) figures_dir else "Figures"
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

# ---- 1) Legend builder that never collapses ----
# expects llm_names and llm_cols already defined upstream
# --- Legend builder with optional 'Best Performer' entry ---
# Uses short LLM names already present in llm_names / llm_cols from earlier steps

# 1) Build legend levels (append 'Best Performer' if we found one)
legend_levels <- c("Ground Truth", sort(llm_names))
if (!is.na(best_pop_id)) {
  legend_levels <- c(legend_levels, "Best Performer")
}

# 2) Data for legend points with shapes
legend_data <- data.frame(
  x   = seq_along(legend_levels),
  y   = 1,
  LLM = factor(legend_levels, levels = legend_levels),
  shape_val = ifelse(legend_levels == "Best Performer", 4, 16)  # 4 = cross, 16 = circle
)

# 3) Color map: start with GT + LLMs, then add 'Best Performer' = red when present
values_map <- c("Ground Truth" = "black", llm_cols)
if (!is.na(best_pop_id)) {
  values_map <- c(values_map, "Best Performer" = "red")
}

# 4) Shape map for legend
shape_map <- setNames(legend_data$shape_val, legend_levels)

# 5) Build the legend plot with explicit shape per entry
legend_plot <- ggplot(legend_data, aes(x = x, y = y)) +
  geom_point(aes(color = LLM, shape = LLM), size = 4, show.legend = TRUE) +
  scale_color_manual(
    values = values_map,
    name   = "Model",
    breaks = legend_levels,
    drop   = FALSE,
    guide = guide_legend(
      nrow = 1,
      override.aes = list(
        shape = shape_map[legend_levels],
        size = 4
      )
    )
  ) +
  scale_shape_manual(
    values = shape_map,
    name   = "Model",
    breaks = legend_levels,
    drop   = FALSE,
    guide = "none"
  ) +
  theme_void() +
  theme(
    legend.position = "bottom",
    legend.text     = element_text(size = 9, margin = margin(r = 8)),
    legend.title    = element_text(size = 10, face = "bold"),
    legend.key.size = unit(0.7, "cm"),
    legend.spacing.x = unit(0.3, "cm"),
    plot.margin = margin(t = 10, r = 80, b = 10, l = 80)
  )

legend_grob <- cowplot::get_legend(legend_plot)

# --- Fallback manual legend row (if guide extraction fails) ---
legend_gg <- if (inherits(legend_grob, "zeroGrob")) {
  key_df <- data.frame(
    LLM = factor(legend_levels, levels = legend_levels),
    x = seq_along(legend_levels), y = 1
  )
  manual_legend <- ggplot(key_df, aes(x, y)) +
    geom_point(aes(color = LLM), size = 4) +
    geom_text(aes(label = LLM), hjust = 0, nudge_x = 0.25, size = 3.5) +
    scale_color_manual(values = values_map, guide = "none", drop = FALSE) +
    coord_cartesian(clip = "off") +
    theme_void() +
    theme(plot.margin = margin(5.5, 5.5, 5.5, 5.5))
  cowplot::ggdraw(manual_legend)
} else {
  cowplot::ggdraw(legend_grob)
}




# ---- 2) Wrap grobs so cowplot treats them like ggplots ----
# p_timeseries_icons is a gtable (from add_facet_icons), so wrap it;
# legend_grob is also a gtable
p_timeseries_icons_gg <- cowplot::ggdraw(p_timeseries_icons)

# If legend extraction failed or yielded a zeroGrob, build a manual key row instead
legend_gg <- if (inherits(legend_grob, "zeroGrob")) {
  # Fallback: a "manual legend row" (no guide-box), visually similar
  key_df <- data.frame(LLM = factor(legend_levels, levels = legend_levels), x = seq_along(legend_levels), y = 1)
  manual_legend <- ggplot(key_df, aes(x, y)) +
    geom_point(aes(color = LLM), size = 4) +
    geom_text(aes(label = LLM), hjust = 0, nudge_x = 0.25, size = 3.5) +
    scale_color_manual(values = c("Ground Truth" = "black", llm_cols), guide = "none", drop = FALSE) +
    coord_cartesian(clip = "off") +
    theme_void() +
    theme(plot.margin = margin(5.5, 5.5, 5.5, 5.5))
  cowplot::ggdraw(manual_legend)
} else {
  cowplot::ggdraw(legend_grob)
}

# ---- 3) Build the upper row (with or without the training panel) ----
upper_row <- if (!is.null(p_training)) {
  cowplot::plot_grid(
    cowplot::ggdraw(p_training),
    p_timeseries_icons_gg,
    ncol = 2,
    rel_widths = c(1, 1),
    align = "h",
    axis = "tb"
  )
} else {
  p_timeseries_icons_gg
}

# ---- 4) Add the legend at the bottom (use finite relative heights) ----
combined_with_legend <- cowplot::plot_grid(
  upper_row,
  legend_gg,
  ncol = 1,
  rel_heights = c(1, 0.16) # Adequate height for legend
)

# ---- 5) Save robustly (ggsave draws grid/cowplot content correctly) ----
ggplot2::ggsave(
  file.path(figures_dir, "NPZ_combined_with_icons.png"),
  combined_with_legend,
  width = 12, height = 10, dpi = 300
)
ggplot2::ggsave(
  file.path(figures_dir, "NPZ_combined_with_icons.svg"),
  combined_with_legend,
  width = 12, height = 10
)

# --- Alternative (manual devices): explicitly draw the grid object ---
# png(file.path(figures_dir, "NPZ_combined_with_icons.png"), width = 12, height = 10, units = "in", res = 300)
# grid::grid.newpage(); grid::grid.draw(combined_with_legend); dev.off()
# svg(file.path(figures_dir, "NPZ_combined_with_icons.svg"), width = 12, height = 10)
# grid::grid.newpage(); grid::grid.draw(combined_with_legend); dev.off()

# List every INDIVIDUAL_* directory that exists in a population (both locations)
list_all_individual_ids <- function(pop_dir) {
  ids_top <- list.dirs(pop_dir, full.names = TRUE, recursive = FALSE)
  ids_top <- basename(ids_top[grepl("^INDIVIDUAL_", basename(ids_top))])

  culled_dir <- file.path(pop_dir, "CULLED")
  ids_culled <- if (dir.exists(culled_dir)) {
    d <- list.dirs(culled_dir, full.names = TRUE, recursive = FALSE)
    basename(d[grepl("^INDIVIDUAL_", basename(d))])
  } else {
    character(0)
  }

  unique(c(ids_top, ids_culled))
}

# Reuse your strict scores.json reader (evaluate_ecological_characteristics.py output)
extract_ecological_scores <- function(metadata_path, individual_dir) {
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
  get_score <- function(name) {
    v <- tryCatch(cs[[name]][["score"]], error = function(e) NA_real_)
    if (is.null(v)) NA_real_ else suppressWarnings(as.numeric(v))
  }

  nutrient_equation_uptake <- get_score("nutrient_equation_uptake")
  nutrient_equation_recycling <- get_score("nutrient_equation_recycling")
  nutrient_equation_mixing <- get_score("nutrient_equation_mixing")
  phytoplankton_equation_growth <- get_score("phytoplankton_equation_growth")
  phytoplankton_equation_grazing_loss <- get_score("phytoplankton_equation_grazing_loss")
  phytoplankton_equation_mortality <- get_score("phytoplankton_equation_mortality")
  phytoplankton_equation_mixing <- get_score("phytoplankton_equation_mixing")
  zooplankton_equation_growth <- get_score("zooplankton_equation_growth")
  zooplankton_equation_mortality <- get_score("zooplankton_equation_mortality")

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

  data.frame(
    individual = basename(individual_dir),
    total_score = suppressWarnings(as.numeric(agg$raw_total)),
    normalized_total = suppressWarnings(as.numeric(agg$normalized_total)),
    final_score = suppressWarnings(as.numeric(agg$final_score)),
    nutrient_equation_uptake = nutrient_equation_uptake,
    nutrient_equation_recycling = nutrient_equation_recycling,
    nutrient_equation_mixing = nutrient_equation_mixing,
    phytoplankton_equation_growth = phytoplankton_equation_growth,
    phytoplankton_equation_grazing_loss = phytoplankton_equation_grazing_loss,
    phytoplankton_equation_mortality = phytoplankton_equation_mortality,
    phytoplankton_equation_mixing = phytoplankton_equation_mixing,
    zooplankton_equation_growth = zooplankton_equation_growth,
    zooplankton_equation_mortality = zooplankton_equation_mortality,
    extras_count = extras_count,
    extras_description = extras_description,
    extras_list_json = extras_list_json,
    stringsAsFactors = FALSE
  )
}

# If scores.json is missing, return a stub row with NA scores so we still include the individual
stub_ecology_row <- function(individual_dir) {
  data.frame(
    individual = basename(individual_dir),
    total_score = NA_real_, normalized_total = NA_real_, final_score = NA_real_,
    nutrient_equation_uptake = NA_real_,
    nutrient_equation_recycling = NA_real_,
    nutrient_equation_mixing = NA_real_,
    phytoplankton_equation_growth = NA_real_,
    phytoplankton_equation_grazing_loss = NA_real_,
    phytoplankton_equation_mortality = NA_real_,
    phytoplankton_equation_mixing = NA_real_,
    zooplankton_equation_growth = NA_real_,
    zooplankton_equation_mortality = NA_real_,
    extras_count = NA_real_,
    extras_description = NA_character_,
    extras_list_json = NA_character_,
    stringsAsFactors = FALSE
  )
}

# Helper: find the directory that contains scores.json for a given individual ID (prefer CULLED if present)
find_scores_dir <- function(pop_dir, individual_id) {
  candidates <- c(
    file.path(pop_dir, "CULLED", individual_id),
    file.path(pop_dir, individual_id)
  )
  hit <- candidates[file.exists(file.path(candidates, "scores.json"))]
  if (length(hit) == 0) NULL else hit[1]
}

# Helper: find *any* directory for an individual (even if scores.json is absent), to read objective_value if available
find_individual_dir_any <- function(pop_dir, individual_id) {
  culled <- file.path(pop_dir, "CULLED", individual_id)
  top <- file.path(pop_dir, individual_id)
  if (dir.exists(culled)) {
    return(culled)
  }
  if (dir.exists(top)) {
    return(top)
  }
  NULL
}

# ---- Build ecology_all from ALL individuals (no kept_ids filter) ----
ecology_results <- list()
for (p in npz_pops) {
  pop_dir <- p$pop_dir

  # Enumerate all INDIVIDUAL_* IDs in top-level and CULLED
  all_ids <- list_all_individual_ids(pop_dir)

  if (length(all_ids) == 0) next

  for (id in all_ids) {
    # Prefer the directory that has scores.json; otherwise fall back to any existing dir
    scores_dir <- find_scores_dir(pop_dir, id)
    indiv_dir <- if (!is.null(scores_dir)) scores_dir else find_individual_dir_any(pop_dir, id)
    if (is.null(indiv_dir)) {
      print("skipping")
      print(indiv_dir)
      next
    }
    eco <- extract_ecological_scores(NA, indiv_dir)
    if (is.null(eco)) eco <- stub_ecology_row(indiv_dir)

    # Attach objective value (may be NA if model_report.json is absent)
    eco$objective_value <- get_objective_value(indiv_dir)
    eco$Population <- basename(pop_dir)
    eco$llm <- if (!is.null(p$meta$llm_choice)) p$meta$llm_choice else "Unknown"

    ecology_results[[length(ecology_results) + 1]] <- eco
  }
}

ecology_all <- if (length(ecology_results) > 0) bind_rows(ecology_results) else NULL
# (From here on, your downstream plots/tables use ecology_all and will automatically
# filter to complete cases where needed, but counts will include every individual.)

# ==== Ecological score (normalized_total) vs Objective value: robust analysis & visuals ====
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
})

stopifnot(!is.null(ecology_all), nrow(ecology_all) > 0)

# -------- Data prep --------
# Use normalized_total as ecological score in [0,1], and objective_value as performance (lower is better).
df <- ecology_all %>%
  transmute(
    Population = as.character(Population),
    individual = as.character(individual),
    llm        = as.character(llm),
    obj        = suppressWarnings(as.numeric(objective_value)),
    eco        = suppressWarnings(as.numeric(normalized_total))
  ) %>%
  # Keep valid rows (objective must be >0 to allow log-scale if needed)
  filter(is.finite(obj), obj > 0, is.finite(eco), eco >= 0, eco <= 1)

if (nrow(df) < 3) stop("Not enough valid rows for analysis (need >= 3).")

# -------- Robust association measures (interpretation: negative => higher eco ↔ lower objective) --------
spearman <- suppressWarnings(cor.test(df$eco, df$obj, method = "spearman", exact = FALSE))
kendall <- suppressWarnings(cor.test(df$eco, df$obj, method = "kendall", exact = FALSE))

# Concordance probability: C = P{obj_i < obj_j | eco_i > eco_j}
concordance_probability <- function(eco, obj, max_pairs = 2e6, seed = 42L) {
  set.seed(seed)
  n <- length(eco)
  if (n < 2) {
    return(NA_real_)
  }
  m <- min(max_pairs, n * (n - 1) / 2)
  i <- sample.int(n, size = m, replace = TRUE)
  j <- sample.int(n, size = m, replace = TRUE)
  keep <- (i != j) & (eco[i] > eco[j]) & is.finite(obj[i]) & is.finite(obj[j])
  i <- i[keep]
  j <- j[keep]
  if (length(i) == 0) {
    return(NA_real_)
  }
  xi <- obj[i]
  xj <- obj[j]
  valid <- (xi != xj)
  if (!any(valid)) {
    return(NA_real_)
  }
  mean(xi[valid] < xj[valid])
}
C <- concordance_probability(df$eco, df$obj)

#####################################
# ==== Ecological score vs Objective: SIMPLE LM (fits include all, y capped at 1.0) ====
# ==== Outliers shown with arrowheads + unified & per‑LLM stats (R², adj‑R², p, n) ====

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
})

# Safety: define short_llm() if not already defined upstream
if (!exists("short_llm")) {
  short_llm <- function(x) ifelse(grepl("/", x), sub(".*/", "", x), x)
}

stopifnot(!is.null(ecology_all), nrow(ecology_all) > 0)

# --- Constants ---
cap_val <- 1.0 # clamp threshold for objective (used for fitting and plotting)
tri_offset <- 0.05 # arrowhead vertical offset above clamp

# --- PLOTTING DATA (ALL points with obj > 0). Create llm_short here to avoid missing column issues. ---
df_plot <- ecology_all %>%
  transmute(
    Population = as.character(Population),
    individual = as.character(individual),
    llm        = as.character(llm),
    obj        = suppressWarnings(as.numeric(objective_value)),
    eco        = suppressWarnings(as.numeric(normalized_total))
  ) %>%
  filter(is.finite(obj), obj > 0, is.finite(eco)) %>%
  mutate(
    eco        = pmin(pmax(eco, 0), 1), # clamp eco to [0,1]
    llm_short  = short_llm(llm), # short label for legend/stats
    is_clamped = obj > cap_val,
    obj_plot   = pmin(obj, cap_val), # visual y value
    tri_y      = ifelse(is_clamped, cap_val + tri_offset, NA_real_)
  )

if (nrow(df_plot) < 3) stop("Not enough valid rows to plot (need >= 3).")

# --- FIT DATA (ALL points, but y is capped at 1.0 for the fit) ---
df_fit <- df_plot %>%
  transmute(
    eco       = eco,
    obj_fit   = obj_plot,
    llm_short = llm_short
  )

# --- Palette (short LLM names) ---
llm_levels <- sort(unique(df_plot$llm_short))
llm_cols <- setNames(scales::hue_pal()(length(llm_levels)), llm_levels)

# --- Unified linear model (capped y) ---
fit_overall <- lm(obj_fit ~ eco, data = df_fit)
overall_sum <- summary(fit_overall)

# --- Per‑LLM linear models (capped y), robust to tiny groups/zero variance ---
llm_splits <- split(df_fit, df_fit$llm_short)

llm_models_list <- lapply(names(llm_splits), function(nm) {
  d <- llm_splits[[nm]]
  # Guard: need at least 2 points and variation in eco to estimate slope
  if (nrow(d) >= 2 && is.finite(sd(d$eco)) && sd(d$eco) > 0) {
    m <- lm(obj_fit ~ eco, data = d)
    s <- summary(m)
    data.frame(
      llm_short = nm,
      n = nrow(d),
      intercept = unname(coef(m)[1]),
      slope = unname(coef(m)[2]),
      r_squared = s$r.squared,
      adj_r_squared = s$adj.r.squared,
      p_value = s$coefficients[2, 4],
      stringsAsFactors = FALSE
    )
  } else {
    data.frame(
      llm_short = nm,
      n = nrow(d),
      intercept = NA_real_,
      slope = NA_real_,
      r_squared = NA_real_,
      adj_r_squared = NA_real_,
      p_value = NA_real_,
      stringsAsFactors = FALSE
    )
  }
})

llm_models <- do.call(rbind, llm_models_list)

# --- Persist summaries ---
results_dir <- "Results/ecology_analysis"
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

# write.csv(
#   llm_models %>% arrange(llm_short),
#   file.path(results_dir, "modeling_by_llm_LINEAR_capped_fits.csv"),
#   row.names = FALSE
# )

overall_out <- data.frame(
  n = nrow(df_fit),
  intercept = unname(coef(fit_overall)[1]),
  slope = unname(coef(fit_overall)[2]),
  r_squared = overall_sum$r.squared,
  adj_r_squared = overall_sum$adj.r.squared,
  p_value_slope = overall_sum$coefficients[2, "Pr(>|t|)"],
  stringsAsFactors = FALSE
)
# write.csv(overall_out, file.path(results_dir, "modeling_overall_LINEAR_capped_fits.csv"), row.names = FALSE)

# --- Unified line predictions (clamped to [0, 1]) ---
grid_eco <- seq(0, 1, length.out = 200)
overall_line_df <- data.frame(
  eco       = grid_eco,
  obj       = pmin(pmax(as.numeric(predict(fit_overall, newdata = data.frame(eco = grid_eco))), 0), cap_val),
  llm_short = "Overall"
)

# --- Vectorized formatter to avoid 'condition has length > 1' errors ---
fmt_num_vec <- function(x) {
  out <- rep("NA", length(x))
  ok <- !is.na(x)
  if (any(ok)) out[ok] <- formatC(x[ok], digits = 3, format = "fg", flag = "#")
  out
}

# --- Annotation text (bottom-left): Unified + one line per LLM ---
unified_line <- sprintf(
  "Unified: R²=%s; p=%s; n=%d",
  fmt_num_vec(overall_out$r_squared),
  fmt_num_vec(overall_out$p_value_slope),
  overall_out$n
)

llm_models_arr <- llm_models %>% arrange(llm_short)
llm_lines <- sprintf(
  "%s: R²=%s; p=%s; n=%d",
  llm_models_arr$llm_short,
  fmt_num_vec(llm_models_arr$r_squared),
  fmt_num_vec(llm_models_arr$p_value),
  llm_models_arr$n
)

ann_text <- paste(c(unified_line, llm_lines), collapse = "\n")

# Position (bottom-left white space). Bump ann_y if text block is long.
y_max <- cap_val + tri_offset + 0.02
ann_x <- 0.02
ann_y <- 0.06 * y_max

# --- Plot ---
fig_dir <- "Figures"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

p_scatter_linear <- ggplot() +
  # points at/under clamp
  geom_point(
    data = df_plot %>% filter(!is_clamped),
    aes(x = eco, y = obj_plot, color = llm_short),
    alpha = 0.8, size = 1.9
  ) +
  # points above clamp (drawn at clamp line)
  geom_point(
    data = df_plot %>% filter(is_clamped),
    aes(x = eco, y = obj_plot, color = llm_short),
    alpha = 0.95, size = 1.9, show.legend = TRUE
  ) +
  # arrowhead triangles above clamp
  {
    if (any(df_plot$is_clamped)) {
      geom_point(
        data = df_plot %>% filter(is_clamped),
        aes(x = eco, y = tri_y - 0.035, color = llm_short, fill = llm_short),
        shape = 24, size = 1.6, stroke = 0.45, show.legend = FALSE
      )
    } else {
      NULL
    }
  } +
  # per‑LLM LM lines (fits on capped y)
  geom_smooth(
    data = df_fit,
    aes(x = eco, y = obj_fit, color = llm_short),
    method = "lm", se = FALSE, linewidth = 1.0
  ) +
  # unified overall LM line in black (also a legend entry)
  geom_path(
    data = overall_line_df,
    aes(x = eco, y = obj, color = llm_short),
    linewidth = 1.2, inherit.aes = FALSE, show.legend = TRUE
  ) +
  # clamp reference
  geom_hline(yintercept = cap_val, linetype = "dotted", color = "grey40") +
  # stats annotation
  annotate("text",
    x = ann_x, y = ann_y,
    label = ann_text,
    hjust = 0, vjust = 0,
    size = 3.3, lineheight = 1.05
  ) +
  # colors (add 'Overall' = black to palette)
  scale_color_manual(values = c(llm_cols, "Overall" = "black"), name = "LLM") +
  scale_fill_manual(values = llm_cols, guide = "none") +
  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, by = 0.1),
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  scale_y_continuous(
    limits = c(0, y_max),
    breaks = c(seq(0, cap_val, by = 0.1), cap_val + tri_offset),
    labels = function(y) ifelse(abs(y - (cap_val + tri_offset)) < 1e-8, "↑", scales::number_format()(y)),
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  labs(
    x = "Ecological score",
    y = "Objective value",
    # title = "Ecological score vs Objective — simple LM fits (y capped at 1.0 for fitting)",
    # subtitle = "All points shown; outliers with arrowheads. Black line = unified LM. Bottom-left: R², adj‑R², p, n for Unified & each LLM."
  ) +
  theme_classic()

# Save
ggsave(file.path(fig_dir, "eco_vs_objective.png"),
  p_scatter_linear,
  width = 9.5, height = 6, dpi = 300
)
ggsave(file.path(fig_dir, "eco_vs_objective.svg"),
  p_scatter_linear,
  width = 9.5, height = 6
)

cat(sprintf(
  "Scatter saved with unified & per‑LLM stats. Fit rows: %d; plot rows: %d; summaries in %s\n",
  nrow(df_fit), nrow(df_plot), results_dir
))

#####################################
# ==== Write analysis_summary_ALL.txt with per‑LLM stats + best/worst by objective AND ecological score ====
suppressPackageStartupMessages({
  library(dplyr)
})


results_dir <- "Results"
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

# Safety: short_llm() should exist already; define if missing.
if (!exists("short_llm")) {
  short_llm <- function(x) ifelse(grepl("/", x), sub(".*/", "", x), x)
}

# Build a clean table for summaries (keep objective_value > 0; coerce numerics)
summary_df <- ecology_all %>%
  mutate(
    objective_value = suppressWarnings(as.numeric(objective_value)),
    total_score = suppressWarnings(as.numeric(total_score)),
    normalized_total = suppressWarnings(as.numeric(normalized_total)),
    final_score = suppressWarnings(as.numeric(final_score)),
    # component / mechanism columns (coerce to numeric safely if present)
    nutrient_equation_uptake = suppressWarnings(as.numeric(nutrient_equation_uptake)),
    nutrient_equation_recycling = suppressWarnings(as.numeric(nutrient_equation_recycling)),
    nutrient_equation_mixing = suppressWarnings(as.numeric(nutrient_equation_mixing)),
    phytoplankton_equation_growth = suppressWarnings(as.numeric(phytoplankton_equation_growth)),
    phytoplankton_equation_grazing_loss = suppressWarnings(as.numeric(phytoplankton_equation_grazing_loss)),
    phytoplankton_equation_mortality = suppressWarnings(as.numeric(phytoplankton_equation_mortality)),
    phytoplankton_equation_mixing = suppressWarnings(as.numeric(phytoplankton_equation_mixing)),
    zooplankton_equation_growth = suppressWarnings(as.numeric(zooplankton_equation_growth)),
    zooplankton_equation_mortality = suppressWarnings(as.numeric(zooplankton_equation_mortality)),
    llm = as.character(llm),
    llm_short = short_llm(llm),
    Population = as.character(Population),
    individual = as.character(individual)
  ) %>%
  filter(is.finite(objective_value), objective_value > 0)

# Helper for compact numeric formatting (scalar-safe)
fmt <- function(x) {
  if (length(x) == 0 || is.na(x)) "NA" else formatC(x, digits = 3, format = "fg", flag = "#")
}

# Detect which mechanism columns are present in this dataset
mechanism_cols <- c(
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
mechanism_cols <- mechanism_cols[mechanism_cols %in% colnames(summary_df)]

# Threshold to consider a mechanism "embedded" 
mech_high_thr <- 0.75

sink(file.path(results_dir, "NPZ_ecology.txt"))

cat("Analysis Summary (ALL NPZ populations)\n")
cat("================\n\n")

# Overall counts
cat("Number of individuals analyzed (objective > 0): ", nrow(summary_df), "\n", sep = "")
cat("Number of NPZ populations: ", length(unique(summary_df$Population)), "\n", sep = "")
cat("Number of LLMs: ", length(unique(summary_df$llm_short)), "\n\n", sep = "")

# Overall summary table
cat("Overall Summary Statistics (objective > 0):\n")
print(summary(summary_df[, c("objective_value", "normalized_total", "final_score", "total_score")], digits = 3))
cat("\n")

# Overall correlations (examples)
if (nrow(summary_df) >= 3) {
  if (any(is.finite(summary_df$total_score))) {
    total_cor <- suppressWarnings(cor.test(summary_df$total_score, summary_df$objective_value))
    cat(sprintf(
      "Correlation (Total Score vs Objective): r = %.3f (p = %.3e)\n",
      total_cor$estimate, total_cor$p.value
    ))
  }
  if (any(is.finite(summary_df$normalized_total))) {
    eco_cor <- suppressWarnings(cor.test(summary_df$normalized_total, summary_df$objective_value))
    cat(sprintf(
      "Correlation (Normalized Ecological Score vs Objective): r = %.3f (p = %.3e)\n",
      eco_cor$estimate, eco_cor$p.value
    ))
  }
  cat("\n")
} else {
  cat("Not enough complete cases to calculate overall correlations.\n\n")
}

# ---- Per‑LLM summaries ----
cat("Per‑LLM Summary Statistics (objective > 0):\n")
cat("------------------------------------------\n\n")

llm_levels <- sort(unique(summary_df$llm_short))

# Collect per‑mechanism CSV rows as we print to the text file
per_mech_rows <- list()

for (llm_name in llm_levels) {
  d <- summary_df %>% filter(llm_short == llm_name)

  cat(sprintf("LLM: %s  (n = %d)\n", llm_name, nrow(d)))

  # Objective stats
  obj_mean <- mean(d$objective_value, na.rm = TRUE)
  obj_sd <- sd(d$objective_value, na.rm = TRUE)
  obj_med <- median(d$objective_value, na.rm = TRUE)
  obj_min <- min(d$objective_value, na.rm = TRUE)
  obj_max <- max(d$objective_value, na.rm = TRUE)
  cat(sprintf(
    "  Objective  -> mean=%s, sd=%s, median=%s, min=%s, max=%s\n",
    fmt(obj_mean), fmt(obj_sd), fmt(obj_med), fmt(obj_min), fmt(obj_max)
  ))

  # Ecological score stats (normalized_total)
  eco_has <- any(is.finite(d$normalized_total))
  if (eco_has) {
    eco_mean <- mean(d$normalized_total, na.rm = TRUE)
    eco_sd <- sd(d$normalized_total, na.rm = TRUE)
    eco_med <- median(d$normalized_total, na.rm = TRUE)
    eco_min <- min(d$normalized_total, na.rm = TRUE)
    eco_max <- max(d$normalized_total, na.rm = TRUE)
    cat(sprintf(
      "  Ecological (normalized_total) -> mean=%s, sd=%s, median=%s, min=%s, max=%s\n",
      fmt(eco_mean), fmt(eco_sd), fmt(eco_med), fmt(eco_min), fmt(eco_max)
    ))
  } else {
    cat("  Ecological (normalized_total) -> no finite values\n")
  }

  # Total score (from characteristic aggregation), if present
  tot_has <- any(is.finite(d$total_score))
  if (tot_has) {
    tot_mean <- mean(d$total_score, na.rm = TRUE)
    tot_sd <- sd(d$total_score, na.rm = TRUE)
    tot_med <- median(d$total_score, na.rm = TRUE)
    tot_min <- min(d$total_score, na.rm = TRUE)
    tot_max <- max(d$total_score, na.rm = TRUE)
    cat(sprintf(
      "  Total Score -> mean=%s, sd=%s, median=%s, min=%s, max=%s\n",
      fmt(tot_mean), fmt(tot_sd), fmt(tot_med), fmt(tot_min), fmt(tot_max)
    ))
  } else {
    cat("  Total Score -> no finite values\n")
  }

  # Final score, if present
  fin_has <- any(is.finite(d$final_score))
  if (fin_has) {
    fin_mean <- mean(d$final_score, na.rm = TRUE)
    fin_sd <- sd(d$final_score, na.rm = TRUE)
    fin_med <- median(d$final_score, na.rm = TRUE)
    fin_min <- min(d$final_score, na.rm = TRUE)
    fin_max <- max(d$final_score, na.rm = TRUE)
    cat(sprintf(
      "  Final Score -> mean=%s, sd=%s, median=%s, min=%s, max=%s\n",
      fmt(fin_mean), fmt(fin_sd), fmt(fin_med), fmt(fin_min), fmt(fin_max)
    ))
  } else {
    cat("  Final Score -> no finite values\n")
  }

  # Best/worst individuals within this LLM (by objective_value; lower is better)
  best_row <- d %>% slice_min(order_by = objective_value, n = 1, with_ties = FALSE)
  worst_row <- d %>% slice_max(order_by = objective_value, n = 1, with_ties = FALSE)

  if (nrow(best_row) == 1) {
    cat(sprintf(
      "  Best by objective (lowest): %s  [Pop: %s]  obj=%s; eco=%s; total=%s; final=%s\n",
      best_row$individual, best_row$Population,
      fmt(best_row$objective_value),
      fmt(best_row$normalized_total),
      fmt(best_row$total_score),
      fmt(best_row$final_score)
    ))
  }
  if (nrow(worst_row) == 1) {
    cat(sprintf(
      "  Worst by objective (highest): %s  [Pop: %s]  obj=%s; eco=%s; total=%s; final=%s\n",
      worst_row$individual, worst_row$Population,
      fmt(worst_row$objective_value),
      fmt(worst_row$normalized_total),
      fmt(worst_row$total_score),
      fmt(worst_row$final_score)
    ))
  }

  # Best/worst individuals within this LLM (by ecological score; HIGHER is better)
  if (eco_has) {
    d_eco <- d %>% filter(is.finite(normalized_total))
    if (nrow(d_eco) >= 1) {
      best_eco <- d_eco %>% slice_max(order_by = normalized_total, n = 1, with_ties = FALSE)
      worst_eco <- d_eco %>% slice_min(order_by = normalized_total, n = 1, with_ties = FALSE)

      cat(sprintf(
        "  Best by ecological score (highest eco): %s  [Pop: %s]  eco=%s; obj=%s; total=%s; final=%s\n",
        best_eco$individual, best_eco$Population,
        fmt(best_eco$normalized_total),
        fmt(best_eco$objective_value),
        fmt(best_eco$total_score),
        fmt(best_eco$final_score)
      ))

      cat(sprintf(
        "  Worst by ecological score (lowest eco): %s  [Pop: %s]  eco=%s; obj=%s; total=%s; final=%s\n",
        worst_eco$individual, worst_eco$Population,
        fmt(worst_eco$normalized_total),
        fmt(worst_eco$objective_value),
        fmt(worst_eco$total_score),
        fmt(worst_eco$final_score)
      ))
    } else {
      cat("  Best/Worst by ecological score: no finite values\n")
    }
  } else {
    cat("  Best/Worst by ecological score: no finite values\n")
  }

  # Optional: correlation (eco vs objective) within this LLM
  if (eco_has && nrow(d) >= 3) {
    llm_cor <- suppressWarnings(cor.test(d$normalized_total, d$objective_value))
    cat(sprintf(
      "  Corr (eco vs objective): r=%s (p=%s)\n",
      fmt(llm_cor$estimate), fmt(llm_cor$p.value)
    ))
  }

  # ---- Per‑mechanism summaries for this LLM ----
  if (length(mechanism_cols) > 0) {
    cat(sprintf("  Per‑mechanism ecological characteristics (threshold for 'embedded' = %.2f):\n", mech_high_thr))

    for (m in mechanism_cols) {
      v <- suppressWarnings(as.numeric(d[[m]]))
      finite <- is.finite(v)
      n_mech <- sum(finite)

      if (n_mech == 0) {
        cat(sprintf("    %s -> no finite values\n", m))
        # accumulate CSV row with no finite values
        per_mech_rows[[length(per_mech_rows) + 1]] <- data.frame(
          llm_short = llm_name, mechanism = m, n = 0,
          mean = NA_real_, sd = NA_real_, median = NA_real_,
          min = NA_real_, max = NA_real_,
          n_high = 0, high_threshold = mech_high_thr,
          best_individual = NA_character_, best_score = NA_real_,
          best_objective = NA_real_, worst_individual = NA_character_,
          worst_score = NA_real_, worst_objective = NA_real_,
          stringsAsFactors = FALSE
        )
      } else {
        m_mean <- mean(v[finite], na.rm = TRUE)
        m_sd <- sd(v[finite], na.rm = TRUE)
        m_med <- median(v[finite], na.rm = TRUE)
        m_min <- min(v[finite], na.rm = TRUE)
        m_max <- max(v[finite], na.rm = TRUE)
        n_high <- sum(v[finite] >= mech_high_thr)

        cat(sprintf(
          "    %s -> n=%d; mean=%s, sd=%s, median=%s, min=%s, max=%s; high(≥%.2f)=%d\n",
          m, n_mech, fmt(m_mean), fmt(m_sd), fmt(m_med), fmt(m_min), fmt(m_max), mech_high_thr, n_high
        ))

        # best/worst by mechanism score (HIGHER is assumed better for mechanism score)
        d_mech <- d %>% filter(is.finite(.data[[m]]))
        best_mech <- d_mech %>% slice_max(order_by = .data[[m]], n = 1, with_ties = FALSE)
        worst_mech <- d_mech %>% slice_min(order_by = .data[[m]], n = 1, with_ties = FALSE)

        if (nrow(best_mech) == 1) {
          cat(sprintf(
            "      Best %s: %s [Pop: %s] score=%s; obj=%s\n",
            m, best_mech$individual, best_mech$Population,
            fmt(best_mech[[m]]), fmt(best_mech$objective_value)
          ))
        }
        if (nrow(worst_mech) == 1) {
          cat(sprintf(
            "      Worst %s: %s [Pop: %s] score=%s; obj=%s\n",
            m, worst_mech$individual, worst_mech$Population,
            fmt(worst_mech[[m]]), fmt(worst_mech$objective_value)
          ))
        }

        # accumulate CSV row
        per_mech_rows[[length(per_mech_rows) + 1]] <- data.frame(
          llm_short = llm_name, mechanism = m, n = n_mech,
          mean = m_mean, sd = m_sd, median = m_med,
          min = m_min, max = m_max,
          n_high = n_high, high_threshold = mech_high_thr,
          best_individual = if (nrow(best_mech) == 1) best_mech$individual else NA_character_,
          best_score = if (nrow(best_mech) == 1) suppressWarnings(as.numeric(best_mech[[m]])) else NA_real_,
          best_objective = if (nrow(best_mech) == 1) best_mech$objective_value else NA_real_,
          worst_individual = if (nrow(worst_mech) == 1) worst_mech$individual else NA_character_,
          worst_score = if (nrow(worst_mech) == 1) suppressWarnings(as.numeric(worst_mech[[m]])) else NA_real_,
          worst_objective = if (nrow(worst_mech) == 1) worst_mech$objective_value else NA_real_,
          stringsAsFactors = FALSE
        )
      }
    }
  } else {
    cat("  Per‑mechanism ecological characteristics: no mechanism columns detected\n")
  }

  cat("\n")
}

sink()

# # ---- Save per‑mechanism summary CSV across all LLMs ----
# if (length(per_mech_rows) > 0) {
#   per_mech_df <- do.call(rbind, per_mech_rows)
#   write.csv(per_mech_df, file.path(results_dir, "per_mechanism_summary_by_llm.csv"), row.names = FALSE)
# } else {
#   # still create an empty file with header for consistency
#   per_mech_df <- data.frame(
#     llm_short = character(), mechanism = character(), n = integer(),
#     mean = double(), sd = double(), median = double(),
#     min = double(), max = double(),
#     n_high = integer(), high_threshold = double(),
#     best_individual = character(), best_score = double(), best_objective = double(),
#     worst_individual = character(), worst_score = double(), worst_objective = double(),
#     stringsAsFactors = FALSE
#   )
#   write.csv(per_mech_df, file.path(results_dir, "per_mechanism_summary_by_llm.csv"), row.names = FALSE)
# }

message("analysis_summary_ALL.txt written with per‑LLM and per‑mechanism statistics; CSV saved: Results/ecology_analysis/per_mechanism_summary_by_llm.csv")
# ==== END analysis_summary_ALL.txt extension ====

# ==== Categorical distributions with fixed bar width per category × LLM ====
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

# Preconditions / helpers
stopifnot(exists("ecology_all"), is.data.frame(ecology_all), nrow(ecology_all) > 0)
if (!exists("short_llm")) {
  short_llm <- function(x) ifelse(grepl("/", x), sub(".*/", "", x), x)
}

# Mechanism columns (reuse if already defined)
if (!exists("mechanism_cols")) {
  mechanism_cols <- c(
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
  mechanism_cols <- mechanism_cols[mechanism_cols %in% colnames(ecology_all)]
}
if (length(mechanism_cols) == 0) {
  stop("No ecological mechanism columns found in ecology_all.")
}

# Categorical mapping: 0..3
score_levels <- c(0L, 1L, 2L, 3L)
score_labels <- c(
  "absent/incorrect",
  "present",
  "alternate",
  "truth match"
)

# --- Identify best performing individual per LLM (by objective value) ---
best_per_llm_cat <- ecology_all %>%
  mutate(
    llm = as.character(llm),
    llm_short = short_llm(llm),
    objective_value = suppressWarnings(as.numeric(objective_value))
  ) %>%
  filter(is.finite(objective_value)) %>%
  group_by(llm_short) %>%
  slice_min(order_by = objective_value, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  select(llm_short, individual, Population, objective_value)

cat("\nBest performers by LLM for categorical plot asterisks:\n")
print(best_per_llm_cat)
cat("\n")

# Extract ecological scores for best performers in long format
best_scores_long <- ecology_all %>%
  mutate(
    llm = as.character(llm),
    llm_short = short_llm(llm)
  ) %>%
  inner_join(best_per_llm_cat, by = c("llm_short", "individual")) %>%
  select(llm_short, individual, all_of(mechanism_cols)) %>%
  pivot_longer(
    cols = all_of(mechanism_cols),
    names_to = "mechanism",
    values_to = "score"
  ) %>%
  mutate(
    score_num = suppressWarnings(as.integer(as.character(score))),
    score_cat = factor(score_num, levels = score_levels, labels = score_labels, ordered = TRUE)
  ) %>%
  filter(!is.na(score_cat))

# Prepare long data for all individuals
df_mech_long <- ecology_all %>%
  mutate(
    llm = as.character(llm),
    llm_short = short_llm(llm)
  ) %>%
  select(Population, individual, llm_short, all_of(mechanism_cols)) %>%
  pivot_longer(
    cols = all_of(mechanism_cols),
    names_to = "mechanism",
    values_to = "score"
  ) %>%
  mutate(
    score_num = suppressWarnings(as.integer(as.character(score))),
    score_cat = factor(score_num, levels = score_levels, labels = score_labels, ordered = TRUE)
  ) %>%
  filter(!is.na(score_cat))

if (nrow(df_mech_long) == 0) stop("No categorical mechanism scores available to plot.")

# All LLM levels we want to keep (even if absent in a particular bin)
llm_levels <- sort(unique(df_mech_long$llm_short))

# Count & COMPLETE the grid so every (mechanism × score_cat × LLM) exists (n=0 if missing)
df_counts <- df_mech_long %>%
  count(mechanism, llm_short, score_cat, name = "n") %>%
  tidyr::complete(
    mechanism,
    score_cat = factor(score_labels, levels = score_labels, ordered = TRUE),
    llm_short = factor(llm_levels, levels = llm_levels),
    fill = list(n = 0)
  ) %>%
  arrange(mechanism, score_cat, llm_short)

# Toggle raw counts vs proportions per (mechanism × LLM)
normalize_counts <- FALSE # set TRUE to show proportions within each LLM per mechanism
if (normalize_counts) {
  df_counts <- df_counts %>%
    group_by(mechanism, llm_short) %>%
    mutate(n_total = sum(n), value = ifelse(n_total > 0, n / n_total, 0)) %>%
    ungroup()
} else {
  df_counts <- df_counts %>%
    mutate(value = n)
}

# Color palette for LLMs
llm_cols <- setNames(scales::hue_pal()(length(llm_levels)), llm_levels)

# Use position_dodge2 with preserve = "single" to keep each bar's width fixed
pos_fixed <- position_dodge2(width = 0.8, preserve = "single", padding = 0.05)

# Prepare asterisk data: create a complete grid with placeholders for all LLMs
# First, mark which combinations should show asterisks (best performers)
best_markers <- best_scores_long %>%
  mutate(show_asterisk = TRUE) %>%
  select(mechanism, llm_short, score_cat, show_asterisk)

# Create complete grid for asterisks (all mechanism x score_cat x llm_short combinations)
asterisk_data <- df_counts %>%
  left_join(best_markers, by = c("mechanism", "llm_short", "score_cat")) %>%
  mutate(
    show_asterisk = ifelse(is.na(show_asterisk), FALSE, show_asterisk),
    # Only show asterisk label where marked AND there's a bar (value > 0)
    label = ifelse(show_asterisk & value > 0, "*", ""),
    # Position asterisk slightly above the bar (or at 0 for empty placeholders)
    y_pos = ifelse(value > 0, value * 1.05, 0)
  )

cat("\nAsterisk data summary (showing asterisks):\n")
print(asterisk_data %>% filter(show_asterisk) %>% select(mechanism, llm_short, score_cat, value, label, y_pos))
cat("\n")

# Create a dummy layer for legend entry (asterisk)
dummy_asterisk <- data.frame(
  score_cat = factor(score_labels[1], levels = score_labels),
  value = NA_real_,
  llm_short = "Best Performer",
  mechanism = mechanism_cols[1]
)

p_mech_dist_cat_fixed <- ggplot(
  df_counts,
  aes(x = score_cat, y = value, fill = llm_short)
) +
  geom_col(position = pos_fixed, width = 0.7, alpha = 0.95) +
  # Add asterisks for best performers - use fill aesthetic to match bar positioning
  {
    if (nrow(asterisk_data) > 0) {
      geom_text(
        data = asterisk_data,
        aes(x = score_cat, y = y_pos, label = label, fill = llm_short),
        position = pos_fixed,
        size = 6,
        # fontface = "bold",
        color = "black",
        show.legend = FALSE
      )
    } else {
      NULL
    }
  } +
  # Add dummy point for legend (invisible, but creates legend entry)
  geom_point(
    data = dummy_asterisk,
    aes(x = score_cat, y = value, shape = llm_short),
    size = 0,
    alpha = 0,
    show.legend = TRUE
  ) +
  facet_wrap(~mechanism, scales = "free_y", ncol = 3,
           labeller = labeller(mechanism = ~stringr::str_to_sentence(gsub("_", " ", gsub("equation", "-", .))))) +
  scale_fill_manual(values = llm_cols, name = "LLM", drop = FALSE) +
  scale_shape_manual(
    values = c("Best Performer" = 8),
    name = "",
    labels = c("Best Performer" = "*  Best Performer")
  ) +
  scale_x_discrete(drop = FALSE) +
  labs(
    x = "Raw score category",
    y = if (normalize_counts) "Proportion within LLM" else "Count of individuals"
  ) +
  theme_classic() +
  theme(
    legend.position = "bottom",
    legend.justification = "center", # optional: center the box at the bottom
    legend.background = element_rect(fill = "white", color = "white"),
    legend.box.background = element_rect(fill = "white", color = NA),
    legend.direction = "horizontal", # legend keys flow horizontally
    legend.box = "horizontal", # multiple legends placed side-by-side
    strip.text = element_text(size = 10),
    axis.text.x = element_text(angle = 30, hjust = 1),
    strip.background = element_blank(),
    legend.spacing.x = unit(0.3, "cm"), # optional: spacing between keys
    legend.key.width = unit(1.2, "lines") # optional: widen keys if needed
  ) +
    guides(
      fill  = guide_legend(order = 1, title = "LLM", nrow = 1, byrow = TRUE),
      shape = guide_legend(order = 2, title = NULL, nrow = 1, byrow = TRUE)
    )

# Save
fig_dir <- "Figures"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
ggsave(file.path(fig_dir, "ecology_mechanism.png"),
  p_mech_dist_cat_fixed,
  width = 11, height = 7.5, dpi = 300
)
ggsave(file.path(fig_dir, "ecology_mechanism.svg"),
  p_mech_dist_cat_fixed,
  width = 11, height = 7.5
)

message("Saved fixed‑width categorical distribution to Figures/ecology_mechanism_distribution_by_llm_categorical_FIXEDWIDTH.{png,svg}")
# ==== END fixed bar width categorical distributions ====
