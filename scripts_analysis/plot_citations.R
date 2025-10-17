# install.packages(c("jsonlite", "dplyr", "tidyr", "ggplot2"))  # if needed

library(jsonlite)
library(dplyr)
library(tidyr)
library(ggplot2)

# ---- Helper: safely pull one numeric field from an individual, trying multiple names ----
extract_numeric_field <- function(ind, candidates) {
    for (nm in candidates) {
        val <- ind[[nm]]
        if (!is.null(val)) {
            num <- suppressWarnings(as.numeric(val))
            if (!is.na(num)) {
                return(num)
            }
        }
    }
    return(NA_real_)
}

# ---- Flatten populations to one row per individual ----
flatten_populations <- function(pop_list, model_type_label, objective_candidates) {
    if (is.null(pop_list) || length(pop_list) == 0) {
        return(NULL)
    }

    rows <- list()
    k <- 1
    for (p in pop_list) {
        inds <- p$individuals
        if (is.null(inds) || length(inds) == 0) next
        for (ind in inds) {
            rows[[k]] <- list(
                model_type                 = model_type_label,
                parameters_metadata_exists = isTRUE(ind$parameters_metadata_exists),
                total_params               = suppressWarnings(as.numeric(ind$total_params)),
                citation_percentage        = suppressWarnings(as.numeric(ind$citation_percentage)),
                semantic_percentage        = suppressWarnings(as.numeric(ind$semantic_percentage)),
                docstore_percentage        = suppressWarnings(as.numeric(ind$docstore_percentage)),
                is_best_performer          = isTRUE(ind$is_best_performer),
                objective_value            = extract_numeric_field(ind, objective_candidates)
            )
            k <- k + 1
        }
    }

    if (length(rows) == 0) {
        return(NULL)
    }
    df <- bind_rows(rows) %>%
        filter(parameters_metadata_exists, !is.na(total_params), total_params > 0) %>%
        mutate(no_citations_pct = 100 - citation_percentage)

    df
}

# ---- Pick the best (lowest objective) row per model_type ----
get_best_rows_by_objective <- function(df, objective_col, summary_list) {
    best_rows <- df %>%
        filter(!is.na(.data[[objective_col]])) %>%
        group_by(model_type) %>%
        slice_min(order_by = .data[[objective_col]], with_ties = FALSE) %>%
        ungroup()

    # Fallback to summary for any missing model_type
    present <- unique(best_rows$model_type)
    need_cots <- !("COTS" %in% present)
    need_npz <- !("NPZ" %in% present)

    add_rows <- list()
    if (need_cots && !is.null(summary_list$cots_best_performers_summary)) {
        s <- summary_list$cots_best_performers_summary
        add_rows[[length(add_rows) + 1]] <- data.frame(
            model_type = "COTS",
            semantic_percentage = suppressWarnings(as.numeric(s$semantic_percentage)),
            docstore_percentage = suppressWarnings(as.numeric(s$docstore_percentage)),
            objective_fill = NA_real_
        )
    }
    if (need_npz && !is.null(summary_list$npz_best_performers_summary)) {
        s <- summary_list$npz_best_performers_summary
        add_rows[[length(add_rows) + 1]] <- data.frame(
            model_type = "NPZ",
            semantic_percentage = suppressWarnings(as.numeric(s$semantic_percentage)),
            docstore_percentage = suppressWarnings(as.numeric(s$docstore_percentage)),
            objective_fill = NA_real_
        )
    }

    if (length(add_rows) > 0) {
        # Normalise columns for bind_rows
        best_rows <- best_rows %>%
            select(model_type, semantic_percentage, docstore_percentage, !!objective_col)
        names(best_rows)[names(best_rows) == objective_col] <- "objective_fill"

        best_rows <- bind_rows(
            best_rows,
            bind_rows(add_rows)
        )
    } else {
        # Rename to a common column for debug printing
        best_rows <- best_rows %>%
            mutate(objective_fill = .data[[objective_col]]) %>%
            select(model_type, semantic_percentage, docstore_percentage, objective_fill)
    }

    # Debug messages
    if (nrow(best_rows) > 0) {
        apply(best_rows, 1, function(r) {
            mt <- r[["model_type"]]
            obj <- as.numeric(r[["objective_fill"]])
            msg <- ifelse(is.na(obj), "from summary", sprintf("%0.6f", obj))
            sem <- as.numeric(r[["semantic_percentage"]])
            doc <- as.numeric(r[["docstore_percentage"]])
            message(sprintf(
                "[DEBUG] Best by objective — %s: objective=%s | semantic=%.3f | local=%.3f",
                mt, msg, sem, doc
            ))
        })
    }

    best_rows
}

# ---- Main: single-panel dodged boxplot with asterisk overlays ----
create_dodged_citations_plot_from_json <- function(
    json_path = "Results/citations_analysis.json",
    output_path = "Figures/citations_boxplot.png",
    objective_field = "objective_value", # set to your field name (e.g., "loss")
    objective_candidates = c("objective_value", "loss", "fitness", "score", "objective") # auto-detect list
    ) {
    if (!file.exists(json_path)) {
        stop(sprintf("JSON file not found: %s", json_path))
    }

    j <- fromJSON(json_path, simplifyVector = FALSE)

    # Per-individual data with objective values
    df_npz <- flatten_populations(j$npz_populations, "NPZ", objective_candidates)
    df_cots <- flatten_populations(j$cots_populations, "COTS", objective_candidates)
    df <- bind_rows(df_cots, df_npz)

    if (is.null(df) || nrow(df) == 0) {
        stop("No valid individuals found to plot (check 'individuals', metadata flags, and total_params > 0).")
    }

    # Choose which objective column to use (explicit or auto-detected)
    objective_col <- if (objective_field %in% names(df)) objective_field else "objective_value"
    df[[objective_col]] <- suppressWarnings(as.numeric(df[[objective_col]]))

    # Build long data for the two citation sources
    df_long <- df %>%
        select(model_type, semantic_percentage, docstore_percentage) %>%
        pivot_longer(
            cols = c(semantic_percentage, docstore_percentage),
            names_to = "source",
            values_to = "pct"
        ) %>%
        mutate(
            source = recode(source,
                semantic_percentage = "Semantic Scholar",
                docstore_percentage = "Local Citations"
            )
        ) %>%
        filter(!is.na(pct))

    # n per (source, model_type)
    counts <- df_long %>%
        group_by(source, model_type) %>%
        summarise(n = dplyr::n(), .groups = "drop")

    # Best rows (lowest objective) per model_type; then reshape to long for overlay points
    best_rows <- get_best_rows_by_objective(df, objective_col, j$summary)
    best_long <- best_rows %>%
        transmute(
            model_type,
            `Semantic Scholar` = suppressWarnings(as.numeric(semantic_percentage)),
            `Local Citations`  = suppressWarnings(as.numeric(docstore_percentage))
        ) %>%
        pivot_longer(
            cols = c(`Semantic Scholar`, `Local Citations`),
            names_to = "source",
            values_to = "pct"
        ) %>%
        filter(!is.na(pct))

    dodge_w <- 0.75

    p <- ggplot(df_long, aes(x = source, y = pct, fill = model_type)) +
        geom_boxplot(
            position = position_dodge(width = dodge_w),
            width = 0.6,
            alpha = 0.6,
            outlier.shape = 21,
            outlier.fill = "white",
            outlier.color = "grey50"
        ) +
        # Mean line within each dodge group
        stat_summary(
            aes(group = model_type),
            fun = mean, geom = "crossbar",
            width = 0.5, colour = "black", fatten = 0,
            position = position_dodge(width = dodge_w)
        ) +
        # Overlay: black asterisks at best (lowest objective) values per model_type & source
        geom_point(
            data = best_long,
            aes(x = source, y = pct, group = model_type),
            shape = 8, colour = "black", size = 3.5,
            position = position_dodge(width = dodge_w)
        ) +
        # n= labels at the top of each dodge group
        geom_text(
            data = counts,
            aes(x = source, y = Inf, label = paste0("n=", n), group = model_type),
            position = position_dodge(width = dodge_w),
            vjust = 1.2, size = 3.2
        ) +
        scale_fill_manual(values = c("COTS" = "#ffcccc", "NPZ" = "#ccccff")) +
        # scale_y_continuous(limits = c(0, 100), expand = expansion(mult = c(0.02, 0.08))) +
        labs(
            x = NULL,
            y = "% of Parameters",
            fill = "Model Type",
            # title = "Citation Integration by Source (Semantic Scholar vs Local)"
        ) +
        theme_minimal(base_size = 12) +
        theme(
            panel.grid.major.y = element_line(linetype = "dashed", colour = "grey70", linewidth = 0.3),
            panel.grid.minor = element_blank(),
            legend.position = "top",
            plot.title = element_text(face = "bold", size = 14)
        ) +
        theme_classic()

    # Ensure output directory exists
    out_dir <- dirname(output_path)
    if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

    ggsave(filename = output_path, plot = p, width = 9, height = 6, dpi = 300, bg = "white")
    message(sprintf("Dodged citation plot saved to: %s", output_path))
}

# ---- Optional: CLI entrypoint ----
# Rscript citations_dodged.R Results/citations_analysis.json Figures/citations_boxplot.png loss
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) {
    json_path <- args[1]
    output_path <- ifelse(length(args) >= 2, args[2], "Figures/citations_boxplot.png")
    objective_arg <- ifelse(length(args) >= 3, args[3], "objective_value")
    create_dodged_citations_plot_from_json(json_path, output_path, objective_field = objective_arg)
}
