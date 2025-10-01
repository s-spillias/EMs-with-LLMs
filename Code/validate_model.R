# Validation and compilation runner for the COTS TMB model

suppressPackageStartupMessages({
  library(TMB)
  library(jsonlite)
})

# Paths
model_dir <- "POPULATIONS/POPULATION_0004/INDIVIDUAL_3AWGFWMY"
model_cpp <- file.path(model_dir, "model.cpp")
params_json <- file.path(model_dir, "parameters.json")
resp_csv <- "Data/timeseries_data_COTS_response.csv"
force_csv <- "Data/timeseries_data_COTS_forcing.csv"

# Helper to simplify column names like "fast_dat (description)" -> "fast_dat"
simplify_names <- function(nms) gsub("\\s*\\(.*\\)$", "", nms)

# Load data
resp <- read.csv(resp_csv, stringsAsFactors = FALSE, check.names = FALSE)
force <- read.csv(force_csv, stringsAsFactors = FALSE, check.names = FALSE)

names(resp) <- simplify_names(names(resp))
names(force) <- simplify_names(names(force))

# Rename to expected TMB names if needed (guarded renaming by position)
# Ensure column order: Year, cots_dat, fast_dat, slow_dat for resp
if (!all(c("Year", "cots_dat", "fast_dat", "slow_dat") %in% names(resp))) {
  # fall back to positional renaming if mixed headers
  if (ncol(resp) >= 4) {
    names(resp)[1:4] <- c("Year", "cots_dat", "fast_dat", "slow_dat")
  }
}
# Ensure column order: Year, sst_dat, cotsimm_dat for force
if (!all(c("Year", "sst_dat", "cotsimm_dat") %in% names(force))) {
  if (ncol(force) >= 3) {
    names(force)[1:3] <- c("Year", "sst_dat", "cotsimm_dat")
  }
}

# Merge and align years
dat <- merge(resp[, c("Year","cots_dat","fast_dat","slow_dat")],
             force[, c("Year","sst_dat","cotsimm_dat")],
             by = "Year", all = FALSE)

# Remove any rows with missing values
dat <- dat[complete.cases(dat), ]

# Build TMB data list
Data <- list(
  Year = as.numeric(dat$Year),
  cots_dat = as.numeric(dat$cots_dat),
  fast_dat = as.numeric(dat$fast_dat),
  slow_dat = as.numeric(dat$slow_dat),
  sst_dat = as.numeric(dat$sst_dat),
  cotsimm_dat = as.numeric(dat$cotsimm_dat)
)

# Load parameters from JSON (robust to df/list; filter only model PARAMETERs)
pj <- fromJSON(params_json, simplifyVector = TRUE)

params_tbl <- pj$parameters
if (is.data.frame(params_tbl)) {
  params_param <- params_tbl[params_tbl$import_type == "PARAMETER", , drop = FALSE]
  param_names <- as.character(params_param$parameter)
  param_values <- as.numeric(params_param$value)
} else if (is.list(params_tbl)) {
  params_param <- Filter(function(x) identical(x$import_type, "PARAMETER"), params_tbl)
  param_names <- vapply(params_param, function(x) x$parameter, character(1))
  param_values <- as.numeric(vapply(params_param, function(x) x$value, numeric(1)))
} else {
  stop("Unrecognized structure for pj$parameters")
}
Parameters <- as.list(stats::setNames(param_values, param_names))

# Compile and load TMB model
compile(model_cpp)
dyn.load(dynlib(model_cpp))

# Build objective function
obj <- MakeADFun(data = Data, parameters = Parameters, DLL = basename(tools::file_path_sans_ext(model_cpp)), silent = TRUE)

# Evaluate once to ensure it runs
nll <- obj$fn()
gr <- obj$gr()

cat(sprintf("Initial NLL: %.6f\n", nll))
cat(sprintf("Gradient norm: %.6f\n", sqrt(sum(gr^2))))
