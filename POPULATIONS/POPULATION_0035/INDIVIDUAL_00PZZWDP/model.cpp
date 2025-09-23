#include <TMB.hpp>

// 1. Data and parameter declarations using TMB macros.
// 2. Process equations:
//    Equation 1: COTS dynamics combine logistic growth with a smooth outbreak trigger and subtract natural mortality.
//    Equation 2: Fast-growing coral progresses via logistic growth minus losses due to COTS predation.
//    Equation 3: Slow-growing coral follows similar dynamics with its own growth and predation parameters.
// 3. Likelihood is computed using lognormal error distributions ensuring all observations (cots_dat, fast_dat, slow_dat) contribute.
template<class Type>
Type objective_function<Type>::operator() () {
  using namespace density;

  // DATA: Year vector and observations for each component (observed with measurement error)
  DATA_VECTOR(Year);          // Years (from the first column of the data file)
  DATA_VECTOR(cots_dat);      // Observed adult COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);      // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);      // Observed slow-growing coral cover (%)

  // PARAMETERS for COTS dynamics (log-transformed where appropriate)
  PARAMETER(log_r_cots);      // Intrinsic growth rate for COTS (year^-1); initial guess ~ log(0.5)
  PARAMETER(log_K_cots);      // Outbreak carrying capacity (individuals/m2); initial guess ~ log(100)
  PARAMETER(outbreak_threshold); // Threshold COTS density triggering outbreak (individuals/m2)
  PARAMETER(outbreak_slope);     // Slope controlling smooth transition of outbreak trigger (unitless)
  PARAMETER(log_mortality);   // Natural mortality rate for COTS (year^-1); initial guess ~ log(0.3)

  // PARAMETERS for fast-growing coral dynamics
  PARAMETER(log_r_fast);      // Intrinsic growth rate for fast-growing coral (year^-1); initial guess ~ log(0.6)
  PARAMETER(log_K_fast);      // Carrying capacity for fast-growing coral (% cover); initial guess ~ log(100)
  PARAMETER(log_alpha_fast);  // Predation rate of COTS on fast-growing coral; initial guess ~ log(0.05)

  // PARAMETERS for slow-growing coral dynamics
  PARAMETER(log_r_slow);      // Intrinsic growth rate for slow-growing coral (year^-1); initial guess ~ log(0.3)
  PARAMETER(log_K_slow);      // Carrying capacity for slow-growing coral (% cover); initial guess ~ log(100)
  PARAMETER(log_alpha_slow);  // Predation rate of COTS on slow-growing coral; initial guess ~ log(0.03)

  // PARAMETERS for observation error (log-transformed standard deviations)
  PARAMETER(log_sd_cots);     // Log error standard deviation for COTS observations; initial guess ~ log(0.1)
  PARAMETER(log_sd_fast);     // Log error standard deviation for fast coral observations; initial guess ~ log(0.1)
  PARAMETER(log_sd_slow);     // Log error standard deviation for slow coral observations; initial guess ~ log(0.1)

  // Transform log-parameters back to their natural scales
  Type r_cots = exp(log_r_cots);       // Intrinsic growth rate for COTS (year^-1)
  Type K_cots = exp(log_K_cots);       // Carrying capacity for COTS (individuals/m2)
  Type mortality = exp(log_mortality); // Natural mortality rate for COTS (year^-1)
  Type r_fast = exp(log_r_fast);       // Intrinsic growth rate for fast coral (year^-1)
  Type K_fast = exp(log_K_fast);       // Carrying capacity for fast coral (% cover)
  Type alpha_fast = exp(log_alpha_fast); // Predation rate on fast coral
  Type r_slow = exp(log_r_slow);       // Intrinsic growth rate for slow coral (year^-1)
  Type K_slow = exp(log_K_slow);       // Carrying capacity for slow coral (% cover)
  Type alpha_slow = exp(log_alpha_slow); // Predation rate on slow coral

  Type sd_cots = exp(log_sd_cots);     // Observation error SD for COTS
  Type sd_fast = exp(log_sd_fast);     // Observation error SD for fast coral
  Type sd_slow = exp(log_sd_slow);     // Observation error SD for slow coral

  // Number of time steps
  int n = cots_dat.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Set initial conditions from the data; these will be used as the starting values.
  cots_pred[0] = cots_dat[0];
  fast_pred[0] = fast_dat[0];
  slow_pred[0] = slow_dat[0];

  // Process model recursion: use previous time step values to predict current state.
  for(int t = 1; t < n; t++){
    // Equation 1: COTS dynamics
    //   (1) Logistic growth: r_cots * previous * (1 - previous/K_cots)
    //   (2) Outbreak trigger: smooth transition via logistic function with threshold outbreak_threshold and slope outbreak_slope.
    //   (3) Natural mortality: mortality * previous COTS
    Type outbreak_multiplier = 1 + (Type(1.0) / (Type(1.0) + exp(-outbreak_slope * (cots_pred[t-1] - outbreak_threshold))));
    cots_pred[t] = cots_pred[t-1] + r_cots * cots_pred[t-1] * (1 - cots_pred[t-1] / K_cots) * outbreak_multiplier - mortality * cots_pred[t-1];

    // Equation 2: Fast-growing coral dynamics
    //   Logistic growth minus losses from predation by COTS.
    fast_pred[t] = fast_pred[t-1] + r_fast * fast_pred[t-1] * (1 - fast_pred[t-1] / K_fast)
                   - alpha_fast * cots_pred[t-1] * fast_pred[t-1] / (fast_pred[t-1] + Type(1e-8));

    // Equation 3: Slow-growing coral dynamics
    //   Logistic growth minus losses from predation by COTS.
    slow_pred[t] = slow_pred[t-1] + r_slow * slow_pred[t-1] * (1 - slow_pred[t-1] / K_slow)
                   - alpha_slow * cots_pred[t-1] * slow_pred[t-1] / (slow_pred[t-1] + Type(1e-8));
  }

  // Likelihood calculation using lognormal error distributions for each time step.
  Type nll = 0.0;
  for(int t = 0; t < n; t++){
    // The dlnorm() function evaluates the log-density of a lognormal distribution.
    nll -= dlnorm(cots_dat[t], log(cots_pred[t] + Type(1e-8)), sd_cots, true);
    nll -= dlnorm(fast_dat[t], log(fast_pred[t] + Type(1e-8)), sd_fast, true);
    nll -= dlnorm(slow_dat[t], log(slow_pred[t] + Type(1e-8)), sd_slow, true);
  }

  // Report predicted trajectories for diagnostics.
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}
