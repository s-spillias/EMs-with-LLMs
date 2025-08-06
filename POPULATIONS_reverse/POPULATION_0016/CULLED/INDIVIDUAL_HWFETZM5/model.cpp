#include <TMB.hpp>
template<class Type>
Type dlnorm_helper(Type x, Type meanlog, Type sdlog, bool give_log){
  // Calculates the lognormal density (or its log) for x.
  Type log_pdf = -log(x) - log(sdlog) - Type(0.5)*log(2.0 * M_PI) - pow(log(x) - meanlog, 2) / (Type(2)*sdlog*sdlog);
  return give_log ? log_pdf : exp(log_pdf);
}
// Template Model Builder (TMB) model for COTS outbreak dynamics on the Great Barrier Reef
// Equations and parameters:
// (1) COTS population dynamics:
//     COTS_pred(t) = COTS_pred(t-1) + r * COTS_pred(t-1) * ( (total_coral/(half_sat + total_coral))*env_eff - COTS_pred(t-1)/K )
//     where total_coral = slow_pred(t-1) + fast_pred(t-1)
// (2) Slow coral dynamics:
//     slow_pred(t) = slow_pred(t-1) - pred_slow * COTS_pred(t-1) * slow_pred(t-1)/(slow_pred(t-1) + 1e-8)
// (3) Fast coral dynamics:
//     fast_pred(t) = fast_pred(t-1) - pred_fast * COTS_pred(t-1) * fast_pred(t-1)/(fast_pred(t-1) + 1e-8)
// 
// Parameter notes (units in comments):
// - log_r: log intrinsic growth rate of COTS (year^-1)
// - log_K: log carrying capacity of COTS (individuals/m^2)
// - log_half_sat: log half-saturation constant for coral cover (unitless fraction)
// - log_env_eff: log environmental efficiency modifier (unitless)
// - log_pred_slow: log predation rate on slow-growing coral (year^-1)
// - log_pred_fast: log predation rate on fast-growing coral (year^-1)
// - log_sigma_*: log observation error (unitless, ensuring sigma > 0)
// 
// Numerical Stability:
// - A small constant (1e-8) is added in denominators and errors to prevent division by zero.
// - Log-transforms of parameters ensure positivity.
// 
// Likelihood Calculation:
// - A lognormal likelihood is used for strictly positive data (COTS, slow coral, and fast coral observations).
// - Only prior time step values are used in predictions to guard against data leakage.
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs
  DATA_VECTOR(time);         // Time vector (years)
  DATA_VECTOR(cots_dat);       // Observed COTS abundance (individuals/m^2)
  DATA_VECTOR(slow_dat);       // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);       // Observed fast-growing coral cover (%)

  // Parameters (logged for positivity)
  PARAMETER(log_r);          // Intrinsic growth rate of COTS (year^-1)
  PARAMETER(log_K);          // Carrying capacity of COTS (individuals/m^2)
  PARAMETER(log_half_sat);   // Half-saturation constant for coral cover (unitless)
  PARAMETER(log_env_eff);    // Environmental efficiency modifier (unitless)
  PARAMETER(log_pred_slow);  // Predation rate on slow-growing coral (year^-1)
  PARAMETER(log_pred_fast);  // Predation rate on fast-growing coral (year^-1)
  PARAMETER(log_sigma_COTS);  // Observation error for COTS
  PARAMETER(log_sigma_slow);  // Observation error for slow coral
  PARAMETER(log_sigma_fast);  // Observation error for fast coral

  // Transform parameters from log-space
  Type r = exp(log_r);                  // Intrinsic growth rate (year^-1)
  Type K = exp(log_K);                  // Carrying capacity (individuals/m^2)
  Type half_sat = exp(log_half_sat);      // Half-saturation constant (unitless)
  Type env_eff = exp(log_env_eff);        // Environmental efficiency modifier
  Type pred_slow = exp(log_pred_slow);    // Predation rate on slow coral (year^-1)
  Type pred_fast = exp(log_pred_fast);    // Predation rate on fast coral (year^-1)
  Type sigma_COTS = exp(log_sigma_COTS) + Type(1e-8);
  Type sigma_slow = exp(log_sigma_slow) + Type(1e-8);
  Type sigma_fast = exp(log_sigma_fast) + Type(1e-8);

  int n = time.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);

  // Initial conditions: assume first observations as starting state
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);

  Type nll = 0.0; // Initialize negative log-likelihood

  // Loop over time steps (using only previous time step values for prediction)
  for(int t = 1; t < n; t++){
    Type total_coral = slow_pred(t-1) + fast_pred(t-1);

    // (1) COTS population update: growth modulated by coral cover saturation and environmental efficiency,
    //     and density-dependent regulation.
    cots_pred(t) = cots_pred(t-1) + r * cots_pred(t-1) * ((total_coral/(half_sat + total_coral)) * env_eff - cots_pred(t-1)/K);

    // (2) Slow coral dynamics: decline driven by predation from COTS.
    slow_pred(t) = slow_pred(t-1) - pred_slow * cots_pred(t-1) * slow_pred(t-1)/(slow_pred(t-1) + Type(1e-8));

    // (3) Fast coral dynamics: decline driven by predation from COTS.
    fast_pred(t) = fast_pred(t-1) - pred_fast * cots_pred(t-1) * fast_pred(t-1)/(fast_pred(t-1) + Type(1e-8));

    if(cots_pred(t) < Type(1e-8)) cots_pred(t) = Type(1e-8);
    if(slow_pred(t) < Type(1e-8)) slow_pred(t) = Type(1e-8);
    if(fast_pred(t) < Type(1e-8)) fast_pred(t) = Type(1e-8);

    // Likelihood: using lognormal error distribution for each component.
    nll -= dlnorm_helper(cots_dat(t), log(cots_pred(t) + Type(1e-8)), sigma_COTS, true);
    nll -= dlnorm_helper(slow_dat(t), log(slow_pred(t) + Type(1e-8)), sigma_slow, true);
    nll -= dlnorm_helper(fast_dat(t), log(fast_pred(t) + Type(1e-8)), sigma_fast, true);
  }

  // REPORT predictions to monitor model output
  REPORT(cots_pred);  // Predicted COTS abundance over time
  REPORT(slow_pred);  // Predicted slow-growing coral cover over time
  REPORT(fast_pred);  // Predicted fast-growing coral cover over time

  return nll;
}
