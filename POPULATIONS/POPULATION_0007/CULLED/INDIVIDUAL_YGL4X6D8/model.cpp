#include <TMB.hpp>

// Helper function for lognormal density
template<class Type>
Type log_dlnorm(Type x, Type meanlog, Type sdlog) {
  return -log(x + Type(1e-8)) - log(sdlog) - 0.5 * log(2.0 * M_PI) - 0.5 * pow((log(x + Type(1e-8)) - meanlog)/sdlog, 2);
}

// Model Overview:
// 1. Data inputs: time series of observed COTS, fast-growing and slow-growing coral covers,
//    sea-surface temperature, and external COTS immigration.
// 2. Parameters: include intrinsic growth rates, carrying capacities, predation efficiencies,
//    coral dynamics parameters, environmental modifiers, and observation error terms.
// 3. Model equations:
//    [1] COTS dynamics: logistic growth modulated by a smooth outbreak mechanism and temperature.
//    [2] Fast and slow coral dynamics: logistic growth limited by max cover and reduced by saturating predation.
//    [3] Likelihood: computed for each observation using lognormal error distributions.
// 4. Predictions are based solely on previous time step values to avoid data leakage.

template<class Type>
Type objective_function<Type>::operator() () {
  using namespace density;
  
  // Data inputs
  DATA_VECTOR(time);        // Time variable (year)
  DATA_VECTOR(cots_dat);    // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);    // Observed fast-growing coral cover (%) (Acropora spp.)
  DATA_VECTOR(slow_dat);    // Observed slow-growing coral cover (%) (Faviidae & Porites spp.)
  DATA_VECTOR(sst_dat);     // Sea-surface temperature (°C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration rate (individuals/m2/year)
  
  int n = time.size();
  
  // Parameters
  PARAMETER(r_cots);       // Intrinsic COTS growth rate (year^-1)
  PARAMETER(K_cots);       // COTS carrying capacity (individuals/m2)
  PARAMETER(pred_eff_fast); // Predation efficiency on fast-growing corals (unitless)
  PARAMETER(pred_eff_slow); // Predation efficiency on slow-growing corals (unitless)
  PARAMETER(growth_fast);   // Growth rate of fast-growing corals (% per year)
  PARAMETER(growth_slow);   // Growth rate of slow-growing corals (% per year)
  PARAMETER(mort_coral);    // Coral mortality rate (year^-1)
  PARAMETER(temp_effect);   // Temperature effect coefficient on COTS growth (unitless)
  PARAMETER(threshold);         // COTS density threshold for outbreak effect (individuals/m2)
  PARAMETER(outbreak_mult);     // Multiplier for outbreak growth (unitless)
  PARAMETER(outbreak_steepness);   // Steepness for outbreak response (unitless)
  PARAMETER(log_sd_cots);   // Log standard deviation for COTS observations
  PARAMETER(log_sd_fast);   // Log standard deviation for fast-growing coral observations
  PARAMETER(log_sd_slow);   // Log standard deviation for slow-growing coral observations
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Predicted state vectors
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Initial conditions: set predictions equal to initial observed values
  cots_pred[0] = cots_dat[0];
  fast_pred[0] = fast_dat[0];
  slow_pred[0] = slow_dat[0];
  
  // Loop over time steps (using only previous time step's predictions)
  for (int t = 1; t < n; t++){
    // [1] COTS Dynamics:
    // Logistic growth with a smooth outbreak effect and temperature modulation.
    Type growth = r_cots * cots_pred[t-1] * (1.0 - cots_pred[t-1] / (K_cots + Type(1e-8)));  // Basic logistic term
    Type outbreak_effect = outbreak_mult / (1.0 + exp(-outbreak_steepness * (cots_pred[t-1] - threshold)));           // Outbreak trigger via flexible, steepness-adjusted threshold
    Type temp_mod = 1.0 + temp_effect * (sst_dat[t-1] - Type(28.0));                              // Temperature modifier (28°C as baseline)
    Type immig = cotsimm_dat[t-1];                                                                  // External larval immigration
    cots_pred[t] = cots_pred[t-1] + (growth + outbreak_effect * cots_pred[t-1]) * temp_mod + immig;
    // Ensure non-negative COTS predictions to avoid numerical issues
    cots_pred[t] = (cots_pred[t] > Type(1e-8)) ? cots_pred[t] : Type(1e-8);
    
    // [2] Fast-growing Coral Dynamics:
    // Logistic growth below 100% cover, minus mortality and saturating predation by COTS.
    Type predation_fast = pred_eff_fast * cots_pred[t-1] * fast_pred[t-1] / (fast_pred[t-1] + Type(1e-8));
    fast_pred[t] = fast_pred[t-1] + growth_fast * fast_pred[t-1] * (1.0 - fast_pred[t-1] / Type(100)) - mort_coral * fast_pred[t-1] - predation_fast;
    
    // [3] Slow-growing Coral Dynamics:
    // Similar to fast corals but with its specific predation efficiency.
    Type predation_slow = pred_eff_slow * cots_pred[t-1] * slow_pred[t-1] / (slow_pred[t-1] + Type(1e-8));
    slow_pred[t] = slow_pred[t-1] + growth_slow * slow_pred[t-1] * (1.0 - slow_pred[t-1] / Type(100)) - mort_coral * slow_pred[t-1] - predation_slow;
    
    // [4] Likelihood Calculation:
    // Ensure predictions are safely bounded to avoid log(0)
    Type safe_cots = CppAD::CondExpGt(cots_pred[t], Type(1e-8), cots_pred[t], Type(1e-8));
    Type safe_fast = CppAD::CondExpGt(fast_pred[t], Type(1e-8), fast_pred[t], Type(1e-8));
    Type safe_slow = CppAD::CondExpGt(slow_pred[t], Type(1e-8), slow_pred[t], Type(1e-8));
    // Lognormal likelihood for each observed data point, with a fixed small constant for numerical stability.
    nll -= log_dlnorm(cots_dat[t] + Type(1e-8), log(safe_cots + Type(1e-8)), log_sd_cots);
    nll -= log_dlnorm(fast_dat[t] + Type(1e-8), log(safe_fast + Type(1e-8)), log_sd_fast);
    nll -= log_dlnorm(slow_dat[t] + Type(1e-8), log(safe_slow + Type(1e-8)), log_sd_slow);
  }
  
  // Reporting predictions
  REPORT(cots_pred); // Predicted COTS abundance (individuals/m2)
  REPORT(fast_pred); // Predicted fast-growing coral cover (%)
  REPORT(slow_pred); // Predicted slow-growing coral cover (%)
  
  return nll;
}
