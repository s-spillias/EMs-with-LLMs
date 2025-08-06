#include <TMB.hpp>

// Sigmoid function for smooth transition
template<class Type>
Type sigmoid(Type x, Type a, Type b) { // a: steepness, b: midpoint
  return Type(1) / (Type(1) + exp(-a * (x - b)));
}

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;

  // DATA SECTION:
  DATA_VECTOR(cots_dat);    // Observed Crown-of-Thorns starfish abundance (individuals/m^2)
  DATA_VECTOR(slow_dat);    // Observed slow-growing coral cover (%) (Faviidae & Porites)
  DATA_VECTOR(fast_dat);    // Observed fast-growing coral cover (%) (Acropora)
  DATA_VECTOR(sst_dat);     // Observed sea-surface temperature (°C)
  DATA_VECTOR(cotsimm_dat); // Observed crown-of-thorns immigration rate (individuals/m^2/year)

  // PARAMETERS SECTION:
  // Log-transformed intrinsic growth rate (annual, unitless after transformation)
  PARAMETER(log_growth_rate);
  // Log-transformed decay rate for slow-growing corals (annual, unitless after transformation)
  PARAMETER(log_decay_slow);
  // Log-transformed decay rate for fast-growing corals (annual, unitless after transformation)
  PARAMETER(log_decay_fast);
  // Temperature effect multiplier (unitless)
  PARAMETER(temp_effect);
  // Standard deviation for slow coral observations (log-scale)
  PARAMETER(sig_slow);
  // Standard deviation for fast coral observations (log-scale)
  PARAMETER(sig_fast);
  // Impact rate of starfish on slow corals (unitless)
  PARAMETER(impact_rate_slow);
  // Impact rate of starfish on fast corals (unitless)
  PARAMETER(impact_rate_fast);
  // Mortality rate for crown-of-thorns starfish (year^-1)
  PARAMETER(mortality_cots);
  // Standard deviation for the lognormal likelihood of starfish observations
  PARAMETER(sig_cots);
  // New parameters for temperature-driven decay modulation via the sigmoid function
  PARAMETER(steepness);
  PARAMETER(midpoint);

  // Transform parameters to ensure positivity
  Type growth_rate = exp(log_growth_rate);   // Intrinsic growth rate (year^-1)
  Type decay_slow  = exp(log_decay_slow);       // Decay rate for slow corals (year^-1)
  Type decay_fast  = exp(log_decay_fast);       // Decay rate for fast corals (year^-1)

  // Likelihood initialization
  int n = cots_dat.size();   // Number of observations
  Type nll = 0.0;

  // --- Model Equations ---
  // 1. Although starfish abundance dynamics are not explicitly modeled here,
  //    the coral cover decline is driven by exponential decay modulated by temperature.
  // 2. For each time step i:
  //    slow_pred(i) = slow0 * exp(-decay_slow * (1 + temp_effect * sigmoid(sst_dat(i), steepness, midpoint) + 1e-8))
  //    fast_pred(i) = fast0 * exp(-decay_fast * (1 + temp_effect * sigmoid(sst_dat(i), steepness, midpoint) + 1e-8))
  
  // Initialize predictions using first observations
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);

  // Loop through each time step starting from 1
  for(int i = 1; i < n; i++){
    // Predict crown-of-thorns starfish:
    //   previous value + immigration forcing - mortality
    cots_pred(i) = cots_pred(i-1) + cotsimm_dat(i) - mortality_cots * cots_pred(i-1) + Type(1e-8);
    // Predict slow-growing corals:
    //   previous cover multiplied by growth, reduced by starfish impact and modulated by temperature-driven decay
    slow_pred(i) = slow_pred(i-1) * growth_rate * exp(-decay_slow * (Type(1) + temp_effect * sigmoid(sst_dat(i), steepness, midpoint))) * (Type(1) - impact_rate_slow * cots_pred(i-1)) + Type(1e-8);
    // Predict fast-growing corals:
    //   previous cover multiplied by growth, reduced by starfish impact and modulated by temperature-driven decay
    fast_pred(i) = fast_pred(i-1) * growth_rate * exp(-decay_fast * (Type(1) + temp_effect * sigmoid(sst_dat(i), steepness, midpoint))) * (Type(1) - impact_rate_fast * cots_pred(i-1)) + Type(1e-8);
  }

  // Likelihood Calculation using lognormal likelihoods (for strictly positive data)
  // dlnorm(x, log_mu, sigma, true) is replaced by: dnorm(log(x), log_mu, sigma, true) - log(x)
  for(int i = 0; i < n; i++){
    Type sd_cots = sig_cots < Type(1e-8) ? Type(1e-8) : sig_cots;
    Type sd_slow = sig_slow < Type(1e-8) ? Type(1e-8) : sig_slow;
    Type sd_fast = sig_fast < Type(1e-8) ? Type(1e-8) : sig_fast;
    nll -= dnorm(log(cots_dat(i) + Type(1e-8)), log(cots_pred(i) + Type(1e-8)), sd_cots, true) - log(cots_dat(i) + Type(1e-8));
    nll -= dnorm(log(slow_dat(i) + Type(1e-8)), log(slow_pred(i) + Type(1e-8)), sd_slow, true) - log(slow_dat(i) + Type(1e-8));
    nll -= dnorm(log(fast_dat(i) + Type(1e-8)), log(fast_pred(i) + Type(1e-8)), sd_fast, true) - log(fast_dat(i) + Type(1e-8));
  }

  // REPORT important model outputs for post-analysis
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(growth_rate);         // Intrinsic growth parameter for corals (unitless)
  REPORT(impact_rate_slow);    // Impact rate of starfish on slow corals
  REPORT(impact_rate_fast);    // Impact rate of starfish on fast corals
  REPORT(mortality_cots);      // Mortality rate for crown-of-thorns starfish (year^-1)
  REPORT(temp_effect);         // Temperature effect multiplier

  return nll;
}
