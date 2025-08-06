#include <TMB.hpp>

 // Template Model Builder model for COTS and coral dynamics
 // Model Equations (recursive formulation):
 // 1. slow_pred[i] = slow_pred[i-1] * growth_slow * (1 - feeding_rate * cots_pred[i-1])
 // 2. fast_pred[i] = fast_pred[i-1] * growth_fast * (1 - feeding_rate * cots_pred[i-1])
 // 3. cots_pred[i] = cots_pred[i-1] * (1 + cotsimm_dat[i])
 // 4. Likelihood: Observations (slow_dat, fast_dat, cots_dat) are modeled using lognormal distributions 
 //    with predictions slow_pred, fast_pred, and cots_pred respectively.

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;
  
  // DATA:
  DATA_VECTOR(slow_dat);    // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);    // Observed fast-growing coral cover (%)
  DATA_VECTOR(cots_dat);    // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(sst_dat);     // Sea-surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat); // COTS immigration rate (individuals/m2/year)
  
  // PARAMETERS:
  PARAMETER(log_growth_slow);        // Log intrinsic growth rate for slow-growing corals (year^-1)
  PARAMETER(log_growth_fast);        // Log intrinsic growth rate for fast-growing corals (year^-1)
  PARAMETER(log_feeding_rate);       // Log feeding (impact) rate of COTS on corals (m2/(individual*year))
  PARAMETER(log_sigma_slow);         // Log observation noise for slow-growing corals
  PARAMETER(log_sigma_fast);         // Log observation noise for fast-growing corals
  PARAMETER(log_sigma_cots);         // Log observation noise for COTS
  PARAMETER(log_init_slow);          // Log initial slow-growing coral cover
  PARAMETER(log_init_fast);          // Log initial fast-growing coral cover
  PARAMETER(log_init_cots);          // Log initial COTS abundance
  PARAMETER(log_gamma_sst);          // Log sensitivity of coral growth to deviation in SST (unitless)

  // TRANSFORMED PARAMETERS:
  Type growth_slow   = exp(log_growth_slow);       // Intrinsic growth rate for slow corals
  Type growth_fast   = exp(log_growth_fast);       // Intrinsic growth rate for fast corals
  Type feeding_rate  = exp(log_feeding_rate);      // Impact rate of COTS on corals
  Type sigma_slow    = exp(log_sigma_slow);         // Observation noise for slow corals
  Type sigma_fast    = exp(log_sigma_fast);         // Observation noise for fast corals
  Type sigma_cots    = exp(log_sigma_cots);         // Observation noise for COTS
  
  Type init_slow     = exp(log_init_slow);           // Initial slow-growing coral cover
  Type init_fast     = exp(log_init_fast);           // Initial fast-growing coral cover
  Type init_cots     = exp(log_init_cots);           // Initial COTS abundance
  Type gamma_sst     = exp(log_gamma_sst);           // Sensitivity of coral growth to SST deviation

  // Numerical stability constant
  Type epsilon = Type(1e-8);
  // Ensure sigma lower bounds
  Type sigma_min = Type(1e-3);
  if(sigma_slow < sigma_min) sigma_slow = sigma_min;
  if(sigma_fast < sigma_min) sigma_fast = sigma_min;
  
  // Model predictions
  int n = slow_dat.size();
  vector<Type> slow_pred(n), fast_pred(n), cots_pred(n);
  
  // Initialize predictions with parameter-based initial conditions
  slow_pred[0] = init_slow;
  fast_pred[0] = init_fast;
  cots_pred[0] = init_cots;
  
  // Recursive predictions using external forcing and previous predictions
  for(int i = 1; i < n; i++){
    // Equation 1: Slow coral prediction: previous slow multiplied by intrinsic growth, modified by SST effect, and reduced by COTS impact
    slow_pred[i] = slow_pred[i-1] * growth_slow * (1 + gamma_sst * (sst_dat[i] - 28)) * (1 - feeding_rate * cots_pred[i-1]);
    // Equation 2: Fast coral prediction: previous fast multiplied by intrinsic growth, modified by SST effect, and reduced by COTS impact
    fast_pred[i] = fast_pred[i-1] * growth_fast * (1 + gamma_sst * (sst_dat[i] - 28)) * (1 - feeding_rate * cots_pred[i-1]);
    // Equation 3: COTS prediction: previous COTS adjusted solely by external immigration
    cots_pred[i] = cots_pred[i-1] * (1 + cotsimm_dat[i]);
  }
  
  // Likelihood calculation: lognormal likelihoods for all response variables
  Type nll = 0.0;
  for(int i = 0; i < n; i++){
    nll -= dlnorm(slow_dat[i] + epsilon, log(slow_pred[i] + epsilon), sigma_slow, true);
    nll -= dlnorm(fast_dat[i] + epsilon, log(fast_pred[i] + epsilon), sigma_fast, true);
    nll -= dlnorm(cots_dat[i] + epsilon, log(cots_pred[i] + epsilon), sigma_cots, true);
  }
  
  // REPORT outputs: model predictions and parameter estimates are reported
  REPORT(slow_pred);      // Predicted slow-growing coral cover
  REPORT(fast_pred);      // Predicted fast-growing coral cover
  REPORT(cots_pred);      // Predicted COTS abundance
  REPORT(growth_slow);    // Intrinsic growth rate for slow corals
  REPORT(growth_fast);    // Intrinsic growth rate for fast corals
  REPORT(feeding_rate);   // Feeding rate of COTS on corals
  REPORT(sigma_slow);     // Observation noise for slow corals
  REPORT(sigma_fast);     // Observation noise for fast corals
  REPORT(sigma_cots);     // Observation noise for COTS
  
  return nll;
}
