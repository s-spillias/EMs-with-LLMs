#include <TMB.hpp>

// Template Model Builder (TMB) model for COTS predation on corals on the Great Barrier Reef
// This model uses smooth transitions and penalty terms to ensure numerical stability and biologically meaningful parameter bounds

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;
  
  // DATA - Observations
  DATA_VECTOR(slow_dat);      // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);      // Observed fast-growing coral cover (%)
  DATA_VECTOR(cots_dat);      // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(sst_dat);       // Sea-surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);   // COTS immigration rate (individuals/m2/year)
  
  // PARAMETERS - Model parameters with comments (units in parentheses)
  PARAMETER(log_growth_slow); // Log intrinsic growth rate for slow corals (year^-1)
  PARAMETER(log_growth_fast); // Log intrinsic growth rate for fast corals (year^-1)
  PARAMETER(beta_slow);       // Effect of COTS on slow corals (unitless)
  PARAMETER(beta_fast);       // Effect of COTS on fast corals (unitless)
  PARAMETER(slow_base);       // Baseline cover for slow-growing corals (%)  
  PARAMETER(fast_base);       // Baseline cover for fast-growing corals (%)
  PARAMETER(log_cots_rate);   // Log scaling factor for COTS abundance related to SST (unitless)
  PARAMETER(sigma_slow);      // Observation error for slow corals (log scale)
  PARAMETER(sigma_fast);      // Observation error for fast corals (log scale)
  PARAMETER(sigma_cots);      // Observation error for COTS abundance (log scale)
  
  // Transform parameters to use intrinsic growth rates directly
  Type growth_slow = log_growth_slow; // Intrinsic growth rate for slow corals
  Type growth_fast = log_growth_fast; // Intrinsic growth rate for fast corals
  Type cots_rate   = exp(log_cots_rate);     // Scaling factor for COTS abundance
  
  // Small constant for numerical stability
  Type eps = Type(1e-8);
  
  int n = slow_dat.size();
  vector<Type> slow_pred(n);  // Predicted slow-growing coral cover (%)
  vector<Type> fast_pred(n);  // Predicted fast-growing coral cover (%)
  vector<Type> cots_pred(n);  // Predicted COTS abundance (individuals/m2)
  
  Type nll = 0.0;
  
  /* 
    Model Equations:
    1. Slow coral cover: slow_pred = slow_base * exp(- beta_slow * cots_dat) + eps
    2. Fast coral cover: fast_pred = fast_base * exp(- beta_fast * cots_dat) + eps
    3. COTS abundance: cots_pred = cots_rate * sst_dat + cotsimm_dat + eps
    Note: The added constant eps ensures numerical stability (e.g., prevents log(0)).
  */
  
  for(int i = 0; i < n; i++){
    slow_pred[i] = slow_base * exp(growth_slow - beta_slow * cots_dat[i]) + eps; // (1)
    fast_pred[i] = fast_base * exp(growth_fast - beta_fast * cots_dat[i]) + eps;   // (2)
    cots_pred[i] = cots_rate * sst_dat[i] + cotsimm_dat[i] + eps;       // (3)
    
    // Likelihood using a lognormal distribution for strictly positive data
    nll -= dnorm(log(slow_dat[i] + eps), log(slow_pred[i]), sigma_slow, true);
    nll -= dnorm(log(fast_dat[i] + eps), log(fast_pred[i]), sigma_fast, true);
    nll -= dnorm(log(cots_dat[i] + eps), log(cots_pred[i]), sigma_cots, true);
  }
  
  // Smooth penalties for parameter bounds using quadratic penalties
  // Example: slow_base and fast_base are bounded between 0 and 100 (%)
  Type penalty_weight = Type(1e6);
  if(slow_base < 0) nll += penalty_weight * ((0 - slow_base) * (0 - slow_base));
  if(slow_base > 100) nll += penalty_weight * ((slow_base - 100) * (slow_base - 100));
  if(fast_base < 0) nll += penalty_weight * ((0 - fast_base) * (0 - fast_base));
  if(fast_base > 100) nll += penalty_weight * ((fast_base - 100) * (fast_base - 100));
  
  // REPORT section: provide essential intermediate predictions and parameter transformations
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(cots_pred);
  REPORT(growth_slow);
  REPORT(growth_fast);
  REPORT(cots_rate);
  
  return nll;
}
