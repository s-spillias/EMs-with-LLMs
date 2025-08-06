#include <TMB.hpp>
using namespace density;

// Template Model Builder objective function
template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS
  // Note: All data vectors are assumed to be of equal length, corresponding to time steps
  DATA_VECTOR(Year);               // Time steps (year), first column as in the data file.
  DATA_VECTOR(cots_dat);           // Observed COTS abundance (individuals/m2) 
  DATA_VECTOR(fast_dat);           // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);           // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);            // Sea-surface temperature (°C) from forcing data
  DATA_VECTOR(cotsimm_dat);        // Larval immigration rate for COTS (individuals/m2/year)
  
  // PARAMETERS
  // COTS dynamics parameters
  PARAMETER(log_r_cots);           // Log intrinsic growth rate (year^-1)
  PARAMETER(log_K_cots);           // Log carrying capacity (individuals/m2)
  PARAMETER(outbreak_threshold);   // SST threshold for triggering outbreak (°C)
  PARAMETER(outbreak_boost);       // Additional outbreak boost (year^-1)
  
  // Coral predation parameters
  PARAMETER(beta_fast);            // Predation impact coefficient on fast coral (% impact per predicted COTS)
  PARAMETER(beta_slow);            // Predation impact coefficient on slow coral (% impact per predicted COTS)
  
  // Likelihood standard deviations (log-scale to enforce positivity)
  PARAMETER(log_sd_cots);          // Log standard deviation for COTS likelihood
  PARAMETER(log_sd_fast);          // Log standard deviation for fast coral likelihood
  PARAMETER(log_sd_slow);          // Log standard deviation for slow coral likelihood
  
  // Transform parameters to ensure positivity and numerical stability
  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type sd_cots = exp(log_sd_cots) + Type(1e-8);
  Type sd_fast = exp(log_sd_fast) + Type(1e-8);
  Type sd_slow = exp(log_sd_slow) + Type(1e-8);
  
  // Number of time steps
  int n = Year.size();
  
  // Initialize predicted state vectors
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Set initial conditions from the data (first observation)
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Numerical constant for stability
  Type eps = Type(1e-8);
  
  // Loop through time steps
  for(int t = 1; t < n; t++){
    // Equation 1: COTS dynamics with logistic growth and outbreak boost
    //   (1) Logistic growth: r_cots * cots_pred(t-1) * (1 - cots_pred(t-1)/K_cots)
    //   (2) Outbreak boost: outbreak_boost modulated by a smooth logistic function of (sst - outbreak_threshold)
    Type logistic_component = r_cots * cots_pred(t-1) * (1 - cots_pred(t-1) / (K_cots + eps));
    // Smooth switch function: logistic with high steepness to mimic a threshold effect.
    Type outbreak_effect = outbreak_boost / (Type(1.0) + exp(-100 * (sst_dat(t-1) - outbreak_threshold)));
    // Equation for COTS prediction:
    cots_pred(t) = cots_pred(t-1) + logistic_component + outbreak_effect + cotsimm_dat(t-1); // including larval immigration as additional source.
    
    // Equation 2: Fast-growing coral dynamics
    //   Coral cover declines due to predation by COTS. Using an exponential decay function.
    fast_pred(t) = fast_pred(t-1) * exp(-beta_fast * cots_pred(t-1));
    
    // Equation 3: Slow-growing coral dynamics
    //   Similarly, slow-growing coral cover declines with increasing COTS predation.
    slow_pred(t) = slow_pred(t-1) * exp(-beta_slow * cots_pred(t-1));
  }
  
  // Likelihood calculation using lognormal observation errors 
  Type nll = 0.0;
  for(int t = 1; t < n; t++){
    // Using lognormal likelihood: observations are strictly positive and span orders of magnitude.
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sd_cots, true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sd_fast, true);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sd_slow, true);
  }
  
  // Reporting predictions (_pred variables) for diagnostics and visualization
  REPORT(cots_pred);    // [1] COTS predicted time series 
  REPORT(fast_pred);    // [2] Fast-growing coral predicted time series
  REPORT(slow_pred);    // [3] Slow-growing coral predicted time series
  
  /*
  Equation List:
  (1) COTS Dynamics: 
      cots_pred[t] = cots_pred[t-1] + r_cots * cots_pred[t-1]*(1 - cots_pred[t-1]/K_cots)
                     + outbreak_boost / (1 + exp(-100*(sst_dat[t-1] - outbreak_threshold))) 
                     + cotsimm_dat[t-1]
  (2) Fast Coral Dynamics:
      fast_pred[t] = fast_pred[t-1] * exp(-beta_fast * cots_pred[t-1])
  (3) Slow Coral Dynamics:
      slow_pred[t] = slow_pred[t-1] * exp(-beta_slow * cots_pred[t-1])
  */
  
  return nll;
}
