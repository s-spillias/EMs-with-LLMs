#include <TMB.hpp>

// 1. COTS dynamics: C[t+1] = C[t] + growth + environmental forcing - predation effects
// 2. Fast coral dynamics: F[t+1] = F[t] + logistic growth - predation by COTS
// 3. Slow coral dynamics: S[t+1] = S[t] + logistic growth - predation by COTS
//
// Notes:
// - All state updates use the previous time step's values only.
// - Small constants (1e-8) are used to prevent division by zero and ensure smooth transitions.
// - Observations are incorporated via lognormal likelihood with fixed minimum standard deviations.
template<class Type>
Type objective_function<Type>::operator() ()
{
  USING_ARRAYS();
  
  // DATA: Observations (from Data/timeseries_data_COTS_response.csv & COTS_forcing.csv)
  DATA_VECTOR(cots_dat);      // Observed COTS densities (individuals/m2)
  DATA_VECTOR(fast_dat);      // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);      // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);       // Sea Surface Temperature (°C)
  DATA_VECTOR(cotsimm_dat);   // COTS larval immigration rate (individuals/m2/year)
  
  int n = cots_dat.size();
  
  // PARAMETERS for COTS dynamics
  PARAMETER(log_C0);      // log initial COTS density
  PARAMETER(r_cots);      // Intrinsic growth rate of COTS (year^-1)
  PARAMETER(beta_fast);   // Predation effect rate of COTS on fast coral (m2/(individual*%))
  PARAMETER(beta_slow);   // Predation effect rate of COTS on slow coral (m2/(individual*%))
  
  // PARAMETERS for coral dynamics
  PARAMETER(g_fast);   // Growth rate for fast coral (% increase per year)
  PARAMETER(g_slow);   // Growth rate for slow coral (% increase per year)
  
  // PARAMETERS for environmental forcing
  PARAMETER(e_sst);      // Effect of SST on COTS growth (individuals/m2 per °C)
  PARAMETER(e_cotsimm);  // Effect of larval immigration on COTS growth (individuals/m2 per unit rate)
  PARAMETER(log_fast0);    // Logarithm of the initial fast coral cover (% cover)
  PARAMETER(log_slow0);    // Logarithm of the initial slow coral cover (% cover)
  
  // PROCESS ERROR parameters (on log scale)
  PARAMETER(log_sigma_cots);
  PARAMETER(log_sigma_fast);
  PARAMETER(log_sigma_slow);
  
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_fast = exp(log_sigma_fast);
  Type sigma_slow = exp(log_sigma_slow);
  
  // Initialize state vectors for predictions
  vector<Type> cots_pred(n);   // COTS predictions (individuals/m2)
  vector<Type> fast_pred(n);   // Fast coral predictions (% cover)
  vector<Type> slow_pred(n);   // Slow coral predictions (% cover)
  
  // Set initial conditions
  cots_pred[0] = exp(log_C0);
  fast_pred[0] = exp(log_fast0);
  slow_pred[0] = exp(log_slow0);
  
  Type nll = 0.0;  // negative log-likelihood
  Type small = Type(1e-8);  // small constant for numerical stability
  
  // Loop over time steps (using previous time step's state)
  for (int t = 0; t < n - 1; t++){
    // Equation 1: COTS dynamics
    // Growth term with saturating effect employing a carrying capacity of 1000 individuals/m2
    Type growth = r_cots * cots_pred[t] * (1 - cots_pred[t] / (Type(1000) + small));
    // Predation effect reduces COTS via consumption of both coral types
    Type pred_effect = beta_fast * fast_pred[t] + beta_slow * slow_pred[t];
    // Environmental forcing modulated by SST and larval immigration
    Type forcing = e_sst * sst_dat[t] + e_cotsimm * cotsimm_dat[t];
    
    cots_pred[t+1] = cots_pred[t] + growth + forcing - pred_effect;
    
    // Equation 2: Fast coral dynamics (logistic growth with predation)
    fast_pred[t+1] = fast_pred[t] + g_fast * fast_pred[t] * (1 - fast_pred[t] / (Type(100) + small))
                  - beta_fast * cots_pred[t] * fast_pred[t];
    
    // Equation 3: Slow coral dynamics (logistic growth with predation)
    slow_pred[t+1] = slow_pred[t] + g_slow * slow_pred[t] * (1 - slow_pred[t] / (Type(80) + small))
                  - beta_slow * cots_pred[t] * slow_pred[t];
  }
  
  // Likelihood: assume lognormal errors to account for positive-only data values
  for (int t = 0; t < n; t++){
    nll -= dlnorm(cots_dat[t] + small, log(cots_pred[t] + small), sigma_cots, true);
    nll -= dlnorm(fast_dat[t] + small, log(fast_pred[t] + small), sigma_fast, true);
    nll -= dlnorm(slow_dat[t] + small, log(slow_pred[t] + small), sigma_slow, true);
  }
  
  // Report predicted time series (COTS, fast coral, and slow coral)
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
