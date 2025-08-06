// model.cpp: TMB model for predicting Crown-of-Thorns starfish dynamics and coral ecosystem impacts
#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() () {
  // DATA:
  DATA_VECTOR(Year);                           // Year: time steps (years)
  DATA_VECTOR(cots_dat);                       // cots_dat: observed COTS density (ind./m2)
  DATA_VECTOR(slow_dat);                       // slow_dat: observed slow-growing coral cover (%) [Faviidae/Porites]
  DATA_VECTOR(fast_dat);                       // fast_dat: observed fast-growing coral cover (%) [Acropora]
  DATA_VECTOR(sst_dat);                        // sst_dat: sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);                    // cotsimm_dat: observed immigration rate (ind./m2/year)

  // PARAMETERS:
  PARAMETER(r_cots);                           // (year^-1) Intrinsic growth rate of COTS
  PARAMETER(cots_K);                           // (ind./m2) Carrying capacity for starfish population
  PARAMETER(environment_effect);               // (year^-1 °C^-1) Modulation of growth rate by SST
  PARAMETER(slow_recov);                       // (% cover/year) Recovery rate for slow-growing corals
  PARAMETER(fast_recov);                       // (% cover/year) Recovery rate for fast-growing corals
  PARAMETER(predation_rate_slow);              // (% cover loss per [ind./m2]) Predation rate on slow corals
  PARAMETER(predation_rate_fast);              // (% cover loss per [ind./m2]) Predation rate on fast corals
  PARAMETER(log_sd_cots);                      // Log-standard deviation for COTS observations
  PARAMETER(log_sd_slow);                      // Log-standard deviation for slow coral observations
  PARAMETER(log_sd_fast);                      // Log-standard deviation for fast coral observations
  
  Type sd_cots = exp(log_sd_cots) + Type(1e-8); // ensure positive SD for COTS
  Type sd_slow = exp(log_sd_slow) + Type(1e-8); // ensure positive SD for slow corals
  Type sd_fast = exp(log_sd_fast) + Type(1e-8); // ensure positive SD for fast corals

  // Number of time steps
  int n = Year.size();
  
  // Initialize predicted state vectors. Predictions are based on previous time step values.
  vector<Type> cots_pred(n);                 // COTS density prediction (ind./m2)
  vector<Type> slow_pred(n);                 // Slow coral cover prediction (%)
  vector<Type> fast_pred(n);                 // Fast coral cover prediction (%)
  
  // Set initial conditions to observed values at time 0
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Negative log-likelihood
  Type nll = 0.0;
  
  /* 
     Equations used:
     (1) COTS dynamics: cots_pred[t] = cots[t-1] * exp((r_cots + environment_effect * sst[t-1]) * (1 - cots[t-1]/(cots_K+1e-8))) + cotsimm_dat[t]
         - Exponential growth moderated by density-dependence and SST, plus external immigration.
     (2) Slow coral dynamics: slow_pred[t] = slow[t-1] + slow_recov*(100 - slow[t-1]) - predation_rate_slow * cots[t-1] * slow[t-1] / (100+1e-8)
         - Recovery towards full cover minus losses from COTS predation.
     (3) Fast coral dynamics: fast_pred[t] = fast[t-1] + fast_recov*(100 - fast[t-1]) - predation_rate_fast * cots[t-1] * fast[t-1] / (100+1e-8)
         - Similar dynamics as slow corals but with different recovery and predation rates.
  */
  
  // Loop over time steps (starting at t=1 to avoid using current observations)
  for(int t = 1; t < n; t++){
    // Equation (1): Starfish population dynamics using previous time step's COTS density and environmental forcing.
    cots_pred(t) = cots_pred(t-1) * exp((r_cots + environment_effect * sst_dat(t-1)) 
                    * (Type(1.0) - cots_pred(t-1)/(cots_K + Type(1e-8))))
                    + cotsimm_dat(t);
                    
    // Equation (2): Slow coral dynamics with recovery and reduction due to predation.
    slow_pred(t) = slow_pred(t-1) 
                   + slow_recov * (Type(100.0) - slow_pred(t-1))
                   - predation_rate_slow * cots_pred(t-1) * slow_pred(t-1) / (Type(100.0) + Type(1e-8));
    
    // Equation (3): Fast coral dynamics.
    fast_pred(t) = fast_pred(t-1) 
                   + fast_recov * (Type(100.0) - fast_pred(t-1))
                   - predation_rate_fast * cots_pred(t-1) * fast_pred(t-1) / (Type(100.0) + Type(1e-8));
    
    // Likelihood calculation: compare log-transformed observed data with predictions using the error structure.
    // Using lognormal error, which is appropriate for strictly positive data.
    nll -= dnorm(log(cots_dat(t)), log(cots_pred(t)), sd_cots, true);
    nll -= dnorm(log(slow_dat(t)), log(slow_pred(t)), sd_slow, true);
    nll -= dnorm(log(fast_dat(t)), log(fast_pred(t)), sd_fast, true);
  }
  
  // REPORT predicted states for each time step
  REPORT(cots_pred);   // Predicted COTS density (ind./m2)
  REPORT(slow_pred);   // Predicted slow coral cover (%)
  REPORT(fast_pred);   // Predicted fast coral cover (%)
  
  return nll;
}
