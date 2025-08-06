#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs
  DATA_VECTOR(Year);                           // Years (matches the first column of the CSV)
  int n = Year.size();                         // Number of time steps (computed from Year vector)
  DATA_VECTOR(cots_dat);                       // Observed COTS adult abundance (individuals/m2)
  DATA_VECTOR(fast_dat);                       // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);                       // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);                        // Sea-surface temperature data (°C)
  DATA_VECTOR(cotsimm_dat);                    // COTS larval immigration rate (individuals/m2/year)

  // Parameters (all must be estimated with biologically meaningful ranges)
  PARAMETER(r);          // Intrinsic growth rate of COTS (year^-1)
  PARAMETER(K);          // Carrying capacity of COTS (individuals/m2)
  PARAMETER(d);          // Natural death rate of COTS (year^-1)
  PARAMETER(alpha);      // Predation effect coefficient on coral (unitless)
  PARAMETER(recovery_fast); // Recovery rate for fast-growing coral (% per year)
  PARAMETER(recovery_slow); // Recovery rate for slow-growing coral (% per year)
  PARAMETER(sd_cots);    // Observation error SD for COTS (log-transformed data)
  PARAMETER(sd_coral);   // Observation error SD for coral (log-transformed data)

  // Small constant for numerical stability
  Type eps = Type(1e-8);

  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Vectors to store predicted states
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // ========================================================================
  // Equation (1): Initial conditions
  // Set initial states as first observed values (or calibrated separately)
  cots_pred[0] = cots_dat[0] + eps;    // Avoid zero by adding small constant
  fast_pred[0] = fast_dat[0] + eps;
  slow_pred[0] = slow_dat[0] + eps;
  
  // Loop over time steps: using previous time step states for predictions
  for(int t = 1; t < n; t++){
    // ========================================================================
    // Equation (2): COTS dynamics
    // cots_pred[t] = cots_pred[t-1] + r * cots_pred[t-1] * (1 - cots_pred[t-1]/K)
    //                + cotsimm_dat[t-1] - d * cots_pred[t-1]
    //                [Incorporates environmental forcing via SST using a simple linear modulator]
    Type env_effect = 1.0 + 0.1 * sst_dat[t-1];  // Environmental multiplier (unitless)
    cots_pred[t] = cots_pred[t-1] 
      + r * cots_pred[t-1] * (1 - cots_pred[t-1] / (K + eps)) * env_effect 
      + cotsimm_dat[t-1] 
      - d * cots_pred[t-1];
    
    // ========================================================================
    // Equation (3): Fast-growing coral dynamics
    // fast_pred[t] = fast_pred[t-1] - alpha * cots_pred[t-1] * fast_pred[t-1] / (fast_pred[t-1] + eps)
    //                + recovery_fast * (100 - fast_pred[t-1])
    fast_pred[t] = fast_pred[t-1] 
      - alpha * cots_pred[t-1] * fast_pred[t-1] / (fast_pred[t-1] + eps)
      + recovery_fast * (100 - fast_pred[t-1]);
    
    // ========================================================================
    // Equation (4): Slow-growing coral dynamics
    // slow_pred[t] = slow_pred[t-1] - alpha * cots_pred[t-1] * slow_pred[t-1] / (slow_pred[t-1] + eps)
    //                + recovery_slow * (100 - slow_pred[t-1])
    slow_pred[t] = slow_pred[t-1] 
      - alpha * cots_pred[t-1] * slow_pred[t-1] / (slow_pred[t-1] + eps)
      + recovery_slow * (100 - slow_pred[t-1]);

    // Ensure predictions remain positive to avoid invalid logarithms
    if(cots_pred[t] < eps) cots_pred[t] = eps;
    if(fast_pred[t] < eps) fast_pred[t] = eps;
    if(slow_pred[t] < eps) slow_pred[t] = eps;
    
    // ========================================================================
    // Likelihood: Observation model using lognormal errors
    // Note: Likelihood is calculated using predictions from the previous state.
    nll -= dnorm(log(cots_dat[t] + eps), log(cots_pred[t] + eps), sd_cots, true);
    nll -= dnorm(log(fast_dat[t] + eps), log(fast_pred[t] + eps), sd_coral, true);
    nll -= dnorm(log(slow_dat[t] + eps), log(slow_pred[t] + eps), sd_coral, true);
  }
  
  // Report predictions (_pred variables)
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
