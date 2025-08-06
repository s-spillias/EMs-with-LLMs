#include <TMB.hpp>

// 1. Data inputs and parameter declarations:
//    - Year: time index from data (year)
//    - cots_dat, fast_dat, slow_dat: Observations for COTS and coral cover
//    - sst_dat, cotsimm_dat: Environmental forcing data (SST) and larval immigration rate
// 2. Parameters are provided on the log scale for numerical stability.
// 3. Equations:
//    (1) COTS dynamics: cots_pred[t] = cots_pred[t-1] + dt * (modulated growth - density dependence + larval immigration)
//    (2) Fast coral dynamics: fast_pred[t] = fast_pred[t-1] + dt * (recovery - predation by COTS)
//    (3) Slow coral dynamics: slow_pred[t] = slow_pred[t-1] + dt * (recovery - predation by COTS)
template<class Type>
Type objective_function<Type>::operator() () {
  // Data Section
  DATA_VECTOR(Year);          // Year observations (year)
  DATA_VECTOR(cots_dat);        // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);        // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);        // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);         // Sea Surface Temperature (°C)
  DATA_VECTOR(cotsimm_dat);     // COTS larval immigration rate (individuals/m2/year)
  
  // Parameter Section (parameters are on a log scale for robust estimation)
  PARAMETER(log_growth_rate);        // Log of intrinsic growth rate of COTS (year^-1)
  PARAMETER(log_attack_rate_fast);   // Log of attack rate on fast-growing coral (m2/% per year)
  PARAMETER(log_attack_rate_slow);   // Log of attack rate on slow-growing coral (m2/% per year)
  PARAMETER(log_threshold);          // Log of coral cover threshold triggering outbreak (%)
  PARAMETER(log_recovery_rate);      // Log of coral recovery rate (year^-1)
  PARAMETER(log_efficiency);         // Log of efficiency converting predation effects to COTS growth
  PARAMETER(half_saturation_coral);  // Half-saturation constant (in % coral cover) for the variable efficiency term
  PARAMETER(sigma_cots);             // Measurement error for COTS (log-scale)
  PARAMETER(sigma_fast);             // Measurement error for fast coral cover (log-scale)
  PARAMETER(sigma_slow);             // Measurement error for slow coral cover (log-scale)
  PARAMETER(log_handling_time_fast);   // Log of handling time for fast coral predation
  PARAMETER(log_handling_time_slow);     // Log of handling time for slow coral predation

  // Transform parameters from log scale
  Type growth_rate = exp(log_growth_rate);
  Type attack_rate_fast = exp(log_attack_rate_fast);
  Type attack_rate_slow = exp(log_attack_rate_slow);
  Type threshold = exp(log_threshold);
  Type recovery_rate = exp(log_recovery_rate);
  Type efficiency = exp(log_efficiency);
  Type hs_coral = half_saturation_coral;
  Type handling_time_fast = exp(log_handling_time_fast);
  Type handling_time_slow = exp(log_handling_time_slow);

  int n = Year.size();
  
  // Initialize predicted state vectors using observed initial conditions
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  cots_pred(0) = cots_dat(0); // Use initial observed COTS abundance
  fast_pred(0) = fast_dat(0); // Use initial observed fast coral cover
  slow_pred(0) = slow_dat(0); // Use initial observed slow coral cover
  
  Type nll = 0.0; // Initialize negative log-likelihood

  // Time loop (t = 1 to n-1) using previous predictions only (no data leakage)
  for(int t = 1; t < n; t++){
    // dt: time-step difference (in years)
    Type dt = Year(t) - Year(t-1);
    
    // Environmental modification: smooth linear modifier based on SST deviation from 26°C
    Type mod_growth = growth_rate * (Type(1) + 0.01 * (sst_dat(t-1) - Type(26)));
    
    // Smooth thresholds for coral resource availability (prevent divide-by-zero issues)
    Type coral_resource_fast = fast_pred(t-1) - threshold;
    Type coral_resource_slow = slow_pred(t-1) - threshold;
    coral_resource_fast = coral_resource_fast / (fabs(coral_resource_fast) + Type(1e-8));
    coral_resource_slow = coral_resource_slow / (fabs(coral_resource_slow) + Type(1e-8));
    
    // Compute effective (variable) conversion efficiency based on total available coral
    Type total_coral = fast_pred(t-1) + slow_pred(t-1);
    Type effective_efficiency = efficiency * total_coral / (hs_coral + total_coral);
    
    // 1. COTS dynamics: logistic-style growth modified by predation and larval immigration.
    cots_pred(t) = cots_pred(t-1)
      + dt * ( mod_growth * cots_pred(t-1) * (Type(1) - cots_pred(t-1) / (threshold + Type(1e-8))) 
               + effective_efficiency * ((attack_rate_fast * cots_pred(t-1) * coral_resource_fast) / (Type(1) + attack_rate_fast * handling_time_fast * cots_pred(t-1))
                                + (attack_rate_slow * cots_pred(t-1) * coral_resource_slow) / (Type(1) + attack_rate_slow * handling_time_slow * cots_pred(t-1)))
               + cotsimm_dat(t-1) );
    
    // 2. Fast coral dynamics: recovery towards 100% cover minus predation by COTS.
    fast_pred(t) = fast_pred(t-1)
      + dt * ( recovery_rate * (Type(100) - fast_pred(t-1))
               - (attack_rate_fast * cots_pred(t-1) * fast_pred(t-1)) / (Type(1) + attack_rate_fast * handling_time_fast * cots_pred(t-1)) );
               
    // 3. Slow coral dynamics: recovery towards 100% cover minus predation by COTS.
    slow_pred(t) = slow_pred(t-1)
      + dt * ( recovery_rate * (Type(100) - slow_pred(t-1))
               - (attack_rate_slow * cots_pred(t-1) * slow_pred(t-1)) / (Type(1) + attack_rate_slow * handling_time_slow * cots_pred(t-1)) );
    
    // Bound predictions to ensure positivity
    if(cots_pred(t) < Type(1e-8)) cots_pred(t) = Type(1e-8);
    if(fast_pred(t) < Type(1e-8)) fast_pred(t) = Type(1e-8);
    if(slow_pred(t) < Type(1e-8)) slow_pred(t) = Type(1e-8);

    // Likelihood contributions: using a lognormal error distribution for strictly positive data.
    // A small constant (1e-8) is added to predictions to ensure numerical stability (prevent log(0)).
    nll -= dnorm(log(cots_dat(t)), log(cots_pred(t) + Type(1e-8)), sigma_cots, true)
           - log(cots_dat(t) + Type(1e-8));
    nll -= dnorm(log(fast_dat(t)), log(fast_pred(t) + Type(1e-8)), sigma_fast, true)
           - log(fast_dat(t) + Type(1e-8));
    nll -= dnorm(log(slow_dat(t)), log(slow_pred(t) + Type(1e-8)), sigma_slow, true)
           - log(slow_dat(t) + Type(1e-8));
  }
  
  // Reporting predicted values for diagnostics
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
