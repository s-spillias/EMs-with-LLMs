#include <TMB.hpp>
#include <cmath>

// Custom log-density for the lognormal distribution.
// Returns the log-density if give_log is true, otherwise returns the density.
template<class Type>
Type dlnorm(Type x, Type mu, Type sigma, bool give_log = true) {
  Type logpdf = -log(x) - log(sigma) - 0.5 * log(2.0 * M_PI) - 0.5 * pow((log(x) - mu) / sigma, 2);
  return (give_log ? logpdf : exp(logpdf));
}

// 1. Data Inputs 
//    - cots_dat (COTS density, individuals/m2)
//    - fast_dat (Fast-growing coral cover in %, e.g., Acropora spp.)
//    - slow_dat (Slow-growing coral cover in %, e.g., Faviidae spp. and Porities spp.)
//    - sst_dat (Sea Surface Temperature in Celsius) [environmental modifier]
//    - cotsimm_dat (COTS larval immigration rate, individuals/m2/year)
template<class Type>
Type objective_function<Type>::operator() () {
  // Data vectors (observations)
  DATA_VECTOR(cots_dat);        // observed COTS densities (ind/m2)
  DATA_VECTOR(fast_dat);        // observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);        // observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);         // sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);     // COTS larval immigration rate (ind/m2/year)
  int n = cots_dat.size();
  
  // 2. Parameters (in log-scale to ensure positivity)
  PARAMETER(log_growth_rate);        // log intrinsic growth rate of COTS (year^-1) [literature]
  PARAMETER(log_mortality_rate);       // log natural mortality rate of COTS (year^-1) [expert opinion]
  PARAMETER(log_predation_eff_fast);   // log predation efficiency on fast-growing coral (m2/ind) [initial estimate]
  PARAMETER(log_predation_eff_slow);   // log predation efficiency on slow-growing coral (m2/ind) [initial estimate]
  PARAMETER(log_recovery_fast);        // log recovery rate for fast-growing coral (% per year) [literature]
  PARAMETER(log_recovery_slow);        // log recovery rate for slow-growing coral (% per year) [literature]
  
  // 3. Initial Conditions (log-scale)
  PARAMETER(initial_log_cots);         // log initial COTS abundance (ind/m2)
  PARAMETER(initial_log_fast);         // log initial fast-growing coral cover (%)
  PARAMETER(initial_log_slow);         // log initial slow-growing coral cover (%)
  
  // 4. Observation error standard deviations (log-scale)
  PARAMETER(log_sigma_cots);   // log standard deviation for COTS observations
  PARAMETER(log_sigma_fast);   // log standard deviation for fast-growing coral observations
  PARAMETER(log_sigma_slow);   // log standard deviation for slow-growing coral observations
  PARAMETER(half_sat_cots_fast);    // half-saturation constant for COTS predation on fast-growing coral (ind/m2)
  PARAMETER(half_sat_cots_slow);    // half-saturation constant for COTS predation on slow-growing coral (ind/m2)
  PARAMETER(log_competition);    // log competition factor representing density-dependent mortality (e.g., cannibalism)
  
  // Transform parameters to natural scale
  Type growth_rate = exp(log_growth_rate);         // intrinsic growth rate (year^-1)
  Type mortality_rate = exp(log_mortality_rate);     // natural mortality rate (year^-1)
  Type predation_eff_fast = exp(log_predation_eff_fast); // predation efficiency on fast coral (m2/ind)
  Type predation_eff_slow = exp(log_predation_eff_slow); // predation efficiency on slow coral (m2/ind)
  Type recovery_fast = exp(log_recovery_fast);       // coral recovery rate fast (% per year)
  Type recovery_slow = exp(log_recovery_slow);       // coral recovery rate slow (% per year)
  Type competition = exp(log_competition);
  
  // Convert initial conditions to natural scale
  vector<Type> cots_pred(n), fast_pred(n), slow_pred(n);
  cots_pred(0) = exp(initial_log_cots);
  fast_pred(0) = exp(initial_log_fast);
  slow_pred(0) = exp(initial_log_slow);
  
  // Small constant for numerical stability to avoid division by zero
  Type eps = Type(1e-8);
  // Time step (assumed 1 year)
  Type dt = 1.0;
  
  /* 
  Equations:
  1) COTS dynamics:
     cots[t] = cots[t-1] +
               dt * [growth_rate * cots[t-1] * trigger(sst_dat[t-1], cotsimm_dat[t-1]) - mortality_rate * cots[t-1]]
     Trigger function: 1 + 0.1*sst_dat + 0.05*cotsimm_dat
     
  2) Fast-growing coral dynamics:
     fast[t] = fast[t-1] +
               dt * [recovery_fast * (max_fast - fast[t-1]) - predation_eff_fast * cots[t-1] * fast[t-1] / (fast[t-1] + eps)]
     where max_fast is the maximum potential cover (%), assumed 100.
     
  3) Slow-growing coral dynamics:
     slow[t] = slow[t-1] +
               dt * [recovery_slow * (max_slow - slow[t-1]) - predation_eff_slow * cots[t-1] * slow[t-1] / (slow[t-1] + eps)]
     where max_slow is assumed to be 100.
  
  Note: Only lagged (t-1) values are used to prevent data leakage.
  */
  for(int t = 1; t < n; t++){
    Type trigger = 1.0 + 0.1 * sst_dat[t - 1] + 0.05 * cotsimm_dat[t - 1]; // Environmental trigger multiplier
    
    // COTS dynamics with resource limitation and mortality, including density-dependent competition
    cots_pred(t) = cots_pred(t - 1) + dt * (growth_rate * cots_pred(t - 1) * trigger - mortality_rate * cots_pred(t - 1) - competition * cots_pred(t - 1) * cots_pred(t - 1));
    
    // Maximum coral cover (percentage)
    Type max_fast = 100.0;
    Type max_slow = 100.0;
    
    // Coral dynamics with recovery and saturating predation losses
    fast_pred(t) = fast_pred(t - 1) + dt * (recovery_fast * (max_fast - fast_pred(t - 1)) - predation_eff_fast * fast_pred(t - 1) * cots_pred(t - 1) / (half_sat_cots_fast + cots_pred(t - 1)));
    slow_pred(t) = slow_pred(t - 1) + dt * (recovery_slow * (max_slow - slow_pred(t - 1)) - predation_eff_slow * slow_pred(t - 1) * cots_pred(t - 1) / (half_sat_cots_slow + cots_pred(t - 1)));
    
    // Ensure predictions remain positive for numerical stability
    if(cots_pred(t) < eps) cots_pred(t) = eps;
    if(fast_pred(t) < eps) fast_pred(t) = eps;
    if(slow_pred(t) < eps) slow_pred(t) = eps;
  }
  
  // Likelihood: using lognormal likelihood to account for data spanning multiple orders of magnitude.
  Type sigma_min = Type(1e-2); // Fixed minimum standard deviation for numerical stability
  Type sigma_cots = (exp(log_sigma_cots) > sigma_min ? exp(log_sigma_cots) : sigma_min);
  Type sigma_fast = (exp(log_sigma_fast) > sigma_min ? exp(log_sigma_fast) : sigma_min);
  Type sigma_slow = (exp(log_sigma_slow) > sigma_min ? exp(log_sigma_slow) : sigma_min);
  
  Type nll = 0.0; // initialize negative log likelihood
  for (int t = 0; t < n; t++){
    nll -= dlnorm(cots_dat(t) + eps, log(cots_pred(t) + eps), sigma_cots, true); // COTS likelihood
    nll -= dlnorm(fast_dat(t) + eps, log(fast_pred(t) + eps), sigma_fast, true);  // fast-growing coral likelihood
    nll -= dlnorm(slow_dat(t) + eps, log(slow_pred(t) + eps), sigma_slow, true);  // slow-growing coral likelihood
  }
  
  // Reporting the predictions for inspection and further analysis
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
