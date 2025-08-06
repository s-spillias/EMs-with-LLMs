#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA SECTION
  DATA_VECTOR(Year);                    // Years of observation
  DATA_VECTOR(cots_dat);                // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(slow_dat);                // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);                // Observed fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);                 // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);             // COTS immigration rate (individuals/m2/year)
  
  // PARAMETER SECTION
  PARAMETER(log_r_cots);                // Log of COTS intrinsic growth rate (year^-1)
  PARAMETER(log_K_cots);                // Log of COTS carrying capacity (individuals/m2)
  PARAMETER(log_alpha_slow);            // Log of COTS predation rate on slow-growing coral (m2/individual/year)
  PARAMETER(log_alpha_fast);            // Log of COTS predation rate on fast-growing coral (m2/individual/year)
  PARAMETER(log_pref_fast);             // Log of COTS preference for fast-growing coral (dimensionless)
  PARAMETER(log_r_slow);                // Log of slow-growing coral intrinsic growth rate (year^-1)
  PARAMETER(log_r_fast);                // Log of fast-growing coral intrinsic growth rate (year^-1)
  PARAMETER(log_K_slow);                // Log of slow-growing coral carrying capacity (%)
  PARAMETER(log_K_fast);                // Log of fast-growing coral carrying capacity (%)
  PARAMETER(log_temp_opt);              // Log of optimal temperature for COTS (°C)
  PARAMETER(log_temp_width);            // Log of temperature tolerance width for COTS (°C)
  PARAMETER(log_temp_coral_threshold);  // Log of temperature threshold for coral bleaching (°C)
  PARAMETER(log_bleach_mort_slow);      // Log of bleaching mortality rate for slow-growing coral (proportion)
  PARAMETER(log_bleach_mort_fast);      // Log of bleaching mortality rate for fast-growing coral (proportion)
  PARAMETER(log_coral_dependency);      // Log of COTS dependency on coral for survival (dimensionless)
  PARAMETER(log_sigma_cots);            // Log of observation error standard deviation for COTS abundance
  PARAMETER(log_sigma_slow);            // Log of observation error standard deviation for slow-growing coral cover
  PARAMETER(log_sigma_fast);            // Log of observation error standard deviation for fast-growing coral cover
  
  // DERIVED PARAMETERS (transform from log-space)
  Type r_cots = exp(log_r_cots);                // COTS intrinsic growth rate (year^-1)
  Type K_cots = exp(log_K_cots);                // COTS carrying capacity (individuals/m2)
  Type alpha_slow = exp(log_alpha_slow);        // COTS predation rate on slow-growing coral (m2/individual/year)
  Type alpha_fast = exp(log_alpha_fast);        // COTS predation rate on fast-growing coral (m2/individual/year)
  Type pref_fast = exp(log_pref_fast);          // COTS preference for fast-growing coral (dimensionless)
  Type r_slow = exp(log_r_slow);                // Slow-growing coral intrinsic growth rate (year^-1)
  Type r_fast = exp(log_r_fast);                // Fast-growing coral intrinsic growth rate (year^-1)
  Type K_slow = exp(log_K_slow);                // Slow-growing coral carrying capacity (%)
  Type K_fast = exp(log_K_fast);                // Fast-growing coral carrying capacity (%)
  Type temp_opt = exp(log_temp_opt);            // Optimal temperature for COTS (°C)
  Type temp_width = exp(log_temp_width);        // Temperature tolerance width for COTS (°C)
  Type temp_coral_threshold = exp(log_temp_coral_threshold);  // Temperature threshold for coral bleaching (°C)
  Type bleach_mort_slow = exp(log_bleach_mort_slow);  // Bleaching mortality rate for slow-growing coral (proportion)
  Type bleach_mort_fast = exp(log_bleach_mort_fast);  // Bleaching mortality rate for fast-growing coral (proportion)
  Type coral_dependency = exp(log_coral_dependency);  // COTS dependency on coral for survival (dimensionless)
  
  // Observation error standard deviations with minimum values
  Type sigma_cots = exp(log_sigma_cots) + Type(0.1);
  Type sigma_slow = exp(log_sigma_slow) + Type(1.0);
  Type sigma_fast = exp(log_sigma_fast) + Type(1.0);
  
  // PREDICTION VECTORS
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  
  // INITIALIZE PREDICTIONS WITH FIRST OBSERVATION
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // PROCESS MODEL
  for(int t = 1; t < n; t++) {
    // 1. COTS population dynamics - simple logistic growth with immigration
    Type cots_growth = r_cots * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1) / K_cots);
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t-1);
    cots_pred(t) = cots_pred(t) < Type(0.01) ? Type(0.01) : cots_pred(t);
    
    // 2. Slow-growing coral dynamics - logistic growth minus COTS predation
    Type slow_growth = r_slow * slow_pred(t-1) * (Type(1.0) - slow_pred(t-1) / K_slow);
    Type slow_predation = alpha_slow * cots_pred(t-1);
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_predation;
    slow_pred(t) = slow_pred(t) < Type(0.1) ? Type(0.1) : slow_pred(t);
    
    // 3. Fast-growing coral dynamics - logistic growth minus COTS predation
    Type fast_growth = r_fast * fast_pred(t-1) * (Type(1.0) - fast_pred(t-1) / K_fast);
    Type fast_predation = alpha_fast * pref_fast * cots_pred(t-1);
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_predation;
    fast_pred(t) = fast_pred(t) < Type(0.1) ? Type(0.1) : fast_pred(t);
  }
  
  // OBSERVATION MODEL
  Type nll = Type(0);
  
  // Normal likelihood for all variables
  for(int t = 0; t < n; t++) {
    nll -= dnorm(cots_dat(t), cots_pred(t), sigma_cots, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_slow, true);
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_fast, true);
  }
  
  // REPORTING SECTION
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(r_cots);
  REPORT(K_cots);
  REPORT(alpha_slow);
  REPORT(alpha_fast);
  REPORT(pref_fast);
  REPORT(r_slow);
  REPORT(r_fast);
  REPORT(K_slow);
  REPORT(K_fast);
  REPORT(temp_opt);
  REPORT(temp_width);
  REPORT(temp_coral_threshold);
  REPORT(bleach_mort_slow);
  REPORT(bleach_mort_fast);
  REPORT(coral_dependency);
  REPORT(sigma_cots);
  REPORT(sigma_slow);
  REPORT(sigma_fast);
  
  return nll;
}
