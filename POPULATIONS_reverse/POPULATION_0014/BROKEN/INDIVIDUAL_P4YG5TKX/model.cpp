#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA SECTION
  DATA_VECTOR(Year);                  // Vector of years for time series data
  DATA_VECTOR(sst_dat);               // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);           // COTS immigration rate (individuals/m²/year)
  DATA_VECTOR(cots_dat);              // Observed COTS density (individuals/m²)
  DATA_VECTOR(slow_dat);              // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);              // Observed fast-growing coral cover (%)
  
  // PARAMETER SECTION
  PARAMETER(r_cots);                  // Intrinsic growth rate of COTS (year⁻¹)
  PARAMETER(K_cots);                  // Carrying capacity of COTS (individuals/m²)
  PARAMETER(temp_opt_cots);           // Optimal temperature for COTS growth (°C)
  PARAMETER(temp_range_cots);         // Temperature tolerance range for COTS (°C)
  PARAMETER(mort_cots);               // Natural mortality rate of COTS (year⁻¹)
  PARAMETER(coral_dep_mort);          // Coral-dependent mortality coefficient for COTS
  
  PARAMETER(r_slow);                  // Intrinsic growth rate of slow-growing corals (year⁻¹)
  PARAMETER(K_slow);                  // Carrying capacity of slow-growing corals (%)
  PARAMETER(temp_opt_slow);           // Optimal temperature for slow-growing corals (°C)
  PARAMETER(temp_range_slow);         // Temperature tolerance range for slow-growing corals (°C)
  
  PARAMETER(r_fast);                  // Intrinsic growth rate of fast-growing corals (year⁻¹)
  PARAMETER(K_fast);                  // Carrying capacity of fast-growing corals (%)
  PARAMETER(temp_opt_fast);           // Optimal temperature for fast-growing corals (°C)
  PARAMETER(temp_range_fast);         // Temperature tolerance range for fast-growing corals (°C)
  
  PARAMETER(alpha_slow);              // COTS feeding rate on slow-growing corals (% cover consumed per COTS/m²/year)
  PARAMETER(alpha_fast);              // COTS feeding rate on fast-growing corals (% cover consumed per COTS/m²/year)
  PARAMETER(pref_fast);               // COTS preference for fast-growing corals (dimensionless)
  PARAMETER(comp_coef);               // Competition coefficient between coral types (dimensionless)
  
  PARAMETER(log_sigma_cots);          // Log of observation error SD for COTS
  PARAMETER(log_sigma_slow);          // Log of observation error SD for slow-growing corals
  PARAMETER(log_sigma_fast);          // Log of observation error SD for fast-growing corals
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize vectors for model predictions
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  
  // Set initial values for first time step
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Time series simulation
  for(int t = 1; t < n; t++) {
    // 1. COTS population dynamics (basic logistic growth)
    cots_pred(t) = cots_pred(t-1) * (1.0 + r_cots * (1.0 - cots_pred(t-1) / K_cots)) + cotsimm_dat(t-1);
    if (cots_pred(t) < 0.0) cots_pred(t) = 0.0;
    
    // 2. Coral dynamics (basic logistic growth with COTS predation)
    slow_pred(t) = slow_pred(t-1) * (1.0 + r_slow * (1.0 - slow_pred(t-1) / K_slow)) - alpha_slow * cots_pred(t-1);
    if (slow_pred(t) < 0.0) slow_pred(t) = 0.0;
    
    fast_pred(t) = fast_pred(t-1) * (1.0 + r_fast * (1.0 - fast_pred(t-1) / K_fast)) - alpha_fast * pref_fast * cots_pred(t-1);
    if (fast_pred(t) < 0.0) fast_pred(t) = 0.0;
  }
  
  // Fixed standard deviations to avoid optimization issues
  Type sigma_cots_fixed = 0.3;
  Type sigma_slow_fixed = 0.3;
  Type sigma_fast_fixed = 0.3;
  
  // Calculate negative log-likelihood using normal distribution on log-transformed data
  for(int t = 0; t < n; t++) {
    // Add small constant to prevent log of zero
    Type eps = 0.01;
    
    // COTS likelihood
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots_fixed, true);
    
    // Slow-growing coral likelihood
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow_fixed, true);
    
    // Fast-growing coral likelihood
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast_fixed, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
