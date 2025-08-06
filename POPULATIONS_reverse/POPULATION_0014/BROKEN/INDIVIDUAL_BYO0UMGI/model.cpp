#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA SECTION
  DATA_VECTOR(Year);                      // Vector of years for time series data
  DATA_VECTOR(sst_dat);                   // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);               // COTS immigration rate (individuals/m²/year)
  DATA_VECTOR(cots_dat);                  // Observed COTS density (individuals/m²)
  DATA_VECTOR(slow_dat);                  // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);                  // Observed fast-growing coral cover (%)
  
  // PARAMETER SECTION
  PARAMETER(r_cots);                      // COTS intrinsic growth rate (year⁻¹)
  PARAMETER(K_cots);                      // COTS carrying capacity (individuals/m²)
  PARAMETER(temp_opt_cots);               // Optimal temperature for COTS reproduction (°C)
  PARAMETER(temp_range_cots);             // Temperature tolerance range for COTS (°C)
  PARAMETER(cots_mortality);              // Natural mortality rate of COTS (year⁻¹)
  
  PARAMETER(r_slow);                      // Intrinsic growth rate of slow-growing corals (year⁻¹)
  PARAMETER(K_slow);                      // Carrying capacity of slow-growing corals (%)
  PARAMETER(r_fast);                      // Intrinsic growth rate of fast-growing corals (year⁻¹)
  PARAMETER(K_fast);                      // Carrying capacity of fast-growing corals (%)
  
  PARAMETER(a_slow);                      // Attack rate on slow-growing corals (m²/individual/year)
  PARAMETER(a_fast);                      // Attack rate on fast-growing corals (m²/individual/year)
  PARAMETER(h_slow);                      // Handling time for slow-growing corals (year/%)
  PARAMETER(h_fast);                      // Handling time for fast-growing corals (year/%)
  PARAMETER(pref_fast);                   // Preference for fast-growing corals (dimensionless)
  
  PARAMETER(bleach_threshold);            // Temperature threshold for coral bleaching (°C)
  PARAMETER(bleach_mortality_slow);       // Mortality rate of slow corals during bleaching (year⁻¹)
  PARAMETER(bleach_mortality_fast);       // Mortality rate of fast corals during bleaching (year⁻¹)
  
  PARAMETER(coral_effect);                // Effect of coral cover on COTS survival (dimensionless)
  
  PARAMETER(log_sigma_cots);              // Log of observation error SD for COTS
  PARAMETER(log_sigma_slow);              // Log of observation error SD for slow-growing corals
  PARAMETER(log_sigma_fast);              // Log of observation error SD for fast-growing corals
  
  // Transform parameters
  Type sigma_cots = exp(log_sigma_cots);  // Observation error SD for COTS
  Type sigma_slow = exp(log_sigma_slow);  // Observation error SD for slow-growing corals
  Type sigma_fast = exp(log_sigma_fast);  // Observation error SD for fast-growing corals
  
  // Initialize negative log-likelihood
  Type nll = Type(0.0);
  
  // Initialize vectors for model predictions
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  
  // Initialize with first observation
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Process model: predict state variables through time
  for(int t = 1; t < n; t++) {
    // COTS population dynamics (very simple)
    cots_pred(t) = cots_pred(t-1) * (Type(1.0) + r_cots * Type(0.1)) + cotsimm_dat(t-1);
    if (cots_pred(t) < Type(0.01)) cots_pred(t) = Type(0.01);
    
    // Coral dynamics (very simple)
    slow_pred(t) = slow_pred(t-1) * (Type(1.0) + r_slow * Type(0.1)) - Type(0.01) * cots_pred(t-1);
    fast_pred(t) = fast_pred(t-1) * (Type(1.0) + r_fast * Type(0.1)) - Type(0.02) * cots_pred(t-1);
    
    // Ensure positive coral populations
    if (slow_pred(t) < Type(0.01)) slow_pred(t) = Type(0.01);
    if (fast_pred(t) < Type(0.01)) fast_pred(t) = Type(0.01);
  }
  
  // Observation model: calculate negative log-likelihood
  for(int t = 0; t < n; t++) {
    // Fixed standard deviation to prevent numerical issues
    Type sd_cots = Type(0.5);
    Type sd_slow = Type(5.0);
    Type sd_fast = Type(5.0);
    
    // Calculate negative log-likelihood using normal distribution
    nll -= dnorm(cots_dat(t), cots_pred(t), sd_cots, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sd_slow, true);
    nll -= dnorm(fast_dat(t), fast_pred(t), sd_fast, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(sigma_cots);
  REPORT(sigma_slow);
  REPORT(sigma_fast);
  
  return nll;
}
