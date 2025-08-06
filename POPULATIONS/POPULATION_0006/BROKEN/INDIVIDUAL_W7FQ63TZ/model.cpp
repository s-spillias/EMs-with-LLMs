#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA SECTION
  DATA_VECTOR(Year);                      // Year of observation
  DATA_VECTOR(sst_dat);                   // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);               // COTS larval immigration rate (individuals/m²/year)
  DATA_VECTOR(cots_dat);                  // Observed COTS density (individuals/m²)
  DATA_VECTOR(fast_dat);                  // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);                  // Observed slow-growing coral cover (%)
  
  // PARAMETER SECTION
  PARAMETER(log_r_cots);                  // Log of COTS intrinsic growth rate (year⁻¹)
  PARAMETER(log_K_cots);                  // Log of COTS carrying capacity (individuals/m²)
  PARAMETER(log_temp_opt_cots);           // Log of optimal temperature for COTS reproduction (°C)
  PARAMETER(log_temp_range_cots);         // Log of temperature tolerance range for COTS (°C)
  PARAMETER(log_imm_effect);              // Log of effect of larval immigration on COTS population (dimensionless)
  
  PARAMETER(log_a_fast);                  // Log of attack rate on fast-growing coral (m²/individual/year)
  PARAMETER(log_a_slow);                  // Log of attack rate on slow-growing coral (m²/individual/year)
  PARAMETER(log_h_fast);                  // Log of handling time for fast-growing coral (year/%)
  PARAMETER(log_h_slow);                  // Log of handling time for slow-growing coral (year/%)
  PARAMETER(log_pref_fast);               // Log of preference for fast-growing coral (dimensionless)
  
  PARAMETER(log_r_fast);                  // Log of intrinsic growth rate of fast-growing coral (year⁻¹)
  PARAMETER(log_r_slow);                  // Log of intrinsic growth rate of slow-growing coral (year⁻¹)
  PARAMETER(log_K_fast);                  // Log of carrying capacity of fast-growing coral (%)
  PARAMETER(log_K_slow);                  // Log of carrying capacity of slow-growing coral (%)
  PARAMETER(log_temp_opt_coral);          // Log of optimal temperature for coral growth (°C)
  PARAMETER(log_temp_range_coral);        // Log of temperature tolerance range for coral (°C)
  PARAMETER(log_coral_recovery);          // Log of recovery rate of coral after disturbance (year⁻¹)
  
  PARAMETER(log_sd_cots);                 // Log of observation error SD for COTS
  PARAMETER(log_sd_fast);                 // Log of observation error SD for fast-growing coral
  PARAMETER(log_sd_slow);                 // Log of observation error SD for slow-growing coral
  
  // Initialize negative log-likelihood
  Type nll = Type(0.0);
  
  // Initialize vectors for predictions
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Set initial values for first time step
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Process model: predict state variables through time
  for(int t = 1; t < n; t++) {
    // 1. COTS population dynamics with immigration
    Type cots_growth = Type(0.5) * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1) / Type(2.0));
    Type immigration = Type(1.0) * cotsimm_dat(t-1);
    
    cots_pred(t) = cots_pred(t-1) + cots_growth + immigration;
    cots_pred(t) = cots_pred(t) < Type(0.0) ? Type(0.0) : cots_pred(t);
    
    // 2. Calculate coral consumption
    Type fast_consumption = Type(0.3) * Type(0.7) * cots_pred(t-1);
    Type slow_consumption = Type(0.2) * Type(0.3) * cots_pred(t-1);
    
    // Ensure consumption doesn't exceed available coral
    fast_consumption = fast_consumption > fast_pred(t-1) ? fast_pred(t-1) : fast_consumption;
    slow_consumption = slow_consumption > slow_pred(t-1) ? slow_pred(t-1) : slow_consumption;
    
    // 3. Coral dynamics with logistic growth and COTS predation
    Type fast_growth = Type(0.8) * fast_pred(t-1) * (Type(1.0) - fast_pred(t-1) / Type(50.0));
    Type slow_growth = Type(0.3) * slow_pred(t-1) * (Type(1.0) - slow_pred(t-1) / Type(40.0));
    
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_consumption;
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_consumption;
    
    // Ensure coral cover stays within reasonable bounds
    fast_pred(t) = fast_pred(t) < Type(0.0) ? Type(0.0) : fast_pred(t);
    fast_pred(t) = fast_pred(t) > Type(100.0) ? Type(100.0) : fast_pred(t);
    slow_pred(t) = slow_pred(t) < Type(0.0) ? Type(0.0) : slow_pred(t);
    slow_pred(t) = slow_pred(t) > Type(100.0) ? Type(100.0) : slow_pred(t);
  }
  
  // Observation model: calculate negative log-likelihood
  for(int t = 0; t < n; t++) {
    // Add small constant to prevent issues with zeros
    Type cots_obs = cots_dat(t) + Type(0.001);
    Type cots_model = cots_pred(t) + Type(0.001);
    Type fast_obs = fast_dat(t) + Type(0.001);
    Type fast_model = fast_pred(t) + Type(0.001);
    Type slow_obs = slow_dat(t) + Type(0.001);
    Type slow_model = slow_pred(t) + Type(0.001);
    
    // Calculate negative log-likelihood
    nll -= dnorm(cots_obs, cots_model, Type(0.3), true);
    nll -= dnorm(fast_obs, fast_model, Type(0.3), true);
    nll -= dnorm(slow_obs, slow_model, Type(0.3), true);
  }
  
  // Report predictions and parameters
  REPORT(nll);
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  // Report fixed parameter values
  Type r_cots = Type(0.5);
  Type K_cots = Type(2.0);
  Type imm_effect = Type(1.0);
  Type a_fast = Type(0.3);
  Type a_slow = Type(0.2);
  Type pref_fast = Type(0.7);
  Type r_fast = Type(0.8);
  Type r_slow = Type(0.3);
  Type K_fast = Type(50.0);
  Type K_slow = Type(40.0);
  
  REPORT(r_cots);
  REPORT(K_cots);
  REPORT(imm_effect);
  REPORT(a_fast);
  REPORT(a_slow);
  REPORT(pref_fast);
  REPORT(r_fast);
  REPORT(r_slow);
  REPORT(K_fast);
  REPORT(K_slow);
  
  // Return negative log-likelihood
  return nll;
}
