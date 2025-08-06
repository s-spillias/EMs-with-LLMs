#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA
  DATA_VECTOR(Year);                  // Years of observation
  DATA_VECTOR(cots_dat);              // Observed COTS abundance (individuals/m^2)
  DATA_VECTOR(slow_dat);              // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);              // Observed fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);               // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);           // COTS immigration rate (individuals/m^2/year)
  
  // PARAMETERS
  // COTS parameters
  PARAMETER(r_cots);                  // Intrinsic growth rate of COTS (year^-1)
  PARAMETER(K_cots);                  // Carrying capacity of COTS (individuals/m^2)
  PARAMETER(m_cots);                  // Natural mortality rate of COTS (year^-1)
  
  // Predation parameters
  PARAMETER(alpha_slow);              // Attack rate on slow-growing corals (m^2/individual/year)
  PARAMETER(alpha_fast);              // Attack rate on fast-growing corals (m^2/individual/year)
  PARAMETER(h_slow);                  // Half-saturation constant for slow-growing corals (%)
  PARAMETER(h_fast);                  // Half-saturation constant for fast-growing corals (%)
  PARAMETER(pref_fast);               // COTS preference for fast-growing corals (proportion)
  
  // Coral parameters
  PARAMETER(r_slow);                  // Intrinsic growth rate of slow-growing corals (year^-1)
  PARAMETER(r_fast);                  // Intrinsic growth rate of fast-growing corals (year^-1)
  PARAMETER(K_slow);                  // Carrying capacity of slow-growing corals (%)
  PARAMETER(K_fast);                  // Carrying capacity of fast-growing corals (%)
  
  // Temperature effect parameters
  PARAMETER(beta_cots_temp);          // Effect of temperature on COTS growth (per °C)
  PARAMETER(temp_opt_cots);           // Optimal temperature for COTS (°C)
  PARAMETER(beta_slow_temp);          // Effect of temperature on slow-growing coral growth (per °C)
  PARAMETER(beta_fast_temp);          // Effect of temperature on fast-growing coral growth (per °C)
  PARAMETER(temp_opt_coral);          // Optimal temperature for coral growth (°C)
  
  // New parameter for coral recovery inhibition during COTS outbreaks
  PARAMETER(cots_inhibit_threshold);  // COTS density threshold for inhibiting coral recovery (individuals/m^2)
  
  // Error parameters
  PARAMETER(sigma_obs_cots);          // Observation error SD for COTS
  PARAMETER(sigma_obs_slow);          // Observation error SD for slow-growing corals
  PARAMETER(sigma_obs_fast);          // Observation error SD for fast-growing corals
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Get data dimensions
  int n_years = Year.size();
  
  // Initialize vectors for model predictions
  vector<Type> cots_pred(n_years);
  vector<Type> slow_pred(n_years);
  vector<Type> fast_pred(n_years);
  
  // Initialize state variables with first observation
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Constants to prevent numerical issues
  Type min_val = Type(0.01);
  Type min_sd = Type(0.1);
  
  // Loop through time steps to calculate predictions and likelihood
  for (int t = 1; t < n_years; t++) {
    // Get previous state
    Type cots_t1 = cots_pred(t-1);
    Type slow_t1 = slow_pred(t-1);
    Type fast_t1 = fast_pred(t-1);
    Type sst = sst_dat(t-1);
    Type cotsimm = cotsimm_dat(t-1);
    
    // Ensure positive values for state variables
    cots_t1 = cots_t1 + min_val;
    slow_t1 = slow_t1 + min_val;
    fast_t1 = fast_t1 + min_val;
    
    // 1. Calculate temperature effects using Gaussian response curves
    Type temp_effect_cots = exp(-pow(sst - temp_opt_cots, 2) / (2.0));
    Type temp_effect_slow = exp(-pow(sst - temp_opt_coral, 2) / (2.0));
    Type temp_effect_fast = exp(-pow(sst - temp_opt_coral, 2) / (2.0));
    
    // 2. Calculate total coral resource availability
    Type total_coral = slow_t1 + fast_t1;
    
    // 3. Calculate COTS predation rates using functional responses
    // Ensure half-saturation constants are positive
    Type h_slow_pos = h_slow + min_val;
    Type h_fast_pos = h_fast + min_val;
    
    // Calculate predation with preference
    Type pref_fast_bounded = pref_fast;
    if (pref_fast_bounded < 0) pref_fast_bounded = 0;
    if (pref_fast_bounded > 1) pref_fast_bounded = 1;
    
    Type pred_slow = alpha_slow * cots_t1 * slow_t1 / (h_slow_pos + slow_t1) * (Type(1.0) - pref_fast_bounded);
    Type pred_fast = alpha_fast * cots_t1 * fast_t1 / (h_fast_pos + fast_t1) * pref_fast_bounded;
    
    // 4. Calculate resource limitation for COTS
    Type resource_limitation = Type(1.0) - exp(-Type(0.1) * total_coral);
    
    // 5. Calculate COTS population dynamics
    Type K_cots_pos = K_cots + min_val;
    Type r_cots_pos = r_cots;
    if (r_cots_pos < 0) r_cots_pos = 0;
    Type m_cots_pos = m_cots;
    if (m_cots_pos < 0) m_cots_pos = 0;
    
    Type cots_growth = r_cots_pos * cots_t1 * (Type(1.0) - cots_t1 / K_cots_pos) * temp_effect_cots * resource_limitation;
    Type cots_mort = m_cots_pos * cots_t1;
    Type cots_next = cots_t1 + cots_growth - cots_mort + cotsimm;
    
    // Ensure non-negative population
    if (cots_next < min_val) cots_next = min_val;
    
    // 6. Calculate coral recovery inhibition factor based on COTS density
    Type cots_inhibit_threshold_pos = cots_inhibit_threshold + min_val;
    
    // Sigmoid function for threshold-dependent inhibition
    Type cots_inhibition = Type(1.0) / (Type(1.0) + exp(-Type(5.0) * (cots_t1 - cots_inhibit_threshold_pos)));
    
    // Recovery inhibition factors
    Type recovery_factor_slow = Type(1.0) - Type(0.8) * cots_inhibition;
    Type recovery_factor_fast = Type(1.0) - Type(0.9) * cots_inhibition; // Fast-growing corals more affected
    
    // 7. Calculate coral dynamics with logistic growth, COTS predation, and recovery inhibition
    Type K_slow_pos = K_slow + min_val;
    Type K_fast_pos = K_fast + min_val;
    
    Type r_slow_pos = r_slow;
    if (r_slow_pos < 0) r_slow_pos = 0;
    
    Type r_fast_pos = r_fast;
    if (r_fast_pos < 0) r_fast_pos = 0;
    
    // Apply recovery inhibition to growth rates
    Type slow_growth = r_slow_pos * slow_t1 * (Type(1.0) - slow_t1 / K_slow_pos) * temp_effect_slow * recovery_factor_slow;
    Type slow_next = slow_t1 + slow_growth - pred_slow;
    
    // Ensure non-negative coral cover
    if (slow_next < min_val) slow_next = min_val;
    
    Type fast_growth = r_fast_pos * fast_t1 * (Type(1.0) - fast_t1 / K_fast_pos) * temp_effect_fast * recovery_factor_fast;
    Type fast_next = fast_t1 + fast_growth - pred_fast;
    
    // Ensure non-negative coral cover
    if (fast_next < min_val) fast_next = min_val;
    
    // 8. Set predictions for the current time step
    cots_pred(t) = cots_next;
    slow_pred(t) = slow_next;
    fast_pred(t) = fast_next;
  }
  
  // Calculate likelihood using log-normal observation model
  Type sigma_obs_cots_pos = sigma_obs_cots + min_sd;
  Type sigma_obs_slow_pos = sigma_obs_slow + min_sd;
  Type sigma_obs_fast_pos = sigma_obs_fast + min_sd;
  
  for (int t = 0; t < n_years; t++) {
    nll -= dnorm(log(cots_dat(t) + min_val), log(cots_pred(t) + min_val), sigma_obs_cots_pos, true);
    nll -= dnorm(log(slow_dat(t) + min_val), log(slow_pred(t) + min_val), sigma_obs_slow_pos, true);
    nll -= dnorm(log(fast_dat(t) + min_val), log(fast_pred(t) + min_val), sigma_obs_fast_pos, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
