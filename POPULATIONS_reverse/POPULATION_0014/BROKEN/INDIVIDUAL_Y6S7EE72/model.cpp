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
  
  // New parameter for coral-COTS feedback
  PARAMETER(coral_recruit_effect);    // Effect of coral cover on COTS recruitment success
  
  // Error parameters
  PARAMETER(sigma_proc_cots);         // Process error SD for COTS
  PARAMETER(sigma_proc_slow);         // Process error SD for slow-growing corals
  PARAMETER(sigma_proc_fast);         // Process error SD for fast-growing corals
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
  Type min_val = 0.01;
  
  // Ensure parameters are in valid ranges
  Type r_cots_safe = r_cots < 0.01 ? 0.01 : r_cots;
  Type K_cots_safe = K_cots < 0.1 ? 0.1 : K_cots;
  Type m_cots_safe = m_cots < 0.01 ? 0.01 : m_cots;
  Type h_slow_safe = h_slow < 1.0 ? 1.0 : h_slow;
  Type h_fast_safe = h_fast < 1.0 ? 1.0 : h_fast;
  Type pref_bounded = pref_fast < 0.0 ? 0.0 : (pref_fast > 1.0 ? 1.0 : pref_fast);
  Type r_slow_safe = r_slow < 0.01 ? 0.01 : r_slow;
  Type r_fast_safe = r_fast < 0.01 ? 0.01 : r_fast;
  Type K_slow_safe = K_slow < 1.0 ? 1.0 : K_slow;
  Type K_fast_safe = K_fast < 1.0 ? 1.0 : K_fast;
  Type coral_effect_safe = coral_recruit_effect < 0.0 ? 0.0 : coral_recruit_effect;
  
  // Ensure observation error parameters are positive
  Type sigma_obs_cots_safe = sigma_obs_cots < 0.1 ? 0.1 : sigma_obs_cots;
  Type sigma_obs_slow_safe = sigma_obs_slow < 0.1 ? 0.1 : sigma_obs_slow;
  Type sigma_obs_fast_safe = sigma_obs_fast < 0.1 ? 0.1 : sigma_obs_fast;
  
  // Add first observations to likelihood
  nll -= dnorm(log(cots_dat(0) + min_val), log(cots_pred(0) + min_val), sigma_obs_cots_safe, true);
  nll -= dnorm(log(slow_dat(0) + min_val), log(slow_pred(0) + min_val), sigma_obs_slow_safe, true);
  nll -= dnorm(log(fast_dat(0) + min_val), log(fast_pred(0) + min_val), sigma_obs_fast_safe, true);
  
  // Loop through time steps to calculate predictions and likelihood
  for (int t = 1; t < n_years; t++) {
    // Get previous state with safety bounds
    Type cots_t1 = cots_pred(t-1) < min_val ? min_val : cots_pred(t-1);
    Type slow_t1 = slow_pred(t-1) < min_val ? min_val : slow_pred(t-1);
    Type fast_t1 = fast_pred(t-1) < min_val ? min_val : fast_pred(t-1);
    
    // Get environmental variables
    Type sst = sst_dat(t-1);
    Type cotsimm = cotsimm_dat(t-1) < 0.0 ? 0.0 : cotsimm_dat(t-1);
    
    // Calculate total coral cover
    Type total_coral = slow_t1 + fast_t1;
    
    // Simplified temperature effects
    Type temp_diff_cots = sst - temp_opt_cots;
    Type temp_diff_coral = sst - temp_opt_coral;
    
    Type temp_effect_cots = 1.0 - 0.01 * temp_diff_cots * temp_diff_cots;
    temp_effect_cots = temp_effect_cots < 0.2 ? 0.2 : (temp_effect_cots > 1.0 ? 1.0 : temp_effect_cots);
    
    Type temp_effect_slow = 1.0 - 0.01 * temp_diff_coral * temp_diff_coral;
    temp_effect_slow = temp_effect_slow < 0.2 ? 0.2 : (temp_effect_slow > 1.0 ? 1.0 : temp_effect_slow);
    
    Type temp_effect_fast = 1.0 - 0.01 * temp_diff_coral * temp_diff_coral;
    temp_effect_fast = temp_effect_fast < 0.2 ? 0.2 : (temp_effect_fast > 1.0 ? 1.0 : temp_effect_fast);
    
    // Calculate predation rates with safety bounds
    Type pred_slow = alpha_slow * cots_t1 * slow_t1 / (h_slow_safe + slow_t1) * (1.0 - pref_bounded);
    pred_slow = pred_slow > slow_t1 * 0.9 ? slow_t1 * 0.9 : pred_slow;
    
    Type pred_fast = alpha_fast * cots_t1 * fast_t1 / (h_fast_safe + fast_t1) * pref_bounded;
    pred_fast = pred_fast > fast_t1 * 0.9 ? fast_t1 * 0.9 : pred_fast;
    
    // Resource limitation for COTS (simplified)
    Type resource_limitation = 1.0 - exp(-0.05 * total_coral);
    
    // NEW: Coral-facilitated recruitment effect
    // This creates a feedback where higher coral cover increases COTS recruitment success
    Type coral_recruit_factor = 1.0 + coral_effect_safe * total_coral / (20.0 + total_coral);
    
    // COTS population dynamics with coral-facilitated recruitment
    Type density_effect = cots_t1 / K_cots_safe > 5.0 ? 5.0 : cots_t1 / K_cots_safe;
    Type cots_growth = r_cots_safe * cots_t1 * (1.0 - density_effect) * temp_effect_cots * resource_limitation * coral_recruit_factor;
    Type cots_mort = m_cots_safe * cots_t1;
    
    // Calculate next state for COTS
    Type cots_next = cots_t1 + cots_growth - cots_mort + cotsimm;
    cots_next = cots_next < min_val ? min_val : cots_next;
    
    // Calculate coral dynamics
    Type slow_growth = r_slow_safe * slow_t1 * (1.0 - slow_t1 / K_slow_safe) * temp_effect_slow;
    Type slow_next = slow_t1 + slow_growth - pred_slow;
    slow_next = slow_next < min_val ? min_val : slow_next;
    
    Type fast_growth = r_fast_safe * fast_t1 * (1.0 - fast_t1 / K_fast_safe) * temp_effect_fast;
    Type fast_next = fast_t1 + fast_growth - pred_fast;
    fast_next = fast_next < min_val ? min_val : fast_next;
    
    // Set predictions for the current time step
    cots_pred(t) = cots_next;
    slow_pred(t) = slow_next;
    fast_pred(t) = fast_next;
    
    // Add to negative log-likelihood with robust error handling
    nll -= dnorm(log(cots_dat(t) + min_val), log(cots_pred(t) + min_val), sigma_obs_cots_safe, true);
    nll -= dnorm(log(slow_dat(t) + min_val), log(slow_pred(t) + min_val), sigma_obs_slow_safe, true);
    nll -= dnorm(log(fast_dat(t) + min_val), log(fast_pred(t) + min_val), sigma_obs_fast_safe, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
