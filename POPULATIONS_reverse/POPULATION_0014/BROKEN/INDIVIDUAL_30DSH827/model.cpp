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
  PARAMETER(cots_density_effect);     // Effect of COTS density on predation efficiency
  
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
  
  // Add first observations to likelihood using squared error instead of dnorm
  Type sigma_cots = 0.3;
  Type sigma_slow = 0.2;
  Type sigma_fast = 0.25;
  
  // Simple squared error for first observation
  nll += 0.5 * pow((log(cots_dat(0) + min_val) - log(cots_pred(0) + min_val)) / sigma_cots, 2);
  nll += 0.5 * pow((log(slow_dat(0) + min_val) - log(slow_pred(0) + min_val)) / sigma_slow, 2);
  nll += 0.5 * pow((log(fast_dat(0) + min_val) - log(fast_pred(0) + min_val)) / sigma_fast, 2);
  
  // Loop through time steps to calculate predictions and likelihood
  for (int t = 1; t < n_years; t++) {
    // Get previous state
    Type cots_t1 = cots_pred(t-1);
    Type slow_t1 = slow_pred(t-1);
    Type fast_t1 = fast_pred(t-1);
    Type sst = sst_dat(t-1);
    Type cotsimm = cotsimm_dat(t-1);
    
    // Ensure non-negative values for state variables
    if (cots_t1 < min_val) cots_t1 = min_val;
    if (slow_t1 < min_val) slow_t1 = min_val;
    if (fast_t1 < min_val) fast_t1 = min_val;
    
    // 1. Calculate temperature effects (simplified)
    Type temp_effect_cots = 1.0;
    Type temp_effect_slow = 1.0;
    Type temp_effect_fast = 1.0;
    
    // 2. Calculate density-dependent predation efficiency
    Type predation_multiplier = 1.0;
    
    // Only apply density effect if COTS density is above threshold
    Type density_threshold = 1.0;
    if (cots_t1 > density_threshold) {
      predation_multiplier = 1.0 + cots_density_effect * (cots_t1 - density_threshold) / (cots_t1 + 1.0);
      if (predation_multiplier > 3.0) predation_multiplier = 3.0;
    }
    
    // 3. Calculate COTS predation rates
    // Ensure positive half-saturation constants
    Type h_slow_pos = h_slow;
    if (h_slow_pos < 0.1) h_slow_pos = 0.1;
    
    Type h_fast_pos = h_fast;
    if (h_fast_pos < 0.1) h_fast_pos = 0.1;
    
    // Bound preference parameter between 0 and 1
    Type pref_fast_bounded = pref_fast;
    if (pref_fast_bounded < 0.0) pref_fast_bounded = 0.0;
    if (pref_fast_bounded > 1.0) pref_fast_bounded = 1.0;
    
    // Calculate predation
    Type pred_slow = alpha_slow * cots_t1 * slow_t1 / (h_slow_pos + slow_t1) * (1.0 - pref_fast_bounded) * predation_multiplier;
    Type pred_fast = alpha_fast * cots_t1 * fast_t1 / (h_fast_pos + fast_t1) * pref_fast_bounded * predation_multiplier;
    
    // Ensure predation doesn't exceed available coral
    if (pred_slow > 0.9 * slow_t1) pred_slow = 0.9 * slow_t1;
    if (pred_fast > 0.9 * fast_t1) pred_fast = 0.9 * fast_t1;
    
    // 4. Calculate resource limitation for COTS
    Type total_coral = slow_t1 + fast_t1;
    Type resource_limitation = 1.0 - exp(-0.1 * total_coral);
    if (resource_limitation < 0.1) resource_limitation = 0.1;
    
    // 5. Calculate COTS population dynamics
    // Ensure positive carrying capacity
    Type K_cots_pos = K_cots;
    if (K_cots_pos < 0.1) K_cots_pos = 0.1;
    
    // Calculate growth
    Type cots_growth = r_cots * cots_t1 * (1.0 - cots_t1 / K_cots_pos) * temp_effect_cots * resource_limitation;
    Type cots_mort = m_cots * cots_t1;
    
    // Ensure mortality doesn't exceed population
    if (cots_mort > 0.9 * cots_t1) cots_mort = 0.9 * cots_t1;
    
    // Calculate next state
    Type cots_next = cots_t1 + cots_growth - cots_mort + cotsimm;
    if (cots_next < min_val) cots_next = min_val;
    
    // 6. Calculate coral dynamics
    // Ensure positive carrying capacities
    Type K_slow_pos = K_slow;
    if (K_slow_pos < 0.1) K_slow_pos = 0.1;
    
    Type K_fast_pos = K_fast;
    if (K_fast_pos < 0.1) K_fast_pos = 0.1;
    
    // Calculate growth
    Type slow_growth = r_slow * slow_t1 * (1.0 - slow_t1 / K_slow_pos) * temp_effect_slow;
    Type fast_growth = r_fast * fast_t1 * (1.0 - fast_t1 / K_fast_pos) * temp_effect_fast;
    
    // Calculate next state
    Type slow_next = slow_t1 + slow_growth - pred_slow;
    Type fast_next = fast_t1 + fast_growth - pred_fast;
    
    // Ensure non-negative cover
    if (slow_next < min_val) slow_next = min_val;
    if (fast_next < min_val) fast_next = min_val;
    
    // 7. Set predictions for the current time step
    cots_pred(t) = cots_next;
    slow_pred(t) = slow_next;
    fast_pred(t) = fast_next;
    
    // 8. Add to negative log-likelihood using squared error instead of dnorm
    nll += 0.5 * pow((log(cots_dat(t) + min_val) - log(cots_pred(t) + min_val)) / sigma_cots, 2);
    nll += 0.5 * pow((log(slow_dat(t) + min_val) - log(slow_pred(t) + min_val)) / sigma_slow, 2);
    nll += 0.5 * pow((log(fast_dat(t) + min_val) - log(fast_pred(t) + min_val)) / sigma_fast, 2);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
