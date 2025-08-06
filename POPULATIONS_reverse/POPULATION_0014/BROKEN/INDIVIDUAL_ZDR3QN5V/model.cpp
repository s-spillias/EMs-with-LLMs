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
  
  // Coral competition parameters
  PARAMETER(comp_effect);             // Strength of competition between coral types
  
  // NEW: Temperature-dependent vulnerability parameters
  PARAMETER(vuln_temp_threshold);     // Temperature threshold above which vulnerability increases (°C)
  PARAMETER(vuln_slow_coef);          // Vulnerability coefficient for slow-growing corals
  PARAMETER(vuln_fast_coef);          // Vulnerability coefficient for fast-growing corals
  
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
  
  // Add first observations to likelihood with fixed SD
  Type fixed_sd = 2.0;
  nll -= dnorm(cots_dat(0), cots_pred(0), fixed_sd, true);
  nll -= dnorm(slow_dat(0), slow_pred(0), fixed_sd, true);
  nll -= dnorm(fast_dat(0), fast_pred(0), fixed_sd, true);
  
  // Loop through time steps to calculate predictions and likelihood
  for (int t = 1; t < n_years; t++) {
    // Get previous state
    Type cots_t1 = cots_pred(t-1);
    Type slow_t1 = slow_pred(t-1);
    Type fast_t1 = fast_pred(t-1);
    Type cotsimm = cotsimm_dat(t-1);
    Type sst = sst_dat(t-1);  // Get temperature from previous time step
    
    // Ensure non-negative state variables
    Type min_val = 1.0;
    if (cots_t1 < min_val) cots_t1 = min_val;
    if (slow_t1 < min_val) slow_t1 = min_val;
    if (fast_t1 < min_val) fast_t1 = min_val;
    
    // 1. Calculate temperature-dependent vulnerability factors
    // Calculate temperature stress above threshold
    Type temp_stress = 0.0;
    if (sst > vuln_temp_threshold) {
      temp_stress = sst - vuln_temp_threshold;
      if (temp_stress > 2.0) temp_stress = 2.0;  // Cap maximum temperature stress effect
    }
    
    // Calculate vulnerability multipliers (linear response to temperature)
    Type vuln_slow_factor = 1.0 + vuln_slow_coef * temp_stress;
    Type vuln_fast_factor = 1.0 + vuln_fast_coef * temp_stress;
    
    // 2. Calculate predation rates with temperature-dependent vulnerability
    // Modified to use total coral in denominator for stability
    Type total_coral = slow_t1 + fast_t1;
    Type pred_slow = alpha_slow * cots_t1 * slow_t1 / (h_slow + total_coral) * (1.0 - pref_fast) * vuln_slow_factor;
    Type pred_fast = alpha_fast * cots_t1 * fast_t1 / (h_fast + total_coral) * pref_fast * vuln_fast_factor;
    
    // Ensure predation doesn't exceed available coral
    if (pred_slow > 0.25 * slow_t1) pred_slow = 0.25 * slow_t1;
    if (pred_fast > 0.25 * fast_t1) pred_fast = 0.25 * fast_t1;
    
    // 3. Calculate COTS population dynamics
    // Calculate temperature effect on COTS growth - simplified linear response
    Type temp_diff = sst - temp_opt_cots;
    Type temp_effect_cots = 1.0 - 0.05 * fabs(temp_diff);
    if (temp_effect_cots < 0.7) temp_effect_cots = 0.7;
    if (temp_effect_cots > 1.0) temp_effect_cots = 1.0;
    
    // Calculate COTS growth with density dependence
    Type dd_term = 1.0 - cots_t1 / K_cots;
    if (dd_term < 0.0) dd_term = 0.0;
    
    Type cots_growth = r_cots * cots_t1 * dd_term * temp_effect_cots;
    
    // Calculate mortality and next state
    Type cots_mort = m_cots * cots_t1;
    Type cots_next = cots_t1 + cots_growth - cots_mort + cotsimm;
    if (cots_next < min_val) cots_next = min_val;
    if (cots_next > 5.0) cots_next = 5.0;  // Add upper bound for stability
    
    // 4. Calculate coral dynamics with temperature effects
    // Calculate temperature effects on coral growth - simplified linear response
    Type temp_diff_coral = sst - temp_opt_coral;
    Type temp_effect_slow = 1.0 - 0.05 * fabs(temp_diff_coral);
    Type temp_effect_fast = 1.0 - 0.08 * fabs(temp_diff_coral);
    
    // Limit temperature effects
    if (temp_effect_slow < 0.7) temp_effect_slow = 0.7;
    if (temp_effect_slow > 1.0) temp_effect_slow = 1.0;
    if (temp_effect_fast < 0.6) temp_effect_fast = 0.6;
    if (temp_effect_fast > 1.0) temp_effect_fast = 1.0;
    
    // Calculate competition terms with improved stability
    Type slow_competition = (slow_t1 + comp_effect * fast_t1) / (K_slow + 10.0);
    if (slow_competition > 0.8) slow_competition = 0.8;
    
    Type fast_competition = (fast_t1 + comp_effect * slow_t1) / (K_fast + 10.0);
    if (fast_competition > 0.8) fast_competition = 0.8;
    
    // Calculate coral growth
    Type slow_growth = r_slow * slow_t1 * (1.0 - slow_competition) * temp_effect_slow;
    Type fast_growth = r_fast * fast_t1 * (1.0 - fast_competition) * temp_effect_fast;
    
    // Calculate next state
    Type slow_next = slow_t1 + slow_growth - pred_slow;
    Type fast_next = fast_t1 + fast_growth - pred_fast;
    
    // Ensure non-negative values and add upper bounds
    if (slow_next < min_val) slow_next = min_val;
    if (fast_next < min_val) fast_next = min_val;
    if (slow_next > K_slow * 0.9) slow_next = K_slow * 0.9;
    if (fast_next > K_fast * 0.9) fast_next = K_fast * 0.9;
    
    // 5. Set predictions for the current time step
    cots_pred(t) = cots_next;
    slow_pred(t) = slow_next;
    fast_pred(t) = fast_next;
    
    // 6. Add to negative log-likelihood (using normal observation model)
    // Use fixed minimum SD to prevent numerical issues
    Type min_sd_obs = 2.0;
    Type cots_sd = sigma_obs_cots;
    Type slow_sd = sigma_obs_slow;
    Type fast_sd = sigma_obs_fast;
    
    if (cots_sd < min_sd_obs) cots_sd = min_sd_obs;
    if (slow_sd < min_sd_obs) slow_sd = min_sd_obs;
    if (fast_sd < min_sd_obs) fast_sd = min_sd_obs;
    
    nll -= dnorm(cots_dat(t), cots_pred(t), cots_sd, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), slow_sd, true);
    nll -= dnorm(fast_dat(t), fast_pred(t), fast_sd, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
