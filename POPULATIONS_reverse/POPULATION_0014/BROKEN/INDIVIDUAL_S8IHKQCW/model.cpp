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
  
  // New parameters for temperature tolerance
  PARAMETER(temp_tolerance_slow);     // Temperature tolerance range for slow-growing corals (°C)
  PARAMETER(temp_tolerance_fast);     // Temperature tolerance range for fast-growing corals (°C)
  
  // Coral competition parameters
  PARAMETER(comp_effect);             // Strength of competition between coral types
  
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
  
  // Add first observations to likelihood
  Type min_sd = 0.3;
  nll -= dnorm(log(cots_dat(0) + 0.1), log(cots_pred(0) + 0.1), min_sd, true);
  nll -= dnorm(log(slow_dat(0) + 0.1), log(slow_pred(0) + 0.1), min_sd, true);
  nll -= dnorm(log(fast_dat(0) + 0.1), log(fast_pred(0) + 0.1), min_sd, true);
  
  // Loop through time steps to calculate predictions and likelihood
  for (int t = 1; t < n_years; t++) {
    // Get previous state
    Type cots_t1 = cots_pred(t-1);
    Type slow_t1 = slow_pred(t-1);
    Type fast_t1 = fast_pred(t-1);
    Type cotsimm = cotsimm_dat(t-1);
    Type sst = sst_dat(t-1);
    
    // Ensure non-negative state variables
    Type min_val = 0.1;
    cots_t1 = (cots_t1 < min_val) ? min_val : cots_t1;
    slow_t1 = (slow_t1 < min_val) ? min_val : slow_t1;
    fast_t1 = (fast_t1 < min_val) ? min_val : fast_t1;
    
    // 1. Calculate total coral resource availability
    Type total_coral = slow_t1 + fast_t1;
    
    // 2. Calculate predation rates with functional response
    Type pred_slow = alpha_slow * cots_t1 * slow_t1 / (h_slow + slow_t1) * (1.0 - pref_fast);
    Type pred_fast = alpha_fast * cots_t1 * fast_t1 / (h_fast + fast_t1) * pref_fast;
    
    // Ensure predation doesn't exceed available coral
    pred_slow = (pred_slow > 0.5 * slow_t1) ? 0.5 * slow_t1 : pred_slow;
    pred_fast = (pred_fast > 0.5 * fast_t1) ? 0.5 * fast_t1 : pred_fast;
    
    // 3. Calculate COTS population dynamics
    // Calculate predation benefit
    Type total_pred = pred_slow + pred_fast;
    Type pred_benefit = 0.1 * total_pred / (total_coral + 10.0);
    
    // Calculate temperature effect on COTS
    Type temp_diff_cots = sst - temp_opt_cots;
    Type temp_effect_cots = exp(-0.25 * temp_diff_cots * temp_diff_cots);
    
    // Calculate COTS growth with density dependence and temperature effect
    Type dd_term = 1.0 - cots_t1 / K_cots;
    dd_term = (dd_term < -0.5) ? -0.5 : dd_term;
    
    Type cots_growth = r_cots * cots_t1 * dd_term * temp_effect_cots * (1.0 + pred_benefit);
    
    // Calculate mortality and next state
    Type cots_mort = m_cots * cots_t1;
    Type cots_next = cots_t1 + cots_growth - cots_mort + cotsimm;
    cots_next = (cots_next < min_val) ? min_val : cots_next;
    
    // 4. Calculate coral dynamics with competition and temperature effects
    // Calculate temperature effects on corals
    Type temp_diff_coral = sst - temp_opt_coral;
    
    // Use fixed variance values to avoid potential numerical issues
    Type var_slow = 4.0; // Fixed variance for slow-growing corals
    Type var_fast = 2.0; // Fixed variance for fast-growing corals
    
    // Avoid using pow() with integer exponents - use multiplication instead
    Type temp_effect_slow = exp(-0.5 * (temp_diff_coral * temp_diff_coral) / var_slow);
    Type temp_effect_fast = exp(-0.5 * (temp_diff_coral * temp_diff_coral) / var_fast);
    
    // Calculate competition terms
    Type slow_competition = (slow_t1 + comp_effect * fast_t1) / K_slow;
    slow_competition = (slow_competition > 0.9) ? 0.9 : slow_competition;
    
    Type fast_competition = (fast_t1 + comp_effect * slow_t1) / K_fast;
    fast_competition = (fast_competition > 0.9) ? 0.9 : fast_competition;
    
    // Calculate coral growth with temperature effects
    Type slow_growth = r_slow * slow_t1 * (1.0 - slow_competition) * temp_effect_slow;
    Type fast_growth = r_fast * fast_t1 * (1.0 - fast_competition) * temp_effect_fast;
    
    // Calculate next state
    Type slow_next = slow_t1 + slow_growth - pred_slow;
    Type fast_next = fast_t1 + fast_growth - pred_fast;
    
    // Ensure non-negative values
    slow_next = (slow_next < min_val) ? min_val : slow_next;
    fast_next = (fast_next < min_val) ? min_val : fast_next;
    
    // 5. Set predictions for the current time step
    cots_pred(t) = cots_next;
    slow_pred(t) = slow_next;
    fast_pred(t) = fast_next;
    
    // 6. Add to negative log-likelihood (using log-normal observation model)
    nll -= dnorm(log(cots_dat(t) + 0.1), log(cots_pred(t) + 0.1), sigma_obs_cots, true);
    nll -= dnorm(log(slow_dat(t) + 0.1), log(slow_pred(t) + 0.1), sigma_obs_slow, true);
    nll -= dnorm(log(fast_dat(t) + 0.1), log(fast_pred(t) + 0.1), sigma_obs_fast, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
