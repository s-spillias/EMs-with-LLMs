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
  Type min_sd = 0.2;  // Increased minimum SD to prevent numerical issues
  nll -= dnorm(log(cots_dat(0) + 0.2), log(cots_pred(0) + 0.2), min_sd, true);
  nll -= dnorm(log(slow_dat(0) + 0.2), log(slow_pred(0) + 0.2), min_sd, true);
  nll -= dnorm(log(fast_dat(0) + 0.2), log(fast_pred(0) + 0.2), min_sd, true);
  
  // Loop through time steps to calculate predictions and likelihood
  for (int t = 1; t < n_years; t++) {
    // Get previous state
    Type cots_t1 = cots_pred(t-1);
    Type slow_t1 = slow_pred(t-1);
    Type fast_t1 = fast_pred(t-1);
    Type cotsimm = cotsimm_dat(t-1);
    
    // Ensure non-negative state variables
    Type min_val = 0.2;
    if (cots_t1 < min_val) cots_t1 = min_val;
    if (slow_t1 < min_val) slow_t1 = min_val;
    if (fast_t1 < min_val) fast_t1 = min_val;
    
    // 1. Calculate total coral resource availability
    Type total_coral = slow_t1 + fast_t1;
    
    // 2. Calculate predation rates - simplified
    Type pred_slow = 0.05 * cots_t1 * slow_t1 / (5.0 + slow_t1) * 0.4;
    Type pred_fast = 0.08 * cots_t1 * fast_t1 / (5.0 + fast_t1) * 0.6;
    
    // Ensure predation doesn't exceed available coral
    if (pred_slow > 0.5 * slow_t1) pred_slow = 0.5 * slow_t1;
    if (pred_fast > 0.5 * fast_t1) pred_fast = 0.5 * fast_t1;
    
    // 3. Calculate COTS population dynamics - simplified
    // Calculate predation benefit
    Type total_pred = pred_slow + pred_fast;
    Type pred_benefit = 0.1 * total_pred / (total_coral + 10.0);
    
    // Calculate COTS growth with density dependence
    Type dd_term = 1.0 - cots_t1 / 2.5;
    if (dd_term < -0.5) dd_term = -0.5;
    
    Type cots_growth = 0.8 * cots_t1 * dd_term * (1.0 + pred_benefit);
    
    // Calculate mortality and next state
    Type cots_mort = 0.3 * cots_t1;
    Type cots_next = cots_t1 + cots_growth - cots_mort + cotsimm;
    if (cots_next < min_val) cots_next = min_val;
    
    // 4. Calculate coral dynamics with competition - simplified
    // Calculate competition terms
    Type slow_competition = (slow_t1 + 0.4 * fast_t1) / 50.0;
    if (slow_competition > 0.9) slow_competition = 0.9;
    
    Type fast_competition = (fast_t1 + 0.4 * slow_t1) / 40.0;
    if (fast_competition > 0.9) fast_competition = 0.9;
    
    // Calculate coral growth
    Type slow_growth = 0.1 * slow_t1 * (1.0 - slow_competition);
    Type fast_growth = 0.3 * fast_t1 * (1.0 - fast_competition);
    
    // Calculate next state
    Type slow_next = slow_t1 + slow_growth - pred_slow;
    Type fast_next = fast_t1 + fast_growth - pred_fast;
    
    // Ensure non-negative values
    if (slow_next < min_val) slow_next = min_val;
    if (fast_next < min_val) fast_next = min_val;
    
    // 5. Set predictions for the current time step
    cots_pred(t) = cots_next;
    slow_pred(t) = slow_next;
    fast_pred(t) = fast_next;
    
    // 6. Add to negative log-likelihood (using log-normal observation model)
    nll -= dnorm(log(cots_dat(t) + 0.2), log(cots_pred(t) + 0.2), min_sd, true);
    nll -= dnorm(log(slow_dat(t) + 0.2), log(slow_pred(t) + 0.2), min_sd, true);
    nll -= dnorm(log(fast_dat(t) + 0.2), log(fast_pred(t) + 0.2), min_sd, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
