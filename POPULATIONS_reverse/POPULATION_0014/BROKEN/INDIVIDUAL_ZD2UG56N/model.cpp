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
  PARAMETER(pred_temp_effect);        // Strength of temperature effect on COTS predation rates
  
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
  Type sd_init = Type(0.5);
  nll -= dnorm(log(cots_dat(0) + Type(1.0)), log(cots_pred(0) + Type(1.0)), sd_init, 1);
  nll -= dnorm(log(slow_dat(0) + Type(1.0)), log(slow_pred(0) + Type(1.0)), sd_init, 1);
  nll -= dnorm(log(fast_dat(0) + Type(1.0)), log(fast_pred(0) + Type(1.0)), sd_init, 1);
  
  // Loop through time steps to calculate predictions and likelihood
  for (int t = 1; t < n_years; t++) {
    // Get previous state
    Type cots_t1 = cots_pred(t-1);
    Type slow_t1 = slow_pred(t-1);
    Type fast_t1 = fast_pred(t-1);
    Type cotsimm = cotsimm_dat(t-1);
    Type sst = sst_dat(t-1);
    
    // Ensure non-negative state variables
    Type min_val = Type(1.0);
    if (cots_t1 < min_val) cots_t1 = min_val;
    if (slow_t1 < min_val) slow_t1 = min_val;
    if (fast_t1 < min_val) fast_t1 = min_val;
    
    // Calculate temperature effect on predation
    // Simple linear effect based on deviation from optimal temperature
    Type temp_diff = sst - temp_opt_cots;
    Type temp_effect = Type(1.0) - Type(0.1) * fabs(temp_diff);
    if (temp_effect < Type(0.5)) temp_effect = Type(0.5);
    if (temp_effect > Type(1.5)) temp_effect = Type(1.5);
    
    // Calculate predation rates with temperature effect
    Type pred_slow = alpha_slow * cots_t1 * slow_t1 / (Type(10.0) + slow_t1) * (Type(1.0) - pref_fast) * temp_effect;
    Type pred_fast = alpha_fast * cots_t1 * fast_t1 / (Type(8.0) + fast_t1) * pref_fast * temp_effect;
    
    // Ensure predation doesn't exceed available coral
    if (pred_slow > Type(0.3) * slow_t1) pred_slow = Type(0.3) * slow_t1;
    if (pred_fast > Type(0.3) * fast_t1) pred_fast = Type(0.3) * fast_t1;
    
    // Calculate COTS population dynamics
    Type total_pred = pred_slow + pred_fast;
    Type pred_benefit = Type(0.1) * total_pred / Type(10.0);
    
    // Calculate COTS growth with density dependence
    Type dd_term = Type(1.0) - cots_t1 / Type(2.5);
    if (dd_term < Type(0.0)) dd_term = Type(0.0);
    
    Type cots_growth = r_cots * cots_t1 * dd_term * (Type(1.0) + pred_benefit);
    Type cots_mort = m_cots * cots_t1;
    Type cots_next = cots_t1 + cots_growth - cots_mort + cotsimm;
    if (cots_next < min_val) cots_next = min_val;
    
    // Calculate coral dynamics with competition
    Type slow_competition = (slow_t1 + comp_effect * fast_t1) / Type(50.0);
    if (slow_competition > Type(0.9)) slow_competition = Type(0.9);
    
    Type fast_competition = (fast_t1 + comp_effect * slow_t1) / Type(40.0);
    if (fast_competition > Type(0.9)) fast_competition = Type(0.9);
    
    Type slow_growth = r_slow * slow_t1 * (Type(1.0) - slow_competition);
    Type fast_growth = r_fast * fast_t1 * (Type(1.0) - fast_competition);
    
    Type slow_next = slow_t1 + slow_growth - pred_slow;
    Type fast_next = fast_t1 + fast_growth - pred_fast;
    
    if (slow_next < min_val) slow_next = min_val;
    if (fast_next < min_val) fast_next = min_val;
    
    // Set predictions for the current time step
    cots_pred(t) = cots_next;
    slow_pred(t) = slow_next;
    fast_pred(t) = fast_next;
    
    // Add to negative log-likelihood
    nll -= dnorm(log(cots_dat(t) + Type(1.0)), log(cots_pred(t) + Type(1.0)), sigma_obs_cots, 1);
    nll -= dnorm(log(slow_dat(t) + Type(1.0)), log(slow_pred(t) + Type(1.0)), sigma_obs_slow, 1);
    nll -= dnorm(log(fast_dat(t) + Type(1.0)), log(fast_pred(t) + Type(1.0)), sigma_obs_fast, 1);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
