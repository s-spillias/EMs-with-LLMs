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
  PARAMETER(temp_pred_effect);        // Temperature effect on COTS predation efficiency (per °C²)
  
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
  Type nll = Type(0.0);
  
  // Get data dimensions
  int n_years = Year.size();
  
  // Initialize vectors for model predictions
  vector<Type> cots_pred(n_years);
  vector<Type> slow_pred(n_years);
  vector<Type> fast_pred(n_years);
  
  // Set minimum values to avoid numerical issues
  Type min_val = Type(1.0);
  Type max_pred_frac = Type(0.3);
  Type min_effect = Type(0.5);
  Type max_effect = Type(1.5);
  Type max_comp = Type(0.9);
  Type eps = Type(1.0);
  
  // Initialize state variables with first observation
  cots_pred(0) = (cots_dat(0) < min_val) ? min_val : cots_dat(0);
  slow_pred(0) = (slow_dat(0) < min_val) ? min_val : slow_dat(0);
  fast_pred(0) = (fast_dat(0) < min_val) ? min_val : fast_dat(0);
  
  // Loop through time steps to calculate predictions
  for (int t = 1; t < n_years; t++) {
    // Get previous state
    Type cots_t1 = (cots_pred(t-1) < min_val) ? min_val : cots_pred(t-1);
    Type slow_t1 = (slow_pred(t-1) < min_val) ? min_val : slow_pred(t-1);
    Type fast_t1 = (fast_pred(t-1) < min_val) ? min_val : fast_pred(t-1);
    Type cotsimm = cotsimm_dat(t-1);
    Type sst = sst_dat(t-1);
    
    // Calculate temperature effect on predation (our key ecological improvement)
    Type temp_diff = sst - temp_opt_cots;
    Type temp_sq = temp_diff * temp_diff;
    Type temp_effect = Type(1.0) - temp_pred_effect * temp_sq;
    
    // Bound the temperature effect
    temp_effect = (temp_effect < min_effect) ? min_effect : temp_effect;
    temp_effect = (temp_effect > max_effect) ? max_effect : temp_effect;
    
    // Calculate predation rates with temperature effect
    Type pred_slow = alpha_slow * cots_t1 * slow_t1 / (h_slow + slow_t1) * (Type(1.0) - pref_fast) * temp_effect;
    Type pred_fast = alpha_fast * cots_t1 * fast_t1 / (h_fast + fast_t1) * pref_fast * temp_effect;
    
    // Limit predation to avoid numerical issues
    pred_slow = (pred_slow > max_pred_frac * slow_t1) ? max_pred_frac * slow_t1 : pred_slow;
    pred_fast = (pred_fast > max_pred_frac * fast_t1) ? max_pred_frac * fast_t1 : pred_fast;
    
    // Calculate COTS population dynamics
    Type dd_term = Type(1.0) - cots_t1 / K_cots;
    dd_term = (dd_term < Type(-0.5)) ? Type(-0.5) : dd_term;
    
    Type cots_growth = r_cots * cots_t1 * dd_term;
    Type cots_mort = m_cots * cots_t1;
    Type cots_next = cots_t1 + cots_growth - cots_mort + cotsimm;
    cots_next = (cots_next < min_val) ? min_val : cots_next;
    
    // Calculate coral dynamics
    // Simple temperature effect on coral growth
    Type slow_temp_effect = Type(1.0) + beta_slow_temp * (sst - temp_opt_coral);
    Type fast_temp_effect = Type(1.0) + beta_fast_temp * (sst - temp_opt_coral);
    
    // Bound temperature effects
    slow_temp_effect = (slow_temp_effect < min_effect) ? min_effect : slow_temp_effect;
    slow_temp_effect = (slow_temp_effect > max_effect) ? max_effect : slow_temp_effect;
    fast_temp_effect = (fast_temp_effect < min_effect) ? min_effect : fast_temp_effect;
    fast_temp_effect = (fast_temp_effect > max_effect) ? max_effect : fast_temp_effect;
    
    // Calculate competition terms
    Type slow_competition = (slow_t1 + comp_effect * fast_t1) / K_slow;
    Type fast_competition = (fast_t1 + comp_effect * slow_t1) / K_fast;
    
    // Bound competition to avoid numerical issues
    slow_competition = (slow_competition > max_comp) ? max_comp : slow_competition;
    fast_competition = (fast_competition > max_comp) ? max_comp : fast_competition;
    
    // Calculate coral growth
    Type slow_growth = r_slow * slow_t1 * (Type(1.0) - slow_competition) * slow_temp_effect;
    Type fast_growth = r_fast * fast_t1 * (Type(1.0) - fast_competition) * fast_temp_effect;
    
    // Calculate next state
    Type slow_next = slow_t1 + slow_growth - pred_slow;
    Type fast_next = fast_t1 + fast_growth - pred_fast;
    
    // Ensure non-negative values
    slow_next = (slow_next < min_val) ? min_val : slow_next;
    fast_next = (fast_next < min_val) ? min_val : fast_next;
    
    // Set predictions for the current time step
    cots_pred(t) = cots_next;
    slow_pred(t) = slow_next;
    fast_pred(t) = fast_next;
  }
  
  // Calculate likelihood using a robust approach
  for (int t = 0; t < n_years; t++) {
    // Use robust likelihood calculation with observation errors
    Type cots_obs = (cots_dat(t) < min_val) ? min_val : cots_dat(t);
    Type slow_obs = (slow_dat(t) < min_val) ? min_val : slow_dat(t);
    Type fast_obs = (fast_dat(t) < min_val) ? min_val : fast_dat(t);
    
    nll -= dnorm(log(cots_obs), log(cots_pred(t)), sigma_obs_cots, true);
    nll -= dnorm(log(slow_obs), log(slow_pred(t)), sigma_obs_slow, true);
    nll -= dnorm(log(fast_obs), log(fast_pred(t)), sigma_obs_fast, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
