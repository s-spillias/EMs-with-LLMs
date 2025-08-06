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
  
  // Recovery delay parameters
  PARAMETER(recovery_delay_slow);     // Recovery delay time for slow-growing corals after predation (years)
  PARAMETER(recovery_delay_fast);     // Recovery delay time for fast-growing corals after predation (years)
  
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
  
  // Vectors to track cumulative predation impact for recovery delay
  vector<Type> slow_pred_impact(n_years);
  vector<Type> fast_pred_impact(n_years);
  
  // Initialize state variables with first observation
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred_impact(0) = Type(0);
  fast_pred_impact(0) = Type(0);
  
  // Minimum value to prevent numerical issues
  Type min_val = Type(1e-4);
  
  // Ensure parameters are positive
  Type r_cots_pos = exp(r_cots);
  Type K_cots_pos = exp(K_cots);
  Type m_cots_pos = exp(m_cots);
  Type alpha_slow_pos = exp(alpha_slow);
  Type alpha_fast_pos = exp(alpha_fast);
  Type h_slow_pos = exp(h_slow);
  Type h_fast_pos = exp(h_fast);
  Type pref_fast_bounded = 1.0 / (1.0 + exp(-pref_fast)); // Logistic transformation to [0,1]
  Type r_slow_pos = exp(r_slow);
  Type r_fast_pos = exp(r_fast);
  Type K_slow_pos = exp(K_slow);
  Type K_fast_pos = exp(K_fast);
  Type recovery_delay_slow_pos = exp(recovery_delay_slow);
  Type recovery_delay_fast_pos = exp(recovery_delay_fast);
  Type sigma_obs_cots_pos = exp(sigma_obs_cots);
  Type sigma_obs_slow_pos = exp(sigma_obs_slow);
  Type sigma_obs_fast_pos = exp(sigma_obs_fast);
  
  // Add first observations to likelihood
  nll -= dnorm(log(cots_dat(0) + min_val), log(cots_pred(0) + min_val), sigma_obs_cots_pos, true);
  nll -= dnorm(log(slow_dat(0) + min_val), log(slow_pred(0) + min_val), sigma_obs_slow_pos, true);
  nll -= dnorm(log(fast_dat(0) + min_val), log(fast_pred(0) + min_val), sigma_obs_fast_pos, true);
  
  // Loop through time steps to calculate predictions and likelihood
  for (int t = 1; t < n_years; t++) {
    // Get previous state
    Type cots_t1 = cots_pred(t-1);
    Type slow_t1 = slow_pred(t-1);
    Type fast_t1 = fast_pred(t-1);
    Type slow_impact_t1 = slow_pred_impact(t-1);
    Type fast_impact_t1 = fast_pred_impact(t-1);
    Type sst = sst_dat(t-1);
    Type cotsimm = cotsimm_dat(t-1);
    
    // 1. Calculate temperature effects using Gaussian response curves
    Type beta_cots_temp_abs = exp(beta_cots_temp);
    Type beta_slow_temp_abs = exp(beta_slow_temp);
    Type beta_fast_temp_abs = exp(beta_fast_temp);
    
    Type temp_effect_cots = exp(-pow(sst - temp_opt_cots, 2) / (2 * pow(beta_cots_temp_abs, 2)));
    Type temp_effect_slow = exp(-pow(sst - temp_opt_coral, 2) / (2 * pow(beta_slow_temp_abs, 2)));
    Type temp_effect_fast = exp(-pow(sst - temp_opt_coral, 2) / (2 * pow(beta_fast_temp_abs, 2)));
    
    // 2. Calculate total coral resource availability
    Type total_coral = slow_t1 + fast_t1 + min_val;
    
    // 3. Calculate COTS predation rates using functional responses
    Type pred_slow = alpha_slow_pos * cots_t1 * slow_t1 / (h_slow_pos + slow_t1) * (Type(1.0) - pref_fast_bounded);
    Type pred_fast = alpha_fast_pos * cots_t1 * fast_t1 / (h_fast_pos + fast_t1) * pref_fast_bounded;
    
    // 4. Calculate resource limitation for COTS
    Type resource_limitation = Type(1.0) - exp(-Type(0.1) * total_coral);
    
    // 5. Calculate COTS population dynamics
    Type cots_growth = r_cots_pos * cots_t1 * (Type(1.0) - cots_t1 / K_cots_pos) * temp_effect_cots * resource_limitation;
    Type cots_mort = m_cots_pos * cots_t1;
    Type cots_next = cots_t1 + cots_growth - cots_mort + cotsimm;
    cots_next = cots_next < min_val ? min_val : cots_next;
    
    // 6. Update predation impact trackers with exponential decay
    Type slow_decay_rate = exp(-Type(1.0) / recovery_delay_slow_pos);
    Type fast_decay_rate = exp(-Type(1.0) / recovery_delay_fast_pos);
    
    Type slow_impact_next = slow_impact_t1 * slow_decay_rate + pred_slow;
    Type fast_impact_next = fast_impact_t1 * fast_decay_rate + pred_fast;
    
    // 7. Calculate recovery inhibition factors
    Type slow_recovery_inhibition = Type(0.1) + Type(0.9) * exp(-slow_impact_next);
    Type fast_recovery_inhibition = Type(0.1) + Type(0.9) * exp(-fast_impact_next);
    
    // 8. Calculate coral dynamics
    Type slow_growth = r_slow_pos * slow_t1 * (Type(1.0) - slow_t1 / K_slow_pos) * temp_effect_slow * slow_recovery_inhibition;
    Type slow_next = slow_t1 + slow_growth - pred_slow;
    slow_next = slow_next < min_val ? min_val : slow_next;
    
    Type fast_growth = r_fast_pos * fast_t1 * (Type(1.0) - fast_t1 / K_fast_pos) * temp_effect_fast * fast_recovery_inhibition;
    Type fast_next = fast_t1 + fast_growth - pred_fast;
    fast_next = fast_next < min_val ? min_val : fast_next;
    
    // 9. Set predictions for the current time step
    cots_pred(t) = cots_next;
    slow_pred(t) = slow_next;
    fast_pred(t) = fast_next;
    slow_pred_impact(t) = slow_impact_next;
    fast_pred_impact(t) = fast_impact_next;
    
    // 10. Add to negative log-likelihood
    nll -= dnorm(log(cots_dat(t) + min_val), log(cots_pred(t) + min_val), sigma_obs_cots_pos, true);
    nll -= dnorm(log(slow_dat(t) + min_val), log(slow_pred(t) + min_val), sigma_obs_slow_pos, true);
    nll -= dnorm(log(fast_dat(t) + min_val), log(fast_pred(t) + min_val), sigma_obs_fast_pos, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(slow_pred_impact);
  REPORT(fast_pred_impact);
  
  return nll;
}
