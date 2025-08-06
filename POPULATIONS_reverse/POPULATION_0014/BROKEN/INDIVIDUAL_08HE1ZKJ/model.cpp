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
  
  // Competition parameters
  PARAMETER(comp_slow_on_fast);       // Competitive effect of slow-growing corals on fast-growing corals
  PARAMETER(comp_fast_on_slow);       // Competitive effect of fast-growing corals on slow-growing corals
  
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
  
  // Minimum values to prevent numerical issues
  Type min_val = Type(0.01);
  Type min_sd = Type(0.01);
  
  // Ensure positive standard deviations
  Type sigma_obs_cots_pos = CppAD::CondExpLt(sigma_obs_cots, min_sd, min_sd, sigma_obs_cots);
  Type sigma_obs_slow_pos = CppAD::CondExpLt(sigma_obs_slow, min_sd, min_sd, sigma_obs_slow);
  Type sigma_obs_fast_pos = CppAD::CondExpLt(sigma_obs_fast, min_sd, min_sd, sigma_obs_fast);
  
  // Loop through time steps to calculate predictions and likelihood
  for (int t = 1; t < n_years; t++) {
    // Get previous state
    Type cots_t1 = cots_pred(t-1);
    Type slow_t1 = slow_pred(t-1);
    Type fast_t1 = fast_pred(t-1);
    Type sst = sst_dat(t-1);
    Type cotsimm = cotsimm_dat(t-1);
    
    // Ensure non-negative state variables
    cots_t1 = CppAD::CondExpLt(cots_t1, min_val, min_val, cots_t1);
    slow_t1 = CppAD::CondExpLt(slow_t1, min_val, min_val, slow_t1);
    fast_t1 = CppAD::CondExpLt(fast_t1, min_val, min_val, fast_t1);
    
    // Calculate total coral resource availability
    Type total_coral = slow_t1 + fast_t1;
    
    // Ensure positive half-saturation constants
    Type h_slow_pos = CppAD::CondExpLt(h_slow, min_val, min_val, h_slow);
    Type h_fast_pos = CppAD::CondExpLt(h_fast, min_val, min_val, h_fast);
    
    // Bound preference between 0 and 1
    Type pref_fast_bounded = CppAD::CondExpLt(pref_fast, Type(0), Type(0), 
                             CppAD::CondExpGt(pref_fast, Type(1), Type(1), pref_fast));
    
    // Calculate predation
    Type pred_slow = alpha_slow * cots_t1 * slow_t1 / (h_slow_pos + slow_t1) * (Type(1.0) - pref_fast_bounded);
    Type pred_fast = alpha_fast * cots_t1 * fast_t1 / (h_fast_pos + fast_t1) * pref_fast_bounded;
    
    // Ensure predation doesn't exceed available coral
    pred_slow = CppAD::CondExpGt(pred_slow, slow_t1 * Type(0.9), slow_t1 * Type(0.9), pred_slow);
    pred_fast = CppAD::CondExpGt(pred_fast, fast_t1 * Type(0.9), fast_t1 * Type(0.9), pred_fast);
    
    // Calculate resource limitation for COTS
    Type resource_limitation = Type(1.0) - exp(-Type(0.1) * total_coral);
    
    // Ensure positive carrying capacity
    Type K_cots_pos = CppAD::CondExpLt(K_cots, min_val, min_val, K_cots);
    
    // Calculate COTS growth
    Type cots_growth = r_cots * cots_t1 * (Type(1.0) - cots_t1 / K_cots_pos) * resource_limitation;
    Type cots_mort = m_cots * cots_t1;
    Type cots_next = cots_t1 + cots_growth - cots_mort + cotsimm;
    cots_next = CppAD::CondExpLt(cots_next, min_val, min_val, cots_next);
    
    // Ensure positive carrying capacities for corals
    Type K_slow_pos = CppAD::CondExpLt(K_slow, min_val, min_val, K_slow);
    Type K_fast_pos = CppAD::CondExpLt(K_fast, min_val, min_val, K_fast);
    
    // Ensure non-negative competition coefficients
    Type comp_slow_on_fast_pos = CppAD::CondExpLt(comp_slow_on_fast, Type(0), Type(0), comp_slow_on_fast);
    Type comp_fast_on_slow_pos = CppAD::CondExpLt(comp_fast_on_slow, Type(0), Type(0), comp_fast_on_slow);
    
    // Calculate effective carrying capacities with competition
    Type K_slow_eff = K_slow_pos / (Type(1.0) + comp_fast_on_slow_pos * fast_t1 / (K_fast_pos + min_val));
    Type K_fast_eff = K_fast_pos / (Type(1.0) + comp_slow_on_fast_pos * slow_t1 / (K_slow_pos + min_val));
    
    // Calculate coral growth
    Type slow_growth = r_slow * slow_t1 * (Type(1.0) - slow_t1 / K_slow_eff);
    Type slow_next = slow_t1 + slow_growth - pred_slow;
    slow_next = CppAD::CondExpLt(slow_next, min_val, min_val, slow_next);
    
    Type fast_growth = r_fast * fast_t1 * (Type(1.0) - fast_t1 / K_fast_eff);
    Type fast_next = fast_t1 + fast_growth - pred_fast;
    fast_next = CppAD::CondExpLt(fast_next, min_val, min_val, fast_next);
    
    // Set predictions for the current time step
    cots_pred(t) = cots_next;
    slow_pred(t) = slow_next;
    fast_pred(t) = fast_next;
  }
  
  // Calculate negative log-likelihood
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
