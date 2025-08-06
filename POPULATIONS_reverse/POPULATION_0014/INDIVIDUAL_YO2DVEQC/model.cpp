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
  Type min_sd = Type(1e-3);  // Minimum standard deviation to prevent numerical issues
  nll -= dnorm(log(cots_dat(0) + min_sd), log(cots_pred(0) + min_sd), sigma_obs_cots + min_sd, true);
  nll -= dnorm(log(slow_dat(0) + min_sd), log(slow_pred(0) + min_sd), sigma_obs_slow + min_sd, true);
  nll -= dnorm(log(fast_dat(0) + min_sd), log(fast_pred(0) + min_sd), sigma_obs_fast + min_sd, true);
  
  // Loop through time steps to calculate predictions and likelihood
  for (int t = 1; t < n_years; t++) {
    // Get previous state
    Type cots_t1 = cots_pred(t-1);
    Type slow_t1 = slow_pred(t-1);
    Type fast_t1 = fast_pred(t-1);
    Type sst = sst_dat(t-1);
    Type cotsimm = cotsimm_dat(t-1);
    
    // 1. Calculate temperature effects using Gaussian response curves
    // Use absolute value of beta parameters to ensure positive width
    Type beta_cots_temp_abs = CppAD::CondExpLt(beta_cots_temp, Type(0), Type(-1.0) * beta_cots_temp, beta_cots_temp);
    Type beta_slow_temp_abs = CppAD::CondExpLt(beta_slow_temp, Type(0), Type(-1.0) * beta_slow_temp, beta_slow_temp);
    Type beta_fast_temp_abs = CppAD::CondExpLt(beta_fast_temp, Type(0), Type(-1.0) * beta_fast_temp, beta_fast_temp);
    
    // Add small constant to prevent division by zero
    beta_cots_temp_abs = beta_cots_temp_abs + Type(1e-8);
    beta_slow_temp_abs = beta_slow_temp_abs + Type(1e-8);
    beta_fast_temp_abs = beta_fast_temp_abs + Type(1e-8);
    
    Type temp_effect_cots = exp(-pow(sst - temp_opt_cots, 2) / (2 * pow(1/beta_cots_temp_abs, 2)));
    Type temp_effect_slow = exp(-pow(sst - temp_opt_coral, 2) / (2 * pow(1/beta_slow_temp_abs, 2)));
    Type temp_effect_fast = exp(-pow(sst - temp_opt_coral, 2) / (2 * pow(1/beta_fast_temp_abs, 2)));
    
    // 2. Calculate total coral resource availability (with small constant to prevent division by zero)
    Type total_coral = slow_t1 + fast_t1 + Type(1e-8);
    
    // 3. Calculate COTS predation rates using functional responses
    // Ensure half-saturation constants are positive
    Type h_slow_pos = CppAD::CondExpLt(h_slow, Type(0), Type(0.1), h_slow);
    Type h_fast_pos = CppAD::CondExpLt(h_fast, Type(0), Type(0.1), h_fast);
    
    Type pred_slow = alpha_slow * cots_t1 * slow_t1 / (h_slow_pos + slow_t1) * (Type(1.0) - pref_fast);
    Type pred_fast = alpha_fast * cots_t1 * fast_t1 / (h_fast_pos + fast_t1) * pref_fast;
    
    // 4. Calculate resource limitation for COTS (smooth transition as resources decline)
    Type resource_limitation = Type(1.0) - exp(-Type(0.1) * total_coral);
    
    // 5. Calculate COTS population dynamics with density dependence, mortality, and immigration
    // Ensure carrying capacity is positive
    Type K_cots_pos = CppAD::CondExpLt(K_cots, Type(0), Type(0.1), K_cots);
    
    Type cots_growth = r_cots * cots_t1 * (Type(1.0) - cots_t1 / K_cots_pos) * temp_effect_cots * resource_limitation;
    Type cots_mort = m_cots * cots_t1;
    Type cots_next = cots_t1 + cots_growth - cots_mort + cotsimm;
    // Use CppAD::CondExpGt instead of max to ensure non-negative population
    cots_next = CppAD::CondExpGt(cots_next, Type(1e-8), cots_next, Type(1e-8));
    
    // 6. Calculate coral dynamics with logistic growth and COTS predation
    // Ensure carrying capacities are positive
    Type K_slow_pos = CppAD::CondExpLt(K_slow, Type(0), Type(0.1), K_slow);
    Type K_fast_pos = CppAD::CondExpLt(K_fast, Type(0), Type(0.1), K_fast);
    
    Type slow_growth = r_slow * slow_t1 * (Type(1.0) - slow_t1 / K_slow_pos) * temp_effect_slow;
    Type slow_next = slow_t1 + slow_growth - pred_slow;
    // Use CppAD::CondExpGt instead of max to ensure non-negative cover
    slow_next = CppAD::CondExpGt(slow_next, Type(1e-8), slow_next, Type(1e-8));
    
    Type fast_growth = r_fast * fast_t1 * (Type(1.0) - fast_t1 / K_fast_pos) * temp_effect_fast;
    Type fast_next = fast_t1 + fast_growth - pred_fast;
    // Use CppAD::CondExpGt instead of max to ensure non-negative cover
    fast_next = CppAD::CondExpGt(fast_next, Type(1e-8), fast_next, Type(1e-8));
    
    // 7. Set predictions for the current time step (without process error)
    cots_pred(t) = cots_next;
    slow_pred(t) = slow_next;
    fast_pred(t) = fast_next;
    
    // 8. Add to negative log-likelihood (using log-normal observation model)
    // Ensure all standard deviations are positive
    Type sigma_obs_cots_pos = CppAD::CondExpLt(sigma_obs_cots, min_sd, min_sd, sigma_obs_cots);
    Type sigma_obs_slow_pos = CppAD::CondExpLt(sigma_obs_slow, min_sd, min_sd, sigma_obs_slow);
    Type sigma_obs_fast_pos = CppAD::CondExpLt(sigma_obs_fast, min_sd, min_sd, sigma_obs_fast);
    
    nll -= dnorm(log(cots_dat(t) + min_sd), log(cots_pred(t) + min_sd), sigma_obs_cots_pos + min_sd, true);
    nll -= dnorm(log(slow_dat(t) + min_sd), log(slow_pred(t) + min_sd), sigma_obs_slow_pos + min_sd, true);
    nll -= dnorm(log(fast_dat(t) + min_sd), log(fast_pred(t) + min_sd), sigma_obs_fast_pos + min_sd, true);
  }
  
  // Add smooth penalties for biologically implausible parameter values
  // Use smooth functions to avoid discontinuities
  nll += Type(100.0) * exp(-Type(10.0) * r_cots) / (Type(1.0) + exp(-Type(10.0) * r_cots));
  nll += Type(100.0) * exp(-Type(10.0) * K_cots) / (Type(1.0) + exp(-Type(10.0) * K_cots));
  nll += Type(100.0) * exp(-Type(10.0) * m_cots) / (Type(1.0) + exp(-Type(10.0) * m_cots));
  nll += Type(100.0) * exp(-Type(10.0) * alpha_slow) / (Type(1.0) + exp(-Type(10.0) * alpha_slow));
  nll += Type(100.0) * exp(-Type(10.0) * alpha_fast) / (Type(1.0) + exp(-Type(10.0) * alpha_fast));
  nll += Type(100.0) * exp(-Type(10.0) * h_slow) / (Type(1.0) + exp(-Type(10.0) * h_slow));
  nll += Type(100.0) * exp(-Type(10.0) * h_fast) / (Type(1.0) + exp(-Type(10.0) * h_fast));
  
  // Penalty for pref_fast outside [0,1]
  Type pref_penalty = CppAD::CondExpLt(pref_fast, Type(0), -pref_fast, Type(0));
  pref_penalty += CppAD::CondExpGt(pref_fast, Type(1), pref_fast - Type(1), Type(0));
  nll += Type(100.0) * pref_penalty;
  
  nll += Type(100.0) * exp(-Type(10.0) * r_slow) / (Type(1.0) + exp(-Type(10.0) * r_slow));
  nll += Type(100.0) * exp(-Type(10.0) * r_fast) / (Type(1.0) + exp(-Type(10.0) * r_fast));
  nll += Type(100.0) * exp(-Type(10.0) * K_slow) / (Type(1.0) + exp(-Type(10.0) * K_slow));
  nll += Type(100.0) * exp(-Type(10.0) * K_fast) / (Type(1.0) + exp(-Type(10.0) * K_fast));
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
