#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Year);                  // Time vector (years)
  DATA_VECTOR(sst_dat);              // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);          // COTS larval immigration rate (individuals/m2/year)
  DATA_VECTOR(cots_dat);             // Observed COTS density (individuals/m2)
  DATA_VECTOR(fast_dat);             // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);             // Observed slow-growing coral cover (%)
  
  // Parameters
  PARAMETER(log_r_cots);             // COTS population growth rate
  PARAMETER(log_allee_threshold);    // Allee effect threshold density
  PARAMETER(log_K_cots);             // COTS carrying capacity
  PARAMETER(log_temp_opt);           // Optimal temperature for COTS survival
  PARAMETER(log_temp_range);         // Temperature tolerance range
  PARAMETER(log_grazing_fast);       // Grazing rate on fast corals
  PARAMETER(log_grazing_slow);       // Grazing rate on slow corals
  PARAMETER(log_r_fast);             // Fast coral growth rate
  PARAMETER(log_r_slow);             // Slow coral growth rate
  PARAMETER(logit_coral_limit);      // Total coral cover limit
  PARAMETER(log_obs_sd_cots);        // Observation error SD for COTS
  PARAMETER(log_obs_sd_fast);        // Observation error SD for fast coral
  PARAMETER(log_obs_sd_slow);        // Observation error SD for slow coral

  
  // Transform parameters with bounds checking
  Type r_cots = exp(log_r_cots);
  Type allee_threshold = exp(log_allee_threshold);
  Type K_cots = exp(log_K_cots);
  Type temp_opt = exp(log_temp_opt);
  Type temp_range = CppAD::CondExpGt(exp(log_temp_range), Type(0.1), exp(log_temp_range), Type(0.1));
  Type grazing_fast = exp(log_grazing_fast);
  Type grazing_slow = exp(log_grazing_slow);
  Type r_fast = exp(log_r_fast);
  Type r_slow = exp(log_r_slow);
  Type coral_limit = invlogit(logit_coral_limit);
  Type obs_sd_cots = CppAD::CondExpGt(exp(log_obs_sd_cots), Type(0.01), exp(log_obs_sd_cots), Type(0.01));
  Type obs_sd_fast = CppAD::CondExpGt(exp(log_obs_sd_fast), Type(0.01), exp(log_obs_sd_fast), Type(0.01));
  Type obs_sd_slow = CppAD::CondExpGt(exp(log_obs_sd_slow), Type(0.01), exp(log_obs_sd_slow), Type(0.01));
  
  // Initialize predicted vectors
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Set initial conditions
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Time series predictions
  for(int t = 1; t < n; t++) {
    // 1. Temperature effect on COTS survival (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t-1) - temp_opt) / temp_range, 2));
    
    // 2. Resource limitation based on total coral cover
    Type total_coral = (fast_pred(t-1) + slow_pred(t-1)) / 100.0; // Convert to proportion
    Type resource_limit = total_coral / (total_coral + eps);
    
    // 3. COTS population dynamics with Allee effect and temperature-dependent recruitment
    Type recruitment = cotsimm_dat(t-1) * temp_effect;
    Type allee_effect = cots_pred(t-1) / (cots_pred(t-1) + allee_threshold + eps);
    Type density_effect = CppAD::CondExpGt(Type(1) - cots_pred(t-1)/K_cots, Type(-1), Type(1) - cots_pred(t-1)/K_cots, Type(-1));
    cots_pred(t) = cots_pred(t-1) * (Type(1) + r_cots * resource_limit * allee_effect * density_effect) + recruitment;
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), Type(0), cots_pred(t), Type(0));
    
    // 4. Coral dynamics with COTS predation
    Type coral_space = (1 - (fast_pred(t-1) + slow_pred(t-1))/100.0/coral_limit);
    coral_space = CppAD::CondExpGt(coral_space, Type(0), coral_space, Type(0));
    
    // Fast-growing coral
    Type fast_growth = r_fast * fast_pred(t-1) * coral_space;
    Type fast_pred_loss = grazing_fast * cots_pred(t-1) * fast_pred(t-1);
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_pred_loss;
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), Type(0), fast_pred(t), Type(0));
    
    // Slow-growing coral
    Type slow_growth = r_slow * slow_pred(t-1) * coral_space;
    Type slow_pred_loss = grazing_slow * cots_pred(t-1) * slow_pred(t-1);
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_pred_loss;
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), Type(0), slow_pred(t), Type(0));
  }
  
  // Observation model using log-normal distribution
  for(int t = 0; t < n; t++) {
    // Add small constant to prevent log(0)
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), obs_sd_cots, true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), obs_sd_fast, true);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), obs_sd_slow, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
