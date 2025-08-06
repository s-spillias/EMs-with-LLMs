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
  PARAMETER(log_K_cots);             // COTS carrying capacity
  PARAMETER(log_allee_threshold);    // Allee effect threshold density
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
  
  // Transform parameters
  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type allee_threshold = exp(log_allee_threshold);
  Type temp_opt = exp(log_temp_opt);
  Type temp_range = exp(log_temp_range);
  Type grazing_fast = exp(log_grazing_fast);
  Type grazing_slow = exp(log_grazing_slow);
  Type r_fast = exp(log_r_fast);
  Type r_slow = exp(log_r_slow);
  Type coral_limit = invlogit(logit_coral_limit);
  Type obs_sd_cots = exp(log_obs_sd_cots);
  Type obs_sd_fast = exp(log_obs_sd_fast);
  Type obs_sd_slow = exp(log_obs_sd_slow);
  
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
    
    // Bounded COTS density to prevent extreme values
    Type bounded_density = CppAD::CondExpGt(cots_pred(t-1), eps, cots_pred(t-1), eps);
    
    // Allee effect with improved numerical stability
    Type allee_ratio = bounded_density / (allee_threshold + eps);
    Type allee_effect = allee_ratio / (Type(1.0) + allee_ratio);
    
    // Logistic growth component with safeguards
    Type carrying_capacity_effect = CppAD::CondExpGt(Type(1.0) - bounded_density/K_cots, Type(0), 
                                                    Type(1.0) - bounded_density/K_cots, Type(0));
    Type base_growth = r_cots * resource_limit * allee_effect * carrying_capacity_effect;
    
    // Bounded growth rate
    Type growth_rate = CppAD::CondExpGt(base_growth, Type(-0.99), base_growth, Type(-0.99));
    
    // Update prediction with safeguards
    cots_pred(t) = bounded_density * (Type(1.0) + growth_rate) + recruitment;
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), eps, cots_pred(t), eps);
    
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
    // Observation model with robust handling of small values
    Type cots_obs = log(CppAD::CondExpGt(cots_dat(t), eps, cots_dat(t), eps));
    Type cots_pred_t = log(CppAD::CondExpGt(cots_pred(t), eps, cots_pred(t), eps));
    nll -= dnorm(cots_obs, cots_pred_t, obs_sd_cots, true);
    
    Type fast_obs = log(CppAD::CondExpGt(fast_dat(t), eps, fast_dat(t), eps));
    Type fast_pred_t = log(CppAD::CondExpGt(fast_pred(t), eps, fast_pred(t), eps));
    nll -= dnorm(fast_obs, fast_pred_t, obs_sd_fast, true);
    
    Type slow_obs = log(CppAD::CondExpGt(slow_dat(t), eps, slow_dat(t), eps));
    Type slow_pred_t = log(CppAD::CondExpGt(slow_pred(t), eps, slow_pred(t), eps));
    nll -= dnorm(slow_obs, slow_pred_t, obs_sd_slow, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
