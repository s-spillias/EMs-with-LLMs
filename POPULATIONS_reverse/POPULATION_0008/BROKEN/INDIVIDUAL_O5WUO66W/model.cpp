#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(cots_dat);      // COTS abundance (individuals/m2)
  DATA_VECTOR(slow_dat);      // Slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);      // Fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);       // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);   // COTS immigration rate (individuals/m2/year)
  
  // Parameters
  PARAMETER(log_r_cots);      // Log COTS population growth rate (year^-1)
  PARAMETER(log_K_cots);      // Log COTS carrying capacity (individuals/m2)
  PARAMETER(log_alpha_slow);   // Log feeding rate on slow corals (% cover/COTS/year)
  PARAMETER(log_alpha_fast);   // Log feeding rate on fast corals (% cover/COTS/year)
  PARAMETER(log_r_slow);      // Log slow coral growth rate (year^-1)
  PARAMETER(log_r_fast);      // Log fast coral growth rate (year^-1)
  PARAMETER(temp_opt);        // Optimal temperature for COTS (Celsius)
  PARAMETER(temp_width);      // Temperature tolerance width (Celsius)
  PARAMETER(log_sigma_cots);  // Log observation error SD for COTS
  PARAMETER(log_sigma_coral); // Log observation error SD for corals
  
  // Transform parameters
  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type alpha_slow = exp(log_alpha_slow);
  Type alpha_fast = exp(log_alpha_fast);
  Type r_slow = exp(log_r_slow);
  Type r_fast = exp(log_r_fast);
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_coral = exp(log_sigma_coral);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Minimum values to prevent numerical issues
  Type min_pop = Type(1e-8);
  Type min_sd = Type(0.01);
  
  // Vectors for predictions
  vector<Type> cots_pred(cots_dat.size());
  vector<Type> slow_pred(slow_dat.size());
  vector<Type> fast_pred(fast_dat.size());
  
  // Initialize predictions with data
  for(int t = 0; t < cots_dat.size(); t++) {
    cots_pred(t) = CppAD::CondExpGe(cots_dat(t), min_pop, cots_dat(t), min_pop);
    slow_pred(t) = CppAD::CondExpGe(slow_dat(t), min_pop, slow_dat(t), min_pop);
    fast_pred(t) = CppAD::CondExpGe(fast_dat(t), min_pop, fast_dat(t), min_pop);
  }
  
  // Process model
  for(int t = 1; t < cots_dat.size(); t++) {
    // 1. Temperature effect on COTS (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t) - temp_opt)/temp_width, 2));
    
    // 2. COTS population dynamics with temperature effect and immigration
    cots_pred(t) = cots_pred(t-1) * (1 + r_cots * temp_effect * (1 - cots_pred(t-1)/K_cots)) + cotsimm_dat(t);
    cots_pred(t) = CppAD::CondExpGe(cots_pred(t), min_pop, cots_pred(t), min_pop);
    
    // 3. Coral dynamics with COTS predation
    slow_pred(t) = slow_pred(t-1) + r_slow * slow_pred(t-1) * (1 - slow_pred(t-1)/100) - 
                   alpha_slow * cots_pred(t-1) * slow_pred(t-1);
    slow_pred(t) = CppAD::CondExpGe(slow_pred(t), min_pop, slow_pred(t), min_pop);
    
    fast_pred(t) = fast_pred(t-1) + r_fast * fast_pred(t-1) * (1 - fast_pred(t-1)/100) - 
                   alpha_fast * cots_pred(t-1) * fast_pred(t-1);
    fast_pred(t) = CppAD::CondExpGe(fast_pred(t), min_pop, fast_pred(t), min_pop);
  }
  
  // Parameter bounds penalties using smooth transitions
  Type pen_r = Type(0.0);
  pen_r += CppAD::CondExpGt(r_cots, Type(2.0), pow(r_cots - Type(2.0), 2), Type(0));
  pen_r += CppAD::CondExpLt(r_cots, Type(0.1), pow(Type(0.1) - r_cots, 2), Type(0));
  nll += Type(100.0) * pen_r;

  Type pen_K = CppAD::CondExpLt(K_cots, Type(0.1), pow(Type(0.1) - K_cots, 2), Type(0));
  nll += Type(100.0) * pen_K;
  
  // Observation model using robust likelihood
  for(int t = 0; t < cots_dat.size(); t++) {
    // Ensure positive observations and predictions
    Type cots_obs = CppAD::CondExpGe(cots_dat(t), min_pop, cots_dat(t), min_pop);
    Type slow_obs = CppAD::CondExpGe(slow_dat(t), min_pop, slow_dat(t), min_pop);
    Type fast_obs = CppAD::CondExpGe(fast_dat(t), min_pop, fast_dat(t), min_pop);
    
    Type cots_pred_t = CppAD::CondExpGe(cots_pred(t), min_pop, cots_pred(t), min_pop);
    Type slow_pred_t = CppAD::CondExpGe(slow_pred(t), min_pop, slow_pred(t), min_pop);
    Type fast_pred_t = CppAD::CondExpGe(fast_pred(t), min_pop, fast_pred(t), min_pop);
    
    // Add observation likelihoods using robust formulation
    nll -= dnorm(log(cots_obs), log(cots_pred_t), sigma_cots + min_sd, true);
    nll -= dnorm(log(slow_obs), log(slow_pred_t), sigma_coral + min_sd, true);
    nll -= dnorm(log(fast_obs), log(fast_pred_t), sigma_coral + min_sd, true);
  }
  
  // Report predictions and parameters
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(r_cots);
  REPORT(K_cots);
  REPORT(alpha_slow);
  REPORT(alpha_fast);
  REPORT(r_slow);
  REPORT(r_fast);
  REPORT(temp_opt);
  REPORT(temp_width);
  
  // Report objective function value
  Type obj_val = nll;
  ADREPORT(obj_val);
  
  return nll;
}
