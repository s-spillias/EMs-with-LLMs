#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Year);                  // Time vector
  DATA_VECTOR(sst_dat);              // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);          // COTS immigration rate (individuals/m2/year)
  DATA_VECTOR(cots_dat);             // Observed COTS density (individuals/m2)
  DATA_VECTOR(slow_dat);             // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);             // Observed fast-growing coral cover (%)
  
  // Parameters
  PARAMETER(r_cots);                 // COTS intrinsic growth rate
  PARAMETER(K_cots);                 // COTS carrying capacity
  PARAMETER(temp_opt);               // Optimal temperature for COTS growth
  PARAMETER(temp_range);             // Temperature tolerance range
  PARAMETER(feed_temp_opt);          // Optimal temperature for COTS feeding
  PARAMETER(attack_rate_fast);       // Attack rate on fast coral
  PARAMETER(attack_rate_slow);       // Attack rate on slow coral
  PARAMETER(handling_time);          // Prey handling time
  PARAMETER(r_fast);                 // Fast coral growth rate
  PARAMETER(r_slow);                 // Slow coral growth rate
  PARAMETER(K_coral);                // Total coral carrying capacity
  PARAMETER(sigma_cots);             // SD for COTS observations
  PARAMETER(sigma_coral);            // SD for coral observations

  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Constants for numerical stability
  const Type eps = Type(1e-8);
  const Type max_val = Type(1e3);
  
  // Transform parameters to valid ranges with exp() for stability
  Type r_cots_pos = exp(r_cots);
  Type K_cots_pos = exp(K_cots);
  Type attack_rate_fast_pos = exp(attack_rate_fast);
  Type attack_rate_slow_pos = exp(attack_rate_slow);
  Type handling_time_pos = exp(handling_time);
  Type r_fast_pos = exp(r_fast);
  Type r_slow_pos = exp(r_slow);
  Type K_coral_pos = exp(K_coral);
  Type sigma_cots_pos = exp(sigma_cots);
  Type sigma_coral_pos = exp(sigma_coral);
  
  // Add large penalties for invalid parameter values
  if(temp_range < 0) nll += 1e10;
  if(temp_opt < 20 || temp_opt > 35) nll += 1e10;
  if(feed_temp_opt < 20 || feed_temp_opt > 35) nll += 1e10;
  
  // Vectors to store predictions
  vector<Type> cots_pred(Year.size());
  vector<Type> slow_pred(Year.size());
  vector<Type> fast_pred(Year.size());
  
  // Initialize first time step with observations
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Time series simulation
  for(int t = 1; t < Year.size(); t++) {
    // 1. Temperature effect on COTS growth (Gaussian response with bounds)
    Type temp_diff = (sst_dat(t-1) - temp_opt) / (temp_range + eps);
    Type temp_effect = exp(-0.5 * pow(temp_diff, 2));
    temp_effect = temp_effect / (1 + temp_effect);  // Bound between 0 and 1
    
    // 2. Temperature-dependent predation with Type II functional responses
    Type feed_temp_diff = (sst_dat(t-1) - feed_temp_opt) / (temp_range + eps);
    Type feed_temp_effect = exp(-0.5 * pow(feed_temp_diff, 2));
    feed_temp_effect = feed_temp_effect / (1 + feed_temp_effect);  // Bound between 0 and 1
    
    Type total_coral = slow_pred(t-1) + fast_pred(t-1);
    Type pred_rate_slow = feed_temp_effect * (attack_rate_slow_pos * slow_pred(t-1)) / 
                         (1 + handling_time_pos * (total_coral + eps));
    Type pred_rate_fast = feed_temp_effect * (attack_rate_fast_pos * fast_pred(t-1)) / 
                         (1 + handling_time_pos * (total_coral + eps));
    
    // Bound predation rates
    pred_rate_slow = CppAD::CondExpGt(pred_rate_slow, max_val,
                                     max_val, pred_rate_slow);
    pred_rate_fast = CppAD::CondExpGt(pred_rate_fast, max_val,
                                     max_val, pred_rate_fast);
    
    // 3. COTS population dynamics with improved stability
    Type density_effect = 1 - cots_pred(t-1) / K_cots_pos;
    density_effect = CppAD::CondExpGt(density_effect, Type(-1),
                                     density_effect, Type(-1));
    
    Type growth = r_cots_pos * temp_effect * density_effect * cots_pred(t-1);
    Type mortality = Type(0.1) * cots_pred(t-1);
    
    cots_pred(t) = cots_pred(t-1) + growth + cotsimm_dat(t-1) - mortality;
    
    // 4. Coral dynamics with competition and improved stability
    Type total_cover = (slow_pred(t-1) + fast_pred(t-1)) / (K_coral_pos + eps);
    Type competition = CppAD::CondExpGt(Type(1) - total_cover, Type(0),
                                      Type(1) - total_cover, Type(0));
    
    // Calculate changes with predation rates
    Type slow_growth = r_slow_pos * slow_pred(t-1) * competition;
    Type fast_growth = r_fast_pos * fast_pred(t-1) * competition;
    Type slow_pred_loss = pred_rate_slow * cots_pred(t-1);
    Type fast_pred_loss = pred_rate_fast * cots_pred(t-1);
    
    // Update states with bounded predictions
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t-1) + slow_growth - slow_pred_loss, eps,
                                   slow_pred(t-1) + slow_growth - slow_pred_loss, eps);
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t-1) + fast_growth - fast_pred_loss, eps,
                                   fast_pred(t-1) + fast_growth - fast_pred_loss, eps);
    
    // 5. Ensure predictions stay positive and bounded
    cots_pred(t) = CppAD::CondExpLt(cots_pred(t), eps,
                                   eps, cots_pred(t));
    slow_pred(t) = CppAD::CondExpLt(slow_pred(t), eps,
                                   eps, slow_pred(t));
    fast_pred(t) = CppAD::CondExpLt(fast_pred(t), eps,
                                   eps, fast_pred(t));
                                   
    // Upper bounds
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), max_val,
                                   max_val, cots_pred(t));
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), K_coral,
                                   K_coral, slow_pred(t));
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), K_coral,
                                   K_coral, fast_pred(t));
  }
  
  // Likelihood calculations using log-normal distribution
  for(int t = 0; t < Year.size(); t++) {
    // COTS likelihood
    nll -= dnorm(log(cots_dat(t) + eps), 
                 log(cots_pred(t) + eps), 
                 sigma_cots_pos, true);
    
    // Coral likelihoods
    nll -= dnorm(log(slow_dat(t) + eps), 
                 log(slow_pred(t) + eps), 
                 sigma_coral_pos, true);
    nll -= dnorm(log(fast_dat(t) + eps), 
                 log(fast_pred(t) + eps), 
                 sigma_coral_pos, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
