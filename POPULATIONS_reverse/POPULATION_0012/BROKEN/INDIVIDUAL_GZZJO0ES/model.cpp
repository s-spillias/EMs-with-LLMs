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
  PARAMETER(temp_opt);               // Optimal temperature for COTS
  PARAMETER(temp_range);             // Temperature tolerance range
  PARAMETER(feed_temp_opt);          // Optimal temperature for feeding
  PARAMETER(feed_temp_range);        // Temperature range for feeding
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
  
  // Add large penalties for invalid parameter values
  if(r_cots < 0) nll += 1e10;
  if(K_cots < 0) nll += 1e10;
  if(temp_range < 0) nll += 1e10;
  if(attack_rate_fast < 0) nll += 1e10;
  if(attack_rate_slow < 0) nll += 1e10;
  if(handling_time < 0) nll += 1e10;
  if(r_fast < 0) nll += 1e10;
  if(r_slow < 0) nll += 1e10;
  if(K_coral < 0) nll += 1e10;
  if(sigma_cots < eps) nll += 1e10;
  if(sigma_coral < eps) nll += 1e10;
  
  // Vectors to store predictions
  vector<Type> cots_pred(Year.size());
  vector<Type> slow_pred(Year.size());
  vector<Type> fast_pred(Year.size());
  
  // Initialize first time step with observations (with bounds)
  cots_pred(0) = CppAD::CondExpLt(cots_dat(0), eps, eps, cots_dat(0));
  slow_pred(0) = CppAD::CondExpLt(slow_dat(0), eps, eps, slow_dat(0));
  fast_pred(0) = CppAD::CondExpLt(fast_dat(0), eps, eps, fast_dat(0));
  
  // Time series simulation
  for(int t = 1; t < Year.size(); t++) {
    // 1. Temperature effect on COTS growth (Gaussian response with bounds)
    Type temp_diff = (sst_dat(t-1) - temp_opt) / (temp_range + eps);
    Type temp_effect = exp(-0.5 * pow(temp_diff, 2));
    temp_effect = temp_effect / (1 + temp_effect);  // Bound between 0 and 1
    
    // 2. Temperature-dependent feeding efficiency (simplified)
    Type temp_diff_feed = (sst_dat(t-1) - feed_temp_opt) / feed_temp_range;
    Type feed_efficiency = Type(0.2) + Type(0.8) * exp(-0.5 * temp_diff_feed * temp_diff_feed);
    
    // Type II functional responses
    Type total_coral = slow_pred(t-1) + fast_pred(t-1) + Type(0.1);
    Type pred_rate_slow = attack_rate_slow * feed_efficiency * slow_pred(t-1) / 
                         (Type(1.0) + handling_time * total_coral);
    Type pred_rate_fast = attack_rate_fast * feed_efficiency * fast_pred(t-1) / 
                         (Type(1.0) + handling_time * total_coral);
    
    // 3. COTS population dynamics
    Type density_effect = Type(1.0) - cots_pred(t-1) / K_cots;
    density_effect = CppAD::CondExpLt(density_effect, Type(0.0), 
                                     Type(0.0), density_effect);
    
    Type growth = r_cots * temp_effect * density_effect * cots_pred(t-1);
    Type mortality = Type(0.1) * cots_pred(t-1);
    
    cots_pred(t) = cots_pred(t-1) + growth + cotsimm_dat(t-1) - mortality;
    
    // 4. Coral dynamics with competition and improved stability
    Type total_cover = (slow_pred(t-1) + fast_pred(t-1)) / K_coral;
    Type competition = 1 - CppAD::CondExpGt(total_cover, Type(1),
                                           Type(1), total_cover);
    
    // Calculate changes
    Type slow_growth = r_slow * slow_pred(t-1) * competition;
    Type fast_growth = r_fast * fast_pred(t-1) * competition;
    Type slow_pred_loss = pred_rate_slow * cots_pred(t-1);
    Type fast_pred_loss = pred_rate_fast * cots_pred(t-1);
    
    // Update states with bounded predictions
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_pred_loss;
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_pred_loss;
    
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
  
  // Likelihood calculations (simplified)
  for(int t = 0; t < Year.size(); t++) {
    if(cots_dat(t) > eps && cots_pred(t) > eps) {
      nll -= dnorm(log(cots_dat(t)), log(cots_pred(t)), sigma_cots, true);
    }
    if(slow_dat(t) > eps && slow_pred(t) > eps) {
      nll -= dnorm(log(slow_dat(t)), log(slow_pred(t)), sigma_coral, true);
    }
    if(fast_dat(t) > eps && fast_pred(t) > eps) {
      nll -= dnorm(log(fast_dat(t)), log(fast_pred(t)), sigma_coral, true);
    }
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
