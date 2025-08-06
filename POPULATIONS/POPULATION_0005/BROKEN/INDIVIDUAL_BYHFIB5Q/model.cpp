#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Year);                  // Time series years
  DATA_VECTOR(sst_dat);              // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);          // COTS larval immigration (individuals/m2/year)
  DATA_VECTOR(cots_dat);             // Observed COTS density (individuals/m2)
  DATA_VECTOR(fast_dat);             // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);             // Observed slow-growing coral cover (%)
  
  // Parameters
  PARAMETER(log_r_cots);             // COTS population growth rate
  PARAMETER(log_K_cots);             // COTS carrying capacity
  PARAMETER(log_temp_effect);        // Temperature effect on COTS recruitment
  PARAMETER(logit_fast_pref);        // Preference for fast-growing coral
  PARAMETER(log_handling_time);      // Handling time for coral consumption
  PARAMETER(log_attack_rate);        // Attack rate on coral
  PARAMETER(log_interference);       // Interference competition coefficient
  PARAMETER(log_r_fast);             // Fast coral growth rate
  PARAMETER(log_r_slow);             // Slow coral growth rate
  PARAMETER(log_obs_sd_cots);        // Observation error SD for COTS
  PARAMETER(log_obs_sd_coral);       // Observation error SD for coral
  
  // Transform parameters
  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type temp_effect = exp(log_temp_effect);
  Type fast_pref = invlogit(logit_fast_pref);
  Type handling_time = exp(log_handling_time);
  Type attack_rate = exp(log_attack_rate);
  Type r_fast = exp(log_r_fast);
  Type r_slow = exp(log_r_slow);
  Type obs_sd_cots = exp(log_obs_sd_cots);
  Type obs_sd_coral = exp(log_obs_sd_coral);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize predicted vectors
  vector<Type> cots_pred(Year.size());
  vector<Type> fast_pred(Year.size());
  vector<Type> slow_pred(Year.size());
  
  // Set initial conditions
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Process model
  for(int t = 1; t < Year.size(); t++) {
    // 1. Temperature-dependent recruitment
    Type temp_recruitment = cotsimm_dat(t-1) * exp(temp_effect * (sst_dat(t-1) - Type(26.0)));
    
    // 2. Coral consumption using multi-species functional response
    Type total_coral = fast_pred(t-1) + slow_pred(t-1) + eps;
    Type fast_proportion = fast_pred(t-1) / total_coral;
    Type slow_proportion = slow_pred(t-1) / total_coral;
    
    // 3. Feeding preferences with simplified density-dependence
    Type interference = exp(log_interference);
    Type density_factor = Type(1.0) / (Type(1.0) + interference * cots_pred(t-1));
    
    // Calculate consumption rates
    Type consumption_fast = (attack_rate * density_factor * fast_pref * fast_pred(t-1) * cots_pred(t-1)) / 
                          (Type(1.0) + handling_time * total_coral);
    Type consumption_slow = (attack_rate * density_factor * (Type(1.0) - fast_pref) * slow_pred(t-1) * cots_pred(t-1)) /
                          (Type(1.0) + handling_time * total_coral);
    
    // 4. COTS population dynamics
    Type growth_term = r_cots * (Type(1.0) - cots_pred(t-1)/K_cots);
    cots_pred(t) = cots_pred(t-1) * (Type(1.0) + growth_term) + temp_recruitment;
    
    // 5. Coral dynamics
    Type fast_growth = r_fast * (Type(1.0) - fast_pred(t-1)/Type(100.0));
    Type slow_growth = r_slow * (Type(1.0) - slow_pred(t-1)/Type(100.0));
    
    fast_pred(t) = fast_pred(t-1) + fast_pred(t-1) * fast_growth - consumption_fast;
    slow_pred(t) = slow_pred(t-1) + slow_pred(t-1) * slow_growth - consumption_slow;
    
    // 6. Simple bounds on predictions
    cots_pred(t) = exp(log(cots_pred(t) + eps));
    fast_pred(t) = exp(log(fast_pred(t) + eps));
    slow_pred(t) = exp(log(slow_pred(t) + eps));
  }
  
  // Observation model
  for(int t = 0; t < Year.size(); t++) {
    if(cots_dat(t) > eps && cots_pred(t) > eps) {
      // Log-normal observation model for COTS when both values are positive
      nll -= dnorm(log(cots_dat(t)), log(cots_pred(t)), obs_sd_cots, true);
    }
    
    // Normal observation model for coral cover with bounds
    if(fast_dat(t) >= Type(0.0) && fast_dat(t) <= Type(100.0)) {
      nll -= dnorm(fast_dat(t), fast_pred(t), obs_sd_coral, true);
    }
    if(slow_dat(t) >= Type(0.0) && slow_dat(t) <= Type(100.0)) {
      nll -= dnorm(slow_dat(t), slow_pred(t), obs_sd_coral, true);
    }
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
