#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;
  // Data
  DATA_VECTOR(cots_dat);      // COTS abundance (individuals/m2)
  DATA_VECTOR(slow_dat);      // Slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);      // Fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);       // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);   // COTS immigration rate (individuals/m2/year)
  
  // Parameters
  PARAMETER(log_r_cots);      // Log COTS population growth rate (year^-1)
  PARAMETER(log_K_cots);      // Log COTS carrying capacity (individuals/m2)
  PARAMETER(log_alpha_slow);   // Log COTS feeding rate on slow corals (m2/individual/year)
  PARAMETER(log_alpha_fast);   // Log COTS feeding rate on fast corals (m2/individual/year)
  PARAMETER(log_r_slow);      // Log slow coral growth rate (year^-1)
  PARAMETER(log_r_fast);      // Log fast coral growth rate (year^-1)
  PARAMETER(log_temp_opt);     // Log optimal temperature for COTS (Celsius)
  PARAMETER(log_temp_width);   // Log temperature tolerance width (Celsius)
  PARAMETER(log_obs_sd);       // Log observation error SD

  // Transform parameters
  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type alpha_slow = exp(log_alpha_slow);
  Type alpha_fast = exp(log_alpha_fast);
  Type r_slow = exp(log_r_slow);
  Type r_fast = exp(log_r_fast);
  Type temp_opt = exp(log_temp_opt);
  Type temp_width = exp(log_temp_width);
  Type obs_sd = exp(log_obs_sd);

  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize vectors for predictions
  vector<Type> cots_pred(cots_dat.size());
  vector<Type> slow_pred(slow_dat.size());
  vector<Type> fast_pred(fast_dat.size());
  
  // Constants for numerical stability
  Type eps = Type(1e-8);
  Type max_coral = Type(100.0);
  Type min_sd = Type(0.1);  // Minimum observation error
  
  // Initial conditions (ensure positive values)
  cots_pred[0] = exp(log_r_cots) * cots_dat[0] + eps;  // Scale initial values
  slow_pred[0] = exp(log_r_slow) * slow_dat[0] + eps;
  fast_pred[0] = exp(log_r_fast) * fast_dat[0] + eps;
  
  // Time series predictions
  for(int t = 1; t < cots_dat.size(); t++) {
    // 1. Temperature effect on COTS (bounded Gaussian response)
    Type temp_diff = (sst_dat[t] - temp_opt) / (temp_width + eps);
    Type temp_effect = exp(-Type(0.5) * temp_diff * temp_diff);
    temp_effect = Type(0.01) + Type(0.99) * temp_effect;  // Bound between 0.01 and 1
    
    // 2. COTS population dynamics with bounded growth
    Type rel_density = cots_pred[t-1] / (K_cots + eps);
    Type dd_term = Type(1.0) / (Type(1.0) + exp(Type(2.0) * (rel_density - Type(0.5))));
    Type cots_growth = r_cots * cots_pred[t-1] * dd_term;
    cots_pred[t] = (cots_pred[t-1] + cots_growth * temp_effect + cotsimm_dat[t]) * 
                   (Type(1.0) / (Type(1.0) + exp(-Type(2.0) * cots_pred[t-1])));
    
    // 3. Coral dynamics with competition and COTS predation
    Type total_cover = slow_pred[t-1] + fast_pred[t-1];
    Type available_space = max_coral * (Type(1.0) - total_cover/max_coral);
    available_space = available_space / (Type(1.0) + exp(-Type(2.0) * available_space));
    
    // Bounded coral growth with smoother transitions
    Type slow_growth = r_slow * slow_pred[t-1] * available_space/max_coral;
    Type fast_growth = r_fast * fast_pred[t-1] * available_space/max_coral;
    
    // Update coral cover with predation using smoother functions
    Type pred_effect_slow = Type(1.0) / (Type(1.0) + alpha_slow * cots_pred[t]);
    Type pred_effect_fast = Type(1.0) / (Type(1.0) + alpha_fast * cots_pred[t]);
    
    slow_pred[t] = (slow_pred[t-1] + slow_growth) * pred_effect_slow;
    fast_pred[t] = (fast_pred[t-1] + fast_growth) * pred_effect_fast;
    
    // Ensure bounds using smoother functions
    slow_pred[t] = slow_pred[t] / (Type(1.0) + slow_pred[t]/max_coral);
    fast_pred[t] = fast_pred[t] / (Type(1.0) + fast_pred[t]/max_coral);
  }
  
  // Observation model using scaled normal distribution
  Type effective_sd = obs_sd + min_sd;  // Ensure positive SD
  
  for(int t = 0; t < cots_dat.size(); t++) {
    // Use scaled differences to improve numerical stability
    Type scale_cots = Type(0.5) * (cots_dat[t] + cots_pred[t]);
    Type scale_slow = Type(0.5) * (slow_dat[t] + slow_pred[t]);
    Type scale_fast = Type(0.5) * (fast_dat[t] + fast_pred[t]);
    
    nll -= dnorm(cots_dat[t]/scale_cots, 
                 cots_pred[t]/scale_cots, 
                 effective_sd, true);
    nll -= dnorm(slow_dat[t]/scale_slow, 
                 slow_pred[t]/scale_slow, 
                 effective_sd, true);
    nll -= dnorm(fast_dat[t]/scale_fast, 
                 fast_pred[t]/scale_fast, 
                 effective_sd, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(temp_opt);
  REPORT(temp_width);
  REPORT(r_cots);
  REPORT(K_cots);
  REPORT(alpha_slow);
  REPORT(alpha_fast);
  REPORT(r_slow);
  REPORT(r_fast);
  
  return nll;
}
