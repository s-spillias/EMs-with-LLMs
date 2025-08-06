#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(cots_dat);        // COTS abundance observations (individuals/m2)
  DATA_VECTOR(slow_dat);        // Slow-growing coral cover observations (%)
  DATA_VECTOR(fast_dat);        // Fast-growing coral cover observations (%)
  DATA_VECTOR(sst_dat);         // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);     // COTS immigration rate (individuals/m2/year)
  
  // Parameters
  PARAMETER(log_r);             // COTS intrinsic growth rate (log scale)
  PARAMETER(log_K);             // COTS carrying capacity (log scale)
  PARAMETER(log_alpha_fast);    // COTS feeding rate on fast coral (log scale)
  PARAMETER(log_alpha_slow);    // COTS feeding rate on slow coral (log scale)
  PARAMETER(log_beta_fast);     // Fast coral growth rate (log scale)
  PARAMETER(log_beta_slow);     // Slow coral growth rate (log scale)
  PARAMETER(log_temp_opt);      // Optimal temperature for coral growth (log scale)
  PARAMETER(log_sigma_cots);    // Observation error SD for COTS (log scale)
  PARAMETER(log_sigma_coral);   // Observation error SD for coral (log scale)

  // Transform parameters
  Type r = exp(log_r);
  Type K = exp(log_K);
  Type alpha_fast = exp(log_alpha_fast);
  Type alpha_slow = exp(log_alpha_slow);
  Type beta_fast = exp(log_beta_fast);
  Type beta_slow = exp(log_beta_slow);
  Type temp_opt = exp(log_temp_opt);
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_coral = exp(log_sigma_coral);

  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  const Type eps = Type(1e-8);
  
  // Vectors for predicted values
  vector<Type> cots_pred(cots_dat.size());
  vector<Type> slow_pred(slow_dat.size());
  vector<Type> fast_pred(fast_dat.size());
  
  // Initial values with bounds
  cots_pred(0) = std::max(Type(0.1), std::min(cots_dat(0), K));
  slow_pred(0) = std::max(Type(1.0), std::min(slow_dat(0), Type(100.0)));
  fast_pred(0) = std::max(Type(1.0), std::min(fast_dat(0), Type(100.0)));

  // Process model
  for(int t = 1; t < cots_dat.size(); t++) {
    // 1. Temperature effect on coral growth (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t) - temp_opt)/Type(2.0), 2));
    
    // 2. COTS population dynamics with scaled growth
    Type rel_density = cots_pred(t-1)/K;
    Type dd_term = std::max(Type(0.0), std::min(Type(1.0) - rel_density, Type(1.0)));
    Type cots_growth = r * cots_pred(t-1) * dd_term / Type(10.0); // Scale down growth rate
    cots_pred(t) = std::max(Type(0.1), std::min(cots_pred(t-1) + cots_growth + cotsimm_dat(t), K));
    
    // 3. Coral dynamics
    // Scaled consumption and growth
    Type fast_consumed = alpha_fast * cots_pred(t-1) * fast_pred(t-1)/Type(1000.0); // Scale down consumption
    Type slow_consumed = alpha_slow * cots_pred(t-1) * slow_pred(t-1)/Type(1000.0);
    
    Type fast_growth = beta_fast * temp_effect * (Type(100.0) - fast_pred(t-1))/Type(10.0); // Scale down growth
    Type slow_growth = beta_slow * temp_effect * (Type(100.0) - slow_pred(t-1))/Type(10.0);
    
    // Update with bounds and minimum values
    fast_pred(t) = std::max(Type(1.0), std::min(fast_pred(t-1) + fast_growth - fast_consumed, Type(100.0)));
    slow_pred(t) = std::max(Type(1.0), std::min(slow_pred(t-1) + slow_growth - slow_consumed, Type(100.0)));
  }
  
  // Simple observation model
  for(int t = 0; t < cots_dat.size(); t++) {
    // Normal likelihood on log scale for COTS
    nll -= dnorm(log(cots_dat(t) + eps), 
                 log(cots_pred(t) + eps), 
                 sigma_cots, 
                 true);
    
    // Normal likelihood for coral cover
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_coral, true);
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_coral, true);
  }

  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(r);
  REPORT(K);
  REPORT(alpha_fast);
  REPORT(alpha_slow);
  REPORT(beta_fast);
  REPORT(beta_slow);
  REPORT(temp_opt);
  
  return nll;
}
