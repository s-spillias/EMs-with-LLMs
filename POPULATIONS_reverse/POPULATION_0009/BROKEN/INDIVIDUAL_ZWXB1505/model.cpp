#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(cots_dat);    // COTS density (individuals/m2)
  DATA_VECTOR(slow_dat);    // Slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);    // Fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);     // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat); // COTS immigration rate (individuals/m2/year)
  
  // Parameters
  PARAMETER(log_r);         // COTS intrinsic growth rate (log scale)
  PARAMETER(log_K);         // COTS carrying capacity (log scale)
  PARAMETER(temp_opt);      // Optimal temperature for COTS growth
  PARAMETER(temp_width);    // Temperature tolerance width
  PARAMETER(log_a_slow);    // Attack rate on slow corals (log scale)
  PARAMETER(log_a_fast);    // Attack rate on fast corals (log scale)
  PARAMETER(log_g_slow);    // Growth rate of slow corals (log scale)
  PARAMETER(log_g_fast);    // Growth rate of fast corals (log scale)
  PARAMETER(log_sd_cots);   // Observation error SD for COTS (log scale)
  PARAMETER(log_sd_coral);  // Observation error SD for corals (log scale)

  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Transform parameters
  Type r = exp(log_r);
  Type K = exp(log_K);
  Type a_slow = exp(log_a_slow);
  Type a_fast = exp(log_a_fast);
  Type g_slow = exp(log_g_slow);
  Type g_fast = exp(log_g_fast);
  Type sd_cots = exp(log_sd_cots);
  Type sd_coral = exp(log_sd_coral);
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Vectors for predictions
  vector<Type> cots_pred(cots_dat.size());
  vector<Type> slow_pred(slow_dat.size());
  vector<Type> fast_pred(fast_dat.size());
  
  // Initial conditions
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Process model
  for(int t = 1; t < cots_dat.size(); t++) {
    // 1. Temperature effect on COTS growth (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t) - temp_opt)/temp_width, 2));
    
    // 2. COTS population dynamics with temperature-dependent growth
    Type cots_growth = r * temp_effect * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1)/(K + eps));
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t);
    
    // 3. Coral dynamics with COTS predation
    Type total_coral = slow_pred(t-1) + fast_pred(t-1) + eps;
    Type pred_pressure = Type(1.0)/(Type(1.0) + exp(-Type(2.0) * cots_pred(t-1))); // Sigmoid predation pressure
    
    // Calculate losses with smooth transitions
    Type slow_loss = a_slow * pred_pressure * slow_pred(t-1)/total_coral;
    Type fast_loss = a_fast * pred_pressure * fast_pred(t-1)/total_coral;
    
    // Logistic growth with smooth bounded losses
    Type slow_growth = g_slow * slow_pred(t-1) * (Type(100.0) - slow_pred(t-1))/Type(100.0);
    Type fast_growth = g_fast * fast_pred(t-1) * (Type(100.0) - fast_pred(t-1))/Type(100.0);
    
    slow_pred(t) = slow_pred(t-1) + slow_growth;
    fast_pred(t) = fast_pred(t-1) + fast_growth;
    
    // Apply bounded losses with smooth transitions
    slow_pred(t) -= slow_loss;
    fast_pred(t) -= fast_loss;
    
    // Ensure predictions stay within biological bounds using sigmoid functions
    Type smooth_scale = Type(10.0);
    
    // Lower bound
    cots_pred(t) = eps + (cots_pred(t) - eps)/(Type(1.0) + exp(-smooth_scale * (cots_pred(t) - eps)));
    slow_pred(t) = eps + (slow_pred(t) - eps)/(Type(1.0) + exp(-smooth_scale * (slow_pred(t) - eps)));
    fast_pred(t) = eps + (fast_pred(t) - eps)/(Type(1.0) + exp(-smooth_scale * (fast_pred(t) - eps)));
    
    // Upper bound
    Type cots_max = K * Type(2.0);
    cots_pred(t) = cots_pred(t)/(Type(1.0) + exp(smooth_scale * (cots_pred(t) - cots_max)));
    slow_pred(t) = slow_pred(t)/(Type(1.0) + exp(smooth_scale * (slow_pred(t) - Type(100.0))));
    fast_pred(t) = fast_pred(t)/(Type(1.0) + exp(smooth_scale * (fast_pred(t) - Type(100.0))));
  }
  
  // Observation model with robust error handling
  for(int t = 0; t < cots_dat.size(); t++) {
    // Log-normal likelihood for COTS
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sd_cots + Type(0.1), true);
    
    // Normal likelihood for coral cover
    nll -= dnorm(slow_dat(t), slow_pred(t), sd_coral + Type(0.1), true);
    nll -= dnorm(fast_dat(t), fast_pred(t), sd_coral + Type(0.1), true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(r);
  REPORT(K);
  REPORT(a_slow);
  REPORT(a_fast);
  REPORT(g_slow);
  REPORT(g_fast);
  
  return nll;
}
