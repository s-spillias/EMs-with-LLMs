#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA SECTION
  DATA_VECTOR(Year);                    // Years of observation
  DATA_VECTOR(cots_dat);                // Observed COTS density (individuals/m2)
  DATA_VECTOR(slow_dat);                // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);                // Observed fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);                 // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);             // COTS immigration rate (individuals/m2/year)
  
  // PARAMETER SECTION
  PARAMETER(log_r_cots);                // Log of COTS intrinsic growth rate (year^-1)
  PARAMETER(log_K_cots);                // Log of COTS carrying capacity (individuals/m2)
  PARAMETER(log_alpha_slow);            // Log of COTS predation rate on slow-growing corals (m2/individual/year)
  PARAMETER(log_alpha_fast);            // Log of COTS predation rate on fast-growing corals (m2/individual/year)
  PARAMETER(log_pref_fast);             // Log of COTS preference for fast-growing corals (dimensionless)
  PARAMETER(log_r_slow);                // Log of slow-growing coral intrinsic growth rate (year^-1)
  PARAMETER(log_r_fast);                // Log of fast-growing coral intrinsic growth rate (year^-1)
  PARAMETER(log_K_slow);                // Log of slow-growing coral carrying capacity (%)
  PARAMETER(log_K_fast);                // Log of fast-growing coral carrying capacity (%)
  PARAMETER(log_temp_opt);              // Log of optimal temperature for COTS (°C)
  PARAMETER(log_temp_width);            // Log of temperature tolerance width for COTS (°C)
  PARAMETER(log_temp_coral_threshold);  // Log of temperature threshold for coral stress (°C)
  PARAMETER(log_temp_coral_slope);      // Log of slope of temperature effect on coral mortality (dimensionless)
  PARAMETER(log_cots_min_coral);        // Log of minimum coral cover needed for COTS survival (%)
  
  // Observation error standard deviations
  PARAMETER(log_sigma_cots);            // Log of SD for COTS observations
  PARAMETER(log_sigma_slow);            // Log of SD for slow-growing coral observations
  PARAMETER(log_sigma_fast);            // Log of SD for fast-growing coral observations
  
  // Transform parameters to natural scale
  Type r_cots = exp(log_r_cots);                // COTS intrinsic growth rate (year^-1)
  Type K_cots = exp(log_K_cots);                // COTS carrying capacity (individuals/m2)
  Type alpha_slow = exp(log_alpha_slow);        // COTS predation rate on slow-growing corals (m2/individual/year)
  Type alpha_fast = exp(log_alpha_fast);        // COTS predation rate on fast-growing corals (m2/individual/year)
  Type pref_fast = exp(log_pref_fast);          // COTS preference for fast-growing corals (dimensionless)
  Type r_slow = exp(log_r_slow);                // Slow-growing coral intrinsic growth rate (year^-1)
  Type r_fast = exp(log_r_fast);                // Fast-growing coral intrinsic growth rate (year^-1)
  Type K_slow = exp(log_K_slow);                // Slow-growing coral carrying capacity (%)
  Type K_fast = exp(log_K_fast);                // Fast-growing coral carrying capacity (%)
  Type temp_opt = exp(log_temp_opt);            // Optimal temperature for COTS (°C)
  Type temp_width = exp(log_temp_width);        // Temperature tolerance width for COTS (°C)
  Type temp_coral_threshold = exp(log_temp_coral_threshold);  // Temperature threshold for coral stress (°C)
  Type temp_coral_slope = exp(log_temp_coral_slope);          // Slope of temperature effect on coral mortality (dimensionless)
  Type cots_min_coral = exp(log_cots_min_coral);              // Minimum coral cover needed for COTS survival (%)
  
  // Observation error standard deviations
  Type sigma_cots = exp(log_sigma_cots);        // SD for COTS observations
  Type sigma_slow = exp(log_sigma_slow);        // SD for slow-growing coral observations
  Type sigma_fast = exp(log_sigma_fast);        // SD for fast-growing coral observations
  
  // Initialize negative log-likelihood
  Type nll = 0;
  
  // Vectors to store predictions
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  
  // Initialize with first observation
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Process model: predict state variables through time
  for(int t = 1; t < n; t++) {
    // Simple logistic growth for COTS with immigration
    cots_pred(t) = cots_pred(t-1) + r_cots * cots_pred(t-1) * (1.0 - cots_pred(t-1) / K_cots) + cotsimm_dat(t-1);
    
    // Simple logistic growth for corals with COTS predation
    slow_pred(t) = slow_pred(t-1) + r_slow * slow_pred(t-1) * (1.0 - slow_pred(t-1) / K_slow) - alpha_slow * cots_pred(t-1);
    fast_pred(t) = fast_pred(t-1) + r_fast * fast_pred(t-1) * (1.0 - fast_pred(t-1) / K_fast) - alpha_fast * pref_fast * cots_pred(t-1);
    
    // Ensure predictions are non-negative
    cots_pred(t) = cots_pred(t) < 0 ? 0 : cots_pred(t);
    slow_pred(t) = slow_pred(t) < 0 ? 0 : slow_pred(t);
    fast_pred(t) = fast_pred(t) < 0 ? 0 : fast_pred(t);
  }
  
  // Observation model: calculate negative log-likelihood
  for(int t = 0; t < n; t++) {
    // Add small constant to prevent log(0)
    Type cots_obs = cots_dat(t) + 0.001;
    Type slow_obs = slow_dat(t) + 0.001;
    Type fast_obs = fast_dat(t) + 0.001;
    
    Type cots_pred_t = cots_pred(t) + 0.001;
    Type slow_pred_t = slow_pred(t) + 0.001;
    Type fast_pred_t = fast_pred(t) + 0.001;
    
    // Normal distribution on log scale
    nll -= dnorm(log(cots_obs), log(cots_pred_t), sigma_cots, true);
    nll -= dnorm(log(slow_obs), log(slow_pred_t), sigma_slow, true);
    nll -= dnorm(log(fast_obs), log(fast_pred_t), sigma_fast, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  // Report transformed parameters
  REPORT(r_cots);
  REPORT(K_cots);
  REPORT(alpha_slow);
  REPORT(alpha_fast);
  REPORT(pref_fast);
  REPORT(r_slow);
  REPORT(r_fast);
  REPORT(K_slow);
  REPORT(K_fast);
  REPORT(temp_opt);
  REPORT(temp_width);
  REPORT(temp_coral_threshold);
  REPORT(temp_coral_slope);
  REPORT(cots_min_coral);
  REPORT(sigma_cots);
  REPORT(sigma_slow);
  REPORT(sigma_fast);
  
  return nll;
}
