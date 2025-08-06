#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs
  DATA_VECTOR(Year);           // Time vector (years)
  DATA_VECTOR(cots_dat);       // Observed COTS density (individuals/m²)
  DATA_VECTOR(slow_dat);       // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);       // Observed fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);        // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);    // COTS immigration rate (individuals/m²/year)
  
  // Parameters
  PARAMETER(log_r_slow);       // Log of slow coral intrinsic growth rate
  PARAMETER(log_r_fast);       // Log of fast coral intrinsic growth rate
  PARAMETER(log_K_slow);       // Log of slow coral carrying capacity
  PARAMETER(log_K_fast);       // Log of fast coral carrying capacity
  PARAMETER(log_alpha_slow);   // Log of COTS attack rate on slow coral
  PARAMETER(log_alpha_fast);   // Log of COTS attack rate on fast coral
  PARAMETER(log_h_slow);       // Log of handling time for slow coral
  PARAMETER(log_h_fast);       // Log of handling time for fast coral
  PARAMETER(log_m);            // Log of COTS density-dependent mortality
  PARAMETER(log_T_opt);        // Log of optimal temperature for COTS
  PARAMETER(log_sigma_T);      // Log of temperature tolerance width
  PARAMETER(log_obs_sd);       // Log of observation error SD
  PARAMETER(log_R50_slow);     // Log of slow coral cover at 50% recruitment facilitation
  
  // Transform parameters
  Type r_slow = exp(log_r_slow);
  Type r_fast = exp(log_r_fast);
  Type K_slow = exp(log_K_slow);
  Type K_fast = exp(log_K_fast);
  Type alpha_slow = exp(log_alpha_slow);
  Type alpha_fast = exp(log_alpha_fast);
  Type h_slow = exp(log_h_slow);
  Type h_fast = exp(log_h_fast);
  Type m = exp(log_m);
  Type T_opt = exp(log_T_opt);
  Type sigma_T = exp(log_sigma_T);
  Type obs_sd = exp(log_obs_sd);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize vectors for predictions
  vector<Type> cots_pred(Year.size());
  vector<Type> slow_pred(Year.size());
  vector<Type> fast_pred(Year.size());
  vector<Type> cotsimm_pred(Year.size());
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Initial conditions
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  cotsimm_pred(0) = cotsimm_dat(0);
  
  // Process model
  for(int t = 1; t < Year.size(); t++) {
    // Temperature scaling function (Gaussian)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t) - T_opt) / sigma_T, 2));
    
    // Total coral cover (competition effect)
    Type total_cover = slow_pred(t-1) + fast_pred(t-1);
    
    // Holling Type II functional responses
    Type f_slow = (alpha_slow * slow_pred(t-1)) / 
                 (1 + alpha_slow * h_slow * slow_pred(t-1) + 
                  alpha_fast * h_fast * fast_pred(t-1));
    Type f_fast = (alpha_fast * fast_pred(t-1)) / 
                 (1 + alpha_slow * h_slow * slow_pred(t-1) + 
                  alpha_fast * h_fast * fast_pred(t-1));
    
    // COTS dynamics
    // Model immigration as temperature-dependent process
    cotsimm_pred(t) = temp_effect * std::max(Type(0), cotsimm_pred(t-1));
    
    cots_pred(t) = cots_pred(t-1) + 
                   temp_effect * (f_slow + f_fast) * cots_pred(t-1) -
                   m * pow(cots_pred(t-1), 2) +
                   cotsimm_pred(t);
    cots_pred(t) = std::max(cots_pred(t), eps);
    
    // Coral dynamics with space limitation and recruitment facilitation
    slow_pred(t) = slow_pred(t-1) + 
                   r_slow * slow_pred(t-1) * (1 - total_cover/K_slow) * 
                   (slow_pred(t-1)/(slow_pred(t-1) + exp(log_R50_slow))) -
                   f_slow * cots_pred(t-1);
    slow_pred(t) = std::max(slow_pred(t), eps);
    
    fast_pred(t) = fast_pred(t-1) + 
                   r_fast * fast_pred(t-1) * (1 - total_cover/K_fast) -
                   f_fast * cots_pred(t-1);
    fast_pred(t) = std::max(fast_pred(t), eps);
  }
  
  // Observation model (lognormal)
  for(int t = 0; t < Year.size(); t++) {
    nll -= dnorm(log(cots_dat(t)), log(cots_pred(t)), obs_sd, true);
    nll -= dnorm(log(slow_dat(t)), log(slow_pred(t)), obs_sd, true);
    nll -= dnorm(log(fast_dat(t)), log(fast_pred(t)), obs_sd, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(cotsimm_pred);
  
  return nll;
}
