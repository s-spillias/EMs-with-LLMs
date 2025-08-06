#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Year);                  // Years of observation
  DATA_VECTOR(sst_dat);              // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);          // COTS immigration rate (individuals/m²/year) - forcing variable
  DATA_VECTOR(cots_dat);             // Observed COTS density (individuals/m²)
  DATA_VECTOR(slow_dat);             // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);             // Observed fast-growing coral cover (%)
  
  // Parameters for immigration model
  PARAMETER(log_sigma_imm);          // Log of immigration observation error SD
  PARAMETER(log_recruit_effect);     // Log of coral abundance effect on COTS recruitment
  
  // Initialize immigration prediction vector
  vector<Type> cotsimm_pred(Year.size());
  cotsimm_pred = cotsimm_dat;  // Use observed immigration as baseline
  
  // Derived parameter - bound recruitment effect
  Type recruit_effect = exp(log_recruit_effect) / (Type(1.0) + exp(log_recruit_effect));
  
  // Parameters
  PARAMETER(r_slow);                 // Growth rate of slow-growing corals
  PARAMETER(r_fast);                 // Growth rate of fast-growing corals
  PARAMETER(K_total);                // Total carrying capacity for corals
  PARAMETER(alpha_cots);             // COTS feeding rate
  PARAMETER(beta_cots);              // Half-saturation constant for COTS feeding
  PARAMETER(pref_fast);              // COTS preference for fast-growing coral
  PARAMETER(m_cots);                 // Natural mortality rate of COTS
  PARAMETER(q_cots);                 // Density-dependent mortality coefficient
  PARAMETER(temp_opt);               // Optimal temperature for coral growth
  PARAMETER(temp_range);             // Temperature range for coral growth
  PARAMETER(log_sigma_cots);         // Log of COTS observation error SD
  PARAMETER(log_sigma_slow);         // Log of slow coral observation error SD
  PARAMETER(log_sigma_fast);         // Log of fast coral observation error SD

  // Derived values
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_slow = exp(log_sigma_slow);
  Type sigma_fast = exp(log_sigma_fast);
  Type sigma_imm = exp(log_sigma_imm);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize predicted vectors
  vector<Type> cots_pred(Year.size());
  vector<Type> slow_pred(Year.size());
  vector<Type> fast_pred(Year.size());
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Initial conditions
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  cotsimm_pred(0) = Type(0.01);  // Start with small baseline immigration
  
  // Process model
  for(int t = 1; t < Year.size(); t++) {
    // Temperature effect on coral growth (Gaussian response)
    Type temp_effect = exp(-pow(sst_dat(t) - temp_opt, 2.0) / (2.0 * pow(temp_range, 2.0)));
    
    // Available space for coral growth
    Type space_available = K_total - (slow_pred(t-1) + fast_pred(t-1));
    space_available = CppAD::CondExpGt(space_available, Type(0), space_available, Type(0));
    
    // COTS functional response
    Type total_coral = slow_pred(t-1) + fast_pred(t-1) + eps;
    Type slow_prop = slow_pred(t-1) / total_coral;
    Type fast_prop = fast_pred(t-1) / total_coral;
    
    // Prey switching in COTS feeding
    Type feeding_slow = alpha_cots * cots_pred(t-1) * slow_pred(t-1) * (1 - pref_fast) * 
                       slow_prop / (beta_cots + total_coral);
    Type feeding_fast = alpha_cots * cots_pred(t-1) * fast_pred(t-1) * pref_fast * 
                       fast_prop / (beta_cots + total_coral);
    
    // State equations
    Type coral_effect = recruit_effect * total_coral / (Type(1.0) + recruit_effect * total_coral);
    Type recruitment = cotsimm_pred(t) * (Type(1.0) + coral_effect);
    cots_pred(t) = cots_pred(t-1) + recruitment - 
                   m_cots * cots_pred(t-1) - 
                   q_cots * pow(cots_pred(t-1), 2.0);
    
    slow_pred(t) = slow_pred(t-1) + 
                   r_slow * slow_pred(t-1) * (space_available/K_total) * temp_effect - 
                   feeding_slow;
    
    fast_pred(t) = fast_pred(t-1) + 
                   r_fast * fast_pred(t-1) * (space_available/K_total) * temp_effect - 
                   feeding_fast;
    
    // Ensure predictions stay positive
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), eps, cots_pred(t), eps);
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), eps, slow_pred(t), eps);
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), eps, fast_pred(t), eps);
  }
  
  // Observation model - lognormal errors
  for(int t = 0; t < Year.size(); t++) {
    // Add small constant to handle zeros in immigration data
    Type imm_obs = cotsimm_dat(t) + eps;
    Type imm_pred = cotsimm_pred(t) + eps;
    
    nll -= dnorm(log(cots_dat(t)), log(cots_pred(t)), sigma_cots, true);
    nll -= dnorm(log(slow_dat(t)), log(slow_pred(t)), sigma_slow, true);
    nll -= dnorm(log(fast_dat(t)), log(fast_pred(t)), sigma_fast, true);
    nll -= dnorm(log(imm_obs), log(imm_pred), sigma_imm, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(cotsimm_pred);  // Report immigration predictions (same as forcing)
  
  return nll;
}
