#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA
  DATA_VECTOR(Year);                     // Vector of years for time series
  DATA_VECTOR(sst_dat);                  // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);              // COTS larval immigration rate (individuals/m²/year)
  DATA_VECTOR(cots_dat);                 // Observed COTS density (individuals/m²)
  DATA_VECTOR(fast_dat);                 // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);                 // Observed slow-growing coral cover (%)
  
  // PARAMETERS
  PARAMETER(log_r_cots);                 // Log of COTS intrinsic growth rate (year⁻¹)
  PARAMETER(log_K_cots);                 // Log of COTS carrying capacity (individuals/m²)
  PARAMETER(log_alpha_fast);             // Log of COTS predation rate on fast coral (% cover/COTS/year)
  PARAMETER(log_alpha_slow);             // Log of COTS predation rate on slow coral (% cover/COTS/year)
  PARAMETER(log_r_fast);                 // Log of fast coral intrinsic growth rate (year⁻¹)
  PARAMETER(log_r_slow);                 // Log of slow coral intrinsic growth rate (year⁻¹)
  PARAMETER(log_K_fast);                 // Log of fast coral carrying capacity (% cover)
  PARAMETER(log_K_slow);                 // Log of slow coral carrying capacity (% cover)
  PARAMETER(logit_beta_fast);            // Logit of competition coefficient of slow on fast coral (dimensionless)
  PARAMETER(logit_beta_slow);            // Logit of competition coefficient of fast on slow coral (dimensionless)
  PARAMETER(log_gamma);                  // Log of COTS mortality due to coral depletion (year⁻¹)
  PARAMETER(log_delta);                  // Log of temperature effect on COTS growth (°C⁻¹)
  PARAMETER(log_temp_opt);               // Log of optimal temperature for COTS (°C)
  PARAMETER(log_imm_survival);           // Log of survival rate of immigrant COTS larvae (dimensionless)
  
  PARAMETER(log_sigma_cots);             // Log of observation error SD for COTS
  PARAMETER(log_sigma_fast);             // Log of observation error SD for fast coral
  PARAMETER(log_sigma_slow);             // Log of observation error SD for slow coral
  
  // Transform parameters to natural scale
  Type r_cots = exp(log_r_cots);         // COTS intrinsic growth rate (year⁻¹)
  Type K_cots = exp(log_K_cots);         // COTS carrying capacity (individuals/m²)
  Type alpha_fast = exp(log_alpha_fast); // COTS predation rate on fast coral (% cover/COTS/year)
  Type alpha_slow = exp(log_alpha_slow); // COTS predation rate on slow coral (% cover/COTS/year)
  Type r_fast = exp(log_r_fast);         // Fast coral intrinsic growth rate (year⁻¹)
  Type r_slow = exp(log_r_slow);         // Slow coral intrinsic growth rate (year⁻¹)
  Type K_fast = exp(log_K_fast);         // Fast coral carrying capacity (% cover)
  Type K_slow = exp(log_K_slow);         // Slow coral carrying capacity (% cover)
  Type beta_fast = 1/(1+exp(-logit_beta_fast)); // Competition coefficient of slow on fast coral (dimensionless)
  Type beta_slow = 1/(1+exp(-logit_beta_slow)); // Competition coefficient of fast on slow coral (dimensionless)
  Type gamma = exp(log_gamma);           // COTS mortality due to coral depletion (year⁻¹)
  Type delta = exp(log_delta);           // Temperature effect on COTS growth (°C⁻¹)
  Type temp_opt = exp(log_temp_opt);     // Optimal temperature for COTS (°C)
  Type imm_survival = exp(log_imm_survival); // Survival rate of immigrant COTS larvae (dimensionless)
  
  // Fixed minimum standard deviations to prevent numerical issues
  Type min_sd = Type(0.5);
  Type sigma_cots = exp(log_sigma_cots);
  sigma_cots = sigma_cots < min_sd ? min_sd : sigma_cots;
  Type sigma_fast = exp(log_sigma_fast);
  sigma_fast = sigma_fast < min_sd ? min_sd : sigma_fast;
  Type sigma_slow = exp(log_sigma_slow);
  sigma_slow = sigma_slow < min_sd ? min_sd : sigma_slow;
  
  // Initialize negative log-likelihood
  Type nll = Type(0.0);
  
  // Initialize vectors for model predictions
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Initialize with first observation (ensure positive values)
  cots_pred(0) = cots_dat(0) + Type(0.01);
  fast_pred(0) = fast_dat(0) + Type(0.01);
  slow_pred(0) = slow_dat(0) + Type(0.01);
  
  // Process model: predict state variables through time
  for(int t = 1; t < n; t++) {
    // 1. COTS population dynamics - basic logistic growth with temperature effect
    Type temp_effect = Type(1.0);
    // Ensure temperature data is valid
    if (sst_dat(t-1) > Type(0.0)) {
      Type temp_diff = (sst_dat(t-1) - temp_opt) / delta;
      temp_effect = exp(-Type(0.5) * temp_diff * temp_diff);
    }
    
    // Simplified COTS growth with temperature effect and immigration
    Type cots_growth = r_cots * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1) / K_cots) * temp_effect;
    Type immigration = imm_survival * cotsimm_dat(t-1);
    
    // Update COTS population
    cots_pred(t) = cots_pred(t-1) + cots_growth + immigration;
    // Ensure positive values
    cots_pred(t) = cots_pred(t) < Type(0.01) ? Type(0.01) : cots_pred(t);
    
    // 2. Fast coral dynamics - basic logistic growth with COTS predation
    Type fast_growth = r_fast * fast_pred(t-1) * (Type(1.0) - fast_pred(t-1) / K_fast);
    Type fast_predation = alpha_fast * cots_pred(t-1);
    
    // Update fast coral
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_predation;
    // Ensure positive values
    fast_pred(t) = fast_pred(t) < Type(0.01) ? Type(0.01) : fast_pred(t);
    
    // 3. Slow coral dynamics - basic logistic growth with COTS predation
    Type slow_growth = r_slow * slow_pred(t-1) * (Type(1.0) - slow_pred(t-1) / K_slow);
    Type slow_predation = alpha_slow * cots_pred(t-1);
    
    // Update slow coral
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_predation;
    // Ensure positive values
    slow_pred(t) = slow_pred(t) < Type(0.01) ? Type(0.01) : slow_pred(t);
  }
  
  // Observation model: calculate negative log-likelihood
  for(int t = 0; t < n; t++) {
    // Add small constant to observations to avoid log(0)
    Type cots_obs = cots_dat(t) + Type(0.01);
    Type fast_obs = fast_dat(t) + Type(0.01);
    Type slow_obs = slow_dat(t) + Type(0.01);
    
    // Use normal distribution on log-transformed data
    nll -= dnorm(log(cots_obs), log(cots_pred(t)), sigma_cots, true);
    nll -= dnorm(log(fast_obs), log(fast_pred(t)), sigma_fast, true);
    nll -= dnorm(log(slow_obs), log(slow_pred(t)), sigma_slow, true);
  }
  
  // Report objective function value
  ADREPORT(nll);
  
  // Report predictions and parameters
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(r_cots);
  REPORT(K_cots);
  REPORT(alpha_fast);
  REPORT(alpha_slow);
  REPORT(r_fast);
  REPORT(r_slow);
  REPORT(K_fast);
  REPORT(K_slow);
  REPORT(beta_fast);
  REPORT(beta_slow);
  REPORT(gamma);
  REPORT(delta);
  REPORT(temp_opt);
  REPORT(imm_survival);
  REPORT(sigma_cots);
  REPORT(sigma_fast);
  REPORT(sigma_slow);
  
  return nll;
}
