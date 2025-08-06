#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA SECTION
  DATA_VECTOR(Year);                  // Vector of years for time series data
  DATA_VECTOR(sst_dat);               // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);           // COTS larval immigration rate (individuals/m²/year)
  DATA_VECTOR(cots_dat);              // Adult COTS abundance (individuals/m²)
  DATA_VECTOR(fast_dat);              // Fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);              // Slow-growing coral cover (%)
  
  // PARAMETER SECTION
  PARAMETER(log_r_cots);              // Log of intrinsic growth rate of COTS (year⁻¹)
  PARAMETER(log_K_cots);              // Log of carrying capacity of COTS (individuals/m²)
  PARAMETER(log_temp_opt);            // Log of optimal temperature for COTS reproduction (°C)
  PARAMETER(log_temp_width);          // Log of temperature tolerance width (°C)
  PARAMETER(log_imm_effect);          // Log of effect of larval immigration on COTS population
  
  PARAMETER(log_a_fast);              // Log of attack rate on fast-growing coral (m²/individual/year)
  PARAMETER(log_a_slow);              // Log of attack rate on slow-growing coral (m²/individual/year)
  PARAMETER(log_h_fast);              // Log of handling time for fast-growing coral (year/%)
  PARAMETER(log_h_slow);              // Log of handling time for slow-growing coral (year/%)
  PARAMETER(log_pref_fast);           // Log of preference for fast-growing coral (dimensionless)
  
  PARAMETER(log_r_fast);              // Log of intrinsic growth rate of fast-growing coral (year⁻¹)
  PARAMETER(log_r_slow);              // Log of intrinsic growth rate of slow-growing coral (year⁻¹)
  PARAMETER(log_K_fast);              // Log of carrying capacity of fast-growing coral (%)
  PARAMETER(log_K_slow);              // Log of carrying capacity of slow-growing coral (%)
  PARAMETER(log_comp_coef);           // Log of competition coefficient between coral types
  
  PARAMETER(log_sd_cots);             // Log of observation error SD for COTS
  PARAMETER(log_sd_fast);             // Log of observation error SD for fast-growing coral
  PARAMETER(log_sd_slow);             // Log of observation error SD for slow-growing coral
  
  // Transform parameters to natural scale
  Type r_cots = 0.5;                  // Fixed intrinsic growth rate of COTS (year⁻¹)
  Type K_cots = 2.0;                  // Fixed carrying capacity of COTS (individuals/m²)
  Type temp_opt = 28.0;               // Fixed optimal temperature for COTS reproduction (°C)
  Type temp_width = 2.0;              // Fixed temperature tolerance width (°C)
  Type imm_effect = 1.0;              // Fixed effect of larval immigration on COTS population
  
  Type a_fast = 0.1;                  // Fixed attack rate on fast-growing coral (m²/individual/year)
  Type a_slow = 0.05;                 // Fixed attack rate on slow-growing coral (m²/individual/year)
  Type h_fast = 0.01;                 // Fixed handling time for fast-growing coral (year/%)
  Type h_slow = 0.02;                 // Fixed handling time for slow-growing coral (year/%)
  Type pref_fast = 0.8;               // Fixed preference for fast-growing coral
  Type pref_slow = 0.2;               // Fixed preference for slow-growing coral
  
  Type r_fast = 0.8;                  // Fixed intrinsic growth rate of fast-growing coral (year⁻¹)
  Type r_slow = 0.3;                  // Fixed intrinsic growth rate of slow-growing coral (year⁻¹)
  Type K_fast = 50.0;                 // Fixed carrying capacity of fast-growing coral (%)
  Type K_slow = 40.0;                 // Fixed carrying capacity of slow-growing coral (%)
  Type comp_coef = 0.5;               // Fixed competition coefficient between coral types
  
  // Fixed standard deviations
  Type sd_cots = 0.3;                 // Fixed observation error SD for COTS
  Type sd_fast = 0.3;                 // Fixed observation error SD for fast-growing coral
  Type sd_slow = 0.3;                 // Fixed observation error SD for slow-growing coral
  
  // Small constant to prevent division by zero
  Type eps = 0.001;
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Vectors to store predictions
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Initialize with first observation
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Time series simulation
  for (int t = 1; t < n; t++) {
    // 1. Temperature effect on COTS reproduction
    Type temp_effect = exp(-0.5 * pow((sst_dat(t-1) - temp_opt) / temp_width, 2));
    
    // 2. COTS population dynamics with temperature effect and immigration
    Type cots_growth = r_cots * cots_pred(t-1) * (1.0 - cots_pred(t-1) / K_cots) * temp_effect;
    Type immigration = imm_effect * cotsimm_dat(t-1);
    cots_pred(t) = cots_pred(t-1) + cots_growth + immigration;
    
    // Ensure COTS population stays positive
    if (cots_pred(t) < eps) cots_pred(t) = eps;
    
    // 3. Coral predation by COTS
    Type fast_consumed = pref_fast * a_fast * cots_pred(t-1) * fast_pred(t-1);
    Type slow_consumed = pref_slow * a_slow * cots_pred(t-1) * slow_pred(t-1);
    
    // Limit consumption to available coral
    if (fast_consumed > 0.5 * fast_pred(t-1)) fast_consumed = 0.5 * fast_pred(t-1);
    if (slow_consumed > 0.5 * slow_pred(t-1)) slow_consumed = 0.5 * slow_pred(t-1);
    
    // 4. Fast-growing coral dynamics
    Type fast_growth = r_fast * fast_pred(t-1) * (1.0 - fast_pred(t-1) / K_fast);
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_consumed;
    
    // Ensure coral cover stays positive
    if (fast_pred(t) < eps) fast_pred(t) = eps;
    
    // 5. Slow-growing coral dynamics
    Type slow_growth = r_slow * slow_pred(t-1) * (1.0 - slow_pred(t-1) / K_slow);
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_consumed;
    
    // Ensure coral cover stays positive
    if (slow_pred(t) < eps) slow_pred(t) = eps;
  }
  
  // Calculate negative log-likelihood
  for (int t = 0; t < n; t++) {
    // Add small constant to avoid log(0)
    Type cots_obs = cots_dat(t) + eps;
    Type cots_model = cots_pred(t) + eps;
    Type fast_obs = fast_dat(t) + eps;
    Type fast_model = fast_pred(t) + eps;
    Type slow_obs = slow_dat(t) + eps;
    Type slow_model = slow_pred(t) + eps;
    
    // Log-transform for lognormal likelihood
    Type log_cots_obs = log(cots_obs);
    Type log_cots_model = log(cots_model);
    Type log_fast_obs = log(fast_obs);
    Type log_fast_model = log(fast_model);
    Type log_slow_obs = log(slow_obs);
    Type log_slow_model = log(slow_model);
    
    // Add to negative log-likelihood
    nll -= dnorm(log_cots_obs, log_cots_model, sd_cots, true);
    nll -= dnorm(log_fast_obs, log_fast_model, sd_fast, true);
    nll -= dnorm(log_slow_obs, log_slow_model, sd_slow, true);
  }
  
  // Report predictions and objective function value
  ADREPORT(nll);
  REPORT(nll);
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(r_cots);
  REPORT(K_cots);
  REPORT(temp_opt);
  REPORT(temp_width);
  REPORT(imm_effect);
  REPORT(a_fast);
  REPORT(a_slow);
  REPORT(h_fast);
  REPORT(h_slow);
  REPORT(pref_fast);
  REPORT(pref_slow);
  REPORT(r_fast);
  REPORT(r_slow);
  REPORT(K_fast);
  REPORT(K_slow);
  REPORT(comp_coef);
  REPORT(sd_cots);
  REPORT(sd_fast);
  REPORT(sd_slow);
  
  return nll;
}
