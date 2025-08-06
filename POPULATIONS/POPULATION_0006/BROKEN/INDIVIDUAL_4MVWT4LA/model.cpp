#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA
  DATA_VECTOR(Year);                  // Years of observation
  DATA_VECTOR(cots_dat);              // Observed COTS abundance (individuals/m^2)
  DATA_VECTOR(fast_dat);              // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);              // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);               // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);           // COTS larval immigration rate (individuals/m^2/year)
  
  // PARAMETERS
  PARAMETER(r_cots);                  // Intrinsic growth rate of COTS population (year^-1)
  PARAMETER(K_cots);                  // Carrying capacity of COTS population (individuals/m^2)
  PARAMETER(m_cots);                  // Natural mortality rate of COTS (year^-1)
  PARAMETER(r_fast);                  // Intrinsic growth rate of fast-growing coral (year^-1)
  PARAMETER(K_fast);                  // Maximum cover of fast-growing coral (%)
  PARAMETER(r_slow);                  // Intrinsic growth rate of slow-growing coral (year^-1)
  PARAMETER(K_slow);                  // Maximum cover of slow-growing coral (%)
  PARAMETER(a_fast);                  // Attack rate of COTS on fast-growing coral (m^2/individual/year)
  PARAMETER(a_slow);                  // Attack rate of COTS on slow-growing coral (m^2/individual/year)
  PARAMETER(h_fast);                  // Handling time for COTS feeding on fast-growing coral (% cover)
  PARAMETER(h_slow);                  // Handling time for COTS feeding on slow-growing coral (% cover)
  PARAMETER(temp_opt);                // Optimal temperature for COTS recruitment (°C)
  PARAMETER(temp_width);              // Temperature range width for COTS recruitment (°C)
  PARAMETER(imm_effect);              // Effect of larval immigration on COTS recruitment (dimensionless)
  PARAMETER(competition);             // Competition coefficient between coral types (dimensionless)
  PARAMETER(bleach_threshold);        // Temperature threshold for coral bleaching (°C)
  PARAMETER(bleach_mortality_fast);   // Mortality rate of fast-growing coral during bleaching (year^-1)
  PARAMETER(bleach_mortality_slow);   // Mortality rate of slow-growing coral during bleaching (year^-1)
  PARAMETER(sigma_cots);              // Observation error standard deviation for COTS abundance (log scale)
  PARAMETER(sigma_fast);              // Observation error standard deviation for fast-growing coral cover (log scale)
  PARAMETER(sigma_slow);              // Observation error standard deviation for slow-growing coral cover (log scale)
  
  // New parameters for improved COTS outbreak dynamics
  PARAMETER(delay_effect);            // Strength of delayed density-dependent recruitment in COTS
  PARAMETER(allee_threshold);         // Population threshold for Allee effect in COTS reproduction
  PARAMETER(nutrient_effect);         // Effect of coral abundance on nutrient availability for COTS larvae
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Number of time steps
  int n_years = Year.size();
  
  // Vectors to store model predictions
  vector<Type> cots_pred(n_years);
  vector<Type> fast_pred(n_years);
  vector<Type> slow_pred(n_years);
  
  // Initialize with first year's data
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(0.01);
  Type sigma_cots_adj = sigma_cots + min_sigma;
  Type sigma_fast_adj = sigma_fast + min_sigma;
  Type sigma_slow_adj = sigma_slow + min_sigma;
  
  // Time series simulation
  for (int t = 1; t < n_years; t++) {
    // Previous time step values - ensure positive values
    Type cots_t0 = cots_pred(t-1);
    if (cots_t0 < eps) cots_t0 = eps;
    
    Type fast_t0 = fast_pred(t-1);
    if (fast_t0 < eps) fast_t0 = eps;
    
    Type slow_t0 = slow_pred(t-1);
    if (slow_t0 < eps) slow_t0 = eps;
    
    Type sst = sst_dat(t-1);
    Type cotsimm = cotsimm_dat(t-1);
    if (cotsimm < 0.0) cotsimm = 0.0;
    
    // Previous year's values for delayed effects (with safety check for t=1)
    Type cots_t1 = (t > 1) ? cots_pred(t-2) : cots_t0;
    if (cots_t1 < eps) cots_t1 = eps;
    
    // 1. Temperature effect on COTS recruitment (using fixed values to avoid instability)
    Type temp_effect = exp(-0.5 * pow((sst - 28.0) / 2.0, 2));
    
    // 2. COTS functional response (Type II) for predation on corals
    Type denominator = 1.0 + a_fast * 10.0 * fast_t0 + a_slow * 15.0 * slow_t0;
    Type pred_fast = (a_fast * fast_t0 * cots_t0) / denominator;
    Type pred_slow = (a_slow * slow_t0 * cots_t0) / denominator;
    
    // 3. Bleaching effect on corals
    Type bleach_effect = 1.0 / (1.0 + exp(-2.0 * (sst - 30.0)));
    
    // 4. COTS population dynamics with improved outbreak mechanisms
    
    // Basic logistic growth
    Type basic_growth = r_cots * cots_t0 * (1.0 - cots_t0 / 2.5);
    
    // Simple delayed recruitment effect
    Type delayed_effect_term = 0.0;
    if (t > 2) {
      // Use population from 2 years ago to influence current growth
      delayed_effect_term = delay_effect * cots_pred(t-3) * 0.1;
    }
    
    // Immigration effect
    Type imm_term = imm_effect * cotsimm / (1.0 + cotsimm);
    
    // Mortality term
    Type mortality = m_cots * cots_t0;
    
    // Update COTS abundance
    cots_pred(t) = cots_t0 + basic_growth + delayed_effect_term + imm_term - mortality;
    if (cots_pred(t) < eps) cots_pred(t) = eps;
    
    // 5. Coral dynamics
    // Fast-growing coral
    Type fast_growth = r_fast * fast_t0 * (1.0 - fast_t0 / 50.0);
    Type fast_bleaching = bleach_mortality_fast * bleach_effect * fast_t0;
    
    // Limit predation to available coral
    Type pred_fast_limited = pred_fast;
    if (pred_fast_limited > fast_t0) pred_fast_limited = fast_t0;
    
    // Update fast-growing coral cover
    fast_pred(t) = fast_t0 + fast_growth - pred_fast_limited - fast_bleaching;
    if (fast_pred(t) < eps) fast_pred(t) = eps;
    
    // Slow-growing coral
    Type slow_growth = r_slow * slow_t0 * (1.0 - slow_t0 / 30.0);
    Type slow_bleaching = bleach_mortality_slow * bleach_effect * slow_t0;
    
    // Limit predation to available coral
    Type pred_slow_limited = pred_slow;
    if (pred_slow_limited > slow_t0) pred_slow_limited = slow_t0;
    
    // Update slow-growing coral cover
    slow_pred(t) = slow_t0 + slow_growth - pred_slow_limited - slow_bleaching;
    if (slow_pred(t) < eps) slow_pred(t) = eps;
  }
  
  // Calculate negative log-likelihood using lognormal distribution
  for (int t = 0; t < n_years; t++) {
    // Add small constant to data and predictions to handle zeros
    Type cots_obs = cots_dat(t);
    if (cots_obs < eps) cots_obs = eps;
    
    Type cots_mod = cots_pred(t);
    if (cots_mod < eps) cots_mod = eps;
    
    Type fast_obs = fast_dat(t);
    if (fast_obs < eps) fast_obs = eps;
    
    Type fast_mod = fast_pred(t);
    if (fast_mod < eps) fast_mod = eps;
    
    Type slow_obs = slow_dat(t);
    if (slow_obs < eps) slow_obs = eps;
    
    Type slow_mod = slow_pred(t);
    if (slow_mod < eps) slow_mod = eps;
    
    // Log-normal likelihood for COTS abundance
    nll -= dnorm(log(cots_obs), log(cots_mod), sigma_cots_adj, true);
    
    // Log-normal likelihood for fast-growing coral cover
    nll -= dnorm(log(fast_obs), log(fast_mod), sigma_fast_adj, true);
    
    // Log-normal likelihood for slow-growing coral cover
    nll -= dnorm(log(slow_obs), log(slow_mod), sigma_slow_adj, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
