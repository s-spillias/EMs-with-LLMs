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
  PARAMETER(cots_threshold);          // Population threshold for COTS outbreak dynamics (individuals/m^2)
  PARAMETER(allee_strength);          // Strength of Allee effect in COTS reproduction (dimensionless)
  
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
    // Previous time step values
    Type cots_t0 = cots_pred(t-1);
    Type fast_t0 = fast_pred(t-1);
    Type slow_t0 = slow_pred(t-1);
    Type sst = sst_dat(t-1);
    Type cotsimm = cotsimm_dat(t-1);
    
    // Ensure positive values for state variables
    if (cots_t0 < eps) cots_t0 = eps;
    if (fast_t0 < eps) fast_t0 = eps;
    if (slow_t0 < eps) slow_t0 = eps;
    
    // 1. Temperature effect on COTS recruitment
    // Gaussian response curve for temperature effect on COTS recruitment
    Type temp_effect = exp(-0.5 * pow((sst - temp_opt) / temp_width, 2));
    
    // 2. COTS functional response (Type II) for predation on corals
    // Holling Type II functional response for COTS predation on fast-growing coral
    Type pred_fast = (a_fast * fast_t0 * cots_t0) / (1.0 + a_fast * h_fast * fast_t0 + a_slow * h_slow * slow_t0 + eps);
    
    // Holling Type II functional response for COTS predation on slow-growing coral
    Type pred_slow = (a_slow * slow_t0 * cots_t0) / (1.0 + a_fast * h_fast * fast_t0 + a_slow * h_slow * slow_t0 + eps);
    
    // 3. Bleaching effect on corals
    // Smooth transition function for bleaching effect
    Type bleach_effect = 1.0 / (1.0 + exp(-2.0 * (sst - bleach_threshold)));
    
    // 4. COTS population dynamics with density-dependent reproduction
    // Basic logistic growth
    Type cots_growth = r_cots * cots_t0 * (1.0 - cots_t0 / K_cots) * temp_effect;
    
    // Apply threshold-dependent boost to reproduction if above threshold
    if (cots_t0 > cots_threshold) {
      // Simple fixed multiplier for reproduction when above threshold
      cots_growth *= 1.5;
    }
    
    // Immigration effect
    Type imm_term = imm_effect * cotsimm;
    
    // Food limitation effect (COTS mortality increases when coral cover is low)
    Type total_coral = fast_t0 + slow_t0 + eps;
    Type food_limitation = m_cots * (1.0 + 0.5 / total_coral);
    
    // Update COTS abundance with bounds checking
    cots_pred(t) = cots_t0 + cots_growth - food_limitation * cots_t0 + imm_term;
    if (cots_pred(t) < eps) cots_pred(t) = eps;
    if (cots_pred(t) > K_cots * 1.5) cots_pred(t) = K_cots * 1.5;
    
    // 5. Coral dynamics
    // Fast-growing coral dynamics with logistic growth, competition, predation, and bleaching
    Type fast_growth = r_fast * fast_t0 * (1.0 - (fast_t0 + competition * slow_t0) / K_fast);
    Type fast_bleaching = bleach_mortality_fast * bleach_effect * fast_t0;
    
    // Update fast-growing coral cover with bounds checking
    fast_pred(t) = fast_t0 + fast_growth - pred_fast - fast_bleaching;
    if (fast_pred(t) < eps) fast_pred(t) = eps;
    if (fast_pred(t) > K_fast) fast_pred(t) = K_fast;
    
    // Slow-growing coral dynamics with logistic growth, competition, predation, and bleaching
    Type slow_growth = r_slow * slow_t0 * (1.0 - (slow_t0 + competition * fast_t0) / K_slow);
    Type slow_bleaching = bleach_mortality_slow * bleach_effect * slow_t0;
    
    // Update slow-growing coral cover with bounds checking
    slow_pred(t) = slow_t0 + slow_growth - pred_slow - slow_bleaching;
    if (slow_pred(t) < eps) slow_pred(t) = eps;
    if (slow_pred(t) > K_slow) slow_pred(t) = K_slow;
  }
  
  // Calculate negative log-likelihood using lognormal distribution
  for (int t = 0; t < n_years; t++) {
    // Add small constant to data and predictions to handle zeros
    Type cots_obs = cots_dat(t) + eps;
    Type cots_mod = cots_pred(t) + eps;
    Type fast_obs = fast_dat(t) + eps;
    Type fast_mod = fast_pred(t) + eps;
    Type slow_obs = slow_dat(t) + eps;
    Type slow_mod = slow_pred(t) + eps;
    
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
