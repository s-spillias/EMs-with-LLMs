#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA
  DATA_VECTOR(Year);                  // Years of observation
  DATA_VECTOR(sst_dat);               // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);           // COTS immigration rate (individuals/m^2/year)
  DATA_VECTOR(cots_dat);              // Observed COTS abundance (individuals/m^2)
  DATA_VECTOR(slow_dat);              // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);              // Observed fast-growing coral cover (%)
  
  // PARAMETERS
  PARAMETER(r_cots);                  // Intrinsic growth rate of COTS (year^-1)
  PARAMETER(K_cots);                  // Carrying capacity of COTS (individuals/m^2)
  PARAMETER(m_cots);                  // Natural mortality rate of COTS (year^-1)
  PARAMETER(a_fast);                  // Attack rate on fast-growing coral (m^2/individual/year)
  PARAMETER(a_slow);                  // Attack rate on slow-growing coral (m^2/individual/year)
  PARAMETER(h_fast);                  // Handling time for fast-growing coral (year/% cover)
  PARAMETER(h_slow);                  // Handling time for slow-growing coral (year/% cover)
  PARAMETER(r_fast);                  // Intrinsic growth rate of fast-growing coral (year^-1)
  PARAMETER(r_slow);                  // Intrinsic growth rate of slow-growing coral (year^-1)
  PARAMETER(K_fast);                  // Carrying capacity of fast-growing coral (% cover)
  PARAMETER(K_slow);                  // Carrying capacity of slow-growing coral (% cover)
  PARAMETER(alpha_fs);                // Competition coefficient: effect of slow on fast coral
  PARAMETER(alpha_sf);                // Competition coefficient: effect of fast on slow coral
  PARAMETER(temp_opt);                // Optimal temperature for coral growth (°C)
  PARAMETER(temp_tol);                // Temperature tolerance range (°C)
  PARAMETER(imm_effect);              // Effect of immigration on COTS population
  PARAMETER(coral_threshold);         // Coral cover threshold for COTS survival (% cover)
  PARAMETER(temp_repro_threshold);    // Temperature threshold for enhanced COTS reproduction (°C)
  PARAMETER(temp_repro_effect);       // Effect of temperature on COTS reproduction (dimensionless)
  PARAMETER(pred_threshold);          // Threshold of predation intensity that impairs coral recovery
  PARAMETER(sigma_cots);              // Observation error SD for COTS (log scale)
  PARAMETER(sigma_slow);              // Observation error SD for slow-growing coral (log scale)
  PARAMETER(sigma_fast);              // Observation error SD for fast-growing coral (log scale)
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Number of time steps
  int n_steps = Year.size();
  
  // Vectors to store model predictions
  vector<Type> cots_pred(n_steps);
  vector<Type> slow_pred(n_steps);
  vector<Type> fast_pred(n_steps);
  
  // Initialize with first observation
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Time series simulation
  for (int t = 1; t < n_steps; t++) {
    // Temperature effect on coral growth (Gaussian response curve)
    Type temp_diff = (sst_dat(t-1) - temp_opt) / temp_tol;
    Type temp_effect = exp(-0.5 * temp_diff * temp_diff);
    
    // Total coral cover (food availability for COTS)
    Type total_coral = slow_pred(t-1) + fast_pred(t-1);
    
    // Functional responses for COTS feeding on corals (Type II)
    Type denom = 1.0 + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1);
    Type F_fast = (a_fast * fast_pred(t-1)) / denom;
    Type F_slow = (a_slow * slow_pred(t-1)) / denom;
    
    // Food limitation effect on COTS (smooth transition at threshold)
    Type food_limitation = 0.1 + 0.9 / (1.0 + exp(-3.0 * (total_coral - coral_threshold)));
    
    // Temperature effect on COTS reproduction
    Type temp_effect_cots = 1.0;
    if (sst_dat(t-1) > temp_repro_threshold) {
      temp_effect_cots = 1.0 + 0.5 * temp_repro_effect;
    }
    
    // COTS population dynamics
    Type density_factor = 1.0 - cots_pred(t-1) / K_cots;
    if (density_factor < 0.0) density_factor = 0.0;
    
    Type cots_growth = r_cots * cots_pred(t-1) * density_factor * food_limitation * temp_effect_cots;
    Type cots_mortality = m_cots * cots_pred(t-1);
    Type cots_immigration = imm_effect * cotsimm_dat(t-1);
    
    // Calculate predation
    Type fast_predation = F_fast * cots_pred(t-1);
    Type slow_predation = F_slow * cots_pred(t-1);
    
    // Ensure predation doesn't exceed available coral
    if (fast_predation > 0.9 * fast_pred(t-1)) fast_predation = 0.9 * fast_pred(t-1);
    if (slow_predation > 0.9 * slow_pred(t-1)) slow_predation = 0.9 * slow_pred(t-1);
    
    // Calculate predation intensity (proportion of coral consumed)
    Type fast_pred_intensity = 0.0;
    Type slow_pred_intensity = 0.0;
    
    if (fast_pred(t-1) > 0.0) fast_pred_intensity = fast_predation / fast_pred(t-1);
    if (slow_pred(t-1) > 0.0) slow_pred_intensity = slow_predation / slow_pred(t-1);
    
    // Recovery threshold effect (sigmoid function)
    Type fast_recovery_factor = 1.0 / (1.0 + exp(5.0 * (fast_pred_intensity - pred_threshold)));
    Type slow_recovery_factor = 1.0 / (1.0 + exp(5.0 * (slow_pred_intensity - pred_threshold)));
    
    // Fast-growing coral dynamics with recovery threshold effect
    Type competition_fast = (fast_pred(t-1) + alpha_fs * slow_pred(t-1)) / K_fast;
    if (competition_fast > 0.95) competition_fast = 0.95;
    
    Type fast_growth = r_fast * fast_pred(t-1) * (1.0 - competition_fast) * temp_effect * 
                      (0.3 + 0.7 * fast_recovery_factor);
    
    // Slow-growing coral dynamics with recovery threshold effect
    Type competition_slow = (slow_pred(t-1) + alpha_sf * fast_pred(t-1)) / K_slow;
    if (competition_slow > 0.95) competition_slow = 0.95;
    
    Type slow_growth = r_slow * slow_pred(t-1) * (1.0 - competition_slow) * temp_effect * 
                      (0.3 + 0.7 * slow_recovery_factor);
    
    // Update populations with bounds to ensure stability
    cots_pred(t) = cots_pred(t-1) + cots_growth - cots_mortality + cots_immigration;
    if (cots_pred(t) < 0.01) cots_pred(t) = 0.01;
    if (cots_pred(t) > 5.0) cots_pred(t) = 5.0;
    
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_predation;
    if (fast_pred(t) < 0.1) fast_pred(t) = 0.1;
    if (fast_pred(t) > K_fast) fast_pred(t) = K_fast;
    
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_predation;
    if (slow_pred(t) < 0.1) slow_pred(t) = 0.1;
    if (slow_pred(t) > K_slow) slow_pred(t) = K_slow;
  }
  
  // Calculate negative log-likelihood
  Type min_sigma = 0.1;
  
  for (int t = 0; t < n_steps; t++) {
    // Ensure observations are positive
    Type cots_obs = cots_dat(t);
    if (cots_obs < 0.01) cots_obs = 0.01;
    
    Type slow_obs = slow_dat(t);
    if (slow_obs < 0.1) slow_obs = 0.1;
    
    Type fast_obs = fast_dat(t);
    if (fast_obs < 0.1) fast_obs = 0.1;
    
    // COTS abundance likelihood
    Type sigma_cots_t = sigma_cots;
    if (sigma_cots_t < min_sigma) sigma_cots_t = min_sigma;
    nll -= dnorm(log(cots_obs), log(cots_pred(t)), sigma_cots_t, true);
    
    // Slow-growing coral cover likelihood
    Type sigma_slow_t = sigma_slow;
    if (sigma_slow_t < min_sigma) sigma_slow_t = min_sigma;
    nll -= dnorm(log(slow_obs), log(slow_pred(t)), sigma_slow_t, true);
    
    // Fast-growing coral cover likelihood
    Type sigma_fast_t = sigma_fast;
    if (sigma_fast_t < min_sigma) sigma_fast_t = min_sigma;
    nll -= dnorm(log(fast_obs), log(fast_pred(t)), sigma_fast_t, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
