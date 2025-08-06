#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA SECTION
  DATA_VECTOR(Year);                  // Vector of years for time series data
  DATA_VECTOR(sst_dat);               // Sea surface temperature data (Celsius)
  DATA_VECTOR(cotsimm_dat);           // COTS larval immigration rate (individuals/m^2/year)
  DATA_VECTOR(cots_dat);              // Observed COTS abundance (individuals/m^2)
  DATA_VECTOR(fast_dat);              // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);              // Observed slow-growing coral cover (%)
  
  // PARAMETER SECTION
  PARAMETER(r_cots);                  // Intrinsic growth rate of COTS (year^-1)
  PARAMETER(K_cots);                  // Carrying capacity of COTS (individuals/m^2)
  PARAMETER(m_cots);                  // Natural mortality rate of COTS (year^-1)
  PARAMETER(alpha_fast);              // Attack rate on fast-growing coral (m^2/individual/year)
  PARAMETER(alpha_slow);              // Attack rate on slow-growing coral (m^2/individual/year)
  PARAMETER(h_fast);                  // Handling time for fast-growing coral (year/% cover)
  PARAMETER(h_slow);                  // Handling time for slow-growing coral (year/% cover)
  PARAMETER(r_fast);                  // Intrinsic growth rate of fast-growing coral (year^-1)
  PARAMETER(r_slow);                  // Intrinsic growth rate of slow-growing coral (year^-1)
  PARAMETER(K_fast);                  // Carrying capacity of fast-growing coral (% cover)
  PARAMETER(K_slow);                  // Carrying capacity of slow-growing coral (% cover)
  PARAMETER(beta_sst);                // Effect of SST on COTS reproduction (dimensionless)
  PARAMETER(sst_opt);                 // Optimal SST for COTS reproduction (Celsius)
  PARAMETER(sst_width);               // Width parameter for temperature response (Celsius)
  PARAMETER(imm_effect);              // Effect of larval immigration (dimensionless)
  PARAMETER(coral_threshold);         // Coral threshold for COTS mortality (% cover)
  PARAMETER(sigma_cots);              // Observation error SD for COTS (log scale)
  PARAMETER(sigma_fast);              // Observation error SD for fast coral (log scale)
  PARAMETER(sigma_slow);              // Observation error SD for slow coral (log scale)
  
  // New parameters for Allee effect and predator-mediated control
  PARAMETER(allee_threshold);         // Population threshold for Allee effect (individuals/m^2)
  PARAMETER(allee_strength);          // Strength of Allee effect (dimensionless)
  PARAMETER(pred_half_sat);           // Half-saturation constant for predator response (individuals/m^2)
  PARAMETER(pred_max);                // Maximum predation rate on COTS (year^-1)
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize vectors for model predictions
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Set initial values (first year)
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-6);
  
  // Run the model for each time step
  for (int t = 1; t < n; t++) {
    // 1. Calculate temperature effect on COTS reproduction using a Gaussian response curve
    // Ensure sst_width is positive to avoid division by zero
    Type sst_width_pos = sst_width + eps;
    Type temp_effect = exp(-pow(sst_dat(t-1) - sst_opt, 2) / (2 * pow(sst_width_pos, 2)));
    
    // 2. Calculate total coral cover (fast + slow) for density dependence
    Type total_coral = fast_pred(t-1) + slow_pred(t-1) + eps;
    
    // 3. Calculate food-dependent mortality modifier (increases when coral is scarce)
    // Ensure coral_threshold is positive for the denominator
    Type coral_denom = coral_threshold * Type(0.1) + eps;
    Type mort_modifier = 1.0 + 1.0 / (1.0 + exp((total_coral - coral_threshold) / coral_denom));
    
    // 4. Calculate functional responses for COTS feeding on corals (Type II)
    // Ensure attack rates and handling times are positive
    Type alpha_fast_pos = alpha_fast + eps;
    Type alpha_slow_pos = alpha_slow + eps;
    Type h_fast_pos = h_fast + eps;
    Type h_slow_pos = h_slow + eps;
    
    Type denominator = 1.0 + alpha_fast_pos * h_fast_pos * fast_pred(t-1) + alpha_slow_pos * h_slow_pos * slow_pred(t-1);
    Type consumption_fast = (alpha_fast_pos * fast_pred(t-1) * cots_pred(t-1)) / denominator;
    Type consumption_slow = (alpha_slow_pos * slow_pred(t-1) * cots_pred(t-1)) / denominator;
    
    // 5. Calculate Allee effect - a positive density dependence at low densities
    // Ensure allee parameters are positive
    Type allee_threshold_pos = allee_threshold + eps;
    Type allee_strength_pos = allee_strength + eps;
    
    // Use a smoother formulation of the Allee effect
    Type allee_denom = pow(allee_threshold_pos, allee_strength_pos) + pow(cots_pred(t-1), allee_strength_pos);
    Type allee_effect = pow(cots_pred(t-1), allee_strength_pos) / allee_denom;
    
    // 6. Calculate predator-mediated mortality - decreases as COTS density increases (predator saturation)
    // Ensure predator parameters are positive
    Type pred_half_sat_pos = pred_half_sat + eps;
    Type pred_max_pos = pred_max + eps;
    
    // Type II functional response for predation
    Type pred_mortality = pred_max_pos * pred_half_sat_pos / (pred_half_sat_pos + cots_pred(t-1));
    
    // 7. Calculate COTS population dynamics with temperature effect, Allee effect, and immigration
    // Ensure growth rate and carrying capacity are positive
    Type r_cots_pos = r_cots + eps;
    Type K_cots_pos = K_cots + eps;
    Type m_cots_pos = m_cots + eps;
    
    // Calculate carrying capacity modifier based on coral availability
    Type carrying_capacity_modifier = total_coral / (K_fast + K_slow + eps);
    
    // Calculate density dependence term with a smooth transition at zero
    Type density_term = 1.0 - cots_pred(t-1) / (K_cots_pos * carrying_capacity_modifier);
    Type density_dependence = 0.5 * (density_term + sqrt(density_term * density_term + eps));
    
    // Calculate growth, mortality, and immigration
    Type cots_growth = r_cots_pos * temp_effect * allee_effect * cots_pred(t-1) * density_dependence;
    Type cots_mortality = (m_cots_pos * mort_modifier + pred_mortality) * cots_pred(t-1);
    Type cots_immigration = imm_effect * cotsimm_dat(t-1);
    
    // 8. Update COTS abundance with a smooth positive constraint
    Type cots_new = cots_pred(t-1) + cots_growth - cots_mortality + cots_immigration;
    cots_pred(t) = 0.5 * (cots_new + sqrt(cots_new * cots_new + eps));
    
    // 9. Calculate coral dynamics with logistic growth and COTS predation
    // Ensure coral growth rates are positive
    Type r_fast_pos = r_fast + eps;
    Type r_slow_pos = r_slow + eps;
    
    // Calculate density dependence for coral growth with smooth transitions
    Type fast_density_term = 1.0 - (fast_pred(t-1) + 0.5 * slow_pred(t-1)) / (K_fast + eps);
    Type fast_density = 0.5 * (fast_density_term + sqrt(fast_density_term * fast_density_term + eps));
    Type fast_growth = r_fast_pos * fast_pred(t-1) * fast_density;
    
    Type slow_density_term = 1.0 - (slow_pred(t-1) + 0.3 * fast_pred(t-1)) / (K_slow + eps);
    Type slow_density = 0.5 * (slow_density_term + sqrt(slow_density_term * slow_density_term + eps));
    Type slow_growth = r_slow_pos * slow_pred(t-1) * slow_density;
    
    // 10. Update coral cover with smooth positive constraints
    Type fast_new = fast_pred(t-1) + fast_growth - consumption_fast;
    Type slow_new = slow_pred(t-1) + slow_growth - consumption_slow;
    
    fast_pred(t) = 0.5 * (fast_new + sqrt(fast_new * fast_new + eps));
    slow_pred(t) = 0.5 * (slow_new + sqrt(slow_new * slow_new + eps));
  }
  
  // Calculate negative log-likelihood using lognormal distribution
  // Add a small constant to observations and predictions to handle zeros
  Type const_obs = Type(1e-3);
  
  for (int t = 0; t < n; t++) {
    // 12. COTS abundance likelihood
    Type cots_obs = cots_dat(t) + const_obs;
    Type cots_model = cots_pred(t) + const_obs;
    nll -= dnorm(log(cots_obs), log(cots_model), sigma_cots, true);
    
    // 13. Fast-growing coral cover likelihood
    Type fast_obs = fast_dat(t) + const_obs;
    Type fast_model = fast_pred(t) + const_obs;
    nll -= dnorm(log(fast_obs), log(fast_model), sigma_fast, true);
    
    // 14. Slow-growing coral cover likelihood
    Type slow_obs = slow_dat(t) + const_obs;
    Type slow_model = slow_pred(t) + const_obs;
    nll -= dnorm(log(slow_obs), log(slow_model), sigma_slow, true);
  }
  
  // Add very soft penalties to constrain parameters within biologically meaningful ranges
  // Using much softer penalties to avoid gradient issues
  nll += 0.00001 * pow(r_cots - 1.0, 2) * (r_cots > 1.0);
  nll += 0.00001 * pow(alpha_fast - 0.5, 2) * (alpha_fast > 0.5);
  nll += 0.00001 * pow(alpha_slow - 0.5, 2) * (alpha_slow > 0.5);
  nll += 0.00001 * pow(allee_strength - 2.0, 2) * (allee_strength > 2.0);
  nll += 0.00001 * pow(pred_max - 0.8, 2) * (pred_max > 0.8);
  
  // REPORT SECTION
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  ADREPORT(r_cots);
  ADREPORT(K_cots);
  ADREPORT(alpha_fast);
  ADREPORT(alpha_slow);
  ADREPORT(r_fast);
  ADREPORT(r_slow);
  ADREPORT(beta_sst);
  ADREPORT(imm_effect);
  ADREPORT(allee_threshold);
  ADREPORT(allee_strength);
  ADREPORT(pred_half_sat);
  ADREPORT(pred_max);
  
  return nll;
}
