#include <TMB.hpp>

// Smooth maximum function for TMB (replaces posfun)
template<class Type>
Type smooth_max(Type x, Type lower_bound, Type eps) {
  // Returns x if x > lower_bound, otherwise smoothly approaches lower_bound
  // Uses a smooth approximation to avoid discontinuities
  Type diff = x - lower_bound;
  return lower_bound + diff * invlogit(diff / eps) + eps;
}

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS
  DATA_VECTOR(Year);                    // Time vector (years)
  DATA_VECTOR(sst_dat);                 // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);             // COTS larval immigration rate (individuals/m²/year)
  DATA_VECTOR(cots_dat);                // Adult COTS abundance (individuals/m²)
  DATA_VECTOR(fast_dat);                // Fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);                // Slow-growing coral cover (%)
  
  // COTS POPULATION PARAMETERS
  PARAMETER(log_cots_recruit_base);     // Baseline recruitment rate from local reproduction (log scale, year⁻¹)
  PARAMETER(log_temp_opt);              // Optimal temperature for COTS larval survival (log scale, °C)
  PARAMETER(log_temp_width);            // Temperature tolerance width (log scale, °C)
  PARAMETER(log_allee_threshold);       // Allee effect threshold density (log scale, individuals/m²)
  PARAMETER(log_allee_strength);        // Strength of Allee effect (log scale, dimensionless)
  PARAMETER(log_cots_mort_base);        // Baseline COTS mortality rate (log scale, year⁻¹)
  PARAMETER(log_cots_mort_starv);       // Starvation mortality coefficient (log scale, m²/% coral cover/year)
  PARAMETER(log_density_mort);          // Density-dependent mortality coefficient (log scale, m²/individual/year)
  PARAMETER(immigration_efficiency);    // Efficiency of larval immigration to recruitment (dimensionless, 0-1)
  
  // CORAL DYNAMICS PARAMETERS
  PARAMETER(log_fast_growth);           // Fast coral intrinsic growth rate (log scale, year⁻¹)
  PARAMETER(log_slow_growth);           // Slow coral intrinsic growth rate (log scale, year⁻¹)
  PARAMETER(log_fast_K);                // Fast coral carrying capacity (log scale, % cover)
  PARAMETER(log_slow_K);                // Slow coral carrying capacity (log scale, % cover)
  PARAMETER(log_competition_coef);      // Interspecific competition coefficient (log scale, dimensionless)
  PARAMETER(log_temp_growth_opt);       // Optimal temperature for coral growth (log scale, °C)
  PARAMETER(log_temp_growth_width);     // Temperature tolerance for coral growth (log scale, °C)
  
  // PREDATION PARAMETERS
  PARAMETER(log_attack_fast);           // Attack rate on fast corals (log scale, m²/individual/year)
  PARAMETER(log_attack_slow);           // Attack rate on slow corals (log scale, m²/individual/year)
  PARAMETER(log_handling_fast);         // Handling time for fast corals (log scale, year/% cover)
  PARAMETER(log_handling_slow);         // Handling time for slow corals (log scale, year/% cover)
  PARAMETER(log_predation_threshold);   // Minimum coral cover for predation (log scale, % cover)
  
  // OBSERVATION ERROR PARAMETERS
  PARAMETER(log_sigma_cots);            // Observation error SD for COTS (log scale)
  PARAMETER(log_sigma_fast);            // Observation error SD for fast corals (log scale)
  PARAMETER(log_sigma_slow);            // Observation error SD for slow corals (log scale)
  
  // Transform parameters from log scale
  Type cots_recruit_base = exp(log_cots_recruit_base);           // Baseline recruitment rate (year⁻¹)
  Type temp_opt = exp(log_temp_opt);                             // Optimal temperature (°C)
  Type temp_width = exp(log_temp_width);                         // Temperature width (°C)
  Type allee_threshold = exp(log_allee_threshold);               // Allee threshold (individuals/m²)
  Type allee_strength = exp(log_allee_strength);                 // Allee strength (dimensionless)
  Type cots_mort_base = exp(log_cots_mort_base);                 // Baseline mortality (year⁻¹)
  Type cots_mort_starv = exp(log_cots_mort_starv);               // Starvation mortality (m²/% cover/year)
  Type density_mort = exp(log_density_mort);                     // Density-dependent mortality (m²/individual/year)
  Type fast_growth = exp(log_fast_growth);                       // Fast coral growth (year⁻¹)
  Type slow_growth = exp(log_slow_growth);                       // Slow coral growth (year⁻¹)
  Type fast_K = exp(log_fast_K);                                 // Fast coral K (% cover)
  Type slow_K = exp(log_slow_K);                                 // Slow coral K (% cover)
  Type competition_coef = exp(log_competition_coef);             // Competition coefficient (dimensionless)
  Type temp_growth_opt = exp(log_temp_growth_opt);               // Optimal growth temperature (°C)
  Type temp_growth_width = exp(log_temp_growth_width);           // Growth temperature width (°C)
  Type attack_fast = exp(log_attack_fast);                       // Attack rate fast (m²/individual/year)
  Type attack_slow = exp(log_attack_slow);                       // Attack rate slow (m²/individual/year)
  Type handling_fast = exp(log_handling_fast);                   // Handling time fast (year/% cover)
  Type handling_slow = exp(log_handling_slow);                   // Handling time slow (year/% cover)
  Type predation_threshold = exp(log_predation_threshold);       // Predation threshold (% cover)
  Type sigma_cots = exp(log_sigma_cots);                         // COTS observation error (individuals/m²)
  Type sigma_fast = exp(log_sigma_fast);                         // Fast coral observation error (% cover)
  Type sigma_slow = exp(log_sigma_slow);                         // Slow coral observation error (% cover)
  
  // Initialize prediction vectors
  int n = Year.size();                                           // Number of time steps
  vector<Type> cots_pred(n);                                     // Predicted COTS abundance (individuals/m²)
  vector<Type> fast_pred(n);                                     // Predicted fast coral cover (% cover)
  vector<Type> slow_pred(n);                                     // Predicted slow coral cover (% cover)
  
  // Set initial conditions from first observation
  cots_pred(0) = cots_dat(0);                                    // Initialize COTS from data (individuals/m²)
  fast_pred(0) = fast_dat(0);                                    // Initialize fast coral from data (% cover)
  slow_pred(0) = slow_dat(0);                                    // Initialize slow coral from data (% cover)
  
  // Small constant for numerical stability
  Type eps = Type(1e-8);                                         // Small constant to prevent division by zero
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(0.01);                                   // Minimum observation error (prevents collapse to zero)
  Type sigma_cots_safe = sigma_cots + min_sigma;                 // Safe COTS error (individuals/m²)
  Type sigma_fast_safe = sigma_fast + min_sigma;                 // Safe fast coral error (% cover)
  Type sigma_slow_safe = sigma_slow + min_sigma;                 // Safe slow coral error (% cover)
  
  // TIME LOOP: Simulate dynamics forward in time
  for(int t = 1; t < n; t++) {
    
    // Use previous time step values to avoid data leakage
    Type cots_prev = cots_pred(t-1);                             // Previous COTS density (individuals/m²)
    Type fast_prev = fast_pred(t-1);                             // Previous fast coral cover (% cover)
    Type slow_prev = slow_pred(t-1);                             // Previous slow coral cover (% cover)
    Type sst_current = sst_dat(t);                               // Current sea surface temperature (°C)
    Type immigration = cotsimm_dat(t);                           // Current larval immigration (individuals/m²/year)
    
    // EQUATION 1: Temperature effect on COTS larval survival (Gaussian response)
    Type temp_diff = sst_current - temp_opt;                     // Temperature deviation from optimum (°C)
    Type temp_effect = exp(-0.5 * pow(temp_diff / (temp_width + eps), 2)); // Temperature survival multiplier (0-1)
    
    // EQUATION 2: Allee effect on COTS recruitment (sigmoid function)
    Type allee_effect = pow(cots_prev, allee_strength) / (pow(allee_threshold, allee_strength) + pow(cots_prev, allee_strength) + eps); // Allee multiplier (0-1)
    
    // EQUATION 3: Total coral availability for COTS food
    Type total_coral = fast_prev + slow_prev + eps;              // Total coral cover (% cover)
    
    // EQUATION 4: COTS recruitment (local + immigration, modified by temperature and Allee effects)
    Type local_recruitment = cots_recruit_base * cots_prev * temp_effect * allee_effect; // Local recruitment (individuals/m²/year)
    Type immigration_recruitment = immigration * immigration_efficiency * temp_effect; // Immigration recruitment (individuals/m²/year)
    Type total_recruitment = local_recruitment + immigration_recruitment; // Total recruitment (individuals/m²/year)
    
    // EQUATION 5: Starvation mortality (increases when coral is depleted)
    Type starvation_mort = cots_mort_starv / (total_coral + eps); // Starvation mortality rate (year⁻¹)
    
    // EQUATION 6: Total COTS mortality (baseline + starvation + density-dependent)
    Type total_cots_mort = cots_mort_base + starvation_mort + density_mort * cots_prev; // Total mortality rate (year⁻¹)
    
    // EQUATION 7: COTS population dynamics (using smooth_max to ensure non-negative values)
    Type cots_change = total_recruitment - total_cots_mort * cots_prev; // Net COTS change (individuals/m²/year)
    Type cots_new = cots_prev + cots_change;                     // New COTS value (individuals/m²)
    cots_pred(t) = smooth_max(cots_new, Type(0.0), eps);         // Update COTS (non-negative, individuals/m²)
    
    // EQUATION 8: Type II functional response for fast coral predation (using smooth_max for threshold)
    Type fast_above_threshold = fast_prev - predation_threshold; // Fast coral relative to threshold (% cover)
    Type fast_available = smooth_max(fast_above_threshold, Type(0.0), eps); // Available fast coral above threshold (% cover)
    Type fast_predation_rate = (attack_fast * fast_available) / (Type(1.0) + attack_fast * handling_fast * fast_available + eps); // Fast coral consumption rate (% cover/individual/year)
    Type fast_predation = fast_predation_rate * cots_prev;       // Total fast coral predation (% cover/year)
    
    // EQUATION 9: Type II functional response for slow coral predation (using smooth_max for threshold)
    Type slow_above_threshold = slow_prev - predation_threshold; // Slow coral relative to threshold (% cover)
    Type slow_available = smooth_max(slow_above_threshold, Type(0.0), eps); // Available slow coral above threshold (% cover)
    Type slow_predation_rate = (attack_slow * slow_available) / (Type(1.0) + attack_slow * handling_slow * slow_available + eps); // Slow coral consumption rate (% cover/individual/year)
    Type slow_predation = slow_predation_rate * cots_prev;       // Total slow coral predation (% cover/year)
    
    // EQUATION 10: Temperature effect on coral growth (Gaussian response)
    Type temp_growth_diff = sst_current - temp_growth_opt;       // Temperature deviation from growth optimum (°C)
    Type temp_growth_effect = exp(-0.5 * pow(temp_growth_diff / (temp_growth_width + eps), 2)); // Temperature growth multiplier (0-1)
    
    // EQUATION 11: Fast coral logistic growth with competition and predation (using smooth_max for non-negative)
    Type fast_space_limit = Type(1.0) - (fast_prev + competition_coef * slow_prev) / (fast_K + eps); // Space limitation for fast coral (0-1)
    Type fast_growth_rate = fast_growth * temp_growth_effect * fast_space_limit; // Net fast coral growth rate (year⁻¹)
    Type fast_change = fast_growth_rate * fast_prev - fast_predation; // Net fast coral change (% cover/year)
    Type fast_new = fast_prev + fast_change;                     // New fast coral value (% cover)
    fast_pred(t) = smooth_max(fast_new, Type(0.0), eps);         // Update fast coral (non-negative, % cover)
    
    // EQUATION 12: Slow coral logistic growth with competition and predation (using smooth_max for non-negative)
    Type slow_space_limit = Type(1.0) - (slow_prev + competition_coef * fast_prev) / (slow_K + eps); // Space limitation for slow coral (0-1)
    Type slow_growth_rate = slow_growth * temp_growth_effect * slow_space_limit; // Net slow coral growth rate (year⁻¹)
    Type slow_change = slow_growth_rate * slow_prev - slow_predation; // Net slow coral change (% cover/year)
    Type slow_new = slow_prev + slow_change;                     // New slow coral value (% cover)
    slow_pred(t) = smooth_max(slow_new, Type(0.0), eps);         // Update slow coral (non-negative, % cover)
  }
  
  // LIKELIHOOD CALCULATION
  Type nll = Type(0.0);                                          // Initialize negative log-likelihood
  
  // EQUATION 13: Lognormal likelihood for COTS observations
  for(int t = 0; t < n; t++) {
    Type cots_obs = cots_dat(t) + eps;                           // Observed COTS (add eps for log transform, individuals/m²)
    Type cots_model = cots_pred(t) + eps;                        // Predicted COTS (add eps for log transform, individuals/m²)
    nll -= dnorm(log(cots_obs), log(cots_model), sigma_cots_safe, true); // Lognormal likelihood for COTS
  }
  
  // EQUATION 14: Lognormal likelihood for fast coral observations
  for(int t = 0; t < n; t++) {
    Type fast_obs = fast_dat(t) + eps;                           // Observed fast coral (add eps for log transform, % cover)
    Type fast_model = fast_pred(t) + eps;                        // Predicted fast coral (add eps for log transform, % cover)
    nll -= dnorm(log(fast_obs), log(fast_model), sigma_fast_safe, true); // Lognormal likelihood for fast coral
  }
  
  // EQUATION 15: Lognormal likelihood for slow coral observations
  for(int t = 0; t < n; t++) {
    Type slow_obs = slow_dat(t) + eps;                           // Observed slow coral (add eps for log transform, % cover)
    Type slow_model = slow_pred(t) + eps;                        // Predicted slow coral (add eps for log transform, % cover)
    nll -= dnorm(log(slow_obs), log(slow_model), sigma_slow_safe, true); // Lognormal likelihood for slow coral
  }
  
  // REPORT PREDICTIONS AND PARAMETERS
  REPORT(cots_pred);                                             // Report predicted COTS abundance
  REPORT(fast_pred);                                             // Report predicted fast coral cover
  REPORT(slow_pred);                                             // Report predicted slow coral cover
  REPORT(cots_recruit_base);                                     // Report baseline recruitment rate
  REPORT(temp_opt);                                              // Report optimal temperature
  REPORT(temp_width);                                            // Report temperature width
  REPORT(allee_threshold);                                       // Report Allee threshold
  REPORT(allee_strength);                                        // Report Allee strength
  REPORT(cots_mort_base);                                        // Report baseline mortality
  REPORT(cots_mort_starv);                                       // Report starvation mortality
  REPORT(density_mort);                                          // Report density-dependent mortality
  REPORT(immigration_efficiency);                                // Report immigration efficiency
  REPORT(fast_growth);                                           // Report fast coral growth
  REPORT(slow_growth);                                           // Report slow coral growth
  REPORT(fast_K);                                                // Report fast coral K
  REPORT(slow_K);                                                // Report slow coral K
  REPORT(competition_coef);                                      // Report competition coefficient
  REPORT(temp_growth_opt);                                       // Report optimal growth temperature
  REPORT(temp_growth_width);                                     // Report growth temperature width
  REPORT(attack_fast);                                           // Report attack rate on fast coral
  REPORT(attack_slow);                                           // Report attack rate on slow coral
  REPORT(handling_fast);                                         // Report handling time for fast coral
  REPORT(handling_slow);                                         // Report handling time for slow coral
  REPORT(predation_threshold);                                   // Report predation threshold
  REPORT(sigma_cots);                                            // Report COTS observation error
  REPORT(sigma_fast);                                            // Report fast coral observation error
  REPORT(sigma_slow);                                            // Report slow coral observation error
  
  return nll;                                                    // Return negative log-likelihood
}
