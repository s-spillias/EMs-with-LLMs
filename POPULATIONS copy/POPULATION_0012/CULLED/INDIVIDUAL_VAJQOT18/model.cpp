#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS
  DATA_VECTOR(Year);                    // Time vector (years)
  DATA_VECTOR(sst_dat);                 // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);             // COTS larval immigration rate (individuals/m²/year)
  DATA_VECTOR(cots_dat);                // Observed adult COTS abundance (individuals/m²)
  DATA_VECTOR(fast_dat);                // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);                // Observed slow-growing coral cover (%)
  
  // COTS POPULATION PARAMETERS
  PARAMETER(log_r_cots);                // Log intrinsic growth rate of COTS (year⁻¹)
  PARAMETER(log_K_cots);                // Log carrying capacity of COTS (individuals/m²)
  PARAMETER(log_allee_threshold);       // Log Allee threshold density (individuals/m²)
  PARAMETER(allee_strength);            // Allee effect strength (dimensionless, 0-1)
  PARAMETER(log_m_cots);                // Log baseline COTS mortality rate (year⁻¹)
  PARAMETER(log_dd_mort);               // Log density-dependent mortality coefficient (m²/individuals/year)
  PARAMETER(larval_survival);           // Baseline larval survival fraction (dimensionless, 0-1)
  
  // TEMPERATURE EFFECTS ON COTS
  PARAMETER(temp_opt_cots);             // Optimal temperature for COTS larvae (°C)
  PARAMETER(log_temp_width_cots);       // Log temperature tolerance width (°C)
  
  // CORAL GROWTH PARAMETERS
  PARAMETER(log_r_fast);                // Log intrinsic growth rate of fast coral (year⁻¹)
  PARAMETER(log_r_slow);                // Log intrinsic growth rate of slow coral (year⁻¹)
  PARAMETER(log_K_coral);               // Log total coral carrying capacity (% cover)
  PARAMETER(competition_fast);          // Competitive advantage of fast coral (dimensionless)
  
  // COTS FEEDING PARAMETERS
  PARAMETER(log_attack_fast);           // Log attack rate on fast coral (m²/individuals/year)
  PARAMETER(log_attack_slow);           // Log attack rate on slow coral (m²/individuals/year)
  PARAMETER(log_handling_fast);         // Log handling time for fast coral (year)
  PARAMETER(log_handling_slow);         // Log handling time for slow coral (year)
  PARAMETER(preference_fast);           // Preference for fast coral when both available (dimensionless, 0-1)
  
  // TEMPERATURE EFFECTS ON CORAL
  PARAMETER(temp_opt_coral);            // Optimal temperature for coral growth (°C)
  PARAMETER(log_temp_width_coral);      // Log temperature tolerance width for coral (°C)
  PARAMETER(log_bleach_threshold);      // Log temperature threshold for bleaching stress (°C above optimum)
  PARAMETER(log_bleach_mort);           // Log bleaching mortality rate (year⁻¹)
  
  // OBSERVATION ERROR PARAMETERS
  PARAMETER(log_sigma_cots);            // Log observation error SD for COTS (log scale)
  PARAMETER(log_sigma_fast);            // Log observation error SD for fast coral (log scale)
  PARAMETER(log_sigma_slow);            // Log observation error SD for slow coral (log scale)
  
  // TRANSFORM PARAMETERS
  Type r_cots = exp(log_r_cots);                          // Intrinsic growth rate of COTS (year⁻¹)
  Type K_cots = exp(log_K_cots);                          // Carrying capacity of COTS (individuals/m²)
  Type allee_threshold = exp(log_allee_threshold);        // Allee threshold density (individuals/m²)
  Type m_cots = exp(log_m_cots);                          // Baseline COTS mortality (year⁻¹)
  Type dd_mort = exp(log_dd_mort);                        // Density-dependent mortality coefficient (m²/individuals/year)
  Type temp_width_cots = exp(log_temp_width_cots);        // Temperature tolerance width for COTS (°C)
  Type r_fast = exp(log_r_fast);                          // Growth rate of fast coral (year⁻¹)
  Type r_slow = exp(log_r_slow);                          // Growth rate of slow coral (year⁻¹)
  Type K_coral = exp(log_K_coral);                        // Total coral carrying capacity (% cover)
  Type attack_fast = exp(log_attack_fast);                // Attack rate on fast coral (m²/individuals/year)
  Type attack_slow = exp(log_attack_slow);                // Attack rate on slow coral (m²/individuals/year)
  Type handling_fast = exp(log_handling_fast);            // Handling time for fast coral (year)
  Type handling_slow = exp(log_handling_slow);            // Handling time for slow coral (year)
  Type temp_width_coral = exp(log_temp_width_coral);      // Temperature tolerance width for coral (°C)
  Type bleach_threshold = exp(log_bleach_threshold);      // Bleaching threshold (°C above optimum)
  Type bleach_mort = exp(log_bleach_mort);                // Bleaching mortality rate (year⁻¹)
  Type sigma_cots = exp(log_sigma_cots);                  // Observation error SD for COTS
  Type sigma_fast = exp(log_sigma_fast);                  // Observation error SD for fast coral
  Type sigma_slow = exp(log_sigma_slow);                  // Observation error SD for slow coral
  
  // INITIALIZE PREDICTION VECTORS
  int n = Year.size();                                     // Number of time steps
  vector<Type> cots_pred(n);                              // Predicted COTS abundance (individuals/m²)
  vector<Type> fast_pred(n);                              // Predicted fast coral cover (%)
  vector<Type> slow_pred(n);                              // Predicted slow coral cover (%)
  
  // SET INITIAL CONDITIONS FROM DATA
  cots_pred(0) = cots_dat(0);                             // Initialize COTS from first observation
  fast_pred(0) = fast_dat(0);                             // Initialize fast coral from first observation
  slow_pred(0) = slow_dat(0);                             // Initialize slow coral from first observation
  
  // NUMERICAL STABILITY CONSTANTS
  Type eps = Type(1e-8);                                   // Small constant to prevent division by zero
  Type min_sigma = Type(0.01);                             // Minimum observation error to prevent numerical issues
  
  // APPLY PARAMETER BOUNDS WITH SOFT PENALTIES
  Type nll = Type(0.0);                                    // Initialize negative log-likelihood
  
  // Bound allee_strength between 0 and 1
  nll -= dnorm(allee_strength, Type(0.5), Type(0.3), true); // Soft prior favoring moderate Allee effects
  
  // Bound larval_survival between 0 and 1
  nll -= dnorm(larval_survival, Type(0.05), Type(0.05), true); // Soft prior for baseline larval survival
  
  // Bound preference_fast between 0 and 1
  nll -= dnorm(preference_fast, Type(0.7), Type(0.2), true); // Soft prior favoring fast coral preference
  
  // Bound competition_fast to be positive
  nll -= dnorm(competition_fast, Type(1.0), Type(0.5), true); // Soft prior for competitive advantage
  
  // TIME LOOP FOR DYNAMIC PREDICTIONS
  for(int t = 1; t < n; t++) {
    
    // EQUATION 1: Temperature effect on COTS larval survival (Gaussian response, tightened)
    Type temp_effect_cots = exp(-pow(sst_dat(t-1) - temp_opt_cots, 2) / (2.0 * pow(temp_width_cots, 2) + eps)); // Temperature multiplier for COTS (0-1)
    
    // EQUATION 2: Allee effect function (sigmoid transition from low to high growth)
    Type allee_effect = Type(1.0) / (Type(1.0) + exp(-allee_strength * (cots_pred(t-1) - allee_threshold))); // Allee multiplier (0-1)
    
    // EQUATION 3: Total coral availability for COTS feeding
    Type total_coral = fast_pred(t-1) + slow_pred(t-1) + eps;  // Total coral cover (%)
    
    // EQUATION 4: Prey preference weighting (switches based on fast coral availability)
    Type pref_weight_fast = preference_fast * (fast_pred(t-1) / (total_coral + eps)); // Preference weight for fast coral (dimensionless)
    Type pref_weight_slow = (Type(1.0) - preference_fast) * (slow_pred(t-1) / (total_coral + eps)); // Preference weight for slow coral (dimensionless)
    Type pref_norm = pref_weight_fast + pref_weight_slow + eps; // Normalization factor
    pref_weight_fast = pref_weight_fast / pref_norm;            // Normalized preference for fast coral
    pref_weight_slow = pref_weight_slow / pref_norm;            // Normalized preference for slow coral
    
    // EQUATION 5: Type II functional response for COTS feeding on fast coral
    Type consumption_fast = (attack_fast * pref_weight_fast * fast_pred(t-1) * cots_pred(t-1)) / (Type(1.0) + attack_fast * handling_fast * pref_weight_fast * fast_pred(t-1) + attack_slow * handling_slow * pref_weight_slow * slow_pred(t-1) + eps); // Fast coral consumed (%/year)
    
    // EQUATION 6: Type II functional response for COTS feeding on slow coral
    Type consumption_slow = (attack_slow * pref_weight_slow * slow_pred(t-1) * cots_pred(t-1)) / (Type(1.0) + attack_fast * handling_fast * pref_weight_fast * fast_pred(t-1) + attack_slow * handling_slow * pref_weight_slow * slow_pred(t-1) + eps); // Slow coral consumed (%/year)
    
    // EQUATION 7: Total food intake for COTS (affects growth)
    Type total_food = consumption_fast + consumption_slow + eps; // Total coral consumed (%/year)
    
    // EQUATION 8: Food-dependent COTS growth modifier
    Type food_effect = total_food / (total_food + Type(5.0));   // Saturation function for food effect (dimensionless)
    
    // EQUATION 9: Density-dependent mortality of COTS
    Type density_mort = dd_mort * cots_pred(t-1);               // Density-dependent mortality rate (year⁻¹)
    
    // EQUATION 10: Larval recruitment with environmental modulation
    Type recruitment = larval_survival * cotsimm_dat(t-1) * temp_effect_cots; // Effective larval recruitment (individuals/m²/year)
    
    // EQUATION 11: COTS population dynamics
    Type cots_growth = r_cots * allee_effect * food_effect * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1) / (K_cots + eps)); // Logistic growth with Allee and food effects
    Type cots_mortality = (m_cots + density_mort) * cots_pred(t-1); // Total mortality
    cots_pred(t) = cots_pred(t-1) + cots_growth - cots_mortality + recruitment; // COTS abundance at time t
    cots_pred(t) = CppAD::CondExpGe(cots_pred(t), eps, cots_pred(t), eps); // Ensure non-negative COTS abundance
    
    // EQUATION 12: Temperature effect on coral growth (Gaussian response)
    Type temp_effect_coral = exp(-pow(sst_dat(t-1) - temp_opt_coral, 2) / (2.0 * pow(temp_width_coral, 2) + eps)); // Temperature multiplier for coral (0-1)
    
    // EQUATION 13: Bleaching mortality (threshold response to high temperature)
    Type temp_diff = sst_dat(t-1) - temp_opt_coral - bleach_threshold; // Temperature difference from bleaching threshold (°C)
    Type temp_anomaly = CppAD::CondExpGe(temp_diff, Type(0.0), temp_diff, Type(0.0)); // Temperature above bleaching threshold (°C)
    Type bleaching_effect = bleach_mort * temp_anomaly;         // Bleaching mortality rate (year⁻¹)
    
    // EQUATION 14: Competition for space (total coral limited by carrying capacity)
    Type total_coral_current = fast_pred(t-1) + slow_pred(t-1) + eps; // Current total coral cover (%)
    Type space_limitation = Type(1.0) - total_coral_current / (K_coral + eps); // Available space fraction (0-1)
    
    // EQUATION 15: Fast coral dynamics
    Type fast_growth = r_fast * temp_effect_coral * fast_pred(t-1) * space_limitation * competition_fast; // Fast coral growth (%/year)
    Type fast_mortality = bleaching_effect * fast_pred(t-1);    // Bleaching mortality of fast coral (%/year)
    Type fast_predation = consumption_fast;                     // COTS predation on fast coral (%/year)
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_mortality - fast_predation; // Fast coral cover at time t
    fast_pred(t) = CppAD::CondExpGe(fast_pred(t), eps, fast_pred(t), eps); // Ensure non-negative fast coral cover
    
    // EQUATION 16: Slow coral dynamics
    Type slow_growth = r_slow * temp_effect_coral * slow_pred(t-1) * space_limitation; // Slow coral growth (%/year)
    Type slow_mortality = bleaching_effect * slow_pred(t-1);    // Bleaching mortality of slow coral (%/year)
    Type slow_predation = consumption_slow;                     // COTS predation on slow coral (%/year)
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_mortality - slow_predation; // Slow coral cover at time t
    slow_pred(t) = CppAD::CondExpGe(slow_pred(t), eps, slow_pred(t), eps); // Ensure non-negative slow coral cover
  }
  
  // LIKELIHOOD CALCULATION
  // Use lognormal distribution for strictly positive data
  Type sigma_cots_use = CppAD::CondExpGe(sigma_cots, min_sigma, sigma_cots, min_sigma); // Apply minimum sigma for COTS
  Type sigma_fast_use = CppAD::CondExpGe(sigma_fast, min_sigma, sigma_fast, min_sigma); // Apply minimum sigma for fast coral
  Type sigma_slow_use = CppAD::CondExpGe(sigma_slow, min_sigma, sigma_slow, min_sigma); // Apply minimum sigma for slow coral
  
  for(int t = 0; t < n; t++) {
    // COTS likelihood (lognormal)
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots_use, true); // Negative log-likelihood for COTS observations
    
    // Fast coral likelihood (lognormal)
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast_use, true); // Negative log-likelihood for fast coral observations
    
    // Slow coral likelihood (lognormal)
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow_use, true); // Negative log-likelihood for slow coral observations
  }
  
  // REPORT PREDICTIONS
  REPORT(cots_pred);                                            // Report predicted COTS abundance
  REPORT(fast_pred);                                            // Report predicted fast coral cover
  REPORT(slow_pred);                                            // Report predicted slow coral cover
  
  // REPORT TRANSFORMED PARAMETERS
  REPORT(r_cots);                                               // Report COTS growth rate
  REPORT(K_cots);                                               // Report COTS carrying capacity
  REPORT(allee_threshold);                                      // Report Allee threshold
  REPORT(m_cots);                                               // Report COTS mortality
  REPORT(dd_mort);                                              // Report density-dependent mortality
  REPORT(temp_width_cots);                                      // Report COTS temperature tolerance
  REPORT(r_fast);                                               // Report fast coral growth rate
  REPORT(r_slow);                                               // Report slow coral growth rate
  REPORT(K_coral);                                              // Report coral carrying capacity
  REPORT(attack_fast);                                          // Report attack rate on fast coral
  REPORT(attack_slow);                                          // Report attack rate on slow coral
  REPORT(handling_fast);                                        // Report handling time for fast coral
  REPORT(handling_slow);                                        // Report handling time for slow coral
  REPORT(temp_width_coral);                                     // Report coral temperature tolerance
  REPORT(bleach_threshold);                                     // Report bleaching threshold
  REPORT(bleach_mort);                                          // Report bleaching mortality
  REPORT(sigma_cots);                                           // Report COTS observation error
  REPORT(sigma_fast);                                           // Report fast coral observation error
  REPORT(sigma_slow);                                           // Report slow coral observation error
  
  return nll;                                                   // Return total negative log-likelihood
}
