#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS - Time series observations
  DATA_VECTOR(Year);                    // Time vector (years)
  DATA_VECTOR(cots_dat);               // Adult COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);               // Fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);               // Slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);                // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);            // COTS larval immigration (individuals/m2/year)
  
  // COTS POPULATION PARAMETERS
  PARAMETER(log_r_cots);                // Log intrinsic recruitment rate (year^-1)
  PARAMETER(log_K_cots);                // Log carrying capacity based on coral availability (individuals/m2)
  PARAMETER(log_allee_threshold);       // Log Allee threshold density (individuals/m2)
  PARAMETER(log_allee_strength);        // Log strength of Allee effect (dimensionless)
  PARAMETER(log_mort_cots);             // Log baseline natural mortality rate (year^-1)
  PARAMETER(log_mort_density_coef);     // Log density-dependent mortality coefficient (m2/individuals/year)
  PARAMETER(log_temp_opt_cots);         // Log optimal temperature for COTS recruitment (Celsius)
  PARAMETER(log_temp_width_cots);       // Log temperature tolerance width (Celsius)
  PARAMETER(immigration_effect);        // Immigration enhancement factor (dimensionless)
  PARAMETER(log_threshold_sensitivity); // Log sensitivity of immigration to Allee threshold proximity
  
  // CORAL PREDATION PARAMETERS
  PARAMETER(log_attack_fast);           // Log attack rate on fast-growing coral (m2/individuals/year)
  PARAMETER(log_attack_slow);           // Log attack rate on slow-growing coral (m2/individuals/year)
  PARAMETER(log_handling_fast);         // Log handling time for fast coral (%^-1 year)
  PARAMETER(log_handling_slow);         // Log handling time for slow coral (%^-1 year)
  PARAMETER(log_conversion_eff);        // Log conversion efficiency of coral to COTS biomass (dimensionless)
  PARAMETER(preference_fast);           // Preference for fast-growing coral (0-1, logit scale)
  
  // CORAL GROWTH PARAMETERS
  PARAMETER(log_r_fast);                // Log intrinsic growth rate fast coral (year^-1)
  PARAMETER(log_r_slow);                // Log intrinsic growth rate slow coral (year^-1)
  PARAMETER(log_K_fast);                // Log carrying capacity fast coral (%)
  PARAMETER(log_K_slow);                // Log carrying capacity slow coral (%)
  PARAMETER(log_temp_opt_coral);        // Log optimal temperature for coral growth (Celsius)
  PARAMETER(log_temp_stress_width);     // Log temperature stress tolerance (Celsius)
  PARAMETER(competition_coef);          // Interspecific competition coefficient (dimensionless)
  
  // OBSERVATION ERROR PARAMETERS
  PARAMETER(log_sigma_cots);            // Log observation error SD for COTS (individuals/m2)
  PARAMETER(log_sigma_fast);            // Log observation error SD for fast coral (%)
  PARAMETER(log_sigma_slow);            // Log observation error SD for slow coral (%)
  
  // Transform parameters to natural scale
  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type allee_threshold = exp(log_allee_threshold);
  Type allee_strength = exp(log_allee_strength);
  Type mort_cots = exp(log_mort_cots);
  Type mort_density_coef = exp(log_mort_density_coef);
  Type temp_opt_cots = exp(log_temp_opt_cots);
  Type temp_width_cots = exp(log_temp_width_cots);
  Type threshold_sensitivity = exp(log_threshold_sensitivity);
  
  Type attack_fast = exp(log_attack_fast);
  Type attack_slow = exp(log_attack_slow);
  Type handling_fast = exp(log_handling_fast);
  Type handling_slow = exp(log_handling_slow);
  Type conversion_eff = exp(log_conversion_eff);
  Type pref_fast = invlogit(preference_fast);
  
  Type r_fast = exp(log_r_fast);
  Type r_slow = exp(log_r_slow);
  Type K_fast = exp(log_K_fast);
  Type K_slow = exp(log_K_slow);
  Type temp_opt_coral = exp(log_temp_opt_coral);
  Type temp_stress_width = exp(log_temp_stress_width);
  
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_fast = exp(log_sigma_fast);
  Type sigma_slow = exp(log_sigma_slow);
  
  // Minimum standard deviations for numerical stability
  Type min_sigma = Type(0.01);
  sigma_cots = sigma_cots + min_sigma;
  sigma_fast = sigma_fast + min_sigma;
  sigma_slow = sigma_slow + min_sigma;
  
  // Initialize prediction vectors
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Set initial conditions from first observation
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Small constant for numerical stability
  Type eps = Type(1e-8);
  
  // Initialize negative log-likelihood
  Type nll = Type(0.0);
  
  // Add first observation to likelihood
  nll -= dnorm(log(cots_dat(0) + eps), log(cots_pred(0) + eps), sigma_cots, true);
  nll -= dnorm(fast_dat(0), fast_pred(0), sigma_fast, true);
  nll -= dnorm(slow_dat(0), slow_pred(0), sigma_slow, true);
  
  // TIME LOOP - Dynamic model equations
  for(int t = 1; t < n; t++) {
    
    // Previous time step values (avoid data leakage)
    Type cots_prev = cots_pred(t-1);
    Type fast_prev = fast_pred(t-1);
    Type slow_prev = slow_pred(t-1);
    Type sst_prev = sst_dat(t-1);
    Type immigration_prev = cotsimm_dat(t-1);
    
    // Ensure non-negative values using CppAD::CondExpGt for smooth max
    cots_prev = CppAD::CondExpGt(cots_prev, eps, cots_prev, eps);
    fast_prev = CppAD::CondExpGt(fast_prev, eps, fast_prev, eps);
    slow_prev = CppAD::CondExpGt(slow_prev, eps, slow_prev, eps);
    
    // EQUATION 1: Temperature effect on COTS recruitment (Gaussian response)
    Type temp_effect_cots = exp(-pow(sst_prev - temp_opt_cots, 2) / (Type(2.0) * pow(temp_width_cots, 2) + eps));
    
    // EQUATION 2: Allee effect function (sigmoid transition from low to high recruitment)
    Type allee_effect = Type(1.0) / (Type(1.0) + exp(-allee_strength * (cots_prev - allee_threshold)));
    
    // EQUATION 3: Total coral availability for COTS food resource
    Type total_coral = fast_prev + slow_prev + eps;
    
    // EQUATION 4: Carrying capacity modulation by coral availability
    Type K_effective = K_cots * (total_coral / (Type(50.0) + eps));
    
    // EQUATION 5: Density-dependent recruitment limitation
    Type density_limit = Type(1.0) - (cots_prev / (K_effective + eps));
    density_limit = CppAD::CondExpGt(density_limit, Type(0.0), density_limit, Type(0.0));
    
    // EQUATION 6: Threshold-sensitive immigration trigger (NEW FORMULATION)
    // Immigration is most effective when local density is near the Allee threshold
    // This creates synergistic outbreak triggering when immigration pulses coincide with primed populations
    Type density_relative_to_threshold = (cots_prev - allee_threshold) / (allee_threshold + eps);
    Type immigration_sensitivity = Type(1.0) / (Type(1.0) + exp(-threshold_sensitivity * density_relative_to_threshold));
    Type immigration_trigger = Type(1.0) + immigration_effect * immigration_prev * immigration_sensitivity;
    
    // EQUATION 7: COTS recruitment rate (combines all factors)
    Type recruitment = r_cots * cots_prev * temp_effect_cots * allee_effect * density_limit * immigration_trigger;
    
    // EQUATION 8: Density-dependent mortality (disease, crowding at high density)
    Type mortality = (mort_cots + mort_density_coef * cots_prev) * cots_prev;
    
    // EQUATION 9: Type II functional response for fast-growing coral predation
    Type predation_fast = (attack_fast * pref_fast * cots_prev * fast_prev) / (Type(1.0) + handling_fast * attack_fast * pref_fast * fast_prev + handling_slow * attack_slow * (Type(1.0) - pref_fast) * slow_prev + eps);
    
    // EQUATION 10: Type II functional response for slow-growing coral predation
    Type predation_slow = (attack_slow * (Type(1.0) - pref_fast) * cots_prev * slow_prev) / (Type(1.0) + handling_fast * attack_fast * pref_fast * fast_prev + handling_slow * attack_slow * (Type(1.0) - pref_fast) * slow_prev + eps);
    
    // EQUATION 11: COTS population change (recruitment - mortality + immigration)
    Type cots_change = recruitment - mortality + immigration_prev;
    Type cots_new = cots_prev + cots_change;
    cots_pred(t) = CppAD::CondExpGt(cots_new, eps, cots_new, eps);
    
    // EQUATION 12: Temperature stress on coral growth (reduced growth at temperature extremes)
    Type temp_stress_coral = exp(-pow(sst_prev - temp_opt_coral, 2) / (Type(2.0) * pow(temp_stress_width, 2) + eps));
    
    // EQUATION 13: Fast coral logistic growth with competition and temperature stress
    Type fast_growth = r_fast * fast_prev * (Type(1.0) - (fast_prev + competition_coef * slow_prev) / (K_fast + eps)) * temp_stress_coral;
    
    // EQUATION 14: Fast coral population change (growth - predation)
    Type fast_change = fast_growth - predation_fast;
    Type fast_new = fast_prev + fast_change;
    fast_new = CppAD::CondExpGt(fast_new, eps, fast_new, eps);
    fast_pred(t) = CppAD::CondExpLt(fast_new, Type(100.0), fast_new, Type(100.0)); // Cap at 100% cover
    
    // EQUATION 15: Slow coral logistic growth with competition and temperature stress
    Type slow_growth = r_slow * slow_prev * (Type(1.0) - (slow_prev + competition_coef * fast_prev) / (K_slow + eps)) * temp_stress_coral;
    
    // EQUATION 16: Slow coral population change (growth - predation)
    Type slow_change = slow_growth - predation_slow;
    Type slow_new = slow_prev + slow_change;
    slow_new = CppAD::CondExpGt(slow_new, eps, slow_new, eps);
    slow_pred(t) = CppAD::CondExpLt(slow_new, Type(100.0), slow_new, Type(100.0)); // Cap at 100% cover
    
    // LIKELIHOOD CONTRIBUTIONS (all observations included)
    // Use lognormal for COTS (strictly positive, spans orders of magnitude)
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots, true);
    
    // Use normal for coral cover (percentage data)
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_fast, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_slow, true);
  }
  
  // SOFT PARAMETER BOUNDS (biological constraints via penalties)
  // COTS parameters should be positive and reasonable
  Type penalty = Type(0.0);
  
  // Recruitment rate: 0.1 to 5.0 year^-1
  penalty += CppAD::CondExpLt(r_cots, Type(0.1), Type(10.0) * pow(Type(0.1) - r_cots, 2), Type(0.0));
  penalty += CppAD::CondExpGt(r_cots, Type(5.0), Type(10.0) * pow(r_cots - Type(5.0), 2), Type(0.0));
  
  // Carrying capacity: 0.1 to 20 individuals/m2
  penalty += CppAD::CondExpLt(K_cots, Type(0.1), Type(10.0) * pow(Type(0.1) - K_cots, 2), Type(0.0));
  penalty += CppAD::CondExpGt(K_cots, Type(20.0), Type(10.0) * pow(K_cots - Type(20.0), 2), Type(0.0));
  
  // Allee threshold: 0.01 to 2.0 individuals/m2
  penalty += CppAD::CondExpLt(allee_threshold, Type(0.01), Type(10.0) * pow(Type(0.01) - allee_threshold, 2), Type(0.0));
  penalty += CppAD::CondExpGt(allee_threshold, Type(2.0), Type(10.0) * pow(allee_threshold - Type(2.0), 2), Type(0.0));
  
  // Mortality rate: 0.05 to 2.0 year^-1
  penalty += CppAD::CondExpLt(mort_cots, Type(0.05), Type(10.0) * pow(Type(0.05) - mort_cots, 2), Type(0.0));
  penalty += CppAD::CondExpGt(mort_cots, Type(2.0), Type(10.0) * pow(mort_cots - Type(2.0), 2), Type(0.0));
  
  // Temperature optimum for COTS: 24 to 30 Celsius
  penalty += CppAD::CondExpLt(temp_opt_cots, Type(24.0), Type(10.0) * pow(Type(24.0) - temp_opt_cots, 2), Type(0.0));
  penalty += CppAD::CondExpGt(temp_opt_cots, Type(30.0), Type(10.0) * pow(temp_opt_cots - Type(30.0), 2), Type(0.0));
  
  // Coral growth rates: 0.01 to 1.0 year^-1
  penalty += CppAD::CondExpLt(r_fast, Type(0.01), Type(10.0) * pow(Type(0.01) - r_fast, 2), Type(0.0));
  penalty += CppAD::CondExpGt(r_fast, Type(1.0), Type(10.0) * pow(r_fast - Type(1.0), 2), Type(0.0));
  penalty += CppAD::CondExpLt(r_slow, Type(0.01), Type(10.0) * pow(Type(0.01) - r_slow, 2), Type(0.0));
  penalty += CppAD::CondExpGt(r_slow, Type(0.5), Type(10.0) * pow(r_slow - Type(0.5), 2), Type(0.0));
  
  // Coral carrying capacities: 10 to 100%
  penalty += CppAD::CondExpLt(K_fast, Type(10.0), Type(10.0) * pow(Type(10.0) - K_fast, 2), Type(0.0));
  penalty += CppAD::CondExpGt(K_fast, Type(100.0), Type(10.0) * pow(K_fast - Type(100.0), 2), Type(0.0));
  penalty += CppAD::CondExpLt(K_slow, Type(10.0), Type(10.0) * pow(Type(10.0) - K_slow, 2), Type(0.0));
  penalty += CppAD::CondExpGt(K_slow, Type(100.0), Type(10.0) * pow(K_slow - Type(100.0), 2), Type(0.0));
  
  // Attack rates: 0.01 to 10.0 m2/individuals/year
  penalty += CppAD::CondExpLt(attack_fast, Type(0.01), Type(10.0) * pow(Type(0.01) - attack_fast, 2), Type(0.0));
  penalty += CppAD::CondExpGt(attack_fast, Type(10.0), Type(10.0) * pow(attack_fast - Type(10.0), 2), Type(0.0));
  penalty += CppAD::CondExpLt(attack_slow, Type(0.01), Type(10.0) * pow(Type(0.01) - attack_slow, 2), Type(0.0));
  penalty += CppAD::CondExpGt(attack_slow, Type(10.0), Type(10.0) * pow(attack_slow - Type(10.0), 2), Type(0.0));
  
  // Threshold sensitivity: 0.1 to 2.0 (controls sharpness of immigration trigger)
  penalty += CppAD::CondExpLt(threshold_sensitivity, Type(0.1), Type(10.0) * pow(Type(0.1) - threshold_sensitivity, 2), Type(0.0));
  penalty += CppAD::CondExpGt(threshold_sensitivity, Type(2.0), Type(10.0) * pow(threshold_sensitivity - Type(2.0), 2), Type(0.0));
  
  nll += penalty;
  
  // REPORT PREDICTIONS AND PARAMETERS
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  REPORT(r_cots);
  REPORT(K_cots);
  REPORT(allee_threshold);
  REPORT(allee_strength);
  REPORT(mort_cots);
  REPORT(mort_density_coef);
  REPORT(temp_opt_cots);
  REPORT(temp_width_cots);
  REPORT(immigration_effect);
  REPORT(threshold_sensitivity);
  
  REPORT(attack_fast);
  REPORT(attack_slow);
  REPORT(handling_fast);
  REPORT(handling_slow);
  REPORT(conversion_eff);
  REPORT(pref_fast);
  
  REPORT(r_fast);
  REPORT(r_slow);
  REPORT(K_fast);
  REPORT(K_slow);
  REPORT(temp_opt_coral);
  REPORT(temp_stress_width);
  REPORT(competition_coef);
  
  REPORT(sigma_cots);
  REPORT(sigma_fast);
  REPORT(sigma_slow);
  
  return nll;
}
