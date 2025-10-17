#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS - Time series observations
  DATA_VECTOR(Year);                    // Time vector (years)
  DATA_VECTOR(cots_dat);                // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);                // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);                // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);                 // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);             // COTS larval immigration rate (individuals/m2/year)
  
  // COTS POPULATION PARAMETERS
  PARAMETER(log_r_cots);                // Log intrinsic growth rate of COTS (year^-1)
  PARAMETER(log_K_cots);                // Log carrying capacity of COTS (individuals/m2)
  PARAMETER(log_allee_threshold);       // Log Allee threshold density (individuals/m2)
  PARAMETER(allee_strength);            // Allee effect strength (dimensionless, 0-1)
  PARAMETER(log_mort_base);             // Log baseline mortality rate (year^-1)
  PARAMETER(log_mort_density);          // Log density-dependent mortality coefficient (m2/individuals/year)
  PARAMETER(log_temp_opt);              // Log optimal temperature for COTS recruitment (Celsius)
  PARAMETER(log_temp_width_lower);      // Log temperature tolerance below optimum (Celsius)
  PARAMETER(log_temp_width_upper);      // Log temperature tolerance above optimum (Celsius)
  PARAMETER(immigration_effect);        // Immigration enhancement factor (dimensionless)
  
  // OUTBREAK DYNAMICS PARAMETERS
  PARAMETER(log_recruitment_threshold); // Log threshold for recruitment amplification (individuals/m2/year)
  PARAMETER(recruitment_cooperativity); // Hill coefficient for recruitment response (dimensionless)
  PARAMETER(immigration_weight);        // Relative importance of immigration vs local reproduction (dimensionless)
  PARAMETER(resource_recruitment_coupling); // Strength of coral-recruitment feedback (dimensionless, 0-1)
  
  // JUVENILE SURVIVAL PARAMETERS
  PARAMETER(log_juvenile_survival_base); // Log baseline juvenile survival rate (dimensionless)
  PARAMETER(juvenile_food_sensitivity);  // Sensitivity of juvenile survival to food (dimensionless, 0-1)
  PARAMETER(log_juvenile_food_halfsat);  // Log half-saturation for food effect on juveniles (% coral)
  
  // CORAL DYNAMICS PARAMETERS
  PARAMETER(log_r_fast);                // Log intrinsic growth rate of fast coral (year^-1)
  PARAMETER(log_r_slow);                // Log intrinsic growth rate of slow coral (year^-1)
  PARAMETER(log_K_coral);               // Log carrying capacity for total coral (%)
  PARAMETER(log_temp_stress_threshold); // Log temperature threshold for coral stress (Celsius)
  PARAMETER(temp_stress_rate);          // Temperature stress mortality rate (year^-1/Celsius)
  
  // PREDATION PARAMETERS
  PARAMETER(log_attack_fast);           // Log attack rate on fast coral (m2/individuals/year)
  PARAMETER(log_attack_slow);           // Log attack rate on slow coral (m2/individuals/year)
  PARAMETER(log_handling_fast);         // Log handling time for fast coral (year)
  PARAMETER(log_handling_slow);         // Log handling time for slow coral (year)
  PARAMETER(log_conversion_eff);        // Log conversion efficiency of coral to COTS (dimensionless)
  PARAMETER(preference_fast);           // Preference for fast coral (dimensionless, 0-1)
  
  // OBSERVATION ERROR PARAMETERS
  PARAMETER(log_sigma_cots);            // Log observation error SD for COTS
  PARAMETER(log_sigma_fast);            // Log observation error SD for fast coral
  PARAMETER(log_sigma_slow);            // Log observation error SD for slow coral
  
  // Transform parameters from log scale
  Type r_cots = exp(log_r_cots);                           // Intrinsic growth rate of COTS (year^-1)
  Type K_cots = exp(log_K_cots);                           // Carrying capacity of COTS (individuals/m2)
  Type allee_threshold = exp(log_allee_threshold);         // Allee threshold (individuals/m2)
  Type mort_base = exp(log_mort_base);                     // Baseline mortality (year^-1)
  Type mort_density = exp(log_mort_density);               // Density-dependent mortality (m2/individuals/year)
  Type temp_opt = exp(log_temp_opt);                       // Optimal temperature (Celsius)
  Type temp_width_lower = exp(log_temp_width_lower);       // Temperature width below optimum (Celsius)
  Type temp_width_upper = exp(log_temp_width_upper);       // Temperature width above optimum (Celsius)
  Type recruitment_threshold = exp(log_recruitment_threshold); // Recruitment threshold (individuals/m2/year)
  Type juvenile_survival_base = exp(log_juvenile_survival_base); // Baseline juvenile survival (proportion)
  Type juvenile_food_halfsat = exp(log_juvenile_food_halfsat); // Half-saturation for food effect (% coral)
  Type r_fast = exp(log_r_fast);                           // Fast coral growth rate (year^-1)
  Type r_slow = exp(log_r_slow);                           // Slow coral growth rate (year^-1)
  Type K_coral = exp(log_K_coral);                         // Coral carrying capacity (%)
  Type temp_stress_threshold = exp(log_temp_stress_threshold); // Temperature stress threshold (Celsius)
  Type attack_fast = exp(log_attack_fast);                 // Attack rate on fast coral (m2/individuals/year)
  Type attack_slow = exp(log_attack_slow);                 // Attack rate on slow coral (m2/individuals/year)
  Type handling_fast = exp(log_handling_fast);             // Handling time fast coral (year)
  Type handling_slow = exp(log_handling_slow);             // Handling time slow coral (year)
  Type conversion_eff = exp(log_conversion_eff);           // Conversion efficiency (dimensionless)
  Type sigma_cots = exp(log_sigma_cots);                   // Observation error COTS
  Type sigma_fast = exp(log_sigma_fast);                   // Observation error fast coral
  Type sigma_slow = exp(log_sigma_slow);                   // Observation error slow coral
  
  // Initialize prediction vectors
  int n = Year.size();                                      // Number of time steps
  vector<Type> cots_pred(n);                                // Predicted COTS abundance
  vector<Type> fast_pred(n);                                // Predicted fast coral cover
  vector<Type> slow_pred(n);                                // Predicted slow coral cover
  
  // Set initial conditions from first observation
  cots_pred(0) = cots_dat(0);                               // Initial COTS from data
  fast_pred(0) = fast_dat(0);                               // Initial fast coral from data
  slow_pred(0) = slow_dat(0);                               // Initial slow coral from data
  
  // Small constant for numerical stability
  Type eps = Type(1e-8);                                    // Small constant to prevent division by zero
  
  // Minimum standard deviations for likelihood
  Type min_sigma = Type(0.01);                              // Minimum SD to prevent numerical issues
  Type sigma_cots_use = sigma_cots + min_sigma;             // Effective SD for COTS
  Type sigma_fast_use = sigma_fast + min_sigma;             // Effective SD for fast coral
  Type sigma_slow_use = sigma_slow + min_sigma;             // Effective SD for slow coral
  
  // TIME LOOP - Forward simulation
  for(int t = 1; t < n; t++) {
    
    // Previous time step values
    Type cots_prev = cots_pred(t-1);                        // COTS at t-1
    Type fast_prev = fast_pred(t-1);                        // Fast coral at t-1
    Type slow_prev = slow_pred(t-1);                        // Slow coral at t-1
    Type sst_curr = sst_dat(t);                             // Current SST
    Type immigration = cotsimm_dat(t);                      // Current immigration
    
    // EQUATION 1: Allee effect function
    // Reduces recruitment at low densities due to reduced mating success
    Type allee_factor = Type(1.0) - allee_strength * exp(-cots_prev / (allee_threshold + eps));
    allee_factor = allee_factor / (Type(1.0) + eps);        // Normalize and stabilize
    
    // EQUATION 2: Asymmetric temperature effect on COTS larval settlement
    // COTS tolerate cooler temperatures better than warmer (asymmetric response)
    Type temp_diff = sst_curr - temp_opt;                   // Deviation from optimum
    Type temp_effect_settlement;                            // Temperature effect on settlement
    if(sst_curr < temp_opt) {
      // Below optimum: gradual decline (wider tolerance)
      temp_effect_settlement = exp(-0.5 * pow(temp_diff / (temp_width_lower + eps), 2));
    } else {
      // Above optimum: steeper decline (narrower tolerance)
      temp_effect_settlement = exp(-0.5 * pow(temp_diff / (temp_width_upper + eps), 2));
    }
    
    // EQUATION 3: Local reproduction potential
    // Combines adult population, Allee effect, and temperature suitability for settlement
    Type local_reproduction = r_cots * cots_prev * allee_factor * temp_effect_settlement; // Local larval production (individuals/m2/year)
    
    // EQUATION 4: Combined larval supply
    // Weighted sum of local reproduction and external immigration
    Type larval_supply = local_reproduction + immigration_weight * immigration; // Total larval supply (individuals/m2/year)
    
    // EQUATION 5: Sigmoidal settlement amplification (Hill function)
    // Creates threshold dynamics for larval settlement success
    Type larval_supply_norm = larval_supply / (recruitment_threshold + eps); // Normalized larval supply
    Type larval_supply_power = pow(larval_supply_norm, recruitment_cooperativity); // Hill function numerator
    Type settlement_amplification = larval_supply_power / (Type(1.0) + larval_supply_power); // Sigmoidal response (0-1)
    
    // EQUATION 6: Larval settlement rate
    // Number of larvae successfully settling (not yet recruited to adult population)
    Type settlement_rate = larval_supply * settlement_amplification; // Settled juveniles (individuals/m2/year)
    
    // EQUATION 7: Food availability for juvenile survival
    // Total coral cover provides food for juvenile COTS
    Type total_coral = fast_prev + slow_prev + eps;         // Total coral available
    
    // EQUATION 8: Food-dependent juvenile survival effect (Michaelis-Menten)
    // Juvenile survival increases with coral food availability, saturating at high cover
    Type food_effect_base = total_coral / (juvenile_food_halfsat + total_coral); // Saturating food response (0-1)
    Type food_effect = Type(1.0) - juvenile_food_sensitivity + 
                      juvenile_food_sensitivity * food_effect_base; // Scaled food effect
    
    // EQUATION 9: Temperature effect on juvenile survival
    // Juveniles also sensitive to temperature (same asymmetric response as settlement)
    Type temp_effect_juvenile;                              // Temperature effect on juvenile survival
    if(sst_curr < temp_opt) {
      temp_effect_juvenile = exp(-0.5 * pow(temp_diff / (temp_width_lower + eps), 2));
    } else {
      temp_effect_juvenile = exp(-0.5 * pow(temp_diff / (temp_width_upper + eps), 2));
    }
    
    // EQUATION 10: Juvenile survival efficiency
    // Combines baseline survival with food and temperature effects
    Type juvenile_survival_efficiency = juvenile_survival_base * food_effect * temp_effect_juvenile; // Survival proportion
    
    // EQUATION 11: Recruitment to adult population
    // Settled juveniles that survive to reproductive adulthood
    Type recruitment_to_adults = settlement_rate * juvenile_survival_efficiency; // New adults (individuals/m2/year)
    
    // EQUATION 12: Resource-dependent recruitment facilitation (legacy coupling)
    // Additional facilitation from high coral cover (habitat quality)
    Type coral_facilitation = total_coral / (K_coral + eps); // Normalized coral availability (0-1)
    Type resource_effect = Type(1.0) - resource_recruitment_coupling + 
                          resource_recruitment_coupling * coral_facilitation; // Resource-dependent facilitation
    
    // EQUATION 13: Final effective recruitment rate
    // Combines juvenile recruitment with resource facilitation
    Type effective_recruitment = recruitment_to_adults * resource_effect; // Final recruitment rate (individuals/m2/year)
    
    // EQUATION 14: Type II functional response for fast coral predation
    // Captures saturation in consumption rate at high prey densities
    Type consumption_fast = (attack_fast * fast_prev * cots_prev) / 
                           (Type(1.0) + attack_fast * handling_fast * fast_prev + eps); // Fast coral consumption (% cover/year)
    
    // EQUATION 15: Type II functional response for slow coral predation
    // COTS switch to slow coral when fast coral is depleted
    Type consumption_slow = (attack_slow * slow_prev * cots_prev) / 
                           (Type(1.0) + attack_slow * handling_slow * slow_prev + eps); // Slow coral consumption (% cover/year)
    
    // EQUATION 16: Prey preference and switching
    // COTS prefer fast coral but switch when it becomes scarce
    Type fast_proportion = fast_prev / total_coral;         // Proportion of fast coral
    Type preference_weight = preference_fast * fast_proportion + 
                            (Type(1.0) - preference_fast) * (Type(1.0) - fast_proportion); // Weighted preference
    
    // EQUATION 17: Weighted consumption rates
    Type consumption_fast_weighted = consumption_fast * preference_weight; // Adjusted fast consumption
    Type consumption_slow_weighted = consumption_slow * (Type(1.0) - preference_weight); // Adjusted slow consumption
    
    // EQUATION 18: Total food intake for COTS
    Type total_consumption = consumption_fast_weighted + consumption_slow_weighted; // Total coral consumed
    
    // EQUATION 19: Density-dependent mortality
    // Increases with crowding (disease, competition)
    Type mortality_dd = mort_base + mort_density * cots_prev; // Total mortality rate (year^-1)
    
    // EQUATION 20: Starvation effect
    // Mortality increases when coral resources are depleted
    Type starvation_factor = Type(1.0) + Type(2.0) * exp(-total_coral / Type(5.0)); // Starvation multiplier
    Type mortality_total = mortality_dd * starvation_factor; // Combined mortality (year^-1)
    
    // EQUATION 21: COTS population growth with two-stage recruitment
    // Combines recruitment (from juvenile survival), predation-derived growth, and mortality
    Type cots_growth = effective_recruitment + // Recruitment from surviving juveniles
                      conversion_eff * total_consumption - // Growth from coral consumption
                      mortality_total * cots_prev; // Mortality losses
    
    // EQUATION 22: Update COTS abundance
    cots_pred(t) = cots_prev + cots_growth;                 // COTS at time t
    cots_pred(t) = cots_pred(t) / (Type(1.0) + eps);        // Stabilize
    if(cots_pred(t) < Type(0.0)) cots_pred(t) = Type(1e-6); // Prevent negative values
    
    // EQUATION 23: Temperature stress on corals
    // Warm temperatures reduce coral growth and increase mortality
    Type temp_stress = Type(0.0);                           // Initialize stress
    if(sst_curr > temp_stress_threshold) {
      temp_stress = temp_stress_rate * (sst_curr - temp_stress_threshold); // Stress mortality (year^-1)
    }
    
    // EQUATION 24: Fast coral dynamics
    // Logistic growth reduced by COTS predation and temperature stress
    Type fast_growth = r_fast * fast_prev * (Type(1.0) - (fast_prev + slow_prev) / (K_coral + eps)) - 
                      consumption_fast_weighted - // COTS predation loss
                      temp_stress * fast_prev; // Temperature stress loss
    
    // EQUATION 25: Update fast coral cover
    fast_pred(t) = fast_prev + fast_growth;                 // Fast coral at time t
    fast_pred(t) = fast_pred(t) / (Type(1.0) + eps);        // Stabilize
    if(fast_pred(t) < Type(0.0)) fast_pred(t) = Type(1e-6); // Prevent negative values
    if(fast_pred(t) > K_coral) fast_pred(t) = K_coral;     // Cap at carrying capacity
    
    // EQUATION 26: Slow coral dynamics
    // Slower growth but more resistant to disturbance
    Type slow_growth = r_slow * slow_prev * (Type(1.0) - (fast_prev + slow_prev) / (K_coral + eps)) - 
                      consumption_slow_weighted - // COTS predation loss
                      Type(0.5) * temp_stress * slow_prev; // Reduced temperature sensitivity
    
    // EQUATION 27: Update slow coral cover
    slow_pred(t) = slow_prev + slow_growth;                 // Slow coral at time t
    slow_pred(t) = slow_pred(t) / (Type(1.0) + eps);        // Stabilize
    if(slow_pred(t) < Type(0.0)) slow_pred(t) = Type(1e-6); // Prevent negative values
    if(slow_pred(t) > K_coral) slow_pred(t) = K_coral;     // Cap at carrying capacity
  }
  
  // LIKELIHOOD CALCULATION
  Type nll = Type(0.0);                                     // Initialize negative log-likelihood
  
  // EQUATION 28: COTS observation likelihood
  // Lognormal distribution for strictly positive abundance data
  for(int t = 0; t < n; t++) {
    Type log_cots_pred = log(cots_pred(t) + eps);          // Log predicted COTS
    Type log_cots_obs = log(cots_dat(t) + eps);            // Log observed COTS
    nll -= dnorm(log_cots_obs, log_cots_pred, sigma_cots_use, true); // Lognormal likelihood
  }
  
  // EQUATION 29: Fast coral observation likelihood
  // Normal distribution for percentage cover data
  for(int t = 0; t < n; t++) {
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_fast_use, true); // Normal likelihood
  }
  
  // EQUATION 30: Slow coral observation likelihood
  // Normal distribution for percentage cover data
  for(int t = 0; t < n; t++) {
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_slow_use, true); // Normal likelihood
  }
  
  // EQUATION 31: Soft parameter bounds using penalties
  // Allee strength bounded between 0 and 1
  Type penalty = Type(0.0);                                 // Initialize penalty
  if(allee_strength < Type(0.0)) penalty += Type(100.0) * pow(allee_strength, 2);
  if(allee_strength > Type(1.0)) penalty += Type(100.0) * pow(allee_strength - Type(1.0), 2);
  
  // EQUATION 32: Preference parameter bounded between 0 and 1
  if(preference_fast < Type(0.0)) penalty += Type(100.0) * pow(preference_fast, 2);
  if(preference_fast > Type(1.0)) penalty += Type(100.0) * pow(preference_fast - Type(1.0), 2);
  
  // EQUATION 33: Conversion efficiency bounded between 0 and 1
  if(conversion_eff < Type(0.0)) penalty += Type(100.0) * pow(conversion_eff, 2);
  if(conversion_eff > Type(1.0)) penalty += Type(100.0) * pow(conversion_eff - Type(1.0), 2);
  
  // EQUATION 34: Resource-recruitment coupling bounded between 0 and 1
  if(resource_recruitment_coupling < Type(0.0)) penalty += Type(100.0) * pow(resource_recruitment_coupling, 2);
  if(resource_recruitment_coupling > Type(1.0)) penalty += Type(100.0) * pow(resource_recruitment_coupling - Type(1.0), 2);
  
  // EQUATION 35: Juvenile food sensitivity bounded between 0 and 1
  if(juvenile_food_sensitivity < Type(0.0)) penalty += Type(100.0) * pow(juvenile_food_sensitivity, 2);
  if(juvenile_food_sensitivity > Type(1.0)) penalty += Type(100.0) * pow(juvenile_food_sensitivity - Type(1.0), 2);
  
  // EQUATION 36: Recruitment cooperativity should be positive
  if(recruitment_cooperativity < Type(1.0)) penalty += Type(100.0) * pow(recruitment_cooperativity - Type(1.0), 2);
  
  // EQUATION 37: Juvenile survival should be between 0 and 1
  if(juvenile_survival_base < Type(0.0)) penalty += Type(100.0) * pow(juvenile_survival_base, 2);
  if(juvenile_survival_base > Type(1.0)) penalty += Type(100.0) * pow(juvenile_survival_base - Type(1.0), 2);
  
  nll += penalty;                                           // Add penalties to likelihood
  
  // REPORTING
  REPORT(cots_pred);                                        // Report predicted COTS
  REPORT(fast_pred);                                        // Report predicted fast coral
  REPORT(slow_pred);                                        // Report predicted slow coral
  REPORT(r_cots);                                           // Report COTS growth rate
  REPORT(K_cots);                                           // Report COTS carrying capacity
  REPORT(allee_threshold);                                  // Report Allee threshold
  REPORT(allee_strength);                                   // Report Allee strength
  REPORT(mort_base);                                        // Report baseline mortality
  REPORT(mort_density);                                     // Report density-dependent mortality
  REPORT(temp_opt);                                         // Report optimal temperature
  REPORT(temp_width_lower);                                 // Report temperature width lower
  REPORT(temp_width_upper);                                 // Report temperature width upper
  REPORT(immigration_effect);                               // Report immigration effect
  REPORT(recruitment_threshold);                            // Report recruitment threshold
  REPORT(recruitment_cooperativity);                        // Report recruitment cooperativity
  REPORT(immigration_weight);                               // Report immigration weight
  REPORT(resource_recruitment_coupling);                    // Report resource-recruitment coupling
  REPORT(juvenile_survival_base);                           // Report baseline juvenile survival
  REPORT(juvenile_food_sensitivity);                        // Report juvenile food sensitivity
  REPORT(juvenile_food_halfsat);                            // Report juvenile food half-saturation
  REPORT(r_fast);                                           // Report fast coral growth
  REPORT(r_slow);                                           // Report slow coral growth
  REPORT(K_coral);                                          // Report coral carrying capacity
  REPORT(temp_stress_threshold);                            // Report temperature stress threshold
  REPORT(temp_stress_rate);                                 // Report temperature stress rate
  REPORT(attack_fast);                                      // Report attack rate fast
  REPORT(attack_slow);                                      // Report attack rate slow
  REPORT(handling_fast);                                    // Report handling time fast
  REPORT(handling_slow);                                    // Report handling time slow
  REPORT(conversion_eff);                                   // Report conversion efficiency
  REPORT(preference_fast);                                  // Report preference for fast coral
  REPORT(sigma_cots);                                       // Report COTS observation error
  REPORT(sigma_fast);                                       // Report fast coral observation error
  REPORT(sigma_slow);                                       // Report slow coral observation error
  
  return nll;                                               // Return negative log-likelihood
}
