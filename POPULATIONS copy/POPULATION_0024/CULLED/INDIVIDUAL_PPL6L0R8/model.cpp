#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS - Time series observations
  DATA_VECTOR(Year);                    // Time vector (years)
  DATA_VECTOR(cots_dat);                // Observed COTS abundance (individuals/m2) - TOTAL (juveniles + adults)
  DATA_VECTOR(fast_dat);                // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);                // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);                 // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);             // COTS larval immigration rate (individuals/m2/year)
  
  // COTS POPULATION PARAMETERS
  PARAMETER(log_r_cots);                // Log intrinsic growth rate of adult COTS (year^-1)
  PARAMETER(log_K_cots);                // Log carrying capacity of total COTS (individuals/m2)
  PARAMETER(log_allee_threshold);       // Log Allee threshold density for adults (individuals/m2)
  PARAMETER(allee_strength);            // Allee effect strength (dimensionless, 0-1)
  PARAMETER(log_mort_base);             // Log baseline mortality rate for adults (year^-1)
  PARAMETER(log_mort_juvenile);         // Log baseline mortality rate for juveniles (year^-1)
  PARAMETER(log_mort_density);          // Log density-dependent mortality coefficient (m2/individuals/year)
  PARAMETER(log_temp_opt);              // Log optimal temperature for COTS recruitment (Celsius)
  PARAMETER(log_temp_width);            // Log temperature tolerance width (Celsius)
  PARAMETER(immigration_effect);        // Immigration enhancement factor (dimensionless)
  PARAMETER(log_recruit_max);           // Log maximum recruitment rate to juvenile stage (individuals/m2/year)
  PARAMETER(recruit_threshold);         // Favorability threshold for recruitment pulse activation (0-1)
  PARAMETER(log_optimal_recruit_density); // Log optimal adult density for recruitment facilitation (individuals/m2)
  PARAMETER(log_recruit_density_width); // Log density width for recruitment facilitation (individuals/m2)
  
  // STAGE TRANSITION PARAMETERS
  PARAMETER(log_maturation_rate);       // Log baseline maturation rate juvenile->adult (year^-1)
  PARAMETER(log_maturation_halfsat);    // Log half-saturation for food-dependent maturation (% coral)
  PARAMETER(maturation_max_multiplier); // Maximum maturation rate multiplier under optimal food
  PARAMETER(juvenile_feeding_fraction); // Fraction of adult feeding rate for juveniles (0-1)
  
  // CORAL DYNAMICS PARAMETERS
  PARAMETER(log_r_fast);                // Log intrinsic growth rate of fast coral (year^-1)
  PARAMETER(log_r_slow);                // Log intrinsic growth rate of slow coral (year^-1)
  PARAMETER(log_K_coral);               // Log carrying capacity for total coral (%)
  PARAMETER(log_temp_stress_threshold); // Log temperature threshold for coral stress (Celsius)
  PARAMETER(temp_stress_rate);          // Temperature stress mortality rate (year^-1/Celsius)
  
  // PREDATION PARAMETERS
  PARAMETER(log_attack_fast);           // Log attack rate on fast coral by adults (m2/individuals/year)
  PARAMETER(log_attack_slow);           // Log attack rate on slow coral by adults (m2/individuals/year)
  PARAMETER(log_handling_fast);         // Log handling time for fast coral (year)
  PARAMETER(log_handling_slow);         // Log handling time for slow coral (year)
  PARAMETER(log_conversion_eff);        // Log conversion efficiency of coral to adult COTS (dimensionless)
  PARAMETER(preference_fast);           // Preference for fast coral (dimensionless, 0-1)
  
  // OBSERVATION ERROR PARAMETERS
  PARAMETER(log_sigma_cots);            // Log observation error SD for total COTS
  PARAMETER(log_sigma_fast);            // Log observation error SD for fast coral
  PARAMETER(log_sigma_slow);            // Log observation error SD for slow coral
  
  // Transform parameters from log scale
  Type r_cots = exp(log_r_cots);                           // Intrinsic growth rate of adults (year^-1)
  Type K_cots = exp(log_K_cots);                           // Carrying capacity of total COTS (individuals/m2)
  Type allee_threshold = exp(log_allee_threshold);         // Allee threshold for adults (individuals/m2)
  Type mort_base = exp(log_mort_base);                     // Baseline mortality adults (year^-1)
  Type mort_juvenile = exp(log_mort_juvenile);             // Baseline mortality juveniles (year^-1)
  Type mort_density = exp(log_mort_density);               // Density-dependent mortality (m2/individuals/year)
  Type temp_opt = exp(log_temp_opt);                       // Optimal temperature (Celsius)
  Type temp_width = exp(log_temp_width);                   // Temperature width (Celsius)
  Type recruit_max = exp(log_recruit_max);                 // Maximum recruitment rate (individuals/m2/year)
  Type optimal_recruit_density = exp(log_optimal_recruit_density); // Optimal adult density for recruitment (individuals/m2)
  Type recruit_density_width = exp(log_recruit_density_width);     // Density width for recruitment (individuals/m2)
  Type maturation_rate = exp(log_maturation_rate);         // Baseline maturation rate (year^-1)
  Type maturation_halfsat = exp(log_maturation_halfsat);   // Half-saturation for maturation (% coral)
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
  
  // Small constant for numerical stability
  Type eps = Type(1e-8);                                    // Small constant to prevent division by zero
  
  // Minimum standard deviations for likelihood
  Type min_sigma = Type(0.01);                              // Minimum SD to prevent numerical issues
  Type sigma_cots_use = sigma_cots + min_sigma;             // Effective SD for COTS
  Type sigma_fast_use = sigma_fast + min_sigma;             // Effective SD for fast coral
  Type sigma_slow_use = sigma_slow + min_sigma;             // Effective SD for slow coral
  
  // Get number of time steps
  int n = Year.size();                                      // Number of time steps
  
  // Initialize prediction vectors - NOW SEPARATE JUVENILE AND ADULT STAGES
  vector<Type> cots_juv_pred(n);   // Juvenile COTS
  vector<Type> cots_adult_pred(n); // Adult COTS
  vector<Type> cots_pred(n);       // Total COTS (for comparison with observations)
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Set initial conditions from observations
  for(int t = 0; t < n; t++) {
    cots_juv_pred(t) = Type(0.0);
    cots_adult_pred(t) = Type(0.0);
    cots_pred(t) = Type(0.0);
    fast_pred(t) = Type(0.0);
    slow_pred(t) = Type(0.0);
  }
  
  // Initialize with assumption that observed COTS are mostly adults (juveniles cryptic)
  cots_adult_pred(0) = cots_dat(0) * Type(0.8);  // 80% adults
  cots_juv_pred(0) = cots_dat(0) * Type(0.2);    // 20% juveniles
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Calculate reference values for standardization in recruitment pulse
  Type max_immigration = Type(0.0);                         // Maximum immigration in dataset
  for(int t = 0; t < n; t++) {
    if(cotsimm_dat(t) > max_immigration) max_immigration = cotsimm_dat(t);
  }
  max_immigration += eps;                                   // Prevent division by zero
  
  // TIME LOOP - Forward simulation to generate predictions
  for(int t = 1; t < n; t++) {
    
    // Previous time step values (using predictions, not observations)
    Type cots_juv_prev = cots_juv_pred(t-1);                // Juveniles at t-1
    Type cots_adult_prev = cots_adult_pred(t-1);            // Adults at t-1
    Type cots_total_prev = cots_juv_prev + cots_adult_prev; // Total COTS at t-1
    Type fast_prev = fast_pred(t-1);                        // Fast coral at t-1
    Type slow_prev = slow_pred(t-1);                        // Slow coral at t-1
    Type sst_curr = sst_dat(t);                             // Current SST
    Type immigration = cotsimm_dat(t);                      // Current immigration
    
    // EQUATION 1: Allee effect function (based on ADULT density)
    // Reduces recruitment at low adult densities due to reduced mating success
    Type allee_factor = Type(1.0) - allee_strength * exp(-cots_adult_prev / (allee_threshold + eps));
    allee_factor = allee_factor / (Type(1.0) + eps);        // Normalize and stabilize
    
    // EQUATION 2: Temperature effect on COTS recruitment
    // Gaussian function centered on optimal temperature
    Type temp_diff = sst_curr - temp_opt;                   // Deviation from optimum
    Type temp_effect = exp(-Type(0.5) * pow(temp_diff / (temp_width + eps), 2)); // Gaussian temperature response
    
    // EQUATION 3: Immigration enhancement
    // Larval immigration boosts local recruitment
    Type immigration_boost = Type(1.0) + immigration_effect * immigration; // Immigration multiplier
    
    // EQUATION 4: Total coral availability for feeding
    Type total_coral = fast_prev + slow_prev + eps;         // Total coral available
    
    // EQUATION 5: Type II functional response for fast coral predation BY ADULTS
    // Captures saturation in consumption rate at high prey densities
    Type consumption_fast_adult = (attack_fast * fast_prev * cots_adult_prev) / 
                                  (Type(1.0) + attack_fast * handling_fast * fast_prev + eps); // Fast coral consumption by adults
    
    // EQUATION 6: Type II functional response for slow coral predation BY ADULTS
    // COTS switch to slow coral when fast coral is depleted
    Type consumption_slow_adult = (attack_slow * slow_prev * cots_adult_prev) / 
                                  (Type(1.0) + attack_slow * handling_slow * slow_prev + eps); // Slow coral consumption by adults
    
    // EQUATION 7: Juvenile feeding (reduced rate)
    // Juveniles are cryptic and feed at much lower rates
    Type consumption_fast_juv = juvenile_feeding_fraction * 
                                (attack_fast * fast_prev * cots_juv_prev) / 
                                (Type(1.0) + attack_fast * handling_fast * fast_prev + eps);
    Type consumption_slow_juv = juvenile_feeding_fraction * 
                                (attack_slow * slow_prev * cots_juv_prev) / 
                                (Type(1.0) + attack_slow * handling_slow * slow_prev + eps);
    
    // EQUATION 8: Prey preference and switching
    // COTS prefer fast coral but switch when it becomes scarce
    Type fast_proportion = fast_prev / total_coral;         // Proportion of fast coral
    Type preference_weight = preference_fast * fast_proportion + 
                            (Type(1.0) - preference_fast) * (Type(1.0) - fast_proportion); // Weighted preference
    
    // EQUATION 9: Weighted consumption rates (adults)
    Type consumption_fast_weighted_adult = consumption_fast_adult * preference_weight;
    Type consumption_slow_weighted_adult = consumption_slow_adult * (Type(1.0) - preference_weight);
    
    // EQUATION 10: Weighted consumption rates (juveniles)
    Type consumption_fast_weighted_juv = consumption_fast_juv * preference_weight;
    Type consumption_slow_weighted_juv = consumption_slow_juv * (Type(1.0) - preference_weight);
    
    // EQUATION 11: Total consumption by each stage
    Type total_consumption_adult = consumption_fast_weighted_adult + consumption_slow_weighted_adult;
    Type total_consumption_juv = consumption_fast_weighted_juv + consumption_slow_weighted_juv;
    
    // EQUATION 12: Total consumption (for coral dynamics)
    Type consumption_fast_total = consumption_fast_weighted_adult + consumption_fast_weighted_juv;
    Type consumption_slow_total = consumption_slow_weighted_adult + consumption_slow_weighted_juv;
    
    // EQUATION 13: Density-dependent mortality (applies to both stages)
    // Increases with total crowding (disease, competition)
    Type mortality_dd = mort_density * cots_total_prev;     // Density-dependent component
    
    // EQUATION 14: Starvation effect
    // Mortality increases when coral resources are depleted
    Type starvation_factor = Type(1.0) + Type(2.0) * exp(-total_coral / Type(5.0)); // Starvation multiplier
    
    // EQUATION 15: Stage-specific total mortality
    Type mortality_total_adult = (mort_base + mortality_dd) * starvation_factor;     // Adult mortality
    Type mortality_total_juv = (mort_juvenile + mortality_dd) * starvation_factor;   // Juvenile mortality (higher baseline)
    
    // EQUATION 16: Recruitment pulse mechanism - Temperature favorability
    // Optimal temperature promotes larval development and settlement
    Type temp_favorability = temp_effect;                   // Temperature component (0-1)
    
    // EQUATION 17: Recruitment pulse mechanism - Immigration favorability
    // High larval supply increases probability of mass settlement
    Type immigration_favorability = immigration / max_immigration; // Immigration component (0-1)
    
    // EQUATION 18: Recruitment pulse mechanism - Food favorability
    // Adequate coral cover ensures juvenile survival post-settlement
    Type food_favorability = total_coral / (K_coral + eps); // Food availability component (0-1)
    
    // EQUATION 19: Composite favorability index for recruitment
    // Multiplicative combination - all three must be reasonably high for mass recruitment
    Type favorability_index = temp_favorability * immigration_favorability * food_favorability;
    
    // EQUATION 20: Recruitment pulse activation
    // Sigmoidal threshold function creates episodic recruitment events
    Type recruit_activation = Type(1.0) / (Type(1.0) + exp(-Type(20.0) * (favorability_index - recruit_threshold)));
    
    // EQUATION 21: Dome-shaped density-dependent recruitment facilitation (based on ADULT density)
    // Creates a "sweet spot" for outbreak amplification at intermediate adult densities
    Type density_deviation = cots_adult_prev - optimal_recruit_density;
    Type density_facilitation = exp(-Type(0.5) * pow(density_deviation / (recruit_density_width + eps), 2));
    
    // EQUATION 22: Recruitment pulse flux TO JUVENILE STAGE
    // Mass recruitment of larvae when conditions are favorable AND adult density is optimal
    Type recruitment_pulse = recruit_max * recruit_activation * favorability_index * density_facilitation;
    
    // EQUATION 23: Food-dependent maturation rate
    // Juveniles mature faster when coral is abundant (Michaelis-Menten kinetics)
    Type food_effect_maturation = (maturation_max_multiplier * total_coral) / 
                                  (maturation_halfsat + total_coral);
    Type maturation_rate_effective = maturation_rate * (Type(1.0) + food_effect_maturation);
    
    // EQUATION 24: Maturation flux from juvenile to adult stage
    // Rate depends on juvenile abundance and food-modulated maturation rate
    Type maturation_flux = maturation_rate_effective * cots_juv_prev;
    
    // EQUATION 25: Adult population growth from existing adults
    // Standard logistic growth with Allee effect, temperature, and immigration modifiers
    // Plus growth from converting consumed coral to new biomass
    Type cots_growth_adults = r_cots * cots_adult_prev * allee_factor * temp_effect * immigration_boost * 
                             (Type(1.0) - cots_total_prev / (K_cots + eps)) + // Logistic growth (total density)
                             conversion_eff * total_consumption_adult; // Growth from coral consumption
    
    // EQUATION 26: JUVENILE DYNAMICS
    // Gains: recruitment pulse
    // Losses: maturation to adult stage, mortality
    Type cots_juv_change = recruitment_pulse - maturation_flux - mortality_total_juv * cots_juv_prev;
    
    // EQUATION 27: ADULT DYNAMICS
    // Gains: maturation from juvenile stage, growth from reproduction and feeding
    // Losses: mortality
    Type cots_adult_change = maturation_flux + cots_growth_adults - mortality_total_adult * cots_adult_prev;
    
    // EQUATION 28: JUVENILE PREDICTION - Update juvenile abundance for time t
    cots_juv_pred(t) = cots_juv_prev + cots_juv_change;
    cots_juv_pred(t) = cots_juv_pred(t) / (Type(1.0) + eps); // Stabilize
    if(cots_juv_pred(t) < Type(0.0)) cots_juv_pred(t) = Type(1e-6); // Prevent negative
    
    // EQUATION 29: ADULT PREDICTION - Update adult abundance for time t
    cots_adult_pred(t) = cots_adult_prev + cots_adult_change;
    cots_adult_pred(t) = cots_adult_pred(t) / (Type(1.0) + eps); // Stabilize
    if(cots_adult_pred(t) < Type(0.0)) cots_adult_pred(t) = Type(1e-6); // Prevent negative
    
    // EQUATION 30: TOTAL COTS PREDICTION (for comparison with observations)
    cots_pred(t) = cots_juv_pred(t) + cots_adult_pred(t);
    
    // EQUATION 31: Temperature stress on corals
    // Warm temperatures reduce coral growth and increase mortality
    Type temp_stress = Type(0.0);                           // Initialize stress
    if(sst_curr > temp_stress_threshold) {
      temp_stress = temp_stress_rate * (sst_curr - temp_stress_threshold); // Stress mortality (year^-1)
    }
    
    // EQUATION 32: Fast coral dynamics
    // Logistic growth reduced by COTS predation (both stages) and temperature stress
    Type fast_growth = r_fast * fast_prev * (Type(1.0) - (fast_prev + slow_prev) / (K_coral + eps)) - 
                      consumption_fast_total - // Total COTS predation loss
                      temp_stress * fast_prev; // Temperature stress loss
    
    // EQUATION 33: FAST CORAL PREDICTION - Update fast coral cover for time t
    fast_pred(t) = fast_prev + fast_growth;
    fast_pred(t) = fast_pred(t) / (Type(1.0) + eps);        // Stabilize
    if(fast_pred(t) < Type(0.0)) fast_pred(t) = Type(1e-6); // Prevent negative
    if(fast_pred(t) > K_coral) fast_pred(t) = K_coral;     // Cap at carrying capacity
    
    // EQUATION 34: Slow coral dynamics
    // Slower growth but more resistant to disturbance
    Type slow_growth = r_slow * slow_prev * (Type(1.0) - (fast_prev + slow_prev) / (K_coral + eps)) - 
                      consumption_slow_total - // Total COTS predation loss
                      Type(0.5) * temp_stress * slow_prev; // Reduced temperature sensitivity
    
    // EQUATION 35: SLOW CORAL PREDICTION - Update slow coral cover for time t
    slow_pred(t) = slow_prev + slow_growth;
    slow_pred(t) = slow_pred(t) / (Type(1.0) + eps);        // Stabilize
    if(slow_pred(t) < Type(0.0)) slow_pred(t) = Type(1e-6); // Prevent negative
    if(slow_pred(t) > K_coral) slow_pred(t) = K_coral;     // Cap at carrying capacity
  }
  
  // LIKELIHOOD CALCULATION
  Type nll = Type(0.0);                                     // Initialize negative log-likelihood
  
  // EQUATION 36: COTS observation likelihood (compare TOTAL COTS to observations)
  // Lognormal distribution for strictly positive abundance data
  for(int t = 0; t < n; t++) {
    Type log_cots_pred = log(cots_pred(t) + eps);          // Log predicted total COTS
    Type log_cots_obs = log(cots_dat(t) + eps);            // Log observed COTS
    nll -= dnorm(log_cots_obs, log_cots_pred, sigma_cots_use, true); // Lognormal likelihood
  }
  
  // EQUATION 37: Fast coral observation likelihood
  // Normal distribution for percentage cover data
  for(int t = 0; t < n; t++) {
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_fast_use, true); // Normal likelihood
  }
  
  // EQUATION 38: Slow coral observation likelihood
  // Normal distribution for percentage cover data
  for(int t = 0; t < n; t++) {
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_slow_use, true); // Normal likelihood
  }
  
  // EQUATION 39: Soft parameter bounds using penalties
  Type penalty = Type(0.0);                                 // Initialize penalty
  
  // Allee strength bounded between 0 and 1
  if(allee_strength < Type(0.0)) penalty += Type(100.0) * pow(allee_strength, 2);
  if(allee_strength > Type(1.0)) penalty += Type(100.0) * pow(allee_strength - Type(1.0), 2);
  
  // EQUATION 40: Preference parameter bounded between 0 and 1
  if(preference_fast < Type(0.0)) penalty += Type(100.0) * pow(preference_fast, 2);
  if(preference_fast > Type(1.0)) penalty += Type(100.0) * pow(preference_fast - Type(1.0), 2);
  
  // EQUATION 41: Conversion efficiency bounded between 0 and 1
  if(conversion_eff < Type(0.0)) penalty += Type(100.0) * pow(conversion_eff, 2);
  if(conversion_eff > Type(1.0)) penalty += Type(100.0) * pow(conversion_eff - Type(1.0), 2);
  
  // EQUATION 42: Recruitment threshold bounded between 0 and 1
  if(recruit_threshold < Type(0.0)) penalty += Type(100.0) * pow(recruit_threshold, 2);
  if(recruit_threshold > Type(1.0)) penalty += Type(100.0) * pow(recruit_threshold - Type(1.0), 2);
  
  // EQUATION 43: Juvenile feeding fraction bounded between 0 and 1
  if(juvenile_feeding_fraction < Type(0.0)) penalty += Type(100.0) * pow(juvenile_feeding_fraction, 2);
  if(juvenile_feeding_fraction > Type(1.0)) penalty += Type(100.0) * pow(juvenile_feeding_fraction - Type(1.0), 2);
  
  // EQUATION 44: Maturation max multiplier should be >= 1
  if(maturation_max_multiplier < Type(1.0)) penalty += Type(100.0) * pow(maturation_max_multiplier - Type(1.0), 2);
  
  nll += penalty;                                           // Add penalties to likelihood
  
  // REPORTING
  REPORT(cots_juv_pred);                                    // Report predicted juvenile COTS
  REPORT(cots_adult_pred);                                  // Report predicted adult COTS
  REPORT(cots_pred);                                        // Report predicted total COTS
  REPORT(fast_pred);                                        // Report predicted fast coral
  REPORT(slow_pred);                                        // Report predicted slow coral
  REPORT(r_cots);                                           // Report adult COTS growth rate
  REPORT(K_cots);                                           // Report total COTS carrying capacity
  REPORT(allee_threshold);                                  // Report Allee threshold
  REPORT(allee_strength);                                   // Report Allee strength
  REPORT(mort_base);                                        // Report baseline adult mortality
  REPORT(mort_juvenile);                                    // Report baseline juvenile mortality
  REPORT(mort_density);                                     // Report density-dependent mortality
  REPORT(temp_opt);                                         // Report optimal temperature
  REPORT(temp_width);                                       // Report temperature width
  REPORT(immigration_effect);                               // Report immigration effect
  REPORT(recruit_max);                                      // Report maximum recruitment rate
  REPORT(recruit_threshold);                                // Report recruitment threshold
  REPORT(optimal_recruit_density);                          // Report optimal recruitment density
  REPORT(recruit_density_width);                            // Report recruitment density width
  REPORT(maturation_rate);                                  // Report baseline maturation rate
  REPORT(maturation_halfsat);                               // Report maturation half-saturation
  REPORT(maturation_max_multiplier);                        // Report maturation max multiplier
  REPORT(juvenile_feeding_fraction);                        // Report juvenile feeding fraction
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
