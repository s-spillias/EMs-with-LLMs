#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs - forcing variables
  DATA_VECTOR(Year);                    // Time vector (years)
  DATA_VECTOR(sst_dat);                 // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);             // COTS larval immigration rate (individuals/m²/year)
  
  // Data inputs - response variables
  DATA_VECTOR(cots_dat);                // Adult COTS abundance (individuals/m²)
  DATA_VECTOR(fast_dat);                // Fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);                // Slow-growing coral cover (%)
  
  // COTS population parameters
  PARAMETER(log_cots_recruit_base);     // Log baseline recruitment rate from accumulated potential (dimensionless)
  PARAMETER(log_recruitment_memory);    // Log time constant for recruitment potential dynamics (year⁻¹)
  PARAMETER(log_recruitment_threshold); // Log threshold of accumulated potential for recruitment (dimensionless)
  PARAMETER(log_temp_effect);           // Log temperature effect on larval survival (°C⁻¹)
  PARAMETER(temp_optimal);              // Optimal temperature for COTS recruitment (°C)
  PARAMETER(log_larval_survival_base);  // Log baseline larval survival multiplier (dimensionless)
  PARAMETER(log_nutrient_sensitivity);  // Log sensitivity to bloom-favorable conditions (°C⁻²)
  PARAMETER(sst_bloom_optimal);         // Optimal SST for phytoplankton blooms (°C)
  PARAMETER(log_nutrient_steepness);    // Log steepness of nutrient threshold response (dimensionless)
  PARAMETER(nutrient_threshold);        // Threshold for nutrient-enhanced survival (dimensionless)
  PARAMETER(log_cots_mort_juvenile);    // Log juvenile COTS mortality rate (year⁻¹)
  PARAMETER(log_cots_mort_adult);       // Log adult COTS mortality rate (year⁻¹)
  PARAMETER(log_maturity_rate);         // Log maturation rate from juvenile to adult (year⁻¹)
  PARAMETER(log_allee_threshold);       // Log Allee effect threshold density (individuals/m²)
  PARAMETER(log_allee_strength);        // Log strength of Allee effect (dimensionless)
  PARAMETER(log_density_mort);          // Log density-dependent mortality coefficient (m²/individuals/year)
  PARAMETER(log_food_limitation);       // Log food limitation coefficient (% cover⁻¹)
  
  // Coral predation parameters
  PARAMETER(log_attack_fast);           // Log baseline attack rate on fast-growing corals (m²/individuals/year)
  PARAMETER(log_attack_slow);           // Log baseline attack rate on slow-growing corals (m²/individuals/year)
  PARAMETER(log_preference_strength);   // Log strength of preference for fast corals when abundant (dimensionless)
  PARAMETER(preference_threshold);      // Fast coral cover threshold for half-maximal preference (% cover)
  PARAMETER(log_preference_steepness);  // Log steepness of prey-switching response (dimensionless)
  PARAMETER(log_handling_time);         // Log handling time for coral consumption (year)
  PARAMETER(log_conversion_eff);        // Log conversion efficiency from coral to COTS biomass (dimensionless)
  
  // Coral growth parameters
  PARAMETER(log_fast_growth);           // Log intrinsic growth rate of fast corals (year⁻¹)
  PARAMETER(log_slow_growth);           // Log intrinsic growth rate of slow corals (year⁻¹)
  PARAMETER(fast_carrying_cap);         // Carrying capacity for fast corals (% cover)
  PARAMETER(slow_carrying_cap);         // Carrying capacity for slow corals (% cover)
  
  // Observation error parameters
  PARAMETER(log_sigma_cots);            // Log standard deviation for COTS observations
  PARAMETER(log_sigma_fast);            // Log standard deviation for fast coral observations
  PARAMETER(log_sigma_slow);            // Log standard deviation for slow coral observations
  
  // Transform parameters to natural scale
  Type cots_recruit_base = exp(log_cots_recruit_base);           // Baseline recruitment scaling factor
  Type recruitment_memory = exp(log_recruitment_memory);         // Recruitment potential time constant
  Type recruitment_threshold = exp(log_recruitment_threshold);   // Recruitment potential threshold
  Type temp_effect = exp(log_temp_effect);                       // Temperature sensitivity parameter
  Type larval_survival_base = exp(log_larval_survival_base);     // Baseline larval survival multiplier
  Type nutrient_sensitivity = exp(log_nutrient_sensitivity);     // Sensitivity to bloom conditions
  Type nutrient_steepness = exp(log_nutrient_steepness);         // Steepness of threshold response
  Type cots_mort_juvenile = exp(log_cots_mort_juvenile);         // Juvenile mortality rate
  Type cots_mort_adult = exp(log_cots_mort_adult);               // Adult mortality rate
  Type maturity_rate = exp(log_maturity_rate);                   // Maturation rate
  Type allee_threshold = exp(log_allee_threshold);               // Density below which Allee effects occur
  Type allee_strength = exp(log_allee_strength);                 // Magnitude of Allee effect
  Type density_mort = exp(log_density_mort);                     // Density-dependent mortality coefficient
  Type food_limitation = exp(log_food_limitation);               // Food limitation strength
  Type attack_fast = exp(log_attack_fast);                       // Baseline attack rate on Acropora
  Type attack_slow = exp(log_attack_slow);                       // Baseline attack rate on massive corals
  Type preference_strength = exp(log_preference_strength);       // Preference strength for fast corals
  Type preference_steepness = exp(log_preference_steepness);     // Steepness of prey-switching
  Type handling_time = exp(log_handling_time);                   // Time spent handling prey
  Type conversion_eff = exp(log_conversion_eff);                 // Biomass conversion efficiency
  Type fast_growth = exp(log_fast_growth);                       // Acropora growth rate
  Type slow_growth = exp(log_slow_growth);                       // Massive coral growth rate
  Type sigma_cots = exp(log_sigma_cots);                         // COTS observation error
  Type sigma_fast = exp(log_sigma_fast);                         // Fast coral observation error
  Type sigma_slow = exp(log_sigma_slow);                         // Slow coral observation error
  
  // Add small constants for numerical stability
  Type eps = Type(1e-8);
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(0.01);
  Type sigma_cots_use = sigma_cots + min_sigma;
  Type sigma_fast_use = sigma_fast + min_sigma;
  Type sigma_slow_use = sigma_slow + min_sigma;
  
  // Get number of time steps
  int n = Year.size();
  
  // Initialize prediction vectors
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> maturity_pred(n);        // Population maturity state (0=juvenile, 1=adult)
  vector<Type> recruit_potential(n);    // Accumulated recruitment potential (dimensionless)
  
  // Set initial conditions from first data point
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  maturity_pred(0) = Type(0.5);         // Start with mixed age structure
  recruit_potential(0) = Type(0.1);     // Start with low recruitment potential
  
  // Initialize negative log-likelihood
  Type nll = Type(0.0);
  
  // Time loop - start from index 1 since initial conditions are set
  for(int t = 1; t < n; t++) {
    
    // Get previous time step values to avoid data leakage
    Type cots_prev = cots_pred(t-1);
    Type fast_prev = fast_pred(t-1);
    Type slow_prev = slow_pred(t-1);
    Type maturity_prev = maturity_pred(t-1);
    Type recruit_pot_prev = recruit_potential(t-1);
    
    // Ensure non-negative values with small floor using CppAD::CondExpGe
    cots_prev = CppAD::CondExpGe(cots_prev, eps, cots_prev, eps);
    fast_prev = CppAD::CondExpGe(fast_prev, eps, fast_prev, eps);
    slow_prev = CppAD::CondExpGe(slow_prev, eps, slow_prev, eps);
    recruit_pot_prev = CppAD::CondExpGe(recruit_pot_prev, eps, recruit_pot_prev, eps);
    
    // Bound maturity between 0 and 1
    maturity_prev = CppAD::CondExpGe(maturity_prev, Type(0.0), maturity_prev, Type(0.0));
    maturity_prev = CppAD::CondExpGe(Type(1.0), maturity_prev, maturity_prev, Type(1.0));
    
    // Total coral cover available as food
    Type total_coral = fast_prev + slow_prev + eps;
    
    // === EQUATION 1: Temperature-dependent larval survival ===
    // Gaussian temperature response centered on optimal temperature
    Type temp_deviation = sst_dat(t-1) - temp_optimal;
    Type temp_response = exp(-temp_effect * temp_deviation * temp_deviation);
    
    // === EQUATION 2: Nutrient-mediated larval survival ===
    // Environmental proxy for phytoplankton bloom conditions
    // Cooler temperatures during wet season often indicate runoff/nutrient pulses
    Type bloom_temp_deviation = sst_dat(t-1) - sst_bloom_optimal;
    Type nutrient_proxy = exp(-nutrient_sensitivity * bloom_temp_deviation * bloom_temp_deviation);
    
    // Sigmoidal response: larval survival amplified when nutrient proxy exceeds threshold
    // This creates the non-linear outbreak trigger mechanism
    Type larval_survival = larval_survival_base / (Type(1.0) + exp(-nutrient_steepness * (nutrient_proxy - nutrient_threshold)));
    
    // === EQUATION 3: Environmental favorability for recruitment ===
    // Combined environmental signal from temperature and nutrient conditions
    Type environmental_favorability = cotsimm_dat(t-1) * temp_response * larval_survival;
    
    // === EQUATION 4: Recruitment potential accumulation dynamics ===
    // Recruitment potential increases with favorable conditions, decays over time
    // This integrates environmental signals over ~2 years (larval development + early juvenile period)
    // Accumulation: favorable conditions add to potential
    // Decay: potential decays at rate recruitment_memory (representing larval/juvenile mortality)
    Type dpotential = recruitment_memory * (environmental_favorability - recruit_pot_prev);
    recruit_potential(t) = CppAD::CondExpGe(recruit_pot_prev + dpotential, eps, recruit_pot_prev + dpotential, eps);
    
    // === EQUATION 5: Threshold-based recruitment from accumulated potential ===
    // Recruitment only occurs when accumulated potential exceeds threshold
    // This creates multi-year lag and requires sustained favorable conditions
    Type potential_above_threshold = CppAD::CondExpGe(recruit_pot_prev, recruitment_threshold, 
                                                       recruit_pot_prev - recruitment_threshold, Type(0.0));
    Type recruitment = cots_recruit_base * potential_above_threshold;
    
    // === EQUATION 6: Age-structured mortality rate ===
    // Mortality interpolates between high juvenile rate and lower adult rate
    // based on population maturity state (0 = all juveniles, 1 = all adults)
    Type cots_mortality_base = cots_mort_juvenile + (cots_mort_adult - cots_mort_juvenile) * maturity_prev;
    
    // === EQUATION 7: Allee effect mortality ===
    // Increased mortality at low densities due to reduced fertilization success
    Type allee_effect = allee_strength * exp(-cots_prev / (allee_threshold + eps));
    
    // === EQUATION 8: Density-dependent mortality ===
    // Mortality increases with crowding
    Type density_effect = density_mort * cots_prev;
    
    // === EQUATION 9: Food limitation mortality ===
    // Mortality increases when coral food is depleted
    Type food_effect = food_limitation / (total_coral + eps);
    
    // === EQUATION 10: Total COTS mortality rate ===
    Type cots_mortality = cots_mortality_base + allee_effect + density_effect + food_effect;
    
    // === EQUATION 11: Dynamic prey preference modifier ===
    // Sigmoidal function: preference high when fast corals abundant, declines as they deplete
    // This creates prey-switching behavior from Acropora to massive corals
    Type preference_modifier = preference_strength / (Type(1.0) + exp(-preference_steepness * (fast_prev - preference_threshold)));
    
    // === EQUATION 12: Effective attack rates with prey-switching ===
    // When fast corals abundant (preference_modifier high):
    //   - Fast coral attack rate enhanced (COTS prefer Acropora)
    //   - Slow coral attack rate reduced (COTS avoid massive corals)
    // When fast corals depleted (preference_modifier low):
    //   - Attack rates approach baseline values (COTS switch to massive corals)
    Type attack_fast_effective = attack_fast * (Type(1.0) + preference_modifier);
    Type attack_slow_effective = attack_slow * (Type(1.0) - Type(0.5) * preference_modifier);
    
    // === EQUATION 13: Type II functional response for fast coral consumption ===
    // Per capita consumption rate with handling time limitation and dynamic preference
    Type consumption_fast = (attack_fast_effective * fast_prev) / 
                           (Type(1.0) + handling_time * (attack_fast_effective * fast_prev + attack_slow_effective * slow_prev) + eps);
    
    // === EQUATION 14: Type II functional response for slow coral consumption ===
    Type consumption_slow = (attack_slow_effective * slow_prev) / 
                           (Type(1.0) + handling_time * (attack_fast_effective * fast_prev + attack_slow_effective * slow_prev) + eps);
    
    // === EQUATION 15: Total consumption converted to COTS growth ===
    Type consumption_total = consumption_fast + consumption_slow;
    Type cots_growth_from_food = conversion_eff * consumption_total * cots_prev;
    
    // === EQUATION 16: COTS population dynamics ===
    // Change in COTS = recruitment + growth from feeding - mortality
    Type dcots = recruitment + cots_growth_from_food - cots_mortality * cots_prev;
    cots_pred(t) = CppAD::CondExpGe(cots_prev + dcots, eps, cots_prev + dcots, eps);
    
    // === EQUATION 17: Population maturity dynamics ===
    // Maturity increases through aging, decreases through recruitment dilution
    // Recruitment brings in juveniles (maturity = 0), diluting population maturity
    Type recruitment_dilution = Type(0.0);
    if(cots_prev > eps) {
      recruitment_dilution = recruitment / (cots_prev + eps);
    }
    
    // Aging increases maturity toward 1 at rate maturity_rate
    Type aging_effect = maturity_rate * (Type(1.0) - maturity_prev);
    
    // Recruitment dilutes maturity toward 0
    Type dilution_effect = recruitment_dilution * (Type(0.0) - maturity_prev);
    
    // Net change in maturity
    Type dmaturity = aging_effect + dilution_effect;
    maturity_pred(t) = maturity_prev + dmaturity;
    
    // Bound maturity between 0 and 1
    maturity_pred(t) = CppAD::CondExpGe(maturity_pred(t), Type(0.0), maturity_pred(t), Type(0.0));
    maturity_pred(t) = CppAD::CondExpGe(Type(1.0), maturity_pred(t), maturity_pred(t), Type(1.0));
    
    // === EQUATION 18: Fast coral predation loss ===
    // Total consumption by entire COTS population with dynamic preference
    Type fast_predation = consumption_fast * cots_prev;
    
    // === EQUATION 19: Fast coral logistic growth ===
    // Growth limited by space availability
    Type fast_logistic_growth = fast_growth * fast_prev * (Type(1.0) - fast_prev / (fast_carrying_cap + eps));
    
    // === EQUATION 20: Fast coral dynamics ===
    Type dfast = fast_logistic_growth - fast_predation;
    fast_pred(t) = CppAD::CondExpGe(fast_prev + dfast, eps, fast_prev + dfast, eps);
    
    // === EQUATION 21: Slow coral predation loss ===
    Type slow_predation = consumption_slow * cots_prev;
    
    // === EQUATION 22: Slow coral logistic growth ===
    Type slow_logistic_growth = slow_growth * slow_prev * (Type(1.0) - slow_prev / (slow_carrying_cap + eps));
    
    // === EQUATION 23: Slow coral dynamics ===
    Type dslow = slow_logistic_growth - slow_predation;
    slow_pred(t) = CppAD::CondExpGe(slow_prev + dslow, eps, slow_prev + dslow, eps);
  }
  
  // Calculate likelihood for all observations
  for(int t = 0; t < n; t++) {
    // Log-normal likelihood for COTS (strictly positive, can span orders of magnitude)
    Type log_cots_pred = log(cots_pred(t) + eps);
    Type log_cots_obs = log(cots_dat(t) + eps);
    nll -= dnorm(log_cots_obs, log_cots_pred, sigma_cots_use, true);
    
    // Normal likelihood for coral cover (percentage data, bounded)
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_fast_use, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_slow_use, true);
  }
  
  // Soft penalties to keep parameters in biologically reasonable ranges
  // These are gentle nudges, not hard constraints
  
  // Juvenile mortality should be higher than adult mortality
  Type mort_ordering_penalty = CppAD::CondExpGe(cots_mort_adult, cots_mort_juvenile, 
                                                 cots_mort_adult - cots_mort_juvenile, Type(0.0));
  nll += Type(0.1) * pow(mort_ordering_penalty, 2);
  
  // Juvenile mortality should be positive but not excessive (0.5 to 5.0 year⁻¹)
  Type juv_mort_upper_penalty = CppAD::CondExpGe(cots_mort_juvenile, Type(5.0), 
                                                  cots_mort_juvenile - Type(5.0), Type(0.0));
  Type juv_mort_lower_penalty = CppAD::CondExpGe(Type(0.5), cots_mort_juvenile, 
                                                  Type(0.5) - cots_mort_juvenile, Type(0.0));
  nll += Type(0.01) * pow(juv_mort_upper_penalty, 2);
  nll += Type(0.01) * pow(juv_mort_lower_penalty, 2);
  
  // Adult mortality should be lower (0.2 to 1.5 year⁻¹)
  Type adult_mort_upper_penalty = CppAD::CondExpGe(cots_mort_adult, Type(1.5), 
                                                    cots_mort_adult - Type(1.5), Type(0.0));
  Type adult_mort_lower_penalty = CppAD::CondExpGe(Type(0.2), cots_mort_adult, 
                                                    Type(0.2) - cots_mort_adult, Type(0.0));
  nll += Type(0.01) * pow(adult_mort_upper_penalty, 2);
  nll += Type(0.01) * pow(adult_mort_lower_penalty, 2);
  
  // Temperature optimum should be in tropical range (26-30°C)
  Type temp_upper_penalty = CppAD::CondExpGe(temp_optimal, Type(32.0), temp_optimal - Type(32.0), Type(0.0));
  Type temp_lower_penalty = CppAD::CondExpGe(Type(24.0), temp_optimal, Type(24.0) - temp_optimal, Type(0.0));
  nll += Type(0.01) * pow(temp_upper_penalty, 2);
  nll += Type(0.01) * pow(temp_lower_penalty, 2);
  
  // Bloom optimal temperature should be cooler than adult optimal (wet season conditions)
  Type bloom_temp_upper_penalty = CppAD::CondExpGe(sst_bloom_optimal, Type(29.0), sst_bloom_optimal - Type(29.0), Type(0.0));
  Type bloom_temp_lower_penalty = CppAD::CondExpGe(Type(23.0), sst_bloom_optimal, Type(23.0) - sst_bloom_optimal, Type(0.0));
  nll += Type(0.01) * pow(bloom_temp_upper_penalty, 2);
  nll += Type(0.01) * pow(bloom_temp_lower_penalty, 2);
  
  // Carrying capacities should be reasonable (10-80% cover)
  Type fast_cap_upper_penalty = CppAD::CondExpGe(fast_carrying_cap, Type(80.0), fast_carrying_cap - Type(80.0), Type(0.0));
  Type fast_cap_lower_penalty = CppAD::CondExpGe(Type(10.0), fast_carrying_cap, Type(10.0) - fast_carrying_cap, Type(0.0));
  Type slow_cap_upper_penalty = CppAD::CondExpGe(slow_carrying_cap, Type(80.0), slow_carrying_cap - Type(80.0), Type(0.0));
  Type slow_cap_lower_penalty = CppAD::CondExpGe(Type(10.0), slow_carrying_cap, Type(10.0) - slow_carrying_cap, Type(0.0));
  nll += Type(0.01) * pow(fast_cap_upper_penalty, 2);
  nll += Type(0.01) * pow(fast_cap_lower_penalty, 2);
  nll += Type(0.01) * pow(slow_cap_upper_penalty, 2);
  nll += Type(0.01) * pow(slow_cap_lower_penalty, 2);
  
  // Conversion efficiency should be less than 1 (can't create biomass from nothing)
  Type conv_penalty = CppAD::CondExpGe(conversion_eff, Type(1.0), conversion_eff - Type(1.0), Type(0.0));
  nll += Type(0.1) * pow(conv_penalty, 2);
  
  // Nutrient threshold should be between 0 and 1
  Type nutrient_thresh_upper_penalty = CppAD::CondExpGe(nutrient_threshold, Type(1.0), nutrient_threshold - Type(1.0), Type(0.0));
  Type nutrient_thresh_lower_penalty = CppAD::CondExpGe(Type(0.0), nutrient_threshold, Type(0.0) - nutrient_threshold, Type(0.0));
  nll += Type(0.01) * pow(nutrient_thresh_upper_penalty, 2);
  nll += Type(0.01) * pow(nutrient_thresh_lower_penalty, 2);
  
  // Preference threshold should be within reasonable coral cover range (5-30%)
  Type pref_thresh_upper_penalty = CppAD::CondExpGe(preference_threshold, Type(30.0), preference_threshold - Type(30.0), Type(0.0));
  Type pref_thresh_lower_penalty = CppAD::CondExpGe(Type(5.0), preference_threshold, Type(5.0) - preference_threshold, Type(0.0));
  nll += Type(0.01) * pow(pref_thresh_upper_penalty, 2);
  nll += Type(0.01) * pow(pref_thresh_lower_penalty, 2);
  
  // Recruitment memory should be positive (0.1 to 10 year⁻¹)
  Type memory_upper_penalty = CppAD::CondExpGe(recruitment_memory, Type(10.0), recruitment_memory - Type(10.0), Type(0.0));
  Type memory_lower_penalty = CppAD::CondExpGe(Type(0.1), recruitment_memory, Type(0.1) - recruitment_memory, Type(0.0));
  nll += Type(0.01) * pow(memory_upper_penalty, 2);
  nll += Type(0.01) * pow(memory_lower_penalty, 2);
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(maturity_pred);
  REPORT(recruit_potential);
  
  // Report transformed parameters for interpretation
  REPORT(cots_recruit_base);
  REPORT(recruitment_memory);
  REPORT(recruitment_threshold);
  REPORT(temp_effect);
  REPORT(temp_optimal);
  REPORT(larval_survival_base);
  REPORT(nutrient_sensitivity);
  REPORT(sst_bloom_optimal);
  REPORT(nutrient_steepness);
  REPORT(nutrient_threshold);
  REPORT(cots_mort_juvenile);
  REPORT(cots_mort_adult);
  REPORT(maturity_rate);
  REPORT(allee_threshold);
  REPORT(allee_strength);
  REPORT(density_mort);
  REPORT(food_limitation);
  REPORT(attack_fast);
  REPORT(attack_slow);
  REPORT(preference_strength);
  REPORT(preference_threshold);
  REPORT(preference_steepness);
  REPORT(handling_time);
  REPORT(conversion_eff);
  REPORT(fast_growth);
  REPORT(slow_growth);
  REPORT(fast_carrying_cap);
  REPORT(slow_carrying_cap);
  REPORT(sigma_cots);
  REPORT(sigma_fast);
  REPORT(sigma_slow);
  
  return nll;
}
