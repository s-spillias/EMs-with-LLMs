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
  PARAMETER(log_cots_recruit_base);     // Log baseline recruitment rate from immigration (dimensionless)
  PARAMETER(log_temp_effect);           // Log temperature effect on larval survival (°C⁻¹)
  PARAMETER(temp_optimal);              // Optimal temperature for COTS recruitment (°C)
  PARAMETER(log_larval_survival_base);  // Log baseline larval survival multiplier (dimensionless)
  PARAMETER(log_nutrient_sensitivity);  // Log sensitivity to bloom-favorable conditions (°C⁻²)
  PARAMETER(sst_bloom_optimal);         // Optimal SST for phytoplankton blooms (°C)
  PARAMETER(log_nutrient_steepness);    // Log steepness of nutrient threshold response (dimensionless)
  PARAMETER(nutrient_threshold);        // Threshold for nutrient-enhanced survival (dimensionless)
  PARAMETER(log_local_fecundity);       // Log local spawning contribution parameter ((%cover)⁻¹)
  PARAMETER(log_cots_mort_juvenile);    // Log juvenile COTS mortality rate (year⁻¹)
  PARAMETER(log_cots_mort_adult);       // Log adult COTS mortality rate (year⁻¹)
  PARAMETER(log_maturity_rate_base);    // Log baseline maturation rate at reference temperature (year⁻¹)
  PARAMETER(log_Q10_maturation);        // Log Q10 temperature coefficient for maturation
  PARAMETER(temp_ref_maturation);       // Reference temperature for baseline maturation rate (°C)
  PARAMETER(log_allee_threshold);       // Log Allee effect threshold density (individuals/m²)
  PARAMETER(log_allee_strength);        // Log strength of Allee effect (dimensionless)
  PARAMETER(log_density_mort);          // Log density-dependent mortality coefficient (m²/individuals/year)
  PARAMETER(log_food_limitation);       // Log food limitation coefficient (% cover⁻¹)
  
  // Coral predation parameters
  PARAMETER(log_attack_fast);           // Log attack rate on fast-growing corals (m²/individuals/year)
  PARAMETER(log_attack_slow);           // Log attack rate on slow-growing corals (m²/individuals/year)
  PARAMETER(log_handling_time);         // Log handling time for coral consumption (year)
  PARAMETER(log_conversion_eff);        // Log conversion efficiency from coral to COTS biomass (dimensionless)
  
  // Coral growth parameters
  PARAMETER(log_fast_growth);           // Log intrinsic growth rate of fast corals (year⁻¹)
  PARAMETER(log_slow_growth);           // Log intrinsic growth rate of slow corals (year⁻¹)
  PARAMETER(fast_carrying_cap);         // Carrying capacity for fast corals (% cover)
  PARAMETER(slow_carrying_cap);         // Carrying capacity for slow corals (% cover)
  
  // Coral thermal performance parameters
  PARAMETER(temp_optimal_fast);         // Optimal temperature for fast coral growth (°C)
  PARAMETER(temp_optimal_slow);         // Optimal temperature for slow coral growth (°C)
  PARAMETER(log_temp_sensitivity_fast); // Log thermal sensitivity for fast coral growth (°C⁻²)
  PARAMETER(log_temp_sensitivity_slow); // Log thermal sensitivity for slow coral growth (°C⁻²)
  
  // Coral bleaching mortality parameters
  PARAMETER(bleach_threshold_fast);     // Temperature threshold for fast coral bleaching mortality (°C)
  PARAMETER(bleach_threshold_slow);     // Temperature threshold for slow coral bleaching mortality (°C)
  PARAMETER(log_bleach_sensitivity_fast); // Log baseline bleaching mortality rate for fast corals (year⁻¹)
  PARAMETER(log_bleach_sensitivity_slow); // Log baseline bleaching mortality rate for slow corals (year⁻¹)
  PARAMETER(log_bleach_steepness);      // Log steepness of exponential thermal stress response (°C⁻¹)
  
  // Observation error parameters
  PARAMETER(log_sigma_cots);            // Log standard deviation for COTS observations
  PARAMETER(log_sigma_fast);            // Log standard deviation for fast coral observations
  PARAMETER(log_sigma_slow);            // Log standard deviation for slow coral observations
  
  // Transform parameters to natural scale
  Type cots_recruit_base = exp(log_cots_recruit_base);           // Baseline recruitment scaling factor
  Type temp_effect = exp(log_temp_effect);                       // Temperature sensitivity parameter
  Type larval_survival_base = exp(log_larval_survival_base);     // Baseline larval survival multiplier
  Type nutrient_sensitivity = exp(log_nutrient_sensitivity);     // Sensitivity to bloom conditions
  Type nutrient_steepness = exp(log_nutrient_steepness);         // Steepness of threshold response
  Type local_fecundity = exp(log_local_fecundity);               // Local spawning contribution
  Type cots_mort_juvenile = exp(log_cots_mort_juvenile);         // Juvenile mortality rate
  Type cots_mort_adult = exp(log_cots_mort_adult);               // Adult mortality rate
  Type maturity_rate_base = exp(log_maturity_rate_base);         // Baseline maturation rate at reference temperature
  Type Q10_maturation = exp(log_Q10_maturation);                 // Q10 temperature coefficient for maturation
  Type allee_threshold = exp(log_allee_threshold);               // Density below which Allee effects occur
  Type allee_strength = exp(log_allee_strength);                 // Magnitude of Allee effect
  Type density_mort = exp(log_density_mort);                     // Density-dependent mortality coefficient
  Type food_limitation = exp(log_food_limitation);               // Food limitation strength
  Type attack_fast = exp(log_attack_fast);                       // Attack rate on Acropora
  Type attack_slow = exp(log_attack_slow);                       // Attack rate on massive corals
  Type handling_time = exp(log_handling_time);                   // Time spent handling prey
  Type conversion_eff = exp(log_conversion_eff);                 // Biomass conversion efficiency
  Type fast_growth = exp(log_fast_growth);                       // Acropora growth rate
  Type slow_growth = exp(log_slow_growth);                       // Massive coral growth rate
  Type temp_sensitivity_fast = exp(log_temp_sensitivity_fast);   // Thermal sensitivity for fast corals
  Type temp_sensitivity_slow = exp(log_temp_sensitivity_slow);   // Thermal sensitivity for slow corals
  Type bleach_sensitivity_fast = exp(log_bleach_sensitivity_fast); // Baseline bleaching mortality for fast corals
  Type bleach_sensitivity_slow = exp(log_bleach_sensitivity_slow); // Baseline bleaching mortality for slow corals
  Type bleach_steepness = exp(log_bleach_steepness);             // Exponential scaling of thermal stress
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
  
  // Set initial conditions from first data point
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  maturity_pred(0) = Type(0.5);         // Start with mixed age structure
  
  // Initialize negative log-likelihood
  Type nll = Type(0.0);
  
  // Time loop - start from index 1 since initial conditions are set
  for(int t = 1; t < n; t++) {
    
    // Get previous time step values to avoid data leakage
    Type cots_prev = cots_pred(t-1);
    Type fast_prev = fast_pred(t-1);
    Type slow_prev = slow_pred(t-1);
    Type maturity_prev = maturity_pred(t-1);
    
    // Ensure non-negative values with small floor using CppAD::CondExpGe
    cots_prev = CppAD::CondExpGe(cots_prev, eps, cots_prev, eps);
    fast_prev = CppAD::CondExpGe(fast_prev, eps, fast_prev, eps);
    slow_prev = CppAD::CondExpGe(slow_prev, eps, slow_prev, eps);
    
    // Bound maturity between 0 and 1
    maturity_prev = CppAD::CondExpGe(maturity_prev, Type(0.0), maturity_prev, Type(0.0));
    maturity_prev = CppAD::CondExpGe(Type(1.0), maturity_prev, maturity_prev, Type(1.0));
    
    // Total coral cover available as food
    Type total_coral = fast_prev + slow_prev + eps;
    
    // Average carrying capacity for coral food availability scaling
    Type avg_carrying_cap = (fast_carrying_cap + slow_carrying_cap) / Type(2.0);
    
    // === EQUATION 1: Temperature-dependent recruitment ===
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
    
    // === EQUATION 3a: Immigration-based recruitment ===
    // Recruitment from external larval immigration with temperature AND nutrient-mediated survival
    Type recruitment_immigration = cots_recruit_base * cotsimm_dat(t-1) * temp_response * larval_survival;
    
    // === EQUATION 3b: Local spawning contribution ===
    // Age-dependent fecundity: mature adults produce vastly more eggs
    // Maturity squared creates non-linear scaling (older populations disproportionately fecund)
    // Food availability affects reproductive output (well-fed adults more fecund)
    Type food_availability = total_coral / (avg_carrying_cap + eps);
    Type recruitment_local = local_fecundity * cots_prev * maturity_prev * maturity_prev * food_availability;
    
    // === EQUATION 3c: Total recruitment with local spawning feedback ===
    // This positive feedback mechanism sustains outbreaks once mature population established
    Type recruitment = recruitment_immigration + recruitment_local;
    
    // === EQUATION 4: Temperature-dependent maturation rate ===
    // Q10 formulation: rate increases exponentially with temperature
    // maturation_rate = maturity_rate_base * Q10^((T - T_ref)/10)
    // This captures the ectotherm metabolic response to temperature
    Type temp_deviation_maturation = sst_dat(t-1) - temp_ref_maturation;
    Type maturity_rate = maturity_rate_base * pow(Q10_maturation, temp_deviation_maturation / Type(10.0));
    
    // === EQUATION 5: Age-structured mortality rate ===
    // Mortality interpolates between high juvenile rate and lower adult rate
    // based on population maturity state (0 = all juveniles, 1 = all adults)
    Type cots_mortality_base = cots_mort_juvenile + (cots_mort_adult - cots_mort_juvenile) * maturity_prev;
    
    // === EQUATION 6: Allee effect mortality ===
    // Increased mortality at low densities due to reduced fertilization success
    Type allee_effect = allee_strength * exp(-cots_prev / (allee_threshold + eps));
    
    // === EQUATION 7: Density-dependent mortality ===
    // Mortality increases with crowding
    Type density_effect = density_mort * cots_prev;
    
    // === EQUATION 8: Food limitation mortality ===
    // Mortality increases when coral food is depleted
    Type food_effect = food_limitation / (total_coral + eps);
    
    // === EQUATION 9: Total COTS mortality rate ===
    Type cots_mortality = cots_mortality_base + allee_effect + density_effect + food_effect;
    
    // === EQUATION 10: Type II functional response for fast coral consumption ===
    // Per capita consumption rate with handling time limitation
    Type consumption_fast = (attack_fast * fast_prev) / (Type(1.0) + handling_time * (attack_fast * fast_prev + attack_slow * slow_prev) + eps);
    
    // === EQUATION 11: Type II functional response for slow coral consumption ===
    Type consumption_slow = (attack_slow * slow_prev) / (Type(1.0) + handling_time * (attack_fast * fast_prev + attack_slow * slow_prev) + eps);
    
    // === EQUATION 12: Total consumption converted to COTS growth ===
    Type consumption_total = consumption_fast + consumption_slow;
    Type cots_growth_from_food = conversion_eff * consumption_total * cots_prev;
    
    // === EQUATION 13: COTS population dynamics ===
    // Change in COTS = recruitment (immigration + local spawning) + growth from feeding - mortality
    Type dcots = recruitment + cots_growth_from_food - cots_mortality * cots_prev;
    cots_pred(t) = CppAD::CondExpGe(cots_prev + dcots, eps, cots_prev + dcots, eps);
    
    // === EQUATION 14: Population maturity dynamics with temperature-dependent maturation ===
    // Maturity increases through aging (now temperature-dependent), decreases through recruitment dilution
    // Recruitment brings in juveniles (maturity = 0), diluting population maturity
    Type recruitment_dilution = Type(0.0);
    if(cots_prev > eps) {
      recruitment_dilution = recruitment / (cots_prev + eps);
    }
    
    // Aging increases maturity toward 1 at temperature-dependent rate
    // Warmer temperatures → faster maturation → more rapid transition to highly fecund adults
    Type aging_effect = maturity_rate * (Type(1.0) - maturity_prev);
    
    // Recruitment dilutes maturity toward 0
    Type dilution_effect = recruitment_dilution * (Type(0.0) - maturity_prev);
    
    // Net change in maturity
    Type dmaturity = aging_effect + dilution_effect;
    maturity_pred(t) = maturity_prev + dmaturity;
    
    // Bound maturity between 0 and 1
    maturity_pred(t) = CppAD::CondExpGe(maturity_pred(t), Type(0.0), maturity_pred(t), Type(0.0));
    maturity_pred(t) = CppAD::CondExpGe(Type(1.0), maturity_pred(t), maturity_pred(t), Type(1.0));
    
    // === EQUATION 15: Fast coral predation loss ===
    // Total consumption by entire COTS population
    Type fast_predation = consumption_fast * cots_prev;
    
    // === EQUATION 16: Temperature-dependent fast coral growth ===
    // Thermal performance curve: Gaussian response centered on optimal temperature
    Type temp_deviation_fast = sst_dat(t-1) - temp_optimal_fast;
    Type temp_response_fast = exp(-temp_sensitivity_fast * temp_deviation_fast * temp_deviation_fast);
    
    // Growth limited by space availability AND temperature
    Type fast_logistic_growth = fast_growth * temp_response_fast * fast_prev * (Type(1.0) - fast_prev / (fast_carrying_cap + eps));
    
    // === EQUATION 17: Bleaching mortality for fast corals ===
    // Exponential mortality when SST exceeds bleaching threshold
    // Only activates above threshold using conditional expression
    Type thermal_anomaly_fast = sst_dat(t-1) - bleach_threshold_fast;
    Type fast_bleaching_mort = CppAD::CondExpGe(thermal_anomaly_fast, Type(0.0),
                                                 bleach_sensitivity_fast * exp(bleach_steepness * thermal_anomaly_fast) * fast_prev,
                                                 Type(0.0));
    
    // === EQUATION 18: Fast coral dynamics with bleaching ===
    Type dfast = fast_logistic_growth - fast_predation - fast_bleaching_mort;
    fast_pred(t) = CppAD::CondExpGe(fast_prev + dfast, eps, fast_prev + dfast, eps);
    
    // === EQUATION 19: Slow coral predation loss ===
    Type slow_predation = consumption_slow * cots_prev;
    
    // === EQUATION 20: Temperature-dependent slow coral growth ===
    // Thermal performance curve: Gaussian response centered on optimal temperature
    Type temp_deviation_slow = sst_dat(t-1) - temp_optimal_slow;
    Type temp_response_slow = exp(-temp_sensitivity_slow * temp_deviation_slow * temp_deviation_slow);
    
    // Growth limited by space availability AND temperature
    Type slow_logistic_growth = slow_growth * temp_response_slow * slow_prev * (Type(1.0) - slow_prev / (slow_carrying_cap + eps));
    
    // === EQUATION 21: Bleaching mortality for slow corals ===
    // Exponential mortality when SST exceeds bleaching threshold
    // Higher threshold than fast corals (greater thermal tolerance)
    Type thermal_anomaly_slow = sst_dat(t-1) - bleach_threshold_slow;
    Type slow_bleaching_mort = CppAD::CondExpGe(thermal_anomaly_slow, Type(0.0),
                                                 bleach_sensitivity_slow * exp(bleach_steepness * thermal_anomaly_slow) * slow_prev,
                                                 Type(0.0));
    
    // === EQUATION 22: Slow coral dynamics with bleaching ===
    Type dslow = slow_logistic_growth - slow_predation - slow_bleaching_mort;
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
  
  // Reference temperature for maturation should be in reasonable range (25-29°C)
  Type temp_ref_upper_penalty = CppAD::CondExpGe(temp_ref_maturation, Type(29.0), temp_ref_maturation - Type(29.0), Type(0.0));
  Type temp_ref_lower_penalty = CppAD::CondExpGe(Type(25.0), temp_ref_maturation, Type(25.0) - temp_ref_maturation, Type(0.0));
  nll += Type(0.01) * pow(temp_ref_upper_penalty, 2);
  nll += Type(0.01) * pow(temp_ref_lower_penalty, 2);
  
  // Q10 should be in biologically reasonable range (1.5 to 4.0)
  Type Q10_upper_penalty = CppAD::CondExpGe(Q10_maturation, Type(4.0), Q10_maturation - Type(4.0), Type(0.0));
  Type Q10_lower_penalty = CppAD::CondExpGe(Type(1.5), Q10_maturation, Type(1.5) - Q10_maturation, Type(0.0));
  nll += Type(0.01) * pow(Q10_upper_penalty, 2);
  nll += Type(0.01) * pow(Q10_lower_penalty, 2);
  
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
  
  // Local fecundity should be positive but not create unrealistic explosions
  Type fecund_upper_penalty = CppAD::CondExpGe(local_fecundity, Type(10.0), local_fecundity - Type(10.0), Type(0.0));
  nll += Type(0.01) * pow(fecund_upper_penalty, 2);
  
  // Coral thermal optima should be in reasonable range (25-29°C)
  Type temp_opt_fast_upper_penalty = CppAD::CondExpGe(temp_optimal_fast, Type(29.0), temp_optimal_fast - Type(29.0), Type(0.0));
  Type temp_opt_fast_lower_penalty = CppAD::CondExpGe(Type(26.0), temp_optimal_fast, Type(26.0) - temp_optimal_fast, Type(0.0));
  Type temp_opt_slow_upper_penalty = CppAD::CondExpGe(temp_optimal_slow, Type(28.0), temp_optimal_slow - Type(28.0), Type(0.0));
  Type temp_opt_slow_lower_penalty = CppAD::CondExpGe(Type(25.0), temp_optimal_slow, Type(25.0) - temp_optimal_slow, Type(0.0));
  nll += Type(0.01) * pow(temp_opt_fast_upper_penalty, 2);
  nll += Type(0.01) * pow(temp_opt_fast_lower_penalty, 2);
  nll += Type(0.01) * pow(temp_opt_slow_upper_penalty, 2);
  nll += Type(0.01) * pow(temp_opt_slow_lower_penalty, 2);
  
  // Fast corals should have higher thermal optimum than slow corals (ecological constraint)
  Type thermal_ordering_penalty = CppAD::CondExpGe(temp_optimal_slow, temp_optimal_fast, 
                                                    temp_optimal_slow - temp_optimal_fast, Type(0.0));
  nll += Type(0.01) * pow(thermal_ordering_penalty, 2);
  
  // Bleaching thresholds should be above growth optima (bleaching occurs at elevated temperatures)
  Type bleach_fast_ordering_penalty = CppAD::CondExpGe(temp_optimal_fast, bleach_threshold_fast,
                                                        temp_optimal_fast - bleach_threshold_fast, Type(0.0));
  Type bleach_slow_ordering_penalty = CppAD::CondExpGe(temp_optimal_slow, bleach_threshold_slow,
                                                        temp_optimal_slow - bleach_threshold_slow, Type(0.0));
  nll += Type(0.01) * pow(bleach_fast_ordering_penalty, 2);
  nll += Type(0.01) * pow(bleach_slow_ordering_penalty, 2);
  
  // Slow corals should have higher bleaching threshold than fast corals (greater thermal tolerance)
  Type bleach_threshold_ordering_penalty = CppAD::CondExpGe(bleach_threshold_fast, bleach_threshold_slow,
                                                             bleach_threshold_fast - bleach_threshold_slow, Type(0.0));
  nll += Type(0.01) * pow(bleach_threshold_ordering_penalty, 2);
  
  // Bleaching thresholds should be in realistic range (29-32°C for GBR)
  Type bleach_fast_upper_penalty = CppAD::CondExpGe(bleach_threshold_fast, Type(31.0), bleach_threshold_fast - Type(31.0), Type(0.0));
  Type bleach_fast_lower_penalty = CppAD::CondExpGe(Type(29.0), bleach_threshold_fast, Type(29.0) - bleach_threshold_fast, Type(0.0));
  Type bleach_slow_upper_penalty = CppAD::CondExpGe(bleach_threshold_slow, Type(32.0), bleach_threshold_slow - Type(32.0), Type(0.0));
  Type bleach_slow_lower_penalty = CppAD::CondExpGe(Type(30.0), bleach_threshold_slow, Type(30.0) - bleach_threshold_slow, Type(0.0));
  nll += Type(0.01) * pow(bleach_fast_upper_penalty, 2);
  nll += Type(0.01) * pow(bleach_fast_lower_penalty, 2);
  nll += Type(0.01) * pow(bleach_slow_upper_penalty, 2);
  nll += Type(0.01) * pow(bleach_slow_lower_penalty, 2);
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(maturity_pred);
  
  // Report transformed parameters for interpretation
  REPORT(cots_recruit_base);
  REPORT(temp_effect);
  REPORT(temp_optimal);
  REPORT(larval_survival_base);
  REPORT(nutrient_sensitivity);
  REPORT(sst_bloom_optimal);
  REPORT(nutrient_steepness);
  REPORT(nutrient_threshold);
  REPORT(local_fecundity);
  REPORT(cots_mort_juvenile);
  REPORT(cots_mort_adult);
  REPORT(maturity_rate_base);
  REPORT(Q10_maturation);
  REPORT(temp_ref_maturation);
  REPORT(allee_threshold);
  REPORT(allee_strength);
  REPORT(density_mort);
  REPORT(food_limitation);
  REPORT(attack_fast);
  REPORT(attack_slow);
  REPORT(handling_time);
  REPORT(conversion_eff);
  REPORT(fast_growth);
  REPORT(slow_growth);
  REPORT(fast_carrying_cap);
  REPORT(slow_carrying_cap);
  REPORT(temp_optimal_fast);
  REPORT(temp_optimal_slow);
  REPORT(temp_sensitivity_fast);
  REPORT(temp_sensitivity_slow);
  REPORT(bleach_threshold_fast);
  REPORT(bleach_threshold_slow);
  REPORT(bleach_sensitivity_fast);
  REPORT(bleach_sensitivity_slow);
  REPORT(bleach_steepness);
  REPORT(sigma_cots);
  REPORT(sigma_fast);
  REPORT(sigma_slow);
  
  return nll;
}
