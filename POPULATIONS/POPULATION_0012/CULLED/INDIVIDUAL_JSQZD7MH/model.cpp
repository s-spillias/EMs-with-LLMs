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
  PARAMETER(log_cots_mort_juvenile);    // Log juvenile COTS mortality rate (year⁻¹)
  PARAMETER(log_cots_mort_adult);       // Log adult COTS mortality rate (year⁻¹)
  PARAMETER(log_maturity_rate);         // Log maturation rate from juvenile to adult (year⁻¹)
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
  PARAMETER(fast_carrying_cap);         // Maximum carrying capacity for fast corals (% cover)
  PARAMETER(slow_carrying_cap);         // Maximum carrying capacity for slow corals (% cover)
  
  // Coral competition parameters
  PARAMETER(total_benthic_space);       // Total available benthic space (% cover)
  PARAMETER(competition_strength);      // Strength of interspecific competition (0-1)
  
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
  Type cots_mort_juvenile = exp(log_cots_mort_juvenile);         // Juvenile mortality rate
  Type cots_mort_adult = exp(log_cots_mort_adult);               // Adult mortality rate
  Type maturity_rate = exp(log_maturity_rate);                   // Maturation rate
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
    
    // === EQUATION 3: Total recruitment with nutrient amplification ===
    // Recruitment from larval immigration with temperature AND nutrient-mediated survival
    Type recruitment = cots_recruit_base * cotsimm_dat(t-1) * temp_response * larval_survival;
    
    // === EQUATION 4: Age-structured mortality rate ===
    // Mortality interpolates between high juvenile rate and lower adult rate
    // based on population maturity state (0 = all juveniles, 1 = all adults)
    Type cots_mortality_base = cots_mort_juvenile + (cots_mort_adult - cots_mort_juvenile) * maturity_prev;
    
    // === EQUATION 5: Allee effect mortality ===
    // Increased mortality at low densities due to reduced fertilization success
    Type allee_effect = allee_strength * exp(-cots_prev / (allee_threshold + eps));
    
    // === EQUATION 6: Density-dependent mortality ===
    // Mortality increases with crowding
    Type density_effect = density_mort * cots_prev;
    
    // === EQUATION 7: Food limitation mortality ===
    // Mortality increases when coral food is depleted
    Type food_effect = food_limitation / (total_coral + eps);
    
    // === EQUATION 8: Total COTS mortality rate ===
    Type cots_mortality = cots_mortality_base + allee_effect + density_effect + food_effect;
    
    // === EQUATION 9: Type II functional response for fast coral consumption ===
    // Per capita consumption rate with handling time limitation
    Type consumption_fast = (attack_fast * fast_prev) / (Type(1.0) + handling_time * (attack_fast * fast_prev + attack_slow * slow_prev) + eps);
    
    // === EQUATION 10: Type II functional response for slow coral consumption ===
    Type consumption_slow = (attack_slow * slow_prev) / (Type(1.0) + handling_time * (attack_fast * fast_prev + attack_slow * slow_prev) + eps);
    
    // === EQUATION 11: Total consumption converted to COTS growth ===
    Type consumption_total = consumption_fast + consumption_slow;
    Type cots_growth_from_food = conversion_eff * consumption_total * cots_prev;
    
    // === EQUATION 12: COTS population dynamics ===
    // Change in COTS = recruitment + growth from feeding - mortality
    Type dcots = recruitment + cots_growth_from_food - cots_mortality * cots_prev;
    cots_pred(t) = CppAD::CondExpGe(cots_prev + dcots, eps, cots_prev + dcots, eps);
    
    // === EQUATION 13: Population maturity dynamics ===
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
    
    // === EQUATION 14: Dynamic effective carrying capacity for fast corals ===
    // Carrying capacity reduced by space occupied by slow corals
    // Competition strength modulates how strongly slow corals inhibit fast coral growth
    Type space_occupied_by_slow = competition_strength * slow_prev / (total_benthic_space + eps);
    Type fast_k_effective = fast_carrying_cap * (Type(1.0) - space_occupied_by_slow);
    // Ensure effective K is positive
    fast_k_effective = CppAD::CondExpGe(fast_k_effective, eps, fast_k_effective, eps);
    
    // === EQUATION 15: Fast coral predation loss ===
    // Total consumption by entire COTS population
    Type fast_predation = consumption_fast * cots_prev;
    
    // === EQUATION 16: Fast coral logistic growth with dynamic carrying capacity ===
    // Growth limited by effective space availability (accounting for competition)
    Type fast_logistic_growth = fast_growth * fast_prev * (Type(1.0) - fast_prev / (fast_k_effective + eps));
    
    // === EQUATION 17: Fast coral dynamics ===
    Type dfast = fast_logistic_growth - fast_predation;
    fast_pred(t) = CppAD::CondExpGe(fast_prev + dfast, eps, fast_prev + dfast, eps);
    
    // === EQUATION 18: Dynamic effective carrying capacity for slow corals ===
    // Carrying capacity reduced by space occupied by fast corals
    Type space_occupied_by_fast = competition_strength * fast_prev / (total_benthic_space + eps);
    Type slow_k_effective = slow_carrying_cap * (Type(1.0) - space_occupied_by_fast);
    // Ensure effective K is positive
    slow_k_effective = CppAD::CondExpGe(slow_k_effective, eps, slow_k_effective, eps);
    
    // === EQUATION 19: Slow coral predation loss ===
    Type slow_predation = consumption_slow * cots_prev;
    
    // === EQUATION 20: Slow coral logistic growth with dynamic carrying capacity ===
    // Growth limited by effective space availability (accounting for competition)
    Type slow_logistic_growth = slow_growth * slow_prev * (Type(1.0) - slow_prev / (slow_k_effective + eps));
    
    // === EQUATION 21: Slow coral dynamics ===
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
  
  // Maximum carrying capacities should be reasonable (10-80% cover)
  Type fast_cap_upper_penalty = CppAD::CondExpGe(fast_carrying_cap, Type(80.0), fast_carrying_cap - Type(80.0), Type(0.0));
  Type fast_cap_lower_penalty = CppAD::CondExpGe(Type(10.0), fast_carrying_cap, Type(10.0) - fast_carrying_cap, Type(0.0));
  Type slow_cap_upper_penalty = CppAD::CondExpGe(slow_carrying_cap, Type(80.0), slow_carrying_cap - Type(80.0), Type(0.0));
  Type slow_cap_lower_penalty = CppAD::CondExpGe(Type(10.0), slow_carrying_cap, Type(10.0) - slow_carrying_cap, Type(0.0));
  nll += Type(0.01) * pow(fast_cap_upper_penalty, 2);
  nll += Type(0.01) * pow(fast_cap_lower_penalty, 2);
  nll += Type(0.01) * pow(slow_cap_upper_penalty, 2);
  nll += Type(0.01) * pow(slow_cap_lower_penalty, 2);
  
  // Total benthic space should be reasonable (70-95% cover)
  Type total_space_upper_penalty = CppAD::CondExpGe(total_benthic_space, Type(95.0), total_benthic_space - Type(95.0), Type(0.0));
  Type total_space_lower_penalty = CppAD::CondExpGe(Type(70.0), total_benthic_space, Type(70.0) - total_benthic_space, Type(0.0));
  nll += Type(0.01) * pow(total_space_upper_penalty, 2);
  nll += Type(0.01) * pow(total_space_lower_penalty, 2);
  
  // Competition strength should be between 0 and 1
  Type comp_upper_penalty = CppAD::CondExpGe(competition_strength, Type(1.0), competition_strength - Type(1.0), Type(0.0));
  Type comp_lower_penalty = CppAD::CondExpGe(Type(0.0), competition_strength, Type(0.0) - competition_strength, Type(0.0));
  nll += Type(0.01) * pow(comp_upper_penalty, 2);
  nll += Type(0.01) * pow(comp_lower_penalty, 2);
  
  // Conversion efficiency should be less than 1 (can't create biomass from nothing)
  Type conv_penalty = CppAD::CondExpGe(conversion_eff, Type(1.0), conversion_eff - Type(1.0), Type(0.0));
  nll += Type(0.1) * pow(conv_penalty, 2);
  
  // Nutrient threshold should be between 0 and 1
  Type nutrient_thresh_upper_penalty = CppAD::CondExpGe(nutrient_threshold, Type(1.0), nutrient_threshold - Type(1.0), Type(0.0));
  Type nutrient_thresh_lower_penalty = CppAD::CondExpGe(Type(0.0), nutrient_threshold, Type(0.0) - nutrient_threshold, Type(0.0));
  nll += Type(0.01) * pow(nutrient_thresh_upper_penalty, 2);
  nll += Type(0.01) * pow(nutrient_thresh_lower_penalty, 2);
  
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
  REPORT(cots_mort_juvenile);
  REPORT(cots_mort_adult);
  REPORT(maturity_rate);
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
  REPORT(total_benthic_space);
  REPORT(competition_strength);
  REPORT(sigma_cots);
  REPORT(sigma_fast);
  REPORT(sigma_slow);
  
  return nll;
}
