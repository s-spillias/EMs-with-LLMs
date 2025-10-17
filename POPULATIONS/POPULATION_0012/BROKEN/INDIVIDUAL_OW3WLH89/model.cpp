#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS - Time series observations
  DATA_VECTOR(Year);                    // Time vector (years)
  DATA_VECTOR(sst_dat);                 // Sea surface temperature observations (°C)
  DATA_VECTOR(cotsimm_dat);             // COTS larval immigration rate (larvae/m²/year)
  DATA_VECTOR(cots_dat);                // Adult COTS abundance observations (individuals/m²)
  DATA_VECTOR(fast_dat);                // Fast-growing coral cover observations (%)
  DATA_VECTOR(slow_dat);                // Slow-growing coral cover observations (%)
  
  // LARVAL STAGE PARAMETERS (NEW - for episodic outbreak dynamics)
  PARAMETER(log_fecundity);             // Log effective fecundity (millions of eggs per adult)
  PARAMETER(log_larval_survival_base);  // Log baseline larval survival rate
  PARAMETER(log_nutrient_effect_strength); // Log nutrient pulse effect on larval survival
  PARAMETER(log_nutrient_threshold);    // Log SST anomaly threshold for nutrient effect
  PARAMETER(log_settlement_rate);       // Log larval settlement rate
  PARAMETER(log_larval_mortality);      // Log larval mortality rate
  
  // COTS POPULATION PARAMETERS
  PARAMETER(log_cots_recruit_base);     // Log baseline COTS recruitment rate (year⁻¹)
  PARAMETER(log_cots_mort_base);        // Log baseline COTS natural mortality rate (year⁻¹)
  PARAMETER(log_allee_threshold);       // Log COTS density for Allee effect threshold (individuals/m²)
  PARAMETER(log_allee_strength);        // Log strength of Allee effect (dimensionless)
  PARAMETER(log_temp_recruit_opt);      // Log optimal temperature for COTS recruitment (°C)
  PARAMETER(log_temp_recruit_width);    // Log temperature tolerance width for recruitment (°C)
  PARAMETER(log_density_mort_rate);     // Log density-dependent mortality coefficient (m²/individuals/year)
  PARAMETER(log_immigration_effect);    // Log immigration contribution to larval pool (dimensionless)
  
  // CORAL GROWTH PARAMETERS
  PARAMETER(log_fast_growth_rate);      // Log fast coral intrinsic growth rate (year⁻¹)
  PARAMETER(log_slow_growth_rate);      // Log slow coral intrinsic growth rate (year⁻¹)
  PARAMETER(log_fast_carrying_cap);     // Log fast coral carrying capacity (%)
  PARAMETER(log_slow_carrying_cap);     // Log slow coral carrying capacity (%)
  PARAMETER(log_coral_competition);     // Log interspecific competition coefficient (dimensionless)
  PARAMETER(log_temp_stress_threshold); // Log temperature threshold for coral stress (°C)
  PARAMETER(log_temp_stress_rate);      // Log coral mortality rate per degree above threshold (year⁻¹/°C)
  
  // PREDATION PARAMETERS
  PARAMETER(log_attack_rate_fast);      // Log COTS attack rate on fast coral (m²/individuals/year)
  PARAMETER(log_attack_rate_slow);      // Log COTS attack rate on slow coral (m²/individuals/year)
  PARAMETER(log_handling_time_fast);    // Log handling time for fast coral (years)
  PARAMETER(log_handling_time_slow);    // Log handling time for slow coral (years)
  PARAMETER(log_preference_switch);     // Log prey switching threshold (% cover)
  PARAMETER(log_conversion_efficiency); // Log conversion of coral to COTS biomass (dimensionless)
  
  // OBSERVATION ERROR PARAMETERS
  PARAMETER(log_sigma_cots);            // Log observation error SD for COTS (individuals/m²)
  PARAMETER(log_sigma_fast);            // Log observation error SD for fast coral (%)
  PARAMETER(log_sigma_slow);            // Log observation error SD for slow coral (%)
  
  // INITIAL CONDITION PARAMETERS (NEW - to avoid data leakage)
  PARAMETER(log_cots_init);             // Log initial adult COTS density
  PARAMETER(log_fast_init);             // Log initial fast coral cover
  PARAMETER(log_slow_init);             // Log initial slow coral cover
  
  // Transform larval parameters from log scale
  Type fecundity = exp(log_fecundity);
  Type larval_survival_base = exp(log_larval_survival_base);
  Type nutrient_effect_strength = exp(log_nutrient_effect_strength);
  Type nutrient_threshold = exp(log_nutrient_threshold);
  Type settlement_rate = exp(log_settlement_rate);
  Type larval_mortality = exp(log_larval_mortality);
  
  // Transform COTS parameters from log scale
  Type cots_recruit_base = exp(log_cots_recruit_base);
  Type cots_mort_base = exp(log_cots_mort_base);
  Type allee_threshold = exp(log_allee_threshold);
  Type allee_strength = exp(log_allee_strength);
  Type temp_recruit_opt = exp(log_temp_recruit_opt);
  Type temp_recruit_width = exp(log_temp_recruit_width);
  Type density_mort_rate = exp(log_density_mort_rate);
  Type immigration_effect = exp(log_immigration_effect);
  
  // Transform coral parameters from log scale
  Type fast_growth_rate = exp(log_fast_growth_rate);
  Type slow_growth_rate = exp(log_slow_growth_rate);
  Type fast_carrying_cap = exp(log_fast_carrying_cap);
  Type slow_carrying_cap = exp(log_slow_carrying_cap);
  Type coral_competition = exp(log_coral_competition);
  Type temp_stress_threshold = exp(log_temp_stress_threshold);
  Type temp_stress_rate = exp(log_temp_stress_rate);
  
  // Transform predation parameters from log scale
  Type attack_rate_fast = exp(log_attack_rate_fast);
  Type attack_rate_slow = exp(log_attack_rate_slow);
  Type handling_time_fast = exp(log_handling_time_fast);
  Type handling_time_slow = exp(log_handling_time_slow);
  Type preference_switch = exp(log_preference_switch);
  Type conversion_efficiency = exp(log_conversion_efficiency);
  
  // Transform observation error parameters from log scale
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_fast = exp(log_sigma_fast);
  Type sigma_slow = exp(log_sigma_slow);
  
  // Transform initial condition parameters from log scale
  Type cots_init = exp(log_cots_init);
  Type fast_init = exp(log_fast_init);
  Type slow_init = exp(log_slow_init);
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(0.01);
  sigma_cots = sigma_cots + min_sigma;
  sigma_fast = sigma_fast + min_sigma;
  sigma_slow = sigma_slow + min_sigma;
  
  // Initialize prediction vectors
  int n = Year.size();
  vector<Type> larvae_pred(n);          // NEW: Larval pool density
  vector<Type> cots_pred(n);            // Adult COTS density
  vector<Type> fast_pred(n);            // Fast coral cover
  vector<Type> slow_pred(n);            // Slow coral cover
  
  // Calculate mean SST for anomaly calculation
  Type mean_sst = Type(0.0);
  for(int t = 0; t < n; t++) {
    mean_sst += sst_dat(t);
  }
  mean_sst = mean_sst / Type(n);
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Initialize negative log-likelihood
  Type nll = Type(0.0);
  
  // TIME LOOP - Dynamic model equations with larval stage
  for(int t = 0; t < n; t++) {
    
    // Declare previous time step values
    Type larvae_prev;
    Type cots_prev;
    Type fast_prev;
    Type slow_prev;
    Type sst_prev;
    Type immigration_prev;
    
    // Set previous values based on time step
    if(t == 0) {
      // For t=0, use initial conditions
      larvae_prev = Type(0.01);
      cots_prev = cots_init;
      fast_prev = fast_init;
      slow_prev = slow_init;
      sst_prev = sst_dat(0);
      immigration_prev = cotsimm_dat(0);
    } else {
      // For t>0, use previous predictions
      larvae_prev = larvae_pred(t-1);
      cots_prev = cots_pred(t-1);
      fast_prev = fast_pred(t-1);
      slow_prev = slow_pred(t-1);
      sst_prev = sst_dat(t-1);
      immigration_prev = cotsimm_dat(t-1);
    }
    
    // ========== LARVAL STAGE DYNAMICS ==========
    
    // EQUATION 1: Allee effect on adult spawning success (sigmoid function)
    Type allee_effect = pow(cots_prev, allee_strength) / (pow(allee_threshold, allee_strength) + pow(cots_prev, allee_strength) + eps);
    
    // EQUATION 2: Larval production from adult spawning
    Type larval_production = fecundity * cots_prev * allee_effect;
    
    // EQUATION 3: SST anomaly for nutrient proxy (negative anomalies = upwelling/mixing)
    Type sst_anomaly = sst_prev - mean_sst;
    
    // EQUATION 4: Nutrient effect on larval survival (threshold response for episodic dynamics)
    // Negative SST anomalies (cooler = nutrient-rich) boost survival
    Type nutrient_effect = Type(1.0) + nutrient_effect_strength / 
                          (Type(1.0) + exp(Type(5.0) * (sst_anomaly + nutrient_threshold)));
    
    // EQUATION 5: Temperature effect on larval survival (permissive window)
    Type temp_diff = sst_prev - temp_recruit_opt;
    Type temp_effect = exp(-0.5 * pow(temp_diff / (temp_recruit_width + eps), 2));
    
    // EQUATION 6: Total larval survival rate (baseline × nutrient boost × temperature window)
    Type larval_survival = larval_survival_base * nutrient_effect * temp_effect;
    
    // EQUATION 7: Immigration adds to larval pool (connectivity between reefs)
    Type immigration_contribution = immigration_effect * immigration_prev;
    
    // EQUATION 8: Larval pool dynamics
    Type larval_gain = larval_production * larval_survival + immigration_contribution;
    Type larval_loss = larval_mortality * larvae_prev + settlement_rate * larvae_prev;
    Type larvae_change = larval_gain - larval_loss;
    larvae_pred(t) = larvae_prev + larvae_change;
    // Prevent negative larvae
    Type larvae_min = Type(0.0001);
    larvae_pred(t) = CppAD::CondExpGt(larvae_pred(t), larvae_min, larvae_pred(t), larvae_min);
    
    // ========== ADULT COTS DYNAMICS ==========
    
    // EQUATION 9: Recruitment from settling larvae to adults
    Type cots_recruitment = cots_recruit_base * settlement_rate * larvae_prev;
    
    // EQUATION 10: Density-dependent COTS mortality
    Type cots_mortality = cots_mort_base + density_mort_rate * cots_prev;
    
    // EQUATION 11: Type II functional response for fast coral predation with preference
    Type fast_available = fast_prev + eps;
    Type preference_fast = Type(1.0) / (Type(1.0) + exp(-Type(5.0) * (fast_prev - preference_switch)));
    Type consumption_fast = (attack_rate_fast * preference_fast * cots_prev * fast_available) / 
                           (Type(1.0) + attack_rate_fast * handling_time_fast * fast_available + eps);
    
    // EQUATION 12: Type II functional response for slow coral predation with switching
    Type slow_available = slow_prev + eps;
    Type preference_slow = Type(1.0) - preference_fast;
    Type consumption_slow = (attack_rate_slow * preference_slow * cots_prev * slow_available) / 
                           (Type(1.0) + attack_rate_slow * handling_time_slow * slow_available + eps);
    
    // EQUATION 13: Total coral consumption and conversion to COTS biomass
    Type total_consumption = consumption_fast + consumption_slow;
    Type cots_gain_from_feeding = conversion_efficiency * total_consumption;
    
    // EQUATION 14: COTS population change
    Type cots_change = cots_recruitment + cots_gain_from_feeding - cots_mortality * cots_prev;
    cots_pred(t) = cots_prev + cots_change;
    // Prevent extinction using smooth lower bound
    Type cots_min = Type(0.001);
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), cots_min, cots_pred(t), cots_min);
    
    // ========== CORAL DYNAMICS ==========
    
    // EQUATION 15: Temperature stress on corals (smooth transition)
    Type temp_excess = sst_prev - temp_stress_threshold;
    Type temp_stress = temp_stress_rate * temp_excess / (Type(1.0) + exp(-Type(10.0) * temp_excess));
    
    // EQUATION 16: Fast coral logistic growth with competition and predation
    Type fast_growth = fast_growth_rate * fast_prev * 
                      (Type(1.0) - (fast_prev + coral_competition * slow_prev) / (fast_carrying_cap + eps));
    Type fast_loss = consumption_fast + temp_stress * fast_prev;
    Type fast_change = fast_growth - fast_loss;
    fast_pred(t) = fast_prev + fast_change;
    // Apply bounds using smooth conditional expressions
    Type fast_min = Type(0.01);
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), fast_min, fast_pred(t), fast_min);
    fast_pred(t) = CppAD::CondExpLt(fast_pred(t), fast_carrying_cap, fast_pred(t), fast_carrying_cap);
    
    // EQUATION 17: Slow coral logistic growth with competition and predation
    Type slow_growth = slow_growth_rate * slow_prev * 
                      (Type(1.0) - (slow_prev + coral_competition * fast_prev) / (slow_carrying_cap + eps));
    Type slow_loss = consumption_slow + temp_stress * slow_prev;
    Type slow_change = slow_growth - slow_loss;
    slow_pred(t) = slow_prev + slow_change;
    // Apply bounds using smooth conditional expressions
    Type slow_min = Type(0.01);
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), slow_min, slow_pred(t), slow_min);
    slow_pred(t) = CppAD::CondExpLt(slow_pred(t), slow_carrying_cap, slow_pred(t), slow_carrying_cap);
  }
  
  // LIKELIHOOD CALCULATION - Compare predictions to observations
  for(int t = 0; t < n; t++) {
    // Lognormal likelihood for COTS (strictly positive, spans orders of magnitude)
    Type log_cots_pred = log(cots_pred(t) + eps);
    Type log_cots_obs = log(cots_dat(t) + eps);
    nll -= dnorm(log_cots_obs, log_cots_pred, sigma_cots, true);
    
    // Normal likelihood for coral cover (percentage data)
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_fast, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_slow, true);
  }
  
  // SOFT CONSTRAINTS - Biological realism penalties
  // Penalize extreme parameter values with smooth quadratic penalties
  
  // Larval survival should be very low (0.0001 to 0.01)
  Type larval_survival_penalty = Type(0.0);
  larval_survival_penalty += CppAD::CondExpLt(larval_survival_base, Type(0.0001), 
                                              Type(10.0) * pow(larval_survival_base - Type(0.0001), 2), Type(0.0));
  larval_survival_penalty += CppAD::CondExpGt(larval_survival_base, Type(0.01), 
                                              Type(10.0) * pow(larval_survival_base - Type(0.01), 2), Type(0.0));
  nll += larval_survival_penalty;
  
  // Nutrient effect should create episodic dynamics (5-100x multiplier)
  Type nutrient_strength_penalty = Type(0.0);
  nutrient_strength_penalty += CppAD::CondExpLt(nutrient_effect_strength, Type(2.0), 
                                                Type(10.0) * pow(nutrient_effect_strength - Type(2.0), 2), Type(0.0));
  nutrient_strength_penalty += CppAD::CondExpGt(nutrient_effect_strength, Type(100.0), 
                                                Type(10.0) * pow(nutrient_effect_strength - Type(100.0), 2), Type(0.0));
  nll += nutrient_strength_penalty;
  
  // Fecundity should be realistic (10-65 million eggs)
  Type fecundity_penalty = Type(0.0);
  fecundity_penalty += CppAD::CondExpLt(fecundity, Type(10.0), 
                                        Type(10.0) * pow(fecundity - Type(10.0), 2), Type(0.0));
  fecundity_penalty += CppAD::CondExpGt(fecundity, Type(200.0), 
                                        Type(10.0) * pow(fecundity - Type(200.0), 2), Type(0.0));
  nll += fecundity_penalty;
  
  // COTS recruitment should be moderate (0.01 to 2.0 year⁻¹)
  Type cots_recruit_penalty = Type(0.0);
  cots_recruit_penalty += CppAD::CondExpLt(cots_recruit_base, Type(0.01), 
                                           Type(10.0) * pow(cots_recruit_base - Type(0.01), 2), Type(0.0));
  cots_recruit_penalty += CppAD::CondExpGt(cots_recruit_base, Type(2.0), 
                                           Type(10.0) * pow(cots_recruit_base - Type(2.0), 2), Type(0.0));
  nll += cots_recruit_penalty;
  
  // COTS mortality should be moderate (0.1 to 1.5 year⁻¹)
  Type cots_mort_penalty = Type(0.0);
  cots_mort_penalty += CppAD::CondExpLt(cots_mort_base, Type(0.1), 
                                        Type(10.0) * pow(cots_mort_base - Type(0.1), 2), Type(0.0));
  cots_mort_penalty += CppAD::CondExpGt(cots_mort_base, Type(1.5), 
                                        Type(10.0) * pow(cots_mort_base - Type(1.5), 2), Type(0.0));
  nll += cots_mort_penalty;
  
  // Coral growth rates should be realistic (fast: 0.05-0.5, slow: 0.01-0.2 year⁻¹)
  Type fast_growth_penalty = Type(0.0);
  fast_growth_penalty += CppAD::CondExpLt(fast_growth_rate, Type(0.05), 
                                          Type(10.0) * pow(fast_growth_rate - Type(0.05), 2), Type(0.0));
  fast_growth_penalty += CppAD::CondExpGt(fast_growth_rate, Type(0.5), 
                                          Type(10.0) * pow(fast_growth_rate - Type(0.5), 2), Type(0.0));
  nll += fast_growth_penalty;
  
  Type slow_growth_penalty = Type(0.0);
  slow_growth_penalty += CppAD::CondExpLt(slow_growth_rate, Type(0.01), 
                                          Type(10.0) * pow(slow_growth_rate - Type(0.01), 2), Type(0.0));
  slow_growth_penalty += CppAD::CondExpGt(slow_growth_rate, Type(0.2), 
                                          Type(10.0) * pow(slow_growth_rate - Type(0.2), 2), Type(0.0));
  nll += slow_growth_penalty;
  
  // Carrying capacities should be reasonable (10-80% cover)
  Type fast_cap_penalty = Type(0.0);
  fast_cap_penalty += CppAD::CondExpLt(fast_carrying_cap, Type(10.0), 
                                       Type(10.0) * pow(fast_carrying_cap - Type(10.0), 2), Type(0.0));
  fast_cap_penalty += CppAD::CondExpGt(fast_carrying_cap, Type(80.0), 
                                       Type(10.0) * pow(fast_carrying_cap - Type(80.0), 2), Type(0.0));
  nll += fast_cap_penalty;
  
  Type slow_cap_penalty = Type(0.0);
  slow_cap_penalty += CppAD::CondExpLt(slow_carrying_cap, Type(10.0), 
                                       Type(10.0) * pow(slow_carrying_cap - Type(10.0), 2), Type(0.0));
  slow_cap_penalty += CppAD::CondExpGt(slow_carrying_cap, Type(80.0), 
                                       Type(10.0) * pow(slow_carrying_cap - Type(80.0), 2), Type(0.0));
  nll += slow_cap_penalty;
  
  // Temperature optimum should be in tropical range (26-30°C)
  Type temp_opt_penalty = Type(0.0);
  temp_opt_penalty += CppAD::CondExpLt(temp_recruit_opt, Type(26.0), 
                                       Type(10.0) * pow(temp_recruit_opt - Type(26.0), 2), Type(0.0));
  temp_opt_penalty += CppAD::CondExpGt(temp_recruit_opt, Type(30.0), 
                                       Type(10.0) * pow(temp_recruit_opt - Type(30.0), 2), Type(0.0));
  nll += temp_opt_penalty;
  
  // Initial conditions should be reasonable
  Type cots_init_penalty = Type(0.0);
  cots_init_penalty += CppAD::CondExpLt(cots_init, Type(0.001), 
                                        Type(10.0) * pow(cots_init - Type(0.001), 2), Type(0.0));
  cots_init_penalty += CppAD::CondExpGt(cots_init, Type(10.0), 
                                        Type(10.0) * pow(cots_init - Type(10.0), 2), Type(0.0));
  nll += cots_init_penalty;
  
  Type fast_init_penalty = Type(0.0);
  fast_init_penalty += CppAD::CondExpLt(fast_init, Type(1.0), 
                                        Type(10.0) * pow(fast_init - Type(1.0), 2), Type(0.0));
  fast_init_penalty += CppAD::CondExpGt(fast_init, Type(80.0), 
                                        Type(10.0) * pow(fast_init - Type(80.0), 2), Type(0.0));
  nll += fast_init_penalty;
  
  Type slow_init_penalty = Type(0.0);
  slow_init_penalty += CppAD::CondExpLt(slow_init, Type(1.0), 
                                        Type(10.0) * pow(slow_init - Type(1.0), 2), Type(0.0));
  slow_init_penalty += CppAD::CondExpGt(slow_init, Type(80.0), 
                                        Type(10.0) * pow(slow_init - Type(80.0), 2), Type(0.0));
  nll += slow_init_penalty;
  
  // REPORTING - Output predictions and parameters
  REPORT(larvae_pred);
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  REPORT(fecundity);
  REPORT(larval_survival_base);
  REPORT(nutrient_effect_strength);
  REPORT(nutrient_threshold);
  REPORT(settlement_rate);
  REPORT(larval_mortality);
  
  REPORT(cots_recruit_base);
  REPORT(cots_mort_base);
  REPORT(allee_threshold);
  REPORT(allee_strength);
  REPORT(temp_recruit_opt);
  REPORT(temp_recruit_width);
  REPORT(density_mort_rate);
  REPORT(immigration_effect);
  
  REPORT(fast_growth_rate);
  REPORT(slow_growth_rate);
  REPORT(fast_carrying_cap);
  REPORT(slow_carrying_cap);
  REPORT(coral_competition);
  REPORT(temp_stress_threshold);
  REPORT(temp_stress_rate);
  
  REPORT(attack_rate_fast);
  REPORT(attack_rate_slow);
  REPORT(handling_time_fast);
  REPORT(handling_time_slow);
  REPORT(preference_switch);
  REPORT(conversion_efficiency);
  
  REPORT(sigma_cots);
  REPORT(sigma_fast);
  REPORT(sigma_slow);
  
  REPORT(cots_init);
  REPORT(fast_init);
  REPORT(slow_init);
  
  return nll;
}
