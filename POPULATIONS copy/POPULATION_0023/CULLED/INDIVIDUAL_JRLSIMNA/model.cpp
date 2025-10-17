#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS
  DATA_VECTOR(Year);                    // Time vector (years)
  DATA_VECTOR(cots_dat);               // Observed total COTS abundance (juveniles + adults, individuals/m2)
  DATA_VECTOR(fast_dat);               // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);               // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);                // Sea surface temperature forcing (Celsius)
  DATA_VECTOR(cotsimm_dat);            // COTS larval immigration forcing (individuals/m2/year)
  
  // COTS POPULATION PARAMETERS - AGE STRUCTURED WITH LARVAL POOL
  PARAMETER(log_r_cots);                // Log intrinsic reproductive rate of adult COTS (year^-1)
  PARAMETER(log_K_cots_base);           // Log baseline carrying capacity of adult COTS (individuals/m2)
  PARAMETER(log_m_cots);                // Log natural mortality rate of adult COTS (year^-1)
  PARAMETER(log_m_cots_juvenile);       // Log natural mortality rate of juvenile COTS (year^-1)
  PARAMETER(log_m_larvae);              // Log natural mortality rate of planktonic larvae (year^-1)
  PARAMETER(log_maturation_rate);       // Log rate of maturation from juvenile to adult (year^-1)
  PARAMETER(juvenile_feeding_efficiency); // Relative feeding efficiency of juveniles vs adults (dimensionless, 0-1)
  PARAMETER(log_allee_threshold);       // Log Allee threshold density for adults (individuals/m2)
  PARAMETER(allee_strength);            // Allee effect strength (dimensionless, 0-1)
  PARAMETER(log_dd_mortality);          // Log density-dependent mortality coefficient for adults (m2/individuals/year)
  
  // LARVAL SETTLEMENT PARAMETERS
  PARAMETER(log_settlement_efficiency);      // Log baseline settlement rate (year^-1)
  PARAMETER(settlement_habitat_dependence);  // Strength of coral substrate requirement (dimensionless, 0-1)
  PARAMETER(log_settlement_saturation);      // Log half-saturation for density-dependent settlement (larvae/m2)
  
  // TEMPERATURE EFFECTS ON COTS
  PARAMETER(temp_opt);                  // Optimal temperature for COTS recruitment (Celsius)
  PARAMETER(log_temp_width);            // Log temperature tolerance width (Celsius)
  PARAMETER(log_temp_effect_max);       // Log maximum temperature effect multiplier (dimensionless)
  
  // LARVAL SURVIVAL PARAMETERS
  PARAMETER(log_larval_survival_efficiency);  // Log efficiency of larval survival under optimal conditions (dimensionless)
  PARAMETER(log_nutrient_half_sat);     // Log half-saturation constant for nutrient effect (nutrient units)
  PARAMETER(log_nutrient_outbreak_threshold);  // Log nutrient threshold for outbreak enhancement (nutrient units)
  PARAMETER(nutrient_outbreak_multiplier);     // Additional boost to larval survival during high nutrients (dimensionless)
  PARAMETER(nutrient_response_steepness);      // Steepness of sigmoid response at threshold (dimensionless)
  
  // CORAL DYNAMICS PARAMETERS
  PARAMETER(log_r_fast);                // Log growth rate of fast-growing coral (year^-1)
  PARAMETER(log_r_slow);                // Log growth rate of slow-growing coral (year^-1)
  PARAMETER(log_K_coral);               // Log carrying capacity for total coral cover (%)
  PARAMETER(competition_strength);      // Competition coefficient between coral types (dimensionless)
  
  // COTS FEEDING PARAMETERS
  PARAMETER(log_attack_fast);           // Log attack rate on fast-growing coral (m2/individuals/year)
  PARAMETER(log_attack_slow);           // Log attack rate on slow-growing coral (m2/individuals/year)
  PARAMETER(log_handling_time);         // Log handling time for coral consumption (year)
  PARAMETER(feeding_preference);        // Preference for fast vs slow coral (dimensionless, >1 prefers fast)
  PARAMETER(log_coral_to_cots);         // Log conversion efficiency of coral to COTS biomass (dimensionless)
  
  // OBSERVATION ERROR PARAMETERS
  PARAMETER(log_sigma_cots);            // Log observation error SD for total COTS (individuals/m2)
  PARAMETER(log_sigma_fast);            // Log observation error SD for fast coral (%)
  PARAMETER(log_sigma_slow);            // Log observation error SD for slow coral (%)
  
  // Transform parameters from log scale
  Type r_cots = exp(log_r_cots);
  Type K_cots_base = exp(log_K_cots_base);
  Type m_cots = exp(log_m_cots);
  Type m_cots_juvenile = exp(log_m_cots_juvenile);
  Type m_larvae = exp(log_m_larvae);
  Type maturation_rate = exp(log_maturation_rate);
  Type allee_threshold = exp(log_allee_threshold);
  Type dd_mortality = exp(log_dd_mortality);
  Type settlement_efficiency = exp(log_settlement_efficiency);
  Type settlement_saturation = exp(log_settlement_saturation);
  Type temp_width = exp(log_temp_width);
  Type temp_effect_max = exp(log_temp_effect_max);
  Type larval_survival_efficiency = exp(log_larval_survival_efficiency);
  Type nutrient_half_sat = exp(log_nutrient_half_sat);
  Type nutrient_outbreak_threshold = exp(log_nutrient_outbreak_threshold);
  Type r_fast = exp(log_r_fast);
  Type r_slow = exp(log_r_slow);
  Type K_coral = exp(log_K_coral);
  Type attack_fast = exp(log_attack_fast);
  Type attack_slow = exp(log_attack_slow);
  Type handling_time = exp(log_handling_time);
  Type coral_to_cots = exp(log_coral_to_cots);
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_fast = exp(log_sigma_fast);
  Type sigma_slow = exp(log_sigma_slow);
  
  // Add minimum sigma values for numerical stability
  Type min_sigma = Type(0.01);
  sigma_cots = sigma_cots + min_sigma;
  sigma_fast = sigma_fast + min_sigma;
  sigma_slow = sigma_slow + min_sigma;
  
  // Initialize prediction vectors - now with explicit larval pool
  int n = Year.size();
  vector<Type> larvae_pred(n);    // Planktonic larval pool
  vector<Type> juvenile_pred(n);  // Juvenile COTS (age 0-2 years)
  vector<Type> adult_pred(n);     // Adult COTS (age 2+ years)
  vector<Type> cots_pred(n);      // Total COTS (for comparison with data)
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Set initial conditions from data
  // Assume initial COTS are split 70% adult, 30% juvenile (reasonable for non-outbreak conditions)
  // Initialize larval pool at low baseline level
  larvae_pred(0) = Type(0.01);  // Small initial larval pool
  juvenile_pred(0) = cots_dat(0) * Type(0.3);
  adult_pred(0) = cots_dat(0) * Type(0.7);
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Small constant for numerical stability
  Type eps = Type(1e-8);
  
  // TIME LOOP: Simulate dynamics forward in time
  for(int t = 0; t < n-1; t++) {
    
    // Current state (using only previous time step values)
    Type larvae_curr = larvae_pred(t);
    Type juvenile_curr = juvenile_pred(t);
    Type adult_curr = adult_pred(t);
    Type fast_curr = fast_pred(t);
    Type slow_curr = slow_pred(t);
    Type sst_curr = sst_dat(t);
    Type immigration_curr = cotsimm_dat(t);
    
    // Default nutrient to baseline value (1.0) if not provided
    Type nutrient_curr = Type(1.0);
    
    // Ensure non-negative values using CppAD::CondExpGt
    larvae_curr = CppAD::CondExpGt(larvae_curr, eps, larvae_curr, eps);
    juvenile_curr = CppAD::CondExpGt(juvenile_curr, eps, juvenile_curr, eps);
    adult_curr = CppAD::CondExpGt(adult_curr, eps, adult_curr, eps);
    fast_curr = CppAD::CondExpGt(fast_curr, eps, fast_curr, eps);
    slow_curr = CppAD::CondExpGt(slow_curr, eps, slow_curr, eps);
    nutrient_curr = CppAD::CondExpGt(nutrient_curr, eps, nutrient_curr, eps);
    
    // EQUATION 1: Temperature effect on COTS recruitment
    // Gaussian function centered at optimal temperature
    Type temp_deviation = sst_curr - temp_opt;
    Type temp_effect = Type(1.0) + (temp_effect_max - Type(1.0)) * exp(-0.5 * pow(temp_deviation / (temp_width + eps), 2));
    
    // EQUATION 2: Allee effect on ADULT COTS reproduction
    // Reduces recruitment at low adult densities (mate-finding limitation)
    Type allee_effect = Type(1.0) - allee_strength * exp(-adult_curr / (allee_threshold + eps));
    allee_effect = CppAD::CondExpGt(allee_effect, Type(0.01), allee_effect, Type(0.01));
    
    // EQUATION 2b: Nutrient effect on larval survival
    // Two-component response: baseline Michaelis-Menten + threshold outbreak boost
    Type nutrient_baseline = nutrient_curr / (nutrient_curr + nutrient_half_sat + eps);
    Type nutrient_deviation = nutrient_curr - nutrient_outbreak_threshold;
    Type outbreak_boost = nutrient_outbreak_multiplier / 
                         (Type(1.0) + exp(-nutrient_response_steepness * nutrient_deviation));
    Type nutrient_effect = nutrient_baseline * (Type(1.0) + outbreak_boost);
    
    // EQUATION 3: Type II functional response for COTS feeding on fast-growing coral
    // Both juveniles and adults feed, but juveniles are less efficient
    Type effective_attack_fast_adult = attack_fast * feeding_preference;
    Type effective_attack_fast_juvenile = effective_attack_fast_adult * juvenile_feeding_efficiency;
    
    Type total_effective_attack_fast = effective_attack_fast_adult * adult_curr + 
                                       effective_attack_fast_juvenile * juvenile_curr;
    Type total_effective_attack_slow = attack_slow * adult_curr + 
                                       attack_slow * juvenile_feeding_efficiency * juvenile_curr;
    
    Type consumption_fast = (total_effective_attack_fast * fast_curr) / 
                           (Type(1.0) + handling_time * (total_effective_attack_fast * fast_curr + 
                            total_effective_attack_slow * slow_curr) + eps);
    
    // EQUATION 4: Type II functional response for COTS feeding on slow-growing coral
    Type consumption_slow = (total_effective_attack_slow * slow_curr) / 
                           (Type(1.0) + handling_time * (total_effective_attack_fast * fast_curr + 
                            total_effective_attack_slow * slow_curr) + eps);
    
    // EQUATION 5: Total coral available for COTS
    Type total_coral = fast_curr + slow_curr + eps;
    
    // EQUATION 6: Dynamic carrying capacity for adult COTS based on coral availability
    Type K_cots = K_cots_base * (total_coral / (K_coral + eps));
    
    // EQUATION 7: Density-dependent mortality of adult COTS
    Type density_mortality = dd_mortality * adult_curr;
    
    // EQUATION 8: ADULT COTS spawning (produces larvae, not juveniles directly)
    // Adults spawn with Allee effect and temperature modulation
    Type adult_spawning = r_cots * adult_curr * (Type(1.0) - adult_curr / (K_cots + eps)) * allee_effect * temp_effect;
    
    // EQUATION 9: LARVAL POOL dynamics
    // Larvae accumulate from local spawning and immigration, experience high mortality, and settle to juveniles
    
    // Larval input from external immigration (modified by survival efficiency and nutrients)
    Type larval_immigration = immigration_curr * larval_survival_efficiency * nutrient_effect;
    
    // Habitat availability for settlement (depends on coral cover)
    Type habitat_availability = Type(1.0) - settlement_habitat_dependence + 
                               settlement_habitat_dependence * (total_coral / (K_coral + eps));
    
    // Settlement rate with density-dependence (Type II saturation)
    // Higher larval densities lead to competition for settlement space
    Type settlement_rate = settlement_efficiency * 
                          (larvae_curr / (larvae_curr + settlement_saturation + eps)) * 
                          habitat_availability * 
                          temp_effect;  // Temperature also affects settlement success
    
    // Larval pool change
    Type larvae_change = adult_spawning + larval_immigration - m_larvae * larvae_curr - settlement_rate * larvae_curr;
    
    // EQUATION 10: JUVENILE COTS population change
    // Juveniles recruited from settling larvae, mature to adults, and experience mortality
    Type juvenile_recruitment = settlement_rate * larvae_curr;
    Type maturation_flux = maturation_rate * juvenile_curr;  // Juveniles maturing to adults
    Type juvenile_change = juvenile_recruitment - maturation_flux - m_cots_juvenile * juvenile_curr;
    
    // EQUATION 11: ADULT COTS population change
    // Adults mature from juveniles and experience mortality (spawning doesn't deplete adults)
    Type adult_change = maturation_flux - m_cots * adult_curr - density_mortality * adult_curr;
    
    // EQUATION 12: Fast-growing coral population change
    Type fast_growth = r_fast * fast_curr * (Type(1.0) - (fast_curr + competition_strength * slow_curr) / (K_coral + eps));
    Type fast_change = fast_growth - consumption_fast;
    
    // EQUATION 13: Slow-growing coral population change
    Type slow_growth = r_slow * slow_curr * (Type(1.0) - (slow_curr + competition_strength * fast_curr) / (K_coral + eps));
    Type slow_change = slow_growth - consumption_slow;
    
    // Update predictions for next time step
    larvae_pred(t+1) = larvae_curr + larvae_change;
    juvenile_pred(t+1) = juvenile_curr + juvenile_change;
    adult_pred(t+1) = adult_curr + adult_change;
    fast_pred(t+1) = fast_curr + fast_change;
    slow_pred(t+1) = slow_curr + slow_change;
    
    // Ensure predictions remain non-negative
    larvae_pred(t+1) = CppAD::CondExpGt(larvae_pred(t+1), eps, larvae_pred(t+1), eps);
    juvenile_pred(t+1) = CppAD::CondExpGt(juvenile_pred(t+1), eps, juvenile_pred(t+1), eps);
    adult_pred(t+1) = CppAD::CondExpGt(adult_pred(t+1), eps, adult_pred(t+1), eps);
    fast_pred(t+1) = CppAD::CondExpGt(fast_pred(t+1), eps, fast_pred(t+1), eps);
    slow_pred(t+1) = CppAD::CondExpGt(slow_pred(t+1), eps, slow_pred(t+1), eps);
    
    // Calculate total COTS for comparison with observations
    cots_pred(t+1) = juvenile_pred(t+1) + adult_pred(t+1);
  }
  
  // LIKELIHOOD CALCULATION
  Type nll = Type(0.0); // Negative log-likelihood
  
  // Likelihood for total COTS observations (lognormal distribution)
  // Observations are total COTS, so compare with juvenile + adult
  for(int t = 0; t < n; t++) {
    Type pred_log = log(cots_pred(t) + eps);
    Type obs_log = log(cots_dat(t) + eps);
    nll -= dnorm(obs_log, pred_log, sigma_cots, true);
  }
  
  // Likelihood for fast-growing coral observations (lognormal distribution)
  for(int t = 0; t < n; t++) {
    Type pred_log = log(fast_pred(t) + eps);
    Type obs_log = log(fast_dat(t) + eps);
    nll -= dnorm(obs_log, pred_log, sigma_fast, true);
  }
  
  // Likelihood for slow-growing coral observations (lognormal distribution)
  for(int t = 0; t < n; t++) {
    Type pred_log = log(slow_pred(t) + eps);
    Type obs_log = log(slow_dat(t) + eps);
    nll -= dnorm(obs_log, pred_log, sigma_slow, true);
  }
  
  // Soft parameter bounds using penalties with CppAD::CondExpGt
  // COTS adult growth rate: reasonable range 0.1-2.0 year^-1
  Type penalty_r_cots_upper = CppAD::CondExpGt(r_cots - Type(2.0), Type(0.0), r_cots - Type(2.0), Type(0.0));
  nll += Type(10.0) * pow(penalty_r_cots_upper, 2);
  Type penalty_r_cots_lower = CppAD::CondExpGt(Type(0.01) - r_cots, Type(0.0), Type(0.01) - r_cots, Type(0.0));
  nll += Type(10.0) * pow(penalty_r_cots_lower, 2);
  
  // Larval mortality: should be high (5-50 year^-1)
  Type penalty_m_larvae_upper = CppAD::CondExpGt(m_larvae - Type(50.0), Type(0.0), m_larvae - Type(50.0), Type(0.0));
  nll += Type(10.0) * pow(penalty_m_larvae_upper, 2);
  Type penalty_m_larvae_lower = CppAD::CondExpGt(Type(1.0) - m_larvae, Type(0.0), Type(1.0) - m_larvae, Type(0.0));
  nll += Type(10.0) * pow(penalty_m_larvae_lower, 2);
  
  // Settlement efficiency: reasonable range 0.01-1.0 year^-1
  Type penalty_settle_upper = CppAD::CondExpGt(settlement_efficiency - Type(1.0), Type(0.0), settlement_efficiency - Type(1.0), Type(0.0));
  nll += Type(10.0) * pow(penalty_settle_upper, 2);
  Type penalty_settle_lower = CppAD::CondExpGt(Type(0.01) - settlement_efficiency, Type(0.0), Type(0.01) - settlement_efficiency, Type(0.0));
  nll += Type(10.0) * pow(penalty_settle_lower, 2);
  
  // Settlement habitat dependence: must be between 0 and 1
  Type penalty_habitat_upper = CppAD::CondExpGt(settlement_habitat_dependence - Type(1.0), Type(0.0), settlement_habitat_dependence - Type(1.0), Type(0.0));
  nll += Type(100.0) * pow(penalty_habitat_upper, 2);
  Type penalty_habitat_lower = CppAD::CondExpGt(Type(0.0) - settlement_habitat_dependence, Type(0.0), Type(0.0) - settlement_habitat_dependence, Type(0.0));
  nll += Type(100.0) * pow(penalty_habitat_lower, 2);
  
  // Maturation rate: reasonable range 0.2-1.0 year^-1 (1-5 year maturation time)
  Type penalty_mat_upper = CppAD::CondExpGt(maturation_rate - Type(1.0), Type(0.0), maturation_rate - Type(1.0), Type(0.0));
  nll += Type(10.0) * pow(penalty_mat_upper, 2);
  Type penalty_mat_lower = CppAD::CondExpGt(Type(0.2) - maturation_rate, Type(0.0), Type(0.2) - maturation_rate, Type(0.0));
  nll += Type(10.0) * pow(penalty_mat_lower, 2);
  
  // Juvenile feeding efficiency: must be between 0.1 and 0.8
  Type penalty_juv_feed_upper = CppAD::CondExpGt(juvenile_feeding_efficiency - Type(0.8), Type(0.0), juvenile_feeding_efficiency - Type(0.8), Type(0.0));
  nll += Type(10.0) * pow(penalty_juv_feed_upper, 2);
  Type penalty_juv_feed_lower = CppAD::CondExpGt(Type(0.1) - juvenile_feeding_efficiency, Type(0.0), Type(0.1) - juvenile_feeding_efficiency, Type(0.0));
  nll += Type(10.0) * pow(penalty_juv_feed_lower, 2);
  
  // Allee strength: must be between 0 and 1
  Type penalty_allee_upper = CppAD::CondExpGt(allee_strength - Type(1.0), Type(0.0), allee_strength - Type(1.0), Type(0.0));
  nll += Type(100.0) * pow(penalty_allee_upper, 2);
  Type penalty_allee_lower = CppAD::CondExpGt(Type(0.0) - allee_strength, Type(0.0), Type(0.0) - allee_strength, Type(0.0));
  nll += Type(100.0) * pow(penalty_allee_lower, 2);
  
  // Competition strength: reasonable range 0-2
  Type penalty_comp_upper = CppAD::CondExpGt(competition_strength - Type(2.0), Type(0.0), competition_strength - Type(2.0), Type(0.0));
  nll += Type(10.0) * pow(penalty_comp_upper, 2);
  Type penalty_comp_lower = CppAD::CondExpGt(Type(0.0) - competition_strength, Type(0.0), Type(0.0) - competition_strength, Type(0.0));
  nll += Type(10.0) * pow(penalty_comp_lower, 2);
  
  // Feeding preference: should be > 0.1
  Type penalty_pref = CppAD::CondExpGt(Type(0.1) - feeding_preference, Type(0.0), Type(0.1) - feeding_preference, Type(0.0));
  nll += Type(10.0) * pow(penalty_pref, 2);
  
  // Temperature optimum: reasonable range 20-32°C
  Type penalty_temp_upper = CppAD::CondExpGt(temp_opt - Type(32.0), Type(0.0), temp_opt - Type(32.0), Type(0.0));
  nll += Type(10.0) * pow(penalty_temp_upper, 2);
  Type penalty_temp_lower = CppAD::CondExpGt(Type(20.0) - temp_opt, Type(0.0), Type(20.0) - temp_opt, Type(0.0));
  nll += Type(10.0) * pow(penalty_temp_lower, 2);
  
  // Larval survival efficiency: should be between 0.01 and 1.0
  Type penalty_larval_upper = CppAD::CondExpGt(larval_survival_efficiency - Type(1.0), Type(0.0), larval_survival_efficiency - Type(1.0), Type(0.0));
  nll += Type(10.0) * pow(penalty_larval_upper, 2);
  Type penalty_larval_lower = CppAD::CondExpGt(Type(0.01) - larval_survival_efficiency, Type(0.0), Type(0.01) - larval_survival_efficiency, Type(0.0));
  nll += Type(10.0) * pow(penalty_larval_lower, 2);
  
  // Nutrient outbreak multiplier: should be between 1.0 and 5.0
  Type penalty_outbreak_mult_upper = CppAD::CondExpGt(nutrient_outbreak_multiplier - Type(5.0), Type(0.0), nutrient_outbreak_multiplier - Type(5.0), Type(0.0));
  nll += Type(10.0) * pow(penalty_outbreak_mult_upper, 2);
  Type penalty_outbreak_mult_lower = CppAD::CondExpGt(Type(1.0) - nutrient_outbreak_multiplier, Type(0.0), Type(1.0) - nutrient_outbreak_multiplier, Type(0.0));
  nll += Type(10.0) * pow(penalty_outbreak_mult_lower, 2);
  
  // Nutrient response steepness: should be between 1.0 and 10.0
  Type penalty_steepness_upper = CppAD::CondExpGt(nutrient_response_steepness - Type(10.0), Type(0.0), nutrient_response_steepness - Type(10.0), Type(0.0));
  nll += Type(10.0) * pow(penalty_steepness_upper, 2);
  Type penalty_steepness_lower = CppAD::CondExpGt(Type(1.0) - nutrient_response_steepness, Type(0.0), Type(1.0) - nutrient_response_steepness, Type(0.0));
  nll += Type(10.0) * pow(penalty_steepness_lower, 2);
  
  // REPORTING
  REPORT(larvae_pred);
  REPORT(juvenile_pred);
  REPORT(adult_pred);
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(r_cots);
  REPORT(K_cots_base);
  REPORT(m_cots);
  REPORT(m_cots_juvenile);
  REPORT(m_larvae);
  REPORT(maturation_rate);
  REPORT(juvenile_feeding_efficiency);
  REPORT(allee_threshold);
  REPORT(allee_strength);
  REPORT(dd_mortality);
  REPORT(settlement_efficiency);
  REPORT(settlement_habitat_dependence);
  REPORT(settlement_saturation);
  REPORT(temp_opt);
  REPORT(temp_width);
  REPORT(temp_effect_max);
  REPORT(larval_survival_efficiency);
  REPORT(nutrient_half_sat);
  REPORT(nutrient_outbreak_threshold);
  REPORT(nutrient_outbreak_multiplier);
  REPORT(nutrient_response_steepness);
  REPORT(r_fast);
  REPORT(r_slow);
  REPORT(K_coral);
  REPORT(competition_strength);
  REPORT(attack_fast);
  REPORT(attack_slow);
  REPORT(handling_time);
  REPORT(feeding_preference);
  REPORT(coral_to_cots);
  REPORT(sigma_cots);
  REPORT(sigma_fast);
  REPORT(sigma_slow);
  
  return nll;
}
