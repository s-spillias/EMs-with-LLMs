#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS - Time series observations
  DATA_VECTOR(Year);                    // Time vector (years)
  DATA_VECTOR(sst_dat);                 // Sea surface temperature observations (°C)
  DATA_VECTOR(cotsimm_dat);             // COTS larval immigration rate (individuals/m²/year)
  DATA_VECTOR(cots_dat);                // Adult COTS abundance observations (individuals/m²)
  DATA_VECTOR(fast_dat);                // Fast-growing coral cover observations (%)
  DATA_VECTOR(slow_dat);                // Slow-growing coral cover observations (%)
  
  // COTS POPULATION PARAMETERS
  PARAMETER(log_cots_recruit_base);     // Log baseline COTS larval production rate (year⁻¹)
  PARAMETER(log_cots_mort_base);        // Log baseline adult COTS natural mortality rate (year⁻¹)
  PARAMETER(log_juvenile_mort_base);    // Log baseline juvenile COTS mortality rate (year⁻¹)
  PARAMETER(log_maturation_rate);       // Log juvenile to adult maturation rate (year⁻¹)
  PARAMETER(log_juvenile_food_effect);  // Log effect of coral on juvenile survival (dimensionless)
  PARAMETER(log_larval_settlement);     // Log larval settlement efficiency (dimensionless)
  PARAMETER(log_allee_threshold);       // Log COTS density for Allee effect threshold (individuals/m²)
  PARAMETER(log_allee_strength);        // Log strength of Allee effect (dimensionless)
  PARAMETER(log_temp_recruit_opt);      // Log optimal temperature for COTS recruitment (°C)
  PARAMETER(log_temp_recruit_width);    // Log temperature tolerance width for recruitment (°C)
  PARAMETER(log_density_mort_rate);     // Log density-dependent mortality coefficient (m²/individuals/year)
  PARAMETER(log_immigration_effect);    // Log immigration contribution to recruitment (dimensionless)
  
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
  
  // Transform parameters from log scale
  Type cots_recruit_base = exp(log_cots_recruit_base);
  Type cots_mort_base = exp(log_cots_mort_base);
  Type juvenile_mort_base = exp(log_juvenile_mort_base);
  Type maturation_rate = exp(log_maturation_rate);
  Type juvenile_food_effect = exp(log_juvenile_food_effect);
  Type larval_settlement = exp(log_larval_settlement);
  Type allee_threshold = exp(log_allee_threshold);
  Type allee_strength = exp(log_allee_strength);
  Type temp_recruit_opt = exp(log_temp_recruit_opt);
  Type temp_recruit_width = exp(log_temp_recruit_width);
  Type density_mort_rate = exp(log_density_mort_rate);
  Type immigration_effect = exp(log_immigration_effect);
  
  Type fast_growth_rate = exp(log_fast_growth_rate);
  Type slow_growth_rate = exp(log_slow_growth_rate);
  Type fast_carrying_cap = exp(log_fast_carrying_cap);
  Type slow_carrying_cap = exp(log_slow_carrying_cap);
  Type coral_competition = exp(log_coral_competition);
  Type temp_stress_threshold = exp(log_temp_stress_threshold);
  Type temp_stress_rate = exp(log_temp_stress_rate);
  
  Type attack_rate_fast = exp(log_attack_rate_fast);
  Type attack_rate_slow = exp(log_attack_rate_slow);
  Type handling_time_fast = exp(log_handling_time_fast);
  Type handling_time_slow = exp(log_handling_time_slow);
  Type preference_switch = exp(log_preference_switch);
  Type conversion_efficiency = exp(log_conversion_efficiency);
  
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_fast = exp(log_sigma_fast);
  Type sigma_slow = exp(log_sigma_slow);
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(0.01);
  sigma_cots = sigma_cots + min_sigma;
  sigma_fast = sigma_fast + min_sigma;
  sigma_slow = sigma_slow + min_sigma;
  
  // Initialize prediction vectors
  int n = Year.size();
  vector<Type> juvenile_pred(n);        // Juvenile COTS density
  vector<Type> adult_pred(n);           // Adult COTS density
  vector<Type> cots_pred(n);            // Total COTS (for comparison with observations)
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Set initial conditions from first observations
  // Assume initial population is 80% adult, 20% juvenile (arbitrary but reasonable)
  adult_pred(0) = cots_dat(0) * Type(0.8);
  juvenile_pred(0) = cots_dat(0) * Type(0.2);
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Initialize negative log-likelihood
  Type nll = Type(0.0);
  
  // TIME LOOP - Dynamic model equations
  for(int t = 1; t < n; t++) {
    
    // Previous time step values
    Type juvenile_prev = juvenile_pred(t-1);
    Type adult_prev = adult_pred(t-1);
    Type fast_prev = fast_pred(t-1);
    Type slow_prev = slow_pred(t-1);
    Type sst_prev = sst_dat(t-1);
    Type immigration_prev = cotsimm_dat(t-1);
    
    // Total coral cover (food availability for juveniles)
    Type total_coral = fast_prev + slow_prev;
    
    // EQUATION 1: Temperature effect on COTS recruitment (Gaussian response)
    Type temp_diff = sst_prev - temp_recruit_opt;
    Type temp_effect = exp(-Type(0.5) * pow(temp_diff / (temp_recruit_width + eps), 2));
    
    // EQUATION 2: Allee effect on COTS recruitment (sigmoid function based on adult density)
    Type allee_effect = pow(adult_prev, allee_strength) / (pow(allee_threshold, allee_strength) + pow(adult_prev, allee_strength) + eps);
    
    // EQUATION 3: Larval production from adults
    Type larval_production = cots_recruit_base * temp_effect * allee_effect * adult_prev;
    
    // EQUATION 4: Immigration contribution to larval pool
    Type immigration_contribution = immigration_effect * immigration_prev;
    
    // EQUATION 5: Total larval settlement to juveniles
    Type larval_settlement_rate = larval_settlement * (larval_production + immigration_contribution);
    
    // EQUATION 6: Food-dependent juvenile survival
    // High coral cover reduces starvation mortality
    // Normalized by carrying capacity to get relative food availability
    Type food_availability = total_coral / ((fast_carrying_cap + slow_carrying_cap) / Type(2.0) + eps);
    Type food_survival_factor = Type(1.0) - juvenile_food_effect * food_availability;
    // Ensure survival factor stays positive and reasonable
    food_survival_factor = CppAD::CondExpGt(food_survival_factor, Type(0.1), food_survival_factor, Type(0.1));
    
    // EQUATION 7: Juvenile mortality (high baseline, reduced by food availability)
    Type juvenile_mortality = juvenile_mort_base * food_survival_factor;
    
    // EQUATION 8: Juvenile maturation to adults
    Type maturation_flux = maturation_rate * juvenile_prev;
    
    // EQUATION 9: Juvenile population change
    Type juvenile_change = larval_settlement_rate - juvenile_mortality * juvenile_prev - maturation_flux;
    juvenile_pred(t) = juvenile_prev + juvenile_change;
    // Prevent extinction using smooth lower bound
    Type juvenile_min = Type(0.0001);
    juvenile_pred(t) = CppAD::CondExpGt(juvenile_pred(t), juvenile_min, juvenile_pred(t), juvenile_min);
    
    // EQUATION 10: Density-dependent adult mortality
    Type adult_mortality = cots_mort_base + density_mort_rate * adult_prev;
    
    // EQUATION 11: Type II functional response for fast coral predation with preference
    Type fast_available = fast_prev + eps;
    Type preference_fast = Type(1.0) / (Type(1.0) + exp(-Type(5.0) * (fast_prev - preference_switch)));
    Type consumption_fast = (attack_rate_fast * preference_fast * adult_prev * fast_available) / 
                           (Type(1.0) + attack_rate_fast * handling_time_fast * fast_available + eps);
    
    // EQUATION 12: Type II functional response for slow coral predation with switching
    Type slow_available = slow_prev + eps;
    Type preference_slow = Type(1.0) - preference_fast;
    Type consumption_slow = (attack_rate_slow * preference_slow * adult_prev * slow_available) / 
                           (Type(1.0) + attack_rate_slow * handling_time_slow * slow_available + eps);
    
    // EQUATION 13: Total coral consumption and conversion to COTS biomass
    Type total_consumption = consumption_fast + consumption_slow;
    Type adult_gain_from_feeding = conversion_efficiency * total_consumption;
    
    // EQUATION 14: Adult population change
    Type adult_change = maturation_flux + adult_gain_from_feeding - adult_mortality * adult_prev;
    adult_pred(t) = adult_prev + adult_change;
    // Prevent extinction using smooth lower bound
    Type adult_min = Type(0.0001);
    adult_pred(t) = CppAD::CondExpGt(adult_pred(t), adult_min, adult_pred(t), adult_min);
    
    // EQUATION 15: Total COTS for comparison with observations
    cots_pred(t) = juvenile_pred(t) + adult_pred(t);
    
    // EQUATION 16: Temperature stress on corals (smooth transition)
    Type temp_excess = sst_prev - temp_stress_threshold;
    Type temp_stress = temp_stress_rate * temp_excess / (Type(1.0) + exp(-Type(10.0) * temp_excess));
    
    // EQUATION 17: Fast coral logistic growth with competition and predation
    Type fast_growth = fast_growth_rate * fast_prev * 
                      (Type(1.0) - (fast_prev + coral_competition * slow_prev) / (fast_carrying_cap + eps));
    Type fast_loss = consumption_fast + temp_stress * fast_prev;
    Type fast_change = fast_growth - fast_loss;
    fast_pred(t) = fast_prev + fast_change;
    // Apply bounds using smooth conditional expressions
    Type fast_min = Type(0.01);
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), fast_min, fast_pred(t), fast_min);
    fast_pred(t) = CppAD::CondExpLt(fast_pred(t), fast_carrying_cap, fast_pred(t), fast_carrying_cap);
    
    // EQUATION 18: Slow coral logistic growth with competition and predation
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
    // Lognormal likelihood for total COTS (strictly positive, spans orders of magnitude)
    Type log_cots_pred = log(cots_pred(t) + eps);
    Type log_cots_obs = log(cots_dat(t) + eps);
    nll -= dnorm(log_cots_obs, log_cots_pred, sigma_cots, true);
    
    // Normal likelihood for coral cover (percentage data)
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_fast, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_slow, true);
  }
  
  // SOFT CONSTRAINTS - Biological realism penalties
  // Penalize extreme parameter values with smooth quadratic penalties
  
  // COTS larval production should be moderate (0.01 to 2.0 year⁻¹)
  Type cots_recruit_penalty = Type(0.0);
  cots_recruit_penalty += CppAD::CondExpLt(cots_recruit_base, Type(0.01), 
                                           Type(10.0) * pow(cots_recruit_base - Type(0.01), 2), Type(0.0));
  cots_recruit_penalty += CppAD::CondExpGt(cots_recruit_base, Type(2.0), 
                                           Type(10.0) * pow(cots_recruit_base - Type(2.0), 2), Type(0.0));
  nll += cots_recruit_penalty;
  
  // Adult COTS mortality should be moderate (0.1 to 1.5 year⁻¹)
  Type cots_mort_penalty = Type(0.0);
  cots_mort_penalty += CppAD::CondExpLt(cots_mort_base, Type(0.1), 
                                        Type(10.0) * pow(cots_mort_base - Type(0.1), 2), Type(0.0));
  cots_mort_penalty += CppAD::CondExpGt(cots_mort_base, Type(1.5), 
                                        Type(10.0) * pow(cots_mort_base - Type(1.5), 2), Type(0.0));
  nll += cots_mort_penalty;
  
  // Juvenile mortality should be high (1.0 to 10.0 year⁻¹)
  Type juvenile_mort_penalty = Type(0.0);
  juvenile_mort_penalty += CppAD::CondExpLt(juvenile_mort_base, Type(1.0), 
                                            Type(10.0) * pow(juvenile_mort_base - Type(1.0), 2), Type(0.0));
  juvenile_mort_penalty += CppAD::CondExpGt(juvenile_mort_base, Type(10.0), 
                                            Type(10.0) * pow(juvenile_mort_base - Type(10.0), 2), Type(0.0));
  nll += juvenile_mort_penalty;
  
  // Maturation rate should be reasonable (0.2 to 1.0 year⁻¹)
  Type maturation_penalty = Type(0.0);
  maturation_penalty += CppAD::CondExpLt(maturation_rate, Type(0.2), 
                                         Type(10.0) * pow(maturation_rate - Type(0.2), 2), Type(0.0));
  maturation_penalty += CppAD::CondExpGt(maturation_rate, Type(1.0), 
                                         Type(10.0) * pow(maturation_rate - Type(1.0), 2), Type(0.0));
  nll += maturation_penalty;
  
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
  
  // REPORTING - Output predictions and parameters
  REPORT(juvenile_pred);
  REPORT(adult_pred);
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  REPORT(cots_recruit_base);
  REPORT(cots_mort_base);
  REPORT(juvenile_mort_base);
  REPORT(maturation_rate);
  REPORT(juvenile_food_effect);
  REPORT(larval_settlement);
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
  
  return nll;
}
