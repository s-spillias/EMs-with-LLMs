#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS
  DATA_VECTOR(Year);                    // Time vector (years)
  DATA_VECTOR(cots_dat);               // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);               // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);               // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);                // Sea surface temperature forcing (Celsius)
  DATA_VECTOR(cotsimm_dat);            // COTS larval immigration forcing (individuals/m2/year)
  
  // COTS POPULATION PARAMETERS
  PARAMETER(log_r_cots);                // Log intrinsic growth rate of COTS (year^-1)
  PARAMETER(log_K_cots_base);           // Log baseline carrying capacity of COTS (individuals/m2)
  PARAMETER(log_m_cots);                // Log natural mortality rate of COTS (year^-1)
  PARAMETER(log_allee_threshold);       // Log Allee threshold density (individuals/m2)
  PARAMETER(allee_strength);            // Allee effect strength (dimensionless, 0-1)
  PARAMETER(log_dd_mortality);          // Log density-dependent mortality coefficient (m2/individuals/year)
  
  // TEMPERATURE EFFECTS ON COTS
  PARAMETER(temp_opt);                  // Optimal temperature for COTS recruitment (Celsius)
  PARAMETER(log_temp_width);            // Log temperature tolerance width (Celsius)
  PARAMETER(log_temp_effect_max);       // Log maximum temperature effect multiplier (dimensionless)
  
  // LARVAL SURVIVAL PARAMETERS
  PARAMETER(log_larval_survival_efficiency);  // Log efficiency of larval survival under optimal conditions (dimensionless)
  PARAMETER(log_nutrient_half_sat);     // Log half-saturation constant for nutrient effect (nutrient units)
  
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
  PARAMETER(log_sigma_cots);            // Log observation error SD for COTS (individuals/m2)
  PARAMETER(log_sigma_fast);            // Log observation error SD for fast coral (%)
  PARAMETER(log_sigma_slow);            // Log observation error SD for slow coral (%)
  
  // Transform parameters from log scale
  Type r_cots = exp(log_r_cots);
  Type K_cots_base = exp(log_K_cots_base);
  Type m_cots = exp(log_m_cots);
  Type allee_threshold = exp(log_allee_threshold);
  Type dd_mortality = exp(log_dd_mortality);
  Type temp_width = exp(log_temp_width);
  Type temp_effect_max = exp(log_temp_effect_max);
  Type larval_survival_efficiency = exp(log_larval_survival_efficiency);
  Type nutrient_half_sat = exp(log_nutrient_half_sat);
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
  
  // Initialize prediction vectors
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Set initial conditions from data
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Small constant for numerical stability
  Type eps = Type(1e-8);
  
  // TIME LOOP: Simulate dynamics forward in time
  for(int t = 0; t < n-1; t++) {
    
    // Current state (using only previous time step values)
    Type cots_curr = cots_pred(t);
    Type fast_curr = fast_pred(t);
    Type slow_curr = slow_pred(t);
    Type sst_curr = sst_dat(t);
    Type immigration_curr = cotsimm_dat(t);
    
    // Ensure non-negative values using CppAD::CondExpGt (if x > eps, return x, else return eps)
    cots_curr = CppAD::CondExpGt(cots_curr, eps, cots_curr, eps);
    fast_curr = CppAD::CondExpGt(fast_curr, eps, fast_curr, eps);
    slow_curr = CppAD::CondExpGt(slow_curr, eps, slow_curr, eps);
    
    // EQUATION 1: Temperature effect on COTS recruitment
    // Gaussian function centered at optimal temperature
    Type temp_deviation = sst_curr - temp_opt;
    Type temp_effect = Type(1.0) + (temp_effect_max - Type(1.0)) * exp(-0.5 * pow(temp_deviation / (temp_width + eps), 2));
    
    // EQUATION 2: Allee effect on COTS recruitment
    // Reduces recruitment at low densities, enhances at high densities
    Type allee_effect = Type(1.0) - allee_strength * exp(-cots_curr / (allee_threshold + eps));
    allee_effect = CppAD::CondExpGt(allee_effect, Type(0.01), allee_effect, Type(0.01)); // Prevent complete recruitment failure
    
    // EQUATION 2b: Nutrient-mediated larval survival effect
    // For now, set to 1.0 (no effect) until nutrient data is provided
    // When nutrient_dat is available, this will be: nutrient_curr / (nutrient_half_sat + nutrient_curr + eps)
    Type nutrient_effect = Type(1.0);
    
    // EQUATION 3: Type II functional response for COTS feeding on fast-growing coral
    // Accounts for handling time and preference
    Type effective_attack_fast = attack_fast * feeding_preference;
    Type consumption_fast = (effective_attack_fast * fast_curr * cots_curr) / 
                           (Type(1.0) + handling_time * (effective_attack_fast * fast_curr + attack_slow * slow_curr) + eps);
    
    // EQUATION 4: Type II functional response for COTS feeding on slow-growing coral
    Type consumption_slow = (attack_slow * slow_curr * cots_curr) / 
                           (Type(1.0) + handling_time * (effective_attack_fast * fast_curr + attack_slow * slow_curr) + eps);
    
    // EQUATION 5: Total coral available for COTS
    Type total_coral = fast_curr + slow_curr + eps;
    
    // EQUATION 6: Dynamic carrying capacity for COTS based on coral availability
    // COTS carrying capacity increases with coral food availability
    Type K_cots = K_cots_base * (total_coral / (K_coral + eps));
    
    // EQUATION 7: Density-dependent mortality of COTS
    // Increases at high densities due to disease, competition, resource limitation
    Type density_mortality = dd_mortality * cots_curr;
    
    // EQUATION 8: COTS population change
    // Includes nutrient-mediated larval survival efficiency
    // Immigration is modulated by: larval_survival_efficiency × nutrient_effect
    Type cots_recruitment = r_cots * cots_curr * (Type(1.0) - cots_curr / (K_cots + eps)) * allee_effect * temp_effect;
    Type larval_recruitment = immigration_curr * larval_survival_efficiency * nutrient_effect;
    Type cots_change = cots_recruitment + larval_recruitment - m_cots * cots_curr - density_mortality * cots_curr;
    
    // EQUATION 9: Fast-growing coral population change
    // Includes: logistic growth, COTS predation, competition with slow-growing coral
    Type fast_growth = r_fast * fast_curr * (Type(1.0) - (fast_curr + competition_strength * slow_curr) / (K_coral + eps));
    Type fast_change = fast_growth - consumption_fast;
    
    // EQUATION 10: Slow-growing coral population change
    // Includes: logistic growth, COTS predation, competition with fast-growing coral
    Type slow_growth = r_slow * slow_curr * (Type(1.0) - (slow_curr + competition_strength * fast_curr) / (K_coral + eps));
    Type slow_change = slow_growth - consumption_slow;
    
    // Update predictions for next time step
    cots_pred(t+1) = cots_curr + cots_change;
    fast_pred(t+1) = fast_curr + fast_change;
    slow_pred(t+1) = slow_curr + slow_change;
    
    // Ensure predictions remain non-negative using CppAD::CondExpGt
    cots_pred(t+1) = CppAD::CondExpGt(cots_pred(t+1), eps, cots_pred(t+1), eps);
    fast_pred(t+1) = CppAD::CondExpGt(fast_pred(t+1), eps, fast_pred(t+1), eps);
    slow_pred(t+1) = CppAD::CondExpGt(slow_pred(t+1), eps, slow_pred(t+1), eps);
  }
  
  // LIKELIHOOD CALCULATION
  Type nll = Type(0.0); // Negative log-likelihood
  
  // Likelihood for COTS observations (lognormal distribution)
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
  // COTS growth rate: reasonable range 0.1-2.0 year^-1
  Type penalty_r_cots_upper = CppAD::CondExpGt(r_cots - Type(2.0), Type(0.0), r_cots - Type(2.0), Type(0.0));
  nll += Type(10.0) * pow(penalty_r_cots_upper, 2);
  Type penalty_r_cots_lower = CppAD::CondExpGt(Type(0.01) - r_cots, Type(0.0), Type(0.01) - r_cots, Type(0.0));
  nll += Type(10.0) * pow(penalty_r_cots_lower, 2);
  
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
  
  // REPORTING
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(r_cots);
  REPORT(K_cots_base);
  REPORT(m_cots);
  REPORT(allee_threshold);
  REPORT(allee_strength);
  REPORT(dd_mortality);
  REPORT(temp_opt);
  REPORT(temp_width);
  REPORT(temp_effect_max);
  REPORT(larval_survival_efficiency);
  REPORT(nutrient_half_sat);
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
