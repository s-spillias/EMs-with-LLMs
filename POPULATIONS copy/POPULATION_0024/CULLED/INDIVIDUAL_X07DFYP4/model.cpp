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
  PARAMETER(log_temp_width);            // Log temperature tolerance width (Celsius)
  PARAMETER(immigration_effect);        // Immigration enhancement factor (dimensionless)
  
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
  Type temp_width = exp(log_temp_width);                   // Temperature width (Celsius)
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
    
    // EQUATION 2: Temperature effect on COTS recruitment
    // Gaussian function centered on optimal temperature
    Type temp_diff = sst_curr - temp_opt;                   // Deviation from optimum
    Type temp_effect = exp(-0.5 * pow(temp_diff / (temp_width + eps), 2)); // Gaussian temperature response
    
    // EQUATION 3: Immigration enhancement
    // Larval immigration boosts local recruitment
    Type immigration_boost = Type(1.0) + immigration_effect * immigration; // Immigration multiplier
    
    // EQUATION 4: Type II functional response for fast coral predation
    // Captures saturation in consumption rate at high prey densities
    Type consumption_fast = (attack_fast * fast_prev * cots_prev) / 
                           (Type(1.0) + attack_fast * handling_fast * fast_prev + eps); // Fast coral consumption (% cover/year)
    
    // EQUATION 5: Type II functional response for slow coral predation
    // COTS switch to slow coral when fast coral is depleted
    Type consumption_slow = (attack_slow * slow_prev * cots_prev) / 
                           (Type(1.0) + attack_slow * handling_slow * slow_prev + eps); // Slow coral consumption (% cover/year)
    
    // EQUATION 6: Prey preference and switching
    // COTS prefer fast coral but switch when it becomes scarce
    Type total_coral = fast_prev + slow_prev + eps;         // Total coral available
    Type fast_proportion = fast_prev / total_coral;         // Proportion of fast coral
    Type preference_weight = preference_fast * fast_proportion + 
                            (Type(1.0) - preference_fast) * (Type(1.0) - fast_proportion); // Weighted preference
    
    // EQUATION 7: Weighted consumption rates
    Type consumption_fast_weighted = consumption_fast * preference_weight; // Adjusted fast consumption
    Type consumption_slow_weighted = consumption_slow * (Type(1.0) - preference_weight); // Adjusted slow consumption
    
    // EQUATION 8: Total food intake for COTS
    Type total_consumption = consumption_fast_weighted + consumption_slow_weighted; // Total coral consumed
    
    // EQUATION 9: Density-dependent mortality
    // Increases with crowding (disease, competition)
    Type mortality_dd = mort_base + mort_density * cots_prev; // Total mortality rate (year^-1)
    
    // EQUATION 10: Starvation effect
    // Mortality increases when coral resources are depleted
    Type starvation_factor = Type(1.0) + Type(2.0) * exp(-total_coral / Type(5.0)); // Starvation multiplier
    Type mortality_total = mortality_dd * starvation_factor; // Combined mortality (year^-1)
    
    // EQUATION 11: COTS population growth
    // Combines recruitment, predation-derived growth, immigration, and mortality
    Type cots_growth = r_cots * cots_prev * allee_factor * temp_effect * immigration_boost * 
                      (Type(1.0) - cots_prev / (K_cots + eps)) + // Logistic growth with all modifiers
                      conversion_eff * total_consumption - // Growth from coral consumption
                      mortality_total * cots_prev; // Mortality losses
    
    // EQUATION 12: Update COTS abundance
    cots_pred(t) = cots_prev + cots_growth;                 // COTS at time t
    cots_pred(t) = cots_pred(t) / (Type(1.0) + eps);        // Stabilize
    if(cots_pred(t) < Type(0.0)) cots_pred(t) = Type(1e-6); // Prevent negative values
    
    // EQUATION 13: Temperature stress on corals
    // Warm temperatures reduce coral growth and increase mortality
    Type temp_stress = Type(0.0);                           // Initialize stress
    if(sst_curr > temp_stress_threshold) {
      temp_stress = temp_stress_rate * (sst_curr - temp_stress_threshold); // Stress mortality (year^-1)
    }
    
    // EQUATION 14: Fast coral dynamics
    // Logistic growth reduced by COTS predation and temperature stress
    Type fast_growth = r_fast * fast_prev * (Type(1.0) - (fast_prev + slow_prev) / (K_coral + eps)) - 
                      consumption_fast_weighted - // COTS predation loss
                      temp_stress * fast_prev; // Temperature stress loss
    
    // EQUATION 15: Update fast coral cover
    fast_pred(t) = fast_prev + fast_growth;                 // Fast coral at time t
    fast_pred(t) = fast_pred(t) / (Type(1.0) + eps);        // Stabilize
    if(fast_pred(t) < Type(0.0)) fast_pred(t) = Type(1e-6); // Prevent negative values
    if(fast_pred(t) > K_coral) fast_pred(t) = K_coral;     // Cap at carrying capacity
    
    // EQUATION 16: Slow coral dynamics
    // Slower growth but more resistant to disturbance
    Type slow_growth = r_slow * slow_prev * (Type(1.0) - (fast_prev + slow_prev) / (K_coral + eps)) - 
                      consumption_slow_weighted - // COTS predation loss
                      Type(0.5) * temp_stress * slow_prev; // Reduced temperature sensitivity
    
    // EQUATION 17: Update slow coral cover
    slow_pred(t) = slow_prev + slow_growth;                 // Slow coral at time t
    slow_pred(t) = slow_pred(t) / (Type(1.0) + eps);        // Stabilize
    if(slow_pred(t) < Type(0.0)) slow_pred(t) = Type(1e-6); // Prevent negative values
    if(slow_pred(t) > K_coral) slow_pred(t) = K_coral;     // Cap at carrying capacity
  }
  
  // LIKELIHOOD CALCULATION
  Type nll = Type(0.0);                                     // Initialize negative log-likelihood
  
  // EQUATION 18: COTS observation likelihood
  // Lognormal distribution for strictly positive abundance data
  for(int t = 0; t < n; t++) {
    Type log_cots_pred = log(cots_pred(t) + eps);          // Log predicted COTS
    Type log_cots_obs = log(cots_dat(t) + eps);            // Log observed COTS
    nll -= dnorm(log_cots_obs, log_cots_pred, sigma_cots_use, true); // Lognormal likelihood
  }
  
  // EQUATION 19: Fast coral observation likelihood
  // Normal distribution for percentage cover data
  for(int t = 0; t < n; t++) {
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_fast_use, true); // Normal likelihood
  }
  
  // EQUATION 20: Slow coral observation likelihood
  // Normal distribution for percentage cover data
  for(int t = 0; t < n; t++) {
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_slow_use, true); // Normal likelihood
  }
  
  // EQUATION 21: Soft parameter bounds using penalties
  // Allee strength bounded between 0 and 1
  Type penalty = Type(0.0);                                 // Initialize penalty
  if(allee_strength < Type(0.0)) penalty += Type(100.0) * pow(allee_strength, 2);
  if(allee_strength > Type(1.0)) penalty += Type(100.0) * pow(allee_strength - Type(1.0), 2);
  
  // EQUATION 22: Preference parameter bounded between 0 and 1
  if(preference_fast < Type(0.0)) penalty += Type(100.0) * pow(preference_fast, 2);
  if(preference_fast > Type(1.0)) penalty += Type(100.0) * pow(preference_fast - Type(1.0), 2);
  
  // EQUATION 23: Conversion efficiency bounded between 0 and 1
  if(conversion_eff < Type(0.0)) penalty += Type(100.0) * pow(conversion_eff, 2);
  if(conversion_eff > Type(1.0)) penalty += Type(100.0) * pow(conversion_eff - Type(1.0), 2);
  
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
  REPORT(temp_width);                                       // Report temperature width
  REPORT(immigration_effect);                               // Report immigration effect
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
