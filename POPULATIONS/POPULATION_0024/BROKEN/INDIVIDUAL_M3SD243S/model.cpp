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
  DATA_VECTOR(cotsimm_dat);            // COTS larval immigration rate (individuals/m2/year)
  
  // COTS POPULATION PARAMETERS
  PARAMETER(log_r_cots);                // Log intrinsic recruitment rate of COTS (year^-1)
  PARAMETER(log_K_cots);                // Log carrying capacity for COTS based on coral resources (individuals/m2)
  PARAMETER(log_m_cots);                // Log baseline natural mortality rate of COTS (year^-1)
  PARAMETER(log_allee_threshold);       // Log Allee threshold density for successful reproduction (individuals/m2)
  PARAMETER(log_temp_opt);              // Log optimal temperature for COTS recruitment (Celsius)
  PARAMETER(log_temp_width);            // Log temperature tolerance width (Celsius)
  PARAMETER(immigration_effect);        // Effect of larval immigration on recruitment (dimensionless)
  
  // CORAL GROWTH PARAMETERS
  PARAMETER(log_r_fast);                // Log intrinsic growth rate of fast-growing corals (year^-1)
  PARAMETER(log_r_slow);                // Log intrinsic growth rate of slow-growing corals (year^-1)
  PARAMETER(log_K_coral);               // Log carrying capacity for total coral cover (%)
  PARAMETER(log_temp_stress_threshold); // Log temperature threshold for coral stress (Celsius)
  PARAMETER(temp_stress_effect);        // Effect of temperature stress on coral growth (dimensionless)
  
  // PREDATION PARAMETERS
  PARAMETER(log_attack_fast);           // Log attack rate on fast-growing corals (m2/individual/year)
  PARAMETER(log_attack_slow);           // Log attack rate on slow-growing corals (m2/individual/year)
  PARAMETER(log_handling_time);         // Log handling time per unit coral consumed (year/%)
  PARAMETER(log_preference_fast);       // Log preference coefficient for fast-growing corals (dimensionless)
  PARAMETER(predation_efficiency);      // Conversion efficiency of coral to COTS biomass (dimensionless)
  
  // OBSERVATION ERROR PARAMETERS
  PARAMETER(log_sigma_cots);            // Log observation error SD for COTS (individuals/m2)
  PARAMETER(log_sigma_fast);            // Log observation error SD for fast corals (%)
  PARAMETER(log_sigma_slow);            // Log observation error SD for slow corals (%)
  
  // Transform parameters from log scale
  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type m_cots = exp(log_m_cots);
  Type allee_threshold = exp(log_allee_threshold);
  Type temp_opt = exp(log_temp_opt);
  Type temp_width = exp(log_temp_width);
  Type r_fast = exp(log_r_fast);
  Type r_slow = exp(log_r_slow);
  Type K_coral = exp(log_K_coral);
  Type temp_stress_threshold = exp(log_temp_stress_threshold);
  Type attack_fast = exp(log_attack_fast);
  Type attack_slow = exp(log_attack_slow);
  Type handling_time = exp(log_handling_time);
  Type preference_fast = exp(log_preference_fast);
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_fast = exp(log_sigma_fast);
  Type sigma_slow = exp(log_sigma_slow);
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(0.01);
  sigma_cots = sigma_cots + min_sigma;
  sigma_fast = sigma_fast + min_sigma;
  sigma_slow = sigma_slow + min_sigma;
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  int n = Year.size();  // Number of time steps
  
  // Initialize prediction vectors
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Set initial conditions from first observation
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Time step (assuming annual data)
  Type dt = Type(1.0);
  
  // PROCESS MODEL: Iterate through time steps
  for(int t = 0; t < (n-1); t++) {
    
    // Current state variables (from previous time step)
    Type C = cots_pred(t);              // Current COTS abundance
    Type F = fast_pred(t);              // Current fast coral cover
    Type S = slow_pred(t);              // Current slow coral cover
    Type T = sst_dat(t);                // Current temperature
    Type I = cotsimm_dat(t);            // Current immigration rate
    
    // Ensure non-negative populations
    C = C + eps;
    F = F + eps;
    S = S + eps;
    
    // EQUATION 1: Temperature effect on COTS recruitment (Gaussian response)
    Type temp_effect = exp(-pow(T - temp_opt, 2) / (Type(2.0) * temp_width * temp_width));
    
    // EQUATION 2: Allee effect (reduced recruitment at low densities due to difficulty finding mates)
    Type allee_effect = C / (C + allee_threshold);
    
    // EQUATION 3: Total coral availability for COTS feeding
    Type total_coral = F + S + eps;
    
    // EQUATION 4: Resource limitation on COTS (logistic term based on coral availability)
    Type resource_limitation = total_coral / (K_coral + eps);
    
    // EQUATION 5: Type II functional response for predation on fast-growing corals
    Type consumption_fast = (attack_fast * preference_fast * F * C) / 
                           (Type(1.0) + attack_fast * preference_fast * handling_time * F + 
                            attack_slow * handling_time * S + eps);
    
    // EQUATION 6: Type II functional response for predation on slow-growing corals
    Type consumption_slow = (attack_slow * S * C) / 
                           (Type(1.0) + attack_fast * preference_fast * handling_time * F + 
                            attack_slow * handling_time * S + eps);
    
    // EQUATION 7: Total consumption by COTS
    Type total_consumption = consumption_fast + consumption_slow;
    
    // EQUATION 8: Density-dependent mortality (increases with crowding)
    Type density_mortality = m_cots * (Type(1.0) + C / (K_cots + eps));
    
    // EQUATION 9: COTS recruitment (temperature-dependent, Allee effect, immigration boost)
    Type recruitment = r_cots * C * temp_effect * allee_effect * resource_limitation + 
                      immigration_effect * I;
    
    // EQUATION 10: COTS population change
    Type dC_dt = recruitment - density_mortality * C + 
                 predation_efficiency * total_consumption;
    
    // EQUATION 11: Temperature stress on coral growth (reduced growth above threshold)
    Type coral_stress = Type(1.0);
    if(T > temp_stress_threshold) {
      coral_stress = exp(-temp_stress_effect * (T - temp_stress_threshold));
    }
    
    // EQUATION 12: Competition for space (total coral cannot exceed carrying capacity)
    Type space_available = Type(1.0) - (F + S) / (K_coral + eps);
    space_available = space_available / (Type(1.0) + exp(-Type(10.0) * space_available)); // Smooth transition
    
    // EQUATION 13: Fast-growing coral dynamics (logistic growth, predation loss, stress)
    Type dF_dt = r_fast * F * space_available * coral_stress - consumption_fast;
    
    // EQUATION 14: Slow-growing coral dynamics (logistic growth, predation loss, stress)
    Type dS_dt = r_slow * S * space_available * coral_stress - consumption_slow;
    
    // Update predictions for next time step using Euler integration
    cots_pred(t+1) = C + dC_dt * dt;
    fast_pred(t+1) = F + dF_dt * dt;
    slow_pred(t+1) = S + dS_dt * dt;
    
    // Ensure predictions remain non-negative using smooth bounding
    cots_pred(t+1) = cots_pred(t+1) / (Type(1.0) + exp(-Type(10.0) * cots_pred(t+1)));
    fast_pred(t+1) = fast_pred(t+1) / (Type(1.0) + exp(-Type(10.0) * fast_pred(t+1)));
    slow_pred(t+1) = slow_pred(t+1) / (Type(1.0) + exp(-Type(10.0) * slow_pred(t+1)));
  }
  
  // LIKELIHOOD CALCULATION
  Type nll = Type(0.0);  // Initialize negative log-likelihood
  
  // Likelihood for COTS observations (lognormal distribution)
  for(int t = 0; t < n; t++) {
    Type pred = cots_pred(t) + eps;
    Type obs = cots_dat(t) + eps;
    nll -= dnorm(log(obs), log(pred), sigma_cots, true);
  }
  
  // Likelihood for fast-growing coral observations (normal distribution)
  for(int t = 0; t < n; t++) {
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_fast, true);
  }
  
  // Likelihood for slow-growing coral observations (normal distribution)
  for(int t = 0; t < n; t++) {
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_slow, true);
  }
  
  // Soft parameter bounds using penalties (to keep parameters biologically reasonable)
  // Penalty for extremely high COTS carrying capacity
  if(K_cots > Type(10.0)) {
    nll += Type(0.1) * pow(K_cots - Type(10.0), 2);
  }
  
  // Penalty for extremely high coral carrying capacity
  if(K_coral > Type(100.0)) {
    nll += Type(0.1) * pow(K_coral - Type(100.0), 2);
  }
  
  // Penalty for negative predation efficiency
  if(predation_efficiency < Type(0.0)) {
    nll += Type(10.0) * pow(predation_efficiency, 2);
  }
  
  // Penalty for predation efficiency > 1 (cannot convert more than 100% of food to biomass)
  if(predation_efficiency > Type(1.0)) {
    nll += Type(10.0) * pow(predation_efficiency - Type(1.0), 2);
  }
  
  // REPORTING
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(r_cots);
  REPORT(K_cots);
  REPORT(m_cots);
  REPORT(allee_threshold);
  REPORT(temp_opt);
  REPORT(temp_width);
  REPORT(immigration_effect);
  REPORT(r_fast);
  REPORT(r_slow);
  REPORT(K_coral);
  REPORT(temp_stress_threshold);
  REPORT(temp_stress_effect);
  REPORT(attack_fast);
  REPORT(attack_slow);
  REPORT(handling_time);
  REPORT(preference_fast);
  REPORT(predation_efficiency);
  REPORT(sigma_cots);
  REPORT(sigma_fast);
  REPORT(sigma_slow);
  
  return nll;
}
