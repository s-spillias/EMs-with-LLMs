#include <TMB.hpp>

// Custom posfun implementation for ensuring positive values
template<class Type>
Type posfun(Type x, Type eps, Type &pen) {
  pen += CppAD::CondExpLt(x, eps, Type(0.01) * pow(x - eps, 2), Type(0.0));
  return CppAD::CondExpGe(x, eps, x, eps / (Type(2.0) - x / eps));
}

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs
  DATA_VECTOR(Year);           // Time vector (years)
  DATA_VECTOR(cots_dat);       // Observed COTS abundance (individuals/m^2)
  DATA_VECTOR(fast_dat);       // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);       // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);        // Sea surface temperature forcing (degrees Celsius)
  DATA_VECTOR(cotsimm_dat);    // COTS larval immigration rate (individuals/m^2/year)
  
  // COTS population parameters
  PARAMETER(r_cots);                    // Intrinsic growth rate of COTS (year^-1); from literature on starfish demography
  PARAMETER(K_cots_base);               // Baseline carrying capacity for COTS (individuals/m^2); from field surveys of maximum densities
  PARAMETER(allee_threshold);           // Allee effect threshold density (individuals/m^2); from reproductive biology studies
  PARAMETER(allee_strength);            // Strength of Allee effect (dimensionless); estimated from outbreak initiation patterns
  PARAMETER(temp_opt);                  // Optimal temperature for COTS recruitment (degrees Celsius); from larval development experiments
  PARAMETER(temp_sensitivity);          // Temperature sensitivity parameter (per degree Celsius); estimated from recruitment-temperature relationships
  PARAMETER(immigration_effect);        // Immigration efficiency (dimensionless); estimated from larval dispersal models
  
  // COTS predation parameters
  PARAMETER(attack_rate_fast);          // Attack rate on fast-growing coral (m^2/individual/year); from feeding rate experiments
  PARAMETER(attack_rate_slow);          // Attack rate on slow-growing coral (m^2/individual/year); from feeding preference studies
  PARAMETER(handling_time_fast);        // Handling time for fast-growing coral (year); from feeding behavior observations
  PARAMETER(handling_time_slow);        // Handling time for slow-growing coral (year); from feeding behavior observations
  PARAMETER(conversion_efficiency);     // Conversion of coral to COTS carrying capacity (dimensionless); estimated from energetics
  
  // Coral growth parameters
  PARAMETER(r_fast);                    // Growth rate of fast-growing coral (year^-1); from coral growth studies on Acropora
  PARAMETER(r_slow);                    // Growth rate of slow-growing coral (year^-1); from coral growth studies on massive corals
  PARAMETER(K_coral_total);             // Total coral carrying capacity (% cover); from reef surveys of maximum coral cover
  PARAMETER(competition_fast);          // Competition coefficient (slow on fast) (dimensionless); estimated from community dynamics
  PARAMETER(competition_slow);          // Competition coefficient (fast on slow) (dimensionless); estimated from community dynamics
  
  // Observation error parameters
  PARAMETER(sigma_cots);                // Observation error SD for COTS (log scale, dimensionless); estimated from data variability
  PARAMETER(sigma_fast);                // Observation error SD for fast coral (log scale, dimensionless); estimated from data variability
  PARAMETER(sigma_slow);                // Observation error SD for slow coral (log scale, dimensionless); estimated from data variability
  
  int n = cots_dat.size();              // Number of time steps
  Type eps = Type(1e-8);                // Small constant to prevent division by zero
  Type min_sd = Type(0.01);             // Minimum standard deviation to prevent numerical issues
  
  // Initialize prediction vectors
  vector<Type> cots_pred(n);            // Predicted COTS abundance
  vector<Type> fast_pred(n);            // Predicted fast-growing coral cover
  vector<Type> slow_pred(n);            // Predicted slow-growing coral cover
  
  // Set initial conditions from first data point
  cots_pred(0) = cots_dat(0);           // Initialize COTS from observed data
  fast_pred(0) = fast_dat(0);           // Initialize fast coral from observed data
  slow_pred(0) = slow_dat(0);           // Initialize slow coral from observed data
  
  // Initialize negative log-likelihood
  Type nll = Type(0.0);                 // Negative log-likelihood accumulator
  
  // Time loop for model dynamics
  for(int t = 1; t < n; t++) {
    
    // Previous time step values (avoid data leakage)
    Type cots_prev = cots_pred(t-1);    // COTS abundance at previous time step
    Type fast_prev = fast_pred(t-1);    // Fast coral cover at previous time step
    Type slow_prev = slow_pred(t-1);    // Slow coral cover at previous time step
    Type sst_curr = sst_dat(t);         // Current sea surface temperature
    Type immigration_curr = cotsimm_dat(t); // Current immigration rate
    
    // Ensure non-negative values using posfun (smooth positive function)
    Type pen = Type(0.0);                                     // Penalty accumulator for posfun
    cots_prev = posfun(cots_prev, Type(0.0), pen);           // Prevent negative COTS abundance
    fast_prev = posfun(fast_prev, Type(0.0), pen);           // Prevent negative fast coral cover
    slow_prev = posfun(slow_prev, Type(0.0), pen);           // Prevent negative slow coral cover
    nll += Type(1000.0) * pen;                               // Add penalty to likelihood if values go negative
    
    // === EQUATION 1: Temperature effect on COTS recruitment ===
    // Gaussian function centered at optimal temperature
    Type temp_deviation = sst_curr - temp_opt;                    // Deviation from optimal temperature
    Type temp_effect = exp(-temp_sensitivity * temp_deviation * temp_deviation); // Temperature multiplier on recruitment (0-1)
    
    // === EQUATION 2: Allee effect on COTS recruitment ===
    // Sigmoid function creating threshold dynamics for outbreak initiation
    Type allee_effect = pow(cots_prev, allee_strength) / (pow(allee_threshold, allee_strength) + pow(cots_prev, allee_strength) + eps); // Allee multiplier (0-1)
    
    // === EQUATION 3: Total coral availability ===
    Type total_coral = fast_prev + slow_prev + eps;               // Total coral cover available for COTS feeding
    
    // === EQUATION 4: Dynamic carrying capacity for COTS ===
    // Carrying capacity depends on available coral prey
    Type K_cots = K_cots_base * (total_coral / (K_coral_total + eps)) * conversion_efficiency; // Dynamic carrying capacity based on coral availability
    K_cots = posfun(K_cots, Type(0.0), pen);                     // Ensure positive carrying capacity
    nll += Type(1000.0) * pen;                                   // Add penalty if K_cots goes negative
    
    // === EQUATION 5: Type II functional response for fast-growing coral predation ===
    Type predation_fast = (attack_rate_fast * fast_prev * cots_prev) / (Type(1.0) + attack_rate_fast * handling_time_fast * fast_prev + eps); // Predation rate on fast coral
    
    // === EQUATION 6: Type II functional response for slow-growing coral predation ===
    Type predation_slow = (attack_rate_slow * slow_prev * cots_prev) / (Type(1.0) + attack_rate_slow * handling_time_slow * slow_prev + eps); // Predation rate on slow coral
    
    // === EQUATION 7: COTS population growth ===
    // Logistic growth with Allee effect, temperature effect, and immigration
    Type cots_growth = r_cots * cots_prev * (Type(1.0) - cots_prev / (K_cots + eps)) * allee_effect * temp_effect; // Density-dependent growth with environmental modifiers
    Type cots_immigration = immigration_effect * immigration_curr; // Immigration contribution to population
    Type cots_change = cots_growth + cots_immigration;            // Total change in COTS abundance
    cots_pred(t) = posfun(cots_prev + cots_change, Type(0.0), pen); // Update COTS abundance (ensure positive)
    nll += Type(1000.0) * pen;                                   // Add penalty if COTS goes negative
    
    // === EQUATION 8: Fast-growing coral dynamics ===
    // Logistic growth with competition and COTS predation
    Type space_occupied_fast = fast_prev + competition_fast * slow_prev; // Effective space occupied by fast coral including competition
    Type fast_growth = r_fast * fast_prev * (Type(1.0) - space_occupied_fast / (K_coral_total + eps)); // Logistic growth with competition
    Type fast_loss = predation_fast;                              // Loss due to COTS predation
    Type fast_change = fast_growth - fast_loss;                   // Net change in fast coral cover
    fast_pred(t) = posfun(fast_prev + fast_change, Type(0.0), pen); // Update fast coral cover (ensure positive)
    nll += Type(1000.0) * pen;                                   // Add penalty if fast coral goes negative
    
    // === EQUATION 9: Slow-growing coral dynamics ===
    // Logistic growth with competition and COTS predation
    Type space_occupied_slow = slow_prev + competition_slow * fast_prev; // Effective space occupied by slow coral including competition
    Type slow_growth = r_slow * slow_prev * (Type(1.0) - space_occupied_slow / (K_coral_total + eps)); // Logistic growth with competition
    Type slow_loss = predation_slow;                              // Loss due to COTS predation
    Type slow_change = slow_growth - slow_loss;                   // Net change in slow coral cover
    slow_pred(t) = posfun(slow_prev + slow_change, Type(0.0), pen); // Update slow coral cover (ensure positive)
    nll += Type(1000.0) * pen;                                   // Add penalty if slow coral goes negative
  }
  
  // === LIKELIHOOD CALCULATION ===
  // Using lognormal distribution for all variables (appropriate for positive, skewed data)
  
  // Apply minimum standard deviation constraint
  Type sigma_cots_safe = sigma_cots + min_sd;                   // Safe COTS observation error (add min_sd to ensure minimum)
  Type sigma_fast_safe = sigma_fast + min_sd;                   // Safe fast coral observation error
  Type sigma_slow_safe = sigma_slow + min_sd;                   // Safe slow coral observation error
  
  for(int t = 0; t < n; t++) {
    // Ensure predictions and observations are positive
    Type pen = Type(0.0);                                         // Penalty accumulator for this time step
    Type cots_pred_safe = posfun(cots_pred(t), eps, pen);        // Safe predicted COTS value
    Type fast_pred_safe = posfun(fast_pred(t), eps, pen);        // Safe predicted fast coral value
    Type slow_pred_safe = posfun(slow_pred(t), eps, pen);        // Safe predicted slow coral value
    Type cots_obs_safe = posfun(cots_dat(t), eps, pen);          // Safe observed COTS value
    Type fast_obs_safe = posfun(fast_dat(t), eps, pen);          // Safe observed fast coral value
    Type slow_obs_safe = posfun(slow_dat(t), eps, pen);          // Safe observed slow coral value
    nll += Type(1000.0) * pen;                                   // Add penalty if any values too small
    
    // === EQUATION 10: COTS likelihood ===
    // Lognormal likelihood for COTS abundance
    nll -= dnorm(log(cots_obs_safe), log(cots_pred_safe), sigma_cots_safe, true); // Negative log-likelihood for COTS
    
    // === EQUATION 11: Fast-growing coral likelihood ===
    // Lognormal likelihood for fast coral cover
    nll -= dnorm(log(fast_obs_safe), log(fast_pred_safe), sigma_fast_safe, true); // Negative log-likelihood for fast coral
    
    // === EQUATION 12: Slow-growing coral likelihood ===
    // Lognormal likelihood for slow coral cover
    nll -= dnorm(log(slow_obs_safe), log(slow_pred_safe), sigma_slow_safe, true); // Negative log-likelihood for slow coral
  }
  
  // Report predicted values
  REPORT(cots_pred);                    // Report predicted COTS abundance
  REPORT(fast_pred);                    // Report predicted fast-growing coral cover
  REPORT(slow_pred);                    // Report predicted slow-growing coral cover
  
  return nll;                           // Return total negative log-likelihood
}
