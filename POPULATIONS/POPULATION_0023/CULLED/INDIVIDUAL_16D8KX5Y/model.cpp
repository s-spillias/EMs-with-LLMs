#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS
  DATA_VECTOR(Year);                    // Time vector (years)
  DATA_VECTOR(cots_dat);               // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);               // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);               // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);                // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);            // COTS larval immigration rate (individuals/m2/year)
  
  // COTS POPULATION PARAMETERS
  PARAMETER(log_r_cots);                // Log intrinsic growth rate of COTS (year^-1)
  PARAMETER(log_K_cots);                // Log carrying capacity for COTS (individuals/m2)
  PARAMETER(log_allee_threshold);       // Log Allee threshold density (individuals/m2)
  PARAMETER(log_mort_cots);             // Log baseline mortality rate of COTS (year^-1)
  PARAMETER(log_dd_mort);               // Log density-dependent mortality coefficient (m2/individuals/year)
  
  // TEMPERATURE EFFECTS ON COTS
  PARAMETER(temp_opt);                  // Optimal temperature for COTS recruitment (Celsius)
  PARAMETER(log_temp_width);            // Log temperature tolerance width (Celsius)
  PARAMETER(log_temp_effect);           // Log maximum temperature effect multiplier (dimensionless)
  
  // LARVAL SURVIVAL ENHANCEMENT
  PARAMETER(log_larval_survival_boost); // Log larval survival multiplier for nutrient-enhanced conditions (dimensionless)
  
  // CORAL DYNAMICS PARAMETERS
  PARAMETER(log_r_fast);                // Log growth rate of fast-growing coral (year^-1)
  PARAMETER(log_r_slow);                // Log growth rate of slow-growing coral (year^-1)
  PARAMETER(log_K_coral);               // Log total coral carrying capacity (%)
  PARAMETER(log_comp_fast);             // Log competition coefficient for fast coral (dimensionless)
  PARAMETER(log_comp_slow);             // Log competition coefficient for slow coral (dimensionless)
  
  // PREDATION PARAMETERS
  PARAMETER(log_attack_fast);           // Log attack rate on fast-growing coral (m2/individuals/year)
  PARAMETER(log_attack_slow);           // Log attack rate on slow-growing coral (m2/individuals/year)
  PARAMETER(log_handling_fast);         // Log handling time for fast coral (year)
  PARAMETER(log_handling_slow);         // Log handling time for slow coral (year)
  PARAMETER(log_preference);            // Log preference for fast vs slow coral (dimensionless, >0 prefers fast)
  
  // OBSERVATION ERROR PARAMETERS
  PARAMETER(log_sigma_cots);            // Log observation error SD for COTS (individuals/m2)
  PARAMETER(log_sigma_fast);            // Log observation error SD for fast coral (%)
  PARAMETER(log_sigma_slow);            // Log observation error SD for slow coral (%)
  
  // TRANSFORM PARAMETERS
  Type r_cots = exp(log_r_cots);                      // Intrinsic growth rate of COTS (year^-1)
  Type K_cots = exp(log_K_cots);                      // Carrying capacity for COTS (individuals/m2)
  Type allee_threshold = exp(log_allee_threshold);    // Allee threshold (individuals/m2)
  Type mort_cots = exp(log_mort_cots);                // Baseline mortality (year^-1)
  Type dd_mort = exp(log_dd_mort);                    // Density-dependent mortality coefficient (m2/individuals/year)
  Type temp_width = exp(log_temp_width);              // Temperature tolerance (Celsius)
  Type temp_effect = exp(log_temp_effect);            // Temperature effect multiplier (dimensionless)
  Type larval_survival_boost = exp(log_larval_survival_boost); // Larval survival multiplier (dimensionless)
  Type r_fast = exp(log_r_fast);                      // Fast coral growth rate (year^-1)
  Type r_slow = exp(log_r_slow);                      // Slow coral growth rate (year^-1)
  Type K_coral = exp(log_K_coral);                    // Coral carrying capacity (%)
  Type comp_fast = exp(log_comp_fast);                // Fast coral competition coefficient (dimensionless)
  Type comp_slow = exp(log_comp_slow);                // Slow coral competition coefficient (dimensionless)
  Type attack_fast = exp(log_attack_fast);            // Attack rate on fast coral (m2/individuals/year)
  Type attack_slow = exp(log_attack_slow);            // Attack rate on slow coral (m2/individuals/year)
  Type handling_fast = exp(log_handling_fast);        // Handling time fast coral (year)
  Type handling_slow = exp(log_handling_slow);        // Handling time slow coral (year)
  Type preference = exp(log_preference);              // Preference coefficient (dimensionless)
  Type sigma_cots = exp(log_sigma_cots);              // Observation error COTS (individuals/m2)
  Type sigma_fast = exp(log_sigma_fast);              // Observation error fast coral (%)
  Type sigma_slow = exp(log_sigma_slow);              // Observation error slow coral (%)
  
  // MINIMUM STANDARD DEVIATIONS FOR NUMERICAL STABILITY
  Type min_sigma = Type(0.01);                        // Minimum SD to prevent numerical issues
  sigma_cots = sigma_cots + min_sigma;                // Add minimum to COTS SD
  sigma_fast = sigma_fast + min_sigma;                // Add minimum to fast coral SD
  sigma_slow = sigma_slow + min_sigma;                // Add minimum to slow coral SD
  
  // INITIALIZE PREDICTION VECTORS
  int n = cots_dat.size();                            // Number of time steps
  vector<Type> cots_pred(n);                          // Predicted COTS abundance (individuals/m2)
  vector<Type> fast_pred(n);                          // Predicted fast coral cover (%)
  vector<Type> slow_pred(n);                          // Predicted slow coral cover (%)
  
  // SET INITIAL CONDITIONS FROM DATA
  cots_pred(0) = cots_dat(0);                         // Initialize COTS from first observation
  fast_pred(0) = fast_dat(0);                         // Initialize fast coral from first observation
  slow_pred(0) = slow_dat(0);                         // Initialize slow coral from first observation
  
  // SMALL CONSTANT FOR NUMERICAL STABILITY
  Type eps = Type(1e-8);                              // Small constant to prevent division by zero
  
  // INITIALIZE NEGATIVE LOG-LIKELIHOOD
  Type nll = Type(0.0);                               // Negative log-likelihood accumulator
  
  // TIME LOOP - PROCESS MODEL
  for(int t = 1; t < n; t++) {
    
    // EQUATION 1: Temperature effect on COTS recruitment
    // Gaussian function centered at optimal temperature
    Type temp_deviation = sst_dat(t-1) - temp_opt;                                    // Deviation from optimal temperature (Celsius)
    Type temp_factor = Type(1.0) + temp_effect * exp(-0.5 * pow(temp_deviation / (temp_width + eps), 2)); // Temperature multiplier on recruitment (dimensionless)
    
    // EQUATION 2: Allee effect on COTS recruitment
    // Positive density dependence at low densities, negative at high densities
    Type allee_factor = cots_pred(t-1) / (allee_threshold + cots_pred(t-1) + eps);   // Allee effect (0 to 1, dimensionless)
    
    // EQUATION 3: Total coral availability for COTS
    Type total_coral = fast_pred(t-1) + slow_pred(t-1) + eps;                        // Total coral cover (%)
    
    // EQUATION 4: Preference-weighted coral consumption (Type II functional response)
    // Fast coral consumption with preference weighting
    Type num_fast = attack_fast * preference * fast_pred(t-1);                       // Numerator for fast coral consumption (m2*%/year)
    Type denom_fast = Type(1.0) + handling_fast * attack_fast * preference * fast_pred(t-1) + handling_slow * attack_slow * slow_pred(t-1) + eps; // Denominator for functional response (dimensionless)
    Type consumption_fast = num_fast / denom_fast;                                    // Per capita fast coral consumption rate (%/individuals/year)
    
    // Slow coral consumption
    Type num_slow = attack_slow * slow_pred(t-1);                                    // Numerator for slow coral consumption (m2*%/year)
    Type denom_slow = Type(1.0) + handling_fast * attack_fast * preference * fast_pred(t-1) + handling_slow * attack_slow * slow_pred(t-1) + eps; // Denominator for functional response (dimensionless)
    Type consumption_slow = num_slow / denom_slow;                                    // Per capita slow coral consumption rate (%/individuals/year)
    
    // EQUATION 5: COTS population growth with nutrient-enhanced larval survival
    // Combines recruitment (with Allee, temperature, and larval survival effects), immigration, and mortality
    Type recruitment = r_cots * allee_factor * temp_factor * larval_survival_boost * (Type(1.0) - cots_pred(t-1) / (K_cots + eps)); // Recruitment term with larval survival boost (year^-1)
    Type mortality = mort_cots + dd_mort * cots_pred(t-1);                           // Total mortality rate (year^-1)
    Type cots_growth = recruitment - mortality;                                       // Net growth rate (year^-1)
    Type cots_temp = cots_pred(t-1) + cots_pred(t-1) * cots_growth + cotsimm_dat(t-1); // COTS abundance next time step (individuals/m2)
    cots_pred(t) = CppAD::CondExpGt(cots_temp, Type(0.0), cots_temp, Type(0.0));    // Ensure non-negative COTS abundance
    
    // EQUATION 6: Fast-growing coral dynamics
    // Logistic growth with competition and COTS predation
    Type fast_space = Type(1.0) - (comp_fast * fast_pred(t-1) + comp_slow * slow_pred(t-1)) / (K_coral + eps); // Available space for fast coral (dimensionless)
    Type fast_growth = r_fast * fast_pred(t-1) * fast_space;                         // Logistic growth of fast coral (%/year)
    Type fast_loss = consumption_fast * cots_pred(t-1);                              // Loss to COTS predation (%/year)
    Type fast_temp = fast_pred(t-1) + fast_growth - fast_loss;                       // Fast coral cover next time step (%)
    fast_pred(t) = CppAD::CondExpGt(fast_temp, Type(0.0), fast_temp, Type(0.0));    // Ensure non-negative coral cover
    
    // EQUATION 7: Slow-growing coral dynamics
    // Logistic growth with competition and COTS predation
    Type slow_space = Type(1.0) - (comp_fast * fast_pred(t-1) + comp_slow * slow_pred(t-1)) / (K_coral + eps); // Available space for slow coral (dimensionless)
    Type slow_growth = r_slow * slow_pred(t-1) * slow_space;                         // Logistic growth of slow coral (%/year)
    Type slow_loss = consumption_slow * cots_pred(t-1);                              // Loss to COTS predation (%/year)
    Type slow_temp = slow_pred(t-1) + slow_growth - slow_loss;                       // Slow coral cover next time step (%)
    slow_pred(t) = CppAD::CondExpGt(slow_temp, Type(0.0), slow_temp, Type(0.0));    // Ensure non-negative coral cover
  }
  
  // LIKELIHOOD CALCULATION
  // Using normal distribution for all observations (could use lognormal if needed)
  for(int t = 0; t < n; t++) {
    nll -= dnorm(cots_dat(t), cots_pred(t), sigma_cots, true);                       // COTS observation likelihood
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_fast, true);                       // Fast coral observation likelihood
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_slow, true);                       // Slow coral observation likelihood
  }
  
  // SOFT PARAMETER BOUNDS (PENALTIES)
  // Penalize biologically unrealistic parameter values
  Type penalty_r_cots = CppAD::CondExpGt(r_cots, Type(2.0), Type(10.0) * pow(r_cots - Type(2.0), 2), Type(0.0)); // Penalty if COTS growth rate too high
  Type penalty_mort = CppAD::CondExpGt(mort_cots, Type(5.0), Type(10.0) * pow(mort_cots - Type(5.0), 2), Type(0.0)); // Penalty if baseline mortality unrealistically high
  Type penalty_K_cots = CppAD::CondExpGt(K_cots, Type(10.0), Type(10.0) * pow(K_cots - Type(10.0), 2), Type(0.0)); // Penalty if COTS carrying capacity unrealistic
  Type temp_low_penalty = CppAD::CondExpLt(temp_opt, Type(20.0), Type(10.0) * pow(Type(20.0) - temp_opt, 2), Type(0.0)); // Penalty if temp too low
  Type temp_high_penalty = CppAD::CondExpGt(temp_opt, Type(32.0), Type(10.0) * pow(temp_opt - Type(32.0), 2), Type(0.0)); // Penalty if temp too high
  Type penalty_K_coral = CppAD::CondExpGt(K_coral, Type(100.0), Type(10.0) * pow(K_coral - Type(100.0), 2), Type(0.0)); // Penalty if coral carrying capacity exceeds 100%
  Type penalty_r_fast = CppAD::CondExpGt(r_fast, Type(0.5), Type(10.0) * pow(r_fast - Type(0.5), 2), Type(0.0)); // Penalty if fast coral growth unrealistic
  Type penalty_r_slow = CppAD::CondExpGt(r_slow, Type(0.2), Type(10.0) * pow(r_slow - Type(0.2), 2), Type(0.0)); // Penalty if slow coral growth unrealistic
  Type penalty_larval_low = CppAD::CondExpLt(larval_survival_boost, Type(0.01), Type(10.0) * pow(Type(0.01) - larval_survival_boost, 2), Type(0.0)); // Penalty if larval survival too low
  Type penalty_larval_high = CppAD::CondExpGt(larval_survival_boost, Type(100.0), Type(10.0) * pow(larval_survival_boost - Type(100.0), 2), Type(0.0)); // Penalty if larval survival unrealistically high
  
  nll += penalty_r_cots + penalty_mort + penalty_K_cots + temp_low_penalty + temp_high_penalty + penalty_K_coral + penalty_r_fast + penalty_r_slow + penalty_larval_low + penalty_larval_high; // Add all penalties to negative log-likelihood
  
  // REPORT PREDICTIONS AND PARAMETERS
  REPORT(cots_pred);                                   // Report predicted COTS abundance
  REPORT(fast_pred);                                   // Report predicted fast coral cover
  REPORT(slow_pred);                                   // Report predicted slow coral cover
  REPORT(r_cots);                                      // Report COTS growth rate
  REPORT(K_cots);                                      // Report COTS carrying capacity
  REPORT(allee_threshold);                             // Report Allee threshold
  REPORT(mort_cots);                                   // Report baseline mortality
  REPORT(dd_mort);                                     // Report density-dependent mortality
  REPORT(temp_opt);                                    // Report optimal temperature
  REPORT(temp_width);                                  // Report temperature tolerance
  REPORT(temp_effect);                                 // Report temperature effect magnitude
  REPORT(larval_survival_boost);                       // Report larval survival multiplier
  REPORT(r_fast);                                      // Report fast coral growth rate
  REPORT(r_slow);                                      // Report slow coral growth rate
  REPORT(K_coral);                                     // Report coral carrying capacity
  REPORT(comp_fast);                                   // Report fast coral competition
  REPORT(comp_slow);                                   // Report slow coral competition
  REPORT(attack_fast);                                 // Report attack rate on fast coral
  REPORT(attack_slow);                                 // Report attack rate on slow coral
  REPORT(handling_fast);                               // Report handling time fast coral
  REPORT(handling_slow);                               // Report handling time slow coral
  REPORT(preference);                                  // Report preference coefficient
  REPORT(sigma_cots);                                  // Report COTS observation error
  REPORT(sigma_fast);                                  // Report fast coral observation error
  REPORT(sigma_slow);                                  // Report slow coral observation error
  
  return nll;                                          // Return negative log-likelihood
}
