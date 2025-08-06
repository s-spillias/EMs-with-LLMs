#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA SECTION
  DATA_VECTOR(Year);                  // Vector of years
  DATA_VECTOR(sst_dat);               // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);           // COTS larval immigration rate (individuals/m^2/year)
  DATA_VECTOR(cots_dat);              // Observed COTS abundance (individuals/m^2)
  DATA_VECTOR(fast_dat);              // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);              // Observed slow-growing coral cover (%)
  
  // PARAMETER SECTION
  PARAMETER(r_cots);                  // Maximum per capita reproduction rate of COTS (year^-1)
  PARAMETER(K_cots);                  // Carrying capacity of COTS (individuals/m^2)
  PARAMETER(m_cots);                  // Natural mortality rate of COTS (year^-1)
  PARAMETER(T_crit);                  // Critical temperature threshold for enhanced COTS reproduction (°C)
  PARAMETER(T_effect);                // Effect size of temperature on COTS reproduction (dimensionless)
  PARAMETER(a_fast);                  // Attack rate on fast-growing coral (m^2/individual/year)
  PARAMETER(a_slow);                  // Attack rate on slow-growing coral (m^2/individual/year)
  PARAMETER(h_fast);                  // Handling time for fast-growing coral (year/% cover)
  PARAMETER(h_slow);                  // Handling time for slow-growing coral (year/% cover)
  PARAMETER(r_fast);                  // Maximum growth rate of fast-growing coral (year^-1)
  PARAMETER(r_slow);                  // Maximum growth rate of slow-growing coral (year^-1)
  PARAMETER(K_fast);                  // Carrying capacity of fast-growing coral (% cover)
  PARAMETER(K_slow);                  // Carrying capacity of slow-growing coral (% cover)
  PARAMETER(alpha_fs);                // Competition coefficient of slow-growing on fast-growing coral (dimensionless)
  PARAMETER(alpha_sf);                // Competition coefficient of fast-growing on slow-growing coral (dimensionless)
  PARAMETER(imm_effect);              // Effect size of larval immigration on COTS population (dimensionless)
  PARAMETER(sigma_cots);              // Observation error standard deviation for COTS abundance (log scale)
  PARAMETER(sigma_fast);              // Observation error standard deviation for fast-growing coral cover (log scale)
  PARAMETER(sigma_slow);              // Observation error standard deviation for slow-growing coral cover (log scale)
  
  // New parameters for improved COTS outbreak dynamics
  PARAMETER(cots_threshold);          // Population threshold for COTS outbreak dynamics (individuals/m^2)
  PARAMETER(outbreak_steepness);      // Steepness of transition to outbreak dynamics (dimensionless)
  PARAMETER(pred_enhancement);        // Enhancement factor for predation during outbreaks (dimensionless)
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Number of time steps
  int n_years = Year.size();
  
  // Vectors to store model predictions
  vector<Type> cots_pred(n_years);
  vector<Type> fast_pred(n_years);
  vector<Type> slow_pred(n_years);
  
  // Initialize with first year's observed values (with safety checks)
  cots_pred(0) = Type(0.5);  // Start with a reasonable value instead of observed
  fast_pred(0) = Type(30.0); // Start with a reasonable value instead of observed
  slow_pred(0) = Type(20.0); // Start with a reasonable value instead of observed
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(0.01);
  Type sigma_cots_adj = sigma_cots + min_sigma;
  Type sigma_fast_adj = sigma_fast + min_sigma;
  Type sigma_slow_adj = sigma_slow + min_sigma;
  
  // Time series simulation
  for (int t = 1; t < n_years; t++) {
    // Temperature effect on COTS reproduction (simple linear effect)
    Type temp_effect = Type(1.0);
    if (sst_dat(t-1) > T_crit) {
      temp_effect = Type(1.0) + T_effect;
    }
    
    // Calculate outbreak state (simple threshold function)
    Type outbreak_state = Type(0.0);
    if (cots_pred(t-1) > cots_threshold) {
      outbreak_state = Type(1.0);
    }
    
    // Enhanced attack rates during outbreaks
    Type a_fast_effective = a_fast;
    Type a_slow_effective = a_slow;
    if (outbreak_state > Type(0.5)) {
      a_fast_effective = a_fast * pred_enhancement;
      a_slow_effective = a_slow * pred_enhancement;
    }
    
    // Type II functional response for COTS predation
    Type denominator = Type(1.0) + a_fast_effective * h_fast * fast_pred(t-1) + a_slow_effective * h_slow * slow_pred(t-1);
    if (denominator < eps) denominator = eps;
    
    Type consumption_fast = (a_fast_effective * fast_pred(t-1) * cots_pred(t-1)) / denominator;
    Type consumption_slow = (a_slow_effective * slow_pred(t-1) * cots_pred(t-1)) / denominator;
    
    // Ensure consumption doesn't exceed available coral
    if (consumption_fast > fast_pred(t-1)) consumption_fast = fast_pred(t-1);
    if (consumption_slow > slow_pred(t-1)) consumption_slow = slow_pred(t-1);
    
    // COTS population dynamics
    Type cots_growth = r_cots * temp_effect * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1) / K_cots);
    
    // Apply outbreak dynamics - higher growth during outbreaks
    if (outbreak_state > Type(0.5)) {
      cots_growth = cots_growth * Type(2.0);
    }
    
    Type cots_mortality = m_cots * cots_pred(t-1);
    Type cots_immigration = imm_effect * cotsimm_dat(t-1);
    
    // Update COTS population
    cots_pred(t) = cots_pred(t-1) + cots_growth - cots_mortality + cots_immigration;
    if (cots_pred(t) < eps) cots_pred(t) = eps;
    if (cots_pred(t) > K_cots * Type(2.0)) cots_pred(t) = K_cots * Type(2.0);
    
    // Fast-growing coral dynamics
    Type fast_growth = r_fast * fast_pred(t-1) * (Type(1.0) - (fast_pred(t-1) + alpha_fs * slow_pred(t-1)) / K_fast);
    
    // Update fast-growing coral
    fast_pred(t) = fast_pred(t-1) + fast_growth - consumption_fast;
    if (fast_pred(t) < eps) fast_pred(t) = eps;
    if (fast_pred(t) > K_fast) fast_pred(t) = K_fast;
    
    // Slow-growing coral dynamics
    Type slow_growth = r_slow * slow_pred(t-1) * (Type(1.0) - (slow_pred(t-1) + alpha_sf * fast_pred(t-1)) / K_slow);
    
    // Update slow-growing coral
    slow_pred(t) = slow_pred(t-1) + slow_growth - consumption_slow;
    if (slow_pred(t) < eps) slow_pred(t) = eps;
    if (slow_pred(t) > K_slow) slow_pred(t) = K_slow;
  }
  
  // Calculate negative log-likelihood using normal distribution on log scale
  for (int t = 0; t < n_years; t++) {
    // Add observation error for COTS abundance
    if (!R_IsNA(asDouble(cots_dat(t))) && cots_dat(t) > eps) {
      nll -= dnorm(log(cots_dat(t)), log(cots_pred(t)), sigma_cots_adj, true);
    }
    
    // Add observation error for fast-growing coral cover
    if (!R_IsNA(asDouble(fast_dat(t))) && fast_dat(t) > eps) {
      nll -= dnorm(log(fast_dat(t)), log(fast_pred(t)), sigma_fast_adj, true);
    }
    
    // Add observation error for slow-growing coral cover
    if (!R_IsNA(asDouble(slow_dat(t))) && slow_dat(t) > eps) {
      nll -= dnorm(log(slow_dat(t)), log(slow_pred(t)), sigma_slow_adj, true);
    }
  }
  
  // Report model predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  // Additional derived quantities for reporting
  vector<Type> temp_effect(n_years);
  vector<Type> consumption_fast(n_years);
  vector<Type> consumption_slow(n_years);
  vector<Type> outbreak_state(n_years);
  vector<Type> a_fast_effective(n_years);
  vector<Type> a_slow_effective(n_years);
  
  for (int t = 0; t < n_years; t++) {
    // Calculate temperature effect for each year
    temp_effect(t) = Type(1.0);
    if (sst_dat(t) > T_crit) {
      temp_effect(t) = Type(1.0) + T_effect;
    }
    
    // Calculate outbreak state for each year
    outbreak_state(t) = Type(0.0);
    if (cots_pred(t) > cots_threshold) {
      outbreak_state(t) = Type(1.0);
    }
    
    // Calculate effective attack rates
    a_fast_effective(t) = a_fast;
    a_slow_effective(t) = a_slow;
    if (outbreak_state(t) > Type(0.5)) {
      a_fast_effective(t) = a_fast * pred_enhancement;
      a_slow_effective(t) = a_slow * pred_enhancement;
    }
    
    // Calculate consumption rates for each year
    if (t > 0) {
      Type denominator = Type(1.0) + a_fast_effective(t-1) * h_fast * fast_pred(t-1) + a_slow_effective(t-1) * h_slow * slow_pred(t-1);
      if (denominator < eps) denominator = eps;
      
      consumption_fast(t) = (a_fast_effective(t-1) * fast_pred(t-1) * cots_pred(t-1)) / denominator;
      consumption_slow(t) = (a_slow_effective(t-1) * slow_pred(t-1) * cots_pred(t-1)) / denominator;
      
      if (consumption_fast(t) > fast_pred(t-1)) consumption_fast(t) = fast_pred(t-1);
      if (consumption_slow(t) > slow_pred(t-1)) consumption_slow(t) = slow_pred(t-1);
    } else {
      consumption_fast(t) = Type(0.0);
      consumption_slow(t) = Type(0.0);
    }
  }
  
  REPORT(temp_effect);
  REPORT(consumption_fast);
  REPORT(consumption_slow);
  REPORT(outbreak_state);
  REPORT(a_fast_effective);
  REPORT(a_slow_effective);
  
  return nll;
}
