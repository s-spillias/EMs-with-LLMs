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
  PARAMETER(pref_threshold);          // Threshold of fast-growing coral cover (%) below which COTS begin switching to slow-growing coral
  PARAMETER(pref_steepness);          // Steepness of the preference switching response (dimensionless)
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-6);
  
  // Number of time steps
  int n_years = Year.size();
  
  // Vectors to store model predictions
  vector<Type> cots_pred(n_years);
  vector<Type> fast_pred(n_years);
  vector<Type> slow_pred(n_years);
  
  // Initialize with first year's observed values
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Ensure positive initial values
  if (cots_pred(0) < eps) cots_pred(0) = eps;
  if (fast_pred(0) < eps) fast_pred(0) = eps;
  if (slow_pred(0) < eps) slow_pred(0) = eps;
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(0.1);
  Type sigma_cots_adj = sigma_cots + min_sigma;
  Type sigma_fast_adj = sigma_fast + min_sigma;
  Type sigma_slow_adj = sigma_slow + min_sigma;
  
  // Ensure parameters are within reasonable bounds
  Type r_cots_bounded = r_cots;
  if (r_cots_bounded < Type(0.1)) r_cots_bounded = Type(0.1);
  if (r_cots_bounded > Type(10.0)) r_cots_bounded = Type(10.0);
  
  Type K_cots_bounded = K_cots;
  if (K_cots_bounded < Type(0.1)) K_cots_bounded = Type(0.1);
  
  Type m_cots_bounded = m_cots;
  if (m_cots_bounded < Type(0.01)) m_cots_bounded = Type(0.01);
  
  // Time series simulation
  for (int t = 1; t < n_years; t++) {
    // Get previous values with safety checks
    Type cots_prev = cots_pred(t-1);
    if (cots_prev < eps) cots_prev = eps;
    
    Type fast_prev = fast_pred(t-1);
    if (fast_prev < eps) fast_prev = eps;
    
    Type slow_prev = slow_pred(t-1);
    if (slow_prev < eps) slow_prev = eps;
    
    // 1. Temperature effect on COTS reproduction (smooth transition around threshold)
    Type temp_diff = sst_dat(t-1) - T_crit;
    Type temp_effect = Type(1.0) + T_effect / (Type(1.0) + exp(-Type(0.5) * temp_diff));
    
    // 2. Calculate preference switching based on fast-growing coral abundance
    Type pref_diff = fast_prev - pref_threshold;
    Type pref_fast = Type(1.0) / (Type(1.0) + exp(-pref_steepness * pref_diff));
    Type pref_slow = Type(1.0) - pref_fast;
    
    // 3. Modify attack rates based on preference
    Type effective_a_fast = a_fast * pref_fast;
    Type effective_a_slow = a_slow * (Type(1.0) + pref_slow * Type(0.5)); // Increase attack on slow coral when preferred coral is scarce
    
    // 4. Type II functional response for COTS predation on fast-growing coral with preference
    Type denominator = Type(1.0) + effective_a_fast * h_fast * fast_prev + effective_a_slow * h_slow * slow_prev;
    if (denominator < eps) denominator = eps;
    
    Type consumption_fast = (effective_a_fast * fast_prev * cots_prev) / denominator;
    // Ensure consumption doesn't exceed available coral
    if (consumption_fast > fast_prev * Type(0.9)) consumption_fast = fast_prev * Type(0.9);
    
    // 5. Type II functional response for COTS predation on slow-growing coral with preference
    Type consumption_slow = (effective_a_slow * slow_prev * cots_prev) / denominator;
    // Ensure consumption doesn't exceed available coral
    if (consumption_slow > slow_prev * Type(0.9)) consumption_slow = slow_prev * Type(0.9);
    
    // 6. COTS population dynamics with temperature effect, density dependence, and immigration
    Type density_factor = Type(1.0) - cots_prev / K_cots_bounded;
    if (density_factor < Type(0.0)) density_factor = Type(0.0);
    Type cots_growth = r_cots_bounded * temp_effect * cots_prev * density_factor;
    
    // 7. Add starvation effect: COTS mortality increases when both coral types are scarce
    Type resource_availability = (fast_prev / K_fast) + (slow_prev / K_slow);
    if (resource_availability < Type(0.0)) resource_availability = Type(0.0);
    if (resource_availability > Type(1.0)) resource_availability = Type(1.0);
    
    Type starvation_effect = Type(1.0) + Type(0.5) * (Type(1.0) - resource_availability);
    
    Type cots_mortality = m_cots_bounded * starvation_effect * cots_prev;
    Type cots_immigration = imm_effect * cotsimm_dat(t-1);
    
    // 8. Update COTS population with bounds to prevent negative or extreme values
    cots_pred(t) = cots_prev + cots_growth - cots_mortality + cots_immigration;
    
    // Apply bounds to COTS population
    if (cots_pred(t) < eps) cots_pred(t) = eps;
    if (cots_pred(t) > K_cots_bounded * Type(2.0)) cots_pred(t) = K_cots_bounded * Type(2.0);
    
    // 9. Fast-growing coral dynamics with competition and predation
    Type competition_fast = Type(1.0) - (fast_prev + alpha_fs * slow_prev) / K_fast;
    if (competition_fast < Type(-1.0)) competition_fast = Type(-1.0);
    Type fast_growth = r_fast * fast_prev * competition_fast;
    
    // 10. Update fast-growing coral with bounds
    fast_pred(t) = fast_prev + fast_growth - consumption_fast;
    
    // Apply bounds to fast-growing coral
    if (fast_pred(t) < eps) fast_pred(t) = eps;
    if (fast_pred(t) > K_fast) fast_pred(t) = K_fast;
    
    // 11. Slow-growing coral dynamics with competition and predation
    Type competition_slow = Type(1.0) - (slow_prev + alpha_sf * fast_prev) / K_slow;
    if (competition_slow < Type(-1.0)) competition_slow = Type(-1.0);
    Type slow_growth = r_slow * slow_prev * competition_slow;
    
    // 12. Update slow-growing coral with bounds
    slow_pred(t) = slow_prev + slow_growth - consumption_slow;
    
    // Apply bounds to slow-growing coral
    if (slow_pred(t) < eps) slow_pred(t) = eps;
    if (slow_pred(t) > K_slow) slow_pred(t) = K_slow;
  }
  
  // Calculate negative log-likelihood using normal distribution on log scale
  for (int t = 0; t < n_years; t++) {
    // 13. Add observation error for COTS abundance
    if (!R_IsNA(asDouble(cots_dat(t))) && cots_dat(t) > Type(0.0)) {
      Type log_cots_obs = log(cots_dat(t) + eps);
      Type log_cots_pred = log(cots_pred(t) + eps);
      nll -= dnorm(log_cots_obs, log_cots_pred, sigma_cots_adj, true);
    }
    
    // 14. Add observation error for fast-growing coral cover
    if (!R_IsNA(asDouble(fast_dat(t))) && fast_dat(t) > Type(0.0)) {
      Type log_fast_obs = log(fast_dat(t) + eps);
      Type log_fast_pred = log(fast_pred(t) + eps);
      nll -= dnorm(log_fast_obs, log_fast_pred, sigma_fast_adj, true);
    }
    
    // 15. Add observation error for slow-growing coral cover
    if (!R_IsNA(asDouble(slow_dat(t))) && slow_dat(t) > Type(0.0)) {
      Type log_slow_obs = log(slow_dat(t) + eps);
      Type log_slow_pred = log(slow_pred(t) + eps);
      nll -= dnorm(log_slow_obs, log_slow_pred, sigma_slow_adj, true);
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
  vector<Type> pref_fast(n_years);
  vector<Type> pref_slow(n_years);
  vector<Type> starvation_effect(n_years);
  
  for (int t = 0; t < n_years; t++) {
    // Calculate temperature effect for each year
    Type temp_diff = sst_dat(t) - T_crit;
    temp_effect(t) = Type(1.0) + T_effect / (Type(1.0) + exp(-Type(0.5) * temp_diff));
    
    // Calculate preference switching
    if (t > 0) {
      Type cots_prev = cots_pred(t-1);
      if (cots_prev < eps) cots_prev = eps;
      
      Type fast_prev = fast_pred(t-1);
      if (fast_prev < eps) fast_prev = eps;
      
      Type slow_prev = slow_pred(t-1);
      if (slow_prev < eps) slow_prev = eps;
      
      Type pref_diff = fast_prev - pref_threshold;
      pref_fast(t) = Type(1.0) / (Type(1.0) + exp(-pref_steepness * pref_diff));
      pref_slow(t) = Type(1.0) - pref_fast(t);
      
      // Calculate effective attack rates
      Type effective_a_fast = a_fast * pref_fast(t);
      Type effective_a_slow = a_slow * (Type(1.0) + pref_slow(t) * Type(0.5));
      
      // Calculate consumption rates for each year
      Type denominator = Type(1.0) + effective_a_fast * h_fast * fast_prev + effective_a_slow * h_slow * slow_prev;
      if (denominator < eps) denominator = eps;
      
      consumption_fast(t) = (effective_a_fast * fast_prev * cots_prev) / denominator;
      if (consumption_fast(t) > fast_prev * Type(0.9)) consumption_fast(t) = fast_prev * Type(0.9);
      
      consumption_slow(t) = (effective_a_slow * slow_prev * cots_prev) / denominator;
      if (consumption_slow(t) > slow_prev * Type(0.9)) consumption_slow(t) = slow_prev * Type(0.9);
      
      // Calculate starvation effect
      Type resource_availability = (fast_prev / K_fast) + (slow_prev / K_slow);
      if (resource_availability < Type(0.0)) resource_availability = Type(0.0);
      if (resource_availability > Type(1.0)) resource_availability = Type(1.0);
      
      starvation_effect(t) = Type(1.0) + Type(0.5) * (Type(1.0) - resource_availability);
    } else {
      Type fast_val = fast_pred(t);
      if (fast_val < eps) fast_val = eps;
      
      Type pref_diff = fast_val - pref_threshold;
      pref_fast(t) = Type(1.0) / (Type(1.0) + exp(-pref_steepness * pref_diff));
      pref_slow(t) = Type(1.0) - pref_fast(t);
      consumption_fast(t) = Type(0.0);
      consumption_slow(t) = Type(0.0);
      starvation_effect(t) = Type(1.0);
    }
  }
  
  REPORT(temp_effect);
  REPORT(consumption_fast);
  REPORT(consumption_slow);
  REPORT(pref_fast);
  REPORT(pref_slow);
  REPORT(starvation_effect);
  
  return nll;
}
