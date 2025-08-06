#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA SECTION
  DATA_VECTOR(Year);                  // Vector of years for time series data
  DATA_VECTOR(sst_dat);               // Sea surface temperature data (Celsius)
  DATA_VECTOR(cotsimm_dat);           // COTS larval immigration rate (individuals/m^2/year)
  DATA_VECTOR(cots_dat);              // Observed COTS abundance (individuals/m^2)
  DATA_VECTOR(fast_dat);              // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);              // Observed slow-growing coral cover (%)
  
  // PARAMETER SECTION
  PARAMETER(r_cots);                  // Intrinsic growth rate of COTS (year^-1)
  PARAMETER(K_cots);                  // Carrying capacity of COTS (individuals/m^2)
  PARAMETER(m_cots);                  // Natural mortality rate of COTS (year^-1)
  PARAMETER(alpha_fast);              // Attack rate on fast-growing coral (m^2/individual/year)
  PARAMETER(alpha_slow);              // Attack rate on slow-growing coral (m^2/individual/year)
  PARAMETER(h_fast);                  // Handling time for fast-growing coral (year/% cover)
  PARAMETER(h_slow);                  // Handling time for slow-growing coral (year/% cover)
  PARAMETER(r_fast);                  // Intrinsic growth rate of fast-growing coral (year^-1)
  PARAMETER(r_slow);                  // Intrinsic growth rate of slow-growing coral (year^-1)
  PARAMETER(K_fast);                  // Carrying capacity of fast-growing coral (% cover)
  PARAMETER(K_slow);                  // Carrying capacity of slow-growing coral (% cover)
  PARAMETER(beta_sst);                // Effect of SST on COTS reproduction (dimensionless)
  PARAMETER(sst_opt);                 // Optimal SST for COTS reproduction (Celsius)
  PARAMETER(sst_width);               // Width parameter for temperature response (Celsius)
  PARAMETER(imm_effect);              // Effect of larval immigration (dimensionless)
  PARAMETER(coral_threshold);         // Coral threshold for COTS mortality (% cover)
  PARAMETER(sigma_cots);              // Observation error SD for COTS (log scale)
  PARAMETER(sigma_fast);              // Observation error SD for fast coral (log scale)
  PARAMETER(sigma_slow);              // Observation error SD for slow coral (log scale)
  
  // New parameters for improved model
  PARAMETER(allee_threshold);         // Population threshold for Allee effect (individuals/m^2)
  PARAMETER(allee_strength);          // Strength of Allee effect (dimensionless)
  PARAMETER(pref_switch_threshold);   // Threshold for switching feeding preference (proportion of K_cots)
  PARAMETER(pref_switch_rate);        // Rate of change in feeding preference (dimensionless)
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize vectors for model predictions
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Set initial values (first year)
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-3);
  
  // Ensure parameters are within reasonable bounds to prevent numerical issues
  Type r_cots_safe = r_cots > 0 ? r_cots : eps;
  Type K_cots_safe = K_cots > 0 ? K_cots : Type(1.0);
  Type m_cots_safe = m_cots > 0 ? m_cots : eps;
  Type alpha_fast_safe = alpha_fast > 0 ? alpha_fast : eps;
  Type alpha_slow_safe = alpha_slow > 0 ? alpha_slow : eps;
  Type h_fast_safe = h_fast > 0 ? h_fast : eps;
  Type h_slow_safe = h_slow > 0 ? h_slow : eps;
  Type r_fast_safe = r_fast > 0 ? r_fast : eps;
  Type r_slow_safe = r_slow > 0 ? r_slow : eps;
  Type K_fast_safe = K_fast > 0 ? K_fast : Type(10.0);
  Type K_slow_safe = K_slow > 0 ? K_slow : Type(10.0);
  Type sst_width_safe = sst_width > 0 ? sst_width : Type(1.0);
  Type allee_threshold_safe = allee_threshold > 0 ? allee_threshold : eps;
  Type allee_strength_safe = allee_strength > 0 ? allee_strength : Type(1.0);
  
  // Run the model for each time step
  for (int t = 1; t < n; t++) {
    // Ensure all values are positive
    Type cots_prev = cots_pred(t-1);
    if (cots_prev < eps) cots_prev = eps;
    
    Type fast_prev = fast_pred(t-1);
    if (fast_prev < eps) fast_prev = eps;
    
    Type slow_prev = slow_pred(t-1);
    if (slow_prev < eps) slow_prev = eps;
    
    // 1. Temperature effect on COTS reproduction (Gaussian response)
    Type temp_diff = sst_dat(t-1) - sst_opt;
    Type temp_effect = exp(-pow(temp_diff, 2) / (2 * pow(sst_width_safe, 2)));
    
    // 2. Total coral cover for density dependence
    Type total_coral = fast_prev + slow_prev;
    
    // 3. Food-dependent mortality modifier
    Type mort_modifier = Type(1.0) + Type(1.0) / (Type(1.0) + exp((total_coral - coral_threshold) / Type(1.0)));
    
    // 4. Allee effect term - simplified implementation
    Type allee_term;
    if (cots_prev < allee_threshold_safe) {
      // Below threshold: reduced reproduction (positive density dependence)
      allee_term = pow(cots_prev / allee_threshold_safe, allee_strength_safe);
    } else {
      // Above threshold: full reproductive potential
      allee_term = Type(1.0);
    }
    
    // 5. Density-dependent feeding preference
    Type density_ratio = cots_prev / K_cots_safe;
    Type pref_modifier;
    
    if (density_ratio <= pref_switch_threshold) {
      // Below threshold: normal feeding preference
      pref_modifier = Type(1.0);
    } else {
      // Above threshold: increased feeding on slow-growing coral
      Type excess_ratio = density_ratio - pref_switch_threshold;
      pref_modifier = Type(1.0) + pref_switch_rate * excess_ratio;
      
      // Cap the modifier to prevent extreme values
      if (pref_modifier > Type(3.0)) pref_modifier = Type(3.0);
    }
    
    // 6. Functional responses for COTS feeding (Type II)
    Type func_response_denom = Type(1.0) + alpha_fast_safe * h_fast_safe * fast_prev + alpha_slow_safe * h_slow_safe * slow_prev;
    
    Type consumption_fast = (alpha_fast_safe * fast_prev * cots_prev) / func_response_denom;
    Type consumption_slow = (alpha_slow_safe * pref_modifier * slow_prev * cots_prev) / func_response_denom;
    
    // Limit consumption to available coral
    if (consumption_fast > fast_prev) consumption_fast = fast_prev;
    if (consumption_slow > slow_prev) consumption_slow = slow_prev;
    
    // 7. COTS population dynamics
    // Carrying capacity depends on available coral
    Type carrying_capacity_modifier = Type(0.5) + Type(0.5) * total_coral / (K_fast_safe + K_slow_safe);
    Type effective_K = K_cots_safe * carrying_capacity_modifier;
    
    // Logistic growth term with bounds
    Type logistic_term = Type(1.0) - cots_prev / effective_K;
    if (logistic_term < Type(-1.0)) logistic_term = Type(-1.0);
    
    // Calculate population change components
    Type cots_growth = r_cots_safe * beta_sst * temp_effect * allee_term * cots_prev * logistic_term;
    Type cots_mortality = m_cots_safe * mort_modifier * cots_prev;
    Type cots_immigration = imm_effect * cotsimm_dat(t-1);
    
    // 8. Update COTS abundance
    cots_pred(t) = cots_prev + cots_growth - cots_mortality + cots_immigration;
    
    // Ensure predictions stay within reasonable bounds
    if (cots_pred(t) < eps) cots_pred(t) = eps;
    if (cots_pred(t) > Type(5.0) * K_cots_safe) cots_pred(t) = Type(5.0) * K_cots_safe;
    
    // 9. Coral dynamics with logistic growth and COTS predation
    // Calculate logistic growth terms with bounds
    Type fast_logistic = Type(1.0) - (fast_prev + Type(0.5) * slow_prev) / K_fast_safe;
    if (fast_logistic < Type(-1.0)) fast_logistic = Type(-1.0);
    
    Type slow_logistic = Type(1.0) - (slow_prev + Type(0.3) * fast_prev) / K_slow_safe;
    if (slow_logistic < Type(-1.0)) slow_logistic = Type(-1.0);
    
    Type fast_growth = r_fast_safe * fast_prev * fast_logistic;
    Type slow_growth = r_slow_safe * slow_prev * slow_logistic;
    
    // 10. Update coral cover
    fast_pred(t) = fast_prev + fast_growth - consumption_fast;
    slow_pred(t) = slow_prev + slow_growth - consumption_slow;
    
    // 11. Ensure coral cover stays positive and below carrying capacity
    if (fast_pred(t) < eps) fast_pred(t) = eps;
    if (fast_pred(t) > K_fast_safe) fast_pred(t) = K_fast_safe;
    
    if (slow_pred(t) < eps) slow_pred(t) = eps;
    if (slow_pred(t) > K_slow_safe) slow_pred(t) = K_slow_safe;
  }
  
  // Calculate negative log-likelihood using lognormal distribution
  Type const_obs = Type(1e-3);
  
  for (int t = 0; t < n; t++) {
    // COTS abundance likelihood
    Type cots_obs = cots_dat(t) + const_obs;
    Type cots_model = cots_pred(t) + const_obs;
    nll -= dnorm(log(cots_obs), log(cots_model), sigma_cots, true);
    
    // Fast-growing coral cover likelihood
    Type fast_obs = fast_dat(t) + const_obs;
    Type fast_model = fast_pred(t) + const_obs;
    nll -= dnorm(log(fast_obs), log(fast_model), sigma_fast, true);
    
    // Slow-growing coral cover likelihood
    Type slow_obs = slow_dat(t) + const_obs;
    Type slow_model = slow_pred(t) + const_obs;
    nll -= dnorm(log(slow_obs), log(slow_model), sigma_slow, true);
  }
  
  // Add weak penalties to constrain parameters to biologically reasonable values
  nll += Type(0.001) * pow(r_cots - Type(0.8), 2);
  nll += Type(0.001) * pow(K_cots - Type(2.5), 2);
  nll += Type(0.001) * pow(m_cots - Type(0.3), 2);
  nll += Type(0.001) * pow(alpha_fast - Type(0.15), 2);
  nll += Type(0.001) * pow(alpha_slow - Type(0.05), 2);
  nll += Type(0.001) * pow(allee_threshold - Type(0.3), 2);
  nll += Type(0.001) * pow(allee_strength - Type(1.5), 2);
  nll += Type(0.001) * pow(pref_switch_threshold - Type(0.8), 2);
  nll += Type(0.001) * pow(pref_switch_rate - Type(2.0), 2);
  
  // REPORT SECTION
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  ADREPORT(r_cots);
  ADREPORT(K_cots);
  ADREPORT(alpha_fast);
  ADREPORT(alpha_slow);
  ADREPORT(r_fast);
  ADREPORT(r_slow);
  ADREPORT(beta_sst);
  ADREPORT(imm_effect);
  ADREPORT(allee_threshold);
  ADREPORT(allee_strength);
  ADREPORT(pref_switch_threshold);
  ADREPORT(pref_switch_rate);
  
  return nll;
}
