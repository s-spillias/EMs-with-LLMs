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
  PARAMETER(delay_effect);            // Strength of delayed density-dependence in COTS reproduction
  PARAMETER(coral_threshold);         // Coral cover threshold affecting COTS reproduction
  
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
  vector<Type> cots_prev_avg(n_years);  // Store moving average of previous COTS densities
  
  // Initialize with first year's observed values (ensure positive values)
  cots_pred(0) = cots_dat(0) + eps;
  fast_pred(0) = fast_dat(0) + eps;
  slow_pred(0) = slow_dat(0) + eps;
  cots_prev_avg(0) = cots_dat(0) + eps;
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(0.1);  // Increased from 0.01 to 0.1 for stability
  Type sigma_cots_adj = sigma_cots + min_sigma;
  Type sigma_fast_adj = sigma_fast + min_sigma;
  Type sigma_slow_adj = sigma_slow + min_sigma;
  
  // Time series simulation
  for (int t = 1; t < n_years; t++) {
    // Calculate moving average of previous COTS densities (with memory decay)
    if (t > 1) {
      cots_prev_avg(t-1) = Type(0.7) * cots_pred(t-1) + Type(0.3) * cots_prev_avg(t-2);
    } else {
      cots_prev_avg(t-1) = cots_pred(t-1);
    }
    
    // Ensure positive values for calculations
    Type cots_prev = cots_pred(t-1) + eps;
    Type fast_prev = fast_pred(t-1) + eps;
    Type slow_prev = slow_pred(t-1) + eps;
    Type cots_prev_avg_t = cots_prev_avg(t-1) + eps;
    
    // 1. Temperature effect on COTS reproduction (smooth transition around threshold)
    Type temp_effect = Type(1.0) + T_effect / (Type(1.0) + exp(-Type(0.5) * (sst_dat(t-1) - T_crit)));
    
    // 2. Calculate total coral cover for resource limitation effect
    Type total_coral = fast_prev + slow_prev;
    
    // 3. Resource limitation effect on COTS reproduction (sigmoid function)
    Type resource_effect = Type(1.0) / (Type(1.0) + exp(-Type(0.3) * (total_coral - coral_threshold)));
    
    // 4. Type II functional response for COTS predation on fast-growing coral
    Type consumption_fast = (a_fast * fast_prev * cots_prev) / 
                           (Type(1.0) + a_fast * h_fast * fast_prev + a_slow * h_slow * slow_prev);
    
    // 5. Type II functional response for COTS predation on slow-growing coral
    Type consumption_slow = (a_slow * slow_prev * cots_prev) / 
                           (Type(1.0) + a_fast * h_fast * fast_prev + a_slow * h_slow * slow_prev);
    
    // 6. Delayed density-dependent effect on COTS reproduction
    Type delayed_dd = Type(1.0) / (Type(1.0) + delay_effect * cots_prev_avg_t / K_cots);
    
    // 7. COTS population dynamics with temperature effect, density dependence, delayed effects, and immigration
    Type cots_growth = r_cots * temp_effect * resource_effect * delayed_dd * cots_prev * 
                      (Type(1.0) - cots_prev / K_cots);
    Type cots_mortality = m_cots * cots_prev;
    Type cots_immigration = imm_effect * cotsimm_dat(t-1);
    
    // 8. Update COTS population
    cots_pred(t) = cots_prev + cots_growth - cots_mortality + cots_immigration;
    cots_pred(t) = cots_pred(t) + eps;  // Ensure positive values
    
    // 9. Fast-growing coral dynamics with competition and predation
    // Adjust growth rate based on current cover to allow faster recovery at low densities
    Type recovery_boost_fast = Type(1.0) + Type(1.0) * exp(-Type(0.1) * fast_prev);
    Type fast_growth = r_fast * recovery_boost_fast * fast_prev * 
                      (Type(1.0) - (fast_prev + alpha_fs * slow_prev) / K_fast);
    
    // 10. Update fast-growing coral
    fast_pred(t) = fast_prev + fast_growth - consumption_fast;
    fast_pred(t) = fast_pred(t) + eps;  // Ensure positive values
    fast_pred(t) = fast_pred(t) < K_fast ? fast_pred(t) : K_fast;  // Ensure below carrying capacity
    
    // 11. Slow-growing coral dynamics with competition and predation
    // Adjust growth rate based on current cover to allow faster recovery at low densities
    Type recovery_boost_slow = Type(1.0) + Type(0.8) * exp(-Type(0.1) * slow_prev);
    Type slow_growth = r_slow * recovery_boost_slow * slow_prev * 
                      (Type(1.0) - (slow_prev + alpha_sf * fast_prev) / K_slow);
    
    // 12. Update slow-growing coral
    slow_pred(t) = slow_prev + slow_growth - consumption_slow;
    slow_pred(t) = slow_pred(t) + eps;  // Ensure positive values
    slow_pred(t) = slow_pred(t) < K_slow ? slow_pred(t) : K_slow;  // Ensure below carrying capacity
  }
  
  // Calculate negative log-likelihood using lognormal distribution
  for (int t = 0; t < n_years; t++) {
    // 13. Add observation error for COTS abundance (lognormal)
    if (!R_IsNA(asDouble(cots_dat(t))) && cots_dat(t) > 0) {
      nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t)), sigma_cots_adj, true);
    }
    
    // 14. Add observation error for fast-growing coral cover (lognormal)
    if (!R_IsNA(asDouble(fast_dat(t))) && fast_dat(t) > 0) {
      nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t)), sigma_fast_adj, true);
    }
    
    // 15. Add observation error for slow-growing coral cover (lognormal)
    if (!R_IsNA(asDouble(slow_dat(t))) && slow_dat(t) > 0) {
      nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t)), sigma_slow_adj, true);
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
  vector<Type> resource_effect(n_years);
  vector<Type> delayed_dd(n_years);
  
  for (int t = 0; t < n_years; t++) {
    // Calculate temperature effect for each year
    temp_effect(t) = Type(1.0) + T_effect / (Type(1.0) + exp(-Type(0.5) * (sst_dat(t) - T_crit)));
    
    // Calculate total coral cover
    Type total_coral = fast_pred(t) + slow_pred(t);
    
    // Calculate resource limitation effect
    resource_effect(t) = Type(1.0) / (Type(1.0) + exp(-Type(0.3) * (total_coral - coral_threshold)));
    
    // Calculate delayed density-dependent effect
    if (t > 0) {
      delayed_dd(t) = Type(1.0) / (Type(1.0) + delay_effect * (cots_prev_avg(t-1) + eps) / K_cots);
    } else {
      delayed_dd(t) = Type(1.0);
    }
    
    // Calculate consumption rates for each year
    if (t > 0) {
      Type cots_prev = cots_pred(t-1) + eps;
      Type fast_prev = fast_pred(t-1) + eps;
      Type slow_prev = slow_pred(t-1) + eps;
      
      consumption_fast(t) = (a_fast * fast_prev * cots_prev) / 
                           (Type(1.0) + a_fast * h_fast * fast_prev + a_slow * h_slow * slow_prev);
      consumption_slow(t) = (a_slow * slow_prev * cots_prev) / 
                           (Type(1.0) + a_fast * h_fast * fast_prev + a_slow * h_slow * slow_prev);
    } else {
      consumption_fast(t) = Type(0.0);
      consumption_slow(t) = Type(0.0);
    }
  }
  
  REPORT(temp_effect);
  REPORT(consumption_fast);
  REPORT(consumption_slow);
  REPORT(resource_effect);
  REPORT(delayed_dd);
  REPORT(cots_prev_avg);
  
  return nll;
}
