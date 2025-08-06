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
  PARAMETER(m_cots_adult);            // Natural mortality rate of adult COTS (year^-1)
  PARAMETER(m_cots_juv);              // Natural mortality rate of juvenile COTS (year^-1)
  PARAMETER(maturation_rate);         // Maturation rate from juvenile to adult COTS (year^-1)
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
  PARAMETER(recovery_delay_fast);     // Recovery delay for fast-growing coral after predation (years)
  PARAMETER(recovery_delay_slow);     // Recovery delay for slow-growing coral after predation (years)
  PARAMETER(sigma_cots);              // Observation error standard deviation for COTS abundance (log scale)
  PARAMETER(sigma_fast);              // Observation error standard deviation for fast-growing coral cover (log scale)
  PARAMETER(sigma_slow);              // Observation error standard deviation for slow-growing coral cover (log scale)
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Number of time steps
  int n_years = Year.size();
  
  // Vectors to store model predictions
  vector<Type> cots_pred(n_years);        // Total COTS (adults + juveniles)
  vector<Type> cots_adult_pred(n_years);  // Adult COTS
  vector<Type> cots_juv_pred(n_years);    // Juvenile COTS
  vector<Type> fast_pred(n_years);        // Fast-growing coral cover
  vector<Type> slow_pred(n_years);        // Slow-growing coral cover
  
  // Vectors to track predation history for recovery delay
  vector<Type> fast_predation_history(n_years);
  vector<Type> slow_predation_history(n_years);
  
  // Initialize with first year's observed values
  cots_pred(0) = cots_dat(0);
  cots_adult_pred(0) = cots_dat(0) * Type(0.6);  // Assume 60% adults initially
  cots_juv_pred(0) = cots_dat(0) * Type(0.4);    // Assume 40% juveniles initially
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_predation_history(0) = Type(0.0);
  slow_predation_history(0) = Type(0.0);
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(0.01);
  Type sigma_cots_adj = sigma_cots + min_sigma;
  Type sigma_fast_adj = sigma_fast + min_sigma;
  Type sigma_slow_adj = sigma_slow + min_sigma;
  
  // Ensure parameters are positive
  Type r_cots_pos = exp(r_cots);
  Type K_cots_pos = exp(K_cots);
  Type m_cots_adult_pos = exp(m_cots_adult);
  Type m_cots_juv_pos = exp(m_cots_juv);
  Type maturation_rate_pos = exp(maturation_rate);
  Type a_fast_pos = exp(a_fast);
  Type a_slow_pos = exp(a_slow);
  Type h_fast_pos = exp(h_fast);
  Type h_slow_pos = exp(h_slow);
  Type r_fast_pos = exp(r_fast);
  Type r_slow_pos = exp(r_slow);
  Type K_fast_pos = exp(K_fast);
  Type K_slow_pos = exp(K_slow);
  Type recovery_delay_fast_pos = exp(recovery_delay_fast);
  Type recovery_delay_slow_pos = exp(recovery_delay_slow);
  
  // Time series simulation
  for (int t = 1; t < n_years; t++) {
    // 1. Temperature effect on COTS reproduction (smooth transition around threshold)
    Type temp_effect = Type(1.0) + T_effect * (Type(1.0) / (Type(1.0) + exp(-Type(5.0) * (sst_dat(t-1) - T_crit))));
    
    // 2. Type II functional response for COTS predation on fast-growing coral (only adults predate)
    Type consumption_fast = (a_fast_pos * fast_pred(t-1) * cots_adult_pred(t-1)) / 
                           (Type(1.0) + a_fast_pos * h_fast_pos * fast_pred(t-1) + a_slow_pos * h_slow_pos * slow_pred(t-1) + eps);
    
    // 3. Type II functional response for COTS predation on slow-growing coral (only adults predate)
    Type consumption_slow = (a_slow_pos * slow_pred(t-1) * cots_adult_pred(t-1)) / 
                           (Type(1.0) + a_fast_pos * h_fast_pos * fast_pred(t-1) + a_slow_pos * h_slow_pos * slow_pred(t-1) + eps);
    
    // Ensure consumption doesn't exceed available coral
    consumption_fast = CppAD::CondExpGt(consumption_fast, fast_pred(t-1), fast_pred(t-1), consumption_fast);
    consumption_slow = CppAD::CondExpGt(consumption_slow, slow_pred(t-1), slow_pred(t-1), consumption_slow);
    
    // 4. Update predation history for recovery delay calculation
    fast_predation_history(t) = consumption_fast;
    slow_predation_history(t) = consumption_slow;
    
    // 5. Calculate recovery inhibition based on recent predation history
    Type fast_recovery_inhibition = Type(0.0);
    Type slow_recovery_inhibition = Type(0.0);
    
    // Convert recovery delay parameters to integers safely
    // Use Type variables for delay calculations instead of integers
    Type fast_delay_value = recovery_delay_fast_pos;
    fast_delay_value = CppAD::CondExpGt(fast_delay_value, Type(t), Type(t), fast_delay_value);
    fast_delay_value = CppAD::CondExpLt(fast_delay_value, Type(1), Type(1), fast_delay_value);
    
    Type slow_delay_value = recovery_delay_slow_pos;
    slow_delay_value = CppAD::CondExpGt(slow_delay_value, Type(t), Type(t), slow_delay_value);
    slow_delay_value = CppAD::CondExpLt(slow_delay_value, Type(1), Type(1), slow_delay_value);
    
    // Calculate recovery inhibition based on predation history
    // Use a fixed maximum lookback to avoid variable loop bounds
    int max_lookback = 10; // Maximum number of years to look back
    
    for (int i = 0; i < max_lookback; i++) {
      if (i < t) { // Only look at valid history
        // Weight by how much this year contributes to the delay
        Type fast_weight = CppAD::CondExpLt(Type(i), fast_delay_value, Type(1.0) / fast_delay_value, Type(0.0));
        Type slow_weight = CppAD::CondExpLt(Type(i), slow_delay_value, Type(1.0) / slow_delay_value, Type(0.0));
        
        fast_recovery_inhibition += fast_predation_history(t-i) * fast_weight;
        slow_recovery_inhibition += slow_predation_history(t-i) * slow_weight;
      }
    }
    
    // Normalize recovery inhibition to [0,1] range using sigmoid function
    fast_recovery_inhibition = Type(1.0) / (Type(1.0) + exp(-Type(5.0) * (fast_recovery_inhibition - Type(0.5))));
    slow_recovery_inhibition = Type(1.0) / (Type(1.0) + exp(-Type(5.0) * (slow_recovery_inhibition - Type(0.2))));
    
    // 6. COTS population dynamics with temperature effect, density dependence, and immigration
    
    // Adult COTS reproduction (only adults reproduce)
    Type density_factor = Type(1.0) - (cots_adult_pred(t-1) + cots_juv_pred(t-1)) / K_cots_pos;
    density_factor = CppAD::CondExpLt(density_factor, Type(0.0), Type(0.0), density_factor);
    
    Type cots_reproduction = r_cots_pos * temp_effect * cots_adult_pred(t-1) * density_factor;
    
    // Juvenile COTS dynamics
    Type juv_maturation = maturation_rate_pos * cots_juv_pred(t-1);
    Type juv_mortality = m_cots_juv_pos * cots_juv_pred(t-1);
    
    // Adult COTS dynamics
    Type adult_mortality = m_cots_adult_pos * cots_adult_pred(t-1);
    
    // Immigration affects juvenile population
    Type cots_immigration = imm_effect * cotsimm_dat(t-1);
    
    // Update juvenile COTS population with bounds checking
    cots_juv_pred(t) = cots_juv_pred(t-1) + cots_reproduction - juv_maturation - juv_mortality + cots_immigration;
    cots_juv_pred(t) = CppAD::CondExpLt(cots_juv_pred(t), Type(0.0), Type(0.0), cots_juv_pred(t));
    
    // Update adult COTS population with bounds checking
    cots_adult_pred(t) = cots_adult_pred(t-1) + juv_maturation - adult_mortality;
    cots_adult_pred(t) = CppAD::CondExpLt(cots_adult_pred(t), Type(0.0), Type(0.0), cots_adult_pred(t));
    
    // Total COTS population (for comparison with observations)
    cots_pred(t) = cots_adult_pred(t) + cots_juv_pred(t);
    
    // 7. Fast-growing coral dynamics with competition, predation, and recovery inhibition
    Type competition_fast = Type(1.0) - (fast_pred(t-1) + alpha_fs * slow_pred(t-1)) / K_fast_pos;
    competition_fast = CppAD::CondExpLt(competition_fast, Type(0.0), Type(0.0), competition_fast);
    
    Type fast_growth = r_fast_pos * fast_pred(t-1) * competition_fast;
    
    // Apply recovery inhibition to growth
    fast_growth *= (Type(1.0) - fast_recovery_inhibition);
    
    // 8. Update fast-growing coral with bounds checking
    fast_pred(t) = fast_pred(t-1) + fast_growth - consumption_fast;
    fast_pred(t) = CppAD::CondExpLt(fast_pred(t), Type(0.0), Type(0.0), fast_pred(t));
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), K_fast_pos, K_fast_pos, fast_pred(t));
    
    // 9. Slow-growing coral dynamics with competition, predation, and recovery inhibition
    Type competition_slow = Type(1.0) - (slow_pred(t-1) + alpha_sf * fast_pred(t-1)) / K_slow_pos;
    competition_slow = CppAD::CondExpLt(competition_slow, Type(0.0), Type(0.0), competition_slow);
    
    Type slow_growth = r_slow_pos * slow_pred(t-1) * competition_slow;
    
    // Apply recovery inhibition to growth
    slow_growth *= (Type(1.0) - slow_recovery_inhibition);
    
    // 10. Update slow-growing coral with bounds checking
    slow_pred(t) = slow_pred(t-1) + slow_growth - consumption_slow;
    slow_pred(t) = CppAD::CondExpLt(slow_pred(t), Type(0.0), Type(0.0), slow_pred(t));
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), K_slow_pos, K_slow_pos, slow_pred(t));
  }
  
  // Calculate negative log-likelihood using lognormal distribution
  for (int t = 0; t < n_years; t++) {
    // 11. Add observation error for COTS abundance (lognormal)
    if (!R_IsNA(asDouble(cots_dat(t))) && cots_pred(t) > eps) {
      Type pred = cots_pred(t) + eps;
      Type obs = cots_dat(t) + eps;
      nll -= dnorm(log(obs), log(pred), sigma_cots_adj, true);
    }
    
    // 12. Add observation error for fast-growing coral cover (lognormal)
    if (!R_IsNA(asDouble(fast_dat(t))) && fast_pred(t) > eps) {
      Type pred = fast_pred(t) + eps;
      Type obs = fast_dat(t) + eps;
      nll -= dnorm(log(obs), log(pred), sigma_fast_adj, true);
    }
    
    // 13. Add observation error for slow-growing coral cover (lognormal)
    if (!R_IsNA(asDouble(slow_dat(t))) && slow_pred(t) > eps) {
      Type pred = slow_pred(t) + eps;
      Type obs = slow_dat(t) + eps;
      nll -= dnorm(log(obs), log(pred), sigma_slow_adj, true);
    }
  }
  
  // Report model predictions
  REPORT(cots_pred);
  REPORT(cots_adult_pred);
  REPORT(cots_juv_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  // Additional derived quantities for reporting
  vector<Type> temp_effect(n_years);
  vector<Type> consumption_fast(n_years);
  vector<Type> consumption_slow(n_years);
  vector<Type> fast_recovery_inhibition(n_years);
  vector<Type> slow_recovery_inhibition(n_years);
  
  for (int t = 0; t < n_years; t++) {
    // Calculate temperature effect for each year
    temp_effect(t) = Type(1.0) + T_effect * (Type(1.0) / (Type(1.0) + exp(-Type(5.0) * (sst_dat(t) - T_crit))));
    
    // Calculate consumption rates and recovery inhibition for each year
    if (t > 0) {
      // Consumption rates
      consumption_fast(t) = (a_fast_pos * fast_pred(t-1) * cots_adult_pred(t-1)) / 
                           (Type(1.0) + a_fast_pos * h_fast_pos * fast_pred(t-1) + a_slow_pos * h_slow_pos * slow_pred(t-1) + eps);
      consumption_slow(t) = (a_slow_pos * slow_pred(t-1) * cots_adult_pred(t-1)) / 
                           (Type(1.0) + a_fast_pos * h_fast_pos * fast_pred(t-1) + a_slow_pos * h_slow_pos * slow_pred(t-1) + eps);
      
      // Ensure consumption doesn't exceed available coral
      consumption_fast(t) = CppAD::CondExpGt(consumption_fast(t), fast_pred(t-1), fast_pred(t-1), consumption_fast(t));
      consumption_slow(t) = CppAD::CondExpGt(consumption_slow(t), slow_pred(t-1), slow_pred(t-1), consumption_slow(t));
      
      // Recovery inhibition
      fast_recovery_inhibition(t) = Type(0.0);
      slow_recovery_inhibition(t) = Type(0.0);
      
      // Use Type variables for delay calculations
      Type fast_delay_value = recovery_delay_fast_pos;
      fast_delay_value = CppAD::CondExpGt(fast_delay_value, Type(t), Type(t), fast_delay_value);
      fast_delay_value = CppAD::CondExpLt(fast_delay_value, Type(1), Type(1), fast_delay_value);
      
      Type slow_delay_value = recovery_delay_slow_pos;
      slow_delay_value = CppAD::CondExpGt(slow_delay_value, Type(t), Type(t), slow_delay_value);
      slow_delay_value = CppAD::CondExpLt(slow_delay_value, Type(1), Type(1), slow_delay_value);
      
      // Use a fixed maximum lookback to avoid variable loop bounds
      int max_lookback = 10; // Maximum number of years to look back
      
      for (int i = 0; i < max_lookback; i++) {
        if (i < t) { // Only look at valid history
          // Weight by how much this year contributes to the delay
          Type fast_weight = CppAD::CondExpLt(Type(i), fast_delay_value, Type(1.0) / fast_delay_value, Type(0.0));
          Type slow_weight = CppAD::CondExpLt(Type(i), slow_delay_value, Type(1.0) / slow_delay_value, Type(0.0));
          
          fast_recovery_inhibition(t) += fast_predation_history(t-i) * fast_weight;
          slow_recovery_inhibition(t) += slow_predation_history(t-i) * slow_weight;
        }
      }
      
      fast_recovery_inhibition(t) = Type(1.0) / (Type(1.0) + exp(-Type(5.0) * (fast_recovery_inhibition(t) - Type(0.5))));
      slow_recovery_inhibition(t) = Type(1.0) / (Type(1.0) + exp(-Type(5.0) * (slow_recovery_inhibition(t) - Type(0.2))));
    } else {
      consumption_fast(t) = Type(0.0);
      consumption_slow(t) = Type(0.0);
      fast_recovery_inhibition(t) = Type(0.0);
      slow_recovery_inhibition(t) = Type(0.0);
    }
  }
  
  REPORT(temp_effect);
  REPORT(consumption_fast);
  REPORT(consumption_slow);
  REPORT(fast_recovery_inhibition);
  REPORT(slow_recovery_inhibition);
  REPORT(fast_predation_history);
  REPORT(slow_predation_history);
  
  return nll;
}
