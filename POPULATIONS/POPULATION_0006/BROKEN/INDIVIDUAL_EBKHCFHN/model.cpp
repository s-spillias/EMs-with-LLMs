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
  PARAMETER(allee_threshold);         // Population threshold below which Allee effects reduce COTS reproduction
  PARAMETER(delay_strength);          // Strength of delayed density dependence in COTS population
  
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
  
  // Initialize with first year's observed values
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(0.01);
  Type sigma_cots_adj = sigma_cots + min_sigma;
  Type sigma_fast_adj = sigma_fast + min_sigma;
  Type sigma_slow_adj = sigma_slow + min_sigma;
  
  // Time series simulation
  for (int t = 1; t < n_years; t++) {
    // 1. Temperature effect on COTS reproduction (smooth transition around threshold)
    Type temp_effect = Type(1.0) + T_effect * (Type(1.0) / (Type(1.0) + exp(-Type(2.0) * (sst_dat(t-1) - T_crit))));
    
    // 2. Type II functional response for COTS predation on fast-growing coral
    Type consumption_fast = (a_fast * fast_pred(t-1) * cots_pred(t-1)) / 
                           (Type(1.0) + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1));
    
    // 3. Type II functional response for COTS predation on slow-growing coral
    Type consumption_slow = (a_slow * slow_pred(t-1) * cots_pred(t-1)) / 
                           (Type(1.0) + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1));
    
    // 4. COTS population dynamics with temperature effect, Allee effect, delayed density dependence, and immigration
    // Calculate Allee effect term (approaches 0 at low densities, approaches 1 at high densities)
    Type allee_param = allee_threshold > eps ? allee_threshold : eps;
    Type allee_effect = cots_pred(t-1) * cots_pred(t-1) / 
                       (allee_param * allee_param + cots_pred(t-1) * cots_pred(t-1));
    
    // Calculate delayed density dependence term
    Type delayed_dd = Type(1.0);
    if (t > 1) {
      // Use population from two time steps ago to create delay
      Type K_safe = K_cots > eps ? K_cots : eps;
      Type delay_param = delay_strength < Type(0.99) ? delay_strength : Type(0.99);
      Type delay_term = Type(1.0) - delay_param * (cots_pred(t-2) / K_safe);
      delayed_dd = delay_term > eps ? delay_term : eps;
    }
    
    // Modified logistic growth with Allee effect and delayed density dependence
    Type r_safe = r_cots > eps ? r_cots : eps;
    Type K_safe = K_cots > eps ? K_cots : eps;
    
    Type cots_growth = r_safe * temp_effect * cots_pred(t-1) * 
                      (Type(1.0) - cots_pred(t-1) / K_safe) * 
                      allee_effect * delayed_dd;
    
    // Ensure growth term is not negative
    cots_growth = cots_growth > Type(0.0) ? cots_growth : Type(0.0);
    
    Type cots_mortality = m_cots * cots_pred(t-1);
    Type cots_immigration = imm_effect * cotsimm_dat(t-1);
    
    // 5. Update COTS population with lower bound to prevent negative values
    cots_pred(t) = cots_pred(t-1) + cots_growth - cots_mortality + cots_immigration;
    cots_pred(t) = cots_pred(t) > Type(0.0) ? cots_pred(t) : Type(0.0);
    
    // 6. Fast-growing coral dynamics with competition and predation
    // Add coral recovery threshold - reduced growth at very low cover
    Type recovery_threshold = Type(0.05); // 5% cover threshold
    Type recovery_factor = fast_pred(t-1) * fast_pred(t-1) / 
                          (recovery_threshold * recovery_threshold + fast_pred(t-1) * fast_pred(t-1));
    
    Type K_fast_safe = K_fast > eps ? K_fast : eps;
    Type fast_growth = r_fast * fast_pred(t-1) * 
                      (Type(1.0) - (fast_pred(t-1) + alpha_fs * slow_pred(t-1)) / K_fast_safe) * 
                      recovery_factor;
    
    // Ensure growth term is not negative
    fast_growth = fast_growth > Type(0.0) ? fast_growth : Type(0.0);
    
    // 7. Update fast-growing coral with bounds
    fast_pred(t) = fast_pred(t-1) + fast_growth - consumption_fast;
    fast_pred(t) = fast_pred(t) > Type(0.0) ? fast_pred(t) : Type(0.0);
    fast_pred(t) = fast_pred(t) < K_fast ? fast_pred(t) : K_fast;
    
    // 8. Slow-growing coral dynamics with competition and predation
    // Add coral recovery threshold - reduced growth at very low cover
    Type slow_recovery_factor = slow_pred(t-1) * slow_pred(t-1) / 
                               (recovery_threshold * recovery_threshold + slow_pred(t-1) * slow_pred(t-1));
    
    Type K_slow_safe = K_slow > eps ? K_slow : eps;
    Type slow_growth = r_slow * slow_pred(t-1) * 
                      (Type(1.0) - (slow_pred(t-1) + alpha_sf * fast_pred(t-1)) / K_slow_safe) * 
                      slow_recovery_factor;
    
    // Ensure growth term is not negative
    slow_growth = slow_growth > Type(0.0) ? slow_growth : Type(0.0);
    
    // 9. Update slow-growing coral with bounds
    slow_pred(t) = slow_pred(t-1) + slow_growth - consumption_slow;
    slow_pred(t) = slow_pred(t) > Type(0.0) ? slow_pred(t) : Type(0.0);
    slow_pred(t) = slow_pred(t) < K_slow ? slow_pred(t) : K_slow;
  }
  
  // Calculate negative log-likelihood using lognormal distribution
  for (int t = 0; t < n_years; t++) {
    // Add small constant to prevent log(0)
    Type cots_obs = cots_dat(t) + eps;
    Type cots_model = cots_pred(t) + eps;
    Type fast_obs = fast_dat(t) + eps;
    Type fast_model = fast_pred(t) + eps;
    Type slow_obs = slow_dat(t) + eps;
    Type slow_model = slow_pred(t) + eps;
    
    // 10. Add observation error for COTS abundance (lognormal)
    if (!R_IsNA(asDouble(cots_dat(t)))) {
      nll -= dnorm(log(cots_obs), log(cots_model), sigma_cots_adj, true);
    }
    
    // 11. Add observation error for fast-growing coral cover (lognormal)
    if (!R_IsNA(asDouble(fast_dat(t)))) {
      nll -= dnorm(log(fast_obs), log(fast_model), sigma_fast_adj, true);
    }
    
    // 12. Add observation error for slow-growing coral cover (lognormal)
    if (!R_IsNA(asDouble(slow_dat(t)))) {
      nll -= dnorm(log(slow_obs), log(slow_model), sigma_slow_adj, true);
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
  vector<Type> allee_effect(n_years);
  vector<Type> delayed_dd(n_years);
  
  for (int t = 0; t < n_years; t++) {
    // Calculate temperature effect for each year
    temp_effect(t) = Type(1.0) + T_effect * (Type(1.0) / (Type(1.0) + exp(-Type(2.0) * (sst_dat(t) - T_crit))));
    
    // Calculate Allee effect for each year
    Type allee_param = allee_threshold > eps ? allee_threshold : eps;
    allee_effect(t) = cots_pred(t) * cots_pred(t) / 
                     (allee_param * allee_param + cots_pred(t) * cots_pred(t));
    
    // Calculate delayed density dependence for each year
    if (t > 1) {
      Type K_safe = K_cots > eps ? K_cots : eps;
      Type delay_param = delay_strength < Type(0.99) ? delay_strength : Type(0.99);
      Type delay_term = Type(1.0) - delay_param * (cots_pred(t-1) / K_safe);
      delayed_dd(t) = delay_term > eps ? delay_term : eps;
    } else {
      delayed_dd(t) = Type(1.0);
    }
    
    // Calculate consumption rates for each year
    if (t > 0) {
      consumption_fast(t) = (a_fast * fast_pred(t-1) * cots_pred(t-1)) / 
                           (Type(1.0) + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1));
      consumption_slow(t) = (a_slow * slow_pred(t-1) * cots_pred(t-1)) / 
                           (Type(1.0) + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1));
    } else {
      consumption_fast(t) = Type(0.0);
      consumption_slow(t) = Type(0.0);
    }
  }
  
  REPORT(temp_effect);
  REPORT(consumption_fast);
  REPORT(consumption_slow);
  REPORT(allee_effect);
  REPORT(delayed_dd);
  
  return nll;
}
