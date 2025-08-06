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
    Type temp_effect = Type(1.0) + T_effect * (Type(1.0) / (Type(1.0) + exp(-Type(5.0) * (sst_dat(t-1) - T_crit))));
    
    // 2. Type II functional response for COTS predation on fast-growing coral
    Type consumption_fast = (a_fast * fast_pred(t-1) * cots_pred(t-1)) / 
                           (Type(1.0) + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1));
    
    // 3. Type II functional response for COTS predation on slow-growing coral
    Type consumption_slow = (a_slow * slow_pred(t-1) * cots_pred(t-1)) / 
                           (Type(1.0) + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1));
    
    // 4. COTS population dynamics with temperature effect, density dependence, and immigration
    Type cots_growth = r_cots * temp_effect * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1) / K_cots);
    Type cots_mortality = m_cots * cots_pred(t-1);
    Type cots_immigration = imm_effect * cotsimm_dat(t-1);
    
    // 5. Update COTS population with smooth lower bound to prevent negative values
    cots_pred(t) = cots_pred(t-1) + cots_growth - cots_mortality + cots_immigration;
    cots_pred(t) = cots_pred(t) / (Type(1.0) + exp(-Type(10.0) * (cots_pred(t) - eps))) * cots_pred(t); // Smooth lower bound
    
    // 6. Fast-growing coral dynamics with competition and predation
    Type fast_growth = r_fast * fast_pred(t-1) * (Type(1.0) - (fast_pred(t-1) + alpha_fs * slow_pred(t-1)) / K_fast);
    
    // 7. Update fast-growing coral with smooth bounds
    fast_pred(t) = fast_pred(t-1) + fast_growth - consumption_fast;
    fast_pred(t) = fast_pred(t) / (Type(1.0) + exp(-Type(10.0) * (fast_pred(t) - eps))) * fast_pred(t); // Smooth lower bound
    fast_pred(t) = K_fast - (K_fast - fast_pred(t)) / (Type(1.0) + exp(-Type(10.0) * (K_fast - fast_pred(t) - eps))) * (K_fast - fast_pred(t)); // Smooth upper bound
    
    // 8. Slow-growing coral dynamics with competition and predation
    Type slow_growth = r_slow * slow_pred(t-1) * (Type(1.0) - (slow_pred(t-1) + alpha_sf * fast_pred(t-1)) / K_slow);
    
    // 9. Update slow-growing coral with smooth bounds
    slow_pred(t) = slow_pred(t-1) + slow_growth - consumption_slow;
    slow_pred(t) = slow_pred(t) / (Type(1.0) + exp(-Type(10.0) * (slow_pred(t) - eps))) * slow_pred(t); // Smooth lower bound
    slow_pred(t) = K_slow - (K_slow - slow_pred(t)) / (Type(1.0) + exp(-Type(10.0) * (K_slow - slow_pred(t) - eps))) * (K_slow - slow_pred(t)); // Smooth upper bound
  }
  
  // Calculate negative log-likelihood using lognormal distribution
  for (int t = 0; t < n_years; t++) {
    // 10. Add observation error for COTS abundance (lognormal)
    if (!R_IsNA(asDouble(cots_dat(t)))) {
      nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots_adj, true);
    }
    
    // 11. Add observation error for fast-growing coral cover (lognormal)
    if (!R_IsNA(asDouble(fast_dat(t)))) {
      nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast_adj, true);
    }
    
    // 12. Add observation error for slow-growing coral cover (lognormal)
    if (!R_IsNA(asDouble(slow_dat(t)))) {
      nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow_adj, true);
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
  
  for (int t = 0; t < n_years; t++) {
    // Calculate temperature effect for each year
    temp_effect(t) = Type(1.0) + T_effect * (Type(1.0) / (Type(1.0) + exp(-Type(5.0) * (sst_dat(t) - T_crit))));
    
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
  
  return nll;
}
