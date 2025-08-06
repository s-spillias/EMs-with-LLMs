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
  PARAMETER(pred_threshold);          // COTS density threshold for predator saturation (individuals/m^2)
  PARAMETER(pred_intensity);          // Intensity of predation pressure on COTS (dimensionless)
  
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
    // 1. Temperature effect on COTS reproduction (simple linear effect)
    Type temp_effect = Type(1.0);
    if (sst_dat(t-1) > T_crit) {
      temp_effect = Type(1.0) + T_effect;
    }
    
    // 2. Type II functional response for COTS predation on fast-growing coral
    Type consumption_fast = (a_fast * fast_pred(t-1) * cots_pred(t-1)) / 
                           (Type(1.0) + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1) + eps);
    
    // 3. Type II functional response for COTS predation on slow-growing coral
    Type consumption_slow = (a_slow * slow_pred(t-1) * cots_pred(t-1)) / 
                           (Type(1.0) + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1) + eps);
    
    // 4. COTS population dynamics with temperature effect, density dependence, and immigration
    Type cots_growth = r_cots * temp_effect * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1) / (K_cots + eps));
    
    // 5. Predator-driven Allee effect - higher mortality at low COTS densities
    // Use a simple step function instead of continuous function
    Type pred_effect = Type(0.0);
    if (cots_pred(t-1) < pred_threshold) {
      pred_effect = pred_intensity;
    }
    
    // Ensure mortality rate is positive
    Type cots_mortality = (m_cots + pred_effect) * cots_pred(t-1);
    
    // Immigration term
    Type cots_immigration = imm_effect * cotsimm_dat(t-1);
    
    // 6. Update COTS population
    cots_pred(t) = cots_pred(t-1) + cots_growth - cots_mortality + cots_immigration;
    
    // Ensure COTS population stays positive
    if (cots_pred(t) < eps) {
      cots_pred(t) = eps;
    }
    
    // 7. Fast-growing coral dynamics with competition and predation
    Type fast_growth = r_fast * fast_pred(t-1) * (Type(1.0) - (fast_pred(t-1) + alpha_fs * slow_pred(t-1)) / (K_fast + eps));
    
    // 8. Update fast-growing coral
    fast_pred(t) = fast_pred(t-1) + fast_growth - consumption_fast;
    
    // Ensure fast coral stays positive and below carrying capacity
    if (fast_pred(t) < eps) {
      fast_pred(t) = eps;
    }
    if (fast_pred(t) > K_fast) {
      fast_pred(t) = K_fast;
    }
    
    // 9. Slow-growing coral dynamics with competition and predation
    Type slow_growth = r_slow * slow_pred(t-1) * (Type(1.0) - (slow_pred(t-1) + alpha_sf * fast_pred(t-1)) / (K_slow + eps));
    
    // 10. Update slow-growing coral
    slow_pred(t) = slow_pred(t-1) + slow_growth - consumption_slow;
    
    // Ensure slow coral stays positive and below carrying capacity
    if (slow_pred(t) < eps) {
      slow_pred(t) = eps;
    }
    if (slow_pred(t) > K_slow) {
      slow_pred(t) = K_slow;
    }
  }
  
  // Calculate negative log-likelihood using lognormal distribution
  for (int t = 0; t < n_years; t++) {
    // 11. Add observation error for COTS abundance (lognormal)
    if (!R_IsNA(asDouble(cots_dat(t))) && cots_dat(t) > 0) {
      Type obs = cots_dat(t);
      Type pred = cots_pred(t);
      nll -= dnorm(log(obs), log(pred), sigma_cots_adj, true);
    }
    
    // 12. Add observation error for fast-growing coral cover (lognormal)
    if (!R_IsNA(asDouble(fast_dat(t))) && fast_dat(t) > 0) {
      Type obs = fast_dat(t);
      Type pred = fast_pred(t);
      nll -= dnorm(log(obs), log(pred), sigma_fast_adj, true);
    }
    
    // 13. Add observation error for slow-growing coral cover (lognormal)
    if (!R_IsNA(asDouble(slow_dat(t))) && slow_dat(t) > 0) {
      Type obs = slow_dat(t);
      Type pred = slow_pred(t);
      nll -= dnorm(log(obs), log(pred), sigma_slow_adj, true);
    }
  }
  
  // Report model predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  // Additional derived quantities for reporting
  vector<Type> temp_effect_vec(n_years);
  vector<Type> consumption_fast_vec(n_years);
  vector<Type> consumption_slow_vec(n_years);
  vector<Type> pred_effect_vec(n_years);
  
  for (int t = 0; t < n_years; t++) {
    // Calculate temperature effect for each year
    temp_effect_vec(t) = Type(1.0);
    if (sst_dat(t) > T_crit) {
      temp_effect_vec(t) = Type(1.0) + T_effect;
    }
    
    // Calculate predation effect for each year
    pred_effect_vec(t) = Type(0.0);
    if (cots_pred(t) < pred_threshold) {
      pred_effect_vec(t) = pred_intensity;
    }
    
    // Calculate consumption rates for each year
    if (t > 0) {
      consumption_fast_vec(t) = (a_fast * fast_pred(t-1) * cots_pred(t-1)) / 
                           (Type(1.0) + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1) + eps);
      consumption_slow_vec(t) = (a_slow * slow_pred(t-1) * cots_pred(t-1)) / 
                           (Type(1.0) + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1) + eps);
    } else {
      consumption_fast_vec(t) = Type(0.0);
      consumption_slow_vec(t) = Type(0.0);
    }
  }
  
  REPORT(temp_effect_vec);
  REPORT(consumption_fast_vec);
  REPORT(consumption_slow_vec);
  REPORT(pred_effect_vec);
  
  return nll;
}
