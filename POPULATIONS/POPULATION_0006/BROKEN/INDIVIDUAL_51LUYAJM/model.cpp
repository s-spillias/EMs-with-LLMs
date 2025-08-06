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
  PARAMETER(nutr_effect);             // Effect size of coral consumption on COTS reproduction (dimensionless)
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-4);
  
  // Number of time steps
  int n_years = Year.size();
  
  // Vectors to store model predictions
  vector<Type> cots_pred(n_years);
  vector<Type> fast_pred(n_years);
  vector<Type> slow_pred(n_years);
  
  // Initialize with first year's observed values (ensure positive values)
  cots_pred(0) = cots_dat(0);
  if (cots_pred(0) < eps) cots_pred(0) = eps;
  
  fast_pred(0) = fast_dat(0);
  if (fast_pred(0) < eps) fast_pred(0) = eps;
  
  slow_pred(0) = slow_dat(0);
  if (slow_pred(0) < eps) slow_pred(0) = eps;
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(0.1);
  Type sigma_cots_adj = sigma_cots + min_sigma;
  Type sigma_fast_adj = sigma_fast + min_sigma;
  Type sigma_slow_adj = sigma_slow + min_sigma;
  
  // Vectors to store consumption rates for reporting
  vector<Type> consumption_fast_vec(n_years);
  vector<Type> consumption_slow_vec(n_years);
  vector<Type> nutr_multiplier_vec(n_years);
  
  // Initialize consumption vectors to avoid NA values
  consumption_fast_vec(0) = Type(0.0);
  consumption_slow_vec(0) = Type(0.0);
  nutr_multiplier_vec(0) = Type(0.5); // Default middle value
  
  // Simplify the model to focus on the nutritional feedback mechanism
  for (int t = 1; t < n_years; t++) {
    // 1. Temperature effect - simplified to avoid gradient issues
    Type temp_effect = Type(1.0);
    
    // 2. Type II functional response for COTS predation on fast-growing coral
    Type denominator = Type(1.0) + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1);
    if (denominator < eps) denominator = eps;
    
    Type consumption_fast = (a_fast * fast_pred(t-1) * cots_pred(t-1)) / denominator;
    if (consumption_fast < 0) consumption_fast = 0;
    if (consumption_fast > fast_pred(t-1) * Type(0.9)) consumption_fast = fast_pred(t-1) * Type(0.9);
    
    // 3. Type II functional response for COTS predation on slow-growing coral
    Type consumption_slow = (a_slow * slow_pred(t-1) * cots_pred(t-1)) / denominator;
    if (consumption_slow < 0) consumption_slow = 0;
    if (consumption_slow > slow_pred(t-1) * Type(0.9)) consumption_slow = slow_pred(t-1) * Type(0.9);
    
    // Store consumption rates for reporting
    consumption_fast_vec(t) = consumption_fast;
    consumption_slow_vec(t) = consumption_slow;
    
    // 4. Nutritional feedback - COTS reproduction is enhanced by coral consumption
    Type total_coral_available = fast_pred(t-1) + slow_pred(t-1);
    if (total_coral_available < eps) total_coral_available = eps;
    
    Type total_consumption = consumption_fast + consumption_slow;
    
    // Calculate consumption ratio with safeguards
    Type consumption_ratio = total_consumption / total_coral_available;
    if (consumption_ratio > Type(1.0)) consumption_ratio = Type(1.0);
    if (consumption_ratio < 0) consumption_ratio = 0;
    
    // Nutritional multiplier with safeguards - simplified to avoid gradient issues
    Type nutr_multiplier = Type(0.2) + Type(0.8) * consumption_ratio;
    if (nutr_multiplier > Type(1.0)) nutr_multiplier = Type(1.0);
    if (nutr_multiplier < Type(0.2)) nutr_multiplier = Type(0.2);
    
    // Store nutritional multiplier for reporting
    nutr_multiplier_vec(t) = nutr_multiplier;
    
    // 5. COTS population dynamics with safeguards
    Type r_cots_safe = r_cots;
    if (r_cots_safe < 0) r_cots_safe = 0;
    
    Type m_cots_safe = m_cots;
    if (m_cots_safe < 0) m_cots_safe = 0;
    
    // Simplified growth model to avoid gradient issues
    Type cots_growth = r_cots_safe * nutr_multiplier * cots_pred(t-1) * Type(0.5);
    Type cots_mortality = m_cots_safe * cots_pred(t-1);
    
    // Update COTS population with safeguards
    cots_pred(t) = cots_pred(t-1) + cots_growth - cots_mortality;
    if (cots_pred(t) < eps) cots_pred(t) = eps;
    if (cots_pred(t) > Type(10.0)) cots_pred(t) = Type(10.0); // Hard upper limit
    
    // 6. Fast-growing coral dynamics with safeguards
    Type r_fast_safe = r_fast;
    if (r_fast_safe < 0) r_fast_safe = 0;
    
    // Simplified growth model to avoid gradient issues
    Type fast_growth = r_fast_safe * fast_pred(t-1) * Type(0.5);
    
    // Update fast-growing coral with safeguards
    fast_pred(t) = fast_pred(t-1) + fast_growth - consumption_fast;
    if (fast_pred(t) < eps) fast_pred(t) = eps;
    if (fast_pred(t) > Type(60.0)) fast_pred(t) = Type(60.0); // Hard upper limit
    
    // 7. Slow-growing coral dynamics with safeguards
    Type r_slow_safe = r_slow;
    if (r_slow_safe < 0) r_slow_safe = 0;
    
    // Simplified growth model to avoid gradient issues
    Type slow_growth = r_slow_safe * slow_pred(t-1) * Type(0.5);
    
    // Update slow-growing coral with safeguards
    slow_pred(t) = slow_pred(t-1) + slow_growth - consumption_slow;
    if (slow_pred(t) < eps) slow_pred(t) = eps;
    if (slow_pred(t) > Type(40.0)) slow_pred(t) = Type(40.0); // Hard upper limit
  }
  
  // Calculate negative log-likelihood using lognormal distribution
  for (int t = 0; t < n_years; t++) {
    // Add observation error for COTS abundance (lognormal)
    if (!R_IsNA(asDouble(cots_dat(t))) && cots_dat(t) > 0) {
      nll -= dnorm(log(cots_dat(t)), log(cots_pred(t)), sigma_cots_adj, true);
    }
    
    // Add observation error for fast-growing coral cover (lognormal)
    if (!R_IsNA(asDouble(fast_dat(t))) && fast_dat(t) > 0) {
      nll -= dnorm(log(fast_dat(t)), log(fast_pred(t)), sigma_fast_adj, true);
    }
    
    // Add observation error for slow-growing coral cover (lognormal)
    if (!R_IsNA(asDouble(slow_dat(t))) && slow_dat(t) > 0) {
      nll -= dnorm(log(slow_dat(t)), log(slow_pred(t)), sigma_slow_adj, true);
    }
  }
  
  // Report model predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(consumption_fast_vec);
  REPORT(consumption_slow_vec);
  REPORT(nutr_multiplier_vec);
  
  return nll;
}
