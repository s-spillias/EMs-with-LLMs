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
  PARAMETER(nutrient_effect);         // Effect of nutrients on COTS recruitment
  PARAMETER(nutrient_threshold);      // SST threshold for nutrient effects
  PARAMETER(pred_refuge);             // Predation refuge coefficient
  
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
  
  // Run the model for each time step
  for (int t = 1; t < n; t++) {
    // Basic variables to prevent numerical issues
    Type cots_t1 = cots_pred(t-1);
    if (cots_t1 < 0.01) cots_t1 = 0.01;
    
    Type fast_t1 = fast_pred(t-1);
    if (fast_t1 < 0.01) fast_t1 = 0.01;
    
    Type slow_t1 = slow_pred(t-1);
    if (slow_t1 < 0.01) slow_t1 = 0.01;
    
    // 1. Simple temperature effect
    Type temp_effect = 1.0;
    if (sst_dat(t-1) > 27.0) {
      temp_effect = 1.2;
    }
    
    // 2. Simple nutrient effect
    Type nutrient_modifier = 1.0;
    if (sst_dat(t-1) > 27.5) {
      nutrient_modifier = 1.3;
    }
    
    // 3. Total coral cover
    Type total_coral = fast_t1 + slow_t1;
    
    // 4. Simple mortality modifier
    Type mort_modifier = 1.0;
    if (total_coral < 5.0) {
      mort_modifier = 1.2;
    }
    
    // 5. Simple predation refuge
    Type refuge_effect = 1.0;
    if (cots_t1 > 1.0) {
      refuge_effect = 0.8;
    }
    
    // 6. Simple consumption rates
    Type consumption_fast = 0.1 * fast_t1 * cots_t1;
    if (consumption_fast > 0.9 * fast_t1) {
      consumption_fast = 0.9 * fast_t1;
    }
    
    Type consumption_slow = 0.05 * slow_t1 * cots_t1;
    if (consumption_slow > 0.9 * slow_t1) {
      consumption_slow = 0.9 * slow_t1;
    }
    
    // 7. COTS population dynamics
    Type cots_growth = 0.5 * temp_effect * nutrient_modifier * cots_t1;
    if (cots_t1 > 2.0) {
      cots_growth = 0.2 * temp_effect * nutrient_modifier * cots_t1;
    }
    
    Type cots_mortality = 0.3 * mort_modifier * refuge_effect * cots_t1;
    Type cots_immigration = 0.1 * cotsimm_dat(t-1);
    
    // 8. Update COTS abundance
    cots_pred(t) = cots_t1 + cots_growth - cots_mortality + cots_immigration;
    if (cots_pred(t) < 0.01) cots_pred(t) = 0.01;
    
    // 9. Coral growth
    Type fast_growth = 0.4 * fast_t1 * (1.0 - fast_t1 / 60.0);
    if (fast_growth < 0.0) fast_growth = 0.0;
    
    Type slow_growth = 0.1 * slow_t1 * (1.0 - slow_t1 / 40.0);
    if (slow_growth < 0.0) slow_growth = 0.0;
    
    // 10. Update coral cover
    fast_pred(t) = fast_t1 + fast_growth - consumption_fast;
    if (fast_pred(t) < 0.01) fast_pred(t) = 0.01;
    
    slow_pred(t) = slow_t1 + slow_growth - consumption_slow;
    if (slow_pred(t) < 0.01) slow_pred(t) = 0.01;
  }
  
  // Calculate negative log-likelihood
  for (int t = 0; t < n; t++) {
    // Add small constant to prevent log(0)
    Type cots_obs = cots_dat(t) + 0.1;
    Type cots_model = cots_pred(t) + 0.1;
    nll -= dnorm(log(cots_obs), log(cots_model), 0.3, true);
    
    Type fast_obs = fast_dat(t) + 0.1;
    Type fast_model = fast_pred(t) + 0.1;
    nll -= dnorm(log(fast_obs), log(fast_model), 0.3, true);
    
    Type slow_obs = slow_dat(t) + 0.1;
    Type slow_model = slow_pred(t) + 0.1;
    nll -= dnorm(log(slow_obs), log(slow_model), 0.3, true);
  }
  
  // REPORT SECTION
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
