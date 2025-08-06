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
  
  // PARAMETER SECTION - Using original parameters without new ones that cause issues
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
  
  // Define constants for the nutritional feedback mechanism
  Type nutrition_factor = Type(0.8);  // Coefficient for nutritional enhancement
  Type nutrition_half_sat = Type(0.3); // Half-saturation constant
  Type recruitment_delay_val = Type(2.0); // Time lag for recruitment
  Type density_mort_val = Type(0.4);  // Density-dependent mortality coefficient
  
  // Time series simulation
  for (int t = 1; t < n_years; t++) {
    // 1. Temperature effect on COTS reproduction (continuous response)
    Type temp_effect = Type(1.0);
    if (sst_dat(t-1) > T_crit) {
      // Smooth transition rather than binary effect
      temp_effect = Type(1.0) + T_effect * (sst_dat(t-1) - T_crit) / Type(2.0);
      temp_effect = std::min(temp_effect, Type(1.0) + T_effect); // Cap the effect
    }
    
    // 2. Functional response for coral consumption
    // Type II functional response for predation on fast-growing coral
    Type consumption_fast = (a_fast * cots_pred(t-1) * fast_pred(t-1)) / 
                           (Type(1.0) + a_fast * h_fast * fast_pred(t-1) + eps);
    
    // Type II functional response for predation on slow-growing coral
    Type consumption_slow = (a_slow * cots_pred(t-1) * slow_pred(t-1)) / 
                           (Type(1.0) + a_slow * h_slow * slow_pred(t-1) + eps);
    
    // Total coral consumption
    Type total_consumption = consumption_fast + consumption_slow;
    
    // 3. Nutritional feedback on reproduction (key improvement)
    // Michaelis-Menten type response of reproduction to consumption
    Type nutrition_effect = (nutrition_factor * total_consumption) / 
                           (nutrition_half_sat + total_consumption + eps);
    
    // 4. Delayed recruitment with proper delay implementation
    Type delayed_recruitment = Type(0.0);
    
    // Convert recruitment_delay to integer and ensure it's within bounds
    int delay = CppAD::Integer(recruitment_delay_val);
    delay = std::max(1, std::min(delay, t));
    
    // Calculate recruitment based on delayed population state
    if (t >= delay) {
      // Logistic growth with nutritional enhancement
      Type effective_r = r_cots * (Type(1.0) + nutrition_effect) * temp_effect;
      delayed_recruitment = effective_r * cots_pred(t-delay) * 
                           (Type(1.0) - cots_pred(t-delay) / (K_cots + eps));
    }
    
    // Ensure recruitment is non-negative
    delayed_recruitment = std::max(Type(0.0), delayed_recruitment);
    
    // 5. Density-dependent mortality (continuous response)
    // Increases non-linearly with population density
    Type base_mortality = m_cots * cots_pred(t-1);
    Type dd_mortality = (density_mort_val * pow(cots_pred(t-1), Type(2.0))) / 
                       (K_cots + cots_pred(t-1) + eps);
    
    // 6. Immigration effect
    Type cots_immigration = imm_effect * cotsimm_dat(t-1);
    
    // 7. Update COTS population
    cots_pred(t) = cots_pred(t-1) + delayed_recruitment - base_mortality - dd_mortality + cots_immigration;
    
    // Ensure population remains positive but not too small
    cots_pred(t) = std::max(Type(0.01), cots_pred(t));
    
    // 8. Fast-growing coral dynamics with competition
    Type fast_growth = r_fast * fast_pred(t-1) * 
                      (Type(1.0) - (fast_pred(t-1) + alpha_fs * slow_pred(t-1)) / (K_fast + eps));
    
    // Update fast-growing coral cover
    fast_pred(t) = fast_pred(t-1) + fast_growth - consumption_fast;
    
    // Ensure coral cover remains within bounds
    fast_pred(t) = std::max(Type(0.01), std::min(K_fast, fast_pred(t)));
    
    // 9. Slow-growing coral dynamics with competition
    Type slow_growth = r_slow * slow_pred(t-1) * 
                      (Type(1.0) - (slow_pred(t-1) + alpha_sf * fast_pred(t-1)) / (K_slow + eps));
    
    // Update slow-growing coral cover
    slow_pred(t) = slow_pred(t-1) + slow_growth - consumption_slow;
    
    // Ensure coral cover remains within bounds
    slow_pred(t) = std::max(Type(0.01), std::min(K_slow, slow_pred(t)));
  }
  
  // Calculate negative log-likelihood
  for (int t = 0; t < n_years; t++) {
    // Add observation error for COTS abundance
    if (!R_IsNA(asDouble(cots_dat(t))) && cots_dat(t) > Type(0.0)) {
      nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots, true);
    }
    
    // Add observation error for fast-growing coral cover
    if (!R_IsNA(asDouble(fast_dat(t))) && fast_dat(t) > Type(0.0)) {
      nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast, true);
    }
    
    // Add observation error for slow-growing coral cover
    if (!R_IsNA(asDouble(slow_dat(t))) && slow_dat(t) > Type(0.0)) {
      nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow, true);
    }
  }
  
  // Report model predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
