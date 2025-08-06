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
  PARAMETER(recruitment_lag);         // Time lag for COTS recruitment (years)
  PARAMETER(nutrient_threshold);      // Threshold for nutrient-enhanced reproduction
  PARAMETER(density_mort_coef);       // Coefficient for density-dependent mortality
  PARAMETER(coral_recovery_threshold); // Threshold for delayed coral recovery
  
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
  Type eps = Type(1e-8);
  
  // Run the model for each time step
  for (int t = 1; t < n; t++) {
    // 1. Calculate temperature effect on COTS reproduction
    // Use a simpler linear response to temperature instead of Gaussian
    Type temp_diff = fabs(sst_dat(t-1) - sst_opt);
    Type temp_effect = 1.0 - beta_sst * (temp_diff / (sst_width + eps));
    temp_effect = temp_effect < 0.0 ? 0.0 : temp_effect;
    
    // 2. Calculate total coral cover (fast + slow) for density dependence
    Type total_coral = fast_pred(t-1) + slow_pred(t-1) + eps;
    
    // 3. Calculate food-dependent mortality modifier (increases when coral is scarce)
    // Simplified linear response
    Type mort_modifier = 1.0 + 0.5 * (1.0 - total_coral / (coral_threshold + total_coral));
    
    // 4. Calculate functional responses for COTS feeding on corals (Type II)
    // Simplified to avoid potential division issues
    Type consumption_denom = 1.0 + alpha_fast * h_fast * fast_pred(t-1) + alpha_slow * h_slow * slow_pred(t-1) + eps;
    Type consumption_fast = (alpha_fast * fast_pred(t-1) * cots_pred(t-1)) / consumption_denom;
    Type consumption_slow = (alpha_slow * slow_pred(t-1) * cots_pred(t-1)) / consumption_denom;
    
    // 5. Implement delayed recruitment effect with fixed 2-year lag
    Type recruitment = 0.0;
    
    if (t >= 2) {
      // Basic delayed recruitment with temperature effect
      Type historical_cots = cots_pred(t-2);
      
      // Simple nutrient effect based on immigration proxy (linear response)
      Type nutrient_effect = 1.0 + 0.5 * (cotsimm_dat(t-2) / (nutrient_threshold + cotsimm_dat(t-2)));
      
      // Calculate recruitment with density dependence
      recruitment = r_cots * historical_cots * (1.0 - historical_cots / K_cots) * temp_effect * nutrient_effect;
      recruitment = recruitment < 0.0 ? 0.0 : recruitment;
    }
    
    // 6. Calculate density-dependent mortality (simplified)
    Type density_ratio = cots_pred(t-1) / (K_cots + eps);
    Type density_mortality = m_cots * mort_modifier * cots_pred(t-1) * (1.0 + density_mort_coef * density_ratio);
    density_mortality = density_mortality < 0.0 ? 0.0 : density_mortality;
    
    // 7. Calculate immigration effect
    Type immigration = imm_effect * cotsimm_dat(t-1);
    immigration = immigration < 0.0 ? 0.0 : immigration;
    
    // 8. Update COTS abundance
    cots_pred(t) = cots_pred(t-1) + recruitment - density_mortality + immigration;
    cots_pred(t) = cots_pred(t) < eps ? eps : cots_pred(t);
    
    // 9. Calculate coral recovery modifiers (simplified linear response)
    Type fast_recovery = fast_pred(t-1) / (coral_recovery_threshold + fast_pred(t-1));
    Type slow_recovery = slow_pred(t-1) / (coral_recovery_threshold + slow_pred(t-1));
    
    // 10. Calculate coral growth with competition
    Type fast_ratio = (fast_pred(t-1) + 0.5 * slow_pred(t-1)) / (K_fast + eps);
    fast_ratio = fast_ratio > 1.0 ? 1.0 : fast_ratio;
    Type fast_growth = r_fast * fast_pred(t-1) * (1.0 - fast_ratio) * fast_recovery;
    
    Type slow_ratio = (slow_pred(t-1) + 0.3 * fast_pred(t-1)) / (K_slow + eps);
    slow_ratio = slow_ratio > 1.0 ? 1.0 : slow_ratio;
    Type slow_growth = r_slow * slow_pred(t-1) * (1.0 - slow_ratio) * slow_recovery;
    
    // 11. Update coral cover
    fast_pred(t) = fast_pred(t-1) + fast_growth - consumption_fast;
    slow_pred(t) = slow_pred(t-1) + slow_growth - consumption_slow;
    
    // 12. Ensure coral cover stays positive
    fast_pred(t) = fast_pred(t) < eps ? eps : fast_pred(t);
    slow_pred(t) = slow_pred(t) < eps ? eps : slow_pred(t);
    
    // 13. Apply upper bounds to prevent unrealistic values
    if (cots_pred(t) > K_cots * 2.0) cots_pred(t) = K_cots * 2.0;
    if (fast_pred(t) > K_fast * 1.2) fast_pred(t) = K_fast * 1.2;
    if (slow_pred(t) > K_slow * 1.2) slow_pred(t) = K_slow * 1.2;
  }
  
  // Calculate negative log-likelihood using lognormal distribution
  Type const_obs = Type(1e-4);
  
  for (int t = 0; t < n; t++) {
    // 14. COTS abundance likelihood
    Type cots_obs = cots_dat(t) + const_obs;
    Type cots_model = cots_pred(t) + const_obs;
    nll -= dnorm(log(cots_obs), log(cots_model), sigma_cots, true);
    
    // 15. Fast-growing coral cover likelihood
    Type fast_obs = fast_dat(t) + const_obs;
    Type fast_model = fast_pred(t) + const_obs;
    nll -= dnorm(log(fast_obs), log(fast_model), sigma_fast, true);
    
    // 16. Slow-growing coral cover likelihood
    Type slow_obs = slow_dat(t) + const_obs;
    Type slow_model = slow_pred(t) + const_obs;
    nll -= dnorm(log(slow_obs), log(slow_model), sigma_slow, true);
  }
  
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
  ADREPORT(nutrient_threshold);
  ADREPORT(density_mort_coef);
  
  return nll;
}
