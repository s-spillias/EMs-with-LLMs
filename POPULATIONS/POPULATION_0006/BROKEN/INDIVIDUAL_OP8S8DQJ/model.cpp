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
  
  // New parameters for threshold-based outbreak dynamics
  PARAMETER(outbreak_threshold);      // Environmental threshold for triggering outbreaks
  PARAMETER(outbreak_steepness);      // Steepness of the outbreak response curve
  PARAMETER(time_lag);                // Time lag for environmental conditions to affect reproduction
  PARAMETER(density_mort_coef);       // Coefficient for density-dependent mortality
  
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
  Type eps = Type(0.01);
  
  // Run the model for each time step
  for (int t = 1; t < n; t++) {
    // 1. Calculate temperature effect on COTS reproduction
    Type temp_effect = exp(-0.5 * pow((sst_dat(t-1) - sst_opt) / (sst_width + eps), 2));
    
    // 2. Calculate total coral cover (fast + slow) for density dependence
    Type total_coral = fast_pred(t-1) + slow_pred(t-1) + eps;
    
    // 3. Calculate food-dependent mortality modifier (increases when coral is scarce)
    Type mort_modifier = 1.0 + 0.5 / (1.0 + exp(0.2 * (total_coral - coral_threshold)));
    
    // 4. Calculate functional responses for COTS feeding on corals (Type II)
    Type denominator = 1.0 + alpha_fast * h_fast * fast_pred(t-1) + alpha_slow * h_slow * slow_pred(t-1) + eps;
    Type consumption_fast = (alpha_fast * fast_pred(t-1) * cots_pred(t-1)) / denominator;
    Type consumption_slow = (alpha_slow * slow_pred(t-1) * cots_pred(t-1)) / denominator;
    
    // 5. Calculate environmental conditions for outbreak potential
    // Use a fixed lag of 2 years
    Type env_condition = 0.0;
    int lag_index = t - 2;
    
    if (lag_index >= 0) {
      // Calculate lagged temperature effect
      Type lagged_temp_effect = exp(-0.5 * pow((sst_dat(lag_index) - sst_opt) / (sst_width + eps), 2));
      
      // Get lagged coral cover
      Type lagged_coral = 0.0;
      if (lag_index > 0) {
        lagged_coral = fast_pred(lag_index-1) + slow_pred(lag_index-1) + eps;
      } else {
        lagged_coral = fast_pred(0) + slow_pred(0) + eps;
      }
      
      // Calculate environmental condition
      Type max_coral = K_fast + K_slow + eps;
      env_condition = lagged_temp_effect * (lagged_coral / max_coral);
    } else {
      // If not enough history, use current conditions
      Type max_coral = K_fast + K_slow + eps;
      env_condition = temp_effect * (total_coral / max_coral);
    }
    
    // 6. Calculate outbreak potential using a simplified sigmoid function
    Type steepness = 5.0; // Fixed value to avoid instability
    Type outbreak_diff = env_condition - outbreak_threshold;
    Type outbreak_potential = 0.5 * (1.0 + tanh(steepness * outbreak_diff));
    
    // 7. Calculate COTS population dynamics with outbreak potential
    Type resource_factor = total_coral / (K_fast + K_slow + eps);
    resource_factor = resource_factor > 1.0 ? 1.0 : resource_factor;
    
    Type effective_K = K_cots * resource_factor;
    effective_K = effective_K < eps ? eps : effective_K;
    
    Type logistic_term = 1.0 - cots_pred(t-1) / effective_K;
    logistic_term = logistic_term < -1.0 ? -1.0 : logistic_term;
    
    // Calculate growth with outbreak effect
    Type growth_multiplier = 1.0 + 0.5 * outbreak_potential;
    Type cots_growth = r_cots * growth_multiplier * temp_effect * cots_pred(t-1) * logistic_term;
    
    // 8. Calculate density-dependent mortality
    Type density_ratio = cots_pred(t-1) / (K_cots + eps);
    density_ratio = density_ratio > 1.0 ? 1.0 : density_ratio;
    
    Type density_effect = 1.0 + density_mort_coef * density_ratio;
    Type cots_mortality = m_cots * mort_modifier * density_effect * cots_pred(t-1);
    
    // 9. Calculate immigration effect
    Type cots_immigration = imm_effect * cotsimm_dat(t-1);
    
    // 10. Update COTS abundance
    cots_pred(t) = cots_pred(t-1) + cots_growth - cots_mortality + cots_immigration;
    
    // Ensure positive values and cap at reasonable maximum
    cots_pred(t) = cots_pred(t) < eps ? eps : cots_pred(t);
    cots_pred(t) = cots_pred(t) > 3.0 ? 3.0 : cots_pred(t);
    
    // 11. Calculate coral dynamics
    Type fast_growth = r_fast * fast_pred(t-1) * (1.0 - (fast_pred(t-1) + 0.5 * slow_pred(t-1)) / (K_fast + eps));
    Type slow_growth = r_slow * slow_pred(t-1) * (1.0 - (slow_pred(t-1) + 0.3 * fast_pred(t-1)) / (K_slow + eps));
    
    // 12. Update coral cover
    fast_pred(t) = fast_pred(t-1) + fast_growth - consumption_fast;
    slow_pred(t) = slow_pred(t-1) + slow_growth - consumption_slow;
    
    // Ensure coral cover stays positive and within reasonable bounds
    fast_pred(t) = fast_pred(t) < eps ? eps : fast_pred(t);
    fast_pred(t) = fast_pred(t) > K_fast * 1.1 ? K_fast * 1.1 : fast_pred(t);
    
    slow_pred(t) = slow_pred(t) < eps ? eps : slow_pred(t);
    slow_pred(t) = slow_pred(t) > K_slow * 1.1 ? K_slow * 1.1 : slow_pred(t);
  }
  
  // Calculate negative log-likelihood using lognormal distribution
  Type const_obs = Type(0.01);
  
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
  ADREPORT(outbreak_threshold);
  ADREPORT(outbreak_steepness);
  ADREPORT(time_lag);
  ADREPORT(density_mort_coef);
  
  return nll;
}
