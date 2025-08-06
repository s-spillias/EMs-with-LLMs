#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA SECTION
  DATA_VECTOR(Year);                      // Vector of years for time series data
  DATA_VECTOR(sst_dat);                   // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);               // COTS larval immigration rate (individuals/m²/year)
  DATA_VECTOR(cots_dat);                  // Observed COTS adult abundance (individuals/m²)
  DATA_VECTOR(fast_dat);                  // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);                  // Observed slow-growing coral cover (%)
  
  // PARAMETER SECTION
  PARAMETER(log_r_cots);                  // Log of COTS intrinsic growth rate (year⁻¹)
  PARAMETER(log_K_cots);                  // Log of COTS carrying capacity (individuals/m²)
  PARAMETER(log_m_cots);                  // Log of COTS natural mortality rate (year⁻¹)
  PARAMETER(log_alpha_fast);              // Log of COTS predation rate on fast-growing coral (m²/individual/year)
  PARAMETER(log_alpha_slow);              // Log of COTS predation rate on slow-growing coral (m²/individual/year)
  PARAMETER(log_r_fast);                  // Log of fast-growing coral intrinsic growth rate (year⁻¹)
  PARAMETER(log_r_slow);                  // Log of slow-growing coral intrinsic growth rate (year⁻¹)
  PARAMETER(log_K_fast);                  // Log of fast-growing coral carrying capacity (%)
  PARAMETER(log_K_slow);                  // Log of slow-growing coral carrying capacity (%)
  PARAMETER(log_beta_fast);               // Log of competition coefficient of slow on fast coral (dimensionless)
  PARAMETER(log_beta_slow);               // Log of competition coefficient of fast on slow coral (dimensionless)
  PARAMETER(log_temp_opt);                // Log of optimal temperature for COTS reproduction (°C)
  PARAMETER(log_temp_width);              // Log of temperature tolerance width for COTS reproduction (°C)
  PARAMETER(log_imm_effect);              // Log of scaling factor for immigration effect (dimensionless)
  PARAMETER(log_coral_threshold);         // Log of coral cover threshold affecting COTS survival (%)
  PARAMETER(log_coral_effect);            // Log of strength of coral effect on COTS survival (dimensionless)
  
  // Observation error standard deviations
  PARAMETER(log_sigma_cots);              // Log of SD for COTS observations
  PARAMETER(log_sigma_fast);              // Log of SD for fast coral observations
  PARAMETER(log_sigma_slow);              // Log of SD for slow coral observations
  
  // Initialize negative log-likelihood
  Type nll = Type(0.0);
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-6);
  
  // Transform parameters to natural scale with safety bounds
  Type r_cots = exp(log_r_cots);          // COTS intrinsic growth rate (year⁻¹)
  Type K_cots = exp(log_K_cots);          // COTS carrying capacity (individuals/m²)
  Type m_cots = exp(log_m_cots);          // COTS natural mortality rate (year⁻¹)
  Type alpha_fast = exp(log_alpha_fast);  // COTS predation rate on fast-growing coral (m²/individual/year)
  Type alpha_slow = exp(log_alpha_slow);  // COTS predation rate on slow-growing coral (m²/individual/year)
  Type r_fast = exp(log_r_fast);          // Fast-growing coral intrinsic growth rate (year⁻¹)
  Type r_slow = exp(log_r_slow);          // Slow-growing coral intrinsic growth rate (year⁻¹)
  Type K_fast = exp(log_K_fast);          // Fast-growing coral carrying capacity (%)
  Type K_slow = exp(log_K_slow);          // Slow-growing coral carrying capacity (%)
  Type beta_fast = exp(log_beta_fast);    // Competition coefficient of slow on fast coral (dimensionless)
  Type beta_slow = exp(log_beta_slow);    // Competition coefficient of fast on slow coral (dimensionless)
  Type temp_opt = exp(log_temp_opt);      // Optimal temperature for COTS reproduction (°C)
  Type temp_width = exp(log_temp_width);  // Temperature tolerance width for COTS reproduction (°C)
  Type imm_effect = exp(log_imm_effect);  // Scaling factor for immigration effect (dimensionless)
  Type coral_threshold = exp(log_coral_threshold); // Coral cover threshold affecting COTS survival (%)
  Type coral_effect = exp(log_coral_effect); // Strength of coral effect on COTS survival (dimensionless)
  
  // Observation error standard deviations with minimum values for stability
  Type sigma_cots = exp(log_sigma_cots) + Type(0.1);  // SD for COTS observations
  Type sigma_fast = exp(log_sigma_fast) + Type(1.0);  // SD for fast coral observations
  Type sigma_slow = exp(log_sigma_slow) + Type(1.0);  // SD for slow coral observations
  
  // Initialize vectors for model predictions
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Set initial values for first time step
  cots_pred(0) = cots_dat(0) + eps;
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Time series simulation with simplified dynamics for stability
  for(int t = 1; t < n; t++) {
    // 1. Temperature effect on COTS reproduction - bounded between 0.2 and 1.0
    Type temp_diff = (sst_dat(t-1) - temp_opt);
    Type temp_width_safe = temp_width + Type(1.0); // Ensure width is not too small
    Type temp_effect = Type(0.2) + Type(0.8) * exp(-0.5 * pow(temp_diff / temp_width_safe, 2));
    
    // 2. Coral effect on COTS survival - bounded between 0.2 and 1.0
    Type total_coral = fast_pred(t-1) + slow_pred(t-1) + eps;
    Type coral_survival = Type(0.2) + Type(0.8) * total_coral / (total_coral + coral_threshold + eps);
    
    // 3. COTS population dynamics with bounded growth
    // Simplified density dependence
    Type density_factor = Type(1.0);
    if (cots_pred(t-1) > K_cots) {
      density_factor = Type(0.5); // Reduce growth if above carrying capacity
    }
    
    // Calculate growth with bounded factors
    Type cots_growth = r_cots * cots_pred(t-1) * density_factor * temp_effect * coral_survival;
    cots_growth = cots_growth < Type(0.0) ? Type(0.0) : cots_growth;
    
    // Calculate mortality
    Type cots_mortality = m_cots * cots_pred(t-1);
    
    // Add immigration effect
    Type cots_immigration = imm_effect * cotsimm_dat(t-1);
    
    // Update COTS population with bounds
    cots_pred(t) = cots_pred(t-1) + cots_growth - cots_mortality + cots_immigration;
    cots_pred(t) = cots_pred(t) < Type(0.01) ? Type(0.01) : cots_pred(t);
    cots_pred(t) = cots_pred(t) > Type(5.0) ? Type(5.0) : cots_pred(t); // Upper bound for stability
    
    // 4. Coral predation - simplified functional response
    Type fast_predation = alpha_fast * cots_pred(t-1) * fast_pred(t-1) / (fast_pred(t-1) + Type(20.0) + eps);
    Type slow_predation = alpha_slow * cots_pred(t-1) * slow_pred(t-1) / (slow_pred(t-1) + Type(20.0) + eps);
    
    // Ensure predation doesn't exceed available coral
    fast_predation = fast_predation > fast_pred(t-1) * Type(0.5) ? fast_pred(t-1) * Type(0.5) : fast_predation;
    slow_predation = slow_predation > slow_pred(t-1) * Type(0.5) ? slow_pred(t-1) * Type(0.5) : slow_predation;
    
    // 5. Coral growth - simplified logistic growth
    // Simplified competition
    Type fast_growth = r_fast * fast_pred(t-1) * (Type(1.0) - fast_pred(t-1) / K_fast);
    Type slow_growth = r_slow * slow_pred(t-1) * (Type(1.0) - slow_pred(t-1) / K_slow);
    
    // Bound growth to prevent extreme changes
    fast_growth = fast_growth < -fast_pred(t-1) * Type(0.2) ? -fast_pred(t-1) * Type(0.2) : fast_growth;
    fast_growth = fast_growth > fast_pred(t-1) * Type(0.5) ? fast_pred(t-1) * Type(0.5) : fast_growth;
    
    slow_growth = slow_growth < -slow_pred(t-1) * Type(0.2) ? -slow_pred(t-1) * Type(0.2) : slow_growth;
    slow_growth = slow_growth > slow_pred(t-1) * Type(0.5) ? slow_pred(t-1) * Type(0.5) : slow_growth;
    
    // 6. Update coral populations
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_predation;
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_predation;
    
    // 7. Ensure coral cover stays within reasonable bounds
    fast_pred(t) = fast_pred(t) < Type(0.0) ? Type(0.0) : fast_pred(t);
    slow_pred(t) = slow_pred(t) < Type(0.0) ? Type(0.0) : slow_pred(t);
    fast_pred(t) = fast_pred(t) > Type(100.0) ? Type(100.0) : fast_pred(t);
    slow_pred(t) = slow_pred(t) > Type(100.0) ? Type(100.0) : slow_pred(t);
  }
  
  // Compute negative log-likelihood with robust error handling
  
  // 8. COTS abundance - use normal distribution on log scale with robust handling
  for(int t = 0; t < n; t++) {
    // Add a constant to prevent log(0)
    Type cots_obs = cots_dat(t) + Type(0.01);
    Type cots_model = cots_pred(t) + Type(0.01);
    
    // Use normal distribution on log-transformed data
    nll -= dnorm(log(cots_obs), log(cots_model), sigma_cots, true);
  }
  
  // 9. Fast-growing coral cover - normal distribution
  for(int t = 0; t < n; t++) {
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_fast, true);
  }
  
  // 10. Slow-growing coral cover - normal distribution
  for(int t = 0; t < n; t++) {
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_slow, true);
  }
  
  // Report model predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  // Report transformed parameters
  REPORT(r_cots);
  REPORT(K_cots);
  REPORT(m_cots);
  REPORT(alpha_fast);
  REPORT(alpha_slow);
  REPORT(r_fast);
  REPORT(r_slow);
  REPORT(K_fast);
  REPORT(K_slow);
  REPORT(beta_fast);
  REPORT(beta_slow);
  REPORT(temp_opt);
  REPORT(temp_width);
  REPORT(imm_effect);
  REPORT(coral_threshold);
  REPORT(coral_effect);
  
  return nll;
}
