#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA SECTION
  DATA_VECTOR(Year);                    // Year of observation
  DATA_VECTOR(sst_dat);                 // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);             // COTS larval immigration rate (individuals/m²/year)
  DATA_VECTOR(cots_dat);                // Observed COTS density (individuals/m²)
  DATA_VECTOR(fast_dat);                // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);                // Observed slow-growing coral cover (%)
  
  // PARAMETER SECTION
  PARAMETER(log_r_cots);                // Log of COTS population growth rate (year⁻¹)
  PARAMETER(log_K_cots);                // Log of COTS carrying capacity (individuals/m²)
  PARAMETER(log_temp_effect);           // Log of temperature effect on COTS reproduction (dimensionless)
  PARAMETER(log_temp_threshold);        // Log of temperature threshold for COTS reproduction (°C)
  PARAMETER(log_imm_effect);            // Log of effect of larval immigration on COTS recruitment (dimensionless)
  
  PARAMETER(log_r_fast);                // Log of intrinsic growth rate of fast-growing coral (year⁻¹)
  PARAMETER(log_r_slow);                // Log of intrinsic growth rate of slow-growing coral (year⁻¹)
  PARAMETER(log_K_fast);                // Log of carrying capacity of fast-growing coral (%)
  PARAMETER(log_K_slow);                // Log of carrying capacity of slow-growing coral (%)
  
  PARAMETER(log_a_fast);                // Log of attack rate on fast-growing coral (m²/individual/year)
  PARAMETER(log_a_slow);                // Log of attack rate on slow-growing coral (m²/individual/year)
  PARAMETER(log_h_fast);                // Log of handling time for fast-growing coral (year/%)
  PARAMETER(log_h_slow);                // Log of handling time for slow-growing coral (year/%)
  
  PARAMETER(log_coral_effect);          // Log of coral cover effect on COTS survival (dimensionless)
  
  PARAMETER(log_sigma_cots);            // Log of observation error SD for COTS
  PARAMETER(log_sigma_fast);            // Log of observation error SD for fast-growing coral
  PARAMETER(log_sigma_slow);            // Log of observation error SD for slow-growing coral
  
  // Parameters for density-dependent predation on COTS
  PARAMETER(log_pred_rate);             // Log of maximum predation rate on COTS (year⁻¹)
  PARAMETER(log_pred_half);             // Log of half-saturation constant for predation (individuals/m²)
  PARAMETER(log_pred_hill);             // Log of Hill coefficient for predator functional response
  
  // Transform parameters to natural scale
  Type r_cots = exp(log_r_cots);                // COTS population growth rate (year⁻¹)
  Type K_cots = exp(log_K_cots);                // COTS carrying capacity (individuals/m²)
  Type temp_effect = exp(log_temp_effect);      // Temperature effect on COTS reproduction (dimensionless)
  Type temp_threshold = exp(log_temp_threshold); // Temperature threshold for COTS reproduction (°C)
  Type imm_effect = exp(log_imm_effect);        // Effect of larval immigration on COTS recruitment (dimensionless)
  
  Type r_fast = exp(log_r_fast);                // Intrinsic growth rate of fast-growing coral (year⁻¹)
  Type r_slow = exp(log_r_slow);                // Intrinsic growth rate of slow-growing coral (year⁻¹)
  Type K_fast = exp(log_K_fast);                // Carrying capacity of fast-growing coral (%)
  Type K_slow = exp(log_K_slow);                // Carrying capacity of slow-growing coral (%)
  
  Type a_fast = exp(log_a_fast);                // Attack rate on fast-growing coral (m²/individual/year)
  Type a_slow = exp(log_a_slow);                // Attack rate on slow-growing coral (m²/individual/year)
  Type h_fast = exp(log_h_fast);                // Handling time for fast-growing coral (year/%)
  Type h_slow = exp(log_h_slow);                // Handling time for slow-growing coral (year/%)
  
  Type coral_effect = exp(log_coral_effect);    // Coral cover effect on COTS survival (dimensionless)
  
  Type sigma_cots = exp(log_sigma_cots);        // Observation error SD for COTS
  Type sigma_fast = exp(log_sigma_fast);        // Observation error SD for fast-growing coral
  Type sigma_slow = exp(log_sigma_slow);        // Observation error SD for slow-growing coral
  
  // Parameters for density-dependent predation
  Type pred_rate = exp(log_pred_rate);          // Maximum predation rate on COTS (year⁻¹)
  Type pred_half = exp(log_pred_half);          // Half-saturation constant for predation (individuals/m²)
  Type pred_hill = Type(2.0);                   // Fixed Hill coefficient for predator functional response
  
  // Set minimum standard deviations to prevent numerical issues
  Type min_sd = Type(0.01);
  if (sigma_cots < min_sd) sigma_cots = min_sd;
  if (sigma_fast < min_sd) sigma_fast = min_sd;
  if (sigma_slow < min_sd) sigma_slow = min_sd;
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize vectors for model predictions
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Initialize with first year's data
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Model equations for each time step
  for(int t = 1; t < n; t++) {
    // 1. Temperature effect on COTS reproduction (logistic function)
    Type temp_diff = sst_dat(t-1) - temp_threshold;
    // Limit extreme values to prevent numerical issues
    if (temp_diff > Type(10.0)) temp_diff = Type(10.0);
    if (temp_diff < Type(-10.0)) temp_diff = Type(-10.0);
    
    Type temp_factor = Type(1.0) / (Type(1.0) + exp(-temp_effect * temp_diff));
    
    // 2. Total coral cover (used for COTS survival)
    Type total_coral = fast_pred(t-1) + slow_pred(t-1) + eps;
    
    // 3. Coral-dependent survival factor for COTS (saturating function)
    Type survival_factor = total_coral / (total_coral + coral_effect);
    
    // 4. Density-dependent predation on COTS (Type III functional response)
    Type cots_density = cots_pred(t-1);
    
    // Calculate predation term - simplified to avoid numerical issues
    Type predation_term = Type(0.0);
    if (cots_density > eps) {
      Type x = cots_density / pred_half;
      // Limit x to prevent overflow
      if (x > Type(100.0)) x = Type(100.0);
      
      // Type III functional response (x²/(1+x²))
      predation_term = pred_rate * (x * x) / (Type(1.0) + (x * x));
    }
    
    // 5. COTS population dynamics with density dependence, temperature effect, and predation
    Type density_factor = Type(1.0);
    if (K_cots > eps) {
      density_factor = Type(1.0) - cots_pred(t-1) / K_cots;
    }
    if (density_factor < Type(0.0)) density_factor = Type(0.0);
    
    Type cots_growth = r_cots * cots_pred(t-1) * density_factor * temp_factor * survival_factor;
    Type immigration = imm_effect * cotsimm_dat(t-1);
    Type predation_loss = predation_term * cots_pred(t-1);
    
    // Update COTS population with safeguards
    cots_pred(t) = cots_pred(t-1) + cots_growth + immigration - predation_loss;
    if (cots_pred(t) < Type(0.0)) cots_pred(t) = Type(0.0);
    
    // 6. Functional responses for COTS predation on corals (Type II)
    Type denominator = Type(1.0) + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1);
    Type consumption_fast = (a_fast * cots_pred(t-1) * fast_pred(t-1)) / denominator;
    Type consumption_slow = (a_slow * cots_pred(t-1) * slow_pred(t-1)) / denominator;
    
    // 7. Coral growth with logistic growth and COTS predation
    Type fast_growth = r_fast * fast_pred(t-1) * (Type(1.0) - fast_pred(t-1) / K_fast);
    Type slow_growth = r_slow * slow_pred(t-1) * (Type(1.0) - slow_pred(t-1) / K_slow);
    
    fast_pred(t) = fast_pred(t-1) + fast_growth - consumption_fast;
    if (fast_pred(t) < Type(0.0)) fast_pred(t) = Type(0.0);
    
    slow_pred(t) = slow_pred(t-1) + slow_growth - consumption_slow;
    if (slow_pred(t) < Type(0.0)) slow_pred(t) = Type(0.0);
    
    // Add upper bounds to prevent extreme values
    Type max_cots = Type(10.0);  // Maximum reasonable COTS density
    Type max_coral = Type(100.0); // Maximum coral cover percentage
    
    if (cots_pred(t) > max_cots) cots_pred(t) = max_cots;
    if (fast_pred(t) > max_coral) fast_pred(t) = max_coral;
    if (slow_pred(t) > max_coral) slow_pred(t) = max_coral;
  }
  
  // Calculate negative log-likelihood using appropriate error distributions
  for(int t = 0; t < n; t++) {
    // 8. Log-normal likelihood for COTS (strictly positive data)
    if (cots_dat(t) > eps && cots_pred(t) > eps) {
      Type log_cots_obs = log(cots_dat(t));
      Type log_cots_pred = log(cots_pred(t));
      nll -= dnorm(log_cots_obs, log_cots_pred, sigma_cots, true);
    }
    
    // 9. Normal likelihood for coral cover percentages
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_fast, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_slow, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(r_cots);
  REPORT(K_cots);
  REPORT(temp_effect);
  REPORT(temp_threshold);
  REPORT(imm_effect);
  REPORT(r_fast);
  REPORT(r_slow);
  REPORT(K_fast);
  REPORT(K_slow);
  REPORT(a_fast);
  REPORT(a_slow);
  REPORT(h_fast);
  REPORT(h_slow);
  REPORT(coral_effect);
  REPORT(sigma_cots);
  REPORT(sigma_fast);
  REPORT(sigma_slow);
  REPORT(pred_rate);
  REPORT(pred_half);
  REPORT(pred_hill);
  
  return nll;
}
