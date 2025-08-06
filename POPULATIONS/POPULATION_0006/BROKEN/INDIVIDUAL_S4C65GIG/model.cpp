#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA
  DATA_VECTOR(Year);                  // Years of observation
  DATA_VECTOR(cots_dat);              // Observed COTS abundance (individuals/m^2)
  DATA_VECTOR(fast_dat);              // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);              // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);               // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);           // COTS larval immigration rate (individuals/m^2/year)
  
  // PARAMETERS
  PARAMETER(r_cots);                  // Intrinsic growth rate of COTS population (year^-1)
  PARAMETER(K_cots);                  // Carrying capacity of COTS population (individuals/m^2)
  PARAMETER(m_cots);                  // Natural mortality rate of COTS (year^-1)
  PARAMETER(r_fast);                  // Intrinsic growth rate of fast-growing coral (year^-1)
  PARAMETER(K_fast);                  // Maximum cover of fast-growing coral (%)
  PARAMETER(r_slow);                  // Intrinsic growth rate of slow-growing coral (year^-1)
  PARAMETER(K_slow);                  // Maximum cover of slow-growing coral (%)
  PARAMETER(a_fast);                  // Attack rate of COTS on fast-growing coral (m^2/individual/year)
  PARAMETER(a_slow);                  // Attack rate of COTS on slow-growing coral (m^2/individual/year)
  PARAMETER(h_fast);                  // Handling time for COTS feeding on fast-growing coral (% cover)
  PARAMETER(h_slow);                  // Handling time for COTS feeding on slow-growing coral (% cover)
  PARAMETER(temp_opt);                // Optimal temperature for COTS recruitment (°C)
  PARAMETER(temp_width);              // Temperature range width for COTS recruitment (°C)
  PARAMETER(imm_effect);              // Effect of larval immigration on COTS recruitment (dimensionless)
  PARAMETER(competition);             // Competition coefficient between coral types (dimensionless)
  PARAMETER(bleach_threshold);        // Temperature threshold for coral bleaching (°C)
  PARAMETER(bleach_mortality_fast);   // Mortality rate of fast-growing coral during bleaching (year^-1)
  PARAMETER(bleach_mortality_slow);   // Mortality rate of slow-growing coral during bleaching (year^-1)
  PARAMETER(sigma_cots);              // Observation error standard deviation for COTS abundance (log scale)
  PARAMETER(sigma_fast);              // Observation error standard deviation for fast-growing coral cover (log scale)
  PARAMETER(sigma_slow);              // Observation error standard deviation for slow-growing coral cover (log scale)
  PARAMETER(allee_threshold);         // Population threshold for Allee effect in COTS (individuals/m^2)
  PARAMETER(allee_strength);          // Strength of Allee effect in COTS population (dimensionless)
  PARAMETER(nutrient_threshold);      // Temperature threshold for nutrient runoff events (°C)
  PARAMETER(nutrient_effect);         // Maximum effect of nutrients on COTS recruitment (dimensionless)
  
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
  
  // Initialize with first year's data
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
    // Previous time step values
    Type cots_t0 = cots_pred(t-1);
    Type fast_t0 = fast_pred(t-1);
    Type slow_t0 = slow_pred(t-1);
    Type sst = sst_dat(t-1);
    Type cotsimm = cotsimm_dat(t-1);
    
    // Ensure positive values for state variables
    cots_t0 = cots_t0 < eps ? eps : cots_t0;
    fast_t0 = fast_t0 < eps ? eps : fast_t0;
    slow_t0 = slow_t0 < eps ? eps : slow_t0;
    
    // 1. Basic temperature effect on COTS recruitment
    // Simple quadratic function for temperature effect
    Type temp_diff = (sst - temp_opt) / temp_width;
    temp_diff = temp_diff > Type(3.0) ? Type(3.0) : (temp_diff < Type(-3.0) ? Type(-3.0) : temp_diff);
    Type temp_effect = Type(1.0) - Type(0.3) * temp_diff * temp_diff;
    temp_effect = temp_effect < Type(0.0) ? Type(0.0) : temp_effect;
    
    // 2. Allee effect - implemented as a simple threshold function
    Type allee_effect = Type(0.2);  // Default low value
    if (cots_t0 > allee_threshold) {
      allee_effect = Type(1.0);  // Full effect above threshold
    } else if (cots_t0 > Type(0.5) * allee_threshold) {
      allee_effect = Type(0.6);  // Intermediate effect
    }
    
    // 3. Nutrient effect - implemented as a simple threshold function
    Type nutrient_pulse = Type(1.0);  // Default baseline
    if (sst > nutrient_threshold) {
      nutrient_pulse = Type(1.0) + nutrient_effect;  // Enhanced effect above threshold
    }
    
    // 4. Simplified predation function
    Type pred_fast = a_fast * cots_t0 * fast_t0 / (Type(10.0) + fast_t0 + slow_t0);
    Type pred_slow = a_slow * cots_t0 * slow_t0 / (Type(10.0) + fast_t0 + slow_t0);
    
    // 5. Simplified bleaching effect
    Type bleach_effect = Type(0.0);  // Default no bleaching
    if (sst > bleach_threshold) {
      bleach_effect = Type(1.0);  // Full bleaching above threshold
    } else if (sst > bleach_threshold - Type(1.0)) {
      bleach_effect = Type(0.5);  // Partial bleaching near threshold
    }
    
    // 6. COTS population dynamics
    // Modified growth rate with Allee effect and nutrient pulse
    Type growth_factor = r_cots * temp_effect * allee_effect * nutrient_pulse;
    
    // Simple logistic growth
    Type cots_growth = growth_factor * cots_t0 * (Type(1.0) - cots_t0 / K_cots);
    
    // Simple immigration term
    Type imm_term = imm_effect * cotsimm / Type(10.0);
    
    // Simple mortality term
    Type mort_term = m_cots * cots_t0;
    
    // Update COTS abundance with bounds
    cots_pred(t) = cots_t0 + cots_growth - mort_term + imm_term;
    cots_pred(t) = cots_pred(t) < eps ? eps : cots_pred(t);
    cots_pred(t) = cots_pred(t) > K_cots * Type(2.0) ? K_cots * Type(2.0) : cots_pred(t);
    
    // 7. Coral dynamics
    // Fast-growing coral
    Type fast_growth = r_fast * fast_t0 * (Type(1.0) - fast_t0 / K_fast);
    Type fast_bleaching = bleach_mortality_fast * bleach_effect * fast_t0;
    
    fast_pred(t) = fast_t0 + fast_growth - pred_fast - fast_bleaching;
    fast_pred(t) = fast_pred(t) < eps ? eps : fast_pred(t);
    fast_pred(t) = fast_pred(t) > K_fast ? K_fast : fast_pred(t);
    
    // Slow-growing coral
    Type slow_growth = r_slow * slow_t0 * (Type(1.0) - slow_t0 / K_slow);
    Type slow_bleaching = bleach_mortality_slow * bleach_effect * slow_t0;
    
    slow_pred(t) = slow_t0 + slow_growth - pred_slow - slow_bleaching;
    slow_pred(t) = slow_pred(t) < eps ? eps : slow_pred(t);
    slow_pred(t) = slow_pred(t) > K_slow ? K_slow : slow_pred(t);
  }
  
  // Calculate negative log-likelihood
  for (int t = 0; t < n_years; t++) {
    // Add small constant to data and predictions to handle zeros
    Type cots_obs = cots_dat(t) + eps;
    Type cots_mod = cots_pred(t) + eps;
    Type fast_obs = fast_dat(t) + eps;
    Type fast_mod = fast_pred(t) + eps;
    Type slow_obs = slow_dat(t) + eps;
    Type slow_mod = slow_pred(t) + eps;
    
    // Log-normal likelihood for COTS abundance
    nll -= dnorm(log(cots_obs), log(cots_mod), sigma_cots_adj, true);
    
    // Log-normal likelihood for fast-growing coral cover
    nll -= dnorm(log(fast_obs), log(fast_mod), sigma_fast_adj, true);
    
    // Log-normal likelihood for slow-growing coral cover
    nll -= dnorm(log(slow_obs), log(slow_mod), sigma_slow_adj, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
