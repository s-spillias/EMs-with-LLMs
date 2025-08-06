#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA SECTION
  DATA_VECTOR(Year);                    // Years of observation
  DATA_VECTOR(sst_dat);                 // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);             // COTS larval immigration rate (individuals/m²/year)
  DATA_VECTOR(cots_dat);                // Observed adult COTS density (individuals/m²)
  DATA_VECTOR(fast_dat);                // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);                // Observed slow-growing coral cover (%)
  
  // PARAMETER SECTION
  PARAMETER(log_r_cots);                // Log of COTS intrinsic growth rate (year⁻¹)
  PARAMETER(log_K_cots);                // Log of COTS carrying capacity (individuals/m²)
  PARAMETER(log_alpha_fast);            // Log of COTS predation rate on fast coral (m²/individual/year)
  PARAMETER(log_alpha_slow);            // Log of COTS predation rate on slow coral (m²/individual/year)
  PARAMETER(log_r_fast);                // Log of fast coral intrinsic growth rate (year⁻¹)
  PARAMETER(log_r_slow);                // Log of slow coral intrinsic growth rate (year⁻¹)
  PARAMETER(log_K_fast);                // Log of fast coral carrying capacity (%)
  PARAMETER(log_K_slow);                // Log of slow coral carrying capacity (%)
  PARAMETER(log_temp_opt);              // Log of optimal temperature for COTS reproduction (°C)
  PARAMETER(log_temp_width);            // Log of temperature tolerance width for COTS (°C)
  PARAMETER(log_imm_effect);            // Log of effect of larval immigration on COTS population
  PARAMETER(log_coral_threshold);       // Log of coral cover threshold for COTS survival (%)
  PARAMETER(log_competition);           // Log of competition coefficient between coral types
  
  PARAMETER(log_sigma_cots);            // Log of observation error SD for COTS
  PARAMETER(log_sigma_fast);            // Log of observation error SD for fast coral
  PARAMETER(log_sigma_slow);            // Log of observation error SD for slow coral
  
  // Transform parameters to natural scale
  Type r_cots = exp(log_r_cots);                // COTS intrinsic growth rate (year⁻¹)
  Type K_cots = exp(log_K_cots);                // COTS carrying capacity (individuals/m²)
  Type alpha_fast = exp(log_alpha_fast);        // COTS predation rate on fast coral (m²/individual/year)
  Type alpha_slow = exp(log_alpha_slow);        // COTS predation rate on slow coral (m²/individual/year)
  Type r_fast = exp(log_r_fast);                // Fast coral intrinsic growth rate (year⁻¹)
  Type r_slow = exp(log_r_slow);                // Slow coral intrinsic growth rate (year⁻¹)
  Type K_fast = exp(log_K_fast);                // Fast coral carrying capacity (%)
  Type K_slow = exp(log_K_slow);                // Slow coral carrying capacity (%)
  Type temp_opt = exp(log_temp_opt);            // Optimal temperature for COTS reproduction (°C)
  Type temp_width = exp(log_temp_width) + Type(0.1);  // Temperature tolerance width (°C)
  Type imm_effect = exp(log_imm_effect);        // Effect of larval immigration on COTS population
  Type coral_threshold = exp(log_coral_threshold); // Coral cover threshold for COTS survival (%)
  Type competition = exp(log_competition);      // Competition coefficient between coral types
  
  // Observation error standard deviations with minimum values
  Type sigma_cots = exp(log_sigma_cots) + Type(0.1);  // Observation error SD for COTS
  Type sigma_fast = exp(log_sigma_fast) + Type(0.1);  // Observation error SD for fast coral
  Type sigma_slow = exp(log_sigma_slow) + Type(0.1);  // Observation error SD for slow coral
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-4);
  
  // Initialize negative log-likelihood
  Type nll = Type(0.0);
  
  // Initialize vectors for model predictions
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Initialize with first year's data
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Simple model equations
  for(int i = 1; i < n; i++) {
    // 1. Temperature effect on COTS reproduction (simplified Gaussian response)
    Type temp_diff = (sst_dat(i-1) - temp_opt) / temp_width;
    Type temp_effect = exp(-0.5 * pow(temp_diff, 2));
    
    // 2. Food limitation effect (combined coral cover)
    Type total_coral = fast_pred(i-1) + slow_pred(i-1) + eps;
    Type food_limitation = 1.0 - exp(-0.05 * total_coral);
    
    // 3. Coral threshold effect on COTS survival
    Type survival_effect = 1.0 / (1.0 + exp(-1.0 * (total_coral - coral_threshold)));
    
    // 4. COTS population dynamics with immigration effect
    Type density_effect = 1.0 - cots_pred(i-1) / (K_cots * food_limitation + eps);
    if (density_effect < 0.0) density_effect = 0.0;
    
    Type cots_growth = r_cots * cots_pred(i-1) * density_effect * temp_effect * survival_effect;
    Type immigration = imm_effect * cotsimm_dat(i-1);
    
    cots_pred(i) = cots_pred(i-1) + cots_growth + immigration;
    if (cots_pred(i) < eps) cots_pred(i) = eps;
    
    // 5. Fast-growing coral dynamics with COTS predation
    Type fast_competition = (fast_pred(i-1) + competition * slow_pred(i-1)) / (K_fast + eps);
    if (fast_competition > 1.0) fast_competition = 1.0;
    
    Type fast_growth = r_fast * fast_pred(i-1) * (1.0 - fast_competition);
    Type fast_predation = alpha_fast * cots_pred(i-1) * fast_pred(i-1) / (total_coral);
    if (fast_predation > fast_pred(i-1)) fast_predation = fast_pred(i-1);
    
    fast_pred(i) = fast_pred(i-1) + fast_growth - fast_predation;
    if (fast_pred(i) < eps) fast_pred(i) = eps;
    
    // 6. Slow-growing coral dynamics with COTS predation
    Type slow_competition = (slow_pred(i-1) + competition * fast_pred(i-1)) / (K_slow + eps);
    if (slow_competition > 1.0) slow_competition = 1.0;
    
    Type slow_growth = r_slow * slow_pred(i-1) * (1.0 - slow_competition);
    Type slow_predation = alpha_slow * cots_pred(i-1) * slow_pred(i-1) / (total_coral);
    if (slow_predation > slow_pred(i-1)) slow_predation = slow_pred(i-1);
    
    slow_pred(i) = slow_pred(i-1) + slow_growth - slow_predation;
    if (slow_pred(i) < eps) slow_pred(i) = eps;
  }
  
  // Likelihood calculations using log-normal distribution
  for(int i = 0; i < n; i++) {
    // Add small constant to prevent log(0)
    Type cots_obs = cots_dat(i) + eps;
    Type fast_obs = fast_dat(i) + eps;
    Type slow_obs = slow_dat(i) + eps;
    
    Type cots_model = cots_pred(i) + eps;
    Type fast_model = fast_pred(i) + eps;
    Type slow_model = slow_pred(i) + eps;
    
    // Log-normal likelihood for COTS
    nll -= dnorm(log(cots_obs), log(cots_model), sigma_cots, true);
    
    // Log-normal likelihood for fast coral
    nll -= dnorm(log(fast_obs), log(fast_model), sigma_fast, true);
    
    // Log-normal likelihood for slow coral
    nll -= dnorm(log(slow_obs), log(slow_model), sigma_slow, true);
  }
  
  // Report objective function value
  REPORT(nll);
  
  // Report predictions and parameters
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(r_cots);
  REPORT(K_cots);
  REPORT(alpha_fast);
  REPORT(alpha_slow);
  REPORT(r_fast);
  REPORT(r_slow);
  REPORT(K_fast);
  REPORT(K_slow);
  REPORT(temp_opt);
  REPORT(temp_width);
  REPORT(imm_effect);
  REPORT(coral_threshold);
  REPORT(competition);
  REPORT(sigma_cots);
  REPORT(sigma_fast);
  REPORT(sigma_slow);
  
  return nll;
}
