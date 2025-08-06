#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA
  DATA_VECTOR(Year);                  // Years of observation
  DATA_VECTOR(cots_dat);              // Observed COTS abundance (individuals/m^2)
  DATA_VECTOR(slow_dat);              // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);              // Observed fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);               // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);           // COTS immigration rate (individuals/m^2/year)
  
  // PARAMETERS
  // COTS parameters
  PARAMETER(r_cots);                  // Intrinsic growth rate of COTS (year^-1)
  PARAMETER(K_cots);                  // Carrying capacity of COTS (individuals/m^2)
  PARAMETER(m_cots);                  // Natural mortality rate of COTS (year^-1)
  
  // Predation parameters
  PARAMETER(alpha_slow);              // Attack rate on slow-growing corals (m^2/individual/year)
  PARAMETER(alpha_fast);              // Attack rate on fast-growing corals (m^2/individual/year)
  PARAMETER(h_slow);                  // Half-saturation constant for slow-growing corals (%)
  PARAMETER(h_fast);                  // Half-saturation constant for fast-growing corals (%)
  PARAMETER(pref_fast);               // COTS preference for fast-growing corals (proportion)
  
  // Coral parameters
  PARAMETER(r_slow);                  // Intrinsic growth rate of slow-growing corals (year^-1)
  PARAMETER(r_fast);                  // Intrinsic growth rate of fast-growing corals (year^-1)
  PARAMETER(K_slow);                  // Carrying capacity of slow-growing corals (%)
  PARAMETER(K_fast);                  // Carrying capacity of fast-growing corals (%)
  
  // Temperature effect parameters
  PARAMETER(beta_cots_temp);          // Effect of temperature on COTS growth (per °C)
  PARAMETER(temp_opt_cots);           // Optimal temperature for COTS (°C)
  PARAMETER(beta_slow_temp);          // Effect of temperature on slow-growing coral growth (per °C)
  PARAMETER(beta_fast_temp);          // Effect of temperature on fast-growing coral growth (per °C)
  PARAMETER(temp_opt_coral);          // Optimal temperature for coral growth (°C)
  
  // Competition parameters
  PARAMETER(comp_effect);             // Competition effect between coral types
  
  // Error parameters
  PARAMETER(sigma_proc_cots);         // Process error SD for COTS
  PARAMETER(sigma_proc_slow);         // Process error SD for slow-growing corals
  PARAMETER(sigma_proc_fast);         // Process error SD for fast-growing corals
  PARAMETER(sigma_obs_cots);          // Observation error SD for COTS
  PARAMETER(sigma_obs_slow);          // Observation error SD for slow-growing corals
  PARAMETER(sigma_obs_fast);          // Observation error SD for fast-growing corals
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Get data dimensions
  int n_years = Year.size();
  
  // Initialize vectors for model predictions
  vector<Type> cots_pred(n_years);
  vector<Type> slow_pred(n_years);
  vector<Type> fast_pred(n_years);
  
  // Initialize state variables with first observation
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Constants to prevent numerical issues
  Type min_sd = Type(0.01);  // Minimum standard deviation
  Type min_val = Type(0.01); // Minimum value for state variables
  
  // Loop through time steps to calculate predictions and likelihood
  for (int t = 1; t < n_years; t++) {
    // Get previous state
    Type cots_t1 = cots_pred(t-1);
    Type slow_t1 = slow_pred(t-1);
    Type fast_t1 = fast_pred(t-1);
    Type sst = sst_dat(t-1);
    Type cotsimm = cotsimm_dat(t-1);
    
    // Ensure positive values for state variables
    cots_t1 = cots_t1 < min_val ? min_val : cots_t1;
    slow_t1 = slow_t1 < min_val ? min_val : slow_t1;
    fast_t1 = fast_t1 < min_val ? min_val : fast_t1;
    
    // 1. Calculate temperature effects using Gaussian response curves
    Type temp_effect_cots = exp(-0.5 * pow((sst - temp_opt_cots) / (1.0 / (beta_cots_temp + 0.01)), 2));
    Type temp_effect_slow = exp(-0.5 * pow((sst - temp_opt_coral) / (1.0 / (beta_slow_temp + 0.01)), 2));
    Type temp_effect_fast = exp(-0.5 * pow((sst - temp_opt_coral) / (1.0 / (beta_fast_temp + 0.01)), 2));
    
    // 2. Calculate total coral resource availability
    Type total_coral = slow_t1 + fast_t1;
    
    // 3. Calculate COTS predation rates using functional responses
    // Ensure half-saturation constants are positive
    Type h_slow_pos = h_slow < 0.1 ? 0.1 : h_slow;
    Type h_fast_pos = h_fast < 0.1 ? 0.1 : h_fast;
    
    // Modify predation rates based on temperature - COTS are more efficient predators at optimal temperatures
    Type pred_efficiency = 0.5 + 0.5 * temp_effect_cots; // Scales from 0.5 to 1.0 based on temperature
    
    // Calculate predation with safeguards against division by zero
    Type pred_slow = alpha_slow * cots_t1 * slow_t1 / (h_slow_pos + slow_t1) * (1.0 - pref_fast) * pred_efficiency;
    Type pred_fast = alpha_fast * cots_t1 * fast_t1 / (h_fast_pos + fast_t1) * pref_fast * pred_efficiency;
    
    // Ensure predation doesn't exceed available coral
    pred_slow = pred_slow > slow_t1 ? slow_t1 : pred_slow;
    pred_fast = pred_fast > fast_t1 ? fast_t1 : pred_fast;
    
    // 4. Calculate resource limitation for COTS (smooth transition as resources decline)
    Type resource_limitation = 1.0 - exp(-0.1 * total_coral);
    
    // 5. Calculate COTS population dynamics with density dependence, mortality, and immigration
    // Ensure carrying capacity is positive
    Type K_cots_pos = K_cots < 0.1 ? 0.1 : K_cots;
    
    // Calculate growth with safeguards
    Type density_effect = 1.0 - cots_t1 / K_cots_pos;
    density_effect = density_effect < 0 ? 0 : density_effect;
    
    Type cots_growth = r_cots * cots_t1 * density_effect * temp_effect_cots * resource_limitation;
    Type cots_mort = m_cots * cots_t1;
    
    // Ensure mortality doesn't exceed population
    cots_mort = cots_mort > cots_t1 ? cots_t1 : cots_mort;
    
    Type cots_next = cots_t1 + cots_growth - cots_mort + cotsimm;
    
    // Ensure non-negative population
    cots_next = cots_next < min_val ? min_val : cots_next;
    
    // 6. Calculate coral dynamics with logistic growth, COTS predation, and competition between coral types
    // Ensure carrying capacities are positive
    Type K_slow_pos = K_slow < 0.1 ? 0.1 : K_slow;
    Type K_fast_pos = K_fast < 0.1 ? 0.1 : K_fast;
    
    // Ensure competition effect is non-negative
    Type comp_effect_pos = comp_effect < 0 ? 0 : comp_effect;
    
    // Calculate competition terms with safeguards
    Type slow_competition = 1.0 - (slow_t1 / K_slow_pos) - comp_effect_pos * (fast_t1 / K_fast_pos);
    Type fast_competition = 1.0 - (fast_t1 / K_fast_pos) - comp_effect_pos * (slow_t1 / K_slow_pos);
    
    // Ensure competition terms don't cause negative growth
    slow_competition = slow_competition < 0 ? 0 : slow_competition;
    fast_competition = fast_competition < 0 ? 0 : fast_competition;
    
    Type slow_growth = r_slow * slow_t1 * slow_competition * temp_effect_slow;
    Type slow_next = slow_t1 + slow_growth - pred_slow;
    
    // Ensure non-negative cover
    slow_next = slow_next < min_val ? min_val : slow_next;
    
    Type fast_growth = r_fast * fast_t1 * fast_competition * temp_effect_fast;
    Type fast_next = fast_t1 + fast_growth - pred_fast;
    
    // Ensure non-negative cover
    fast_next = fast_next < min_val ? min_val : fast_next;
    
    // 7. Set predictions for the current time step
    cots_pred(t) = cots_next;
    slow_pred(t) = slow_next;
    fast_pred(t) = fast_next;
  }
  
  // Calculate negative log-likelihood using log-normal observation model
  for (int t = 0; t < n_years; t++) {
    // Ensure positive values for observations and predictions
    Type cots_obs = cots_dat(t) + min_val;
    Type slow_obs = slow_dat(t) + min_val;
    Type fast_obs = fast_dat(t) + min_val;
    
    Type cots_mod = cots_pred(t) + min_val;
    Type slow_mod = slow_pred(t) + min_val;
    Type fast_mod = fast_pred(t) + min_val;
    
    // Ensure positive standard deviations
    Type sigma_cots = sigma_obs_cots < min_sd ? min_sd : sigma_obs_cots;
    Type sigma_slow = sigma_obs_slow < min_sd ? min_sd : sigma_obs_slow;
    Type sigma_fast = sigma_obs_fast < min_sd ? min_sd : sigma_obs_fast;
    
    // Add to negative log-likelihood
    nll -= dnorm(log(cots_obs), log(cots_mod), sigma_cots, true);
    nll -= dnorm(log(slow_obs), log(slow_mod), sigma_slow, true);
    nll -= dnorm(log(fast_obs), log(fast_mod), sigma_fast, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
