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
  
  // Define minimum values
  Type min_val = Type(0.001);
  
  // Add first observations to likelihood
  nll -= dnorm(log(cots_dat(0) + min_val), log(cots_pred(0) + min_val), sigma_obs_cots + min_val, true);
  nll -= dnorm(log(slow_dat(0) + min_val), log(slow_pred(0) + min_val), sigma_obs_slow + min_val, true);
  nll -= dnorm(log(fast_dat(0) + min_val), log(fast_pred(0) + min_val), sigma_obs_fast + min_val, true);
  
  // Loop through time steps to calculate predictions and likelihood
  for (int t = 1; t < n_years; t++) {
    // Get previous state
    Type cots_t1 = cots_pred(t-1);
    Type slow_t1 = slow_pred(t-1);
    Type fast_t1 = fast_pred(t-1);
    Type sst = sst_dat(t-1);
    Type cotsimm = cotsimm_dat(t-1);
    
    // Temperature effects (simplified)
    Type temp_effect_cots = exp(-0.1 * pow(sst - temp_opt_cots, 2));
    Type temp_effect_slow = exp(-0.1 * pow(sst - temp_opt_coral, 2));
    Type temp_effect_fast = exp(-0.1 * pow(sst - temp_opt_coral, 2));
    
    // Calculate predation rates
    Type pred_slow = alpha_slow * cots_t1 * slow_t1 / (h_slow + slow_t1) * (1.0 - pref_fast);
    Type pred_fast = alpha_fast * cots_t1 * fast_t1 / (h_fast + fast_t1) * pref_fast;
    
    // Calculate COTS population dynamics
    Type cots_growth = r_cots * cots_t1 * (1.0 - cots_t1 / K_cots) * temp_effect_cots;
    Type cots_mort = m_cots * cots_t1;
    Type cots_next = cots_t1 + cots_growth - cots_mort + cotsimm;
    cots_next = cots_next < min_val ? min_val : cots_next;
    
    // Calculate coral dynamics with competition
    // Use fixed competition parameters
    Type comp_slow_fast_val = Type(0.8);  // Competition effect of fast-growing on slow-growing corals
    Type comp_fast_slow_val = Type(0.5);  // Competition effect of slow-growing on fast-growing corals
    
    Type competition_slow = Type(1.0) - (slow_t1 + comp_slow_fast_val * fast_t1) / K_slow;
    competition_slow = competition_slow < Type(0.0) ? Type(0.0) : competition_slow;
    
    Type competition_fast = Type(1.0) - (fast_t1 + comp_fast_slow_val * slow_t1) / K_fast;
    competition_fast = competition_fast < Type(0.0) ? Type(0.0) : competition_fast;
    
    Type slow_growth = r_slow * slow_t1 * competition_slow * temp_effect_slow;
    Type slow_next = slow_t1 + slow_growth - pred_slow;
    slow_next = slow_next < min_val ? min_val : slow_next;
    
    Type fast_growth = r_fast * fast_t1 * competition_fast * temp_effect_fast;
    Type fast_next = fast_t1 + fast_growth - pred_fast;
    fast_next = fast_next < min_val ? min_val : fast_next;
    
    // Set predictions for the current time step
    cots_pred(t) = cots_next;
    slow_pred(t) = slow_next;
    fast_pred(t) = fast_next;
    
    // Add to negative log-likelihood
    nll -= dnorm(log(cots_dat(t) + min_val), log(cots_pred(t) + min_val), sigma_obs_cots + min_val, true);
    nll -= dnorm(log(slow_dat(t) + min_val), log(slow_pred(t) + min_val), sigma_obs_slow + min_val, true);
    nll -= dnorm(log(fast_dat(t) + min_val), log(fast_pred(t) + min_val), sigma_obs_fast + min_val, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
