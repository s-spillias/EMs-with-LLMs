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
  PARAMETER(r_cots);                  // Intrinsic growth rate of COTS (year^-1)
  PARAMETER(K_cots);                  // Carrying capacity of COTS (individuals/m^2)
  PARAMETER(m_cots);                  // Natural mortality rate of COTS (year^-1)
  PARAMETER(alpha_slow);              // Attack rate on slow-growing corals (m^2/individual/year)
  PARAMETER(alpha_fast);              // Attack rate on fast-growing corals (m^2/individual/year)
  PARAMETER(h_slow);                  // Half-saturation constant for slow-growing corals (%)
  PARAMETER(h_fast);                  // Half-saturation constant for fast-growing corals (%)
  PARAMETER(pref_fast);               // COTS preference for fast-growing corals (proportion)
  PARAMETER(r_slow);                  // Intrinsic growth rate of slow-growing corals (year^-1)
  PARAMETER(r_fast);                  // Intrinsic growth rate of fast-growing corals (year^-1)
  PARAMETER(K_slow);                  // Carrying capacity of slow-growing corals (%)
  PARAMETER(K_fast);                  // Carrying capacity of fast-growing corals (%)
  PARAMETER(K_total);                 // Total carrying capacity for all coral types (%)
  PARAMETER(comp_fast_on_slow);       // Competitive effect of fast-growing corals on slow-growing corals
  PARAMETER(comp_slow_on_fast);       // Competitive effect of slow-growing corals on fast-growing corals
  PARAMETER(beta_cots_temp);          // Effect of temperature on COTS growth (per °C)
  PARAMETER(temp_opt_cots);           // Optimal temperature for COTS (°C)
  PARAMETER(beta_slow_temp);          // Effect of temperature on slow-growing coral growth (per °C)
  PARAMETER(beta_fast_temp);          // Effect of temperature on fast-growing coral growth (per °C)
  PARAMETER(temp_opt_coral);          // Optimal temperature for coral growth (°C)
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
  Type eps = Type(0.01);
  
  // Add first observations to likelihood
  nll -= dnorm(log(cots_dat(0) + eps), log(cots_pred(0) + eps), Type(0.3), true);
  nll -= dnorm(log(slow_dat(0) + eps), log(slow_pred(0) + eps), Type(0.3), true);
  nll -= dnorm(log(fast_dat(0) + eps), log(fast_pred(0) + eps), Type(0.3), true);
  
  // Loop through time steps to calculate predictions and likelihood
  for (int t = 1; t < n_years; t++) {
    // Get previous state
    Type cots_t1 = cots_pred(t-1);
    Type slow_t1 = slow_pred(t-1);
    Type fast_t1 = fast_pred(t-1);
    
    // Ensure non-negative state variables
    cots_t1 = cots_t1 < eps ? eps : cots_t1;
    slow_t1 = slow_t1 < eps ? eps : slow_t1;
    fast_t1 = fast_t1 < eps ? eps : fast_t1;
    
    // Calculate total coral cover
    Type total_coral = slow_t1 + fast_t1;
    
    // Calculate space limitation (simplified)
    Type space_limit = Type(1.0) - total_coral / (K_total + eps);
    space_limit = space_limit < Type(0.0) ? Type(0.0) : space_limit;
    
    // Calculate competition effects (simplified)
    Type comp_effect_slow = Type(1.0) - comp_fast_on_slow * fast_t1 / (K_total + eps);
    comp_effect_slow = comp_effect_slow < Type(0.2) ? Type(0.2) : comp_effect_slow;
    
    Type comp_effect_fast = Type(1.0) - comp_slow_on_fast * slow_t1 / (K_total + eps);
    comp_effect_fast = comp_effect_fast < Type(0.2) ? Type(0.2) : comp_effect_fast;
    
    // Calculate coral growth with competition
    Type slow_growth = r_slow * slow_t1 * (Type(1.0) - slow_t1 / (K_slow + eps)) * comp_effect_slow * space_limit;
    Type fast_growth = r_fast * fast_t1 * (Type(1.0) - fast_t1 / (K_fast + eps)) * comp_effect_fast * space_limit;
    
    // Calculate COTS predation
    Type pred_slow = alpha_slow * cots_t1 * slow_t1 / (h_slow + total_coral) * (Type(1.0) - pref_fast);
    Type pred_fast = alpha_fast * cots_t1 * fast_t1 / (h_fast + total_coral) * pref_fast;
    
    // Limit predation
    pred_slow = pred_slow > Type(0.5) * slow_t1 ? Type(0.5) * slow_t1 : pred_slow;
    pred_fast = pred_fast > Type(0.5) * fast_t1 ? Type(0.5) * fast_t1 : pred_fast;
    
    // Calculate COTS population dynamics
    Type cots_growth = r_cots * cots_t1 * (Type(1.0) - cots_t1 / (K_cots + eps));
    Type cots_mort = m_cots * cots_t1;
    
    // Calculate next states
    Type cots_next = cots_t1 + cots_growth - cots_mort + cotsimm_dat(t-1);
    Type slow_next = slow_t1 + slow_growth - pred_slow;
    Type fast_next = fast_t1 + fast_growth - pred_fast;
    
    // Ensure non-negative values
    cots_next = cots_next < eps ? eps : cots_next;
    slow_next = slow_next < eps ? eps : slow_next;
    fast_next = fast_next < eps ? eps : fast_next;
    
    // Set predictions
    cots_pred(t) = cots_next;
    slow_pred(t) = slow_next;
    fast_pred(t) = fast_next;
    
    // Add to negative log-likelihood
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), Type(0.3), true);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), Type(0.3), true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), Type(0.3), true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
