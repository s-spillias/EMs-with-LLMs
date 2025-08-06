#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA SECTION
  DATA_VECTOR(Year);                  // Years of observation
  DATA_VECTOR(cots_dat);              // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);              // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);              // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);               // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);           // COTS larval immigration rate (individuals/m2/year)
  
  // PARAMETER SECTION
  // COTS parameters
  PARAMETER(r_cots);                  // COTS intrinsic growth rate (year^-1)
  PARAMETER(K_cots);                  // COTS carrying capacity (individuals/m2)
  PARAMETER(m_cots);                  // COTS natural mortality rate (year^-1)
  PARAMETER(alpha_cots);              // COTS density-dependent mortality coefficient (dimensionless)
  PARAMETER(beta_sst_cots);           // Effect of SST on COTS reproduction (°C^-1)
  PARAMETER(sst_opt_cots);            // Optimal SST for COTS reproduction (°C)
  PARAMETER(sst_tol_cots);            // SST tolerance range for COTS (°C)
  
  // Coral parameters
  PARAMETER(r_fast);                  // Fast-growing coral intrinsic growth rate (year^-1)
  PARAMETER(K_fast);                  // Fast-growing coral carrying capacity (%)
  PARAMETER(r_slow);                  // Slow-growing coral intrinsic growth rate (year^-1)
  PARAMETER(K_slow);                  // Slow-growing coral carrying capacity (%)
  PARAMETER(beta_sst_coral);          // Effect of SST on coral growth (°C^-1)
  PARAMETER(sst_opt_coral);           // Optimal SST for coral growth (°C)
  PARAMETER(sst_tol_coral);           // SST tolerance range for coral (°C)
  
  // Predation parameters
  PARAMETER(a_fast);                  // Attack rate on fast-growing coral (m2/individual/year)
  PARAMETER(a_slow);                  // Attack rate on slow-growing coral (m2/individual/year)
  PARAMETER(h_fast);                  // Handling time for fast-growing coral (year/%)
  PARAMETER(h_slow);                  // Handling time for slow-growing coral (year/%)
  PARAMETER(pref_fast);               // Preference for fast-growing coral (dimensionless)
  PARAMETER(min_coral_thresh);        // Minimum coral cover threshold for COTS survival (%)
  
  // Observation error parameters
  PARAMETER(log_sigma_cots);          // Log of observation error SD for COTS
  PARAMETER(log_sigma_fast);          // Log of observation error SD for fast-growing coral
  PARAMETER(log_sigma_slow);          // Log of observation error SD for slow-growing coral
  
  // Process error parameters - not used in this version for stability
  PARAMETER(log_sigma_proc_cots);     // Log of process error SD for COTS
  PARAMETER(log_sigma_proc_fast);     // Log of process error SD for fast-growing coral
  PARAMETER(log_sigma_proc_slow);     // Log of process error SD for slow-growing coral
  
  // Transform parameters
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_fast = exp(log_sigma_fast);
  Type sigma_slow = exp(log_sigma_slow);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize predicted values vectors
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Set initial values for first time step
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Process model: predict state variables through time
  for(int t = 1; t < n; t++) {
    // 1. Calculate temperature effects on growth rates using a simpler formulation
    Type sst_effect_cots = exp(-0.5 * pow((sst_dat(t-1) - sst_opt_cots) / (sst_tol_cots + 0.1), 2));
    Type sst_effect_coral = exp(-0.5 * pow((sst_dat(t-1) - sst_opt_coral) / (sst_tol_coral + 0.1), 2));
    
    // 2. Calculate total coral cover (used for density dependence)
    Type total_coral = fast_pred(t-1) + slow_pred(t-1) + 0.1;
    
    // 3. Calculate coral-dependent COTS survival - simplified
    Type coral_effect = total_coral / (total_coral + min_coral_thresh);
    
    // 4. Calculate predation rates using simplified functional response
    Type pred_rate_fast = a_fast * pref_fast / (1.0 + h_fast * fast_pred(t-1) + h_slow * slow_pred(t-1));
    Type pred_rate_slow = a_slow * (1.0 - pref_fast) / (1.0 + h_fast * fast_pred(t-1) + h_slow * slow_pred(t-1));
    
    // 5. COTS population dynamics with simplified terms
    Type reproduction = r_cots * cots_pred(t-1) * (1.0 - cots_pred(t-1) / (K_cots + 0.1)) * sst_effect_cots * coral_effect;
    if (reproduction < 0.0) reproduction = 0.0;
    
    Type mortality = m_cots * cots_pred(t-1) * (1.0 + alpha_cots * cots_pred(t-1));
    if (mortality < 0.0) mortality = 0.0;
    
    Type immigration = cotsimm_dat(t-1);
    
    // 6. Coral dynamics with simplified terms
    Type fast_growth = r_fast * fast_pred(t-1) * (1.0 - (fast_pred(t-1) + slow_pred(t-1)) / (K_fast + 0.1)) * sst_effect_coral;
    if (fast_growth < 0.0) fast_growth = 0.0;
    
    Type slow_growth = r_slow * slow_pred(t-1) * (1.0 - (fast_pred(t-1) + slow_pred(t-1)) / (K_slow + 0.1)) * sst_effect_coral;
    if (slow_growth < 0.0) slow_growth = 0.0;
    
    Type fast_predation = pred_rate_fast * cots_pred(t-1) * fast_pred(t-1);
    if (fast_predation < 0.0) fast_predation = 0.0;
    if (fast_predation > fast_pred(t-1)) fast_predation = 0.9 * fast_pred(t-1);
    
    Type slow_predation = pred_rate_slow * cots_pred(t-1) * slow_pred(t-1);
    if (slow_predation < 0.0) slow_predation = 0.0;
    if (slow_predation > slow_pred(t-1)) slow_predation = 0.9 * slow_pred(t-1);
    
    // 7. Update state variables with deterministic process
    cots_pred(t) = cots_pred(t-1) + reproduction - mortality + immigration;
    if (cots_pred(t) < 0.001) cots_pred(t) = 0.001;
    if (cots_pred(t) > K_cots) cots_pred(t) = 0.99 * K_cots;
    
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_predation;
    if (fast_pred(t) < 0.001) fast_pred(t) = 0.001;
    if (fast_pred(t) > K_fast) fast_pred(t) = 0.99 * K_fast;
    
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_predation;
    if (slow_pred(t) < 0.001) slow_pred(t) = 0.001;
    if (slow_pred(t) > K_slow) slow_pred(t) = 0.99 * K_slow;
  }
  
  // Observation model: calculate likelihood of observations given predictions
  for(int t = 0; t < n; t++) {
    // Use normal distribution on log-transformed data with fixed minimum SD
    Type log_cots_obs = log(cots_dat(t) + 0.001);
    Type log_cots_pred = log(cots_pred(t) + 0.001);
    nll -= dnorm(log_cots_obs, log_cots_pred, sigma_cots + 0.1, true);
    
    Type log_fast_obs = log(fast_dat(t) + 0.001);
    Type log_fast_pred = log(fast_pred(t) + 0.001);
    nll -= dnorm(log_fast_obs, log_fast_pred, sigma_fast + 0.1, true);
    
    Type log_slow_obs = log(slow_dat(t) + 0.001);
    Type log_slow_pred = log(slow_pred(t) + 0.001);
    nll -= dnorm(log_slow_obs, log_slow_pred, sigma_slow + 0.1, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
