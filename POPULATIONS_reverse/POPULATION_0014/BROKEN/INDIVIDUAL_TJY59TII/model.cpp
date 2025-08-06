#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA SECTION
  DATA_VECTOR(Year);                  // Years of observation
  DATA_VECTOR(cots_dat);              // Adult COTS abundance (individuals/m2)
  DATA_VECTOR(slow_dat);              // Slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);              // Fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);               // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);           // COTS immigration rate (individuals/m2/year)
  
  // PARAMETER SECTION
  PARAMETER(r_cots);                  // COTS intrinsic growth rate (year^-1)
  PARAMETER(K_cots);                  // COTS carrying capacity (individuals/m2)
  PARAMETER(m_cots);                  // COTS natural mortality rate (year^-1)
  PARAMETER(a_slow);                  // COTS attack rate on slow-growing coral (m2/individual/year)
  PARAMETER(a_fast);                  // COTS attack rate on fast-growing coral (m2/individual/year)
  PARAMETER(pref_fast);               // COTS preference for fast-growing coral (dimensionless)
  PARAMETER(h_cots);                  // COTS half-saturation constant for feeding (% cover)
  PARAMETER(r_slow);                  // Slow-growing coral intrinsic growth rate (year^-1)
  PARAMETER(r_fast);                  // Fast-growing coral intrinsic growth rate (year^-1)
  PARAMETER(K_coral);                 // Total coral carrying capacity (% cover)
  PARAMETER(comp_slow);               // Competitive effect of slow-growing coral on fast-growing coral (dimensionless)
  PARAMETER(comp_fast);               // Competitive effect of fast-growing coral on slow-growing coral (dimensionless)
  PARAMETER(temp_opt);                // Optimal temperature for coral growth (°C)
  PARAMETER(temp_tol);                // Temperature tolerance range for coral (°C)
  PARAMETER(temp_mort);               // Temperature mortality coefficient (year^-1/°C)
  PARAMETER(log_sigma_cots);          // Log of observation error SD for COTS
  PARAMETER(log_sigma_slow);          // Log of observation error SD for slow-growing coral
  PARAMETER(log_sigma_fast);          // Log of observation error SD for fast-growing coral
  
  // Transform parameters
  Type sigma_cots = exp(log_sigma_cots);  // Observation error SD for COTS
  Type sigma_slow = exp(log_sigma_slow);  // Observation error SD for slow-growing coral
  Type sigma_fast = exp(log_sigma_fast);  // Observation error SD for fast-growing coral
  
  // Initialize negative log-likelihood
  Type nll = 0;
  
  // Initialize vectors for predictions
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  
  // Set initial values for first time step
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-6);
  
  // PROCESS MODEL: Simulate dynamics through time
  for(int t = 1; t < n; t++) {
    // 1. Simple linear models for all state variables
    
    // Temperature effect (0-1 scale)
    Type temp_diff = sst_dat(t-1) - temp_opt;
    Type temp_effect = exp(-0.5 * temp_diff * temp_diff / (temp_tol * temp_tol + eps));
    
    // COTS population dynamics (logistic growth + immigration - mortality)
    Type cots_growth = r_cots * cots_pred(t-1) * (1.0 - cots_pred(t-1) / (K_cots + eps));
    if (cots_growth < 0.0) cots_growth = 0.0;
    
    // Coral predation by COTS
    Type total_feeding = a_slow * slow_pred(t-1) + pref_fast * a_fast * fast_pred(t-1);
    Type feeding_slow = a_slow * cots_pred(t-1) * slow_pred(t-1) / (h_cots + total_feeding + eps);
    Type feeding_fast = pref_fast * a_fast * cots_pred(t-1) * fast_pred(t-1) / (h_cots + total_feeding + eps);
    
    // Coral growth
    Type slow_growth = r_slow * slow_pred(t-1) * temp_effect;
    Type fast_growth = r_fast * fast_pred(t-1) * temp_effect;
    
    // Update state variables
    cots_pred(t) = cots_pred(t-1) + cots_growth - m_cots * cots_pred(t-1) + cotsimm_dat(t-1);
    slow_pred(t) = slow_pred(t-1) + slow_growth - feeding_slow;
    fast_pred(t) = fast_pred(t-1) + fast_growth - feeding_fast;
    
    // Apply bounds to predictions
    if (cots_pred(t) < 0.0) cots_pred(t) = 0.0;
    if (slow_pred(t) < 0.0) slow_pred(t) = 0.0;
    if (fast_pred(t) < 0.0) fast_pred(t) = 0.0;
    
    if (slow_pred(t) > K_coral) slow_pred(t) = K_coral;
    if (fast_pred(t) > K_coral) fast_pred(t) = K_coral;
  }
  
  // OBSERVATION MODEL: Calculate negative log-likelihood
  Type min_sd = 0.1;  // Minimum standard deviation
  
  for(int t = 0; t < n; t++) {
    // Normal observation model for all variables
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots + min_sd, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_slow + min_sd, true);
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_fast + min_sd, true);
  }
  
  // REPORTING SECTION
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(sigma_cots);
  REPORT(sigma_slow);
  REPORT(sigma_fast);
  
  return nll;
}
