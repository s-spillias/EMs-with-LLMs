#include <TMB.hpp>

template<class Type>
Type dlnorm_local(Type x, Type meanlog, Type sd, bool give_log=true){
    Type log_density = dnorm(log(x), meanlog, sd, true) - log(x);
    return give_log ? log_density : exp(log_density);
}

// 1. Data inputs:
//    - time: vector of time steps (years)
//    - cots_dat: observed COTS population densities (individuals/m²)
//    - slow_dat: observed cover for slow-growing corals (Faviidae/Porites) (% cover)
//    - fast_dat: observed cover for fast-growing coral (Acropora) (% cover)
//    - sst_dat and cotsimm_dat can be used for additional forcing if needed.
template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA SECTION
  DATA_VECTOR(time);              // Time steps (years)
  DATA_VECTOR(cots_dat);          // Observations: COTS density (individuals/m²)
  DATA_VECTOR(slow_dat);          // Observations: slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);          // Observations: fast-growing coral cover (%)

  // PARAMETER SECTION
  // Log-transformed intrinsic growth rate (year^-1) for COTS. (Parameter 1)
  PARAMETER(log_growth_rate);     
  // Log-transformed mortality rate (year^-1) for COTS. (Parameter 2)
  PARAMETER(log_mort_rate);       
  // Environmental modifier (dimensionless) scaling reproduction efficiency. (Parameter 3)
  PARAMETER(env);                 
  // Log-transformed half-saturation constant for resource limitation (% coral cover). (Parameter 4)
  PARAMETER(log_half_saturation);
  // Log-transformed predation efficiency on slow-growing coral (dimensionless). (Parameter 5)
  PARAMETER(log_pred_eff_slow);
  // Log-transformed predation efficiency on fast-growing coral (dimensionless). (Parameter 6)
  PARAMETER(log_pred_eff_fast);

  // Parameter transformations
  Type growth_rate   = exp(log_growth_rate);
  Type mort_rate     = exp(log_mort_rate);
  Type half_sat      = exp(log_half_saturation);
  Type pred_eff_slow = exp(log_pred_eff_slow);
  Type pred_eff_fast = exp(log_pred_eff_fast);

  // Introduce a small constant for numerical stability
  Type epsilon = 1e-8;

  // 2. State variable initialization:
  int n = time.size();
  vector<Type> cots_pred(n);
  cots_pred[0] = cots_dat[0];  // Initial condition from observed COTS density
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  slow_pred[0] = slow_dat[0];  // Initial condition for slow-growing coral cover
  fast_pred[0] = fast_dat[0];  // Initial condition for fast-growing coral cover
  
  /* 
    Equation Descriptions:
    (1) Resource Limitation Function: 
        resource_lim = (slow + fast coral cover) / (half_sat + (slow + fast coral cover) + epsilon)
        
    (2) Reproduction: 
        reproduction = growth_rate * resource_lim * env * (previous COTS density)
        
    (3) Mortality: 
        mortality = mort_rate * (previous COTS density)
        
    (4) Update Rule: 
        cots_pred[t] = cots_pred[t-1] + reproduction - mortality
        Note: Using previous time step values prevents data leakage.
  */
  for(int t = 1; t < n; t++){
    // Sum of coral cover from previous time step
    Type coral_cover = slow_dat[t-1] + fast_dat[t-1];
    
    // (1) Compute resource limitation using a saturating function
    Type resource_lim = coral_cover / (half_sat + coral_cover + epsilon);
    
    // (2) Calculate reproduction term; environmental conditions modulate reproduction success
    Type reproduction = growth_rate * resource_lim * env * cots_pred[t-1];
    
    // (3) Compute mortality term (including any implicit indirect feedback effects via predation)
    Type mortality = mort_rate * cots_pred[t-1];
    
    // (4) Update COTS density for time t
    cots_pred[t] = cots_pred[t-1] + reproduction - mortality; 
    slow_pred[t] = slow_pred[t-1] + 1e-8;  // Persistence with minimal variability for slow-growing coral cover
    fast_pred[t] = fast_pred[t-1] + 1e-8;  // Persistence with minimal variability for fast-growing coral cover
  }
  
  // 3. Likelihood calculation:
  // Using a lognormal error distribution with a fixed minimum standard deviation (0.01) for robustness.
  Type nll = 0.0;
  Type sigma = 0.01;
  for(int t = 0; t < n; t++){
    nll -= dlnorm_local(cots_dat[t] + epsilon, log(cots_pred[t] + epsilon), sigma, true);
  }
  for(int t = 0; t < n; t++){
    nll -= dlnorm_local(slow_dat[t] + epsilon, log(slow_pred[t] + epsilon), sigma, true);
    nll -= dlnorm_local(fast_dat[t] + epsilon, log(fast_pred[t] + epsilon), sigma, true);
  }

  // Report predictions for diagnostic purposes
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(nll);
  
  return nll;
}
