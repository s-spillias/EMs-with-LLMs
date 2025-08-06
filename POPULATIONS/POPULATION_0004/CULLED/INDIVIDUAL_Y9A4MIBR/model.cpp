/*
  TMB Model for simulating COTS outbreaks on the Great Barrier Reef

  Model Equations:
  1. COTS Dynamics:  
     - If previous COTS density is below outbreak_threshold, logistic growth is used.
     - If above the threshold, a boom effect (scaled by boom_efficiency) is activated, followed by a bust (scaled by bust_rate).
  2. Fast Coral Dynamics:
     - Fast-growing coral cover declines due to predation (using a saturating functional response) and recovers at a rate coral_recovery_fast.
  3. Slow Coral Dynamics:
     - Slow-growing coral cover declines due to predation and recovers at a rate coral_recovery_slow.
     
  Reporting:
  - All predicted state variables (cots_pred, fast_pred, slow_pred) are reported via REPORT().
  
  Numerical Stability:
  - A small constant (small = 1e-8) is added in denominators and log-transformations to avoid division by zero or log(0).
  
  Note:
  - Only previous time step values are used for predictions to preclude data leakage.
*/

#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data input:
  DATA_VECTOR(time);          // Time steps (years)
  DATA_VECTOR(cots_dat);      // Observed COTS density (individuals/m2)
  DATA_VECTOR(fast_dat);      // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);      // Observed slow-growing coral cover (%)

  // Parameters for ecological processes:
  // 1. COTS outbreak dynamics parameters:
  PARAMETER(growth_rate);           // Intrinsic outbreak growth rate (year^-1)
  PARAMETER(carrying_capacity);     // Maximum sustainable COTS density (individuals/m2)
  PARAMETER(outbreak_threshold);    // Threshold COTS density that triggers outbreak dynamics (individuals/m2)
  PARAMETER(boom_efficiency);       // Efficiency multiplier for explosive outbreak growth (unitless)
  PARAMETER(bust_rate);             // Decline rate during outbreak bust phase (year^-1)
  
  // 2. Predation on coral parameters:
  PARAMETER(predation_eff_rate_fast);  // Predation efficiency on fast-growing coral (per unit COTS density)
  PARAMETER(predation_eff_rate_slow);  // Predation efficiency on slow-growing coral (per unit COTS density)
  
  // 3. Coral recovery dynamics:
  PARAMETER(coral_recovery_fast);   // Recovery rate for fast-growing coral (% per year)
  PARAMETER(coral_recovery_slow);   // Recovery rate for slow-growing coral (% per year)
  
  // 4. Initial conditions:
  PARAMETER(cots_0);                // Initial COTS density (individuals/m2)
  PARAMETER(fast_0);                // Initial fast-growing coral cover (%)
  PARAMETER(slow_0);                // Initial slow-growing coral cover (%)
  PARAMETER(env_modifier);          // Environmental modifier scaling outbreak dynamics
  
  int n = time.size();            // Number of time steps
  Type nll = 0.0;                 // Negative log likelihood
  Type small = Type(1e-8);        // Small constant for numerical stability
  
  // Vectors for predictions:
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Initial condition assignment (Equation 0):
  cots_pred(0) = cots_0;
  fast_pred(0) = fast_0;
  slow_pred(0) = slow_0;
  
  // Loop over time (starting at t=1 to avoid data leakage)
  for(int t = 1; t < n; t++){
    // Retrieve the previous state
    Type cots_prev = cots_pred(t-1);
    Type fast_prev = fast_pred(t-1);
    Type slow_prev = slow_pred(t-1);
    
    // Equation 1: COTS dynamics with smooth transition for outbreak (boom-bust behavior)
    // A logistic function is used to create a smooth outbreak indicator.
    Type outbreak_indicator = 1 / ( 1 + exp( - (cots_prev - outbreak_threshold) ) );
    Type logistic_growth = env_modifier * growth_rate * cots_prev * (1 - cots_prev/(carrying_capacity + small));
    Type boom = boom_efficiency * outbreak_indicator * cots_prev;
    Type bust = bust_rate * outbreak_indicator * cots_prev;
    cots_pred(t) = cots_prev + logistic_growth + boom - bust;
    cots_pred(t) = cots_pred(t) > small ? cots_pred(t) : small;  // Ensure non-negative
    
    // Equation 2: Fast coral dynamics with predation and recovery
    Type predation_fast = predation_eff_rate_fast * cots_prev * fast_prev/(fast_prev + small);
    fast_pred(t) = fast_prev - predation_fast + coral_recovery_fast * (100 - fast_prev);
    fast_pred(t) = fast_pred(t) > small ? fast_pred(t) : small;
    
    // Equation 3: Slow coral dynamics with predation and recovery
    Type predation_slow = predation_eff_rate_slow * cots_prev * slow_prev/(slow_prev + small);
    slow_pred(t) = slow_prev - predation_slow + coral_recovery_slow * (100 - slow_prev);
    slow_pred(t) = slow_pred(t) > small ? slow_pred(t) : small;
  }
  
  // Likelihood calculation: Lognormal likelihood with fixed measurement error sd = 0.1.
  // Log-transform used due to data spanning multiple orders of magnitude.
  for(int t = 0; t < n; t++){
    nll -= dnorm(log(cots_dat(t) + small), log(cots_pred(t) + small), Type(0.1), 1);
    nll -= dnorm(log(fast_dat(t) + small), log(fast_pred(t) + small), Type(0.1), 1);
    nll -= dnorm(log(slow_dat(t) + small), log(slow_pred(t) + small), Type(0.1), 1);
  }
  
  // Reporting prediction variables
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
