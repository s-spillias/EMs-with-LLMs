/*
Template Model Builder (TMB) model for simulating episodic outbreaks of 
Crown-of-Thorns Starfish (COTS) on the Great Barrier Reef.

Equations (numbered):
1. COTS dynamics:
    cots_pred[t] = cots_pred[t-1] + growth_rate * cots_pred[t-1] * (1 - cots_pred[t-1]/carrying_capacity) * outbreak_effect
                   - predation_eff * cots_pred[t-1] * (fast_pred[t-1] / (fast_pred[t-1] + Type(1e-8)))
   - Uses a logistic growth component modulated by an outbreak trigger.
   - The outbreak_effect is a smooth transition function based on past COTS density.
2. Fast-growing coral dynamics:
    fast_pred[t] = fast_pred[t-1] + coral_regen_fast * (1 - fast_pred[t-1]) - 0.05 * cots_pred[t-1] * (fast_pred[t-1] / (fast_pred[t-1] + Type(1e-8)))
   - Coral regeneration minus loss due to COTS predation.
3. Slow-growing coral dynamics:
    slow_pred[t] = slow_pred[t-1] + coral_regen_slow * (1 - slow_pred[t-1]) - 0.03 * cots_pred[t-1] * (slow_pred[t-1] / (slow_pred[t-1] + Type(1e-8)))
   - Lower predation impact is assumed compared to fast-growing coral.

Notes on Numerical Stability:
- A small constant (1e-8) prevents division by zero.
- Smooth transitions (via a logistic outbreak effect) are used instead of hard thresholds.
- All likelihood calculations include every observation using dnorm with a fixed minimum SD.

*/

#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA VARIABLES (observations)
  DATA_VECTOR(time);             // Time variable (from data file, e.g., Year)
  DATA_VECTOR(cots_dat);         // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);         // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);         // Observed slow-growing coral cover (%)
  
  // PARAMETERS (in log-scale where appropriate for positivity)
  PARAMETER(log_growth_rate);        // log(Intrinsic growth rate of COTS, year^-1)
  PARAMETER(log_carrying_capacity);  // log(Carrying capacity for COTS, individuals/m2)
  PARAMETER(log_predation_eff);      // log(Predation efficiency on corals, unitless)
  PARAMETER(log_outbreak_threshold); // log(Threshold density for triggering outbreak, individuals/m2)
  
  // These parameters are taken directly (assumed positive already)
  PARAMETER(trigger_steepness);      // Steepness of outbreak trigger (unitless)
  PARAMETER(outbreak_curvature);     // Modifier of outbreak trigger curvature
  PARAMETER(log_coral_regen_fast);   // log(Regeneration rate for fast-growing coral, year^-1)
  PARAMETER(log_coral_regen_slow);   // log(Regeneration rate for slow-growing coral, year^-1)
  
  // Transform parameters back from log-scale
  Type growth_rate       = exp(log_growth_rate);         // (year^-1)
  Type carrying_capacity = exp(log_carrying_capacity);     // (individuals/m2)
  Type predation_eff     = exp(log_predation_eff);         // (unitless fraction)
  Type outbreak_threshold= exp(log_outbreak_threshold);    // (individuals/m2)
  Type coral_regen_fast  = exp(log_coral_regen_fast);      // (year^-1)
  Type coral_regen_slow  = exp(log_coral_regen_slow);      // (year^-1)
  
  // Number of time steps (assumed to correspond to years)
  int n = time.size();
  
  // Initialize prediction vectors: using 'n' rows
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Initial conditions: use the first observation as starting point
  cots_pred[0] = cots_dat[0];
  fast_pred[0] = fast_dat[0];
  slow_pred[0] = slow_dat[0];
  
  // Small constant for numerical stability
  Type eps = Type(1e-8);
  
  // Negative log-likelihood accumulator
  Type nll = 0;
  
  // Loop over time (starting from 1 to avoid using current observation in prediction)
  for(int t = 1; t < n; t++){
    // 1. Calculate a smooth outbreak trigger effect (logistic function)
    // outbreak_effect ranges from near 0 (no outbreak) to 1 (full outbreak)
    Type outbreak_effect = 1.0 / (1.0 + exp(-trigger_steepness * outbreak_curvature * (cots_pred[t-1] - outbreak_threshold)));
    
    // 2. Update COTS predictions using logistic growth modulated by outbreak trigger and predation on coral:
    // Equation 1: COTS dynamic update
    cots_pred[t] = cots_pred[t-1] 
                   + growth_rate * cots_pred[t-1] * (1.0 - cots_pred[t-1] / (carrying_capacity + eps)) * outbreak_effect 
                   - predation_eff * cots_pred[t-1] * (fast_pred[t-1] / (fast_pred[t-1] + eps));
    
    // 3. Update fast-growing coral predictions with regeneration and loss due to COTS predation:
    // Equation 2: Fast coral dynamic update
    fast_pred[t] = fast_pred[t-1] 
                   + coral_regen_fast * (1.0 - fast_pred[t-1]) 
                   - Type(0.05) * cots_pred[t-1] * (fast_pred[t-1] / (fast_pred[t-1] + eps));
    
    // 4. Update slow-growing coral predictions with slower regeneration and lower predation impact:
    // Equation 3: Slow coral dynamic update
    slow_pred[t] = slow_pred[t-1] 
                   + coral_regen_slow * (1.0 - slow_pred[t-1]) 
                   - Type(0.03) * cots_pred[t-1] * (slow_pred[t-1] / (slow_pred[t-1] + eps));
    
    // (Optional) Add smooth penalties if parameters stray out of biologically meaningful ranges.
    // For example, one might add: nll += square(max(Type(0), some_parameter - upper_bound)) * penalty_coeff;
  }
  
  // Likelihood calculations using fixed minimum standard deviations (0.1) to stabilize log-likelihood
  for (int t = 0; t < n; t++){
    // Using dnorm with log flag true; these error distributions could alternatively be lognormal.
    nll -= dnorm(cots_dat[t], cots_pred[t], Type(0.1), true);
    nll -= dnorm(fast_dat[t], fast_pred[t], Type(0.1), true);
    nll -= dnorm(slow_dat[t], slow_pred[t], Type(0.1), true);
    
    // REPORT predicted values for diagnostics
    REPORT(cots_pred[t]);
    REPORT(fast_pred[t]);
    REPORT(slow_pred[t]);
  }
  
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  return nll;
}
