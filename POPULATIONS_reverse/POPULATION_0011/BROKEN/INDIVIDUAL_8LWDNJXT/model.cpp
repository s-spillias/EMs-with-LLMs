#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() (void) {
  using namespace density;
  
  // DATA SECTION
  DATA_VECTOR(Year);         // Time variable (years)
  DATA_VECTOR(cots_dat);       // Observed Crown-of-Thorns Starfish density (individuals/m2)
  DATA_VECTOR(slow_dat);       // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);       // Observed fast-growing coral cover (%)
  
  // PARAMETER SECTION
  PARAMETER(log_r);            // Log intrinsic growth rate of starfish (year^-1)
  PARAMETER(log_K);            // Log carrying capacity (log(individuals/m2))
  PARAMETER(log_delta);        // Log predation effect coefficient (unitless scaling)
  PARAMETER(log_gamma_s);      // Log predation rate on slow coral (year^-1)
  PARAMETER(log_gamma_f);      // Log predation rate on fast coral (year^-1)
  PARAMETER(sigma);            // Observation error standard deviation (log-scale)
  PARAMETER(log_cots0);        // Log initial COTS density (log(individuals/m2))
  PARAMETER(log_slow0);        // Log initial slow-growing coral cover (log(% cover))
  PARAMETER(log_fast0);        // Log initial fast-growing coral cover (log(% cover))
  
  // TRANSFORM PARAMETERS
  Type r = exp(log_r);         // Intrinsic growth rate (year^-1)
  Type K = exp(log_K);         // Carrying capacity (individuals/m2)
  Type delta = exp(log_delta); // Predation effect coefficient
  Type gamma_s = exp(log_gamma_s); // Predation rate on slow coral (year^-1)
  Type gamma_f = exp(log_gamma_f); // Predation rate on fast coral (year^-1)
  Type cots0 = exp(log_cots0);         // Initial COTS density (individuals/m2)
  Type slow0 = exp(log_slow0);         // Initial slow-growing coral cover (% cover)
  Type fast0 = exp(log_fast0);         // Initial fast-growing coral cover (% cover)
  
  // INITIALIZING PREDICTIONS
  int n = Year.size();
  vector<Type> cots_pred(n);   // Predicted COTS density
  vector<Type> slow_pred(n);   // Predicted slow-growing coral cover
  vector<Type> fast_pred(n);   // Predicted fast-growing coral cover
  
  // Initialization: set initial states from parameters (to avoid data leakage)
  cots_pred(0) = cots0;
  slow_pred(0) = slow0;
  fast_pred(0) = fast0;
  
  // MODEL EQUATIONS:
  // 1. Starfish dynamics: logistic growth with resource limitation modified by coral availability.
  // 2. Slow coral dynamics: decline driven by starfish predation using a saturating functional response.
  // 3. Fast coral dynamics: decline driven by starfish predation using a saturating functional response.
  // 4. Smooth transitions and small constant (1e-8) are used to ensure numerical stability.
  
  for(int t = 0; t < n-1; t++){
    // Saturating functions for coral predation (avoid division by zero)
    Type slow_effect = slow_pred(t) / (slow_pred(t) + Type(1e-8));
    Type fast_effect = fast_pred(t) / (fast_pred(t) + Type(1e-8));
    
    // (1) Update starfish population:
    // Equation: cots_pred[t+1] = cots_pred[t] + r * cots_pred[t] * (1 - cots_pred[t] / K) - delta * cots_pred[t]*(slow_effect+fast_effect)
    cots_pred(t+1) = cots_pred(t) + r * cots_pred(t) * (1 - cots_pred(t) / (K + Type(1e-8))) - delta * cots_pred(t) * (slow_effect + fast_effect);
    
    // (2) Update slow coral cover:
    // Equation: slow_pred[t+1] = slow_pred[t] - gamma_s * cots_pred[t] * slow_effect
    slow_pred(t+1) = slow_pred(t) - gamma_s * cots_pred(t) * slow_effect;
    
    // (3) Update fast coral cover:
    // Equation: fast_pred[t+1] = fast_pred[t] - gamma_f * cots_pred(t) * fast_effect
    fast_pred(t+1) = fast_pred(t) - gamma_f * cots_pred(t) * fast_effect;
  }
  
  // LIKELIHOOD CALCULATION:
  // Link initial observations to the estimated initial states and subsequent observations to model predictions.
  Type nll = 0.0;
  
  // Likelihood for the initial state (t = 0)
  nll -= dlnorm(cots_dat(0), log(cots0 + Type(1e-8)), sigma + Type(1e-8), true);
  nll -= dlnorm(slow_dat(0), log(slow0 + Type(1e-8)), sigma + Type(1e-8), true);
  nll -= dlnorm(fast_dat(0), log(fast0 + Type(1e-8)), sigma + Type(1e-8), true);
  
  // Likelihood for subsequent time points (t >= 1)
  for(int t = 0; t < n-1; t++){
    nll -= dlnorm(cots_dat(t+1), log(cots_pred(t+1) + Type(1e-8)), sigma + Type(1e-8), true);
    nll -= dlnorm(slow_dat(t+1), log(slow_pred(t+1) + Type(1e-8)), sigma + Type(1e-8), true);
    nll -= dlnorm(fast_dat(t+1), log(fast_pred(t+1) + Type(1e-8)), sigma + Type(1e-8), true);
  }
  
  // REPORT PREDICTED VALUES
  REPORT(cots_pred);   // Predicted COTS densities
  REPORT(slow_pred);   // Predicted slow-growing coral cover
  REPORT(fast_pred);   // Predicted fast-growing coral cover
  
  return nll;
}
