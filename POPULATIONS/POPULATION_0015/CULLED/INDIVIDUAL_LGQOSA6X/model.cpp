#include <TMB.hpp>
/*
Model Description:
1. COTS Dynamics: cots_pred[t] = cots[t-1] + growth_rate * f(resource limitation) - decline_rate * cots[t-1]
2. Coral Dynamics: coral cover declines due to COTS predation with process-specific efficiencies.
3. Functional responses: A logistic function is used to smoothly activate outbreak dynamics when COTS exceed a threshold.
4. Likelihood: Lognormal observation error is assumed on the log-transformed predictions.
5. Numerical Stability: Small constants (1e-8) are added to denominators to avoid division by zero and achieve smoother transitions.
*/

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs: time vector and observed data from the data file (time as first column)
  DATA_VECTOR(time);                   // time (years)
  DATA_VECTOR(cots_dat);               // Adult Crown-of-Thorns starfish abundance (individuals/m2)
  DATA_VECTOR(fast_dat);               // Fast-growing coral cover (%) (Acropora spp.)
  DATA_VECTOR(slow_dat);               // Slow-growing coral cover (%) (Faviidae spp. and Poritidae spp.)
  
  // Parameters:
  PARAMETER(log_growth_rate);          // Log intrinsic growth rate of COTS (year^-1)
  PARAMETER(log_decline_rate);         // Log decline rate of COTS (year^-1)
  PARAMETER(log_threshold);            // Log threshold triggering outbreak for COTS (individuals/m2)
  PARAMETER(eff_predation_fast);       // Efficiency of predation on fast-growing corals (unitless)
  PARAMETER(eff_predation_slow);       // Efficiency of predation on slow-growing corals (unitless)
  PARAMETER(lnoise);                   // Log of observation noise standard deviation
  PARAMETER(log_hfast);                // Log half-saturation constant for fast-growing corals
  PARAMETER(log_hslow);                // Log half-saturation constant for slow-growing corals
  
  // Transform parameters for model use
  Type growth_rate = exp(log_growth_rate);     // Convert intrinsic growth rate from log-scale
  Type decline_rate = exp(log_decline_rate);     // Convert decline rate from log-scale
  Type threshold = exp(log_threshold);           // Outbreak threshold on natural scale
  Type sigma = exp(lnoise) + Type(1e-8);           // Observation noise standard deviation with numerical stability
  Type hfast = exp(log_hfast);                    // Half-saturation constant for fast-growing corals predation
  Type hslow = exp(log_hslow);                    // Half-saturation constant for slow-growing corals predation
  int n = time.size();  // Number of time points
  
  // Initialize predicted vectors for COTS and coral covers
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Set initial state using the first observed data point (avoid data leakage)
  cots_pred[0] = cots_dat[0];
  fast_pred[0] = fast_dat[0];
  slow_pred[0] = slow_dat[0];
  
  // Initialize negative log likelihood
  Type nll = 0.0;
  
  // Loop over time steps starting at t=1 (using previous time step values only)
  for (int t = 1; t < n; t++){
      // 1. Determine the outbreak effect with a smooth logistic (saturating) function:
      //    outbreak_indicator approximates a threshold effect near the value of 'threshold'.
      Type outbreak_indicator = 1.0 / (1.0 + exp(-10 * (cots_pred[t-1] - threshold)));
      
      // 2. COTS Dynamics:
      //    Equation: cots_pred = previous value + growth - decline
      cots_pred[t] = cots_pred[t-1] + growth_rate * outbreak_indicator * cots_pred[t-1] - decline_rate * cots_pred[t-1];
      
      // 3. Coral Dynamics:
      //    Fast and slow corals decline due to predation by increasing COTS numbers.
      fast_pred[t] = fast_pred[t-1] - eff_predation_fast * outbreak_indicator * cots_pred[t-1] * fast_pred[t-1] / (hfast + fast_pred[t-1]);
      slow_pred[t] = slow_pred[t-1] - eff_predation_slow * outbreak_indicator * cots_pred[t-1] * slow_pred[t-1] / (hslow + slow_pred[t-1]);
      
      // 4. Likelihood calculation: Lognormal errors on the log-transformed predictions to account for strictly positive values.
      nll -= dnorm(log(cots_dat[t]), log(cots_pred[t]), sigma, true);
      nll -= dnorm(log(fast_dat[t]), log(fast_pred[t]), sigma, true);
      nll -= dnorm(log(slow_dat[t]), log(slow_pred[t]), sigma, true);
  }
  
  // Report predicted values for post-processing and diagnostics
  REPORT(cots_pred);    // Predicted COTS abundances
  REPORT(fast_pred);    // Predicted fast-growing coral covers
  REPORT(slow_pred);    // Predicted slow-growing coral covers
  
  return nll;  // Return the negative log-likelihood
}
