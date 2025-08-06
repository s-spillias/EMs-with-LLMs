#include <TMB.hpp>

// Simple TMB model for coral dynamics using logistic growth and constant predation
// Equations:
// 1) Slow coral dynamics: 
//      slow_pred[t] = slow_pred[t-1] + growth_slow * slow_pred[t-1]*(1 - slow_pred[t-1]/100) - predation_slow * slow_pred[t-1]
// 2) Fast coral dynamics: 
//      fast_pred[t] = fast_pred[t-1] + growth_fast * fast_pred[t-1]*(1 - fast_pred[t-1]/100) - predation_fast * fast_pred[t-1]
// 3) Likelihood: lognormal error distributions comparing predictions to observations

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA inputs:
  DATA_VECTOR(slow_dat);       // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);       // Observed fast-growing coral cover (%)

  // PARAMETERS:
  PARAMETER(growth_slow);      // Growth rate for slow-growing corals (per year)
  PARAMETER(predation_slow);   // Predation rate on slow-growing corals (per year)
  PARAMETER(growth_fast);      // Growth rate for fast-growing corals (per year)
  PARAMETER(predation_fast);   // Predation rate on fast-growing corals (per year)
  PARAMETER_VECTOR(sigma);     // Observation standard deviations: sigma[0] for slow coral, sigma[1] for fast coral

  Type nll = 0.0;              // Negative log-likelihood
  Type eps = Type(1e-8);       // Small constant for numerical stability
  int n = slow_dat.size();
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);

  // Initialize predictions using the first observations
  slow_pred[0] = slow_dat[0];
  fast_pred[0] = fast_dat[0];

  Type K = Type(100.0);        // Carrying capacity for coral cover (%)

  for (int t = 1; t < n; t++) {
    // Slow coral dynamics with logistic growth and constant predation
    slow_pred[t] = slow_pred[t-1] + growth_slow * slow_pred[t-1] * (Type(1.0) - slow_pred[t-1] / K) - predation_slow * slow_pred[t-1];
    // Fast coral dynamics with logistic growth and constant predation
    fast_pred[t] = fast_pred[t-1] + growth_fast * fast_pred[t-1] * (Type(1.0) - fast_pred[t-1] / K) - predation_fast * fast_pred[t-1];

    // Clamping predictions to remain strictly positive
    slow_pred[t] = slow_pred[t] > eps ? slow_pred[t] : eps;
    fast_pred[t] = fast_pred[t] > eps ? fast_pred[t] : eps;

    Type sigma_slow = sigma[0] < Type(1e-8) ? Type(1e-8) : sigma[0];
    Type sigma_fast = sigma[1] < Type(1e-8) ? Type(1e-8) : sigma[1];

    // Likelihood calculation using lognormal errors for coral cover observations
    nll -= dnorm(log(slow_dat[t] + eps), log(slow_pred[t] + eps), sigma_slow, true);
    nll -= dnorm(log(fast_dat[t] + eps), log(fast_pred[t] + eps), sigma_fast, true);
  }

  // Report the model predictions
  ADREPORT(slow_pred);
  ADREPORT(fast_pred);
  
  return nll;
}
