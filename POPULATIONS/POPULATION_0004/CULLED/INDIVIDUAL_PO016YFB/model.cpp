#include <TMB.hpp>

// 1. DATA INPUTS:
//    - cots_dat: observed COTS abundance (individuals/m^2)
//    - fast_dat: observed fast-growing coral cover (%)
//    - slow_dat: observed slow-growing coral cover (%)
//    - time: time steps (Year)
//
// 2. PARAMETERS (with units and sources provided in parameters.json):
//    - r0: Baseline intrinsic growth rate for COTS (year^-1)
//    - r1: Additional growth rate during outbreak events (year^-1)
//    - K: Carrying capacity for COTS (individuals/m^2)
//    - outbreak_threshold: COTS abundance threshold triggering outbreak (individuals/m^2)
//    - lambda: Steepness of the outbreak trigger logistic function (unitless)
//    - gamma: Factor for predation-driven COTS decline (unitless)
//    - growth_fast: Intrinsic growth rate for fast-growing coral (% per year)
//    - growth_slow: Intrinsic growth rate for slow-growing coral (% per year)
//    - delta_fast: Impact of COTS predation on fast-growing coral (per individual per year)
//    - delta_slow: Impact of COTS predation on slow-growing coral (per individual per year)
//    - cots0, fast0, slow0: Initial conditions for COTS and the two coral groups
//    - log_sigma_*: Log standard deviations for observation error
//
// 3. EQUATION DESCRIPTIONS:
//    (1) Outbreak trigger: outbreak = 1/(1+exp(-lambda*(cots_prev - outbreak_threshold)))
//    (2) COTS dynamics: 
//        cots[t] = cots[t-1] + (r0 + r1*outbreak) * cots[t-1] * (1 - cots[t-1]/(K + epsilon))
//                   - gamma * cots[t-1] * (fast[t-1] + slow[t-1])
//    (3) Fast coral dynamics:
//        fast[t] = fast[t-1] + growth_fast * (1 - fast[t-1]) * (1 - 1/(1+exp(-delta_fast*(cots[t-1]-epsilon))))
//    (4) Slow coral dynamics:
//        slow[t] = slow[t-1] + growth_slow * (1 - slow[t-1]) * (1 - 1/(1+exp(-delta_slow*(cots[t-1]-epsilon))))
//
//    Numerical stability is maintained using a small constant epsilon = 1e-8.
//    Observations are linked to predictions using lognormal likelihoods to ensure positivity.
 
template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA: observed time series
  DATA_VECTOR(cots_dat);   // COTS abundance observations (individuals/m^2)
  DATA_VECTOR(fast_dat);   // Fast-growing coral cover observations (%)
  DATA_VECTOR(slow_dat);   // Slow-growing coral cover observations (%)
  DATA_VECTOR(time);       // Time (Year)

  // PARAMETERS
  PARAMETER(r0);             // Baseline intrinsic growth rate (year^-1)
  PARAMETER(r1);             // Outbreak-induced growth rate increment (year^-1)
  PARAMETER(K);              // Carrying capacity for COTS (individuals/m^2)
  PARAMETER(outbreak_threshold); // Threshold to trigger outbreak (individuals/m^2)
  PARAMETER(lambda);         // Logistic slope for outbreak trigger (unitless)
  PARAMETER(gamma);          // Predation impact factor on COTS (unitless)
  PARAMETER(growth_fast);    // Growth rate for fast-growing coral (% per year)
  PARAMETER(growth_slow);    // Growth rate for slow-growing coral (% per year)
  PARAMETER(delta_fast);     // COTS predation effect on fast-growing coral (per individual per year)
  PARAMETER(delta_slow);     // COTS predation effect on slow-growing coral (per individual per year)
  PARAMETER(cots0);          // Initial COTS abundance (individuals/m^2)
  PARAMETER(fast0);          // Initial fast-growing coral cover (fraction)
  PARAMETER(slow0);          // Initial slow-growing coral cover (fraction)
  PARAMETER(log_sigma_cots); // Log observation error for COTS
  PARAMETER(log_sigma_fast); // Log observation error for fast coral
  PARAMETER(log_sigma_slow); // Log observation error for slow coral

  Type sigma_cots = exp(log_sigma_cots); // Observation error for COTS
  Type sigma_fast = exp(log_sigma_fast); // Observation error for fast coral
  Type sigma_slow = exp(log_sigma_slow); // Observation error for slow coral

  int n = cots_dat.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Set initial conditions
  cots_pred(0) = cots0;
  fast_pred(0) = fast0;
  slow_pred(0) = slow0;

  Type nll = 0;             // Negative log-likelihood accumulator
  Type epsilon = Type(1e-8);  // Small constant to prevent numerical issues

  // Loop over time steps (starting from t=1, using only lagged values)
  for(int t = 1; t < n; t++){
    // Outbreak trigger (smooth transition):
    Type outbreak = Type(1) / (Type(1) + exp(-lambda * (cots_pred(t-1) - outbreak_threshold)));
    
    // COTS dynamic equation:
    cots_pred(t) = cots_pred(t-1) 
                    + (r0 + r1 * outbreak) * cots_pred(t-1) * (Type(1) - cots_pred(t-1)/(K + epsilon))
                    - gamma * cots_pred(t-1) * (fast_pred(t-1) + slow_pred(t-1));
    
    // Fast-growing coral dynamic equation:
    fast_pred(t) = fast_pred(t-1)
                    + growth_fast * (Type(1) - fast_pred(t-1))
                      * (Type(1) - Type(1) / (Type(1) + exp(-delta_fast * (cots_pred(t-1) - epsilon))));
    
    // Slow-growing coral dynamic equation:
    slow_pred(t) = slow_pred(t-1)
                    + growth_slow * (Type(1) - slow_pred(t-1))
                      * (Type(1) - Type(1) / (Type(1) + exp(-delta_slow * (cots_pred(t-1) - epsilon))));
    
    // Likelihood: use lognormal to accommodate strictly positive predictions
    // dlnorm(x, meanlog, sdlog, true) = dnorm(log(x), meanlog, sdlog, true) - log(x)
    nll -= dnorm(log(cots_dat(t)), log(cots_pred(t) + epsilon), sigma_cots, true)
           - log(cots_dat(t) + epsilon);
    nll -= dnorm(log(fast_dat(t)), log(fast_pred(t) + epsilon), sigma_fast, true)
           - log(fast_dat(t) + epsilon);
    nll -= dnorm(log(slow_dat(t)), log(slow_pred(t) + epsilon), sigma_slow, true)
           - log(slow_dat(t) + epsilon);
  }

  // Report predicted trajectories for inspection
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}
