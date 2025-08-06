#include <TMB.hpp>  // TMB header
using namespace density;

// 1. Data inputs
// [1] cots_dat: Adult Crown-of-Thorns starfish abundance (individuals/m2)
// [2] slow_dat: Observed slow-growing coral cover (%) 
// [3] fast_dat: Observed fast-growing coral cover (%)
template<class Type>
Type objective_function<Type>::operator() ()
{
  DATA_VECTOR(cots_dat);    // Starfish density data, individuals/m2
  DATA_VECTOR(slow_dat);    // Observed slow-growing coral cover data (%)
  DATA_VECTOR(fast_dat);    // Observed fast-growing coral cover data (%)
  DATA_VECTOR(sst_dat);     // Sea Surface Temperature data (°C)
  DATA_VECTOR(cotsimm_dat); // Crown-of-thorns immigration rate (individuals/m2/year)

  // 2. Model parameters with comments:
  PARAMETER(slow_intercept);  // (slow_intercept): Baseline slow-growing coral cover (%) from expert opinion
  PARAMETER(slow_slope);      // (slow_slope): Effect of COTS on slow-growing coral cover (% per individuals/m2) from literature
  PARAMETER(fast_intercept);  // (fast_intercept): Baseline fast-growing coral cover (%) from expert opinion
  PARAMETER(fast_slope);      // (fast_slope): Effect of COTS on fast-growing coral cover (% per individuals/m2) from literature
  PARAMETER(log_sigma);       // (log_sigma): Log-standard deviation for observation errors; ensures sigma>0
  PARAMETER(density_dependence); // (density_dependence): Density-dependence factor limiting coral cover growth.
  PARAMETER(sst_effect); // New parameter: effect of Sea Surface Temperature on coral cover predictions.

  // 3. Numerical stability constant to prevent division by zero, etc.
  Type eps = Type(1e-8);  // Small constant for numerical stability

  int n = cots_dat.size();   // Number of observations
  Type nll = 0.0;            // Initialize negative log-likelihood accumulator

  // 4. Compute predictions using external forcing and lagged predictions:
  vector<Type> cots_pred(n); // Predicted Crown-of-Thorns starfish abundance (individuals/m2)
  vector<Type> slow_pred(n); // Predicted slow-growing coral cover (%)
  vector<Type> fast_pred(n); // Predicted fast-growing coral cover (%)

  // Initialize predictions with first observation/forcing values
  cots_pred(0) = cotsimm_dat(0);  // Set initial COTS prediction using immigration data
  slow_pred(0) = slow_dat(0);      // Set initial slow coral cover prediction from observation
  fast_pred(0) = fast_dat(0);      // Set initial fast coral cover prediction from observation

  // Recursive prediction equations for t>=1:
  for (int i = 1; i < n; i++){
    // Equation 1: COTS prediction: update previous COTS level towards current immigration forcing
    // cots_pred(i) = cots_pred(i-1) + 0.5 * (cotsimm_dat(i) - cots_pred(i-1))
    cots_pred(i) = cots_pred(i-1) + Type(0.5) * (cotsimm_dat(i) - cots_pred(i-1));

    // Equation 2: Slow-growing coral cover prediction with density-dependent regulation and SST effect
    slow_pred(i) = slow_pred(i-1) * slow_intercept * (Type(1.0) - slow_slope * cots_pred(i-1)) * exp(sst_effect * sst_dat(i)) / (Type(1.0) + density_dependence * slow_pred(i-1));

    // Equation 3: Fast-growing coral cover prediction with density-dependent regulation and SST effect
    fast_pred(i) = fast_pred(i-1) * fast_intercept * (Type(1.0) - fast_slope * cots_pred(i-1)) * exp(sst_effect * sst_dat(i)) / (Type(1.0) + density_dependence * fast_pred(i-1));
  }

  // 5. Observation error processing with minimal sigma
  Type sigma = exp(log_sigma) + eps;  // Observation standard deviation with enforced lower bound

  // 6. Likelihood calculation using lognormal likelihoods:
  // For each observation, the log-transformed observed data is modeled as normally distributed 
  // around the log of the predicted value.
  for (int i = 0; i < n; i++){
    nll -= dnorm(log(cots_dat(i) + eps), log(cots_pred(i) + eps), sigma, true);   // Eq.3 for COTS
    nll -= dnorm(log(slow_dat(i) + eps), log(slow_pred(i) + eps), sigma, true);       // Eq.3 for slow coral
    nll -= dnorm(log(fast_dat(i) + eps), log(fast_pred(i) + eps), sigma, true);         // Eq.3 for fast coral
  }
  vector<Type> sst_pred(n);
  for (int i = 0; i < n; i++){
    sst_pred(i) = sst_dat(i);
  }
  vector<Type> cotsimm_pred(n);
  for (int i = 0; i < n; i++){
    cotsimm_pred(i) = cotsimm_dat(i);
  }

  // 7. Reporting section: Report predictions and sigma for further analysis and diagnostics.
  REPORT(cots_pred);    // cots_pred: predicted Crown-of-Thorns starfish abundance
  REPORT(cotsimm_pred); // cotsimm_pred: predicted Crown-of-Thorns starfish immigration rate
  REPORT(sst_pred);     // sst_pred: predicted Sea Surface Temperature (°C)
  REPORT(slow_pred);    // slow_pred: predicted slow-growing coral cover (%)
  REPORT(fast_pred);    // fast_pred: predicted fast-growing coral cover (%)
  REPORT(sigma);        // sigma: effective observation error standard deviation

  return nll;
}
