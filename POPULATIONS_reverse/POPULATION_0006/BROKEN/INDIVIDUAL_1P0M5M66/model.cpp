#include <TMB.hpp>

// TMB model for COTS feeding on slow and fast-growing corals.
// Model Equations:
// 1. COTS dynamics: cots_pred = cots_dat + growth_cots * cots_dat * (small increment), where growth_cots is the intrinsic growth rate (year^-1).
// 2. Slow coral dynamics: slow_pred = slow_dat - predation_slow * cots_dat, where predation_slow (m2 per individual) reduces coral cover.
// 3. Fast coral dynamics: fast_pred = fast_dat - predation_fast * cots_dat, where predation_fast (m2 per individual) reduces coral cover.
// Each equation includes a small constant (Type(1e-8)) for numerical stability.
// Observations are modeled using lognormal errors for corals (ensuring strictly positive predictions) and normal errors for COTS.

// Template and objective function definition
template<class Type>
Type objective_function<Type>::operator() (void) {
  // DATA: Vectors are populated from external inputs
  DATA_VECTOR(cots_dat);   // Adult Crown-of-Thorns starfish abundance (individuals/m2)
  DATA_VECTOR(slow_dat);   // Slow-growing coral cover (%) for Faviidae and Porites
  DATA_VECTOR(fast_dat);   // Fast-growing coral cover (%) for Acropora
  DATA_VECTOR(sst_dat);    // Sea Surface Temperature forcing (°C)
  DATA_VECTOR(cotsimm_dat);  // COTS immigration rate forcing (individuals/m2/year)

  // PARAMETERS
  // growth_cots: intrinsic growth rate (year^-1)
  PARAMETER(growth_cots);
  // predation_slow: predation rate on slow-growing corals (m2 per individual)
  PARAMETER(predation_slow);
  // predation_fast: predation rate on fast-growing corals (m2 per individual)
  PARAMETER(predation_fast);
  // log_sd: log-scale of the observational standard deviation to ensure positivity
  PARAMETER(log_sd);

  // Transform parameter for observational standard deviation and add a small constant for stability.
  Type sd = exp(log_sd) + Type(1e-8);

  // Negative log-likelihood
  Type nll = 0.0;

  // Number of observations:
  int n = cots_dat.size();

  // Vectors to store predicted values for reporting (with '_pred' suffix)
  vector<Type> cots_pred(n);   // (1) Predicted COTS abundance (individuals/m2)
  vector<Type> slow_pred(n);   // (2) Predicted slow coral cover (%) adjusted for predation
  vector<Type> fast_pred(n);   // (3) Predicted fast coral cover (%) adjusted for predation

  

  // Even simpler model predictions:
  // (1) COTS prediction: linear scaling of immigration forcing.
  for (int i = 0; i < n; i++) {
    cots_pred(i) = cotsimm_dat(i) * (1 + growth_cots);
  }
  // (2) Slow coral prediction: constant prediction using exponential decay from 100%.
  Type slow_const = Type(100) * exp(-predation_slow);
  // (3) Fast coral prediction: constant prediction using exponential decay from 100%.
  Type fast_const = Type(100) * exp(-predation_fast);
  for (int i = 0; i < n; i++) {
    slow_pred(i) = slow_const;
    fast_pred(i) = fast_const;
  }

  // Likelihood Calculation using the predicted values:
  for (int i = 0; i < n; i++) {
      // a. Lognormal likelihood for slow-growing corals.
      nll -= dnorm(log(slow_dat(i) + Type(1e-8)), log(slow_pred(i) + Type(1e-8)), sd, true);
      
      // b. Lognormal likelihood for fast-growing corals.
      nll -= dnorm(log(fast_dat(i) + Type(1e-8)), log(fast_pred(i) + Type(1e-8)), sd, true);
      
      // c. Normal likelihood for COTS abundance.
      nll -= dnorm(cots_dat(i), cots_pred(i), sd, true);
  }

  // Reporting predictions for inspection and further analysis
  ADREPORT(cots_pred);   // Report predicted COTS abundances
  ADREPORT(slow_pred);   // Report predicted slow coral values
  ADREPORT(fast_pred);   // Report predicted fast coral values

  return nll;
}
