#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Small constant to ensure numerical stability
  Type eps = Type(1e-8);

  //---- DATA INPUTS ----
  // Time vector (years)
  DATA_VECTOR(Year); // Year: time (year)
  // Observed variables:
  DATA_VECTOR(cots_dat); // COTS abundance (individuals/m^2)
  DATA_VECTOR(slow_dat); // Slow-growing coral cover (%) for Faviidae/Porites
  DATA_VECTOR(fast_dat); // Fast-growing coral cover (%) for Acropora
  // Forcing data:
  DATA_VECTOR(sst_dat);  // Sea Surface Temperature (°C)
  DATA_VECTOR(cotsimm_dat); // COTS immigration rate (individuals/m^2/year)

  //---- MODEL PARAMETERS ----
  // 1. Starfish dynamics parameters
  PARAMETER(log_growth_rate);  // Log intrinsic growth rate (year^-1)
  PARAMETER(log_K);            // Log carrying capacity (individuals/m^2)
  PARAMETER(beta_sst);         // SST modulation coefficient (°C^-1)

  // 2. Coral dynamics parameters
  PARAMETER(recruitment_slow); // Recruitment rate for slow-growing corals (year^-1)
  PARAMETER(recruitment_fast); // Recruitment rate for fast-growing corals (year^-1)
  PARAMETER(log_predation_rate_slow); // Log predation rate on slow-growing corals
  PARAMETER(log_predation_rate_fast); // Log predation rate on fast-growing corals
  PARAMETER(threshold_slow);          // Saturation threshold for slow coral (percentage)
  PARAMETER(threshold_fast);          // Saturation threshold for fast coral (percentage)

  // 3. Observation error parameters (log-transformed to ensure positivity)
  PARAMETER(log_sigma_cots); // Log standard deviation for COTS observations
  PARAMETER(log_sigma_slow); // Log standard deviation for slow coral observations
  PARAMETER(log_sigma_fast); // Log standard deviation for fast coral observations
  PARAMETER(cots0);         // Initial COTS abundance (individuals/m^2)
  PARAMETER(slow0);         // Initial slow coral cover (%)
  PARAMETER(fast0);         // Initial fast coral cover (%)

  //---- PARAMETER TRANSFORMATIONS ----
  Type growth_rate = exp(log_growth_rate); // Intrinsic growth rate (year^-1)
  Type K = exp(log_K);                     // Carrying capacity (individuals/m^2)
  Type predation_rate_slow = std::min(exp(log_predation_rate_slow), Type(1000.0)); // Predation rate (slow coral) capped for numerical stability
  Type predation_rate_fast = std::min(exp(log_predation_rate_fast), Type(1000.0)); // Predation rate (fast coral) capped for numerical stability
  Type sigma_cots = exp(log_sigma_cots) + eps; // Observation error SD for COTS
  Type sigma_slow = exp(log_sigma_slow) + eps; // Observation error SD for slow coral
  Type sigma_fast = exp(log_sigma_fast) + eps; // Observation error SD for fast coral

  //---- MODEL SETUP ----
  int n = Year.size(); // Number of time steps

  // Vectors to store predicted values; initial values set from data to avoid leakage.
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  cots_pred(0) = cots0; // Initial COTS abundance from parameter
  slow_pred(0) = slow0; // Initial slow coral cover from parameter
  fast_pred(0) = fast0; // Initial fast coral cover from parameter

  // Negative log-likelihood initialization
  Type nll = 0.0;

  // Likelihood contributions for initial states using lognormal density transformation
  nll -= ( dnorm(log(cots_dat(0) + eps), log(cots0 + eps), sigma_cots, true) - log(cots_dat(0) + eps) );
  nll -= ( dnorm(log(slow_dat(0) + eps), log(slow0 + eps), sigma_slow, true) - log(slow_dat(0) + eps) );
  nll -= ( dnorm(log(fast_dat(0) + eps), log(fast0 + eps), sigma_fast, true) - log(fast_dat(0) + eps) );

  // Model Equations:
  // 1. COTS dynamics: logistic growth modulated by environmental SST effects and forced immigration.
  // 2. Slow coral dynamics: logistic growth minus predation loss captured by a saturating functional response.
  // 3. Fast coral dynamics: analogous to slow coral dynamics with distinct parameters.
  // (Predictions for time t>=1 are computed using the state at time t-1 to avoid data leakage.)

  for (int t = 1; t < n; t++) {
    // Effective growth rate modulated by deviation from a reference SST (27°C)
    Type effective_growth = growth_rate * (1 + beta_sst * (sst_dat(t-1) - Type(27.0)));

    // Equation 1: COTS population update using logistic growth and external immigration
    cots_pred(t) = cots_pred(t-1) + effective_growth * cots_pred(t-1) * (1 - cots_pred(t-1) / (K + eps))
                   + cotsimm_dat(t-1);

    // Equation 2: Slow coral cover update: logistic growth with interspecific competition minus predation
    slow_pred(t) = slow_pred(t-1)
                   + recruitment_slow * slow_pred(t-1) * (1 - (slow_pred(t-1) + fast_pred(t-1))/(Type(100.0) + eps))
                   - predation_rate_slow * cots_pred(t-1) * slow_pred(t-1)
                     / (slow_pred(t-1) + threshold_slow + eps);

    // Equation 3: Fast coral cover update: logistic growth with interspecific competition minus predation
    fast_pred(t) = fast_pred(t-1)
                   + recruitment_fast * fast_pred(t-1) * (1 - (slow_pred(t-1) + fast_pred(t-1))/(Type(100.0) + eps))
                   - predation_rate_fast * cots_pred(t-1) * fast_pred(t-1)
                     / (fast_pred(t-1) + threshold_fast + eps);
    // Ensure predictions remain positive to avoid log of non-positive numbers in likelihoods
    cots_pred(t) = CppAD::CondExpLt(cots_pred(t), eps, eps, cots_pred(t));
    slow_pred(t) = CppAD::CondExpLt(slow_pred(t), eps, eps, slow_pred(t));
    fast_pred(t) = CppAD::CondExpLt(fast_pred(t), eps, eps, fast_pred(t));

    // Likelihood Calculation:
    // Use lognormal error distributions for strictly positive observed data (for t>=1).
    nll -= ( dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots, true) - log(cots_dat(t) + eps) );
    nll -= ( dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow, true) - log(slow_dat(t) + eps) );
    nll -= ( dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast, true) - log(fast_dat(t) + eps) );
  }

  // Report model predictions (_pred variables)
  REPORT(cots_pred); // Predicted COTS abundance
  REPORT(slow_pred); // Predicted slow-growing coral cover
  REPORT(fast_pred); // Predicted fast-growing coral cover

  return nll;
}
