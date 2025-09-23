#include <TMB.hpp>

// TMB Model for simulating episodic COTS outbreaks and coral dynamics
// Equations description:
// 1. COTS Population Dynamics: Logistic-like growth modulated by both a carrying capacity influenced by coral regeneration and an environmental effect term.
// 2. Fast-growing Coral Dynamics: Growth based on regeneration minus loss due to selective predation by COTS.
// 3. Slow-growing Coral Dynamics: Similar to fast-growing coral, but with a different predation rate.
// 4. Likelihood: For each time step (t>=1), the lognormal likelihood is computed for observed COTS, fast-growing coral, and slow-growing coral, using predictions from the previous time step.
// Note: All predictions use previous time step variables to avoid data leakage.
// Parameters are bounded with smooth penalties via TMB, and small constants (1e-8) ensure numerical stability.

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA: Observations (all strictly positive; expected to be log-transformed by likelihood function)
  DATA_VECTOR(cots_dat);      // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);      // Observed fast-growing coral cover (%) (Acropora spp.)
  DATA_VECTOR(slow_dat);      // Observed slow-growing coral cover (%) (Faviidae/Porites spp.)
  // DATA: Assume time is provided implicitly as the order in the data vectors

  // PARAMETERS:
  PARAMETER(log_std);   // Log-scale standard deviation for observation error (from parameters.json)
  PARAMETER(cots_r);    // [1] Intrinsic growth rate of COTS (year^-1)
  PARAMETER(pred_fast); // [2] Predation rate on fast-growing coral by COTS (year^-1 per individual)
  PARAMETER(pred_slow); // [3] Predation rate on slow-growing coral by COTS (year^-1 per individual)
  PARAMETER(coral_regen); // [4] Coral regeneration rate (year^-1)
  PARAMETER(env_effect);  // [5] Environmental modifier for COTS growth (unitless)

  // Initialize prediction vectors for model outputs
  int n = cots_dat.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Use initial observations as starting conditions (avoid data leakage, predictions start from t0)
  cots_pred[0] = cots_dat[0];
  fast_pred[0] = fast_dat[0];
  slow_pred[0] = slow_dat[0];

  // Small constant for numerical stability
  Type eps = Type(1e-8);

  // Initialize negative log likelihood
  Type nll = 0.0;

  // Main loop: iterate over time steps (starting at t = 1)
  for(int t = 1; t < n; t++){
    // Equation 1: COTS population dynamics with logistic-like growth
    // Formula: cots_pred[t] = previous value + growth * previous value * (1 - previous value/(previous value+coral_regen)) * (1 + env_effect)
    cots_pred[t] = cots_pred[t-1] 
                   + cots_r * cots_pred[t-1] * (Type(1.0) - cots_pred[t-1]/(cots_pred[t-1] + coral_regen + eps)) * (Type(1.0) + env_effect);

    // Equation 2: Fast-growing coral dynamics with regeneration and predation by COTS
    fast_pred[t] = fast_pred[t-1] 
                   + coral_regen * fast_pred[t-1] 
                   - pred_fast * cots_pred[t-1] * fast_pred[t-1] / (fast_pred[t-1] + eps);

    // Equation 3: Slow-growing coral dynamics with regeneration and predation by COTS
    slow_pred[t] = slow_pred[t-1] 
                   + coral_regen * slow_pred[t-1] 
                   - pred_slow * cots_pred[t-1] * slow_pred[t-1] / (slow_pred[t-1] + eps);

    // Ensure predictions remain non-negative using smooth transitions
    cots_pred[t] = CppAD::CondExpGt(cots_pred[t], Type(0), cots_pred[t], eps);
    fast_pred[t] = CppAD::CondExpGt(fast_pred[t], Type(0), fast_pred[t], eps);
    slow_pred[t] = CppAD::CondExpGt(slow_pred[t], Type(0), slow_pred[t], eps);

    // Equation 4: Add likelihood contributions using lognormal errors for each variable
    // Lognormal density: density = dnorm(log(x), mu, sigma, true) - log(x)
    nll -= dnorm(log(cots_dat[t] + eps), log(cots_pred[t] + eps), log_std, true) - log(cots_dat[t] + eps);
    nll -= dnorm(log(fast_dat[t] + eps), log(fast_pred[t] + eps), log_std, true) - log(fast_dat[t] + eps);
    nll -= dnorm(log(slow_dat[t] + eps), log(slow_pred[t] + eps), log_std, true) - log(slow_dat[t] + eps);
  }

  // REPORT model predictions for inspection (_pred variables corresponding to _dat observations)
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}
