#include <TMB.hpp>

// 1. Data and parameters are defined, where 'time' is the vector of observation years,
//    'cots_dat', 'fast_dat', and 'slow_dat' are observed data for COTS, fast-growing coral,
//    and slow-growing coral respectively.
// 2. Equations:
//    (1) COTS dynamics: cots[t+1] = cots[t] + (r_cots + extra_growth * outbreak_effect)*cots[t]
//         - m_cots*cots[t] - efficiency*(attack_fast*fast[t] + attack_slow*slow[t])*cots[t]
//         where outbreak_effect is defined as a logistic function of (cots[t] - outbreak_threshold).
//    (2) Fast-growing coral dynamics: fast[t+1] = fast[t] + regrow_fast*(100 - fast[t])
//         - attack_fast * cots[t] * fast[t]
//    (3) Slow-growing coral dynamics: slow[t+1] = slow[t] + regrow_slow*(100 - slow[t])
//         - attack_slow * cots[t] * slow[t]
// 3. Likelihood: For each time step, a lognormal likelihood is used on the observations.
// 4. Numerical stability: A small constant (eps) is added where needed.
// 5. All predicted variables (_pred) are reported via REPORT() for diagnostics.
// 6. Importantly, only previous time step values are used to compute predictions.
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs
  DATA_VECTOR(time);       // Years of observations
  DATA_VECTOR(cots_dat);   // COTS abundance observations (ind/m^2)
  DATA_VECTOR(fast_dat);   // Fast-growing coral cover observations (%)
  DATA_VECTOR(slow_dat);   // Slow-growing coral cover observations (%)

  // Process parameters (log-transformed for positivity)
  PARAMETER(log_r_cots);           // Log intrinsic growth rate of COTS (log(year^-1))
  PARAMETER(log_m_cots);           // Log mortality rate of COTS (log(year^-1))
  PARAMETER(log_attack_fast);      // Log predation rate on fast-growing coral (log(m^2/(ind*year)))
  PARAMETER(log_attack_slow);      // Log predation rate on slow-growing coral (log(m^2/(ind*year)))
  PARAMETER(log_efficiency);       // Log conversion efficiency from coral consumed to COTS growth (log(unitless))
  PARAMETER(log_outbreak_threshold); // Log outbreak initiation threshold for COTS (log(ind/m^2))
  PARAMETER(log_extra_growth);     // Log extra growth rate during outbreak events (log(year^-1))

  // Coral parameters (log-transformed)
  PARAMETER(log_regrow_fast);      // Log regrowth rate for fast-growing coral (% per year)
  PARAMETER(log_regrow_slow);      // Log regrowth rate for slow-growing coral (% per year)
  PARAMETER(log_mort_fast);        // Log mortality rate for fast-growing coral (log(year^-1))
  PARAMETER(log_mort_slow);        // Log mortality rate for slow-growing coral (log(year^-1))

  // Transform parameters to natural scale
  Type r_cots = exp(log_r_cots);               // Intrinsic growth rate of COTS (year^-1)
  Type m_cots = exp(log_m_cots);               // Natural mortality rate of COTS (year^-1)
  Type attack_fast = exp(log_attack_fast);     // Predation rate on fast-growing coral (m^2/(ind*year))
  Type attack_slow = exp(log_attack_slow);     // Predation rate on slow-growing coral (m^2/(ind*year))
  Type efficiency = exp(log_efficiency);       // Conversion efficiency (unitless)
  Type outbreak_threshold = exp(log_outbreak_threshold); // Outbreak threshold (ind/m^2)
  Type extra_growth = exp(log_extra_growth);     // Extra growth rate during outbreak (year^-1)

  Type regrow_fast = exp(log_regrow_fast);       // Regrowth rate for fast-growing coral (% per year)
  Type regrow_slow = exp(log_regrow_slow);       // Regrowth rate for slow-growing coral (% per year)
  Type mort_fast = exp(log_mort_fast);           // Mortality rate for fast-growing coral (year^-1)
  Type mort_slow = exp(log_mort_slow);           // Mortality rate for slow-growing coral (year^-1)

  int n = time.size();  // Number of time steps

  // Initialize predicted state vectors
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Initialize predicted observation vectors
  vector<Type> cots_pred_obs(n);
  vector<Type> fast_pred_obs(n);
  vector<Type> slow_pred_obs(n);

  // Initial conditions set from the first observation
  cots_pred[0] = cots_dat[0];
  fast_pred[0] = fast_dat[0];
  slow_pred[0] = slow_dat[0];

  // Set initial predicted observations equal to state predictions
  cots_pred_obs[0] = cots_pred[0];
  fast_pred_obs[0] = fast_pred[0];
  slow_pred_obs[0] = slow_pred[0];

  Type eps = Type(1e-8);  // small constant for numerical stability
  Type nll = 0.0;         // negative log likelihood

  // --- Likelihood Calculation for Initial Observation ---
  Type sigma = Type(0.1) + eps;
  nll -= dnorm(log(cots_dat[0] + eps), log(cots_pred_obs[0] + eps), sigma, true); // COTS initial likelihood
  nll -= dnorm(log(fast_dat[0] + eps), log(fast_pred_obs[0] + eps), sigma, true);   // Fast-growing coral initial likelihood
  nll -= dnorm(log(slow_dat[0] + eps), log(slow_pred_obs[0] + eps), sigma, true);   // Slow-growing coral initial likelihood

  // Loop over time steps (using values from the previous time step for prediction)
  for(int t = 0; t < n - 1; t++){
    // --- Equation 1: COTS dynamics ---
    Type outbreak_effect = Type(1.0) / (Type(1.0) + exp(-(cots_pred[t] - outbreak_threshold)));
    Type r_effect = r_cots + extra_growth * outbreak_effect;
    Type cots_growth = r_effect * cots_pred[t];
    Type cots_loss = m_cots * cots_pred[t];
    Type coral_predation = efficiency * (attack_fast*fast_pred[t] + attack_slow*slow_pred[t]) * cots_pred[t];
    cots_pred[t+1] = cots_pred[t] + cots_growth - cots_loss - coral_predation;

    // --- Equation 2: Fast-growing coral dynamics ---
    fast_pred[t+1] = fast_pred[t] + regrow_fast*(Type(100.0) - fast_pred[t]) - attack_fast*cots_pred[t]*fast_pred[t];

    // --- Equation 3: Slow-growing coral dynamics ---
    slow_pred[t+1] = slow_pred[t] + regrow_slow*(Type(100.0) - slow_pred[t]) - attack_slow*cots_pred[t]*slow_pred[t];

    // Assign predicted observation equations to avoid data leakage
    cots_pred_obs[t+1] = cots_pred[t+1];
    fast_pred_obs[t+1] = fast_pred[t+1];
    slow_pred_obs[t+1] = slow_pred[t+1];

    // --- Likelihood Calculation ---
    nll -= dnorm(log(cots_dat[t+1] + eps), log(cots_pred_obs[t+1] + eps), sigma, true);
    nll -= dnorm(log(fast_dat[t+1] + eps), log(fast_pred_obs[t+1] + eps), sigma, true);
    nll -= dnorm(log(slow_dat[t+1] + eps), log(slow_pred_obs[t+1] + eps), sigma, true);
  }

  // Define observed prediction equations to avoid data leakage - these are the state predictions used for the likelihood.
  vector<Type> cots_pred_obs = cots_pred;
  vector<Type> fast_pred_obs = fast_pred;
  vector<Type> slow_pred_obs = slow_pred;

  // Report predicted observations corresponding to the data, as required.
  REPORT(cots_pred_obs); // Predicted COTS abundance over time
  REPORT(fast_pred_obs); // Predicted fast-growing coral cover over time
  REPORT(slow_pred_obs); // Predicted slow-growing coral cover over time

  // Also report state variable predictions as derived parameters.
  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);

  return nll;
}
