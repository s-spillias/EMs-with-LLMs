#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------
  // DATA INPUTS
  // ------------------------------------------------------------------------
  DATA_VECTOR(Year);          // Vector of years for the time series
  DATA_VECTOR(cots_dat);      // Observed COTS density (individuals/m2)
  DATA_VECTOR(fast_dat);      // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);      // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);       // Observed sea-surface temperature (Celsius) - available for future model versions
  DATA_VECTOR(cotsimm_dat);   // COTS larval immigration rate (individuals/m2/year)

  // ------------------------------------------------------------------------
  // PARAMETERS
  // ------------------------------------------------------------------------
  PARAMETER(a_c);      // COTS maximum food intake rate parameter (year^-1)
  PARAMETER(h_c);      // COTS food saturation parameter, inverse of half-saturation constant (1/% cover)
  PARAMETER(e_c);      // COTS assimilation efficiency, converting food to growth (dimensionless)
  PARAMETER(m_c_base); // COTS baseline natural mortality rate (year^-1)
  PARAMETER(pred_rate); // Maximum predation rate on COTS by generalist predators (year^-1)
  PARAMETER(pred_h);   // COTS density at which predator satiation is half-maximal (individuals/m^2)
  PARAMETER(m_c_den);  // COTS density-dependent mortality coefficient (m^2 / (individual * year))
  PARAMETER(gamma);    // Coral loss conversion factor from COTS predation ((% cover * m^2) / individual)
  PARAMETER(pref_f);   // COTS preference for fast-growing coral (dimensionless, 0-1)
  PARAMETER(r_f);      // Intrinsic growth rate of fast-growing coral (year^-1)
  PARAMETER(K_f);      // Carrying capacity of fast-growing coral (%)
  PARAMETER(alpha_fs); // Competition coefficient of slow coral on fast coral (dimensionless)
  PARAMETER(r_s);      // Intrinsic growth rate of slow-growing coral (year^-1)
  PARAMETER(K_s);      // Carrying capacity of slow-growing coral (%)
  PARAMETER(alpha_sf); // Competition coefficient of fast coral on slow coral (dimensionless)
  PARAMETER(log_sigma_cots); // Log of standard deviation for COTS observation error
  PARAMETER(log_sigma_fast); // Log of standard deviation for fast coral observation error
  PARAMETER(log_sigma_slow); // Log of standard deviation for slow coral observation error

  // ------------------------------------------------------------------------
  // MODEL SETUP
  // ------------------------------------------------------------------------
  int n_obs = Year.size(); // Number of observations in the time series

  // Initialize prediction vectors
  vector<Type> cots_pred(n_obs);
  vector<Type> fast_pred(n_obs);
  vector<Type> slow_pred(n_obs);

  // Set initial conditions from the first data point
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Initialize negative log-likelihood
  Type nll = 0.0;

  // ------------------------------------------------------------------------
  // PARAMETER BOUNDS (using smooth penalties)
  // ------------------------------------------------------------------------
  // Penalize if positive-only parameters go below zero
  if (a_c < 0.0) nll -= dnorm(a_c, Type(0.0), Type(0.1), true);
  if (h_c < 0.0) nll -= dnorm(h_c, Type(0.0), Type(0.1), true);
  if (m_c_base < 0.0) nll -= dnorm(m_c_base, Type(0.0), Type(0.1), true);
  if (pred_rate < 0.0) nll -= dnorm(pred_rate, Type(0.0), Type(0.1), true);
  if (pred_h < 0.0) nll -= dnorm(pred_h, Type(0.0), Type(0.1), true);
  if (m_c_den < 0.0) nll -= dnorm(m_c_den, Type(0.0), Type(0.1), true);
  if (gamma < 0.0) nll -= dnorm(gamma, Type(0.0), Type(0.1), true);
  if (r_f < 0.0) nll -= dnorm(r_f, Type(0.0), Type(0.1), true);
  if (K_f < 0.0) nll -= dnorm(K_f, Type(0.0), Type(0.1), true);
  if (alpha_fs < 0.0) nll -= dnorm(alpha_fs, Type(0.0), Type(0.1), true);
  if (r_s < 0.0) nll -= dnorm(r_s, Type(0.0), Type(0.1), true);
  if (K_s < 0.0) nll -= dnorm(K_s, Type(0.0), Type(0.1), true);
  if (alpha_sf < 0.0) nll -= dnorm(alpha_sf, Type(0.0), Type(0.1), true);

  // Penalize if proportional parameters (0-1) go out of bounds
  if (e_c < 0.0) nll -= dnorm(e_c, Type(0.0), Type(0.1), true);
  if (e_c > 1.0) nll -= dnorm(e_c, Type(1.0), Type(0.1), true);
  if (pref_f < 0.0) nll -= dnorm(pref_f, Type(0.0), Type(0.1), true);
  if (pref_f > 1.0) nll -= dnorm(pref_f, Type(1.0), Type(0.1), true);

  // ------------------------------------------------------------------------
  // ECOLOGICAL PROCESS MODEL (Time-step simulation)
  // ------------------------------------------------------------------------
  // Equation descriptions:
  // 1. COTS food intake: A Holling Type II functional response where intake saturates with total available coral food.
  // 2. COTS population dynamics: A differential equation including growth from assimilated food, a complex mortality term, and larval immigration.
  // 2a. COTS mortality: Includes baseline mortality, depensatory mortality from predator satiation (predator pit), and linear density-dependent mortality at high densities.
  // 3. Coral predation loss: Total coral biomass consumed by the COTS population, partitioned between fast and slow corals based on preference and availability.
  // 4. Fast-growing coral dynamics: Logistic growth, reduced by competition from slow corals and by COTS predation.
  // 5. Slow-growing coral dynamics: Logistic growth, reduced by competition from fast corals and by COTS predation.
  // 6. Positivity constraints: Ensure that population densities do not fall below a small positive value to maintain numerical stability.

  for (int t = 1; t < n_obs; ++t) {
    // Calculate total available food, weighted by COTS preference
    Type available_food = pref_f * fast_pred(t-1) + (1.0 - pref_f) * slow_pred(t-1);

    // Eq 1: COTS per-capita food intake rate (Holling Type II)
    Type food_intake = a_c * available_food / (1.0 + h_c * available_food + 1e-8);

    // Eq 2: COTS population dynamics (Euler step, dt=1 year)
    Type cots_growth = e_c * food_intake; // Growth from assimilated food
    // Eq 2a: COTS mortality (baseline + predator pit + density-dependence)
    Type cots_mortality = m_c_base + (pred_rate / (pred_h + cots_pred(t-1))) + m_c_den * cots_pred(t-1);
    cots_pred(t) = cots_pred(t-1) + cots_pred(t-1) * (cots_growth - cots_mortality) + cotsimm_dat(t-1);

    // Eq 3: Total coral predation loss, partitioned by type
    Type total_predation_effect = gamma * food_intake * cots_pred(t-1); // Total % cover loss rate
    Type fast_predation_loss = total_predation_effect * (pref_f * fast_pred(t-1)) / (available_food + 1e-8);
    Type slow_predation_loss = total_predation_effect * ((1.0 - pref_f) * slow_pred(t-1)) / (available_food + 1e-8);

    // Eq 4: Fast-growing coral dynamics (Euler step, dt=1 year)
    Type fast_growth = r_f * fast_pred(t-1) * (1.0 - (fast_pred(t-1) + alpha_fs * slow_pred(t-1)) / (K_f + 1e-8));
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_predation_loss;

    // Eq 5: Slow-growing coral dynamics (Euler step, dt=1 year)
    Type slow_growth = r_s * slow_pred(t-1) * (1.0 - (slow_pred(t-1) + alpha_sf * fast_pred(t-1)) / (K_s + 1e-8));
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_predation_loss;

    // Eq 6: Enforce positivity for all state variables
    cots_pred(t) = CppAD::CondExpGe(cots_pred(t), Type(0.0), cots_pred(t), Type(1e-8));
    fast_pred(t) = CppAD::CondExpGe(fast_pred(t), Type(0.0), fast_pred(t), Type(1e-8));
    slow_pred(t) = CppAD::CondExpGe(slow_pred(t), Type(0.0), slow_pred(t), Type(1e-8));
  }

  // ------------------------------------------------------------------------
  // LIKELIHOOD CALCULATION
  // ------------------------------------------------------------------------
  // Use a lognormal distribution for strictly positive data (abundances, cover)
  // This is implemented by assuming log(data) is normally distributed
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_fast = exp(log_sigma_fast);
  Type sigma_slow = exp(log_sigma_slow);

  for (int t = 0; t < n_obs; ++t) {
    nll -= dnorm(log(cots_dat(t) + 1e-8), log(cots_pred(t)), sigma_cots, true);
    nll -= dnorm(log(fast_dat(t) + 1e-8), log(fast_pred(t)), sigma_fast, true);
    nll -= dnorm(log(slow_dat(t) + 1e-8), log(slow_pred(t)), sigma_slow, true);
  }

  // ------------------------------------------------------------------------
  // REPORTING SECTION
  // ------------------------------------------------------------------------
  // Report predicted time series for plotting and analysis
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  // Report parameters and their derived values
  REPORT(a_c);
  REPORT(h_c);
  REPORT(e_c);
  REPORT(m_c_base);
  REPORT(pred_rate);
  REPORT(pred_h);
  REPORT(m_c_den);
  REPORT(gamma);
  REPORT(pref_f);
  REPORT(r_f);
  REPORT(K_f);
  REPORT(alpha_fs);
  REPORT(r_s);
  REPORT(K_s);
  REPORT(alpha_sf);
  REPORT(sigma_cots);
  REPORT(sigma_fast);
  REPORT(sigma_slow);
  
  // Report standard errors for predicted values
  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);

  return nll;
}
