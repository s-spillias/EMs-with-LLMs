#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- Data Inputs ---
  // These are mapped from the input data file in parameters.json
  DATA_VECTOR(Time_days); // Time points of observations (days)
  DATA_VECTOR(N_dat);     // Observed Nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);     // Observed Phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);     // Observed Zooplankton concentration (g C m^-3)

  // --- Model Parameters ---
  // Initial Conditions
  PARAMETER(log_N0);  // Log of the initial Nutrient concentration (g C m^-3).
  PARAMETER(log_P0);  // Log of the initial Phytoplankton concentration (g C m^-3).
  PARAMETER(log_Z0);  // Log of the initial Zooplankton concentration (g C m^-3).

  // Ecological Process Rates
  PARAMETER(V_max);   // Maximum phytoplankton growth rate (day^-1). Governs the max rate of primary production.
  PARAMETER(K_N);     // Half-saturation constant for nutrient uptake (g C m^-3). Nutrient level at which P growth is half of V_max.
  PARAMETER(g_max);   // Maximum zooplankton grazing rate (day^-1). Governs the max rate of phytoplankton consumption.
  PARAMETER(K_P);     // Half-saturation constant for grazing (g C m^-3). Phytoplankton density at which grazing is half of g_max.
  PARAMETER(beta);    // Zooplankton assimilation efficiency (dimensionless, 0-1). Fraction of grazed P converted to Z biomass.
  PARAMETER(m_P);     // Phytoplankton linear mortality rate (day^-1). Includes natural death and sinking.
  PARAMETER(m_Z);     // Zooplankton quadratic mortality rate ((g C m^-3)^-1 day^-1). Represents density-dependent losses like higher predation.

  // --- Observation Error Parameters ---
  PARAMETER(log_sd_N); // Log of the standard deviation for Nutrient observation error.
  PARAMETER(log_sd_P); // Log of the standard deviation for Phytoplankton observation error.
  PARAMETER(log_sd_Z); // Log of the standard deviation for Zooplankton observation error.

  // Transform log standard deviations to natural scale
  Type sd_N = exp(log_sd_N);
  Type sd_P = exp(log_sd_P);
  Type sd_Z = exp(log_sd_Z);

  // --- Initialize Negative Log-Likelihood and Penalties ---
  Type nll = 0.0;

  // Smooth penalties to enforce biologically meaningful parameter bounds
  // Penalty for beta (assimilation efficiency) to be between 0 and 1
  if (beta < 0.0) nll += 1000.0 * beta * beta;
  if (beta > 1.0) nll += 1000.0 * (beta - 1.0) * (beta - 1.0);

  // Penalties for rate and constant parameters to be positive
  if (V_max < 0.0) nll += 1000.0 * V_max * V_max;
  if (K_N < 0.0)   nll += 1000.0 * K_N * K_N;
  if (g_max < 0.0) nll += 1000.0 * g_max * g_max;
  if (K_P < 0.0)   nll += 1000.0 * K_P * K_P;
  if (m_P < 0.0)   nll += 1000.0 * m_P * m_P;
  if (m_Z < 0.0)   nll += 1000.0 * m_Z * m_Z;

  // --- Model Predictions ---
  int n_obs = N_dat.size();
  vector<Type> N_pred(n_obs);
  vector<Type> P_pred(n_obs);
  vector<Type> Z_pred(n_obs);

  // Initialize predictions with estimated initial conditions
  N_pred(0) = exp(log_N0);
  P_pred(0) = exp(log_P0);
  Z_pred(0) = exp(log_Z0);

  // Constants for numerical stability
  Type epsilon = 1e-8; // Small value to prevent division by zero or log(0)
  Type sigma_min = 1e-4; // Fixed minimum standard deviation for likelihood stability

  // --- Time Integration Loop (Euler Method) ---
  for (int i = 1; i < n_obs; ++i) {
    Type dt = Time_days(i) - Time_days(i-1);

    // State variables from the previous time step
    Type N_prev = N_pred(i-1);
    Type P_prev = P_pred(i-1);
    Type Z_prev = Z_pred(i-1);

    // --- Ecological Process Equations ---
    // 1. Phytoplankton growth (uptake), limited by nutrients (Michaelis-Menten)
    Type uptake = V_max * N_prev / (K_N + N_prev + epsilon) * P_prev;

    // 2. Zooplankton grazing on phytoplankton (Holling Type II functional response)
    Type grazing = g_max * P_prev / (K_P + P_prev + epsilon) * Z_prev;

    // --- System of Ordinary Differential Equations (ODEs) ---
    // 3. Rate of change for Nutrients (dN/dt)
    Type dNdt = -uptake + (1.0 - beta) * grazing + m_P * P_prev + m_Z * Z_prev * Z_prev;

    // 4. Rate of change for Phytoplankton (dP/dt)
    Type dPdt = uptake - grazing - m_P * P_prev;

    // 5. Rate of change for Zooplankton (dZ/dt)
    Type dZdt = beta * grazing - m_Z * Z_prev * Z_prev;

    // Update state variables using Euler forward method
    N_pred(i) = N_prev + dNdt * dt;
    P_pred(i) = P_prev + dPdt * dt;
    Z_pred(i) = Z_prev + dZdt * dt;

    // Ensure predictions are non-negative using a smooth conditional (prevents issues in log-likelihood)
    N_pred(i) = CppAD::CondExpGe(N_pred(i), epsilon, N_pred(i), epsilon);
    P_pred(i) = CppAD::CondExpGe(P_pred(i), epsilon, P_pred(i), epsilon);
    Z_pred(i) = CppAD::CondExpGe(Z_pred(i), epsilon, Z_pred(i), epsilon);
  }

  // --- Likelihood Calculation ---
  // Use a lognormal error distribution, suitable for positive-only concentration data.
  // This is equivalent to a normal distribution on the log-transformed data and predictions.
  for (int i = 0; i < n_obs; ++i) {
    // Add minimum variance for stability, preventing sd from becoming zero
    Type total_sd_N = sqrt(sd_N * sd_N + sigma_min * sigma_min);
    Type total_sd_P = sqrt(sd_P * sd_P + sigma_min * sigma_min);
    Type total_sd_Z = sqrt(sd_Z * sd_Z + sigma_min * sigma_min);

    // Add log-likelihood contribution for each state variable at each time point
    nll -= dnorm(log(N_dat(i) + epsilon), log(N_pred(i)), total_sd_N, true);
    nll -= dnorm(log(P_dat(i) + epsilon), log(P_pred(i)), total_sd_P, true);
    nll -= dnorm(log(Z_dat(i) + epsilon), log(Z_pred(i)), total_sd_Z, true);
  }

  // --- Reporting Section ---
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);

  // Report derived quantities for uncertainty estimation
  ADREPORT(sd_N);
  ADREPORT(sd_P);
  ADREPORT(sd_Z);

  return nll;
}
