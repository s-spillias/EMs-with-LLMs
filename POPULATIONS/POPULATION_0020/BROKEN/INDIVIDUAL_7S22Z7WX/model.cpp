#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA ---
  DATA_VECTOR(Time_days); // Time vector from the data, in days.
  DATA_VECTOR(N_dat);     // Observed Nutrient concentration (g C m^-3).
  DATA_VECTOR(P_dat);     // Observed Phytoplankton concentration (g C m^-3).
  DATA_VECTOR(Z_dat);     // Observed Zooplankton concentration (g C m^-3).

  // --- PARAMETERS ---
  // These are on a transformed scale to allow for unconstrained optimization.
  PARAMETER(log_V_max);      // Log of maximum phytoplankton growth rate (day^-1).
  PARAMETER(log_K_N);        // Log of half-saturation constant for nutrient uptake (g C m^-3).
  PARAMETER(log_g_max);      // Log of maximum zooplankton grazing rate (day^-1).
  PARAMETER(log_K_P);        // Log of half-saturation constant for grazing (g C m^-3).
  PARAMETER(logit_beta);     // Logit of zooplankton assimilation efficiency (dimensionless).
  PARAMETER(log_m_P);        // Log of phytoplankton mortality rate (day^-1).
  PARAMETER(log_m_Z_quad);   // Log of zooplankton quadratic mortality rate ((g C m^-3)^-1 day^-1).

  // Observation error parameters
  PARAMETER(log_sigma_N); // Log of the standard deviation for Nutrient observations.
  PARAMETER(log_sigma_P); // Log of the standard deviation for Phytoplankton observations.
  PARAMETER(log_sigma_Z); // Log of the standard deviation for Zooplankton observations.

  // --- TRANSFORM PARAMETERS ---
  // Transform from unconstrained scale to natural scale for use in the model.
  Type V_max = exp(log_V_max);
  Type K_N = exp(log_K_N);
  Type g_max = exp(log_g_max);
  Type K_P = exp(log_K_P);
  Type beta = invlogit(logit_beta); // invlogit() transforms from (-inf, inf) to (0, 1).
  Type m_P = exp(log_m_P);
  Type m_Z_quad = exp(log_m_Z_quad);

  Type sigma_N = exp(log_sigma_N);
  Type sigma_P = exp(log_sigma_P);
  Type sigma_Z = exp(log_sigma_Z);

  // --- MODEL SETUP ---
  int n_obs = Time_days.size(); // Number of observations.

  // Predicted state variables
  vector<Type> N_pred(n_obs);
  vector<Type> P_pred(n_obs);
  vector<Type> Z_pred(n_obs);

  // Initialize predictions with the first data point.
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);

  // Initialize negative log-likelihood.
  Type nll = 0.0;

  // --- SIMULATION AND LIKELIHOOD ---
  // Time-stepping loop for model simulation.
  for (int i = 1; i < n_obs; ++i) {
    Type dt = Time_days(i) - Time_days(i-1);

    // Previous state values for the ODE solver.
    Type N_prev = N_pred(i-1);
    Type P_prev = P_pred(i-1);
    Type Z_prev = Z_pred(i-1);

    // --- Ecological process rates based on previous time step ---
    // Nutrient uptake by phytoplankton (Michaelis-Menten kinetics).
    Type uptake = V_max * (N_prev / (K_N + N_prev + Type(1e-8))) * P_prev;

    // Phytoplankton grazing by zooplankton (Holling Type II functional response).
    Type grazing = g_max * (P_prev / (K_P + P_prev + Type(1e-8))) * Z_prev;

    // --- System of Ordinary Differential Equations (ODEs) ---
    // 1. dN/dt: Change in Nutrient concentration.
    //    - Negative term for uptake by phytoplankton.
    //    - Positive term for nutrient recycling from sloppy zooplankton grazing and excretion.
    //    - Positive term for remineralization of dead phytoplankton.
    //    - Positive term for remineralization of dead zooplankton.
    Type dN_dt = -uptake + (Type(1.0) - beta) * grazing + m_P * P_prev + m_Z_quad * Z_prev * Z_prev;

    // 2. dP/dt: Change in Phytoplankton concentration.
    //    - Positive term for growth via nutrient uptake.
    //    - Negative term for grazing by zooplankton.
    //    - Negative term for mortality.
    Type dP_dt = uptake - grazing - m_P * P_prev;

    // 3. dZ/dt: Change in Zooplankton concentration.
    //    - Positive term for growth from assimilated phytoplankton.
    //    - Negative term for quadratic mortality.
    Type dZ_dt = beta * grazing - m_Z_quad * Z_prev * Z_prev;

    // --- Forward Euler integration to predict next state ---
    N_pred(i) = N_prev + dN_dt * dt;
    P_pred(i) = P_prev + dP_dt * dt;
    Z_pred(i) = Z_prev + dZ_dt * dt;

    // --- Numerical stability: ensure predictions are non-negative ---
    // Use CondExp for a smooth approximation of an if-statement to avoid log(0).
    N_pred(i) = CppAD::CondExpGt(N_pred(i), Type(0.0), N_pred(i), Type(1e-9));
    P_pred(i) = CppAD::CondExpGt(P_pred(i), Type(0.0), P_pred(i), Type(1e-9));
    Z_pred(i) = CppAD::CondExpGt(Z_pred(i), Type(0.0), Z_pred(i), Type(1e-9));

    // --- Likelihood calculation ---
    // Lognormal distribution for observation error, robust for positive-only data.
    // A fixed minimum standard deviation prevents numerical issues.
    Type min_sigma = Type(1e-4);
    nll -= dnorm(log(N_dat(i)), log(N_pred(i)), CppAD::CondExpGt(sigma_N, min_sigma, sigma_N, min_sigma), true);
    nll -= dnorm(log(P_dat(i)), log(P_pred(i)), CppAD::CondExpGt(sigma_P, min_sigma, sigma_P, min_sigma), true);
    nll -= dnorm(log(Z_dat(i)), log(Z_pred(i)), CppAD::CondExpGt(sigma_Z, min_sigma, sigma_Z, min_sigma), true);
  }

  // --- REPORTING SECTION ---
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  
  // Report natural scale parameters for interpretation.
  ADREPORT(V_max);
  ADREPORT(K_N);
  ADREPORT(g_max);
  ADREPORT(K_P);
  ADREPORT(beta);
  ADREPORT(m_P);
  ADREPORT(m_Z_quad);
  ADREPORT(sigma_N);
  ADREPORT(sigma_P);
  ADREPORT(sigma_Z);

  return nll;
}
