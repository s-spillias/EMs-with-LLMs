#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------
  // DATA
  // ------------------------------------------------------------------------
  
  // Observed data vectors
  DATA_VECTOR(Time); // Time vector of observations
  DATA_VECTOR(N_dat); // Observed Nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat); // Observed Phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat); // Observed Zooplankton concentration (g C m^-3)

  // ------------------------------------------------------------------------
  // PARAMETERS
  // ------------------------------------------------------------------------

  // Phytoplankton growth parameters
  PARAMETER(V_max);   // Maximum phytoplankton uptake rate (day^-1)
  PARAMETER(K_N);     // Half-saturation constant for nutrient uptake (g C m^-3)

  // Zooplankton grazing parameters
  PARAMETER(g_max);   // Maximum grazing rate of zooplankton (day^-1)
  PARAMETER(K_P);     // Half-saturation constant for grazing (g C m^-3)
  PARAMETER(beta);    // Zooplankton assimilation efficiency (dimensionless)

  // Mortality and regeneration parameters
  PARAMETER(m_P);      // Phytoplankton linear mortality rate (day^-1)
  PARAMETER(m_P_quad); // Phytoplankton quadratic mortality rate ((g C m^-3)^-1 day^-1)
  PARAMETER(m_Z_lin);  // Zooplankton linear mortality rate (day^-1)
  PARAMETER(m_Z);      // Zooplankton quadratic mortality rate ((g C m^-3)^-1 day^-1)
  PARAMETER(gamma);    // Zooplankton excretion rate (day^-1)
  PARAMETER(epsilon);  // Remineralization rate of dead organic matter (day^-1)

  // Observation error parameters (log scale to ensure positivity)
  PARAMETER(log_sigma_N); // Log standard deviation for Nutrient observations
  PARAMETER(log_sigma_P); // Log standard deviation for Phytoplankton observations
  PARAMETER(log_sigma_Z); // Log standard deviation for Zooplankton observations

  // Transform log standard deviations to natural scale
  Type sigma_N = exp(log_sigma_N);
  Type sigma_P = exp(log_sigma_P);
  Type sigma_Z = exp(log_sigma_Z);

  // ------------------------------------------------------------------------
  // MODEL SETUP
  // ------------------------------------------------------------------------

  int n_obs = Time.size();
  vector<Type> N_pred(n_obs);
  vector<Type> P_pred(n_obs);
  vector<Type> Z_pred(n_obs);

  // Initialize state variables at time 0 with the first observation
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);

  // Negative log-likelihood
  Type nll = 0.0;

  // ------------------------------------------------------------------------
  // MODEL DYNAMICS & LIKELIHOOD
  // ------------------------------------------------------------------------

  // Loop over time steps (from the second observation onwards)
  for (int i = 1; i < n_obs; ++i) {
    // Time step duration
    Type dt = Time(i) - Time(i - 1);

    // State variables from the previous time step
    Type N = N_pred(i - 1);
    Type P = P_pred(i - 1);
    Type Z = Z_pred(i - 1);

    // --- Flux calculations ---

    // Phytoplankton nutrient uptake (Michaelis-Menten)
    Type uptake = V_max * N / (K_N + N) * P;

    // Zooplankton grazing on phytoplankton (Holling Type III)
    Type grazing = g_max * P * P / (K_P * K_P + P * P) * Z;

    // Phytoplankton mortality (linear + quadratic)
    Type p_mortality = m_P * P + m_P_quad * P * P;

    // Zooplankton mortality (linear + quadratic)
    Type z_mortality = m_Z_lin * Z + m_Z * Z * Z;

    // Nutrient regeneration from sloppy feeding (unassimilated grazing)
    Type sloppy_feeding = (1.0 - beta) * grazing;

    // Nutrient regeneration from zooplankton excretion
    Type excretion = gamma * Z;

    // Nutrient regeneration from remineralization of dead organic matter
    Type remineralization = epsilon * (p_mortality + z_mortality);

    // --- Differential equations (Euler forward method) ---
    Type dN = (-uptake + sloppy_feeding + excretion + remineralization) * dt;
    Type dP = (uptake - grazing - p_mortality) * dt;
    Type dZ = (beta * grazing - z_mortality) * dt;

    // --- Update state variables ---
    N_pred(i) = N + dN;
    P_pred(i) = P + dP;
    Z_pred(i) = Z + dZ;

    // --- Safeguards to prevent negative concentrations ---
    if (N_pred(i) < 0) N_pred(i) = 0;
    if (P_pred(i) < 0) P_pred(i) = 0;
    if (Z_pred(i) < 0) Z_pred(i) = 0;

    // --- Likelihood calculation ---
    // Compare predictions with observations at the current time step
    // Use dnorm with log=true for log-likelihood
    nll -= dnorm(N_dat(i), N_pred(i), sigma_N, true);
    nll -= dnorm(P_dat(i), P_pred(i), sigma_P, true);
    nll -= dnorm(Z_dat(i), Z_pred(i), sigma_Z, true);
  }

  // ------------------------------------------------------------------------
  // REPORT PREDICTIONS & RETURN NEGATIVE LOG-LIKELIHOOD
  // ------------------------------------------------------------------------
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  
  return nll;
}
