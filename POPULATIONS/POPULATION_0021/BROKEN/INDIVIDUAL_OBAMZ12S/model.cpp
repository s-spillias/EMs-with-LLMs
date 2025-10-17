#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- Data Inputs ---
  DATA_VECTOR(Time_days);     // Time vector (days). The variable name 'Time (days)' was sanitized.
  DATA_VECTOR(N_dat);         // Observed Nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);         // Observed Phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);         // Observed Zooplankton concentration (g C m^-3)

  // --- Model Parameters ---
  PARAMETER(V_max);           // Maximum phytoplankton growth rate (day^-1)
  PARAMETER(K_N);             // Nutrient half-saturation constant for phytoplankton uptake (g C m^-3)
  PARAMETER(I_max);           // Maximum zooplankton ingestion rate (day^-1)
  PARAMETER(K_P);             // Phytoplankton half-saturation constant for zooplankton grazing (g C m^-3)
  PARAMETER(beta);            // Zooplankton assimilation efficiency (dimensionless)
  PARAMETER(m_P);             // Phytoplankton linear mortality rate (day^-1)
  PARAMETER(m_Z);             // Zooplankton quadratic mortality rate (m^3 (g C)^-1 day^-1)
  
  // --- Error Model Parameters ---
  PARAMETER(log_sigma_N);     // Log of the standard deviation for the Nutrient lognormal error
  PARAMETER(log_sigma_P);     // Log of the standard deviation for the Phytoplankton lognormal error
  PARAMETER(log_sigma_Z);     // Log of the standard deviation for the Zooplankton lognormal error

  // --- Parameter Transformations ---
  Type sigma_N = exp(log_sigma_N); // Standard deviation for Nutrient observations
  Type sigma_P = exp(log_sigma_P); // Standard deviation for Phytoplankton observations
  Type sigma_Z = exp(log_sigma_Z); // Standard deviation for Zooplankton observations

  // --- Negative Log-Likelihood Initialization ---
  Type nll = 0.0;

  // --- Soft Penalties for Parameter Bounds ---
  // This ensures parameters remain within biologically meaningful ranges using smooth penalties.
  if (V_max < 0.0) nll -= dnorm(V_max, Type(0.0), Type(0.1), true);
  if (K_N < 0.0) nll -= dnorm(K_N, Type(0.0), Type(0.1), true);
  if (I_max < 0.0) nll -= dnorm(I_max, Type(0.0), Type(0.1), true);
  if (K_P < 0.0) nll -= dnorm(K_P, Type(0.0), Type(0.1), true);
  if (m_P < 0.0) nll -= dnorm(m_P, Type(0.0), Type(0.1), true);
  if (m_Z < 0.0) nll -= dnorm(m_Z, Type(0.0), Type(0.1), true);
  if (beta < 0.0) nll -= dnorm(beta, Type(0.0), Type(0.1), true); // Penalty for beta < 0
  if (beta > 1.0) nll -= dnorm(beta, Type(1.0), Type(0.1), true); // Penalty for beta > 1

  // --- Model Implementation ---
  int n_steps = Time_days.size(); // Total number of time steps

  // Create vectors to store model predictions
  vector<Type> N_pred(n_steps);
  vector<Type> P_pred(n_steps);
  vector<Type> Z_pred(n_steps);

  // Initialize model predictions with the first observation
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);

  // --- Model Equations Description ---
  // The model is a system of ordinary differential equations (ODEs) describing the change in N, P, and Z over time.
  // It is solved numerically using the Euler forward method.
  // 1. dN/dt = -Uptake + Excretion + Remineralization
  //    Nutrient concentration changes due to consumption by phytoplankton (Uptake), excretion from zooplankton, and remineralization of dead organic matter.
  // 2. dP/dt = Uptake - Grazing - Mortality_P
  //    Phytoplankton concentration changes due to growth (Uptake), being eaten by zooplankton (Grazing), and natural mortality.
  // 3. dZ/dt = Assimilation - Mortality_Z
  //    Zooplankton concentration changes due to growth from assimilated phytoplankton (Assimilation) and mortality.
  //
  // Key processes:
  // - Uptake (Phytoplankton growth): V_max * (N / (K_N + N)) * P
  // - Grazing (Zooplankton feeding): I_max * (P / (K_P + P)) * Z
  // - Assimilation (Zooplankton growth): beta * Grazing
  // - Excretion (Sloppy eating): (1 - beta) * Grazing
  // - Mortality_P (Phytoplankton death): m_P * P
  // - Mortality_Z (Zooplankton death): m_Z * Z^2
  // - Remineralization: Mortality_P and Mortality_Z are returned to the nutrient pool.

  // Time-stepping loop for model simulation
  for (int i = 1; i < n_steps; ++i) {
    Type dt = Time_days(i) - Time_days(i-1); // Time step duration

    // State variables from the previous time step's prediction
    Type N_prev = N_pred(i-1);
    Type P_prev = P_pred(i-1);
    Type Z_prev = Z_pred(i-1);

    // Ecological process rates, with small constants for numerical stability
    Type uptake = V_max * (N_prev / (K_N + N_prev + Type(1e-8))) * P_prev;
    Type grazing = I_max * (P_prev / (K_P + P_prev + Type(1e-8))) * Z_prev;
    
    // System of ODEs
    Type dN_dt = -uptake + (Type(1.0) - beta) * grazing + m_P * P_prev + m_Z * Z_prev * Z_prev;
    Type dP_dt = uptake - grazing - m_P * P_prev;
    Type dZ_dt = beta * grazing - m_Z * Z_prev * Z_prev;

    // Euler forward integration step
    N_pred(i) = N_prev + dN_dt * dt;
    P_pred(i) = P_prev + dP_dt * dt;
    Z_pred(i) = Z_prev + dZ_dt * dt;

    // Enforce positivity with a smooth conditional to prevent negative concentrations
    N_pred(i) = CppAD::CondExpGt(N_pred(i), Type(0.0), N_pred(i), Type(1e-8));
    P_pred(i) = CppAD::CondExpGt(P_pred(i), Type(0.0), P_pred(i), Type(1e-8));
    Z_pred(i) = CppAD::CondExpGt(Z_pred(i), Type(0.0), Z_pred(i), Type(1e-8));
  }

  // --- Likelihood Calculation ---
  // Assumes a lognormal error distribution, suitable for strictly positive data like concentrations.
  for (int i = 0; i < n_steps; ++i) {
    // Add a small constant to data to avoid log(0) in case of zero measurements
    nll -= dnorm(log(N_dat(i) + Type(1e-8)), log(N_pred(i)), sigma_N, true);
    nll -= dnorm(log(P_dat(i) + Type(1e-8)), log(P_pred(i)), sigma_P, true);
    nll -= dnorm(log(Z_dat(i) + Type(1e-8)), log(Z_pred(i)), sigma_Z, true);
  }

  // --- Reporting Section ---
  // Report predicted time series for analysis and visualization
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  
  // Report parameters and their transformed values for inspection
  REPORT(V_max);
  REPORT(K_N);
  REPORT(I_max);
  REPORT(K_P);
  REPORT(beta);
  REPORT(m_P);
  REPORT(m_Z);
  REPORT(sigma_N);
  REPORT(sigma_P);
  REPORT(sigma_Z);

  return nll;
}
