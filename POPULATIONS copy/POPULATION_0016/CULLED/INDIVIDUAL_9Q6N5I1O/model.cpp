#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA SECTION
  DATA_VECTOR(Time);      // Time vector from the data file (days)
  DATA_VECTOR(N_dat);     // Observed Nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);     // Observed Phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);     // Observed Zooplankton concentration (g C m^-3)

  // PARAMETER SECTION
  PARAMETER(V_max);       // Maximum phytoplankton growth rate (day^-1)
  PARAMETER(K_N);         // Half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(I_max);       // Maximum grazing rate of zooplankton (day^-1)
  PARAMETER(K_P);         // Half-saturation constant for phytoplankton grazing (g C m^-3)
  PARAMETER(beta);        // Zooplankton assimilation efficiency (dimensionless, 0-1)
  PARAMETER(m_P);         // Phytoplankton mortality rate (day^-1)
  PARAMETER(m_Z);         // Zooplankton mortality rate (day^-1)
  PARAMETER(log_sigma_N); // Log of the std dev for Nutrient observation error
  PARAMETER(log_sigma_P); // Log of the std dev for Phytoplankton observation error
  PARAMETER(log_sigma_Z); // Log of the std dev for Zooplankton observation error

  // Initialize the negative log-likelihood
  Type nll = 0.0;

  // Parameter transformations and bounds
  // Transform log-standard deviations to standard deviations
  Type sigma_N = exp(log_sigma_N);
  Type sigma_P = exp(log_sigma_P);
  Type sigma_Z = exp(log_sigma_Z);

  // Apply smooth penalties to enforce biologically meaningful parameter ranges
  if (V_max < 0.0) nll += 100 * pow(V_max, 2);         // Penalty for negative rate
  if (K_N < 0.0) nll += 100 * pow(K_N, 2);           // Penalty for negative concentration
  if (I_max < 0.0) nll += 100 * pow(I_max, 2);         // Penalty for negative rate
  if (K_P < 0.0) nll += 100 * pow(K_P, 2);           // Penalty for negative concentration
  if (m_P < 0.0) nll += 100 * pow(m_P, 2);           // Penalty for negative mortality
  if (m_Z < 0.0) nll += 100 * pow(m_Z, 2);           // Penalty for negative mortality
  if (beta < 0.0) nll += 100 * pow(beta, 2);         // Penalty for efficiency < 0
  if (beta > 1.0) nll += 100 * pow(beta - 1.0, 2);   // Penalty for efficiency > 1

  // Get the number of time steps
  int n_steps = Time.size();

  // Create vectors to store model predictions
  vector<Type> N_pred(n_steps);
  vector<Type> P_pred(n_steps);
  vector<Type> Z_pred(n_steps);

  // Initialize predictions with the first data point
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);

  // --- Model Equations ---
  // 1. dN/dt = -Uptake + Remineralization
  //    dN/dt = -(V_max * N / (K_N + N)) * P + (1 - beta) * (I_max * P / (K_P + P)) * Z + m_P*P + m_Z*Z
  // 2. dP/dt = Uptake - Grazing - Mortality
  //    dP/dt = (V_max * N / (K_N + N)) * P - (I_max * P / (K_P + P)) * Z - m_P*P
  // 3. dZ/dt = Assimilated_Grazing - Mortality
  //    dZ/dt = beta * (I_max * P / (K_P + P)) * Z - m_Z*Z

  // Time-stepping loop for numerical integration (Forward Euler method)
  for (int i = 1; i < n_steps; ++i) {
    Type dt = Time(i) - Time(i-1); // Time step duration

    // Retrieve state variables from the previous time step
    Type N_prev = N_pred(i-1);
    Type P_prev = P_pred(i-1);
    Type Z_prev = Z_pred(i-1);

    // --- Calculate process rates based on the previous state ---
    // Phytoplankton nutrient uptake (Michaelis-Menten kinetics)
    Type N_uptake = V_max * N_prev / (K_N + N_prev + Type(1e-8)) * P_prev;

    // Zooplankton grazing on phytoplankton (Holling Type II functional response)
    Type P_grazing = I_max * P_prev / (K_P + P_prev + Type(1e-8)) * Z_prev;

    // --- Calculate the derivatives (change per day) for each state variable ---
    Type dNdt = -N_uptake + (Type(1.0) - beta) * P_grazing + m_P * P_prev + m_Z * Z_prev;
    Type dPdt = N_uptake - P_grazing - m_P * P_prev;
    Type dZdt = beta * P_grazing - m_Z * Z_prev;

    // --- Update state variables using the Forward Euler method ---
    N_pred(i) = N_prev + dNdt * dt;
    P_pred(i) = P_prev + dPdt * dt;
    Z_pred(i) = Z_prev + dZdt * dt;

    // --- Enforce positivity of state variables ---
    // Prevent negative concentrations, which are biologically impossible,
    // and add a penalty to the negative log-likelihood.
    if (N_pred(i) < 0.0) {
      nll -= N_pred(i) * 1000.0; // Add a strong penalty for negativity
      N_pred(i) = Type(1e-8);    // Reset to a small positive value
    }
    if (P_pred(i) < 0.0) {
      nll -= P_pred(i) * 1000.0; // Add a strong penalty for negativity
      P_pred(i) = Type(1e-8);    // Reset to a small positive value
    }
    if (Z_pred(i) < 0.0) {
      nll -= Z_pred(i) * 1000.0; // Add a strong penalty for negativity
      Z_pred(i) = Type(1e-8);    // Reset to a small positive value
    }
  }

  // LIKELIHOOD CALCULATION
  // Compare model predictions with observed data using a lognormal error distribution.
  // This is appropriate for strictly positive data like concentrations.
  for (int i = 0; i < n_steps; ++i) {
    // Add a small constant to prevent log(0) issues with data or predictions
    nll -= dnorm(log(N_dat(i) + Type(1e-8)), log(N_pred(i) + Type(1e-8)), sigma_N, true);
    nll -= dnorm(log(P_dat(i) + Type(1e-8)), log(P_pred(i) + Type(1e-8)), sigma_P, true);
    nll -= dnorm(log(Z_dat(i) + Type(1e-8)), log(Z_pred(i) + Type(1e-8)), sigma_Z, true);
  }

  // REPORTING SECTION
  // Report predicted time series for plotting and evaluation
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);

  // Report estimated parameters and their standard errors
  ADREPORT(V_max);
  ADREPORT(K_N);
  ADREPORT(I_max);
  ADREPORT(K_P);
  ADREPORT(beta);
  ADREPORT(m_P);
  ADREPORT(m_Z);
  ADREPORT(sigma_N);
  ADREPORT(sigma_P);
  ADREPORT(sigma_Z);

  return nll;
}
