#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------
  // DATA INPUTS
  // ------------------------------------------------------------------------
  
  // These are the observed data points for the model to fit against.
  DATA_VECTOR(Time);      // Time points of observations (days)
  DATA_VECTOR(N_dat);     // Observed nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);     // Observed phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);     // Observed zooplankton concentration (g C m^-3)

  // ------------------------------------------------------------------------
  // PARAMETER DECLARATIONS
  // ------------------------------------------------------------------------
  
  // These parameters govern the biological and ecological processes in the model.
  PARAMETER(V_max);       // Max phytoplankton specific growth rate (day^-1). From literature.
  PARAMETER(K_n);         // Half-saturation for nutrient uptake (g C m^-3). From literature.
  PARAMETER(g_max);       // Max zooplankton specific grazing rate (day^-1). From literature.
  PARAMETER(K_p);         // Half-saturation for grazing (g C m^-3). From literature.
  PARAMETER(m_p);         // Phytoplankton mortality rate (day^-1). From literature.
  PARAMETER(m_z);         // Zooplankton mortality rate (day^-1). From literature.
  PARAMETER(beta);        // Zooplankton assimilation efficiency (dimensionless). From literature.
  
  // These parameters define the observation error model.
  PARAMETER(log_sd_N);    // Log of the standard deviation for nutrient observations. Initial estimate.
  PARAMETER(log_sd_P);    // Log of the standard deviation for phytoplankton observations. Initial estimate.
  PARAMETER(log_sd_Z);    // Log of the standard deviation for zooplankton observations. Initial estimate.

  // ------------------------------------------------------------------------
  // MODEL EQUATIONS
  // ------------------------------------------------------------------------
  // This model uses a system of ordinary differential equations (ODEs) to describe
  // the change in N, P, and Z concentrations over time.
  //
  // 1. dN/dt = -Uptake + (1 - beta) * Grazing + m_p * P + m_z * Z
  //    (Change in Nutrients = -Phytoplankton_Uptake + Zooplankton_Excretion + Mortality_Recycling)
  //
  // 2. dP/dt = Uptake - Grazing - m_p * P
  //    (Change in Phytoplankton = Growth_from_Uptake - Grazing_Loss - Mortality_Loss)
  //
  // 3. dZ/dt = beta * Grazing - m_z * Z
  //    (Change in Zooplankton = Assimilated_Growth - Mortality_Loss)
  //
  // Where:
  //    Uptake = V_max * (N / (K_n + N)) * P
  //    Grazing = g_max * (P / (K_p + P)) * Z
  // ------------------------------------------------------------------------

  // Initialize negative log-likelihood
  Type nll = 0.0;

  // Parameter bounds penalties (smooth quadratic penalty)
  // These penalties discourage the optimizer from selecting biologically unrealistic parameter values.
  if (V_max < 0.0) { nll += 100.0 * pow(V_max - 0.0, 2); }
  if (K_n < 0.0) { nll += 100.0 * pow(K_n - 0.0, 2); }
  if (g_max < 0.0) { nll += 100.0 * pow(g_max - 0.0, 2); }
  if (K_p < 0.0) { nll += 100.0 * pow(K_p - 0.0, 2); }
  if (m_p < 0.0) { nll += 100.0 * pow(m_p - 0.0, 2); }
  if (m_z < 0.0) { nll += 100.0 * pow(m_z - 0.0, 2); }
  if (beta < 0.0 || beta > 1.0) { nll += 100.0 * pow(beta - 0.5, 2); }

  // Get number of time steps
  int n_steps = Time.size();

  // Create vectors to store model predictions
  vector<Type> N_pred(n_steps);
  vector<Type> P_pred(n_steps);
  vector<Type> Z_pred(n_steps);

  // Initialize model predictions with the first data point
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);

  // Small constant to prevent division by zero or log of zero
  Type stability_const = 1e-8;

  // Main loop for simulating the model over time using the Euler method
  for (int i = 1; i < n_steps; ++i) {
    Type dt = Time(i) - Time(i-1);

    // Use predicted values from the previous time step for calculations
    Type N_prev = N_pred(i-1);
    Type P_prev = P_pred(i-1);
    Type Z_prev = Z_pred(i-1);

    // --- Calculate process rates ---
    
    // Phytoplankton nutrient uptake (Michaelis-Menten kinetics)
    Type uptake = V_max * (N_prev / (K_n + N_prev + stability_const)) * P_prev;
    
    // Zooplankton grazing on phytoplankton (Holling Type II functional response)
    Type grazing = g_max * (P_prev / (K_p + P_prev + stability_const)) * Z_prev;

    // --- Calculate derivatives (rate of change) for each state variable ---
    Type dNdt = -uptake + (1.0 - beta) * grazing + m_p * P_prev + m_z * Z_prev;
    Type dPdt = uptake - grazing - m_p * P_prev;
    Type dZdt = beta * grazing - m_z * Z_prev;

    // --- Update state variables using Euler forward method ---
    N_pred(i) = N_prev + dNdt * dt;
    P_pred(i) = P_prev + dPdt * dt;
    Z_pred(i) = Z_prev + dZdt * dt;

    // --- Enforce positivity constraint ---
    // This ensures concentrations do not become negative, which is biologically impossible.
    N_pred(i) = CppAD::CondExpGe(N_pred(i), Type(0.0), N_pred(i), stability_const);
    P_pred(i) = CppAD::CondExpGe(P_pred(i), Type(0.0), P_pred(i), stability_const);
    Z_pred(i) = CppAD::CondExpGe(Z_pred(i), Type(0.0), Z_pred(i), stability_const);
  }

  // ------------------------------------------------------------------------
  // LIKELIHOOD CALCULATION
  // ------------------------------------------------------------------------
  
  // Transform log standard deviations to standard deviations
  Type sd_N = exp(log_sd_N);
  Type sd_P = exp(log_sd_P);
  Type sd_Z = exp(log_sd_Z);

  // Fixed minimum standard deviation to prevent numerical issues with very small data values
  Type min_sd = 1e-4;

  // Compare model predictions with observed data to calculate likelihood
  // A lognormal error distribution is assumed, suitable for positive-only data.
  for (int i = 0; i < n_steps; ++i) {
    nll -= dnorm(log(N_dat(i)), log(N_pred(i) + stability_const), sd_N + min_sd, true);
    nll -= dnorm(log(P_dat(i)), log(P_pred(i) + stability_const), sd_P + min_sd, true);
    nll -= dnorm(log(Z_dat(i)), log(Z_pred(i) + stability_const), sd_Z + min_sd, true);
  }

  // ------------------------------------------------------------------------
  // REPORTING SECTION
  // ------------------------------------------------------------------------
  
  // Report predicted time series for plotting and evaluation
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  
  return nll;
}
