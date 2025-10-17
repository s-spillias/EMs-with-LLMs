#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // DATA
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  DATA_VECTOR(Time);          // Time vector for the simulation (days)
  DATA_VECTOR(N_dat);         // Observed nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);         // Observed phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);         // Observed zooplankton concentration (g C m^-3)

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // PARAMETERS
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  PARAMETER(V_max);      // Maximum phytoplankton growth rate (day^-1)
  PARAMETER(K_N);        // Nutrient half-saturation constant for phytoplankton (g C m^-3)
  PARAMETER(g_max);      // Maximum zooplankton grazing rate (day^-1)
  PARAMETER(K_P);        // Phytoplankton half-saturation constant for zooplankton (g C m^-3)
  PARAMETER(beta);       // Zooplankton assimilation efficiency (dimensionless)
  PARAMETER(m_P);        // Phytoplankton mortality rate (day^-1)
  PARAMETER(m_Z);        // Zooplankton mortality rate (day^-1)
  PARAMETER(m_Z2);       // Quadratic zooplankton mortality rate ((g C m^-3)^-1 day^-1)

  // Observation error parameters
  PARAMETER(log_sigma_N); // Log of the standard deviation for Nutrient observations
  PARAMETER(log_sigma_P); // Log of the standard deviation for Phytoplankton observations
  PARAMETER(log_sigma_Z); // Log of the standard deviation for Zooplankton observations

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // DERIVED QUANTITIES
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Type sigma_N = exp(log_sigma_N); // Standard deviation for Nutrient observations
  Type sigma_P = exp(log_sigma_P); // Standard deviation for Phytoplankton observations
  Type sigma_Z = exp(log_sigma_Z); // Standard deviation for Zooplankton observations

  int n_steps = Time.size();
  vector<Type> N_pred(n_steps);
  vector<Type> P_pred(n_steps);
  vector<Type> Z_pred(n_steps);

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // MODEL EQUATIONS
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  /*
  The model simulates the dynamics of Nutrients (N), Phytoplankton (P), and Zooplankton (Z)
  using a set of ordinary differential equations (ODEs), solved with the forward Euler method.

  1. Phytoplankton Growth (Uptake):
     Uptake = V_max * (N / (K_N + N)) * P
     - Phytoplankton growth is limited by nutrient concentration (N) following Michaelis-Menten kinetics.

  2. Zooplankton Grazing:
     Grazing = g_max * (P / (K_P + P)) * Z
     - Zooplankton consume phytoplankton following a Holling Type II functional response.

  3. Nutrient Dynamics (dN/dt):
     dN/dt = -Uptake + (1 - beta) * Grazing + m_P * P + m_Z * Z + m_Z2 * Z^2
     - Nutrients decrease due to phytoplankton uptake.
     - Nutrients increase from unassimilated grazing, and instantaneous remineralization of dead phytoplankton and zooplankton.
     - The quadratic zooplankton mortality term also contributes to remineralization.

  4. Phytoplankton Dynamics (dP/dt):
     dP/dt = Uptake - Grazing - m_P * P
     - Phytoplankton biomass increases via nutrient uptake and decreases due to zooplankton grazing and natural mortality.

  5. Zooplankton Dynamics (dZ/dt):
     dZ/dt = beta * Grazing - m_Z * Z - m_Z2 * Z^2
     - Zooplankton biomass increases by assimilating a fraction (beta) of the grazed phytoplankton.
     - It decreases due to linear natural mortality and density-dependent quadratic mortality (e.g., predation).
  */

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // MODEL IMPLEMENTATION
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Initialize predictions with the first data point
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);

  // Time-stepping loop (Forward Euler method)
  for (int i = 0; i < n_steps - 1; ++i) {
    Type dt = Time(i+1) - Time(i);

    // Use predicted values from the previous time step for calculations
    Type N = N_pred(i);
    Type P = P_pred(i);
    Type Z = Z_pred(i);

    // Add small constants to denominators to prevent division by zero
    Type n_limitation = N / (K_N + N + Type(1e-8));
    Type p_limitation = P / (K_P + P + Type(1e-8));

    // Calculate process rates
    Type uptake = V_max * n_limitation * P;
    Type grazing = g_max * p_limitation * Z;

    // Calculate the change in state variables (ODEs)
    Type dN = -uptake + (Type(1.0) - beta) * grazing + m_P * P + m_Z * Z + m_Z2 * Z * Z;
    Type dP = uptake - grazing - m_P * P;
    Type dZ = beta * grazing - m_Z * Z - m_Z2 * Z * Z;

    // Update state variables for the next time step
    N_pred(i+1) = N + dN * dt;
    P_pred(i+1) = P + dP * dt;
    Z_pred(i+1) = Z + dZ * dt;

    // Ensure predictions remain non-negative
    // Using if-statements to be compatible with AD types, as max() was causing compilation issues.
    if (N_pred(i+1) < Type(1e-8)) {
      N_pred(i+1) = Type(1e-8);
    }
    if (P_pred(i+1) < Type(1e-8)) {
      P_pred(i+1) = Type(1e-8);
    }
    if (Z_pred(i+1) < Type(1e-8)) {
      Z_pred(i+1) = Type(1e-8);
    }
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // LIKELIHOOD CALCULATION
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Type nll = 0.0;

  // Lognormal likelihood for observations.
  // This is appropriate for strictly positive data like concentrations.
  // The TMB dlnorm function was not found during compilation, so we are using
  // the equivalent formulation based on the normal distribution on log-transformed data
  // with a Jacobian correction.
  // log-likelihood(x) = dnorm(log(x), meanlog, sdlog, true) - log(x)
  for (int i = 0; i < n_steps; ++i) {
    // Add a small constant to prevent log(0)
    Type log_N_dat_i = log(N_dat(i) + Type(1e-8));
    nll -= (dnorm(log_N_dat_i, log(N_pred(i)), sigma_N, true) - log_N_dat_i);

    Type log_P_dat_i = log(P_dat(i) + Type(1e-8));
    nll -= (dnorm(log_P_dat_i, log(P_pred(i)), sigma_P, true) - log_P_dat_i);

    Type log_Z_dat_i = log(Z_dat(i) + Type(1e-8));
    nll -= (dnorm(log_Z_dat_i, log(Z_pred(i)), sigma_Z, true) - log_Z_dat_i);
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // PARAMETER BOUNDS (SOFT PENALTIES)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Penalize parameter values that fall outside biologically plausible ranges.
  // This helps guide the optimizer and prevents it from exploring unrealistic parameter space.
  // The penalties are quadratic, increasing further away from the bound.
  Type penalty_weight = 100.0;

  if (V_max < 0.0) { nll += penalty_weight * pow(V_max - 0.0, 2); }
  if (V_max > 5.0) { nll += penalty_weight * pow(V_max - 5.0, 2); }

  if (K_N < 0.0) { nll += penalty_weight * pow(K_N - 0.0, 2); }
  if (K_N > 2.0) { nll += penalty_weight * pow(K_N - 2.0, 2); }

  if (g_max < 0.0) { nll += penalty_weight * pow(g_max - 0.0, 2); }
  if (g_max > 5.0) { nll += penalty_weight * pow(g_max - 5.0, 2); }

  if (K_P < 0.0) { nll += penalty_weight * pow(K_P - 0.0, 2); }
  if (K_P > 2.0) { nll += penalty_weight * pow(K_P - 2.0, 2); }

  if (beta < 0.1) { nll += penalty_weight * pow(beta - 0.1, 2); }
  if (beta > 1.0) { nll += penalty_weight * pow(beta - 1.0, 2); }

  if (m_P < 0.0) { nll += penalty_weight * pow(m_P - 0.0, 2); }
  if (m_P > 1.0) { nll += penalty_weight * pow(m_P - 1.0, 2); }

  if (m_Z < 0.0) { nll += penalty_weight * pow(m_Z - 0.0, 2); }
  if (m_Z > 1.0) { nll += penalty_weight * pow(m_Z - 1.0, 2); }

  if (m_Z2 < 0.0) { nll += penalty_weight * pow(m_Z2 - 0.0, 2); }
  if (m_Z2 > 0.5) { nll += penalty_weight * pow(m_Z2 - 0.5, 2); }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // REPORTING SECTION
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  REPORT(sigma_N);
  REPORT(sigma_P);
  REPORT(sigma_Z);
  REPORT(nll);

  return nll;
}
