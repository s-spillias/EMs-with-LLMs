#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------
  // DATA
  // ------------------------------------------------------------------------
  
  // These variable names must match the 'parameter' field in parameters.json
  DATA_VECTOR(Time_days); // Time vector (days)
  DATA_VECTOR(N_dat);     // Observed Nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);     // Observed Phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);     // Observed Zooplankton concentration (g C m^-3)

  // ------------------------------------------------------------------------
  // PARAMETERS
  // ------------------------------------------------------------------------
  
  // These variable names must match the 'parameter' field in parameters.json
  PARAMETER(V_max);   // Maximum phytoplankton uptake rate (day^-1)
  PARAMETER(K_N);     // Nutrient uptake half-saturation constant (g C m^-3)
  PARAMETER(mu_P);    // Phytoplankton mortality rate (day^-1)
  PARAMETER(I_max);   // Maximum zooplankton grazing rate (day^-1)
  PARAMETER(K_P);     // Grazing half-saturation constant (g C m^-3)
  PARAMETER(beta);    // Zooplankton assimilation efficiency (dimensionless)
  PARAMETER(mu_Z);    // Zooplankton mortality rate (day^-1)
  PARAMETER(epsilon); // Remineralization rate (day^-1)

  // Observation error parameters
  PARAMETER(log_sigma_N); // Log of the standard deviation for Nutrient observations
  PARAMETER(log_sigma_P); // Log of the standard deviation for Phytoplankton observations
  PARAMETER(log_sigma_Z); // Log of the standard deviation for Zooplankton observations

  // ------------------------------------------------------------------------
  // MODEL EQUATIONS
  // ------------------------------------------------------------------------
  //
  // 1. Nutrient Uptake (by Phytoplankton): V_max * (N / (K_N + N)) * P
  // 2. Grazing (of Phytoplankton by Zooplankton): I_max * (P / (K_P + P)) * Z
  // 3. Phytoplankton Mortality: mu_P * P
  // 4. Zooplankton Mortality: mu_Z * Z
  // 5. Remineralization: epsilon * (Phytoplankton Mortality + Zooplankton Mortality + Unassimilated Grazing)
  //
  // Resulting differential equations:
  // 6. dN/dt = -Uptake + Remineralization
  // 7. dP/dt =  Uptake - Grazing - Phytoplankton Mortality
  // 8. dZ/dt =  (beta * Grazing) - Zooplankton Mortality
  //
  // ------------------------------------------------------------------------
  
  Type nll = 0.0; // Initialize negative log-likelihood

  // ------------------------------------------------------------------------
  // PARAMETER BOUNDS (using smooth penalties)
  // ------------------------------------------------------------------------
  Type penalty_weight = 100.0;
  if (V_max < 0.0)   nll += penalty_weight * pow(V_max - 0.0, 2);
  if (V_max > 5.0)   nll += penalty_weight * pow(V_max - 5.0, 2);
  if (K_N < 0.01)    nll += penalty_weight * pow(K_N - 0.01, 2);
  if (K_N > 1.0)     nll += penalty_weight * pow(K_N - 1.0, 2);
  if (mu_P < 0.0)    nll += penalty_weight * pow(mu_P - 0.0, 2);
  if (mu_P > 0.5)    nll += penalty_weight * pow(mu_P - 0.5, 2);
  if (I_max < 0.0)   nll += penalty_weight * pow(I_max - 0.0, 2);
  if (I_max > 2.0)   nll += penalty_weight * pow(I_max - 2.0, 2);
  if (K_P < 0.01)    nll += penalty_weight * pow(K_P - 0.01, 2);
  if (K_P > 1.0)     nll += penalty_weight * pow(K_P - 1.0, 2);
  if (beta < 0.1)    nll += penalty_weight * pow(beta - 0.1, 2);
  if (beta > 1.0)    nll += penalty_weight * pow(beta - 1.0, 2);
  if (mu_Z < 0.0)    nll += penalty_weight * pow(mu_Z - 0.0, 2);
  if (mu_Z > 0.5)    nll += penalty_weight * pow(mu_Z - 0.5, 2);
  if (epsilon < 0.0) nll += penalty_weight * pow(epsilon - 0.0, 2);
  if (epsilon > 1.0) nll += penalty_weight * pow(epsilon - 1.0, 2);

  // ------------------------------------------------------------------------
  // MODEL PREDICTIONS (using Forward Euler integration)
  // ------------------------------------------------------------------------
  
  int n_obs = Time_days.size();
  vector<Type> N_pred(n_obs);
  vector<Type> P_pred(n_obs);
  vector<Type> Z_pred(n_obs);

  // Initialize predictions with the first data point
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);

  // Loop through time steps
  for (int i = 1; i < n_obs; ++i) {
    Type dt = Time_days(i) - Time_days(i-1);

    // Values from the previous time step
    Type N_prev = N_pred(i-1);
    Type P_prev = P_pred(i-1);
    Type Z_prev = Z_pred(i-1);

    // Ecological process rates
    Type uptake = V_max * (N_prev / (K_N + N_prev)) * P_prev;
    Type grazing = I_max * (P_prev / (K_P + P_prev + Type(1e-8))) * Z_prev; // Add small constant for stability
    Type p_mortality = mu_P * P_prev;
    Type z_mortality = mu_Z * Z_prev;
    Type unassimilated_grazing = (Type(1.0) - beta) * grazing;
    Type remineralization = epsilon * (p_mortality + z_mortality + unassimilated_grazing);

    // Differentials
    Type dN_dt = -uptake + remineralization;
    Type dP_dt = uptake - grazing - p_mortality;
    Type dZ_dt = (beta * grazing) - z_mortality;

    // Update state variables using Forward Euler method
    N_pred(i) = N_prev + dN_dt * dt;
    P_pred(i) = P_prev + dP_dt * dt;
    Z_pred(i) = Z_prev + dZ_dt * dt;

    // Enforce positivity to prevent negative concentrations
    if (N_pred(i) < 0.0) N_pred(i) = Type(1e-8);
    if (P_pred(i) < 0.0) P_pred(i) = Type(1e-8);
    if (Z_pred(i) < 0.0) Z_pred(i) = Type(1e-8);
  }

  // ------------------------------------------------------------------------
  // LIKELIHOOD CALCULATION
  // ------------------------------------------------------------------------
  
  Type min_sd = 1e-8; // Minimum standard deviation for numerical stability
  Type sigma_N = exp(log_sigma_N) + min_sd;
  Type sigma_P = exp(log_sigma_P) + min_sd;
  Type sigma_Z = exp(log_sigma_Z) + min_sd;

  // Lognormal error distribution for strictly positive concentration data
  // Add a small constant to avoid log(0)
  for (int i=0; i<n_obs; i++){
    nll -= dnorm(log(N_dat(i) + min_sd), log(N_pred(i) + min_sd), sigma_N, true);
    nll -= dnorm(log(P_dat(i) + min_sd), log(P_pred(i) + min_sd), sigma_P, true);
    nll -= dnorm(log(Z_dat(i) + min_sd), log(Z_pred(i) + min_sd), sigma_Z, true);
  }

  // ------------------------------------------------------------------------
  // REPORTING
  // ------------------------------------------------------------------------
  
  // These variables will be available in the output report
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  
  return nll;
}
