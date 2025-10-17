#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA VECTORS
  DATA_VECTOR(Time);    // Time points of observations (days)
  DATA_VECTOR(N_dat);   // Observed Nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);   // Observed Phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);   // Observed Zooplankton concentration (g C m^-3)

  // PARAMETERS
  PARAMETER(V_max);       // Maximum phytoplankton growth rate (day^-1)
  PARAMETER(K_N);         // Nutrient half-saturation constant for phytoplankton uptake (g C m^-3)
  PARAMETER(mu_P);        // Phytoplankton mortality rate (day^-1)
  PARAMETER(I_max);       // Maximum zooplankton grazing rate (day^-1)
  PARAMETER(K_P);         // Phytoplankton half-saturation constant for zooplankton grazing (g C m^-3)
  PARAMETER(beta);        // Zooplankton assimilation efficiency (dimensionless)
  PARAMETER(mu_Z);        // Zooplankton mortality rate (day^-1)
  PARAMETER(lambda);      // Zooplankton excretion/remineralization rate (day^-1)
  PARAMETER(log_sigma_N); // Log of the standard deviation for the Nutrient observation error
  PARAMETER(log_sigma_P); // Log of the standard deviation for the Phytoplankton observation error
  PARAMETER(log_sigma_Z); // Log of the standard deviation for the Zooplankton observation error

  // ------------------------------------------------------------------------
  // MODEL EQUATIONS (Ordinary Differential Equations)
  // 1. dN/dt = -Uptake + P_Mortality_Remineralization + Z_Mortality_Remineralization + Z_Excretion_Remineralization
  //    dN/dt = -V_max * (N / (K_N + N)) * P + mu_P * P + mu_Z * Z + lambda * Z
  // 2. dP/dt = Growth - Grazing - Mortality
  //    dP/dt = V_max * (N / (K_N + N)) * P - I_max * (P / (K_P + P)) * Z - mu_P * P
  // 3. dZ/dt = Assimilated_Grazing - Mortality - Excretion
  //    dZ/dt = beta * I_max * (P / (K_P + P)) * Z - mu_Z * Z - lambda * Z
  // ------------------------------------------------------------------------

  int n_obs = Time.size();
  vector<Type> N_pred(n_obs);
  vector<Type> P_pred(n_obs);
  vector<Type> Z_pred(n_obs);

  // Initialize predictions with the first data point
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);

  // Use Euler method to integrate the differential equations over time
  for (int i = 1; i < n_obs; ++i) {
    Type dt = Time(i) - Time(i-1);

    // Values from the previous time step
    Type N_prev = N_pred(i-1);
    Type P_prev = P_pred(i-1);
    Type Z_prev = Z_pred(i-1);

    // Ecological process rates, with small constants to prevent division by zero
    Type uptake = V_max * (N_prev / (K_N + N_prev + Type(1e-8))) * P_prev;
    Type grazing = I_max * (P_prev / (K_P + P_prev + Type(1e-8))) * Z_prev;

    // Calculate the change in each state variable
    Type dN_dt = -uptake + mu_P * P_prev + mu_Z * Z_prev + lambda * Z_prev;
    Type dP_dt = uptake - grazing - mu_P * P_prev;
    Type dZ_dt = beta * grazing - mu_Z * Z_prev - lambda * Z_prev;

    // Update state variables for the current time step
    N_pred(i) = N_prev + dN_dt * dt;
    P_pred(i) = P_prev + dP_dt * dt;
    Z_pred(i) = Z_prev + dZ_dt * dt;

    // Enforce positivity with a smooth approximation to prevent negative concentrations
    N_pred(i) = CppAD::CondExpGt(N_pred(i), Type(0.0), N_pred(i), Type(1e-8));
    P_pred(i) = CppAD::CondExpGt(P_pred(i), Type(0.0), P_pred(i), Type(1e-8));
    Z_pred(i) = CppAD::CondExpGt(Z_pred(i), Type(0.0), Z_pred(i), Type(1e-8));
  }

  // --- Likelihood Calculation ---
  Type nll = 0.0; // Initialize negative log-likelihood

  // Unpack sigmas and add a minimum value for numerical stability
  Type sigma_N = exp(log_sigma_N) + Type(1e-6);
  Type sigma_P = exp(log_sigma_P) + Type(1e-6);
  Type sigma_Z = exp(log_sigma_Z) + Type(1e-6);

  // Calculate likelihood using a log-normal distribution for each state variable
  for (int i = 0; i < n_obs; ++i) {
    nll -= dnorm(log(N_dat(i) + Type(1e-8)), log(N_pred(i) + Type(1e-8)), sigma_N, true);
    nll -= dnorm(log(P_dat(i) + Type(1e-8)), log(P_pred(i) + Type(1e-8)), sigma_P, true);
    nll -= dnorm(log(Z_dat(i) + Type(1e-8)), log(Z_pred(i) + Type(1e-8)), sigma_Z, true);
  }

  // --- Smooth Penalties for Parameter Bounds ---
  // Penalize parameters that stray outside their biologically meaningful ranges
  if (V_max < 0.1) nll += pow(V_max - 0.1, 2);
  if (V_max > 5.0) nll += pow(V_max - 5.0, 2);
  if (K_N < 0.01) nll += pow(K_N - 0.01, 2);
  if (K_N > 1.0) nll += pow(K_N - 1.0, 2);
  if (mu_P < 0.01) nll += pow(mu_P - 0.01, 2);
  if (mu_P > 0.5) nll += pow(mu_P - 0.5, 2);
  if (I_max < 0.1) nll += pow(I_max - 0.1, 2);
  if (I_max > 5.0) nll += pow(I_max - 5.0, 2);
  if (K_P < 0.05) nll += pow(K_P - 0.05, 2);
  if (K_P > 2.0) nll += pow(K_P - 2.0, 2);
  if (beta < 0.1) nll += pow(beta - 0.1, 2);
  if (beta > 0.9) nll += pow(beta - 0.9, 2);
  if (mu_Z < 0.01) nll += pow(mu_Z - 0.01, 2);
  if (mu_Z > 0.5) nll += pow(mu_Z - 0.5, 2);
  if (lambda < 0.01) nll += pow(lambda - 0.01, 2);
  if (lambda > 0.5) nll += pow(lambda - 0.5, 2);

  // --- Reporting Section ---
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);

  ADREPORT(V_max);
  ADREPORT(K_N);
  ADREPORT(mu_P);
  ADREPORT(I_max);
  ADREPORT(K_P);
  ADREPORT(beta);
  ADREPORT(mu_Z);
  ADREPORT(lambda);
  ADREPORT(sigma_N);
  ADREPORT(sigma_P);
  ADREPORT(sigma_Z);

  return nll;
}
