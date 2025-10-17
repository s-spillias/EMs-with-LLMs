#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------
  // DATA
  // ------------------------------------------------------------------------
  
  // Time vector
  DATA_VECTOR(Time_days); // Time points of observations (days)
  
  // Observed data
  DATA_VECTOR(N_dat); // Nutrient concentration observations (g C m^-3)
  DATA_VECTOR(P_dat); // Phytoplankton concentration observations (g C m^-3)
  DATA_VECTOR(Z_dat); // Zooplankton concentration observations (g C m^-3)
  
  // ------------------------------------------------------------------------
  // PARAMETERS
  // ------------------------------------------------------------------------
  
  // These parameters are transformed to ensure they remain in a valid biological range.
  // For example, rates are log-transformed to ensure positivity.
  
  // Phytoplankton growth
  PARAMETER(log_Vm);      // log of maximum phytoplankton growth rate (day^-1)
  PARAMETER(log_Ks);      // log of nutrient uptake half-saturation constant (g C m^-3)
  
  // Zooplankton grazing
  PARAMETER(log_gmax);    // log of maximum zooplankton grazing rate (day^-1)
  PARAMETER(log_Kp);      // log of grazing half-saturation constant (g C m^-3)
  PARAMETER(logit_beta);  // logit of zooplankton assimilation efficiency (dimensionless)
  
  // Mortality
  PARAMETER(log_mP);      // log of phytoplankton linear mortality rate (day^-1)
  PARAMETER(log_mZ);      // log of zooplankton quadratic mortality rate ((g C m^-3)^-1 day^-1)
  
  // Observation error
  PARAMETER(log_sigma_N); // log of standard deviation for Nutrient observations
  PARAMETER(log_sigma_P); // log of standard deviation for Phytoplankton observations
  PARAMETER(log_sigma_Z); // log of standard deviation for Zooplankton observations
  
  // --- Parameter transformations ---
  Type Vm = exp(log_Vm);          // Vm: Maximum phytoplankton growth rate (day^-1). Determines the max rate of nutrient uptake.
  Type Ks = exp(log_Ks);          // Ks: Nutrient uptake half-saturation constant (g C m^-3). Nutrient level at which growth is half of Vm.
  Type gmax = exp(log_gmax);      // gmax: Maximum zooplankton grazing rate (day^-1). Max rate of phytoplankton consumption.
  Type Kp = exp(log_Kp);          // Kp: Grazing half-saturation constant (g C m^-3). Phytoplankton density at which grazing is half of gmax.
  Type beta = 1.0 / (1.0 + exp(-logit_beta)); // beta: Zooplankton assimilation efficiency (dimensionless, 0-1). Fraction of grazed P converted to Z biomass.
  Type mP = exp(log_mP);          // mP: Phytoplankton linear mortality rate (day^-1). Rate of non-grazing related death.
  Type mZ = exp(log_mZ);          // mZ: Zooplankton quadratic mortality rate ((g C m^-3)^-1 day^-1). Represents predation on zooplankton.
  
  Type sigma_N = exp(log_sigma_N); // sigma_N: Observation error SD for N.
  Type sigma_P = exp(log_sigma_P); // sigma_P: Observation error SD for P.
  Type sigma_Z = exp(log_sigma_Z); // sigma_Z: Observation error SD for Z.
  
  // ------------------------------------------------------------------------
  // MODEL EQUATIONS
  // ------------------------------------------------------------------------
  
  /*
  The model uses a system of ordinary differential equations (ODEs) to describe the change in
  Nutrient (N), Phytoplankton (P), and Zooplankton (Z) concentrations over time.
  The equations are solved numerically using the forward Euler method.
  
  Equations:
  1. dN/dt = -Uptake + Excretion + Remineralization
     - Uptake: Vm * (N / (Ks + N)) * P
     - Excretion: (1 - beta) * Grazing
     - Remineralization: mP * P + mZ * Z^2
  2. dP/dt = Uptake - Grazing - Phytoplankton_Mortality
     - Grazing: gmax * (P / (Kp + P)) * Z
     - Phytoplankton_Mortality: mP * P
  3. dZ/dt = Assimilation - Zooplankton_Mortality
     - Assimilation: beta * Grazing
     - Zooplankton_Mortality: mZ * Z^2
  */
  
  int n_obs = Time_days.size();
  vector<Type> N_pred(n_obs);
  vector<Type> P_pred(n_obs);
  vector<Type> Z_pred(n_obs);
  
  // Initialize predicted state vectors with the first observation
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);
  
  // --- Time-stepping loop (Euler method) ---
  for (int i = 1; i < n_obs; ++i) {
    Type dt = Time_days(i) - Time_days(i - 1);
    
    // Concentrations from the previous time step
    Type N_prev = N_pred(i - 1);
    Type P_prev = P_pred(i - 1);
    Type Z_prev = Z_pred(i - 1);
    
    // Add small constant (1e-8) to denominators to prevent division by zero
    Type small_const = 1e-8;
    
    // --- Ecological process rates ---
    // 1. Phytoplankton nutrient uptake (Michaelis-Menten)
    Type uptake = Vm * (N_prev / (Ks + N_prev + small_const)) * P_prev;
    
    // 2. Zooplankton grazing on phytoplankton (Holling Type II)
    Type grazing = gmax * (P_prev / (Kp + P_prev + small_const)) * Z_prev;
    
    // 3. Phytoplankton mortality
    Type p_mortality = mP * P_prev;
    
    // 4. Zooplankton mortality (quadratic)
    Type z_mortality = mZ * Z_prev * Z_prev;
    
    // 5. Nutrient regeneration from zooplankton excretion
    Type excretion = (1.0 - beta) * grazing;
    
    // --- State variable updates ---
    // Change in Nutrient concentration
    Type dN = -uptake + excretion + p_mortality + z_mortality;
    N_pred(i) = N_prev + dN * dt;
    
    // Change in Phytoplankton concentration
    Type dP = uptake - grazing - p_mortality;
    P_pred(i) = P_prev + dP * dt;
    
    // Change in Zooplankton concentration
    Type dZ = beta * grazing - z_mortality;
    Z_pred(i) = Z_prev + dZ * dt;

    // Ensure predictions are non-negative
    if (N_pred(i) < 0) N_pred(i) = 0;
    if (P_pred(i) < 0) P_pred(i) = 0;
    if (Z_pred(i) < 0) Z_pred(i) = 0;
  }
  
  // ------------------------------------------------------------------------
  // LIKELIHOOD
  // ------------------------------------------------------------------------
  
  Type nll = 0.0; // Initialize negative log-likelihood
  
  // Add a fixed minimum standard deviation to prevent numerical issues
  Type min_sd = 1e-4;
  
  // Lognormal likelihood for each state variable
  // This assumes observations are lognormally distributed around the predicted mean.
  for (int i = 0; i < n_obs; ++i) {
    // Add small constant to predictions to avoid log(0)
    nll -= dlnorm(N_dat(i), log(N_pred(i) + small_const), sigma_N + min_sd, true);
    nll -= dlnorm(P_dat(i), log(P_pred(i) + small_const), sigma_P + min_sd, true);
    nll -= dlnorm(Z_dat(i), log(Z_pred(i) + small_const), sigma_Z + min_sd, true);
  }
  
  // ------------------------------------------------------------------------
  // REPORTING
  // ------------------------------------------------------------------------
  
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  
  ADREPORT(N_pred);
  ADREPORT(P_pred);
  ADREPORT(Z_pred);
  
  return nll;
}
