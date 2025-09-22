#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs
  DATA_VECTOR(Time);                    // Time vector (days)
  DATA_VECTOR(N_dat);                   // Observed nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);                   // Observed phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);                   // Observed zooplankton concentration (g C m^-3)
  
  // Model parameters
  PARAMETER(log_r);                     // Log maximum phytoplankton growth rate (day^-1)
  PARAMETER(log_K_N);                   // Log half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(log_mu_P);                  // Log phytoplankton mortality rate (day^-1)
  PARAMETER(log_g_max);                 // Log maximum zooplankton grazing rate (day^-1)
  PARAMETER(log_K_P);                   // Log half-saturation constant for grazing (g C m^-3)
  PARAMETER(log_e);                     // Log grazing efficiency (dimensionless)
  PARAMETER(log_mu_Z);                  // Log zooplankton mortality rate (day^-1)
  PARAMETER(log_gamma);                 // Log nutrient recycling efficiency (dimensionless)
  PARAMETER(log_sigma_N);               // Log observation error for nutrients
  PARAMETER(log_sigma_P);               // Log observation error for phytoplankton
  PARAMETER(log_sigma_Z);               // Log observation error for zooplankton
  
  // Transform parameters to natural scale with small constants for stability
  Type r = exp(log_r);                  // Maximum phytoplankton growth rate (day^-1)
  Type K_N = exp(log_K_N) + Type(1e-8); // Half-saturation for nutrient uptake (g C m^-3)
  Type mu_P = exp(log_mu_P);            // Phytoplankton mortality rate (day^-1)
  Type g_max = exp(log_g_max);          // Maximum grazing rate (day^-1)
  Type K_P = exp(log_K_P) + Type(1e-8); // Half-saturation for grazing (g C m^-3)
  Type e = exp(log_e);                  // Grazing efficiency (0-1)
  Type mu_Z = exp(log_mu_Z);            // Zooplankton mortality rate (day^-1)
  Type gamma = exp(log_gamma);          // Nutrient recycling efficiency (0-1)
  Type sigma_N = exp(log_sigma_N) + Type(1e-4); // Minimum observation error for N
  Type sigma_P = exp(log_sigma_P) + Type(1e-4); // Minimum observation error for P
  Type sigma_Z = exp(log_sigma_Z) + Type(1e-4); // Minimum observation error for Z
  
  int n_obs = Time.size();              // Number of observations
  
  // Initialize state variables with first observations
  vector<Type> N_pred(n_obs);           // Predicted nutrient concentration
  vector<Type> P_pred(n_obs);           // Predicted phytoplankton concentration
  vector<Type> Z_pred(n_obs);           // Predicted zooplankton concentration
  
  // Set initial conditions with minimum bounds
  N_pred(0) = CppAD::CondExpLt(N_dat(0), Type(1e-6), Type(1e-6), N_dat(0)); // Initial nutrient concentration
  P_pred(0) = CppAD::CondExpLt(P_dat(0), Type(1e-6), Type(1e-6), P_dat(0)); // Initial phytoplankton concentration
  Z_pred(0) = CppAD::CondExpLt(Z_dat(0), Type(1e-6), Type(1e-6), Z_dat(0)); // Initial zooplankton concentration
  
  // Numerical integration using Euler method
  for(int i = 1; i < n_obs; i++) {
    Type dt = Time(i) - Time(i-1);      // Time step (days)
    
    // Previous time step values (avoid data leakage)
    Type N_prev = N_pred(i-1);          // Previous nutrient concentration
    Type P_prev = P_pred(i-1);          // Previous phytoplankton concentration
    Type Z_prev = Z_pred(i-1);          // Previous zooplankton concentration
    
    // Functional responses
    Type f_N = N_prev / (K_N + N_prev);  // Michaelis-Menten nutrient limitation (0-1)
    Type f_P = P_prev / (K_P + P_prev);  // Holling Type II grazing response (0-1)
    
    // Process rates
    Type uptake = r * f_N * P_prev;      // Nutrient uptake by phytoplankton (g C m^-3 day^-1)
    Type grazing = g_max * f_P * Z_prev; // Zooplankton grazing rate (g C m^-3 day^-1)
    Type P_mortality = mu_P * P_prev;    // Phytoplankton mortality (g C m^-3 day^-1)
    Type Z_mortality = mu_Z * Z_prev;    // Zooplankton mortality (g C m^-3 day^-1)
    Type recycling = gamma * (P_mortality + Z_mortality); // Nutrient recycling (g C m^-3 day^-1)
    
    // Differential equations (Euler integration)
    // Equation 1: dN/dt = -uptake + recycling
    Type dN_dt = -uptake + recycling;
    
    // Equation 2: dP/dt = uptake - grazing - P_mortality  
    Type dP_dt = uptake - grazing - P_mortality;
    
    // Equation 3: dZ/dt = e * grazing - Z_mortality
    Type dZ_dt = e * grazing - Z_mortality;
    
    // Update state variables
    Type N_new = N_prev + dt * dN_dt;    // New nutrient concentration
    Type P_new = P_prev + dt * dP_dt;    // New phytoplankton concentration  
    Type Z_new = Z_prev + dt * dZ_dt;    // New zooplankton concentration
    
    // Ensure non-negative concentrations using CppAD::CondExpLt
    N_pred(i) = CppAD::CondExpLt(N_new, Type(1e-6), Type(1e-6), N_new); // Minimum nutrient concentration
    P_pred(i) = CppAD::CondExpLt(P_new, Type(1e-6), Type(1e-6), P_new); // Minimum phytoplankton concentration
    Z_pred(i) = CppAD::CondExpLt(Z_new, Type(1e-6), Type(1e-6), Z_new); // Minimum zooplankton concentration
  }
  
  // Calculate negative log-likelihood
  Type nll = 0.0;                       // Initialize negative log-likelihood
  
  // Likelihood for all observations using normal distribution on log scale
  for(int i = 0; i < n_obs; i++) {
    // Add small constants to prevent log(0)
    Type N_obs_safe = N_dat(i) + Type(1e-8);     // Safe observed nutrient
    Type P_obs_safe = P_dat(i) + Type(1e-8);     // Safe observed phytoplankton
    Type Z_obs_safe = Z_dat(i) + Type(1e-8);     // Safe observed zooplankton
    Type N_pred_safe = N_pred(i) + Type(1e-8);   // Safe predicted nutrient
    Type P_pred_safe = P_pred(i) + Type(1e-8);   // Safe predicted phytoplankton
    Type Z_pred_safe = Z_pred(i) + Type(1e-8);   // Safe predicted zooplankton
    
    // Normal likelihood on log scale (equivalent to lognormal)
    nll -= dnorm(log(N_obs_safe), log(N_pred_safe), sigma_N, true);
    nll -= dnorm(log(P_obs_safe), log(P_pred_safe), sigma_P, true);
    nll -= dnorm(log(Z_obs_safe), log(Z_pred_safe), sigma_Z, true);
  }
  
  // Parameter bounds using smooth penalties
  // Growth rate penalty (should be positive but not excessive)
  nll += Type(0.1) * CppAD::CondExpGt(log_r, Type(2.0), pow(log_r - Type(2.0), 2), Type(0.0)); // Soft upper bound
  nll += Type(0.1) * CppAD::CondExpLt(log_r, Type(-5.0), pow(log_r - Type(-5.0), 2), Type(0.0)); // Soft lower bound
  
  // Grazing efficiency penalty (should be between 0 and 1)
  nll += Type(0.1) * CppAD::CondExpGt(log_e, Type(0.0), pow(log_e, 2), Type(0.0)); // Soft upper bound at 1.0
  nll += Type(0.1) * CppAD::CondExpLt(log_e, Type(-3.0), pow(log_e - Type(-3.0), 2), Type(0.0)); // Soft lower bound
  
  // Recycling efficiency penalty (should be between 0 and 1)  
  nll += Type(0.1) * CppAD::CondExpGt(log_gamma, Type(0.0), pow(log_gamma, 2), Type(0.0)); // Soft upper bound at 1.0
  nll += Type(0.1) * CppAD::CondExpLt(log_gamma, Type(-3.0), pow(log_gamma - Type(-3.0), 2), Type(0.0)); // Soft lower bound
  
  // Check for numerical issues
  if(!isfinite(asDouble(nll))) {
    nll = Type(1e10);                   // Return large but finite value if NaN/Inf
  }
  
  // Report predicted values
  REPORT(N_pred);                       // Report nutrient predictions
  REPORT(P_pred);                       // Report phytoplankton predictions
  REPORT(Z_pred);                       // Report zooplankton predictions
  
  return nll;                           // Return negative log-likelihood
}
