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
  Type sigma_N = exp(log_sigma_N) + Type(1e-6); // Minimum observation error for N
  Type sigma_P = exp(log_sigma_P) + Type(1e-6); // Minimum observation error for P
  Type sigma_Z = exp(log_sigma_Z) + Type(1e-6); // Minimum observation error for Z
  
  int n_obs = Time.size();              // Number of observations
  
  // Initialize state variables with first observations
  vector<Type> N_pred(n_obs);           // Predicted nutrient concentration
  vector<Type> P_pred(n_obs);           // Predicted phytoplankton concentration
  vector<Type> Z_pred(n_obs);           // Predicted zooplankton concentration
  
  // Set initial conditions
  N_pred(0) = N_dat(0);                 // Initial nutrient concentration
  P_pred(0) = P_dat(0);                 // Initial phytoplankton concentration
  Z_pred(0) = Z_dat(0);                 // Initial zooplankton concentration
  
  // Numerical integration using Euler method
  for(int i = 1; i < n_obs; i++) {
    Type dt = Time(i) - Time(i-1);      // Time step (days)
    
    // Previous time step values (avoid data leakage)
    Type N_prev = N_pred(i-1) + Type(1e-8); // Add small constant for stability
    Type P_prev = P_pred(i-1) + Type(1e-8); // Add small constant for stability
    Type Z_prev = Z_pred(i-1) + Type(1e-8); // Add small constant for stability
    
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
    N_pred(i) = N_prev + dt * dN_dt;     // New nutrient concentration
    P_pred(i) = P_prev + dt * dP_dt;     // New phytoplankton concentration  
    Z_pred(i) = Z_prev + dt * dZ_dt;     // New zooplankton concentration
    
    // Ensure non-negative concentrations using CppAD::CondExpGt
    N_pred(i) = CppAD::CondExpGt(N_pred(i), Type(1e-8), N_pred(i), Type(1e-8)); // Minimum nutrient concentration
    P_pred(i) = CppAD::CondExpGt(P_pred(i), Type(1e-8), P_pred(i), Type(1e-8)); // Minimum phytoplankton concentration
    Z_pred(i) = CppAD::CondExpGt(Z_pred(i), Type(1e-8), Z_pred(i), Type(1e-8)); // Minimum zooplankton concentration
  }
  
  // Calculate negative log-likelihood
  Type nll = 0.0;                       // Initialize negative log-likelihood
  
  // Likelihood for all observations using lognormal distribution
  for(int i = 0; i < n_obs; i++) {
    // Nutrient observations
    nll -= dnorm(log(N_dat(i) + Type(1e-8)), log(N_pred(i) + Type(1e-8)), sigma_N, true);
    
    // Phytoplankton observations  
    nll -= dnorm(log(P_dat(i) + Type(1e-8)), log(P_pred(i) + Type(1e-8)), sigma_P, true);
    
    // Zooplankton observations
    nll -= dnorm(log(Z_dat(i) + Type(1e-8)), log(Z_pred(i) + Type(1e-8)), sigma_Z, true);
  }
  
  // Parameter bounds using smooth penalties with CppAD::CondExpGt
  // Growth rate penalty (should be positive but not excessive)
  Type r_upper_penalty = CppAD::CondExpGt(log_r - Type(2.0), Type(0.0), log_r - Type(2.0), Type(0.0));
  Type r_lower_penalty = CppAD::CondExpGt(Type(-5.0) - log_r, Type(0.0), Type(-5.0) - log_r, Type(0.0));
  nll += Type(0.5) * pow(r_upper_penalty, 2); // Soft upper bound at ~7.4 day^-1
  nll += Type(0.5) * pow(r_lower_penalty, 2); // Soft lower bound at ~0.007 day^-1
  
  // Grazing efficiency penalty (should be between 0 and 1)
  Type e_upper_penalty = CppAD::CondExpGt(log_e - Type(0.0), Type(0.0), log_e - Type(0.0), Type(0.0));
  Type e_lower_penalty = CppAD::CondExpGt(Type(-3.0) - log_e, Type(0.0), Type(-3.0) - log_e, Type(0.0));
  nll += Type(0.5) * pow(e_upper_penalty, 2); // Soft upper bound at 1.0
  nll += Type(0.5) * pow(e_lower_penalty, 2); // Soft lower bound at ~0.05
  
  // Recycling efficiency penalty (should be between 0 and 1)  
  Type gamma_upper_penalty = CppAD::CondExpGt(log_gamma - Type(0.0), Type(0.0), log_gamma - Type(0.0), Type(0.0));
  Type gamma_lower_penalty = CppAD::CondExpGt(Type(-3.0) - log_gamma, Type(0.0), Type(-3.0) - log_gamma, Type(0.0));
  nll += Type(0.5) * pow(gamma_upper_penalty, 2); // Soft upper bound at 1.0
  nll += Type(0.5) * pow(gamma_lower_penalty, 2); // Soft lower bound at ~0.05
  
  // Report predicted values
  REPORT(N_pred);                       // Report nutrient predictions
  REPORT(P_pred);                       // Report phytoplankton predictions
  REPORT(Z_pred);                       // Report zooplankton predictions
  
  return nll;                           // Return negative log-likelihood
}
