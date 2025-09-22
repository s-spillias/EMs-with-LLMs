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
  PARAMETER(log_mu_Z);                  // Log zooplankton mortality rate (day^-1)
  PARAMETER(log_e);                     // Log zooplankton growth efficiency (dimensionless)
  PARAMETER(log_gamma);                 // Log nutrient recycling efficiency (dimensionless)
  PARAMETER(log_N_in);                  // Log external nutrient input rate (g C m^-3 day^-1)
  PARAMETER(log_sigma_N);               // Log observation error for nutrients
  PARAMETER(log_sigma_P);               // Log observation error for phytoplankton
  PARAMETER(log_sigma_Z);               // Log observation error for zooplankton
  
  // Transform parameters to natural scale with biological bounds
  Type r = exp(log_r);                  // Maximum phytoplankton growth rate (day^-1), typical range 0.1-2.0
  Type K_N = exp(log_K_N);              // Half-saturation for nutrient uptake (g C m^-3), typical range 0.01-1.0
  Type mu_P = exp(log_mu_P);            // Phytoplankton mortality rate (day^-1), typical range 0.01-0.5
  Type g_max = exp(log_g_max);          // Maximum grazing rate (day^-1), typical range 0.1-1.0
  Type K_P = exp(log_K_P);              // Half-saturation for grazing (g C m^-3), typical range 0.01-0.5
  Type mu_Z = exp(log_mu_Z);            // Zooplankton mortality rate (day^-1), typical range 0.01-0.3
  Type e = Type(1.0) / (Type(1.0) + exp(-log_e));  // Growth efficiency (0-1), typical range 0.1-0.8
  Type gamma = Type(1.0) / (Type(1.0) + exp(-log_gamma)); // Recycling efficiency (0-1), typical range 0.1-0.9
  Type N_in = exp(log_N_in);            // External nutrient input (g C m^-3 day^-1), typical range 0.001-0.1
  Type sigma_N = exp(log_sigma_N) + Type(1e-6);  // Observation error for N with minimum bound
  Type sigma_P = exp(log_sigma_P) + Type(1e-6);  // Observation error for P with minimum bound  
  Type sigma_Z = exp(log_sigma_Z) + Type(1e-6);  // Observation error for Z with minimum bound
  
  // Initialize state variables
  int n_obs = Time.size();
  vector<Type> N_pred(n_obs);           // Predicted nutrient concentration
  vector<Type> P_pred(n_obs);           // Predicted phytoplankton concentration
  vector<Type> Z_pred(n_obs);           // Predicted zooplankton concentration
  
  // Set initial conditions from first observation
  N_pred(0) = N_dat(0);                 // Initial nutrient concentration
  P_pred(0) = P_dat(0);                 // Initial phytoplankton concentration
  Z_pred(0) = Z_dat(0);                 // Initial zooplankton concentration
  
  // Numerical integration using Euler method
  for(int i = 1; i < n_obs; i++) {
    Type dt = Time(i) - Time(i-1);      // Time step size (days)
    
    // Previous time step values (avoid data leakage)
    Type N_prev = N_pred(i-1) + Type(1e-8);  // Add small constant for numerical stability
    Type P_prev = P_pred(i-1) + Type(1e-8);  // Add small constant for numerical stability
    Type Z_prev = Z_pred(i-1) + Type(1e-8);  // Add small constant for numerical stability
    
    // Equation 1: Nutrient limitation function (Michaelis-Menten kinetics)
    Type f_N = N_prev / (K_N + N_prev);
    
    // Equation 2: Grazing function (Type II functional response)
    Type f_P = P_prev / (K_P + P_prev);
    
    // Equation 3: Phytoplankton growth rate (nutrient-limited)
    Type growth_P = r * f_N * P_prev;
    
    // Equation 4: Zooplankton grazing rate
    Type grazing = g_max * f_P * Z_prev;
    
    // Equation 5: Nutrient recycling from mortality and excretion
    Type recycling = gamma * (mu_P * P_prev + (Type(1.0) - e) * grazing + mu_Z * Z_prev);
    
    // Differential equations for NPZ dynamics
    // Equation 6: Nutrient dynamics
    Type dN_dt = N_in - growth_P + recycling;
    
    // Equation 7: Phytoplankton dynamics  
    Type dP_dt = growth_P - mu_P * P_prev - grazing;
    
    // Equation 8: Zooplankton dynamics
    Type dZ_dt = e * grazing - mu_Z * Z_prev;
    
    // Update state variables using Euler integration
    N_pred(i) = N_prev + dt * dN_dt;    // Nutrient concentration at time i
    P_pred(i) = P_prev + dt * dP_dt;    // Phytoplankton concentration at time i
    Z_pred(i) = Z_prev + dt * dZ_dt;    // Zooplankton concentration at time i
    
    // Ensure non-negative concentrations using smooth maximum function
    Type min_val = Type(1e-8);          // Minimum allowed concentration
    N_pred(i) = (N_pred(i) + sqrt(N_pred(i) * N_pred(i) + min_val * min_val)) / Type(2.0);
    P_pred(i) = (P_pred(i) + sqrt(P_pred(i) * P_pred(i) + min_val * min_val)) / Type(2.0);
    Z_pred(i) = (Z_pred(i) + sqrt(Z_pred(i) * Z_pred(i) + min_val * min_val)) / Type(2.0);
  }
  
  // Calculate negative log-likelihood
  Type nll = Type(0.0);
  
  // Likelihood for nutrient observations (lognormal distribution)
  for(int i = 0; i < n_obs; i++) {
    Type N_obs = N_dat(i) + Type(1e-8);  // Add small constant to prevent log(0)
    Type N_model = N_pred(i) + Type(1e-8); // Add small constant to prevent log(0)
    nll -= dnorm(log(N_obs), log(N_model), sigma_N, true);
  }
  
  // Likelihood for phytoplankton observations (lognormal distribution)
  for(int i = 0; i < n_obs; i++) {
    Type P_obs = P_dat(i) + Type(1e-8);  // Add small constant to prevent log(0)
    Type P_model = P_pred(i) + Type(1e-8); // Add small constant to prevent log(0)
    nll -= dnorm(log(P_obs), log(P_model), sigma_P, true);
  }
  
  // Likelihood for zooplankton observations (lognormal distribution)
  for(int i = 0; i < n_obs; i++) {
    Type Z_obs = Z_dat(i) + Type(1e-8);  // Add small constant to prevent log(0)
    Type Z_model = Z_pred(i) + Type(1e-8); // Add small constant to prevent log(0)
    nll -= dnorm(log(Z_obs), log(Z_model), sigma_Z, true);
  }
  
  // Soft biological constraints using penalty functions
  // Constraint 1: Growth rate should be reasonable for phytoplankton
  if(r > Type(3.0)) nll += Type(10.0) * pow(r - Type(3.0), 2);
  
  // Constraint 2: Mortality rates should not exceed growth rates
  if(mu_P > r) nll += Type(10.0) * pow(mu_P - r, 2);
  if(mu_Z > g_max) nll += Type(10.0) * pow(mu_Z - g_max, 2);
  
  // Constraint 3: Half-saturation constants should be reasonable
  if(K_N > Type(2.0)) nll += Type(5.0) * pow(K_N - Type(2.0), 2);
  if(K_P > Type(1.0)) nll += Type(5.0) * pow(K_P - Type(1.0), 2);
  
  // Report predicted values and parameters
  REPORT(N_pred);                       // Predicted nutrient concentrations
  REPORT(P_pred);                       // Predicted phytoplankton concentrations
  REPORT(Z_pred);                       // Predicted zooplankton concentrations
  REPORT(r);                           // Maximum phytoplankton growth rate
  REPORT(K_N);                         // Nutrient half-saturation constant
  REPORT(mu_P);                        // Phytoplankton mortality rate
  REPORT(g_max);                       // Maximum grazing rate
  REPORT(K_P);                         // Grazing half-saturation constant
  REPORT(mu_Z);                        // Zooplankton mortality rate
  REPORT(e);                           // Growth efficiency
  REPORT(gamma);                       // Recycling efficiency
  REPORT(N_in);                        // External nutrient input
  
  return nll;
}
