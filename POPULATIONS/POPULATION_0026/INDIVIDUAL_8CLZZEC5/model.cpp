#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs
  DATA_VECTOR(Time);                    // Time vector in days
  DATA_VECTOR(N_dat);                   // Observed nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);                   // Observed phytoplankton concentration (g C m^-3)  
  DATA_VECTOR(Z_dat);                   // Observed zooplankton concentration (g C m^-3)
  
  // Model parameters
  PARAMETER(log_r);                     // Log maximum phytoplankton growth rate (day^-1)
  PARAMETER(log_K_N);                   // Log half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(log_m_P);                   // Log phytoplankton natural mortality rate (day^-1)
  PARAMETER(log_g_max);                 // Log maximum zooplankton grazing rate (day^-1)
  PARAMETER(log_K_P);                   // Log half-saturation constant for zooplankton grazing (g C m^-3)
  PARAMETER(log_e);                     // Log zooplankton assimilation efficiency (dimensionless)
  PARAMETER(log_m_Z);                   // Log zooplankton natural mortality rate (day^-1)
  PARAMETER(log_gamma);                 // Log nutrient recycling efficiency (dimensionless)
  PARAMETER(log_sigma_N);               // Log observation error standard deviation for nutrients
  PARAMETER(log_sigma_P);               // Log observation error standard deviation for phytoplankton
  PARAMETER(log_sigma_Z);               // Log observation error standard deviation for zooplankton
  
  // Transform parameters to natural scale with biological bounds
  Type r = exp(log_r);                  // Maximum phytoplankton growth rate (day^-1), from literature on marine phytoplankton
  Type K_N = exp(log_K_N);              // Half-saturation for nutrient uptake (g C m^-3), typical oceanic values
  Type m_P = exp(log_m_P);              // Phytoplankton mortality rate (day^-1), natural death and sinking
  Type g_max = exp(log_g_max);          // Maximum grazing rate (day^-1), zooplankton feeding capacity
  Type K_P = exp(log_K_P);              // Half-saturation for grazing (g C m^-3), prey handling limitations
  Type e = invlogit(log_e);             // Assimilation efficiency (0-1), using TMB's invlogit function
  Type m_Z = exp(log_m_Z);              // Zooplankton mortality rate (day^-1), natural death and higher predation
  Type gamma = invlogit(log_gamma);     // Recycling efficiency (0-1), using TMB's invlogit function
  Type sigma_N = exp(log_sigma_N);      // Observation error for nutrients, measurement uncertainty
  Type sigma_P = exp(log_sigma_P);      // Observation error for phytoplankton, sampling and measurement error
  Type sigma_Z = exp(log_sigma_Z);      // Observation error for zooplankton, sampling and measurement error
  
  // Add small constants for numerical stability
  Type eps = Type(1e-8);                // Small constant to prevent division by zero
  
  // Initialize state variables and predictions
  int n_obs = N_dat.size();             // Number of observations
  vector<Type> N_pred(n_obs);           // Predicted nutrient concentrations
  vector<Type> P_pred(n_obs);           // Predicted phytoplankton concentrations
  vector<Type> Z_pred(n_obs);           // Predicted zooplankton concentrations
  
  // Set initial conditions from first observations
  N_pred(0) = N_dat(0);                 // Initial nutrient concentration from data
  P_pred(0) = P_dat(0);                 // Initial phytoplankton concentration from data
  Z_pred(0) = Z_dat(0);                 // Initial zooplankton concentration from data
  
  // Numerical integration using Euler method
  for(int i = 1; i < n_obs; i++) {
    Type dt = Time(i) - Time(i-1);      // Time step size (days)
    
    // Previous time step values (avoid data leakage)
    Type N_prev = N_pred(i-1) + eps;    // Previous nutrient concentration with stability constant
    Type P_prev = P_pred(i-1) + eps;    // Previous phytoplankton concentration with stability constant  
    Type Z_prev = Z_pred(i-1) + eps;    // Previous zooplankton concentration with stability constant
    
    // Equation 1: Phytoplankton growth rate with Michaelis-Menten nutrient limitation
    Type growth_rate = r * (N_prev / (K_N + N_prev)) * P_prev;
    
    // Equation 2: Zooplankton grazing rate with Type II functional response
    Type grazing_rate = g_max * (P_prev / (K_P + P_prev)) * Z_prev;
    
    // Equation 3: Phytoplankton natural mortality
    Type P_mortality = m_P * P_prev;
    
    // Equation 4: Zooplankton natural mortality  
    Type Z_mortality = m_Z * Z_prev;
    
    // Equation 5: Nutrient recycling from mortality
    Type recycling = gamma * (P_mortality + Z_mortality);
    
    // Differential equations for NPZ dynamics
    // Equation 6: Nutrient dynamics (consumption by phytoplankton, recycling from mortality)
    Type dN_dt = -growth_rate + recycling;
    
    // Equation 7: Phytoplankton dynamics (growth, grazing loss, natural mortality)
    Type dP_dt = growth_rate - grazing_rate - P_mortality;
    
    // Equation 8: Zooplankton dynamics (growth from grazing, natural mortality)
    Type dZ_dt = e * grazing_rate - Z_mortality;
    
    // Update predictions using Euler integration
    Type N_new = N_prev + dt * dN_dt;   // Nutrient concentration at next time step
    Type P_new = P_prev + dt * dP_dt;   // Phytoplankton concentration at next time step
    Type Z_new = Z_prev + dt * dZ_dt;   // Zooplankton concentration at next time step
    
    // Ensure non-negative concentrations for biological realism
    N_pred(i) = (N_new > eps) ? N_new : eps; // Prevent negative nutrient concentrations
    P_pred(i) = (P_new > eps) ? P_new : eps; // Prevent negative phytoplankton concentrations
    Z_pred(i) = (Z_new > eps) ? Z_new : eps; // Prevent negative zooplankton concentrations
  }
  
  // Calculate negative log-likelihood
  Type nll = Type(0.0);                 // Initialize negative log-likelihood
  
  // Add minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(1e-6);          // Minimum observation error to prevent numerical instability
  Type sigma_N_safe = (sigma_N > min_sigma) ? sigma_N : min_sigma; // Safe nutrient observation error
  Type sigma_P_safe = (sigma_P > min_sigma) ? sigma_P : min_sigma; // Safe phytoplankton observation error  
  Type sigma_Z_safe = (sigma_Z > min_sigma) ? sigma_Z : min_sigma; // Safe zooplankton observation error
  
  // Likelihood contributions from all observations
  for(int i = 0; i < n_obs; i++) {
    // Normal likelihood for nutrient observations
    nll -= dnorm(N_dat(i), N_pred(i), sigma_N_safe, true);
    
    // Normal likelihood for phytoplankton observations
    nll -= dnorm(P_dat(i), P_pred(i), sigma_P_safe, true);
    
    // Normal likelihood for zooplankton observations  
    nll -= dnorm(Z_dat(i), Z_pred(i), sigma_Z_safe, true);
  }
  
  // Add weak priors to prevent extreme parameter values
  nll -= dnorm(log_r, Type(-0.693), Type(1.0), true);        // Prior on growth rate
  nll -= dnorm(log_K_N, Type(-2.303), Type(1.0), true);      // Prior on nutrient half-saturation
  nll -= dnorm(log_m_P, Type(-2.303), Type(1.0), true);      // Prior on phytoplankton mortality
  nll -= dnorm(log_g_max, Type(-0.693), Type(1.0), true);    // Prior on maximum grazing rate
  nll -= dnorm(log_K_P, Type(-2.303), Type(1.0), true);      // Prior on grazing half-saturation
  nll -= dnorm(log_e, Type(0.0), Type(1.0), true);           // Prior on assimilation efficiency
  nll -= dnorm(log_m_Z, Type(-2.303), Type(1.0), true);      // Prior on zooplankton mortality
  nll -= dnorm(log_gamma, Type(-0.405), Type(1.0), true);    // Prior on recycling efficiency
  
  // Report predictions and derived quantities
  REPORT(N_pred);                       // Report predicted nutrient concentrations
  REPORT(P_pred);                       // Report predicted phytoplankton concentrations
  REPORT(Z_pred);                       // Report predicted zooplankton concentrations
  REPORT(r);                            // Report transformed growth rate
  REPORT(K_N);                          // Report transformed nutrient half-saturation
  REPORT(m_P);                          // Report transformed phytoplankton mortality
  REPORT(g_max);                        // Report transformed maximum grazing rate
  REPORT(K_P);                          // Report transformed grazing half-saturation
  REPORT(e);                            // Report transformed assimilation efficiency
  REPORT(m_Z);                          // Report transformed zooplankton mortality
  REPORT(gamma);                        // Report transformed recycling efficiency
  
  return nll;                           // Return negative log-likelihood for minimization
}
