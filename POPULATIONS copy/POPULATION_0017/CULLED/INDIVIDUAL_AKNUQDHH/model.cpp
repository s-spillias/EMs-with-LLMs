#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS
  DATA_VECTOR(Time);  // Time in days
  DATA_VECTOR(N_dat);  // Nutrient concentration observations (g C m^-3)
  DATA_VECTOR(P_dat);  // Phytoplankton concentration observations (g C m^-3)
  DATA_VECTOR(Z_dat);  // Zooplankton concentration observations (g C m^-3)
  
  // PARAMETERS
  PARAMETER(log_r);  // Log of maximum phytoplankton growth rate (day^-1)
  PARAMETER(log_K_N);  // Log of half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(log_g);  // Log of maximum zooplankton grazing rate (day^-1)
  PARAMETER(log_K_P);  // Log of half-saturation constant for zooplankton grazing (g C m^-3)
  PARAMETER(log_m_P);  // Log of phytoplankton mortality rate (day^-1)
  PARAMETER(log_m_Z);  // Log of zooplankton quadratic mortality rate (m^3 g C^-1 day^-1)
  PARAMETER(logit_epsilon);  // Logit of zooplankton assimilation efficiency (dimensionless, 0-1)
  PARAMETER(logit_delta);  // Logit of zooplankton excretion efficiency (dimensionless, 0-1)
  PARAMETER(logit_gamma);  // Logit of nutrient recycling efficiency (dimensionless, 0-1)
  PARAMETER(log_sigma_N);  // Log of observation error SD for nutrients (g C m^-3)
  PARAMETER(log_sigma_P);  // Log of observation error SD for phytoplankton (g C m^-3)
  PARAMETER(log_sigma_Z);  // Log of observation error SD for zooplankton (g C m^-3)
  
  // TRANSFORM PARAMETERS to natural scale with biological constraints
  Type r = exp(log_r);  // Maximum phytoplankton growth rate (day^-1), must be positive
  Type K_N = exp(log_K_N);  // Half-saturation for nutrient uptake (g C m^-3), must be positive
  Type g = exp(log_g);  // Maximum grazing rate (day^-1), must be positive
  Type K_P = exp(log_K_P);  // Half-saturation for grazing (g C m^-3), must be positive
  Type m_P = exp(log_m_P);  // Phytoplankton mortality rate (day^-1), must be positive
  Type m_Z = exp(log_m_Z);  // Zooplankton mortality rate (m^3 g C^-1 day^-1), must be positive
  Type epsilon = Type(1.0) / (Type(1.0) + exp(-logit_epsilon));  // Assimilation efficiency (0-1), logistic transform
  Type delta = Type(1.0) / (Type(1.0) + exp(-logit_delta));  // Excretion efficiency (0-1), logistic transform
  Type gamma = Type(1.0) / (Type(1.0) + exp(-logit_gamma));  // Recycling efficiency (0-1), logistic transform
  Type sigma_N = exp(log_sigma_N);  // Observation error SD for N (g C m^-3), must be positive
  Type sigma_P = exp(log_sigma_P);  // Observation error SD for P (g C m^-3), must be positive
  Type sigma_Z = exp(log_sigma_Z);  // Observation error SD for Z (g C m^-3), must be positive
  
  // MINIMUM STANDARD DEVIATIONS to prevent numerical issues
  Type min_sigma = Type(1e-4);  // Minimum SD (g C m^-3) to ensure numerical stability
  sigma_N = sigma_N + min_sigma;  // Add minimum to nutrient observation error
  sigma_P = sigma_P + min_sigma;  // Add minimum to phytoplankton observation error
  sigma_Z = sigma_Z + min_sigma;  // Add minimum to zooplankton observation error
  
  // INITIALIZE PREDICTION VECTORS
  int n = Time.size();  // Number of time steps in the data
  vector<Type> N_pred(n);  // Predicted nutrient concentrations (g C m^-3)
  vector<Type> P_pred(n);  // Predicted phytoplankton concentrations (g C m^-3)
  vector<Type> Z_pred(n);  // Predicted zooplankton concentrations (g C m^-3)
  
  // SET INITIAL CONDITIONS from first observation
  N_pred(0) = N_dat(0);  // Initialize nutrients from data (g C m^-3)
  P_pred(0) = P_dat(0);  // Initialize phytoplankton from data (g C m^-3)
  Z_pred(0) = Z_dat(0);  // Initialize zooplankton from data (g C m^-3)
  
  // NUMERICAL STABILITY CONSTANTS
  Type eps = Type(1e-8);  // Small constant to prevent division by zero
  
  // SIMULATE DYNAMICS using Euler integration
  for(int i = 1; i < n; i++) {
    Type dt = Time(i) - Time(i-1);  // Time step size (days)
    
    // Get previous time step values to avoid data leakage
    Type N_prev = N_pred(i-1);  // Nutrient concentration at previous time (g C m^-3)
    Type P_prev = P_pred(i-1);  // Phytoplankton concentration at previous time (g C m^-3)
    Type Z_prev = Z_pred(i-1);  // Zooplankton concentration at previous time (g C m^-3)
    
    // Ensure non-negative concentrations for biological realism using CppAD::CondExpGe
    N_prev = CppAD::CondExpGe(N_prev, Type(0.0), N_prev, Type(0.0));  // Nutrients cannot be negative (g C m^-3)
    P_prev = CppAD::CondExpGe(P_prev, Type(0.0), P_prev, Type(0.0));  // Phytoplankton cannot be negative (g C m^-3)
    Z_prev = CppAD::CondExpGe(Z_prev, Type(0.0), Z_prev, Type(0.0));  // Zooplankton cannot be negative (g C m^-3)
    
    // EQUATION 1: Nutrient limitation function (Monod/Michaelis-Menten kinetics)
    Type f_N = N_prev / (K_N + N_prev + eps);  // Nutrient limitation factor (0-1, dimensionless)
    
    // EQUATION 2: Phytoplankton uptake rate
    Type uptake = r * f_N * P_prev;  // Nutrient uptake by phytoplankton (g C m^-3 day^-1)
    
    // EQUATION 3: Zooplankton grazing function (Type II functional response)
    Type grazing = g * P_prev / (K_P + P_prev + eps) * Z_prev;  // Grazing rate (g C m^-3 day^-1)
    
    // EQUATION 4: Phytoplankton mortality
    Type P_mortality = m_P * P_prev;  // Phytoplankton natural mortality (g C m^-3 day^-1)
    
    // EQUATION 5: Zooplankton mortality (quadratic/density-dependent)
    Type Z_mortality = m_Z * Z_prev * Z_prev;  // Zooplankton mortality (g C m^-3 day^-1)
    
    // EQUATION 6: Zooplankton excretion (fast nutrient recycling)
    Type excretion = delta * grazing;  // Direct nutrient regeneration from grazing (g C m^-3 day^-1)
    
    // EQUATION 7: Nutrient recycling from mortality (slow pathway)
    Type recycling = gamma * (P_mortality + Z_mortality);  // Nutrient remineralization (g C m^-3 day^-1)
    
    // EQUATION 8: Nutrient dynamics (dN/dt) - now includes both fast (excretion) and slow (recycling) pathways
    Type dN_dt = -uptake + excretion + recycling;  // Change in nutrient concentration (g C m^-3 day^-1)
    
    // EQUATION 9: Phytoplankton dynamics (dP/dt)
    Type dP_dt = uptake - grazing - P_mortality;  // Change in phytoplankton concentration (g C m^-3 day^-1)
    
    // EQUATION 10: Zooplankton dynamics (dZ/dt)
    Type dZ_dt = epsilon * grazing - Z_mortality;  // Change in zooplankton concentration (g C m^-3 day^-1)
    
    // UPDATE STATE VARIABLES using Euler method
    N_pred(i) = N_prev + dN_dt * dt;  // Update nutrient concentration (g C m^-3)
    P_pred(i) = P_prev + dP_dt * dt;  // Update phytoplankton concentration (g C m^-3)
    Z_pred(i) = Z_prev + dZ_dt * dt;  // Update zooplankton concentration (g C m^-3)
    
    // Ensure predictions remain non-negative using CppAD::CondExpGe
    N_pred(i) = CppAD::CondExpGe(N_pred(i), Type(0.0), N_pred(i), Type(0.0));  // Prevent negative nutrients (g C m^-3)
    P_pred(i) = CppAD::CondExpGe(P_pred(i), Type(0.0), P_pred(i), Type(0.0));  // Prevent negative phytoplankton (g C m^-3)
    Z_pred(i) = CppAD::CondExpGe(Z_pred(i), Type(0.0), Z_pred(i), Type(0.0));  // Prevent negative zooplankton (g C m^-3)
  }
  
  // CALCULATE NEGATIVE LOG-LIKELIHOOD
  Type nll = Type(0.0);  // Initialize negative log-likelihood
  
  // Add likelihood contributions from all observations
  for(int i = 0; i < n; i++) {
    // Nutrient observations (normal distribution)
    nll -= dnorm(N_dat(i), N_pred(i), sigma_N, true);  // Log-likelihood for nutrient data
    
    // Phytoplankton observations (normal distribution)
    nll -= dnorm(P_dat(i), P_pred(i), sigma_P, true);  // Log-likelihood for phytoplankton data
    
    // Zooplankton observations (normal distribution)
    nll -= dnorm(Z_dat(i), Z_pred(i), sigma_Z, true);  // Log-likelihood for zooplankton data
  }
  
  // SOFT PARAMETER CONSTRAINTS using penalties (to guide optimization to biologically realistic ranges)
  // These are gentle penalties, not hard constraints
  
  // Phytoplankton growth rate should be reasonable (0.1 to 2.0 day^-1)
  nll += CppAD::CondExpLt(r, Type(0.05), Type(10.0) * pow(Type(0.05) - r, 2), Type(0.0));  // Penalty if too low
  nll += CppAD::CondExpGt(r, Type(3.0), Type(10.0) * pow(r - Type(3.0), 2), Type(0.0));  // Penalty if too high
  
  // Grazing rate should be reasonable (0.05 to 1.5 day^-1)
  nll += CppAD::CondExpLt(g, Type(0.02), Type(10.0) * pow(Type(0.02) - g, 2), Type(0.0));  // Penalty if too low
  nll += CppAD::CondExpGt(g, Type(2.0), Type(10.0) * pow(g - Type(2.0), 2), Type(0.0));  // Penalty if too high
  
  // Assimilation efficiency should be between 0.1 and 0.9
  nll += CppAD::CondExpLt(epsilon, Type(0.1), Type(10.0) * pow(Type(0.1) - epsilon, 2), Type(0.0));  // Penalty if too low
  nll += CppAD::CondExpGt(epsilon, Type(0.9), Type(10.0) * pow(epsilon - Type(0.9), 2), Type(0.0));  // Penalty if too high
  
  // Excretion efficiency should be between 0.2 and 0.4
  nll += CppAD::CondExpLt(delta, Type(0.2), Type(10.0) * pow(Type(0.2) - delta, 2), Type(0.0));  // Penalty if too low
  nll += CppAD::CondExpGt(delta, Type(0.4), Type(10.0) * pow(delta - Type(0.4), 2), Type(0.0));  // Penalty if too high
  
  // Recycling efficiency should be between 0.1 and 0.9
  nll += CppAD::CondExpLt(gamma, Type(0.1), Type(10.0) * pow(Type(0.1) - gamma, 2), Type(0.0));  // Penalty if too low
  nll += CppAD::CondExpGt(gamma, Type(0.9), Type(10.0) * pow(gamma - Type(0.9), 2), Type(0.0));  // Penalty if too high
  
  // Mass balance constraint: epsilon + delta should be less than 1.0 (some material lost to fecal pellets/detritus)
  Type efficiency_sum = epsilon + delta;  // Total fraction of grazed material accounted for
  nll += CppAD::CondExpGt(efficiency_sum, Type(0.95), Type(50.0) * pow(efficiency_sum - Type(0.95), 2), Type(0.0));  // Strong penalty if sum exceeds 0.95
  
  // REPORT PREDICTIONS AND PARAMETERS
  REPORT(N_pred);  // Report predicted nutrient concentrations
  REPORT(P_pred);  // Report predicted phytoplankton concentrations
  REPORT(Z_pred);  // Report predicted zooplankton concentrations
  REPORT(r);  // Report phytoplankton growth rate
  REPORT(K_N);  // Report nutrient half-saturation constant
  REPORT(g);  // Report grazing rate
  REPORT(K_P);  // Report grazing half-saturation constant
  REPORT(m_P);  // Report phytoplankton mortality rate
  REPORT(m_Z);  // Report zooplankton mortality rate
  REPORT(epsilon);  // Report assimilation efficiency
  REPORT(delta);  // Report excretion efficiency
  REPORT(gamma);  // Report recycling efficiency
  REPORT(sigma_N);  // Report nutrient observation error
  REPORT(sigma_P);  // Report phytoplankton observation error
  REPORT(sigma_Z);  // Report zooplankton observation error
  
  return nll;  // Return negative log-likelihood for optimization
}
