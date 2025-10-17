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
  PARAMETER(log_r);  // Log maximum phytoplankton growth rate (day^-1)
  PARAMETER(log_K_N);  // Log half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(log_g);  // Log maximum zooplankton grazing rate (day^-1)
  PARAMETER(log_K_P);  // Log half-saturation constant for grazing (g C m^-3)
  PARAMETER(log_e);  // Log zooplankton assimilation efficiency (dimensionless, 0-1)
  PARAMETER(log_m_P);  // Log phytoplankton mortality rate (day^-1)
  PARAMETER(log_m_Z);  // Log zooplankton linear mortality rate (day^-1)
  PARAMETER(log_m_Z2);  // Log zooplankton quadratic mortality rate (day^-1 (g C m^-3)^-1)
  PARAMETER(log_gamma);  // Log nutrient recycling efficiency from phytoplankton mortality (dimensionless, 0-1)
  PARAMETER(log_delta);  // Log zooplankton excretion rate (day^-1)
  PARAMETER(log_sigma_N);  // Log observation error SD for nutrients (g C m^-3)
  PARAMETER(log_sigma_P);  // Log observation error SD for phytoplankton (g C m^-3)
  PARAMETER(log_sigma_Z);  // Log observation error SD for zooplankton (g C m^-3)
  
  // Transform parameters from log scale to natural scale
  Type r = exp(log_r);  // Maximum phytoplankton growth rate (day^-1), typical range 0.1-2.0
  Type K_N = exp(log_K_N);  // Half-saturation for nutrient uptake (g C m^-3), typical range 0.01-0.5
  Type g = exp(log_g);  // Maximum grazing rate (day^-1), typical range 0.1-1.0
  Type K_P = exp(log_K_P);  // Half-saturation for grazing (g C m^-3), typical range 0.05-0.5
  Type e = exp(log_e);  // Assimilation efficiency (dimensionless), typical range 0.2-0.8
  Type m_P = exp(log_m_P);  // Phytoplankton mortality (day^-1), typical range 0.01-0.2
  Type m_Z = exp(log_m_Z);  // Zooplankton linear mortality (day^-1), typical range 0.01-0.2
  Type m_Z2 = exp(log_m_Z2);  // Zooplankton quadratic mortality (day^-1 (g C m^-3)^-1), typical range 0.1-2.0
  Type gamma = exp(log_gamma);  // Nutrient recycling efficiency (dimensionless), typical range 0.3-0.9
  Type delta = exp(log_delta);  // Zooplankton excretion rate (day^-1), typical range 0.05-0.3
  Type sigma_N = exp(log_sigma_N);  // Observation error for N (g C m^-3)
  Type sigma_P = exp(log_sigma_P);  // Observation error for P (g C m^-3)
  Type sigma_Z = exp(log_sigma_Z);  // Observation error for Z (g C m^-3)
  
  // Add small constant for numerical stability
  Type eps = Type(1e-8);  // Small constant to prevent division by zero
  
  // Minimum observation standard deviations to prevent numerical issues
  Type min_sigma = Type(0.001);  // Minimum SD (g C m^-3)
  sigma_N = sigma_N + min_sigma;  // Ensure sigma_N is not too small
  sigma_P = sigma_P + min_sigma;  // Ensure sigma_P is not too small
  sigma_Z = sigma_Z + min_sigma;  // Ensure sigma_Z is not too small
  
  // Soft bounds on efficiency parameters using penalties
  Type nll = Type(0.0);  // Initialize negative log-likelihood
  
  // Penalty to keep assimilation efficiency between 0 and 1
  if (e < Type(0.0)) nll += Type(100.0) * pow(e, 2);  // Quadratic penalty if e < 0
  if (e > Type(1.0)) nll += Type(100.0) * pow(e - Type(1.0), 2);  // Quadratic penalty if e > 1
  
  // Penalty to keep recycling efficiency between 0 and 1
  if (gamma < Type(0.0)) nll += Type(100.0) * pow(gamma, 2);  // Quadratic penalty if gamma < 0
  if (gamma > Type(1.0)) nll += Type(100.0) * pow(gamma - Type(1.0), 2);  // Quadratic penalty if gamma > 1
  
  // Get number of time points
  int n = Time.size();  // Number of observations
  
  // Initialize prediction vectors
  vector<Type> N_pred(n);  // Predicted nutrient concentration (g C m^-3)
  vector<Type> P_pred(n);  // Predicted phytoplankton concentration (g C m^-3)
  vector<Type> Z_pred(n);  // Predicted zooplankton concentration (g C m^-3)
  
  // Set initial conditions from first observation
  N_pred(0) = N_dat(0);  // Initial nutrient concentration from data (g C m^-3)
  P_pred(0) = P_dat(0);  // Initial phytoplankton concentration from data (g C m^-3)
  Z_pred(0) = Z_dat(0);  // Initial zooplankton concentration from data (g C m^-3)
  
  // Forward simulation using Euler integration
  for(int i = 1; i < n; i++) {
    Type dt = Time(i) - Time(i-1);  // Time step (days)
    
    // Get previous time step values (avoid data leakage)
    Type N_prev = N_pred(i-1);  // Nutrient at previous time (g C m^-3)
    Type P_prev = P_pred(i-1);  // Phytoplankton at previous time (g C m^-3)
    Type Z_prev = Z_pred(i-1);  // Zooplankton at previous time (g C m^-3)
    
    // Ensure non-negative concentrations
    N_prev = N_prev + eps;  // Add small constant for numerical stability
    P_prev = P_prev + eps;  // Add small constant for numerical stability
    Z_prev = Z_prev + eps;  // Add small constant for numerical stability
    
    // EQUATION 1: Nutrient uptake by phytoplankton (Michaelis-Menten)
    Type uptake = r * (N_prev / (K_N + N_prev)) * P_prev;  // Nutrient uptake rate (g C m^-3 day^-1)
    
    // EQUATION 2: Grazing by zooplankton (Holling Type II functional response)
    Type grazing = g * (P_prev / (K_P + P_prev)) * Z_prev;  // Grazing rate (g C m^-3 day^-1)
    
    // EQUATION 3: Phytoplankton mortality
    Type P_mortality = m_P * P_prev;  // Phytoplankton mortality rate (g C m^-3 day^-1)
    
    // EQUATION 4: Zooplankton mortality (linear + quadratic density dependence)
    Type Z_mortality = m_Z * Z_prev + m_Z2 * Z_prev * Z_prev;  // Zooplankton mortality rate (g C m^-3 day^-1)
    
    // EQUATION 5: Zooplankton excretion
    Type excretion = delta * Z_prev;  // Zooplankton excretion rate (g C m^-3 day^-1)
    
    // EQUATION 6: Nutrient recycling from phytoplankton mortality
    Type P_recycling = gamma * P_mortality;  // Nutrient recycling from dead phytoplankton (g C m^-3 day^-1)
    
    // EQUATION 7: Nutrient recycling from zooplankton mortality
    Type Z_recycling = gamma * Z_mortality;  // Nutrient recycling from dead zooplankton (g C m^-3 day^-1)
    
    // STATE EQUATIONS (differential equations integrated with Euler method)
    
    // EQUATION 8: dN/dt = -uptake + P_recycling + Z_recycling + excretion
    Type dN_dt = -uptake + P_recycling + Z_recycling + excretion;  // Rate of change of nutrients (g C m^-3 day^-1)
    
    // EQUATION 9: dP/dt = uptake - grazing - P_mortality
    Type dP_dt = uptake - grazing - P_mortality;  // Rate of change of phytoplankton (g C m^-3 day^-1)
    
    // EQUATION 10: dZ/dt = e * grazing - Z_mortality - excretion
    Type dZ_dt = e * grazing - Z_mortality - excretion;  // Rate of change of zooplankton (g C m^-3 day^-1)
    
    // Update predictions using Euler integration
    N_pred(i) = N_prev + dN_dt * dt;  // Nutrient concentration at current time (g C m^-3)
    P_pred(i) = P_prev + dP_dt * dt;  // Phytoplankton concentration at current time (g C m^-3)
    Z_pred(i) = Z_prev + dZ_dt * dt;  // Zooplankton concentration at current time (g C m^-3)
    
    // Ensure predictions remain non-negative (biological constraint)
    N_pred(i) = N_pred(i) < Type(0.0) ? Type(0.0) : N_pred(i);  // Prevent negative nutrients
    P_pred(i) = P_pred(i) < Type(0.0) ? Type(0.0) : P_pred(i);  // Prevent negative phytoplankton
    Z_pred(i) = Z_pred(i) < Type(0.0) ? Type(0.0) : Z_pred(i);  // Prevent negative zooplankton
  }
  
  // LIKELIHOOD CALCULATION
  // Use normal distribution for observation errors (could use lognormal if data spans many orders of magnitude)
  
  for(int i = 0; i < n; i++) {
    // Nutrient observations
    nll -= dnorm(N_dat(i), N_pred(i), sigma_N, true);  // Negative log-likelihood for nutrient observations
    
    // Phytoplankton observations
    nll -= dnorm(P_dat(i), P_pred(i), sigma_P, true);  // Negative log-likelihood for phytoplankton observations
    
    // Zooplankton observations
    nll -= dnorm(Z_dat(i), Z_pred(i), sigma_Z, true);  // Negative log-likelihood for zooplankton observations
  }
  
  // REPORTING
  REPORT(N_pred);  // Report predicted nutrient concentrations
  REPORT(P_pred);  // Report predicted phytoplankton concentrations
  REPORT(Z_pred);  // Report predicted zooplankton concentrations
  REPORT(r);  // Report maximum phytoplankton growth rate
  REPORT(K_N);  // Report half-saturation constant for nutrient uptake
  REPORT(g);  // Report maximum grazing rate
  REPORT(K_P);  // Report half-saturation constant for grazing
  REPORT(e);  // Report assimilation efficiency
  REPORT(m_P);  // Report phytoplankton mortality rate
  REPORT(m_Z);  // Report zooplankton linear mortality rate
  REPORT(m_Z2);  // Report zooplankton quadratic mortality rate
  REPORT(gamma);  // Report nutrient recycling efficiency
  REPORT(delta);  // Report zooplankton excretion rate
  REPORT(sigma_N);  // Report observation error SD for nutrients
  REPORT(sigma_P);  // Report observation error SD for phytoplankton
  REPORT(sigma_Z);  // Report observation error SD for zooplankton
  
  return nll;  // Return total negative log-likelihood
}
