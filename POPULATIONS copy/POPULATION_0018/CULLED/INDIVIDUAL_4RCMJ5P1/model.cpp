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
  PARAMETER(log_V_max);  // Log maximum phytoplankton nutrient uptake rate (day^-1)
  PARAMETER(log_K_N);  // Log half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(log_m_P);  // Log phytoplankton mortality rate (day^-1)
  PARAMETER(log_g_max);  // Log maximum zooplankton grazing rate (day^-1)
  PARAMETER(log_K_P);  // Log half-saturation constant for grazing (g C m^-3)
  PARAMETER(log_m_Z);  // Log zooplankton mortality rate (day^-1)
  PARAMETER(logit_epsilon);  // Logit zooplankton assimilation efficiency (dimensionless, 0-1)
  PARAMETER(logit_gamma);  // Logit nutrient recycling efficiency (dimensionless, 0-1)
  PARAMETER(log_I_0);  // Log surface light intensity (W m^-2)
  PARAMETER(log_k_w);  // Log background light attenuation coefficient (m^-1)
  PARAMETER(log_k_c);  // Log specific light attenuation for phytoplankton (m^2 (g C)^-1)
  PARAMETER(log_H);  // Log mixed layer depth (m)
  PARAMETER(log_I_k);  // Log light saturation parameter (W m^-2)
  PARAMETER(log_sigma_N);  // Log observation error SD for nutrients (g C m^-3)
  PARAMETER(log_sigma_P);  // Log observation error SD for phytoplankton (g C m^-3)
  PARAMETER(log_sigma_Z);  // Log observation error SD for zooplankton (g C m^-3)
  
  // Transform parameters to natural scale
  Type V_max = exp(log_V_max);  // Maximum phytoplankton nutrient uptake rate (day^-1), typically 0.5-2.0
  Type K_N = exp(log_K_N);  // Half-saturation for nutrient uptake (g C m^-3), typically 0.01-0.1
  Type m_P = exp(log_m_P);  // Phytoplankton mortality rate (day^-1), typically 0.01-0.1
  Type g_max = exp(log_g_max);  // Maximum zooplankton grazing rate (day^-1), typically 0.2-1.0
  Type K_P = exp(log_K_P);  // Half-saturation for grazing (g C m^-3), typically 0.05-0.3
  Type m_Z = exp(log_m_Z);  // Zooplankton mortality rate (day^-1), typically 0.01-0.2
  Type epsilon = Type(1.0) / (Type(1.0) + exp(-logit_epsilon));  // Assimilation efficiency (0-1), typically 0.3-0.7
  Type gamma = Type(1.0) / (Type(1.0) + exp(-logit_gamma));  // Recycling efficiency (0-1), typically 0.3-0.7
  Type I_0 = exp(log_I_0);  // Surface light intensity (W m^-2), typically 100-300
  Type k_w = exp(log_k_w);  // Background light attenuation (m^-1), typically 0.04-0.2
  Type k_c = exp(log_k_c);  // Phytoplankton self-shading (m^2 (g C)^-1), typically 0.02-0.1
  Type H = exp(log_H);  // Mixed layer depth (m), typically 20-100
  Type I_k = exp(log_I_k);  // Light saturation parameter (W m^-2), typically 30-100
  Type sigma_N = exp(log_sigma_N);  // Observation error SD for nutrients
  Type sigma_P = exp(log_sigma_P);  // Observation error SD for phytoplankton
  Type sigma_Z = exp(log_sigma_Z);  // Observation error SD for zooplankton
  
  // Add minimum SD to prevent numerical issues
  Type min_sigma = Type(1e-4);  // Minimum standard deviation to ensure numerical stability
  sigma_N = sigma_N + min_sigma;  // Ensure sigma_N is not too small
  sigma_P = sigma_P + min_sigma;  // Ensure sigma_P is not too small
  sigma_Z = sigma_Z + min_sigma;  // Ensure sigma_Z is not too small
  
  // Initialize prediction vectors
  int n_obs = Time.size();  // Number of observations
  vector<Type> N_pred(n_obs);  // Predicted nutrient concentrations
  vector<Type> P_pred(n_obs);  // Predicted phytoplankton concentrations
  vector<Type> Z_pred(n_obs);  // Predicted zooplankton concentrations
  
  // Set initial conditions from first observation
  N_pred(0) = N_dat(0);  // Initialize nutrients from data
  P_pred(0) = P_dat(0);  // Initialize phytoplankton from data
  Z_pred(0) = Z_dat(0);  // Initialize zooplankton from data
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);  // Small constant for numerical stability
  
  // Forward simulation using Euler integration
  for(int i = 1; i < n_obs; i++) {
    Type dt = Time(i) - Time(i-1);  // Time step size (days)
    
    // Get previous state (avoid using current observations)
    Type N_prev = N_pred(i-1);  // Nutrient concentration at previous time step
    Type P_prev = P_pred(i-1);  // Phytoplankton concentration at previous time step
    Type Z_prev = Z_pred(i-1);  // Zooplankton concentration at previous time step
    
    // Ensure non-negative concentrations using CppAD::CondExpGt (TMB-compatible)
    N_prev = CppAD::CondExpGt(N_prev, eps, N_prev, eps);  // Prevent negative nutrients
    P_prev = CppAD::CondExpGt(P_prev, eps, P_prev, eps);  // Prevent negative phytoplankton
    Z_prev = CppAD::CondExpGt(Z_prev, eps, Z_prev, eps);  // Prevent negative zooplankton
    
    // EQUATION 1: Calculate light limitation factor
    // Total light attenuation coefficient at previous time step
    Type k_total = k_w + k_c * P_prev;  // Total attenuation (m^-1) = background + phytoplankton self-shading
    
    // Depth-averaged light intensity using analytical solution of Beer-Lambert law
    // I_avg = (I_0 / (k_total * H)) * (1 - exp(-k_total * H))
    Type k_H = k_total * H;  // Dimensionless optical depth
    Type I_avg = I_0 * (Type(1.0) - exp(-k_H)) / (k_H + eps);  // Average light in mixed layer (W m^-2)
    
    // Light limitation factor using exponential saturation function
    // f_I = 1 - exp(-I_avg / I_k), ranges from 0 (no light) to 1 (saturating light)
    Type f_I = Type(1.0) - exp(-I_avg / (I_k + eps));  // Light limitation factor (dimensionless, 0-1)
    
    // EQUATION 2: Nutrient limitation factor (Monod/Michaelis-Menten kinetics)
    Type f_N = N_prev / (K_N + N_prev + eps);  // Nutrient limitation factor (dimensionless, 0-1)
    
    // EQUATION 3: Nutrient uptake by phytoplankton (limited by both nutrients AND light)
    Type uptake = V_max * f_N * f_I * P_prev;  // Nutrient uptake rate (g C m^-3 day^-1)
    
    // EQUATION 4: Grazing by zooplankton (Holling Type II functional response)
    Type grazing = g_max * (P_prev / (K_P + P_prev + eps)) * Z_prev;  // Grazing rate (g C m^-3 day^-1)
    
    // EQUATION 5: Phytoplankton mortality
    Type P_mortality = m_P * P_prev;  // Phytoplankton mortality rate (g C m^-3 day^-1)
    
    // EQUATION 6: Zooplankton mortality
    Type Z_mortality = m_Z * Z_prev;  // Zooplankton mortality rate (g C m^-3 day^-1)
    
    // EQUATION 7: Nutrient recycling from mortality and excretion
    Type recycling = gamma * (P_mortality + (Type(1.0) - epsilon) * grazing + Z_mortality);  // Nutrient recycling rate (g C m^-3 day^-1)
    
    // EQUATION 8: Nutrient dynamics (dN/dt)
    Type dN_dt = -uptake + recycling;  // Rate of change of nutrients
    
    // EQUATION 9: Phytoplankton dynamics (dP/dt)
    Type dP_dt = uptake - grazing - P_mortality;  // Rate of change of phytoplankton
    
    // EQUATION 10: Zooplankton dynamics (dZ/dt)
    Type dZ_dt = epsilon * grazing - Z_mortality;  // Rate of change of zooplankton
    
    // Update predictions using Euler method
    N_pred(i) = N_prev + dt * dN_dt;  // Update nutrient concentration
    P_pred(i) = P_prev + dt * dP_dt;  // Update phytoplankton concentration
    Z_pred(i) = Z_prev + dt * dZ_dt;  // Update zooplankton concentration
    
    // Ensure non-negative predictions using CppAD::CondExpGt (TMB-compatible)
    N_pred(i) = CppAD::CondExpGt(N_pred(i), eps, N_pred(i), eps);  // Prevent negative predicted nutrients
    P_pred(i) = CppAD::CondExpGt(P_pred(i), eps, P_pred(i), eps);  // Prevent negative predicted phytoplankton
    Z_pred(i) = CppAD::CondExpGt(Z_pred(i), eps, Z_pred(i), eps);  // Prevent negative predicted zooplankton
  }
  
  // Initialize negative log-likelihood
  Type nll = Type(0.0);  // Negative log-likelihood accumulator
  
  // Calculate likelihood for all observations using normal distribution
  for(int i = 0; i < n_obs; i++) {
    // EQUATION 11: Likelihood for nutrient observations
    nll -= dnorm(N_dat(i), N_pred(i), sigma_N, true);  // Add nutrient observation likelihood
    
    // EQUATION 12: Likelihood for phytoplankton observations
    nll -= dnorm(P_dat(i), P_pred(i), sigma_P, true);  // Add phytoplankton observation likelihood
    
    // EQUATION 13: Likelihood for zooplankton observations
    nll -= dnorm(Z_dat(i), Z_pred(i), sigma_Z, true);  // Add zooplankton observation likelihood
  }
  
  // Soft parameter bounds using penalties to maintain biological realism
  // EQUATION 14: Penalty for extremely high uptake rates (V_max > 5.0 day^-1 is unrealistic)
  Type V_max_penalty = CppAD::CondExpGt(V_max, Type(5.0), V_max - Type(5.0), Type(0.0));  // Calculate penalty amount
  nll += Type(10.0) * pow(V_max_penalty, 2);  // Penalize V_max > 5.0
  
  // EQUATION 15: Penalty for extremely high grazing rates (g_max > 3.0 day^-1 is unrealistic)
  Type g_max_penalty = CppAD::CondExpGt(g_max, Type(3.0), g_max - Type(3.0), Type(0.0));  // Calculate penalty amount
  nll += Type(10.0) * pow(g_max_penalty, 2);  // Penalize g_max > 3.0
  
  // EQUATION 16: Penalty for extremely high mortality rates (m_P or m_Z > 1.0 day^-1 is unrealistic)
  Type m_P_penalty = CppAD::CondExpGt(m_P, Type(1.0), m_P - Type(1.0), Type(0.0));  // Calculate m_P penalty amount
  Type m_Z_penalty = CppAD::CondExpGt(m_Z, Type(1.0), m_Z - Type(1.0), Type(0.0));  // Calculate m_Z penalty amount
  nll += Type(10.0) * pow(m_P_penalty, 2);  // Penalize m_P > 1.0
  nll += Type(10.0) * pow(m_Z_penalty, 2);  // Penalize m_Z > 1.0
  
  // Report predictions and parameters
  REPORT(N_pred);  // Report predicted nutrient concentrations
  REPORT(P_pred);  // Report predicted phytoplankton concentrations
  REPORT(Z_pred);  // Report predicted zooplankton concentrations
  REPORT(V_max);  // Report maximum uptake rate
  REPORT(K_N);  // Report nutrient half-saturation constant
  REPORT(m_P);  // Report phytoplankton mortality rate
  REPORT(g_max);  // Report maximum grazing rate
  REPORT(K_P);  // Report grazing half-saturation constant
  REPORT(m_Z);  // Report zooplankton mortality rate
  REPORT(epsilon);  // Report assimilation efficiency
  REPORT(gamma);  // Report recycling efficiency
  REPORT(I_0);  // Report surface light intensity
  REPORT(k_w);  // Report background light attenuation
  REPORT(k_c);  // Report phytoplankton self-shading coefficient
  REPORT(H);  // Report mixed layer depth
  REPORT(I_k);  // Report light saturation parameter
  REPORT(sigma_N);  // Report nutrient observation error
  REPORT(sigma_P);  // Report phytoplankton observation error
  REPORT(sigma_Z);  // Report zooplankton observation error
  
  return nll;  // Return negative log-likelihood for minimization
}
