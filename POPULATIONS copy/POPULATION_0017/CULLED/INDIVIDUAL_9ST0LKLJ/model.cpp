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
  PARAMETER(log_V_max);  // Log maximum nutrient uptake rate by phytoplankton (day^-1)
  PARAMETER(log_K_N);  // Log half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(log_I);  // Log effective light intensity in mixed layer (μmol photons m^-2 s^-1)
  PARAMETER(log_K_I);  // Log half-saturation constant for light limitation (μmol photons m^-2 s^-1)
  PARAMETER(log_mu_P);  // Log phytoplankton natural mortality rate (day^-1)
  PARAMETER(log_g_max);  // Log maximum zooplankton grazing rate (day^-1)
  PARAMETER(log_K_P);  // Log half-saturation constant for grazing (g C m^-3)
  PARAMETER(log_mu_Z);  // Log zooplankton linear mortality rate (day^-1)
  PARAMETER(log_mu_Z2);  // Log zooplankton quadratic mortality rate (day^-1 (g C m^-3)^-1)
  PARAMETER(logit_epsilon);  // Logit zooplankton assimilation efficiency (dimensionless, 0-1)
  PARAMETER(logit_gamma_P);  // Logit nutrient recycling efficiency from phytoplankton mortality (dimensionless, 0-1)
  PARAMETER(logit_gamma_Z);  // Logit nutrient recycling efficiency from zooplankton losses (dimensionless, 0-1)
  PARAMETER(log_sigma_N);  // Log observation error standard deviation for nutrients (g C m^-3)
  PARAMETER(log_sigma_P);  // Log observation error standard deviation for phytoplankton (g C m^-3)
  PARAMETER(log_sigma_Z);  // Log observation error standard deviation for zooplankton (g C m^-3)
  
  // TRANSFORM PARAMETERS to natural scale with biological constraints
  Type V_max = exp(log_V_max);  // Maximum nutrient uptake rate (day^-1), must be positive
  Type K_N = exp(log_K_N);  // Half-saturation for nutrients (g C m^-3), must be positive
  Type I = exp(log_I);  // Effective light intensity (μmol photons m^-2 s^-1), must be positive
  Type K_I = exp(log_K_I);  // Half-saturation for light (μmol photons m^-2 s^-1), must be positive
  Type mu_P = exp(log_mu_P);  // Phytoplankton mortality (day^-1), must be positive
  Type g_max = exp(log_g_max);  // Maximum grazing rate (day^-1), must be positive
  Type K_P = exp(log_K_P);  // Half-saturation for grazing (g C m^-3), must be positive
  Type mu_Z = exp(log_mu_Z);  // Zooplankton linear mortality (day^-1), must be positive
  Type mu_Z2 = exp(log_mu_Z2);  // Zooplankton quadratic mortality (day^-1 (g C m^-3)^-1), must be positive
  Type epsilon = Type(1.0) / (Type(1.0) + exp(-logit_epsilon));  // Assimilation efficiency (0-1), bounded using logistic transform
  Type gamma_P = Type(1.0) / (Type(1.0) + exp(-logit_gamma_P));  // Phytoplankton recycling efficiency (0-1), bounded using logistic transform
  Type gamma_Z = Type(1.0) / (Type(1.0) + exp(-logit_gamma_Z));  // Zooplankton recycling efficiency (0-1), bounded using logistic transform
  Type sigma_N = exp(log_sigma_N);  // Observation error SD for N (g C m^-3), must be positive
  Type sigma_P = exp(log_sigma_P);  // Observation error SD for P (g C m^-3), must be positive
  Type sigma_Z = exp(log_sigma_Z);  // Observation error SD for Z (g C m^-3), must be positive
  
  // Add minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(1e-4);  // Minimum observation error (g C m^-3)
  sigma_N = sigma_N + min_sigma;  // Ensure sigma_N is not too small
  sigma_P = sigma_P + min_sigma;  // Ensure sigma_P is not too small
  sigma_Z = sigma_Z + min_sigma;  // Ensure sigma_Z is not too small
  
  // INITIALIZE PREDICTION VECTORS
  int n_obs = Time.size();  // Number of time points
  vector<Type> N_pred(n_obs);  // Predicted nutrient concentration (g C m^-3)
  vector<Type> P_pred(n_obs);  // Predicted phytoplankton concentration (g C m^-3)
  vector<Type> Z_pred(n_obs);  // Predicted zooplankton concentration (g C m^-3)
  
  // Set initial conditions from first observation
  N_pred(0) = N_dat(0);  // Initial nutrient concentration from data (g C m^-3)
  P_pred(0) = P_dat(0);  // Initial phytoplankton concentration from data (g C m^-3)
  Z_pred(0) = Z_dat(0);  // Initial zooplankton concentration from data (g C m^-3)
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);  // Small constant for numerical stability
  
  // SIMULATE DYNAMICS using Euler integration
  for(int i = 1; i < n_obs; i++) {
    Type dt = Time(i) - Time(i-1);  // Time step (days)
    
    // Get previous state (avoid data leakage by using only previous predictions)
    Type N_prev = N_pred(i-1);  // Nutrient at previous time (g C m^-3)
    Type P_prev = P_pred(i-1);  // Phytoplankton at previous time (g C m^-3)
    Type Z_prev = Z_pred(i-1);  // Zooplankton at previous time (g C m^-3)
    
    // Ensure non-negative concentrations with smooth lower bound
    N_prev = N_prev + eps;  // Prevent negative nutrients
    P_prev = P_prev + eps;  // Prevent negative phytoplankton
    Z_prev = Z_prev + eps;  // Prevent negative zooplankton
    
    // EQUATION 1: Nutrient uptake by phytoplankton with nutrient AND light co-limitation
    // Multiplicative co-limitation: both nutrient and light must be available
    Type nutrient_limitation = N_prev / (K_N + N_prev + eps);  // Michaelis-Menten nutrient limitation term (dimensionless, 0-1)
    Type light_limitation = I / (K_I + I + eps);  // Michaelis-Menten light limitation term (dimensionless, 0-1)
    Type uptake = V_max * nutrient_limitation * light_limitation * P_prev;  // Nutrient uptake rate with co-limitation (g C m^-3 day^-1)
    
    // EQUATION 2: Zooplankton grazing on phytoplankton (Holling Type II functional response)
    Type grazing = g_max * (P_prev / (K_P + P_prev + eps)) * Z_prev;  // Grazing rate (g C m^-3 day^-1)
    
    // EQUATION 3: Phytoplankton mortality
    Type P_mortality = mu_P * P_prev;  // Phytoplankton mortality rate (g C m^-3 day^-1)
    
    // EQUATION 4: Zooplankton mortality (linear + quadratic density dependence)
    Type Z_mortality = mu_Z * Z_prev + mu_Z2 * Z_prev * Z_prev;  // Zooplankton mortality rate (g C m^-3 day^-1)
    
    // EQUATION 5: Nutrient recycling from phytoplankton and zooplankton losses
    Type N_recycling_P = gamma_P * P_mortality;  // Nutrient recycling from phytoplankton mortality (g C m^-3 day^-1)
    Type N_recycling_Z = gamma_Z * Z_mortality;  // Nutrient recycling from zooplankton mortality (g C m^-3 day^-1)
    Type N_recycling_grazing = gamma_Z * (Type(1.0) - epsilon) * grazing;  // Nutrient recycling from inefficient grazing (g C m^-3 day^-1)
    
    // EQUATION 6: Rate of change for Nutrients (dN/dt)
    Type dN_dt = -uptake + N_recycling_P + N_recycling_Z + N_recycling_grazing;  // Net nutrient change (g C m^-3 day^-1)
    
    // EQUATION 7: Rate of change for Phytoplankton (dP/dt)
    Type dP_dt = uptake - grazing - P_mortality;  // Net phytoplankton change (g C m^-3 day^-1)
    
    // EQUATION 8: Rate of change for Zooplankton (dZ/dt)
    Type dZ_dt = epsilon * grazing - Z_mortality;  // Net zooplankton change (g C m^-3 day^-1)
    
    // Update predictions using Euler method
    N_pred(i) = N_prev + dN_dt * dt;  // Nutrient at current time (g C m^-3)
    P_pred(i) = P_prev + dP_dt * dt;  // Phytoplankton at current time (g C m^-3)
    Z_pred(i) = Z_prev + dZ_dt * dt;  // Zooplankton at current time (g C m^-3)
    
    // Apply smooth lower bounds to prevent negative values
    N_pred(i) = N_pred(i) + eps;  // Ensure non-negative nutrients
    P_pred(i) = P_pred(i) + eps;  // Ensure non-negative phytoplankton
    Z_pred(i) = Z_pred(i) + eps;  // Ensure non-negative zooplankton
  }
  
  // CALCULATE NEGATIVE LOG-LIKELIHOOD
  Type nll = Type(0.0);  // Initialize negative log-likelihood
  
  // Likelihood for all nutrient observations (normal distribution)
  for(int i = 0; i < n_obs; i++) {
    nll -= dnorm(N_dat(i), N_pred(i), sigma_N, true);  // Add nutrient observation likelihood
  }
  
  // Likelihood for all phytoplankton observations (normal distribution)
  for(int i = 0; i < n_obs; i++) {
    nll -= dnorm(P_dat(i), P_pred(i), sigma_P, true);  // Add phytoplankton observation likelihood
  }
  
  // Likelihood for all zooplankton observations (normal distribution)
  for(int i = 0; i < n_obs; i++) {
    nll -= dnorm(Z_dat(i), Z_pred(i), sigma_Z, true);  // Add zooplankton observation likelihood
  }
  
  // REPORT PREDICTIONS AND PARAMETERS
  REPORT(N_pred);  // Report predicted nutrient concentrations
  REPORT(P_pred);  // Report predicted phytoplankton concentrations
  REPORT(Z_pred);  // Report predicted zooplankton concentrations
  REPORT(V_max);  // Report maximum uptake rate
  REPORT(K_N);  // Report nutrient half-saturation
  REPORT(I);  // Report effective light intensity
  REPORT(K_I);  // Report light half-saturation
  REPORT(mu_P);  // Report phytoplankton mortality
  REPORT(g_max);  // Report maximum grazing rate
  REPORT(K_P);  // Report grazing half-saturation
  REPORT(mu_Z);  // Report zooplankton linear mortality
  REPORT(mu_Z2);  // Report zooplankton quadratic mortality
  REPORT(epsilon);  // Report assimilation efficiency
  REPORT(gamma_P);  // Report phytoplankton recycling efficiency
  REPORT(gamma_Z);  // Report zooplankton recycling efficiency
  REPORT(sigma_N);  // Report nutrient observation error
  REPORT(sigma_P);  // Report phytoplankton observation error
  REPORT(sigma_Z);  // Report zooplankton observation error
  
  return nll;  // Return total negative log-likelihood
}
