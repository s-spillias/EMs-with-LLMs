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
  PARAMETER(log_g_max);  // Log maximum grazing rate by zooplankton (day^-1)
  PARAMETER(log_K_P);  // Log half-saturation constant for grazing (g C m^-3)
  PARAMETER(log_m_P);  // Log phytoplankton mortality rate (day^-1)
  PARAMETER(log_m_Z);  // Log zooplankton quadratic mortality rate (m^3 g C^-1 day^-1)
  PARAMETER(logit_alpha);  // Logit assimilation efficiency of zooplankton (dimensionless, 0-1)
  PARAMETER(logit_gamma);  // Logit nutrient recycling efficiency (dimensionless, 0-1)
  PARAMETER(log_I_0);  // Log surface light intensity (μmol photons m^-2 s^-1)
  PARAMETER(log_k_w);  // Log background water light attenuation coefficient (m^-1)
  PARAMETER(log_k_c);  // Log phytoplankton-specific light attenuation coefficient (m^2 g C^-1)
  PARAMETER(log_K_I);  // Log half-saturation constant for light (μmol photons m^-2 s^-1)
  PARAMETER(log_MLD);  // Log mixed layer depth (m)
  PARAMETER(log_sigma_N);  // Log observation error SD for nutrients (g C m^-3)
  PARAMETER(log_sigma_P);  // Log observation error SD for phytoplankton (g C m^-3)
  PARAMETER(log_sigma_Z);  // Log observation error SD for zooplankton (g C m^-3)
  
  // Transform parameters to natural scale with biological bounds
  Type V_max = exp(log_V_max);  // Maximum nutrient uptake rate (day^-1), literature values: 0.5-2.0
  Type K_N = exp(log_K_N);  // Half-saturation for nutrients (g C m^-3), literature values: 0.01-0.1
  Type g_max = exp(log_g_max);  // Maximum grazing rate (day^-1), literature values: 0.2-1.0
  Type K_P = exp(log_K_P);  // Half-saturation for grazing (g C m^-3), literature values: 0.05-0.3
  Type m_P = exp(log_m_P);  // Phytoplankton mortality (day^-1), literature values: 0.01-0.1
  Type m_Z = exp(log_m_Z);  // Zooplankton mortality (m^3 g C^-1 day^-1), literature values: 0.01-0.5
  Type alpha = Type(1.0) / (Type(1.0) + exp(-logit_alpha));  // Assimilation efficiency (0-1), literature values: 0.2-0.4
  Type gamma = Type(1.0) / (Type(1.0) + exp(-logit_gamma));  // Recycling efficiency (0-1), literature values: 0.3-0.7
  Type I_0 = exp(log_I_0);  // Surface light intensity (μmol photons m^-2 s^-1), literature values: 100-2000
  Type k_w = exp(log_k_w);  // Background water attenuation (m^-1), literature values: 0.02-0.2
  Type k_c = exp(log_k_c);  // Phytoplankton attenuation (m^2 g C^-1), literature values: 0.01-0.1
  Type K_I = exp(log_K_I);  // Half-saturation for light (μmol photons m^-2 s^-1), literature values: 20-100
  Type MLD = exp(log_MLD);  // Mixed layer depth (m), literature values: 10-200
  Type sigma_N = exp(log_sigma_N);  // Observation error SD for N (g C m^-3)
  Type sigma_P = exp(log_sigma_P);  // Observation error SD for P (g C m^-3)
  Type sigma_Z = exp(log_sigma_Z);  // Observation error SD for Z (g C m^-3)
  
  // Add small constants for numerical stability
  Type eps = Type(1e-8);  // Small constant to prevent division by zero
  Type min_sigma = Type(0.001);  // Minimum observation error to prevent numerical issues
  
  // Apply minimum sigma values for numerical stability
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
  
  // Forward simulation using Euler integration
  for(int i = 1; i < n_obs; i++) {
    Type dt = Time(i) - Time(i-1);  // Time step (days)
    
    // Get previous state values (avoid data leakage)
    Type N_prev = N_pred(i-1);  // Nutrient concentration at previous time step
    Type P_prev = P_pred(i-1);  // Phytoplankton concentration at previous time step
    Type Z_prev = Z_pred(i-1);  // Zooplankton concentration at previous time step
    
    // Ensure non-negative concentrations with smooth lower bound
    N_prev = N_prev + eps;  // Add small constant for numerical stability
    P_prev = P_prev + eps;  // Add small constant for numerical stability
    Z_prev = Z_prev + eps;  // Add small constant for numerical stability
    
    // EQUATION 1: Light attenuation and limitation
    // Calculate total attenuation coefficient (background + phytoplankton self-shading)
    Type k_total = k_w + k_c * P_prev;  // Total light attenuation coefficient (m^-1)
    
    // Calculate depth-integrated average light in mixed layer using Beer-Lambert law
    // I_avg = I_0 * (1 - exp(-k_total * MLD)) / (k_total * MLD)
    Type light_extinction = exp(-k_total * MLD);  // Light extinction factor
    Type I_avg = I_0 * (Type(1.0) - light_extinction) / (k_total * MLD + eps);  // Average light in mixed layer (μmol photons m^-2 s^-1)
    
    // Calculate light limitation factor (Monod-type response)
    Type f_light = I_avg / (K_I + I_avg + eps);  // Light limitation factor (0-1)
    
    // EQUATION 2: Nutrient limitation by phytoplankton (Michaelis-Menten)
    Type f_nutrient = N_prev / (K_N + N_prev + eps);  // Nutrient limitation factor (0-1)
    
    // EQUATION 3: Nutrient uptake by phytoplankton (co-limited by light and nutrients)
    Type uptake = V_max * f_nutrient * f_light * P_prev;  // Nutrient uptake rate (g C m^-3 day^-1)
    
    // EQUATION 4: Zooplankton grazing on phytoplankton (Holling Type II)
    Type grazing = g_max * (P_prev / (K_P + P_prev + eps)) * Z_prev;  // Grazing rate (g C m^-3 day^-1)
    
    // EQUATION 5: Phytoplankton mortality
    Type P_mortality = m_P * P_prev;  // Linear phytoplankton mortality (g C m^-3 day^-1)
    
    // EQUATION 6: Zooplankton mortality (quadratic/density-dependent)
    Type Z_mortality = m_Z * Z_prev * Z_prev;  // Quadratic zooplankton mortality (g C m^-3 day^-1)
    
    // EQUATION 7: Nutrient recycling from mortality and inefficient grazing
    Type recycling = gamma * (P_mortality + Z_mortality + (Type(1.0) - alpha) * grazing);  // Nutrient recycling (g C m^-3 day^-1)
    
    // EQUATION 8: Nutrient dynamics (dN/dt)
    Type dN_dt = -uptake + recycling;  // Rate of change of nutrients (g C m^-3 day^-1)
    
    // EQUATION 9: Phytoplankton dynamics (dP/dt)
    Type dP_dt = uptake - grazing - P_mortality;  // Rate of change of phytoplankton (g C m^-3 day^-1)
    
    // EQUATION 10: Zooplankton dynamics (dZ/dt)
    Type dZ_dt = alpha * grazing - Z_mortality;  // Rate of change of zooplankton (g C m^-3 day^-1)
    
    // Update predictions using Euler method
    N_pred(i) = N_prev + dN_dt * dt;  // Update nutrient concentration
    P_pred(i) = P_prev + dP_dt * dt;  // Update phytoplankton concentration
    Z_pred(i) = Z_prev + dZ_dt * dt;  // Update zooplankton concentration
    
    // Apply soft lower bounds to keep concentrations positive
    N_pred(i) = N_pred(i) + eps;  // Ensure N stays positive
    P_pred(i) = P_pred(i) + eps;  // Ensure P stays positive
    Z_pred(i) = Z_pred(i) + eps;  // Ensure Z stays positive
  }
  
  // LIKELIHOOD CALCULATION
  Type nll = Type(0.0);  // Initialize negative log-likelihood
  
  // Add observation likelihoods for all time points
  for(int i = 0; i < n_obs; i++) {
    // Nutrient observations (normal distribution)
    nll -= dnorm(N_dat(i), N_pred(i), sigma_N, true);  // Negative log-likelihood for nutrient observations
    
    // Phytoplankton observations (normal distribution)
    nll -= dnorm(P_dat(i), P_pred(i), sigma_P, true);  // Negative log-likelihood for phytoplankton observations
    
    // Zooplankton observations (normal distribution)
    nll -= dnorm(Z_dat(i), Z_pred(i), sigma_Z, true);  // Negative log-likelihood for zooplankton observations
  }
  
  // Soft parameter bounds using penalties (to keep parameters in biologically reasonable ranges)
  // Penalty for V_max if too high (> 5.0 day^-1) - using smooth penalty
  Type V_max_excess = V_max - Type(5.0);  // Calculate excess above threshold
  nll += Type(10.0) * V_max_excess * V_max_excess * CppAD::CondExpGt(V_max, Type(5.0), Type(1.0), Type(0.0));  // Soft upper bound penalty for V_max
  
  // Penalty for g_max if too high (> 3.0 day^-1) - using smooth penalty
  Type g_max_excess = g_max - Type(3.0);  // Calculate excess above threshold
  nll += Type(10.0) * g_max_excess * g_max_excess * CppAD::CondExpGt(g_max, Type(3.0), Type(1.0), Type(0.0));  // Soft upper bound penalty for g_max
  
  // Penalty for m_P if too high (> 0.5 day^-1) - using smooth penalty
  Type m_P_excess = m_P - Type(0.5);  // Calculate excess above threshold
  nll += Type(10.0) * m_P_excess * m_P_excess * CppAD::CondExpGt(m_P, Type(0.5), Type(1.0), Type(0.0));  // Soft upper bound penalty for m_P
  
  // Penalty for m_Z if too high (> 2.0 m^3 g C^-1 day^-1) - using smooth penalty
  Type m_Z_excess = m_Z - Type(2.0);  // Calculate excess above threshold
  nll += Type(10.0) * m_Z_excess * m_Z_excess * CppAD::CondExpGt(m_Z, Type(2.0), Type(1.0), Type(0.0));  // Soft upper bound penalty for m_Z
  
  // Penalty for I_0 if too high (> 2000 μmol photons m^-2 s^-1) - using smooth penalty
  Type I_0_excess = I_0 - Type(2000.0);  // Calculate excess above threshold
  nll += Type(10.0) * I_0_excess * I_0_excess * CppAD::CondExpGt(I_0, Type(2000.0), Type(1.0), Type(0.0));  // Soft upper bound penalty for I_0
  
  // Penalty for k_w if too high (> 0.2 m^-1) - using smooth penalty
  Type k_w_excess = k_w - Type(0.2);  // Calculate excess above threshold
  nll += Type(10.0) * k_w_excess * k_w_excess * CppAD::CondExpGt(k_w, Type(0.2), Type(1.0), Type(0.0));  // Soft upper bound penalty for k_w
  
  // Penalty for k_c if too high (> 0.1 m^2 g C^-1) - using smooth penalty
  Type k_c_excess = k_c - Type(0.1);  // Calculate excess above threshold
  nll += Type(10.0) * k_c_excess * k_c_excess * CppAD::CondExpGt(k_c, Type(0.1), Type(1.0), Type(0.0));  // Soft upper bound penalty for k_c
  
  // Penalty for MLD if too high (> 200 m) - using smooth penalty
  Type MLD_excess = MLD - Type(200.0);  // Calculate excess above threshold
  nll += Type(10.0) * MLD_excess * MLD_excess * CppAD::CondExpGt(MLD, Type(200.0), Type(1.0), Type(0.0));  // Soft upper bound penalty for MLD
  
  // REPORTING
  REPORT(N_pred);  // Report predicted nutrient concentrations
  REPORT(P_pred);  // Report predicted phytoplankton concentrations
  REPORT(Z_pred);  // Report predicted zooplankton concentrations
  REPORT(V_max);  // Report maximum nutrient uptake rate
  REPORT(K_N);  // Report half-saturation constant for nutrients
  REPORT(g_max);  // Report maximum grazing rate
  REPORT(K_P);  // Report half-saturation constant for grazing
  REPORT(m_P);  // Report phytoplankton mortality rate
  REPORT(m_Z);  // Report zooplankton mortality rate
  REPORT(alpha);  // Report assimilation efficiency
  REPORT(gamma);  // Report recycling efficiency
  REPORT(I_0);  // Report surface light intensity
  REPORT(k_w);  // Report background water attenuation
  REPORT(k_c);  // Report phytoplankton attenuation
  REPORT(K_I);  // Report half-saturation constant for light
  REPORT(MLD);  // Report mixed layer depth
  REPORT(sigma_N);  // Report observation error for nutrients
  REPORT(sigma_P);  // Report observation error for phytoplankton
  REPORT(sigma_Z);  // Report observation error for zooplankton
  
  return nll;  // Return total negative log-likelihood
}
