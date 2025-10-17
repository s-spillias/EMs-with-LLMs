#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs
  DATA_VECTOR(Time);  // Time in days
  DATA_VECTOR(N_dat);  // Nutrient concentration observations (g C m^-3)
  DATA_VECTOR(P_dat);  // Phytoplankton concentration observations (g C m^-3)
  DATA_VECTOR(Z_dat);  // Zooplankton concentration observations (g C m^-3)
  
  // Parameters for phytoplankton dynamics
  PARAMETER(r);  // Maximum phytoplankton growth rate (day^-1)
  PARAMETER(K_N);  // Half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(m_P);  // Phytoplankton mortality rate (day^-1)
  
  // Parameters for zooplankton dynamics
  PARAMETER(g_max);  // Maximum zooplankton grazing rate (day^-1)
  PARAMETER(K_P);  // Half-saturation constant for grazing (g C m^-3)
  PARAMETER(e);  // Zooplankton assimilation efficiency (dimensionless, 0-1)
  PARAMETER(m_Z);  // Zooplankton mortality rate (day^-1)
  
  // Nutrient recycling parameters
  PARAMETER(gamma);  // Fraction of phytoplankton mortality recycled to nutrients (dimensionless, 0-1)
  PARAMETER(delta);  // Fraction of zooplankton mortality recycled to nutrients (dimensionless, 0-1)
  
  // Light limitation parameters
  PARAMETER(k_w);  // Background light attenuation by water (m^-1)
  PARAMETER(k_c);  // Specific light attenuation by phytoplankton (m^2 (g C)^-1)
  PARAMETER(z_mix);  // Mixed layer depth (m)
  
  // Observation error parameters
  PARAMETER(log_sigma_N);  // Log-scale standard deviation for nutrient observations
  PARAMETER(log_sigma_P);  // Log-scale standard deviation for phytoplankton observations
  PARAMETER(log_sigma_Z);  // Log-scale standard deviation for zooplankton observations
  
  // Transform log-scale parameters to natural scale
  Type sigma_N = exp(log_sigma_N);  // Standard deviation for nutrient observation error
  Type sigma_P = exp(log_sigma_P);  // Standard deviation for phytoplankton observation error
  Type sigma_Z = exp(log_sigma_Z);  // Standard deviation for zooplankton observation error
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(1e-4);  // Minimum observation error to ensure numerical stability
  sigma_N = sigma_N + min_sigma;  // Add minimum to nutrient error
  sigma_P = sigma_P + min_sigma;  // Add minimum to phytoplankton error
  sigma_Z = sigma_Z + min_sigma;  // Add minimum to zooplankton error
  
  // Initialize prediction vectors
  int n = Time.size();  // Number of time points in the dataset
  vector<Type> N_pred(n);  // Predicted nutrient concentrations
  vector<Type> P_pred(n);  // Predicted phytoplankton concentrations
  vector<Type> Z_pred(n);  // Predicted zooplankton concentrations
  
  // Set initial conditions from first observation
  N_pred(0) = N_dat(0);  // Initialize nutrient from first data point
  P_pred(0) = P_dat(0);  // Initialize phytoplankton from first data point
  Z_pred(0) = Z_dat(0);  // Initialize zooplankton from first data point
  
  // Small constant to prevent division by zero
  Type epsilon = Type(1e-8);  // Small value added to denominators for numerical stability
  
  // Soft constraints on parameters using smooth penalties
  Type nll = Type(0.0);  // Initialize negative log-likelihood
  
  // Soft bounds for biological realism (using quadratic penalties outside reasonable ranges)
  Type penalty_weight = Type(10.0);  // Weight for soft constraint penalties
  
  // Growth rate should be positive and reasonable (0.1 to 3.0 day^-1)
  if(r < Type(0.1)) nll += penalty_weight * pow(Type(0.1) - r, 2);  // Penalize if growth rate too low
  if(r > Type(3.0)) nll += penalty_weight * pow(r - Type(3.0), 2);  // Penalize if growth rate too high
  
  // Half-saturation constants should be positive and reasonable (0.001 to 1.0 g C m^-3)
  if(K_N < Type(0.001)) nll += penalty_weight * pow(Type(0.001) - K_N, 2);  // Penalize if K_N too low
  if(K_N > Type(1.0)) nll += penalty_weight * pow(K_N - Type(1.0), 2);  // Penalize if K_N too high
  if(K_P < Type(0.001)) nll += penalty_weight * pow(Type(0.001) - K_P, 2);  // Penalize if K_P too low
  if(K_P > Type(1.0)) nll += penalty_weight * pow(K_P - Type(1.0), 2);  // Penalize if K_P too high
  
  // Mortality rates should be positive and reasonable (0.01 to 1.0 day^-1)
  if(m_P < Type(0.01)) nll += penalty_weight * pow(Type(0.01) - m_P, 2);  // Penalize if phytoplankton mortality too low
  if(m_P > Type(1.0)) nll += penalty_weight * pow(m_P - Type(1.0), 2);  // Penalize if phytoplankton mortality too high
  if(m_Z < Type(0.01)) nll += penalty_weight * pow(Type(0.01) - m_Z, 2);  // Penalize if zooplankton mortality too low
  if(m_Z > Type(1.0)) nll += penalty_weight * pow(m_Z - Type(1.0), 2);  // Penalize if zooplankton mortality too high
  
  // Grazing rate should be positive and reasonable (0.1 to 2.0 day^-1)
  if(g_max < Type(0.1)) nll += penalty_weight * pow(Type(0.1) - g_max, 2);  // Penalize if grazing rate too low
  if(g_max > Type(2.0)) nll += penalty_weight * pow(g_max - Type(2.0), 2);  // Penalize if grazing rate too high
  
  // Efficiency and recycling fractions should be between 0 and 1
  if(e < Type(0.0)) nll += penalty_weight * pow(e, 2);  // Penalize negative assimilation efficiency
  if(e > Type(1.0)) nll += penalty_weight * pow(e - Type(1.0), 2);  // Penalize assimilation efficiency > 1
  if(gamma < Type(0.0)) nll += penalty_weight * pow(gamma, 2);  // Penalize negative recycling fraction
  if(gamma > Type(1.0)) nll += penalty_weight * pow(gamma - Type(1.0), 2);  // Penalize recycling fraction > 1
  if(delta < Type(0.0)) nll += penalty_weight * pow(delta, 2);  // Penalize negative recycling fraction
  if(delta > Type(1.0)) nll += penalty_weight * pow(delta - Type(1.0), 2);  // Penalize recycling fraction > 1
  
  // Light attenuation parameters should be positive and reasonable
  if(k_w < Type(0.02)) nll += penalty_weight * pow(Type(0.02) - k_w, 2);  // Penalize if background attenuation too low
  if(k_w > Type(0.2)) nll += penalty_weight * pow(k_w - Type(0.2), 2);  // Penalize if background attenuation too high
  if(k_c < Type(0.01)) nll += penalty_weight * pow(Type(0.01) - k_c, 2);  // Penalize if phytoplankton attenuation too low
  if(k_c > Type(0.1)) nll += penalty_weight * pow(k_c - Type(0.1), 2);  // Penalize if phytoplankton attenuation too high
  
  // Mixed layer depth should be positive and reasonable (10 to 100 m)
  if(z_mix < Type(10.0)) nll += penalty_weight * pow(Type(10.0) - z_mix, 2);  // Penalize if mixed layer too shallow
  if(z_mix > Type(100.0)) nll += penalty_weight * pow(z_mix - Type(100.0), 2);  // Penalize if mixed layer too deep
  
  // Forward simulation using Euler integration
  for(int i = 1; i < n; i++) {
    Type dt = Time(i) - Time(i-1);  // Time step size (days)
    
    // Get previous time step values (avoid data leakage)
    Type N_prev = N_pred(i-1);  // Nutrient concentration at previous time step
    Type P_prev = P_pred(i-1);  // Phytoplankton concentration at previous time step
    Type Z_prev = Z_pred(i-1);  // Zooplankton concentration at previous time step
    
    // Ensure non-negative concentrations with smooth transition
    N_prev = N_prev + epsilon;  // Add small constant to prevent negative values
    P_prev = P_prev + epsilon;  // Add small constant to prevent negative values
    Z_prev = Z_prev + epsilon;  // Add small constant to prevent negative values
    
    // Calculate light limitation factor (Beer-Lambert law with self-shading)
    // Light availability decreases exponentially with water attenuation and phytoplankton biomass
    Type light_attenuation = k_w * z_mix + k_c * P_prev * z_mix;  // Total light attenuation in mixed layer
    Type light_limitation = exp(-light_attenuation);  // Light availability factor (0-1)
    
    // Calculate process rates
    // Equation 1: Phytoplankton growth with dual limitation (nutrients AND light)
    Type nutrient_limitation = N_prev / (K_N + N_prev);  // Michaelis-Menten nutrient limitation (0-1)
    Type uptake = r * nutrient_limitation * light_limitation * P_prev;  // Nutrient uptake by phytoplankton (g C m^-3 day^-1)
    
    // Equation 2: Zooplankton grazing on phytoplankton (Holling Type II functional response)
    Type grazing = g_max * (P_prev / (K_P + P_prev)) * Z_prev;  // Phytoplankton consumption by zooplankton (g C m^-3 day^-1)
    
    // Equation 3: Phytoplankton natural mortality
    Type P_mortality = m_P * P_prev;  // Phytoplankton death rate (g C m^-3 day^-1)
    
    // Equation 4: Zooplankton mortality (density-dependent)
    Type Z_mortality = m_Z * Z_prev * Z_prev;  // Zooplankton death rate, quadratic to represent density dependence (g C m^-3 day^-1)
    
    // Equation 5: Nutrient recycling from phytoplankton mortality
    Type N_recycling_P = gamma * P_mortality;  // Nutrients returned from dead phytoplankton (g C m^-3 day^-1)
    
    // Equation 6: Nutrient recycling from zooplankton mortality and excretion
    Type N_recycling_Z = delta * Z_mortality + (Type(1.0) - e) * grazing;  // Nutrients from zooplankton waste and inefficient assimilation (g C m^-3 day^-1)
    
    // Equation 7: Rate of change for nutrients (dN/dt)
    Type dN_dt = -uptake + N_recycling_P + N_recycling_Z;  // Net change in nutrient concentration (g C m^-3 day^-1)
    
    // Equation 8: Rate of change for phytoplankton (dP/dt)
    Type dP_dt = uptake - grazing - P_mortality;  // Net change in phytoplankton concentration (g C m^-3 day^-1)
    
    // Equation 9: Rate of change for zooplankton (dZ/dt)
    Type dZ_dt = e * grazing - Z_mortality;  // Net change in zooplankton concentration (g C m^-3 day^-1)
    
    // Update predictions using Euler method
    N_pred(i) = N_prev + dN_dt * dt;  // Update nutrient concentration
    P_pred(i) = P_prev + dP_dt * dt;  // Update phytoplankton concentration
    Z_pred(i) = Z_prev + dZ_dt * dt;  // Update zooplankton concentration
    
    // Ensure predictions remain non-negative
    N_pred(i) = CppAD::CondExpGt(N_pred(i), Type(0.0), N_pred(i), Type(epsilon));  // Set to epsilon if negative
    P_pred(i) = CppAD::CondExpGt(P_pred(i), Type(0.0), P_pred(i), Type(epsilon));  // Set to epsilon if negative
    Z_pred(i) = CppAD::CondExpGt(Z_pred(i), Type(0.0), Z_pred(i), Type(epsilon));  // Set to epsilon if negative
  }
  
  // Calculate likelihood for all observations
  for(int i = 0; i < n; i++) {
    // Normal likelihood for nutrient observations
    nll -= dnorm(N_dat(i), N_pred(i), sigma_N, true);  // Add negative log-likelihood for nutrient data
    
    // Normal likelihood for phytoplankton observations
    nll -= dnorm(P_dat(i), P_pred(i), sigma_P, true);  // Add negative log-likelihood for phytoplankton data
    
    // Normal likelihood for zooplankton observations
    nll -= dnorm(Z_dat(i), Z_pred(i), sigma_Z, true);  // Add negative log-likelihood for zooplankton data
  }
  
  // Report predictions and parameters
  REPORT(N_pred);  // Report predicted nutrient concentrations
  REPORT(P_pred);  // Report predicted phytoplankton concentrations
  REPORT(Z_pred);  // Report predicted zooplankton concentrations
  REPORT(sigma_N);  // Report nutrient observation error
  REPORT(sigma_P);  // Report phytoplankton observation error
  REPORT(sigma_Z);  // Report zooplankton observation error
  REPORT(r);  // Report phytoplankton growth rate
  REPORT(K_N);  // Report nutrient half-saturation constant
  REPORT(m_P);  // Report phytoplankton mortality rate
  REPORT(g_max);  // Report maximum grazing rate
  REPORT(K_P);  // Report grazing half-saturation constant
  REPORT(e);  // Report assimilation efficiency
  REPORT(m_Z);  // Report zooplankton mortality rate
  REPORT(gamma);  // Report phytoplankton recycling fraction
  REPORT(delta);  // Report zooplankton recycling fraction
  REPORT(k_w);  // Report background light attenuation
  REPORT(k_c);  // Report phytoplankton light attenuation
  REPORT(z_mix);  // Report mixed layer depth
  
  return nll;  // Return total negative log-likelihood
}
