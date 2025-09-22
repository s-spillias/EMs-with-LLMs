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
  PARAMETER(log_g_max);                 // Log maximum zooplankton grazing rate (day^-1)
  PARAMETER(log_K_P);                   // Log half-saturation constant for zooplankton grazing (g C m^-3)
  PARAMETER(log_m_P);                   // Log phytoplankton mortality rate (day^-1)
  PARAMETER(log_m_Z);                   // Log zooplankton mortality rate (day^-1)
  PARAMETER(log_e);                     // Log zooplankton assimilation efficiency (dimensionless)
  PARAMETER(log_gamma);                 // Log nutrient recycling efficiency (dimensionless)
  PARAMETER(log_N_in);                  // Log external nutrient input rate (g C m^-3 day^-1)
  PARAMETER(log_sigma_N);               // Log observation error standard deviation for nutrients
  PARAMETER(log_sigma_P);               // Log observation error standard deviation for phytoplankton
  PARAMETER(log_sigma_Z);               // Log observation error standard deviation for zooplankton
  
  // Transform parameters to natural scale with bounds checking
  Type r = exp(log_r);                  // Maximum phytoplankton growth rate (0.1-5.0 day^-1)
  Type K_N = exp(log_K_N);              // Nutrient half-saturation (0.01-1.0 g C m^-3)
  Type g_max = exp(log_g_max);          // Maximum grazing rate (0.1-3.0 day^-1)
  Type K_P = exp(log_K_P);              // Grazing half-saturation (0.01-1.0 g C m^-3)
  Type m_P = exp(log_m_P);              // Phytoplankton mortality (0.01-1.0 day^-1)
  Type m_Z = exp(log_m_Z);              // Zooplankton mortality (0.01-1.0 day^-1)
  Type e = Type(1.0) / (Type(1.0) + exp(-log_e));  // Assimilation efficiency (0-1)
  Type gamma = Type(1.0) / (Type(1.0) + exp(-log_gamma)); // Recycling efficiency (0-1)
  Type N_in = exp(log_N_in);            // External nutrient input (0.001-0.1 g C m^-3 day^-1)
  Type sigma_N = exp(log_sigma_N);      // Nutrient observation error std dev
  Type sigma_P = exp(log_sigma_P);      // Phytoplankton observation error std dev  
  Type sigma_Z = exp(log_sigma_Z);      // Zooplankton observation error std dev
  
  // Add small constants for numerical stability
  Type eps = Type(1e-8);
  
  // Check for valid parameter values
  if(!isfinite(asDouble(r)) || !isfinite(asDouble(K_N)) || !isfinite(asDouble(g_max)) || 
     !isfinite(asDouble(K_P)) || !isfinite(asDouble(m_P)) || !isfinite(asDouble(m_Z)) ||
     !isfinite(asDouble(e)) || !isfinite(asDouble(gamma)) || !isfinite(asDouble(N_in)) ||
     !isfinite(asDouble(sigma_N)) || !isfinite(asDouble(sigma_P)) || !isfinite(asDouble(sigma_Z))) {
    return Type(1e10);  // Return large penalty for invalid parameters
  }
  
  // Initialize negative log-likelihood
  Type nll = Type(0.0);
  
  int n_obs = Time.size();
  
  // Initialize prediction vectors
  vector<Type> N_pred(n_obs);          // Predicted nutrient concentrations
  vector<Type> P_pred(n_obs);          // Predicted phytoplankton concentrations
  vector<Type> Z_pred(n_obs);          // Predicted zooplankton concentrations
  
  // Set initial conditions from first observations
  N_pred(0) = N_dat(0);                // Initial nutrient concentration
  P_pred(0) = P_dat(0);                // Initial phytoplankton concentration
  Z_pred(0) = Z_dat(0);                // Initial zooplankton concentration
  
  // Numerical integration using Euler method
  for(int i = 1; i < n_obs; i++) {
    Type dt = Time(i) - Time(i-1);     // Time step size
    
    // Previous time step values (avoid data leakage)
    Type N_prev = N_pred(i-1);         // Previous nutrient concentration
    Type P_prev = P_pred(i-1);         // Previous phytoplankton concentration  
    Type Z_prev = Z_pred(i-1);         // Previous zooplankton concentration
    
    // Add small constants to prevent division by zero
    N_prev = N_prev + eps;
    P_prev = P_prev + eps;
    Z_prev = Z_prev + eps;
    
    // Equation 1: Nutrient uptake rate by phytoplankton (Michaelis-Menten kinetics)
    Type uptake = r * (N_prev / (K_N + N_prev)) * P_prev;
    
    // Equation 2: Zooplankton grazing rate on phytoplankton (Holling Type II functional response)
    Type grazing = g_max * (P_prev / (K_P + P_prev)) * Z_prev;
    
    // Equation 3: Phytoplankton natural mortality
    Type P_mortality = m_P * P_prev;
    
    // Equation 4: Zooplankton natural mortality  
    Type Z_mortality = m_Z * Z_prev;
    
    // Equation 5: Nutrient recycling from mortality and inefficient grazing
    Type recycling = gamma * (P_mortality + Z_mortality + (Type(1.0) - e) * grazing);
    
    // Differential equations for NPZ dynamics
    // Equation 6: dN/dt = external input + recycling - uptake
    Type dN_dt = N_in + recycling - uptake;
    
    // Equation 7: dP/dt = uptake - grazing - mortality
    Type dP_dt = uptake - grazing - P_mortality;
    
    // Equation 8: dZ/dt = efficient grazing - mortality
    Type dZ_dt = e * grazing - Z_mortality;
    
    // Update predictions using Euler integration
    N_pred(i) = N_prev + dt * dN_dt;   // New nutrient concentration
    P_pred(i) = P_prev + dt * dP_dt;   // New phytoplankton concentration
    Z_pred(i) = Z_prev + dt * dZ_dt;   // New zooplankton concentration
    
    // Ensure non-negative concentrations
    if(N_pred(i) < eps) N_pred(i) = eps;
    if(P_pred(i) < eps) P_pred(i) = eps;
    if(Z_pred(i) < eps) Z_pred(i) = eps;
    
    // Check for numerical issues
    if(!isfinite(asDouble(N_pred(i))) || !isfinite(asDouble(P_pred(i))) || !isfinite(asDouble(Z_pred(i)))) {
      return Type(1e10);  // Return large penalty for numerical issues
    }
  }
  
  // Calculate likelihood using normal distribution with safe standard deviations
  Type min_sigma = Type(0.01);         // Minimum standard deviation to prevent numerical issues
  Type sigma_N_safe = sigma_N + min_sigma;
  Type sigma_P_safe = sigma_P + min_sigma;
  Type sigma_Z_safe = sigma_Z + min_sigma;
  
  // Add observation likelihood for all data points
  for(int i = 0; i < n_obs; i++) {
    Type nll_N = -dnorm(N_dat(i), N_pred(i), sigma_N_safe, true);  // Nutrient likelihood
    Type nll_P = -dnorm(P_dat(i), P_pred(i), sigma_P_safe, true);  // Phytoplankton likelihood
    Type nll_Z = -dnorm(Z_dat(i), Z_pred(i), sigma_Z_safe, true);  // Zooplankton likelihood
    
    // Check for numerical issues in likelihood calculation
    if(!isfinite(asDouble(nll_N)) || !isfinite(asDouble(nll_P)) || !isfinite(asDouble(nll_Z))) {
      return Type(1e10);  // Return large penalty for numerical issues
    }
    
    nll += nll_N + nll_P + nll_Z;
  }
  
  // Report predictions and parameters
  REPORT(N_pred);                      // Report predicted nutrient concentrations
  REPORT(P_pred);                      // Report predicted phytoplankton concentrations
  REPORT(Z_pred);                      // Report predicted zooplankton concentrations
  REPORT(r);                           // Report maximum growth rate
  REPORT(K_N);                         // Report nutrient half-saturation
  REPORT(g_max);                       // Report maximum grazing rate
  REPORT(K_P);                         // Report grazing half-saturation
  REPORT(m_P);                         // Report phytoplankton mortality
  REPORT(m_Z);                         // Report zooplankton mortality
  REPORT(e);                           // Report assimilation efficiency
  REPORT(gamma);                       // Report recycling efficiency
  REPORT(N_in);                        // Report external nutrient input
  
  return nll;                          // Return negative log-likelihood
}
