#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data section
  DATA_VECTOR(Time);            // Time points for observations (days)
  DATA_VECTOR(N_dat);           // Observed nutrient concentrations (g C m^-3)
  DATA_VECTOR(P_dat);           // Observed phytoplankton concentrations (g C m^-3)
  DATA_VECTOR(Z_dat);           // Observed zooplankton concentrations (g C m^-3)
  
  // Parameter section
  PARAMETER(log_vmax);          // Log of max nutrient uptake rate (day^-1)
  PARAMETER(log_km);            // Log of half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(log_gamma);         // Log of phytoplankton growth efficiency (dimensionless)
  PARAMETER(log_gmax);          // Log of max zooplankton grazing rate (day^-1)
  PARAMETER(log_ks);            // Log of half-saturation constant for grazing (g C m^-3)
  PARAMETER(log_alpha);         // Log of zooplankton assimilation efficiency (dimensionless)
  PARAMETER(log_mp);            // Log of phytoplankton mortality rate (day^-1)
  PARAMETER(log_mz);            // Log of zooplankton mortality rate (day^-1)
  PARAMETER(log_beta);          // Log of nutrient recycling fraction (dimensionless)
  
  // Transform parameters to natural scale with biological constraints
  Type vmax = exp(log_vmax);    // Max nutrient uptake rate (0.5-2.0 day^-1)
  Type km = exp(log_km);        // Half-saturation for nutrients (0.1-1.0 g C m^-3)
  Type gamma = exp(log_gamma);   // Growth efficiency (0.3-0.8)
  Type gmax = exp(log_gmax);    // Max grazing rate (0.1-1.0 day^-1)
  Type ks = exp(log_ks);        // Half-saturation for grazing (0.1-1.0 g C m^-3)
  Type alpha = exp(log_alpha);   // Assimilation efficiency (0.3-0.8)
  Type mp = exp(log_mp);        // Phytoplankton mortality (0.05-0.2 day^-1)
  Type mz = exp(log_mz);        // Zooplankton mortality (0.05-0.2 day^-1)
  Type beta = exp(log_beta);    // Recycling fraction (0.5-0.9)
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Initialize negative log-likelihood and predictions
  Type nll = 0.0;
  vector<Type> N_pred(Time.size());
  vector<Type> P_pred(Time.size());
  vector<Type> Z_pred(Time.size());
  
  // Initial conditions
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);
  
  // Euler integration of NPZ dynamics with stability checks
  for(int t = 1; t < Time.size(); t++) {
    Type dt = Time(t) - Time(t-1);
    
    // 1. Nutrient uptake by phytoplankton (Michaelis-Menten)
    Type uptake = vmax * N_pred(t-1) * P_pred(t-1) / (km + N_pred(t-1) + eps);
    uptake = CppAD::CondExpGe(uptake, Type(0), uptake, Type(0)); // Ensure non-negative
    
    // 2. Zooplankton grazing on phytoplankton (Holling Type II)
    Type grazing = gmax * P_pred(t-1) * Z_pred(t-1) / (ks + P_pred(t-1) + eps);
    grazing = CppAD::CondExpGe(grazing, Type(0), grazing, Type(0)); // Ensure non-negative
    
    // 3. System dynamics with bounds checking
    Type dN = -uptake + beta * mp * P_pred(t-1) + beta * mz * Z_pred(t-1) + beta * (1 - alpha) * grazing;
    Type dP = gamma * uptake - grazing - mp * P_pred(t-1);
    Type dZ = alpha * grazing - mz * Z_pred(t-1);
    
    // Update with stability constraints
    N_pred(t) = CppAD::CondExpGe(N_pred(t-1) + dt * dN, eps, N_pred(t-1) + dt * dN, eps);
    P_pred(t) = CppAD::CondExpGe(P_pred(t-1) + dt * dP, eps, P_pred(t-1) + dt * dP, eps);
    Z_pred(t) = CppAD::CondExpGe(Z_pred(t-1) + dt * dZ, eps, Z_pred(t-1) + dt * dZ, eps);
  }
  
  // Observation model using lognormal error
  Type sigma_N = Type(0.2);  // Minimum SD for nutrient observations
  Type sigma_P = Type(0.2);  // Minimum SD for phytoplankton observations
  Type sigma_Z = Type(0.2);  // Minimum SD for zooplankton observations
  
  for(int t = 0; t < Time.size(); t++) {
    // Add lognormal likelihood contributions
    nll -= dnorm(log(N_dat(t)), log(N_pred(t) + eps), sigma_N, true);
    nll -= dnorm(log(P_dat(t)), log(P_pred(t) + eps), sigma_P, true);
    nll -= dnorm(log(Z_dat(t)), log(Z_pred(t) + eps), sigma_Z, true);
  }
  
  // Add smooth penalties to keep parameters in biological ranges
  nll += 0.1 * pow(log_vmax - log(1.0), 2);    // Center around 1.0 day^-1
  nll += 0.1 * pow(log_km - log(0.3), 2);      // Center around 0.3 g C m^-3
  nll += 0.1 * pow(log_gamma - log(0.5), 2);   // Center around 0.5
  nll += 0.1 * pow(log_gmax - log(0.5), 2);    // Center around 0.5 day^-1
  nll += 0.1 * pow(log_ks - log(0.3), 2);      // Center around 0.3 g C m^-3
  nll += 0.1 * pow(log_alpha - log(0.5), 2);   // Center around 0.5
  nll += 0.1 * pow(log_mp - log(0.1), 2);      // Center around 0.1 day^-1
  nll += 0.1 * pow(log_mz - log(0.1), 2);      // Center around 0.1 day^-1
  nll += 0.1 * pow(log_beta - log(0.7), 2);    // Center around 0.7
  
  // Report all values
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  REPORT(vmax);
  REPORT(km);
  REPORT(gamma);
  REPORT(gmax);
  REPORT(ks);
  REPORT(alpha);
  REPORT(mp);
  REPORT(mz);
  REPORT(beta);
  
  return nll;
}
