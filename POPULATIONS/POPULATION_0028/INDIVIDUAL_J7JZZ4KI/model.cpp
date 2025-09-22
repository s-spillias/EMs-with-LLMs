#include <TMB.hpp>

// This TMB model simulates the dynamics of Nutrient (N), Phytoplankton (P) and Zooplankton (Z)
// in the oceanic mixed layer using a discrete-time approximation with Euler forward integration.

template<class Type>
Type objective_function<Type>::operator() () {
  using namespace density;
  
  // Data inputs:
  DATA_VECTOR(time);         // Time in days
  DATA_VECTOR(N_dat);        // Observed nutrient concentrations (g C m^-3)
  DATA_VECTOR(P_dat);        // Observed phytoplankton concentrations (g C m^-3)
  DATA_VECTOR(Z_dat);        // Observed zooplankton concentrations (g C m^-3)
  
  // Initial state parameters:
  PARAMETER(N0);             // Initial nutrient concentration (g C m^-3), from literature or initial estimate
  PARAMETER(P0);             // Initial phytoplankton concentration (g C m^-3), from literature or initial estimate
  PARAMETER(Z0);             // Initial zooplankton concentration (g C m^-3), from literature or initial estimate
  
  // Process parameters:
  PARAMETER(growth_rate);    // Phytoplankton intrinsic growth rate (day^-1); literature based or estimated
  PARAMETER(half_sat);       // Half-saturation constant for nutrient uptake (g C m^-3); expert opinion
  PARAMETER(grazing_rate);   // Zooplankton grazing rate (day^-1); literature based or estimated
  PARAMETER(efficiency);     // Conversion efficiency from grazed phytoplankton to zooplankton (dimensionless; range 0-1)
  PARAMETER(mortality_P);    // Phytoplankton mortality rate (day^-1); literature or expert estimate
  PARAMETER(mortality_Z);    // Zooplankton mortality rate (day^-1); literature or expert estimate
  
  // Error parameters for likelihood:
  PARAMETER(log_sigma_N);    // Log-transformed observation error for nutrient data
  PARAMETER(log_sigma_P);    // Log-transformed observation error for phytoplankton data
  PARAMETER(log_sigma_Z);    // Log-transformed observation error for zooplankton data
  
  // Transform error parameters to ensure positivity (add a small constant for numerical stability)
  Type sigma_N = exp(log_sigma_N) + Type(1e-8);  // Standard deviation for nutrient error
  Type sigma_P = exp(log_sigma_P) + Type(1e-8);  // Standard deviation for phytoplankton error
  Type sigma_Z = exp(log_sigma_Z) + Type(1e-8);  // Standard deviation for zooplankton error

  int n = time.size();
  vector<Type> N_pred(n);    // Predicted nutrient concentrations
  vector<Type> P_pred(n);    // Predicted phytoplankton concentrations
  vector<Type> Z_pred(n);    // Predicted zooplankton concentrations

  // Set initial conditions
  N_pred(0) = N0;  // t = 0 nutrient concentration
  P_pred(0) = P0;  // t = 0 phytoplankton concentration
  Z_pred(0) = Z0;  // t = 0 zooplankton concentration

  Type eps = Type(1e-8);  // Small constant to prevent division by zero and negative biomass

  // Numbered equations of the model:
  // 1. Nutrient: dN/dt = - uptake + recycling from grazing (unassimilated fraction)
  // 2. Phytoplankton: dP/dt = uptake (growth) - grazing - mortality
  // 3. Zooplankton: dZ/dt = grazing (converted by efficiency) - mortality

  // Discrete time simulation (Euler integration): use the previous time step state only
  for (int i = 1; i < n; i++){
    Type dt = time(i) - time(i-1);  // Time step length (days)
    
    // Resource limitation modeled by a Michaelis-Menten (saturating) function for nutrient uptake:
    Type uptake = growth_rate * P_pred(i-1) * N_pred(i-1) / (half_sat + N_pred(i-1) + eps);  // (g C m^-3 day^-1)
    
    // Grazing modeled with a saturating functional response:
    Type grazing = grazing_rate * Z_pred(i-1) * P_pred(i-1) / (half_sat + P_pred(i-1) + eps);  // (g C m^-3 day^-1)
    
    // Recycling: a fraction (1 - efficiency) of grazed biomass returns as nutrients:
    Type recycling = grazing * (1 - efficiency);  // (g C m^-3 day^-1)
    
    // Update nutrient concentration (Equation 1)
    N_pred(i) = N_pred(i-1) - uptake * dt + recycling * dt;
    
    // Update phytoplankton concentration (Equation 2)
    P_pred(i) = P_pred(i-1) + (uptake - grazing - mortality_P * P_pred(i-1)) * dt;
    
    // Update zooplankton concentration (Equation 3)
    Z_pred(i) = Z_pred(i-1) + (grazing * efficiency - mortality_Z * Z_pred(i-1)) * dt;
    
    // Ensure prediced concentrations do not fall below the stable minimum value
    N_pred(i) = fmax(N_pred(i), eps);
    P_pred(i) = fmax(P_pred(i), eps);
    Z_pred(i) = fmax(Z_pred(i), eps);
  }
  
  // Likelihood calculation: using lognormal error distributions for all observed data
  Type nll = 0.0;
  for (int i = 0; i < n; i++){
    nll -= dnorm(log(N_dat(i) + eps), log(N_pred(i) + eps), sigma_N, true);
    nll -= dnorm(log(P_dat(i) + eps), log(P_pred(i) + eps), sigma_P, true);
    nll -= dnorm(log(Z_dat(i) + eps), log(Z_pred(i) + eps), sigma_Z, true);
  }
  
  // Report model predictions for further analysis
  REPORT(N_pred);  // Report predicted nutrient concentrations
  REPORT(P_pred);  // Report predicted phytoplankton concentrations
  REPORT(Z_pred);  // Report predicted zooplankton concentrations
  
  return nll;
}
