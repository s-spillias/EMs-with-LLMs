#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS
  DATA_VECTOR(Time);  // Time in days
  DATA_VECTOR(N_dat);  // Observed nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);  // Observed phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);  // Observed zooplankton concentration (g C m^-3)
  
  // PARAMETERS - Nutrient dynamics
  PARAMETER(log_N_input);  // Log of external nutrient input rate (g C m^-3 day^-1) - mixing from deep water
  PARAMETER(log_k_N);  // Log of half-saturation constant for nutrient uptake (g C m^-3) - Michaelis-Menten parameter
  PARAMETER(log_recycle_P);  // Log of nutrient recycling rate from phytoplankton mortality (day^-1) - remineralization efficiency
  PARAMETER(log_recycle_Z);  // Log of nutrient recycling rate from zooplankton excretion (day^-1) - excretion efficiency
  
  // PARAMETERS - Phytoplankton dynamics
  PARAMETER(log_r_P);  // Log of maximum phytoplankton growth rate (day^-1) - photosynthetic capacity
  PARAMETER(log_m_P);  // Log of phytoplankton mortality rate (day^-1) - natural death and sinking
  PARAMETER(log_uptake_efficiency);  // Log of nutrient uptake efficiency (dimensionless) - conversion efficiency of N to P
  
  // PARAMETERS - Zooplankton dynamics
  PARAMETER(log_g_max);  // Log of maximum grazing rate (day^-1) - maximum consumption rate
  PARAMETER(log_k_P);  // Log of half-saturation constant for grazing (g C m^-3) - Holling Type II parameter
  PARAMETER(log_m_Z);  // Log of zooplankton mortality rate (day^-1) - natural death and predation
  PARAMETER(log_assimilation);  // Log of assimilation efficiency (dimensionless) - conversion efficiency of P to Z
  
  // OBSERVATION ERROR PARAMETERS
  PARAMETER(log_sigma_N);  // Log of observation error SD for nutrients (g C m^-3)
  PARAMETER(log_sigma_P);  // Log of observation error SD for phytoplankton (g C m^-3)
  PARAMETER(log_sigma_Z);  // Log of observation error SD for zooplankton (g C m^-3)
  
  // Transform parameters from log scale to natural scale
  Type N_input = exp(log_N_input);  // External nutrient input rate
  Type k_N = exp(log_k_N);  // Half-saturation for nutrient uptake
  Type recycle_P = exp(log_recycle_P);  // Phytoplankton recycling rate
  Type recycle_Z = exp(log_recycle_Z);  // Zooplankton recycling rate
  Type r_P = exp(log_r_P);  // Maximum phytoplankton growth rate
  Type m_P = exp(log_m_P);  // Phytoplankton mortality rate
  Type uptake_efficiency = exp(log_uptake_efficiency);  // Nutrient uptake efficiency
  Type g_max = exp(log_g_max);  // Maximum grazing rate
  Type k_P = exp(log_k_P);  // Half-saturation for grazing
  Type m_Z = exp(log_m_Z);  // Zooplankton mortality rate
  Type assimilation = exp(log_assimilation);  // Assimilation efficiency
  Type sigma_N = exp(log_sigma_N);  // Observation error for N
  Type sigma_P = exp(log_sigma_P);  // Observation error for P
  Type sigma_Z = exp(log_sigma_Z);  // Observation error for Z
  
  // Add minimum observation error to prevent numerical issues
  Type min_sigma = Type(0.001);  // Minimum SD to prevent division by zero
  sigma_N = sigma_N + min_sigma;  // Ensure sigma_N is not too small
  sigma_P = sigma_P + min_sigma;  // Ensure sigma_P is not too small
  sigma_Z = sigma_Z + min_sigma;  // Ensure sigma_Z is not too small
  
  // Soft constraints to keep parameters in biologically reasonable ranges
  Type nll = Type(0.0);  // Initialize negative log-likelihood
  Type penalty_weight = Type(10.0);  // Weight for soft constraint penalties
  
  // Penalize if uptake_efficiency is far from reasonable range (0.1 to 1.0)
  if(uptake_efficiency < Type(0.05)) nll += penalty_weight * pow(Type(0.05) - uptake_efficiency, 2);
  if(uptake_efficiency > Type(1.5)) nll += penalty_weight * pow(uptake_efficiency - Type(1.5), 2);
  
  // Penalize if assimilation is far from reasonable range (0.1 to 0.9)
  if(assimilation < Type(0.05)) nll += penalty_weight * pow(Type(0.05) - assimilation, 2);
  if(assimilation > Type(0.95)) nll += penalty_weight * pow(assimilation - Type(0.95), 2);
  
  // Initialize prediction vectors
  int n_obs = Time.size();  // Number of observations
  vector<Type> N_pred(n_obs);  // Predicted nutrient concentration
  vector<Type> P_pred(n_obs);  // Predicted phytoplankton concentration
  vector<Type> Z_pred(n_obs);  // Predicted zooplankton concentration
  
  // Set initial conditions from first observation
  N_pred(0) = N_dat(0);  // Initial nutrient concentration from data
  P_pred(0) = P_dat(0);  // Initial phytoplankton concentration from data
  Z_pred(0) = Z_dat(0);  // Initial zooplankton concentration from data
  
  // Small constant to prevent division by zero
  Type epsilon = Type(1e-8);  // Small constant for numerical stability
  
  // NUMERICAL INTEGRATION using Euler method
  for(int i = 1; i < n_obs; i++) {
    Type dt = Time(i) - Time(i-1);  // Time step size (days)
    
    // Get previous state (avoid using current time step data)
    Type N_prev = N_pred(i-1);  // Nutrient at previous time step
    Type P_prev = P_pred(i-1);  // Phytoplankton at previous time step
    Type Z_prev = Z_pred(i-1);  // Zooplankton at previous time step
    
    // Ensure non-negative concentrations
    N_prev = N_prev < Type(0.0) ? Type(0.0) : N_prev;  // Prevent negative nutrients
    P_prev = P_prev < Type(0.0) ? Type(0.0) : P_prev;  // Prevent negative phytoplankton
    Z_prev = Z_prev < Type(0.0) ? Type(0.0) : Z_prev;  // Prevent negative zooplankton
    
    // EQUATION 1: Nutrient uptake by phytoplankton (Michaelis-Menten kinetics)
    Type nutrient_uptake = (r_P * N_prev * P_prev) / (k_N + N_prev + epsilon);  // Nutrient-limited phytoplankton growth
    
    // EQUATION 2: Zooplankton grazing on phytoplankton (Holling Type II functional response)
    Type grazing = (g_max * P_prev * Z_prev) / (k_P + P_prev + epsilon);  // Density-dependent grazing
    
    // EQUATION 3: Nutrient recycling from mortality and excretion
    Type nutrient_regeneration = recycle_P * m_P * P_prev + recycle_Z * m_Z * Z_prev;  // Remineralization from dead biomass
    
    // EQUATION 4: Phytoplankton growth (nutrient-limited)
    Type phytoplankton_growth = uptake_efficiency * nutrient_uptake;  // Conversion of nutrients to phytoplankton
    
    // EQUATION 5: Zooplankton growth (food-limited)
    Type zooplankton_growth = assimilation * grazing;  // Conversion of phytoplankton to zooplankton
    
    // DIFFERENTIAL EQUATIONS
    // dN/dt: Nutrient dynamics
    Type dN_dt = N_input - nutrient_uptake + nutrient_regeneration;  // Input - uptake + recycling
    
    // dP/dt: Phytoplankton dynamics
    Type dP_dt = phytoplankton_growth - m_P * P_prev - grazing;  // Growth - mortality - grazing
    
    // dZ/dt: Zooplankton dynamics
    Type dZ_dt = zooplankton_growth - m_Z * Z_prev;  // Growth - mortality
    
    // Update predictions using Euler method
    N_pred(i) = N_prev + dt * dN_dt;  // Forward Euler integration for nutrients
    P_pred(i) = P_prev + dt * dP_dt;  // Forward Euler integration for phytoplankton
    Z_pred(i) = Z_prev + dt * dZ_dt;  // Forward Euler integration for zooplankton
    
    // Ensure predictions remain non-negative
    N_pred(i) = N_pred(i) < Type(0.0) ? Type(0.0) : N_pred(i);  // Bound nutrients at zero
    P_pred(i) = P_pred(i) < Type(0.0) ? Type(0.0) : P_pred(i);  // Bound phytoplankton at zero
    Z_pred(i) = Z_pred(i) < Type(0.0) ? Type(0.0) : Z_pred(i);  // Bound zooplankton at zero
  }
  
  // LIKELIHOOD CALCULATION - compare predictions to observations
  for(int i = 0; i < n_obs; i++) {
    // Normal likelihood for nutrient observations
    nll -= dnorm(N_dat(i), N_pred(i), sigma_N, true);  // Log-likelihood for nutrient data
    
    // Normal likelihood for phytoplankton observations
    nll -= dnorm(P_dat(i), P_pred(i), sigma_P, true);  // Log-likelihood for phytoplankton data
    
    // Normal likelihood for zooplankton observations
    nll -= dnorm(Z_dat(i), Z_pred(i), sigma_Z, true);  // Log-likelihood for zooplankton data
  }
  
  // REPORT predicted values for plotting and diagnostics
  REPORT(N_pred);  // Report predicted nutrient concentrations
  REPORT(P_pred);  // Report predicted phytoplankton concentrations
  REPORT(Z_pred);  // Report predicted zooplankton concentrations
  
  // ADREPORT for standard errors on predictions
  ADREPORT(N_pred);  // Standard errors for nutrient predictions
  ADREPORT(P_pred);  // Standard errors for phytoplankton predictions
  ADREPORT(Z_pred);  // Standard errors for zooplankton predictions
  
  return nll;  // Return total negative log-likelihood
}
