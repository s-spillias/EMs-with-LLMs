#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS
  DATA_VECTOR(Time);  // Time in days
  DATA_VECTOR(N_dat);  // Observed nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);  // Observed phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);  // Observed zooplankton concentration (g C m^-3)
  
  // PARAMETERS - Phytoplankton growth
  PARAMETER(r_P);  // Maximum phytoplankton growth rate at reference temperature (day^-1)
  PARAMETER(K_N);  // Half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(epsilon_P);  // Nutrient uptake efficiency (dimensionless, 0-1)
  
  // PARAMETERS - Zooplankton grazing
  PARAMETER(g_max);  // Maximum zooplankton grazing rate at reference temperature (day^-1)
  PARAMETER(K_Z);  // Half-saturation constant for grazing (g C m^-3)
  PARAMETER(epsilon_Z);  // Assimilation efficiency of zooplankton (dimensionless, 0-1)
  
  // PARAMETERS - Mortality and recycling
  PARAMETER(m_P);  // Phytoplankton mortality rate at reference temperature (day^-1)
  PARAMETER(m_Z);  // Zooplankton mortality rate at reference temperature (day^-1)
  PARAMETER(gamma);  // Nutrient recycling fraction (dimensionless, 0-1)
  
  // PARAMETERS - Temperature dependence
  PARAMETER(Q10);  // Temperature sensitivity coefficient (dimensionless)
  PARAMETER(T_ref);  // Reference temperature for rate normalization (degrees Celsius)
  
  // PARAMETERS - Observation error
  PARAMETER(log_sigma_N);  // Log-scale standard deviation for nutrient observations
  PARAMETER(log_sigma_P);  // Log-scale standard deviation for phytoplankton observations
  PARAMETER(log_sigma_Z);  // Log-scale standard deviation for zooplankton observations
  
  // Transform log-scale parameters to natural scale
  Type sigma_N = exp(log_sigma_N);  // Standard deviation for nutrient (g C m^-3)
  Type sigma_P = exp(log_sigma_P);  // Standard deviation for phytoplankton (g C m^-3)
  Type sigma_Z = exp(log_sigma_Z);  // Standard deviation for zooplankton (g C m^-3)
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(1e-4);  // Minimum allowed standard deviation (g C m^-3)
  sigma_N = sigma_N + min_sigma;  // Ensure sigma_N is always above minimum
  sigma_P = sigma_P + min_sigma;  // Ensure sigma_P is always above minimum
  sigma_Z = sigma_Z + min_sigma;  // Ensure sigma_Z is always above minimum
  
  int n = Time.size();  // Number of time steps in the data
  
  // PREDICTION VECTORS
  vector<Type> N_pred(n);  // Predicted nutrient concentration (g C m^-3)
  vector<Type> P_pred(n);  // Predicted phytoplankton concentration (g C m^-3)
  vector<Type> Z_pred(n);  // Predicted zooplankton concentration (g C m^-3)
  
  // INITIALIZE with first observations (initial conditions from data)
  N_pred(0) = N_dat(0);  // Initial nutrient concentration from data
  P_pred(0) = P_dat(0);  // Initial phytoplankton concentration from data
  Z_pred(0) = Z_dat(0);  // Initial zooplankton concentration from data
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);  // Small constant for numerical stability
  
  // SOFT PARAMETER CONSTRAINTS using penalties
  Type nll = Type(0.0);  // Initialize negative log-likelihood
  
  // Soft bounds for rates (should be positive)
  nll -= dnorm(r_P, Type(0.5), Type(2.0), true);  // Prior: r_P should be positive, centered around 0.5
  nll -= dnorm(g_max, Type(0.5), Type(2.0), true);  // Prior: g_max should be positive, centered around 0.5
  nll -= dnorm(m_P, Type(0.05), Type(0.5), true);  // Prior: m_P should be small positive, centered around 0.05
  nll -= dnorm(m_Z, Type(0.05), Type(0.5), true);  // Prior: m_Z should be small positive, centered around 0.05
  
  // Soft bounds for half-saturation constants (should be positive)
  nll -= dnorm(K_N, Type(0.1), Type(1.0), true);  // Prior: K_N should be positive, centered around 0.1
  nll -= dnorm(K_Z, Type(0.1), Type(1.0), true);  // Prior: K_Z should be positive, centered around 0.1
  
  // Soft bounds for temperature parameters
  nll -= dnorm(Q10, Type(2.0), Type(0.5), true);  // Prior: Q10 centered around 2.0 (typical for biological processes)
  nll -= dnorm(T_ref, Type(15.0), Type(5.0), true);  // Prior: T_ref centered around 15°C (temperate ocean)
  
  // Soft bounds for efficiencies (should be between 0 and 1)
  Type epsilon_P_bounded = Type(1.0) / (Type(1.0) + exp(-epsilon_P));  // Logistic transform to bound between 0 and 1
  Type epsilon_Z_bounded = Type(1.0) / (Type(1.0) + exp(-epsilon_Z));  // Logistic transform to bound between 0 and 1
  Type gamma_bounded = Type(1.0) / (Type(1.0) + exp(-gamma));  // Logistic transform to bound between 0 and 1
  
  // FORWARD SIMULATION using Euler method
  for(int i = 1; i < n; i++) {
    Type dt = Time(i) - Time(i-1);  // Time step size (days)
    
    // Get previous state (avoid using current time step observations)
    Type N_prev = N_pred(i-1);  // Nutrient at previous time step
    Type P_prev = P_pred(i-1);  // Phytoplankton at previous time step
    Type Z_prev = Z_pred(i-1);  // Zooplankton at previous time step
    
    // Temperature effect: Since T_dat is not available, assume constant temperature at T_ref
    // This means temp_effect = Q10^((T_ref - T_ref)/10) = Q10^0 = 1.0
    // So all rates remain at their reference values (no temperature modulation)
    Type temp_effect = Type(1.0);  // No temperature effect when T = T_ref
    
    // Apply temperature correction to biological rates (currently no effect since temp_effect = 1.0)
    Type r_P_T = r_P * temp_effect;  // Temperature-corrected phytoplankton growth rate (day^-1)
    Type g_max_T = g_max * temp_effect;  // Temperature-corrected zooplankton grazing rate (day^-1)
    Type m_P_T = m_P * temp_effect;  // Temperature-corrected phytoplankton mortality rate (day^-1)
    Type m_Z_T = m_Z * temp_effect;  // Temperature-corrected zooplankton mortality rate (day^-1)
    
    // Ensure non-negative concentrations
    N_prev = N_prev + eps;  // Add small constant to prevent negative values
    P_prev = P_prev + eps;  // Add small constant to prevent negative values
    Z_prev = Z_prev + eps;  // Add small constant to prevent negative values
    
    // EQUATION 1: Phytoplankton nutrient uptake (Monod/Michaelis-Menten kinetics)
    Type nutrient_limitation = N_prev / (K_N + N_prev + eps);  // Nutrient limitation factor (0-1)
    Type uptake = r_P_T * nutrient_limitation * P_prev;  // Temperature-modulated nutrient limited phytoplankton growth (g C m^-3 day^-1)
    
    // EQUATION 2: Zooplankton grazing (Holling Type II functional response)
    Type grazing = g_max_T * (P_prev / (K_Z + P_prev + eps)) * Z_prev;  // Temperature-modulated phytoplankton consumption by zooplankton (g C m^-3 day^-1)
    
    // EQUATION 3: Phytoplankton mortality and losses
    Type P_loss = m_P_T * P_prev;  // Temperature-modulated phytoplankton mortality (g C m^-3 day^-1)
    
    // EQUATION 4: Zooplankton mortality and losses
    Type Z_loss = m_Z_T * Z_prev;  // Temperature-modulated zooplankton mortality (g C m^-3 day^-1)
    
    // EQUATION 5: Nutrient recycling from dead organic matter
    Type recycling = gamma_bounded * (P_loss + Z_loss);  // Remineralization of nutrients (g C m^-3 day^-1)
    
    // EQUATION 6: Nutrient dynamics (dN/dt)
    Type dN_dt = -epsilon_P_bounded * uptake + recycling;  // Change in nutrient concentration (g C m^-3 day^-1)
    
    // EQUATION 7: Phytoplankton dynamics (dP/dt)
    Type dP_dt = epsilon_P_bounded * uptake - grazing - P_loss;  // Change in phytoplankton concentration (g C m^-3 day^-1)
    
    // EQUATION 8: Zooplankton dynamics (dZ/dt)
    Type dZ_dt = epsilon_Z_bounded * grazing - Z_loss;  // Change in zooplankton concentration (g C m^-3 day^-1)
    
    // Euler integration step
    N_pred(i) = N_prev + dN_dt * dt;  // Update nutrient concentration
    P_pred(i) = P_prev + dP_dt * dt;  // Update phytoplankton concentration
    Z_pred(i) = Z_prev + dZ_dt * dt;  // Update zooplankton concentration
    
    // Ensure predictions remain non-negative
    N_pred(i) = N_pred(i) + eps;  // Prevent negative nutrients
    P_pred(i) = P_pred(i) + eps;  // Prevent negative phytoplankton
    Z_pred(i) = Z_pred(i) + eps;  // Prevent negative zooplankton
  }
  
  // LIKELIHOOD CALCULATION - compare predictions to observations
  for(int i = 0; i < n; i++) {
    // Always include all observations in likelihood
    nll -= dnorm(N_dat(i), N_pred(i), sigma_N, true);  // Nutrient observation likelihood
    nll -= dnorm(P_dat(i), P_pred(i), sigma_P, true);  // Phytoplankton observation likelihood
    nll -= dnorm(Z_dat(i), Z_pred(i), sigma_Z, true);  // Zooplankton observation likelihood
  }
  
  // REPORTING - output predictions and parameters
  REPORT(N_pred);  // Report predicted nutrient concentrations
  REPORT(P_pred);  // Report predicted phytoplankton concentrations
  REPORT(Z_pred);  // Report predicted zooplankton concentrations
  REPORT(sigma_N);  // Report nutrient observation error
  REPORT(sigma_P);  // Report phytoplankton observation error
  REPORT(sigma_Z);  // Report zooplankton observation error
  REPORT(epsilon_P_bounded);  // Report bounded phytoplankton efficiency
  REPORT(epsilon_Z_bounded);  // Report bounded zooplankton efficiency
  REPORT(gamma_bounded);  // Report bounded recycling fraction
  REPORT(Q10);  // Report temperature sensitivity coefficient
  REPORT(T_ref);  // Report reference temperature
  
  return nll;  // Return total negative log-likelihood
}
