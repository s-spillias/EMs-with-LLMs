#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs
  DATA_VECTOR(Time_days_);  // Time vector in days
  DATA_VECTOR(N_dat_Nutrient_concentration_in_g_C_m_3_);      // Observed nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat_Phytoplankton_concentration_in_g_C_m_3_);      // Observed phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat_Zooplankton_concentration_in_g_C_m_3_);      // Observed zooplankton concentration (g C m^-3)
  
  // Parameters - Phytoplankton dynamics
  PARAMETER(r_max);        // Maximum phytoplankton growth rate (day^-1)
  PARAMETER(K_N);          // Half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(m_P);          // Phytoplankton mortality rate (day^-1)
  
  // Parameters - Zooplankton dynamics
  PARAMETER(g_max);        // Maximum zooplankton grazing rate (day^-1)
  PARAMETER(K_P);          // Half-saturation constant for grazing (g C m^-3)
  PARAMETER(m_Z);          // Zooplankton mortality rate (day^-1)
  PARAMETER(epsilon);      // Grazing efficiency (assimilation efficiency, dimensionless)
  PARAMETER(gamma);        // Zooplankton excretion rate (day^-1)
  
  // Observation error parameters
  PARAMETER(log_sigma_N);  // Log-scale standard deviation for nutrient observations
  PARAMETER(log_sigma_P);  // Log-scale standard deviation for phytoplankton observations
  PARAMETER(log_sigma_Z);  // Log-scale standard deviation for zooplankton observations
  
  // Transform log-scale parameters to natural scale
  Type sigma_N = exp(log_sigma_N);  // Standard deviation for nutrient observations (g C m^-3)
  Type sigma_P = exp(log_sigma_P);  // Standard deviation for phytoplankton observations (g C m^-3)
  Type sigma_Z = exp(log_sigma_Z);  // Standard deviation for zooplankton observations (g C m^-3)
  
  // Add minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(1e-4);  // Minimum standard deviation (g C m^-3)
  sigma_N = sigma_N + min_sigma;  // Ensure sigma_N is not too small
  sigma_P = sigma_P + min_sigma;  // Ensure sigma_P is not too small
  sigma_Z = sigma_Z + min_sigma;  // Ensure sigma_Z is not too small
  
  // Initialize negative log-likelihood
  Type nll = Type(0.0);  // Negative log-likelihood accumulator
  
  // Add soft constraints to keep parameters in biologically reasonable ranges
  Type penalty_weight = Type(10.0);  // Weight for penalty terms
  
  // Soft lower bounds (using smooth exponential penalties)
  nll -= penalty_weight * log(r_max + Type(1e-8));      // Penalize r_max approaching 0
  nll -= penalty_weight * log(K_N + Type(1e-8));        // Penalize K_N approaching 0
  nll -= penalty_weight * log(m_P + Type(1e-8));        // Penalize m_P approaching 0
  nll -= penalty_weight * log(g_max + Type(1e-8));      // Penalize g_max approaching 0
  nll -= penalty_weight * log(K_P + Type(1e-8));        // Penalize K_P approaching 0
  nll -= penalty_weight * log(m_Z + Type(1e-8));        // Penalize m_Z approaching 0
  nll -= penalty_weight * log(epsilon + Type(1e-8));    // Penalize epsilon approaching 0
  nll -= penalty_weight * log(Type(1.0) - epsilon + Type(1e-8));  // Penalize epsilon approaching 1
  nll -= penalty_weight * log(gamma + Type(1e-8));      // Penalize gamma approaching 0
  
  // Soft upper bounds (using smooth quadratic penalties)
  Type r_max_excess = r_max - Type(5.0);  // Excess above upper bound for r_max
  nll += penalty_weight * CppAD::CondExpGt(r_max_excess, Type(0.0), r_max_excess * r_max_excess, Type(0.0));  // Penalize r_max > 5 day^-1
  
  Type g_max_excess = g_max - Type(5.0);  // Excess above upper bound for g_max
  nll += penalty_weight * CppAD::CondExpGt(g_max_excess, Type(0.0), g_max_excess * g_max_excess, Type(0.0));  // Penalize g_max > 5 day^-1
  
  Type m_P_excess = m_P - Type(1.0);  // Excess above upper bound for m_P
  nll += penalty_weight * CppAD::CondExpGt(m_P_excess, Type(0.0), m_P_excess * m_P_excess, Type(0.0));  // Penalize m_P > 1 day^-1
  
  Type m_Z_excess = m_Z - Type(1.0);  // Excess above upper bound for m_Z
  nll += penalty_weight * CppAD::CondExpGt(m_Z_excess, Type(0.0), m_Z_excess * m_Z_excess, Type(0.0));  // Penalize m_Z > 1 day^-1
  
  Type gamma_excess = gamma - Type(1.0);  // Excess above upper bound for gamma
  nll += penalty_weight * CppAD::CondExpGt(gamma_excess, Type(0.0), gamma_excess * gamma_excess, Type(0.0));  // Penalize gamma > 1 day^-1
  
  // Get number of time steps
  int n = Time_days_.size();  // Number of observations
  
  // Initialize prediction vectors
  vector<Type> N_pred(n);  // Predicted nutrient concentration (g C m^-3)
  vector<Type> P_pred(n);  // Predicted phytoplankton concentration (g C m^-3)
  vector<Type> Z_pred(n);  // Predicted zooplankton concentration (g C m^-3)
  
  // Set initial conditions from first observation
  N_pred(0) = N_dat_Nutrient_concentration_in_g_C_m_3_(0);  // Initial nutrient concentration from data (g C m^-3)
  P_pred(0) = P_dat_Phytoplankton_concentration_in_g_C_m_3_(0);  // Initial phytoplankton concentration from data (g C m^-3)
  Z_pred(0) = Z_dat_Zooplankton_concentration_in_g_C_m_3_(0);  // Initial zooplankton concentration from data (g C m^-3)
  
  // Time integration using Euler method
  for(int i = 1; i < n; i++) {
    // Calculate time step
    Type dt = Time_days_(i) - Time_days_(i-1);  // Time step size (days)
    
    // Get previous state values (avoid data leakage)
    Type N_prev = N_pred(i-1);  // Nutrient concentration at previous time step (g C m^-3)
    Type P_prev = P_pred(i-1);  // Phytoplankton concentration at previous time step (g C m^-3)
    Type Z_prev = Z_pred(i-1);  // Zooplankton concentration at previous time step (g C m^-3)
    
    // Add small constant to prevent division by zero
    Type eps = Type(1e-8);  // Small constant for numerical stability
    
    // Equation 1: Nutrient uptake rate by phytoplankton (Michaelis-Menten)
    Type uptake = r_max * (N_prev / (K_N + N_prev + eps)) * P_prev;  // Nutrient uptake (g C m^-3 day^-1)
    
    // Equation 2: Zooplankton grazing rate (Holling Type II functional response)
    Type grazing = g_max * (P_prev / (K_P + P_prev + eps)) * Z_prev;  // Grazing rate (g C m^-3 day^-1)
    
    // Equation 3: Phytoplankton mortality
    Type P_mortality = m_P * P_prev;  // Phytoplankton mortality (g C m^-3 day^-1)
    
    // Equation 4: Zooplankton mortality
    Type Z_mortality = m_Z * Z_prev;  // Zooplankton mortality (g C m^-3 day^-1)
    
    // Equation 5: Zooplankton excretion
    Type excretion = gamma * Z_prev;  // Zooplankton excretion (g C m^-3 day^-1)
    
    // Equation 6: Nutrient recycling from phytoplankton mortality
    Type P_recycling = P_mortality;  // Nutrients from dead phytoplankton (g C m^-3 day^-1)
    
    // Equation 7: Nutrient recycling from zooplankton mortality
    Type Z_recycling = Z_mortality;  // Nutrients from dead zooplankton (g C m^-3 day^-1)
    
    // Equation 8: Nutrient recycling from inefficient grazing (sloppy feeding and excretion)
    Type grazing_recycling = (Type(1.0) - epsilon) * grazing + excretion;  // Nutrients from grazing losses (g C m^-3 day^-1)
    
    // Equation 9: Rate of change of nutrients
    Type dN_dt = -uptake + P_recycling + Z_recycling + grazing_recycling;  // Change in nutrient concentration (g C m^-3 day^-1)
    
    // Equation 10: Rate of change of phytoplankton
    Type dP_dt = uptake - grazing - P_mortality;  // Change in phytoplankton concentration (g C m^-3 day^-1)
    
    // Equation 11: Rate of change of zooplankton
    Type dZ_dt = epsilon * grazing - Z_mortality - excretion;  // Change in zooplankton concentration (g C m^-3 day^-1)
    
    // Update state variables using Euler integration with smooth lower bound at 0
    Type N_new = N_prev + dN_dt * dt;  // Tentative new nutrient concentration (g C m^-3)
    N_pred(i) = CppAD::CondExpGt(N_new, Type(0.0), N_new, Type(0.0));  // Bounded nutrient concentration (g C m^-3)
    
    Type P_new = P_prev + dP_dt * dt;  // Tentative new phytoplankton concentration (g C m^-3)
    P_pred(i) = CppAD::CondExpGt(P_new, Type(0.0), P_new, Type(0.0));  // Bounded phytoplankton concentration (g C m^-3)
    
    Type Z_new = Z_prev + dZ_dt * dt;  // Tentative new zooplankton concentration (g C m^-3)
    Z_pred(i) = CppAD::CondExpGt(Z_new, Type(0.0), Z_new, Type(0.0));  // Bounded zooplankton concentration (g C m^-3)
  }
  
  // Calculate likelihood for all observations
  for(int i = 0; i < n; i++) {
    // Nutrient likelihood (normal distribution)
    nll -= dnorm(N_dat_Nutrient_concentration_in_g_C_m_3_(i), N_pred(i), sigma_N, true);  // Negative log-likelihood for nutrient observations
    
    // Phytoplankton likelihood (normal distribution)
    nll -= dnorm(P_dat_Phytoplankton_concentration_in_g_C_m_3_(i), P_pred(i), sigma_P, true);  // Negative log-likelihood for phytoplankton observations
    
    // Zooplankton likelihood (normal distribution)
    nll -= dnorm(Z_dat_Zooplankton_concentration_in_g_C_m_3_(i), Z_pred(i), sigma_Z, true);  // Negative log-likelihood for zooplankton observations
  }
  
  // Report predicted values
  REPORT(N_pred);  // Report predicted nutrient concentrations
  REPORT(P_pred);  // Report predicted phytoplankton concentrations
  REPORT(Z_pred);  // Report predicted zooplankton concentrations
  
  // Report parameters
  REPORT(r_max);    // Report maximum phytoplankton growth rate
  REPORT(K_N);      // Report nutrient half-saturation constant
  REPORT(m_P);      // Report phytoplankton mortality rate
  REPORT(g_max);    // Report maximum grazing rate
  REPORT(K_P);      // Report grazing half-saturation constant
  REPORT(m_Z);      // Report zooplankton mortality rate
  REPORT(epsilon);  // Report grazing efficiency
  REPORT(gamma);    // Report excretion rate
  REPORT(sigma_N);  // Report nutrient observation error
  REPORT(sigma_P);  // Report phytoplankton observation error
  REPORT(sigma_Z);  // Report zooplankton observation error
  
  return nll;  // Return total negative log-likelihood
}
