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
  PARAMETER(r);  // Maximum phytoplankton growth rate (day^-1)
  PARAMETER(K_N);  // Half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(m_P);  // Phytoplankton mortality rate (day^-1)
  PARAMETER(g_max);  // Maximum zooplankton grazing rate (day^-1)
  PARAMETER(K_P);  // Half-saturation constant for grazing (g C m^-3)
  PARAMETER(e);  // Zooplankton assimilation efficiency (dimensionless, 0-1)
  PARAMETER(m_Z);  // Linear zooplankton mortality rate (day^-1)
  PARAMETER(m_Z2);  // Quadratic zooplankton mortality coefficient (day^-1 (g C m^-3)^-1)
  PARAMETER(log_sigma_N);  // Log standard deviation for nutrient observations
  PARAMETER(log_sigma_P);  // Log standard deviation for phytoplankton observations
  PARAMETER(log_sigma_Z);  // Log standard deviation for zooplankton observations
  
  // Transform log standard deviations to ensure positivity
  Type sigma_N = exp(log_sigma_N);  // Standard deviation for nutrient (g C m^-3)
  Type sigma_P = exp(log_sigma_P);  // Standard deviation for phytoplankton (g C m^-3)
  Type sigma_Z = exp(log_sigma_Z);  // Standard deviation for zooplankton (g C m^-3)
  
  // Add minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(1e-6);  // Minimum standard deviation (g C m^-3)
  sigma_N = sigma_N + min_sigma;  // Ensure sigma_N is not too small
  sigma_P = sigma_P + min_sigma;  // Ensure sigma_P is not too small
  sigma_Z = sigma_Z + min_sigma;  // Ensure sigma_Z is not too small
  
  // INITIALIZE PREDICTIONS
  int n = Time.size();  // Number of time points
  vector<Type> N_pred(n);  // Predicted nutrient concentrations (g C m^-3)
  vector<Type> P_pred(n);  // Predicted phytoplankton concentrations (g C m^-3)
  vector<Type> Z_pred(n);  // Predicted zooplankton concentrations (g C m^-3)
  
  // Set initial conditions from first observation
  N_pred(0) = N_dat(0);  // Initial nutrient concentration from data (g C m^-3)
  P_pred(0) = P_dat(0);  // Initial phytoplankton concentration from data (g C m^-3)
  Z_pred(0) = Z_dat(0);  // Initial zooplankton concentration from data (g C m^-3)
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);  // Small constant for numerical stability
  
  // SIMULATE DYNAMICS
  for(int i = 1; i < n; i++) {
    Type dt = Time(i) - Time(i-1);  // Time step (days)
    
    // Get previous state (avoid data leakage by using only previous predictions)
    Type N_prev = N_pred(i-1);  // Nutrient at previous time step (g C m^-3)
    Type P_prev = P_pred(i-1);  // Phytoplankton at previous time step (g C m^-3)
    Type Z_prev = Z_pred(i-1);  // Zooplankton at previous time step (g C m^-3)
    
    // Ensure non-negative concentrations using CppAD::CondExpGe (TMB-compatible)
    N_prev = CppAD::CondExpGe(N_prev, Type(0.0), N_prev, Type(0.0));  // Prevent negative nutrients (g C m^-3)
    P_prev = CppAD::CondExpGe(P_prev, Type(0.0), P_prev, Type(0.0));  // Prevent negative phytoplankton (g C m^-3)
    Z_prev = CppAD::CondExpGe(Z_prev, Type(0.0), Z_prev, Type(0.0));  // Prevent negative zooplankton (g C m^-3)
    
    // EQUATION 1: Nutrient uptake by phytoplankton (Monod/Michaelis-Menten kinetics)
    Type nutrient_limitation = N_prev / (K_N + N_prev + eps);  // Dimensionless (0-1), nutrient limitation factor
    
    // EQUATION 2: Phytoplankton growth rate
    Type phyto_growth = r * nutrient_limitation * P_prev;  // g C m^-3 day^-1, nutrient-limited growth
    
    // EQUATION 3: Phytoplankton mortality
    Type phyto_mortality = m_P * P_prev;  // g C m^-3 day^-1, linear mortality
    
    // EQUATION 4: Zooplankton grazing (Holling Type II functional response)
    Type grazing_rate = (g_max * P_prev) / (K_P + P_prev + eps);  // day^-1, per-capita grazing rate
    Type grazing = grazing_rate * Z_prev;  // g C m^-3 day^-1, total grazing flux
    
    // EQUATION 5: Zooplankton growth (assimilation of grazed phytoplankton)
    Type zoo_growth = e * grazing;  // g C m^-3 day^-1, assimilated carbon from grazing
    
    // EQUATION 6: Zooplankton mortality (linear + quadratic density-dependent)
    Type zoo_mortality_linear = m_Z * Z_prev;  // g C m^-3 day^-1, density-independent mortality
    Type zoo_mortality_quadratic = m_Z2 * Z_prev * Z_prev;  // g C m^-3 day^-1, density-dependent predation
    Type zoo_mortality = zoo_mortality_linear + zoo_mortality_quadratic;  // g C m^-3 day^-1, total mortality
    
    // EQUATION 7: Nutrient recycling (from phytoplankton mortality, zooplankton mortality, and inefficient grazing)
    Type nutrient_recycling = phyto_mortality + zoo_mortality + (Type(1.0) - e) * grazing;  // g C m^-3 day^-1, total nutrient return
    
    // EQUATION 8: Rate of change for nutrients (dN/dt)
    Type dN_dt = -phyto_growth + nutrient_recycling;  // g C m^-3 day^-1, net nutrient change
    
    // EQUATION 9: Rate of change for phytoplankton (dP/dt)
    Type dP_dt = phyto_growth - phyto_mortality - grazing;  // g C m^-3 day^-1, net phytoplankton change
    
    // EQUATION 10: Rate of change for zooplankton (dZ/dt)
    Type dZ_dt = zoo_growth - zoo_mortality;  // g C m^-3 day^-1, net zooplankton change
    
    // Update predictions using Euler integration
    N_pred(i) = N_prev + dN_dt * dt;  // Nutrient at current time step (g C m^-3)
    P_pred(i) = P_prev + dP_dt * dt;  // Phytoplankton at current time step (g C m^-3)
    Z_pred(i) = Z_prev + dZ_dt * dt;  // Zooplankton at current time step (g C m^-3)
    
    // Ensure non-negative predictions using CppAD::CondExpGe (TMB-compatible)
    N_pred(i) = CppAD::CondExpGe(N_pred(i), Type(0.0), N_pred(i), Type(0.0));  // Prevent negative nutrients (g C m^-3)
    P_pred(i) = CppAD::CondExpGe(P_pred(i), Type(0.0), P_pred(i), Type(0.0));  // Prevent negative phytoplankton (g C m^-3)
    Z_pred(i) = CppAD::CondExpGe(Z_pred(i), Type(0.0), Z_pred(i), Type(0.0));  // Prevent negative zooplankton (g C m^-3)
  }
  
  // CALCULATE NEGATIVE LOG-LIKELIHOOD
  Type nll = 0.0;  // Initialize negative log-likelihood
  
  // Add Gaussian likelihood for all observations
  for(int i = 0; i < n; i++) {
    nll -= dnorm(N_dat(i), N_pred(i), sigma_N, true);  // Nutrient observation likelihood
    nll -= dnorm(P_dat(i), P_pred(i), sigma_P, true);  // Phytoplankton observation likelihood
    nll -= dnorm(Z_dat(i), Z_pred(i), sigma_Z, true);  // Zooplankton observation likelihood
  }
  
  // SOFT PARAMETER CONSTRAINTS (biological realism penalties)
  // Growth rate should be positive and reasonable (0.01 to 5.0 day^-1)
  if(r < Type(0.01)) nll += Type(100.0) * pow(Type(0.01) - r, 2);  // Penalty for r too small
  if(r > Type(5.0)) nll += Type(100.0) * pow(r - Type(5.0), 2);  // Penalty for r too large
  
  // Half-saturation constants should be positive and reasonable (0.001 to 2.0 g C m^-3)
  if(K_N < Type(0.001)) nll += Type(100.0) * pow(Type(0.001) - K_N, 2);  // Penalty for K_N too small
  if(K_N > Type(2.0)) nll += Type(100.0) * pow(K_N - Type(2.0), 2);  // Penalty for K_N too large
  if(K_P < Type(0.001)) nll += Type(100.0) * pow(Type(0.001) - K_P, 2);  // Penalty for K_P too small
  if(K_P > Type(2.0)) nll += Type(100.0) * pow(K_P - Type(2.0), 2);  // Penalty for K_P too large
  
  // Mortality rates should be positive and reasonable (0.001 to 2.0 day^-1)
  if(m_P < Type(0.001)) nll += Type(100.0) * pow(Type(0.001) - m_P, 2);  // Penalty for m_P too small
  if(m_P > Type(2.0)) nll += Type(100.0) * pow(m_P - Type(2.0), 2);  // Penalty for m_P too large
  if(m_Z < Type(0.001)) nll += Type(100.0) * pow(Type(0.001) - m_Z, 2);  // Penalty for m_Z too small
  if(m_Z > Type(2.0)) nll += Type(100.0) * pow(m_Z - Type(2.0), 2);  // Penalty for m_Z too large
  
  // Quadratic mortality coefficient should be positive and reasonable (0.001 to 1.0 day^-1 (g C m^-3)^-1)
  if(m_Z2 < Type(0.001)) nll += Type(100.0) * pow(Type(0.001) - m_Z2, 2);  // Penalty for m_Z2 too small
  if(m_Z2 > Type(1.0)) nll += Type(100.0) * pow(m_Z2 - Type(1.0), 2);  // Penalty for m_Z2 too large
  
  // Grazing rate should be positive and reasonable (0.01 to 5.0 day^-1)
  if(g_max < Type(0.01)) nll += Type(100.0) * pow(Type(0.01) - g_max, 2);  // Penalty for g_max too small
  if(g_max > Type(5.0)) nll += Type(100.0) * pow(g_max - Type(5.0), 2);  // Penalty for g_max too large
  
  // Assimilation efficiency should be between 0 and 1
  if(e < Type(0.0)) nll += Type(1000.0) * pow(e, 2);  // Strong penalty for negative efficiency
  if(e > Type(1.0)) nll += Type(1000.0) * pow(e - Type(1.0), 2);  // Strong penalty for efficiency > 1
  
  // REPORT PREDICTIONS
  REPORT(N_pred);  // Report predicted nutrient concentrations
  REPORT(P_pred);  // Report predicted phytoplankton concentrations
  REPORT(Z_pred);  // Report predicted zooplankton concentrations
  
  return nll;  // Return negative log-likelihood for optimization
}
