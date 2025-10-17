#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS
  DATA_VECTOR(Time);  // Time in days
  DATA_VECTOR(N_dat);  // Nutrient concentration observations (g C m^-3)
  DATA_VECTOR(P_dat);  // Phytoplankton concentration observations (g C m^-3)
  DATA_VECTOR(Z_dat);  // Zooplankton concentration observations (g C m^-3)
  
  // PARAMETERS - Phytoplankton growth and nutrient uptake
  PARAMETER(r);  // Maximum phytoplankton growth rate (day^-1) - determined from laboratory culture experiments
  PARAMETER(K_N);  // Half-saturation constant for nutrient uptake (g C m^-3) - from nutrient addition experiments
  
  // PARAMETERS - Zooplankton grazing
  PARAMETER(g_max);  // Maximum grazing rate (day^-1) - from feeding experiments
  PARAMETER(K_P);  // Half-saturation constant for grazing (g C m^-3) - from functional response studies
  PARAMETER(epsilon);  // Assimilation efficiency (dimensionless, 0-1) - fraction of ingested phytoplankton converted to zooplankton biomass
  
  // PARAMETERS - Mortality rates
  PARAMETER(m_P);  // Phytoplankton mortality rate (day^-1) - from dilution experiments and natural decay studies
  PARAMETER(m_Z);  // Zooplankton density-dependent mortality coefficient (m^3 (g C)^-1 day^-1) - represents predation by higher trophic levels
  
  // PARAMETERS - Nutrient recycling
  PARAMETER(gamma_P);  // Fraction of dead phytoplankton remineralized to nutrients (dimensionless, 0-1) - from decomposition studies
  PARAMETER(gamma_Z);  // Fraction of dead zooplankton remineralized to nutrients (dimensionless, 0-1) - from decomposition studies
  PARAMETER(excretion);  // Zooplankton nutrient excretion rate as fraction of ingestion (dimensionless, 0-1) - from metabolic studies
  
  // OBSERVATION ERROR PARAMETERS
  PARAMETER(log_sigma_N);  // Log-scale standard deviation for nutrient observations
  PARAMETER(log_sigma_P);  // Log-scale standard deviation for phytoplankton observations
  PARAMETER(log_sigma_Z);  // Log-scale standard deviation for zooplankton observations
  
  // Transform log-scale parameters to natural scale
  Type sigma_N = exp(log_sigma_N);  // Standard deviation for nutrient observation error (g C m^-3)
  Type sigma_P = exp(log_sigma_P);  // Standard deviation for phytoplankton observation error (g C m^-3)
  Type sigma_Z = exp(log_sigma_Z);  // Standard deviation for zooplankton observation error (g C m^-3)
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(1e-4);  // Minimum observation error to ensure numerical stability
  sigma_N = CppAD::CondExpGt(sigma_N, min_sigma, sigma_N, min_sigma);  // Enforce minimum for nutrient error
  sigma_P = CppAD::CondExpGt(sigma_P, min_sigma, sigma_P, min_sigma);  // Enforce minimum for phytoplankton error
  sigma_Z = CppAD::CondExpGt(sigma_Z, min_sigma, sigma_Z, min_sigma);  // Enforce minimum for zooplankton error
  
  int n_obs = Time.size();  // Number of time points in the dataset
  
  // PREDICTION VECTORS - store model predictions at each time point
  vector<Type> N_pred(n_obs);  // Predicted nutrient concentration (g C m^-3)
  vector<Type> P_pred(n_obs);  // Predicted phytoplankton concentration (g C m^-3)
  vector<Type> Z_pred(n_obs);  // Predicted zooplankton concentration (g C m^-3)
  
  // INITIALIZE with observed initial conditions
  N_pred(0) = N_dat(0);  // Set initial nutrient concentration from data
  P_pred(0) = P_dat(0);  // Set initial phytoplankton concentration from data
  Z_pred(0) = Z_dat(0);  // Set initial zooplankton concentration from data
  
  // Small constant to prevent division by zero
  Type epsilon_small = Type(1e-8);  // Numerical stability constant
  
  // SOFT PARAMETER CONSTRAINTS using smooth penalties
  Type nll = Type(0.0);  // Initialize negative log-likelihood
  
  // Penalize parameters outside biologically reasonable ranges
  nll -= dnorm(r, Type(0.5), Type(1.0), true);  // Prior: growth rate centered at 0.5 day^-1 with SD=1.0
  nll -= dnorm(g_max, Type(0.5), Type(1.0), true);  // Prior: grazing rate centered at 0.5 day^-1 with SD=1.0
  nll -= dnorm(K_N, Type(0.05), Type(0.1), true);  // Prior: nutrient half-saturation centered at 0.05 with SD=0.1
  nll -= dnorm(K_P, Type(0.1), Type(0.2), true);  // Prior: grazing half-saturation centered at 0.1 with SD=0.2
  
  // Penalize efficiency and fraction parameters if they drift outside [0,1]
  if(epsilon < Type(0.0)) nll += Type(100.0) * pow(epsilon, 2);  // Quadratic penalty for negative assimilation efficiency
  if(epsilon > Type(1.0)) nll += Type(100.0) * pow(epsilon - Type(1.0), 2);  // Quadratic penalty for assimilation efficiency > 1
  if(gamma_P < Type(0.0)) nll += Type(100.0) * pow(gamma_P, 2);  // Quadratic penalty for negative phytoplankton remineralization
  if(gamma_P > Type(1.0)) nll += Type(100.0) * pow(gamma_P - Type(1.0), 2);  // Quadratic penalty for phytoplankton remineralization > 1
  if(gamma_Z < Type(0.0)) nll += Type(100.0) * pow(gamma_Z, 2);  // Quadratic penalty for negative zooplankton remineralization
  if(gamma_Z > Type(1.0)) nll += Type(100.0) * pow(gamma_Z - Type(1.0), 2);  // Quadratic penalty for zooplankton remineralization > 1
  if(excretion < Type(0.0)) nll += Type(100.0) * pow(excretion, 2);  // Quadratic penalty for negative excretion rate
  if(excretion > Type(1.0)) nll += Type(100.0) * pow(excretion - Type(1.0), 2);  // Quadratic penalty for excretion rate > 1
  
  // Penalize negative mortality rates
  if(m_P < Type(0.0)) nll += Type(100.0) * pow(m_P, 2);  // Quadratic penalty for negative phytoplankton mortality
  if(m_Z < Type(0.0)) nll += Type(100.0) * pow(m_Z, 2);  // Quadratic penalty for negative zooplankton mortality coefficient
  
  // FORWARD SIMULATION using Euler integration
  for(int i = 1; i < n_obs; i++) {
    Type dt = Time(i) - Time(i-1);  // Time step size (days)
    
    // Get previous time step values (avoid data leakage)
    Type N_prev = N_pred(i-1);  // Nutrient concentration at previous time step
    Type P_prev = P_pred(i-1);  // Phytoplankton concentration at previous time step
    Type Z_prev = Z_pred(i-1);  // Zooplankton concentration at previous time step
    
    // Ensure non-negative concentrations with smooth lower bound
    N_prev = CppAD::CondExpGt(N_prev, epsilon_small, N_prev, epsilon_small);  // Prevent negative nutrient concentration
    P_prev = CppAD::CondExpGt(P_prev, epsilon_small, P_prev, epsilon_small);  // Prevent negative phytoplankton concentration
    Z_prev = CppAD::CondExpGt(Z_prev, epsilon_small, Z_prev, epsilon_small);  // Prevent negative zooplankton concentration
    
    // EQUATION 1: Nutrient uptake by phytoplankton (Michaelis-Menten kinetics)
    Type nutrient_limitation = N_prev / (K_N + N_prev + epsilon_small);  // Monod equation for nutrient limitation (dimensionless, 0-1)
    Type phyto_uptake = r * nutrient_limitation * P_prev;  // Nutrient uptake rate (g C m^-3 day^-1)
    
    // EQUATION 2: Zooplankton grazing on phytoplankton (Type II functional response)
    Type grazing_rate = (g_max * P_prev) / (K_P + P_prev + epsilon_small);  // Holling Type II functional response (day^-1)
    Type total_grazing = grazing_rate * Z_prev;  // Total phytoplankton consumed (g C m^-3 day^-1)
    
    // EQUATION 3: Phytoplankton mortality
    Type phyto_mortality = m_P * P_prev;  // Natural phytoplankton death rate (g C m^-3 day^-1)
    
    // EQUATION 4: Zooplankton density-dependent mortality (quadratic term for predation)
    Type zoo_mortality = m_Z * Z_prev * Z_prev;  // Density-dependent zooplankton mortality representing predation by higher trophic levels (g C m^-3 day^-1)
    
    // EQUATION 5: Nutrient remineralization from dead organic matter
    Type nutrient_from_phyto = gamma_P * phyto_mortality;  // Nutrients recycled from dead phytoplankton (g C m^-3 day^-1)
    Type nutrient_from_zoo = gamma_Z * zoo_mortality;  // Nutrients recycled from dead zooplankton (g C m^-3 day^-1)
    Type nutrient_from_excretion = excretion * total_grazing;  // Nutrients excreted by zooplankton (g C m^-3 day^-1)
    
    // EQUATION 6: Zooplankton growth from assimilated phytoplankton
    Type zoo_growth = epsilon * total_grazing;  // Zooplankton biomass gain from grazing (g C m^-3 day^-1)
    
    // DIFFERENTIAL EQUATIONS
    // dN/dt: Nutrient dynamics
    Type dN_dt = -phyto_uptake + nutrient_from_phyto + nutrient_from_zoo + nutrient_from_excretion;  // Net nutrient change (g C m^-3 day^-1)
    
    // dP/dt: Phytoplankton dynamics
    Type dP_dt = phyto_uptake - total_grazing - phyto_mortality;  // Net phytoplankton change (g C m^-3 day^-1)
    
    // dZ/dt: Zooplankton dynamics
    Type dZ_dt = zoo_growth - zoo_mortality;  // Net zooplankton change (g C m^-3 day^-1)
    
    // EULER INTEGRATION
    N_pred(i) = N_prev + dt * dN_dt;  // Update nutrient concentration
    P_pred(i) = P_prev + dt * dP_dt;  // Update phytoplankton concentration
    Z_pred(i) = Z_prev + dt * dZ_dt;  // Update zooplankton concentration
    
    // Ensure predictions remain non-negative
    N_pred(i) = CppAD::CondExpGt(N_pred(i), epsilon_small, N_pred(i), epsilon_small);  // Enforce non-negative nutrients
    P_pred(i) = CppAD::CondExpGt(P_pred(i), epsilon_small, P_pred(i), epsilon_small);  // Enforce non-negative phytoplankton
    Z_pred(i) = CppAD::CondExpGt(Z_pred(i), epsilon_small, Z_pred(i), epsilon_small);  // Enforce non-negative zooplankton
  }
  
  // LIKELIHOOD CALCULATION - compare predictions to observations
  for(int i = 0; i < n_obs; i++) {
    // Normal likelihood for nutrient observations
    nll -= dnorm(N_dat(i), N_pred(i), sigma_N, true);  // Log-likelihood contribution from nutrient data
    
    // Normal likelihood for phytoplankton observations
    nll -= dnorm(P_dat(i), P_pred(i), sigma_P, true);  // Log-likelihood contribution from phytoplankton data
    
    // Normal likelihood for zooplankton observations
    nll -= dnorm(Z_dat(i), Z_pred(i), sigma_Z, true);  // Log-likelihood contribution from zooplankton data
  }
  
  // REPORT predicted values for plotting and diagnostics
  REPORT(N_pred);  // Report predicted nutrient concentrations
  REPORT(P_pred);  // Report predicted phytoplankton concentrations
  REPORT(Z_pred);  // Report predicted zooplankton concentrations
  REPORT(sigma_N);  // Report nutrient observation error
  REPORT(sigma_P);  // Report phytoplankton observation error
  REPORT(sigma_Z);  // Report zooplankton observation error
  
  return nll;  // Return total negative log-likelihood for optimization
}
