#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs
  DATA_VECTOR(Time);                    // Time vector (days)
  DATA_VECTOR(N_dat);                   // Observed nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);                   // Observed phytoplankton concentration (g C m^-3)  
  DATA_VECTOR(Z_dat);                   // Observed zooplankton concentration (g C m^-3)
  
  // Model parameters
  PARAMETER(log_r);                     // Log maximum phytoplankton growth rate (day^-1)
  PARAMETER(log_K_N);                   // Log half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(log_m_P);                   // Log phytoplankton mortality rate (day^-1)
  PARAMETER(log_g_max);                 // Log maximum zooplankton grazing rate (day^-1)
  PARAMETER(log_K_P);                   // Log half-saturation constant for zooplankton grazing (g C m^-3)
  PARAMETER(log_e);                     // Log zooplankton assimilation efficiency (dimensionless)
  PARAMETER(log_m_Z);                   // Log zooplankton mortality rate (day^-1)
  PARAMETER(log_gamma);                 // Log nutrient recycling efficiency from mortality (dimensionless)
  PARAMETER(log_delta);                 // Log nutrient recycling efficiency from zooplankton excretion (dimensionless)
  PARAMETER(log_N_in);                  // Log external nutrient input rate (g C m^-3 day^-1)
  PARAMETER(log_sigma_N);               // Log observation error standard deviation for nutrients
  PARAMETER(log_sigma_P);               // Log observation error standard deviation for phytoplankton
  PARAMETER(log_sigma_Z);               // Log observation error standard deviation for zooplankton
  
  // Transform parameters to natural scale with small constants for numerical stability
  Type r = exp(log_r);                  // Maximum phytoplankton growth rate (day^-1)
  Type K_N = exp(log_K_N) + Type(1e-8); // Half-saturation for nutrient uptake (g C m^-3)
  Type m_P = exp(log_m_P);              // Phytoplankton mortality rate (day^-1)
  Type g_max = exp(log_g_max);          // Maximum zooplankton grazing rate (day^-1)
  Type K_P = exp(log_K_P) + Type(1e-8); // Half-saturation for grazing (g C m^-3)
  Type e = Type(1.0) / (Type(1.0) + exp(-log_e)); // Assimilation efficiency (0-1, logistic transform)
  Type m_Z = exp(log_m_Z);              // Zooplankton mortality rate (day^-1)
  Type gamma = Type(1.0) / (Type(1.0) + exp(-log_gamma)); // Recycling efficiency from mortality (0-1)
  Type delta = Type(1.0) / (Type(1.0) + exp(-log_delta)); // Recycling efficiency from excretion (0-1)
  Type N_in = exp(log_N_in);            // External nutrient input (g C m^-3 day^-1)
  Type sigma_N = exp(log_sigma_N) + Type(1e-6); // Minimum observation error for nutrients
  Type sigma_P = exp(log_sigma_P) + Type(1e-6); // Minimum observation error for phytoplankton
  Type sigma_Z = exp(log_sigma_Z) + Type(1e-6); // Minimum observation error for zooplankton
  
  int n_obs = Time.size();              // Number of observations
  
  // Initialize state variables with observed initial conditions
  vector<Type> N_pred(n_obs);           // Predicted nutrient concentration
  vector<Type> P_pred(n_obs);           // Predicted phytoplankton concentration
  vector<Type> Z_pred(n_obs);           // Predicted zooplankton concentration
  
  // Set initial conditions from first observation
  N_pred(0) = N_dat(0);                 // Initial nutrient concentration
  P_pred(0) = P_dat(0);                 // Initial phytoplankton concentration
  Z_pred(0) = Z_dat(0);                 // Initial zooplankton concentration
  
  // Numerical integration using Euler method
  for(int i = 1; i < n_obs; i++) {
    Type dt = Time(i) - Time(i-1);      // Time step (days)
    
    // Previous time step values (avoid data leakage)
    Type N_prev = N_pred(i-1) + Type(1e-8); // Add small constant for numerical stability
    Type P_prev = P_pred(i-1) + Type(1e-8); // Add small constant for numerical stability
    Type Z_prev = Z_pred(i-1) + Type(1e-8); // Add small constant for numerical stability
    
    // Equation 1: Nutrient-limited phytoplankton growth (Michaelis-Menten kinetics)
    Type phyto_growth = r * (N_prev / (K_N + N_prev)) * P_prev;
    
    // Equation 2: Zooplankton grazing on phytoplankton (Type II functional response)
    Type grazing = g_max * (P_prev / (K_P + P_prev)) * Z_prev;
    
    // Equation 3: Phytoplankton mortality
    Type phyto_mortality = m_P * P_prev;
    
    // Equation 4: Zooplankton mortality  
    Type zoo_mortality = m_Z * Z_prev;
    
    // Equation 5: Nutrient recycling from phytoplankton mortality
    Type nutrient_recycling_P = gamma * phyto_mortality;
    
    // Equation 6: Nutrient recycling from zooplankton mortality
    Type nutrient_recycling_Z = gamma * zoo_mortality;
    
    // Equation 7: Nutrient excretion from zooplankton (unassimilated grazing)
    Type nutrient_excretion = delta * (Type(1.0) - e) * grazing;
    
    // State variable derivatives
    Type dN_dt = N_in - phyto_growth + nutrient_recycling_P + nutrient_recycling_Z + nutrient_excretion;
    Type dP_dt = phyto_growth - grazing - phyto_mortality;
    Type dZ_dt = e * grazing - zoo_mortality;
    
    // Update state variables using Euler integration
    N_pred(i) = N_prev + dt * dN_dt;
    P_pred(i) = P_prev + dt * dP_dt;
    Z_pred(i) = Z_prev + dt * dZ_dt;
    
    // Ensure non-negative concentrations using smooth maximum function
    Type min_val = Type(1e-8);
    N_pred(i) = Type(0.5) * (N_pred(i) + sqrt(N_pred(i) * N_pred(i) + min_val * min_val));
    P_pred(i) = Type(0.5) * (P_pred(i) + sqrt(P_pred(i) * P_pred(i) + min_val * min_val));
    Z_pred(i) = Type(0.5) * (Z_pred(i) + sqrt(Z_pred(i) * Z_pred(i) + min_val * min_val));
  }
  
  // Calculate negative log-likelihood
  Type nll = Type(0.0);
  
  // Likelihood for nutrient observations (lognormal distribution)
  for(int i = 0; i < n_obs; i++) {
    Type pred_log = log(N_pred(i) + Type(1e-8));
    Type obs_log = log(N_dat(i) + Type(1e-8));
    nll -= dnorm(obs_log, pred_log, sigma_N, true);
  }
  
  // Likelihood for phytoplankton observations (lognormal distribution)
  for(int i = 0; i < n_obs; i++) {
    Type pred_log = log(P_pred(i) + Type(1e-8));
    Type obs_log = log(P_dat(i) + Type(1e-8));
    nll -= dnorm(obs_log, pred_log, sigma_P, true);
  }
  
  // Likelihood for zooplankton observations (lognormal distribution)
  for(int i = 0; i < n_obs; i++) {
    Type pred_log = log(Z_pred(i) + Type(1e-8));
    Type obs_log = log(Z_dat(i) + Type(1e-8));
    nll -= dnorm(obs_log, pred_log, sigma_Z, true);
  }
  
  // Soft biological constraints using penalty functions
  // Constraint 1: Growth rate should be reasonable for phytoplankton
  if(r > Type(5.0)) nll += Type(10.0) * pow(r - Type(5.0), 2);
  
  // Constraint 2: Assimilation efficiency should be realistic
  if(e > Type(0.8)) nll += Type(10.0) * pow(e - Type(0.8), 2);
  if(e < Type(0.1)) nll += Type(10.0) * pow(Type(0.1) - e, 2);
  
  // Constraint 3: Recycling efficiencies should be reasonable
  if(gamma > Type(0.9)) nll += Type(10.0) * pow(gamma - Type(0.9), 2);
  if(delta > Type(0.9)) nll += Type(10.0) * pow(delta - Type(0.9), 2);
  
  // Report predicted values
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  
  return nll;
}
