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
  PARAMETER(log_K);                     // Log half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(log_g);                     // Log maximum zooplankton grazing rate (day^-1)
  PARAMETER(log_h);                     // Log half-saturation constant for zooplankton grazing (g C m^-3)
  PARAMETER(log_e);                     // Log zooplankton assimilation efficiency (dimensionless, 0-1)
  PARAMETER(log_m_p);                   // Log phytoplankton mortality rate (day^-1)
  PARAMETER(log_m_z);                   // Log zooplankton mortality rate (day^-1)
  PARAMETER(log_d);                     // Log nutrient recycling efficiency (dimensionless, 0-1)
  PARAMETER(log_N_in);                  // Log external nutrient input rate (g C m^-3 day^-1)
  PARAMETER(log_sigma_N);               // Log observation error for nutrients
  PARAMETER(log_sigma_P);               // Log observation error for phytoplankton
  PARAMETER(log_sigma_Z);               // Log observation error for zooplankton
  
  // Transform parameters to natural scale with biological bounds
  Type r = exp(log_r);                  // Maximum phytoplankton growth rate (day^-1)
  Type K = exp(log_K);                  // Half-saturation for nutrient uptake (g C m^-3)
  Type g = exp(log_g);                  // Maximum zooplankton grazing rate (day^-1)
  Type h = exp(log_h);                  // Half-saturation for grazing (g C m^-3)
  Type e = Type(1.0) / (Type(1.0) + exp(-log_e)); // Assimilation efficiency (0-1)
  Type m_p = exp(log_m_p);              // Phytoplankton mortality rate (day^-1)
  Type m_z = exp(log_m_z);              // Zooplankton mortality rate (day^-1)
  Type d = Type(1.0) / (Type(1.0) + exp(-log_d)); // Recycling efficiency (0-1)
  Type N_in = exp(log_N_in);            // External nutrient input (g C m^-3 day^-1)
  Type sigma_N = exp(log_sigma_N) + Type(1e-4); // Observation error for N
  Type sigma_P = exp(log_sigma_P) + Type(1e-4); // Observation error for P
  Type sigma_Z = exp(log_sigma_Z) + Type(1e-4); // Observation error for Z
  
  int n_obs = Time.size();              // Number of observations
  
  // Initialize state variables
  vector<Type> N_pred(n_obs);           // Predicted nutrient concentration
  vector<Type> P_pred(n_obs);           // Predicted phytoplankton concentration
  vector<Type> Z_pred(n_obs);           // Predicted zooplankton concentration
  
  // Set initial conditions from first observation with minimum bounds
  N_pred(0) = fmax(N_dat(0), Type(1e-6)); // Initial nutrient concentration
  P_pred(0) = fmax(P_dat(0), Type(1e-6)); // Initial phytoplankton concentration
  Z_pred(0) = fmax(Z_dat(0), Type(1e-6)); // Initial zooplankton concentration
  
  // Numerical integration using Euler method
  for(int i = 1; i < n_obs; i++) {
    Type dt = Time(i) - Time(i-1);      // Time step (days)
    
    // Previous time step values (avoid data leakage)
    Type N_prev = N_pred(i-1);          // Previous nutrient concentration
    Type P_prev = P_pred(i-1);          // Previous phytoplankton concentration  
    Type Z_prev = Z_pred(i-1);          // Previous zooplankton concentration
    
    // Ensure minimum values for numerical stability
    N_prev = fmax(N_prev, Type(1e-6));  // Numerical stability for nutrients
    P_prev = fmax(P_prev, Type(1e-6));  // Numerical stability for phytoplankton
    Z_prev = fmax(Z_prev, Type(1e-6));  // Numerical stability for zooplankton
    
    // Ecological process rates
    Type nutrient_limitation = N_prev / (K + N_prev); // Michaelis-Menten nutrient uptake
    Type phyto_growth = r * nutrient_limitation * P_prev; // Nutrient-limited phytoplankton growth
    Type grazing_rate = g * P_prev / (h + P_prev) * Z_prev; // Type II functional response grazing
    Type phyto_mortality = m_p * P_prev; // Phytoplankton natural mortality
    Type zoo_mortality = m_z * Z_prev;   // Zooplankton natural mortality
    Type nutrient_recycling = d * (phyto_mortality + zoo_mortality); // Remineralization
    
    // Differential equations for NPZ model
    // 1. dN/dt = N_in + nutrient_recycling - phyto_growth
    // 2. dP/dt = phyto_growth - grazing_rate - phyto_mortality  
    // 3. dZ/dt = e * grazing_rate - zoo_mortality
    
    Type dN_dt = N_in + nutrient_recycling - phyto_growth; // Nutrient dynamics
    Type dP_dt = phyto_growth - grazing_rate - phyto_mortality; // Phytoplankton dynamics
    Type dZ_dt = e * grazing_rate - zoo_mortality; // Zooplankton dynamics
    
    // Update state variables using Euler integration
    N_pred(i) = N_prev + dt * dN_dt;    // Update nutrient concentration
    P_pred(i) = P_prev + dt * dP_dt;    // Update phytoplankton concentration
    Z_pred(i) = Z_prev + dt * dZ_dt;    // Update zooplankton concentration
    
    // Ensure non-negative concentrations with minimum bounds
    N_pred(i) = fmax(N_pred(i), Type(1e-6)); // Prevent negative nutrients
    P_pred(i) = fmax(P_pred(i), Type(1e-6)); // Prevent negative phytoplankton
    Z_pred(i) = fmax(Z_pred(i), Type(1e-6)); // Prevent negative zooplankton
  }
  
  // Calculate negative log-likelihood
  Type nll = Type(0.0);                 // Initialize negative log-likelihood
  
  // Likelihood for nutrient observations (normal distribution on log scale)
  for(int i = 0; i < n_obs; i++) {
    Type pred_log = log(N_pred(i));     // Log predicted nutrients
    Type obs_log = log(fmax(N_dat(i), Type(1e-6))); // Log observed nutrients
    nll -= dnorm(obs_log, pred_log, sigma_N, true); // Lognormal likelihood for nutrients
  }
  
  // Likelihood for phytoplankton observations (normal distribution on log scale)
  for(int i = 0; i < n_obs; i++) {
    Type pred_log = log(P_pred(i));     // Log predicted phytoplankton
    Type obs_log = log(fmax(P_dat(i), Type(1e-6))); // Log observed phytoplankton
    nll -= dnorm(obs_log, pred_log, sigma_P, true); // Lognormal likelihood for phytoplankton
  }
  
  // Likelihood for zooplankton observations (normal distribution on log scale)
  for(int i = 0; i < n_obs; i++) {
    Type pred_log = log(Z_pred(i));     // Log predicted zooplankton
    Type obs_log = log(fmax(Z_dat(i), Type(1e-6))); // Log observed zooplankton
    nll -= dnorm(obs_log, pred_log, sigma_Z, true); // Lognormal likelihood for zooplankton
  }
  
  // Soft biological constraints using penalty functions
  Type penalty = Type(0.0);             // Initialize penalty term
  
  // Penalty for unrealistic growth rate (should be < 5 day^-1)
  if(r > Type(5.0)) penalty += Type(10.0) * pow(r - Type(5.0), 2);
  
  // Penalty for unrealistic grazing rate (should be < 2 day^-1)  
  if(g > Type(2.0)) penalty += Type(10.0) * pow(g - Type(2.0), 2);
  
  // Penalty for unrealistic mortality rates (should be < 1 day^-1)
  if(m_p > Type(1.0)) penalty += Type(10.0) * pow(m_p - Type(1.0), 2);
  if(m_z > Type(1.0)) penalty += Type(10.0) * pow(m_z - Type(1.0), 2);
  
  nll += penalty;                       // Add penalties to objective function
  
  // Check for numerical issues
  if(!isfinite(asDouble(nll))) {
    nll = Type(1e10);                   // Return large finite value if NaN/Inf
  }
  
  // Report predicted values and parameters
  REPORT(N_pred);                       // Report predicted nutrient concentrations
  REPORT(P_pred);                       // Report predicted phytoplankton concentrations
  REPORT(Z_pred);                       // Report predicted zooplankton concentrations
  REPORT(r);                            // Report phytoplankton growth rate
  REPORT(K);                            // Report nutrient half-saturation
  REPORT(g);                            // Report zooplankton grazing rate
  REPORT(h);                            // Report grazing half-saturation
  REPORT(e);                            // Report assimilation efficiency
  REPORT(m_p);                          // Report phytoplankton mortality
  REPORT(m_z);                          // Report zooplankton mortality
  REPORT(d);                            // Report recycling efficiency
  REPORT(N_in);                         // Report nutrient input rate
  
  return nll;                           // Return negative log-likelihood
}
