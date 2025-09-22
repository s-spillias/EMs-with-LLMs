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
  PARAMETER(log_K_P);                   // Log phytoplankton carrying capacity (g C m^-3)
  PARAMETER(log_g_max);                 // Log maximum zooplankton grazing rate (day^-1)
  PARAMETER(log_K_Z);                   // Log half-saturation constant for zooplankton grazing (g C m^-3)
  PARAMETER(log_m_P);                   // Log phytoplankton mortality rate (day^-1)
  PARAMETER(log_m_Z);                   // Log zooplankton mortality rate (day^-1)
  PARAMETER(log_e);                     // Log zooplankton assimilation efficiency (dimensionless)
  PARAMETER(log_gamma);                 // Log nutrient recycling efficiency (dimensionless)
  PARAMETER(log_N_in);                  // Log external nutrient input rate (g C m^-3 day^-1)
  PARAMETER(log_sigma_N);               // Log observation error for nutrients
  PARAMETER(log_sigma_P);               // Log observation error for phytoplankton
  PARAMETER(log_sigma_Z);               // Log observation error for zooplankton
  
  // Transform parameters to natural scale with biological bounds
  Type r = exp(log_r);                  // Maximum phytoplankton growth rate (day^-1), typical range 0.1-2.0
  Type K_N = exp(log_K_N);              // Half-saturation for nutrient uptake (g C m^-3), typical range 0.01-1.0
  Type K_P = exp(log_K_P);              // Phytoplankton carrying capacity (g C m^-3), typical range 0.1-10.0
  Type g_max = exp(log_g_max);          // Maximum zooplankton grazing rate (day^-1), typical range 0.1-5.0
  Type K_Z = exp(log_K_Z);              // Half-saturation for grazing (g C m^-3), typical range 0.01-1.0
  Type m_P = exp(log_m_P);              // Phytoplankton mortality rate (day^-1), typical range 0.01-0.5
  Type m_Z = exp(log_m_Z);              // Zooplankton mortality rate (day^-1), typical range 0.01-1.0
  Type e = Type(0.2) + Type(0.6) * exp(log_e) / (Type(1.0) + exp(log_e));  // Assimilation efficiency (0.2-0.8)
  Type gamma = Type(0.1) + Type(0.8) * exp(log_gamma) / (Type(1.0) + exp(log_gamma)); // Recycling efficiency (0.1-0.9)
  Type N_in = exp(log_N_in);            // External nutrient input (g C m^-3 day^-1), typical range 0.001-0.1
  Type sigma_N = exp(log_sigma_N);      // Observation error for N
  Type sigma_P = exp(log_sigma_P);      // Observation error for P
  Type sigma_Z = exp(log_sigma_Z);      // Observation error for Z
  
  int n_obs = Time.size();              // Number of observations
  
  // Initialize state variables
  vector<Type> N_pred(n_obs);           // Predicted nutrient concentration
  vector<Type> P_pred(n_obs);           // Predicted phytoplankton concentration  
  vector<Type> Z_pred(n_obs);           // Predicted zooplankton concentration
  
  // Set initial conditions from first observations
  N_pred(0) = N_dat(0);                 // Initial nutrient concentration
  P_pred(0) = P_dat(0);                 // Initial phytoplankton concentration
  Z_pred(0) = Z_dat(0);                 // Initial zooplankton concentration
  
  // Numerical integration using Euler method
  for(int i = 1; i < n_obs; i++) {
    Type dt = Time(i) - Time(i-1);      // Time step size (days)
    
    // Previous time step values (avoid data leakage)
    Type N_prev = N_pred(i-1);          // Previous nutrient concentration
    Type P_prev = P_pred(i-1);          // Previous phytoplankton concentration
    Type Z_prev = Z_pred(i-1);          // Previous zooplankton concentration
    
    // Ensure positive values with small minimum
    N_prev = N_prev + Type(1e-10);      // Numerical stability for nutrients
    P_prev = P_prev + Type(1e-10);      // Numerical stability for phytoplankton
    Z_prev = Z_prev + Type(1e-10);      // Numerical stability for zooplankton
    
    // Equation 1: Nutrient limitation function (Michaelis-Menten)
    Type f_N = N_prev / (K_N + N_prev + Type(1e-10));
    
    // Equation 2: Phytoplankton growth with nutrient limitation and simple carrying capacity
    Type carrying_capacity_factor = K_P / (K_P + P_prev + Type(1e-10));
    Type P_growth = r * f_N * P_prev * carrying_capacity_factor;
    
    // Equation 3: Zooplankton functional response (Type II)
    Type f_Z = g_max * P_prev / (K_Z + P_prev + Type(1e-10));
    
    // Equation 4: Zooplankton grazing rate
    Type grazing = f_Z * Z_prev;
    
    // Equation 5: Phytoplankton mortality
    Type P_mortality = m_P * P_prev;
    
    // Equation 6: Zooplankton mortality  
    Type Z_mortality = m_Z * Z_prev;
    
    // Equation 7: Nutrient recycling from mortality and inefficient grazing
    Type N_recycling = gamma * (P_mortality + Z_mortality + (Type(1.0) - e) * grazing);
    
    // Differential equations
    // Equation 8: dN/dt = external input + recycling - phytoplankton uptake
    Type dN_dt = N_in + N_recycling - P_growth;
    
    // Equation 9: dP/dt = growth - grazing - mortality
    Type dP_dt = P_growth - grazing - P_mortality;
    
    // Equation 10: dZ/dt = efficient grazing - mortality
    Type dZ_dt = e * grazing - Z_mortality;
    
    // Update state variables using Euler integration
    N_pred(i) = N_prev + dt * dN_dt;    // Update nutrient concentration
    P_pred(i) = P_prev + dt * dP_dt;    // Update phytoplankton concentration
    Z_pred(i) = Z_prev + dt * dZ_dt;    // Update zooplankton concentration
    
    // Simple bounds to prevent negative values
    if(N_pred(i) < Type(1e-10)) N_pred(i) = Type(1e-10);  // Prevent negative nutrients
    if(P_pred(i) < Type(1e-10)) P_pred(i) = Type(1e-10);  // Prevent negative phytoplankton
    if(Z_pred(i) < Type(1e-10)) Z_pred(i) = Type(1e-10);  // Prevent negative zooplankton
  }
  
  // Calculate negative log-likelihood
  Type nll = Type(0.0);                 // Initialize negative log-likelihood
  
  // Add very weak penalties for extreme parameter values
  nll += Type(0.0001) * log_r * log_r;           // Very weak penalty on growth rate
  nll += Type(0.0001) * log_K_N * log_K_N;       // Very weak penalty on K_N
  nll += Type(0.0001) * log_g_max * log_g_max;   // Very weak penalty on grazing rate
  
  // Likelihood for nutrient observations (normal distribution on log scale)
  for(int i = 0; i < n_obs; i++) {
    if(N_pred(i) > Type(0.0) && N_dat(i) > Type(0.0)) {
      Type pred_log = log(N_pred(i));    // Log predicted value
      Type obs_log = log(N_dat(i));      // Log observed value
      nll -= dnorm(obs_log, pred_log, sigma_N, true);  // Lognormal likelihood for nutrients
    }
  }
  
  // Likelihood for phytoplankton observations (normal distribution on log scale)
  for(int i = 0; i < n_obs; i++) {
    if(P_pred(i) > Type(0.0) && P_dat(i) > Type(0.0)) {
      Type pred_log = log(P_pred(i));    // Log predicted value
      Type obs_log = log(P_dat(i));      // Log observed value
      nll -= dnorm(obs_log, pred_log, sigma_P, true);  // Lognormal likelihood for phytoplankton
    }
  }
  
  // Likelihood for zooplankton observations (normal distribution on log scale)
  for(int i = 0; i < n_obs; i++) {
    if(Z_pred(i) > Type(0.0) && Z_dat(i) > Type(0.0)) {
      Type pred_log = log(Z_pred(i));    // Log predicted value
      Type obs_log = log(Z_dat(i));      // Log observed value
      nll -= dnorm(obs_log, pred_log, sigma_Z, true);  // Lognormal likelihood for zooplankton
    }
  }
  
  // Report predicted values
  REPORT(N_pred);                       // Report predicted nutrient concentrations
  REPORT(P_pred);                       // Report predicted phytoplankton concentrations
  REPORT(Z_pred);                       // Report predicted zooplankton concentrations
  
  return nll;                           // Return negative log-likelihood
}
