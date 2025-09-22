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
  Type e = Type(1.0) / (Type(1.0) + exp(-log_e));  // Assimilation efficiency (0-1), typical range 0.1-0.8
  Type gamma = Type(1.0) / (Type(1.0) + exp(-log_gamma)); // Recycling efficiency (0-1), typical range 0.1-0.9
  Type N_in = exp(log_N_in);            // External nutrient input (g C m^-3 day^-1), typical range 0.001-0.1
  Type sigma_N = exp(log_sigma_N) + Type(1e-6);  // Observation error for N with minimum bound
  Type sigma_P = exp(log_sigma_P) + Type(1e-6);  // Observation error for P with minimum bound
  Type sigma_Z = exp(log_sigma_Z) + Type(1e-6);  // Observation error for Z with minimum bound
  
  // Add smooth penalties for parameter bounds
  Type penalty = Type(0.0);
  penalty -= dnorm(log_r, Type(log(0.5)), Type(2.0), true);        // Soft constraint around reasonable growth rate
  penalty -= dnorm(log_K_N, Type(log(0.1)), Type(2.0), true);     // Soft constraint around reasonable K_N
  penalty -= dnorm(log_K_P, Type(log(1.0)), Type(2.0), true);     // Soft constraint around reasonable K_P
  penalty -= dnorm(log_g_max, Type(log(1.0)), Type(2.0), true);   // Soft constraint around reasonable grazing
  penalty -= dnorm(log_m_P, Type(log(0.1)), Type(2.0), true);     // Soft constraint around reasonable mortality
  penalty -= dnorm(log_m_Z, Type(log(0.2)), Type(2.0), true);     // Soft constraint around reasonable mortality
  
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
    
    // Add small constants to prevent division by zero
    N_prev = N_prev + Type(1e-8);       // Numerical stability for nutrients
    P_prev = P_prev + Type(1e-8);       // Numerical stability for phytoplankton
    Z_prev = Z_prev + Type(1e-8);       // Numerical stability for zooplankton
    
    // Equation 1: Nutrient limitation function (Michaelis-Menten)
    Type f_N = N_prev / (K_N + N_prev);
    
    // Equation 2: Phytoplankton growth with nutrient limitation and carrying capacity
    Type P_growth = r * f_N * P_prev * (Type(1.0) - P_prev / K_P);
    
    // Equation 3: Zooplankton functional response (Type II)
    Type f_Z = g_max * P_prev / (K_Z + P_prev);
    
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
    
    // Ensure non-negative concentrations using smooth transitions
    Type min_val = Type(1e-8);          // Minimum allowed concentration
    N_pred(i) = CppAD::CondExpGt(N_pred(i), min_val, N_pred(i), min_val);  // Prevent negative nutrients
    P_pred(i) = CppAD::CondExpGt(P_pred(i), min_val, P_pred(i), min_val);  // Prevent negative phytoplankton
    Z_pred(i) = CppAD::CondExpGt(Z_pred(i), min_val, Z_pred(i), min_val);  // Prevent negative zooplankton
  }
  
  // Calculate negative log-likelihood
  Type nll = penalty;                   // Start with parameter penalties
  
  // Likelihood for nutrient observations (lognormal distribution)
  for(int i = 0; i < n_obs; i++) {
    Type pred_log = log(N_pred(i) + Type(1e-8));  // Log predicted value with stability
    Type obs_log = log(N_dat(i) + Type(1e-8));    // Log observed value with stability
    nll -= dnorm(obs_log, pred_log, sigma_N, true);  // Lognormal likelihood for nutrients
  }
  
  // Likelihood for phytoplankton observations (lognormal distribution)
  for(int i = 0; i < n_obs; i++) {
    Type pred_log = log(P_pred(i) + Type(1e-8));  // Log predicted value with stability
    Type obs_log = log(P_dat(i) + Type(1e-8));    // Log observed value with stability
    nll -= dnorm(obs_log, pred_log, sigma_P, true);  // Lognormal likelihood for phytoplankton
  }
  
  // Likelihood for zooplankton observations (lognormal distribution)
  for(int i = 0; i < n_obs; i++) {
    Type pred_log = log(Z_pred(i) + Type(1e-8));  // Log predicted value with stability
    Type obs_log = log(Z_dat(i) + Type(1e-8));    // Log observed value with stability
    nll -= dnorm(obs_log, pred_log, sigma_Z, true);  // Lognormal likelihood for zooplankton
  }
  
  // Report predicted values
  REPORT(N_pred);                       // Report predicted nutrient concentrations
  REPORT(P_pred);                       // Report predicted phytoplankton concentrations
  REPORT(Z_pred);                       // Report predicted zooplankton concentrations
  
  return nll;                           // Return negative log-likelihood
}
