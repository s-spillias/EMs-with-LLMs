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
  PARAMETER(log_g_max);                 // Log maximum zooplankton grazing rate (day^-1)
  PARAMETER(log_K_P);                   // Log half-saturation constant for grazing (g C m^-3)
  PARAMETER(log_e);                     // Log grazing efficiency (dimensionless, 0-1)
  PARAMETER(log_m_P);                   // Log phytoplankton mortality rate (day^-1)
  PARAMETER(log_m_Z);                   // Log zooplankton mortality rate (day^-1)
  PARAMETER(log_gamma);                 // Log nutrient recycling efficiency (dimensionless, 0-1)
  PARAMETER(log_N_in);                  // Log external nutrient input rate (g C m^-3 day^-1)
  PARAMETER(log_sigma_N);               // Log observation error for nutrients
  PARAMETER(log_sigma_P);               // Log observation error for phytoplankton
  PARAMETER(log_sigma_Z);               // Log observation error for zooplankton
  
  // Transform parameters to natural scale with biological bounds
  Type r = exp(log_r);                  // Maximum phytoplankton growth rate (day^-1)
  Type K_N = exp(log_K_N);              // Half-saturation for nutrient uptake (g C m^-3)
  Type g_max = exp(log_g_max);          // Maximum zooplankton grazing rate (day^-1)
  Type K_P = exp(log_K_P);              // Half-saturation for grazing (g C m^-3)
  Type e = Type(1.0) / (Type(1.0) + exp(-log_e)); // Grazing efficiency (0-1, logistic transform)
  Type m_P = exp(log_m_P);              // Phytoplankton mortality rate (day^-1)
  Type m_Z = exp(log_m_Z);              // Zooplankton mortality rate (day^-1)
  Type gamma = Type(1.0) / (Type(1.0) + exp(-log_gamma)); // Recycling efficiency (0-1, logistic transform)
  Type N_in = exp(log_N_in);            // External nutrient input (g C m^-3 day^-1)
  Type sigma_N = exp(log_sigma_N);      // Observation error for nutrients
  Type sigma_P = exp(log_sigma_P);      // Observation error for phytoplankton
  Type sigma_Z = exp(log_sigma_Z);      // Observation error for zooplankton
  
  // Initialize negative log-likelihood
  Type nll = Type(0.0);
  
  // Add soft penalties to keep parameters within biological bounds
  nll -= dnorm(log_r, Type(log(0.5)), Type(2.0), true);        // Penalize extreme growth rates
  nll -= dnorm(log_g_max, Type(log(0.3)), Type(2.0), true);    // Penalize extreme grazing rates
  nll -= dnorm(log_e, Type(0.0), Type(2.0), true);             // Penalize extreme efficiencies
  nll -= dnorm(log_gamma, Type(0.0), Type(2.0), true);         // Penalize extreme recycling
  
  // Get number of time points
  int n_time = Time.size();
  
  // Initialize prediction vectors
  vector<Type> N_pred(n_time);
  vector<Type> P_pred(n_time);
  vector<Type> Z_pred(n_time);
  
  // Set initial conditions from first observations
  N_pred(0) = N_dat(0);                 // Initial nutrient concentration
  P_pred(0) = P_dat(0);                 // Initial phytoplankton concentration
  Z_pred(0) = Z_dat(0);                 // Initial zooplankton concentration
  
  // Time integration using Euler method
  for(int i = 1; i < n_time; i++) {
    Type dt = Time(i) - Time(i-1);      // Time step size (days)
    
    // Previous time step values (avoid data leakage)
    Type N_prev = N_pred(i-1);          // Previous nutrient concentration
    Type P_prev = P_pred(i-1);          // Previous phytoplankton concentration  
    Type Z_prev = Z_pred(i-1);          // Previous zooplankton concentration
    
    // Add small constants to prevent division by zero
    Type N_safe = N_prev + Type(1e-8);  // Numerically stable nutrient concentration
    Type P_safe = P_prev + Type(1e-8);  // Numerically stable phytoplankton concentration
    
    // Equation 1: Phytoplankton growth rate with Michaelis-Menten nutrient limitation
    Type growth_rate = r * (N_safe / (K_N + N_safe)); // Nutrient-limited growth (day^-1)
    
    // Equation 2: Zooplankton grazing rate with Holling Type II functional response
    Type grazing_rate = g_max * (P_safe / (K_P + P_safe)); // Density-dependent grazing (day^-1)
    
    // Equation 3: Nutrient dynamics - uptake by phytoplankton, recycling, external input
    Type dN_dt = -growth_rate * P_prev + gamma * (m_P * P_prev + m_Z * Z_prev) + N_in;
    
    // Equation 4: Phytoplankton dynamics - growth, grazing mortality, natural mortality
    Type dP_dt = growth_rate * P_prev - grazing_rate * Z_prev - m_P * P_prev;
    
    // Equation 5: Zooplankton dynamics - grazing with efficiency, natural mortality
    Type dZ_dt = e * grazing_rate * Z_prev - m_Z * Z_prev;
    
    // Update predictions using Euler integration with TMB-compatible maximum function
    Type N_new = N_prev + dt * dN_dt;
    Type P_new = P_prev + dt * dP_dt;
    Type Z_new = Z_prev + dt * dZ_dt;
    
    // Ensure non-negative values using smooth conditional expressions
    N_pred(i) = CppAD::CondExpGt(N_new, Type(1e-8), N_new, Type(1e-8)); // Ensure non-negative nutrients
    P_pred(i) = CppAD::CondExpGt(P_new, Type(1e-8), P_new, Type(1e-8)); // Ensure non-negative phytoplankton
    Z_pred(i) = CppAD::CondExpGt(Z_new, Type(1e-8), Z_new, Type(1e-8)); // Ensure non-negative zooplankton
  }
  
  // Calculate likelihood for all observations
  Type min_sigma_N = Type(0.01);        // Minimum observation error for nutrients
  Type min_sigma_P = Type(0.001);       // Minimum observation error for phytoplankton
  Type min_sigma_Z = Type(0.001);       // Minimum observation error for zooplankton
  
  for(int i = 0; i < n_time; i++) {
    // Use TMB-compatible maximum for sigma values
    Type sigma_N_safe = CppAD::CondExpGt(sigma_N, min_sigma_N, sigma_N, min_sigma_N);
    Type sigma_P_safe = CppAD::CondExpGt(sigma_P, min_sigma_P, sigma_P, min_sigma_P);
    Type sigma_Z_safe = CppAD::CondExpGt(sigma_Z, min_sigma_Z, sigma_Z, min_sigma_Z);
    
    // Use lognormal likelihood for strictly positive concentrations
    nll -= dnorm(log(N_dat(i) + Type(1e-8)), log(N_pred(i) + Type(1e-8)), 
                 sigma_N_safe, true); // Nutrient likelihood
    nll -= dnorm(log(P_dat(i) + Type(1e-8)), log(P_pred(i) + Type(1e-8)), 
                 sigma_P_safe, true); // Phytoplankton likelihood
    nll -= dnorm(log(Z_dat(i) + Type(1e-8)), log(Z_pred(i) + Type(1e-8)), 
                 sigma_Z_safe, true); // Zooplankton likelihood
  }
  
  // Report predictions and parameters
  REPORT(N_pred);                       // Predicted nutrient concentrations
  REPORT(P_pred);                       // Predicted phytoplankton concentrations
  REPORT(Z_pred);                       // Predicted zooplankton concentrations
  REPORT(r);                            // Maximum phytoplankton growth rate
  REPORT(K_N);                          // Nutrient half-saturation constant
  REPORT(g_max);                        // Maximum zooplankton grazing rate
  REPORT(K_P);                          // Grazing half-saturation constant
  REPORT(e);                            // Grazing efficiency
  REPORT(m_P);                          // Phytoplankton mortality rate
  REPORT(m_Z);                          // Zooplankton mortality rate
  REPORT(gamma);                        // Nutrient recycling efficiency
  REPORT(N_in);                         // External nutrient input rate
  
  return nll;
}
