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
  PARAMETER(log_m_P);                   // Log phytoplankton mortality rate (day^-1)
  PARAMETER(log_m_Z);                   // Log zooplankton mortality rate (day^-1)
  PARAMETER(log_e);                     // Log grazing efficiency (dimensionless)
  PARAMETER(log_gamma);                 // Log nutrient recycling efficiency (dimensionless)
  PARAMETER(log_N_in);                  // Log external nutrient input rate (g C m^-3 day^-1)
  PARAMETER(log_sigma_N);               // Log observation error for nutrients
  PARAMETER(log_sigma_P);               // Log observation error for phytoplankton
  PARAMETER(log_sigma_Z);               // Log observation error for zooplankton
  
  // Transform parameters to natural scale with biological bounds
  Type r = exp(log_r);                  // Maximum phytoplankton growth rate (0.1-5.0 day^-1)
  Type K_N = exp(log_K_N);              // Nutrient half-saturation (0.01-1.0 g C m^-3)
  Type g_max = exp(log_g_max);          // Maximum grazing rate (0.1-3.0 day^-1)
  Type K_P = exp(log_K_P);              // Grazing half-saturation (0.01-1.0 g C m^-3)
  Type m_P = exp(log_m_P);              // Phytoplankton mortality (0.01-1.0 day^-1)
  Type m_Z = exp(log_m_Z);              // Zooplankton mortality (0.01-1.0 day^-1)
  Type e = Type(1.0) / (Type(1.0) + exp(-log_e));  // Grazing efficiency (0-1, sigmoid)
  Type gamma = Type(1.0) / (Type(1.0) + exp(-log_gamma)); // Recycling efficiency (0-1, sigmoid)
  Type N_in = exp(log_N_in);            // External nutrient input (0.001-1.0 g C m^-3 day^-1)
  Type sigma_N = exp(log_sigma_N) + Type(1e-4);  // Minimum observation error for numerical stability
  Type sigma_P = exp(log_sigma_P) + Type(1e-4);  // Minimum observation error for numerical stability
  Type sigma_Z = exp(log_sigma_Z) + Type(1e-4);  // Minimum observation error for numerical stability
  
  int n_obs = Time.size();              // Number of observations
  
  // Initialize state variables with first observations
  vector<Type> N_pred(n_obs);           // Predicted nutrient concentration
  vector<Type> P_pred(n_obs);           // Predicted phytoplankton concentration
  vector<Type> Z_pred(n_obs);           // Predicted zooplankton concentration
  
  // Set initial conditions from first observation
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
    
    // Simple numerical stability - add small positive values
    Type N_stable = N_prev + Type(1e-8); // Nutrient with small constant
    Type P_stable = P_prev + Type(1e-8); // Phytoplankton with small constant
    Type Z_stable = Z_prev + Type(1e-8); // Zooplankton with small constant
    
    // Ecological rate calculations
    Type nutrient_limitation = N_stable / (K_N + N_stable);  // Michaelis-Menten nutrient uptake
    Type phyto_growth = r * nutrient_limitation * P_stable;  // Nutrient-limited phytoplankton growth
    Type grazing_rate = g_max * P_stable / (K_P + P_stable); // Holling Type II functional response
    Type grazing = grazing_rate * Z_stable;                  // Total grazing pressure
    Type phyto_mortality = m_P * P_stable;                   // Phytoplankton natural mortality
    Type zoo_mortality = m_Z * Z_stable;                     // Zooplankton natural mortality
    Type nutrient_recycling = gamma * (phyto_mortality + zoo_mortality); // Nutrient regeneration
    
    // Differential equations for NPZ model
    Type dN_dt = N_in - phyto_growth + nutrient_recycling;   // Nutrient dynamics: input - uptake + recycling
    Type dP_dt = phyto_growth - grazing - phyto_mortality;   // Phytoplankton dynamics: growth - grazing - mortality
    Type dZ_dt = e * grazing - zoo_mortality;                // Zooplankton dynamics: efficient grazing - mortality
    
    // Euler integration step
    N_pred(i) = N_prev + dt * dN_dt;     // Update nutrient concentration
    P_pred(i) = P_prev + dt * dP_dt;     // Update phytoplankton concentration
    Z_pred(i) = Z_prev + dt * dZ_dt;     // Update zooplankton concentration
    
    // Simple bounds to prevent negative values
    if(N_pred(i) < Type(1e-8)) N_pred(i) = Type(1e-8);  // Minimum nutrient concentration
    if(P_pred(i) < Type(1e-8)) P_pred(i) = Type(1e-8);  // Minimum phytoplankton concentration
    if(Z_pred(i) < Type(1e-8)) Z_pred(i) = Type(1e-8);  // Minimum zooplankton concentration
  }
  
  // Likelihood calculation using normal distribution
  Type nll = 0.0;                       // Initialize negative log-likelihood
  
  for(int i = 0; i < n_obs; i++) {
    // Nutrient observations likelihood
    nll -= dnorm(N_dat(i), N_pred(i), sigma_N, true);
    // Phytoplankton observations likelihood  
    nll -= dnorm(P_dat(i), P_pred(i), sigma_P, true);
    // Zooplankton observations likelihood
    nll -= dnorm(Z_dat(i), Z_pred(i), sigma_Z, true);
  }
  
  // Biological parameter bounds using weak penalties
  nll += Type(0.01) * pow(log_r, 2);                        // Weak penalty on growth rate
  nll += Type(0.01) * pow(log_g_max, 2);                    // Weak penalty on grazing rate
  nll += Type(0.01) * pow(log_K_N + Type(2.3), 2);         // Weak penalty on nutrient half-saturation
  nll += Type(0.01) * pow(log_K_P + Type(2.3), 2);         // Weak penalty on grazing half-saturation
  
  // Report predicted values for output
  REPORT(N_pred);                       // Report predicted nutrient concentrations
  REPORT(P_pred);                       // Report predicted phytoplankton concentrations
  REPORT(Z_pred);                       // Report predicted zooplankton concentrations
  REPORT(r);                           // Report transformed growth rate
  REPORT(K_N);                         // Report transformed nutrient half-saturation
  REPORT(g_max);                       // Report transformed maximum grazing rate
  REPORT(K_P);                         // Report transformed grazing half-saturation
  REPORT(m_P);                         // Report transformed phytoplankton mortality
  REPORT(m_Z);                         // Report transformed zooplankton mortality
  REPORT(e);                           // Report transformed grazing efficiency
  REPORT(gamma);                       // Report transformed recycling efficiency
  REPORT(N_in);                        // Report transformed nutrient input
  
  return nll;                          // Return negative log-likelihood for minimization
}

/*
Equation Descriptions:
1. Nutrient dynamics: dN/dt = N_in - r*(N/(K_N+N))*P + gamma*(m_P*P + m_Z*Z)
2. Phytoplankton dynamics: dP/dt = r*(N/(K_N+N))*P - g_max*(P/(K_P+P))*Z - m_P*P  
3. Zooplankton dynamics: dZ/dt = e*g_max*(P/(K_P+P))*Z - m_Z*Z
4. Nutrient limitation follows Michaelis-Menten kinetics
5. Grazing follows Holling Type II functional response
6. Nutrient recycling occurs through mortality and excretion
*/
