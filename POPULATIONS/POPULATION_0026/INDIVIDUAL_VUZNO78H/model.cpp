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
  PARAMETER(log_d);                     // Log nutrient recycling efficiency (dimensionless)
  PARAMETER(log_S);                     // Log external nutrient supply rate (g C m^-3 day^-1)
  PARAMETER(log_sigma_N);               // Log observation error standard deviation for nutrients
  PARAMETER(log_sigma_P);               // Log observation error standard deviation for phytoplankton
  PARAMETER(log_sigma_Z);               // Log observation error standard deviation for zooplankton
  
  // Transform parameters from log scale with biological bounds
  Type r = exp(log_r);                  // Maximum phytoplankton growth rate (day^-1), from literature on marine phytoplankton
  Type K_N = exp(log_K_N);              // Half-saturation for nutrient uptake (g C m^-3), typical oceanic values
  Type m_P = exp(log_m_P);              // Phytoplankton mortality rate (day^-1), natural death + sinking losses
  Type g_max = exp(log_g_max);          // Maximum zooplankton grazing rate (day^-1), from feeding experiments
  Type K_P = exp(log_K_P);              // Half-saturation for grazing (g C m^-3), prey density for half-max feeding
  Type e = Type(1.0) / (Type(1.0) + exp(-log_e)); // Assimilation efficiency (0-1), logistic transformation for bounds
  Type m_Z = exp(log_m_Z);              // Zooplankton mortality rate (day^-1), natural death + predation
  Type d = Type(1.0) / (Type(1.0) + exp(-log_d)); // Recycling efficiency (0-1), fraction of dead matter recycled
  Type S = exp(log_S);                  // External nutrient supply (g C m^-3 day^-1), upwelling + inputs
  Type sigma_N = exp(log_sigma_N) + Type(1e-6); // Observation error for N with minimum bound
  Type sigma_P = exp(log_sigma_P) + Type(1e-6); // Observation error for P with minimum bound  
  Type sigma_Z = exp(log_sigma_Z) + Type(1e-6); // Observation error for Z with minimum bound
  
  int n_obs = Time.size();              // Number of observations
  
  // Initialize state variables
  vector<Type> N_pred(n_obs);           // Predicted nutrient concentration
  vector<Type> P_pred(n_obs);           // Predicted phytoplankton concentration
  vector<Type> Z_pred(n_obs);           // Predicted zooplankton concentration
  
  // Set initial conditions from first observation
  N_pred(0) = N_dat(0);                 // Initial nutrient concentration from data
  P_pred(0) = P_dat(0);                 // Initial phytoplankton concentration from data
  Z_pred(0) = Z_dat(0);                 // Initial zooplankton concentration from data
  
  // Numerical integration using Euler method
  for(int i = 1; i < n_obs; i++) {
    Type dt = Time(i) - Time(i-1);      // Time step size (days)
    
    // Previous time step values (avoid data leakage) with smooth lower bounds
    Type N_prev = N_pred(i-1) + Type(1e-8); // Add small constant for numerical stability
    Type P_prev = P_pred(i-1) + Type(1e-8); // Add small constant for numerical stability
    Type Z_prev = Z_pred(i-1) + Type(1e-8); // Add small constant for numerical stability
    
    // Equation 1: Phytoplankton growth rate with Michaelis-Menten nutrient limitation
    Type phyto_growth = r * (N_prev / (K_N + N_prev)) * P_prev;
    
    // Equation 2: Zooplankton grazing rate with Type II functional response
    Type grazing = g_max * (P_prev / (K_P + P_prev)) * Z_prev;
    
    // Equation 3: Nutrient recycling from phytoplankton mortality
    Type nutrient_from_phyto = d * m_P * P_prev;
    
    // Equation 4: Nutrient recycling from zooplankton mortality and excretion
    Type nutrient_from_zoo = d * m_Z * Z_prev + (Type(1.0) - e) * grazing;
    
    // Differential equations for NPZ dynamics
    // Equation 5: Nutrient dynamics - supply + recycling - uptake
    Type dN_dt = S + nutrient_from_phyto + nutrient_from_zoo - phyto_growth;
    
    // Equation 6: Phytoplankton dynamics - growth - mortality - grazing
    Type dP_dt = phyto_growth - m_P * P_prev - grazing;
    
    // Equation 7: Zooplankton dynamics - efficient grazing - mortality  
    Type dZ_dt = e * grazing - m_Z * Z_prev;
    
    // Update state variables using Euler integration with smooth lower bounds
    Type N_new = N_prev + dt * dN_dt;   // Forward Euler step for nutrients
    Type P_new = P_prev + dt * dP_dt;   // Forward Euler step for phytoplankton
    Type Z_new = Z_prev + dt * dZ_dt;   // Forward Euler step for zooplankton
    
    // Apply smooth lower bounds using sqrt transformation
    N_pred(i) = (N_new + sqrt(N_new * N_new + Type(4e-16))) / Type(2.0); // Smooth max with 1e-8
    P_pred(i) = (P_new + sqrt(P_new * P_new + Type(4e-16))) / Type(2.0); // Smooth max with 1e-8
    Z_pred(i) = (Z_new + sqrt(Z_new * Z_new + Type(4e-16))) / Type(2.0); // Smooth max with 1e-8
  }
  
  // Calculate negative log-likelihood
  Type nll = 0.0;                       // Initialize negative log-likelihood
  
  // Likelihood contributions from all observations using normal distribution
  for(int i = 0; i < n_obs; i++) {
    nll -= dnorm(N_dat(i), N_pred(i), sigma_N, true); // Nutrient observation likelihood
    nll -= dnorm(P_dat(i), P_pred(i), sigma_P, true); // Phytoplankton observation likelihood  
    nll -= dnorm(Z_dat(i), Z_pred(i), sigma_Z, true); // Zooplankton observation likelihood
  }
  
  // Report predicted values for plotting and diagnostics
  REPORT(N_pred);                       // Report predicted nutrient concentrations
  REPORT(P_pred);                       // Report predicted phytoplankton concentrations
  REPORT(Z_pred);                       // Report predicted zooplankton concentrations
  REPORT(r);                            // Report transformed growth rate parameter
  REPORT(K_N);                          // Report transformed nutrient half-saturation
  REPORT(m_P);                          // Report transformed phytoplankton mortality
  REPORT(g_max);                        // Report transformed maximum grazing rate
  REPORT(K_P);                          // Report transformed grazing half-saturation
  REPORT(e);                            // Report transformed assimilation efficiency
  REPORT(m_Z);                          // Report transformed zooplankton mortality
  REPORT(d);                            // Report transformed recycling efficiency
  REPORT(S);                            // Report transformed nutrient supply
  
  return nll;                           // Return negative log-likelihood for optimization
}
