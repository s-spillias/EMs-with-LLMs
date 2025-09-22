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
  PARAMETER(log_a);                     // Log zooplankton maximum grazing rate (day^-1)
  PARAMETER(log_b);                     // Log zooplankton half-saturation constant for grazing (g C m^-3)
  PARAMETER(log_e);                     // Log zooplankton assimilation efficiency (dimensionless)
  PARAMETER(log_m);                     // Log zooplankton mortality rate (day^-1)
  PARAMETER(log_gamma);                 // Log nutrient recycling efficiency (dimensionless)
  PARAMETER(log_d);                     // Log phytoplankton mortality rate (day^-1)
  PARAMETER(log_s);                     // Log external nutrient supply rate (g C m^-3 day^-1)
  
  // Observation error parameters
  PARAMETER(log_sigma_N);               // Log standard deviation for nutrient observations
  PARAMETER(log_sigma_P);               // Log standard deviation for phytoplankton observations
  PARAMETER(log_sigma_Z);               // Log standard deviation for zooplankton observations
  
  // Transform parameters to natural scale
  Type r = exp(log_r);                  // Maximum phytoplankton growth rate (day^-1)
  Type K = exp(log_K);                  // Half-saturation constant for nutrient uptake (g C m^-3)
  Type a = exp(log_a);                  // Zooplankton maximum grazing rate (day^-1)
  Type b = exp(log_b);                  // Zooplankton half-saturation constant for grazing (g C m^-3)
  Type e = exp(log_e);                  // Zooplankton assimilation efficiency (dimensionless)
  Type m = exp(log_m);                  // Zooplankton mortality rate (day^-1)
  Type gamma = exp(log_gamma);          // Nutrient recycling efficiency (dimensionless)
  Type d = exp(log_d);                  // Phytoplankton mortality rate (day^-1)
  Type s = exp(log_s);                  // External nutrient supply rate (g C m^-3 day^-1)
  
  Type sigma_N = exp(log_sigma_N);      // Standard deviation for nutrient observations
  Type sigma_P = exp(log_sigma_P);      // Standard deviation for phytoplankton observations
  Type sigma_Z = exp(log_sigma_Z);      // Standard deviation for zooplankton observations
  
  // Apply biological bounds using smooth penalties
  Type penalty = Type(0.0);
  penalty -= dnorm(log_e, log(Type(0.3)), Type(1.0), true);  // Efficiency around 30% with flexibility
  penalty -= dnorm(log_gamma, log(Type(0.5)), Type(1.0), true); // Recycling efficiency around 50%
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(1e-6);
  sigma_N = sigma_N + min_sigma;        // Prevent sigma from being too small
  sigma_P = sigma_P + min_sigma;        // Prevent sigma from being too small
  sigma_Z = sigma_Z + min_sigma;        // Prevent sigma from being too small
  
  int n_obs = Time.size();              // Number of observations
  
  // Initialize prediction vectors
  vector<Type> N_pred(n_obs);           // Predicted nutrient concentration
  vector<Type> P_pred(n_obs);           // Predicted phytoplankton concentration
  vector<Type> Z_pred(n_obs);           // Predicted zooplankton concentration
  
  // Set initial conditions from first observation
  N_pred(0) = N_dat(0);                 // Initial nutrient concentration
  P_pred(0) = P_dat(0);                 // Initial phytoplankton concentration
  Z_pred(0) = Z_dat(0);                 // Initial zooplankton concentration
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Numerical integration using Euler method
  for(int i = 1; i < n_obs; i++) {
    Type dt = Time(i) - Time(i-1);      // Time step (days)
    
    // Previous time step values (avoid data leakage)
    Type N_prev = N_pred(i-1);          // Previous nutrient concentration
    Type P_prev = P_pred(i-1);          // Previous phytoplankton concentration
    Type Z_prev = Z_pred(i-1);          // Previous zooplankton concentration
    
    // Ensure positive values with small epsilon
    N_prev = fmax(N_prev, eps);         // Prevent negative nutrients
    P_prev = fmax(P_prev, eps);         // Prevent negative phytoplankton
    Z_prev = fmax(Z_prev, eps);         // Prevent negative zooplankton
    
    // Equation 1: Phytoplankton growth rate with Michaelis-Menten nutrient limitation
    Type phyto_growth = r * (N_prev / (K + N_prev)) * P_prev;
    
    // Equation 2: Zooplankton grazing rate with Type II functional response
    Type grazing = a * (P_prev / (b + P_prev)) * Z_prev;
    
    // Equation 3: Nutrient recycling from zooplankton excretion and phytoplankton mortality
    Type nutrient_recycling = gamma * (grazing * (Type(1.0) - e) + d * P_prev + m * Z_prev);
    
    // Equation 4: Zooplankton growth from assimilated phytoplankton
    Type zoo_growth = e * grazing;
    
    // Differential equations for NPZ dynamics
    // Equation 5: dN/dt = external supply + recycling - phytoplankton uptake
    Type dN_dt = s + nutrient_recycling - phyto_growth;
    
    // Equation 6: dP/dt = phytoplankton growth - grazing - natural mortality
    Type dP_dt = phyto_growth - grazing - d * P_prev;
    
    // Equation 7: dZ/dt = zooplankton growth - natural mortality
    Type dZ_dt = zoo_growth - m * Z_prev;
    
    // Update predictions using Euler integration
    N_pred(i) = N_prev + dt * dN_dt;    // Update nutrient concentration
    P_pred(i) = P_prev + dt * dP_dt;    // Update phytoplankton concentration
    Z_pred(i) = Z_prev + dt * dZ_dt;    // Update zooplankton concentration
    
    // Ensure predictions remain positive
    N_pred(i) = fmax(N_pred(i), eps);   // Bound nutrients above zero
    P_pred(i) = fmax(P_pred(i), eps);   // Bound phytoplankton above zero
    Z_pred(i) = fmax(Z_pred(i), eps);   // Bound zooplankton above zero
  }
  
  // Calculate negative log-likelihood
  Type nll = Type(0.0);
  
  // Likelihood for all observations using normal distribution
  for(int i = 0; i < n_obs; i++) {
    nll -= dnorm(N_dat(i), N_pred(i), sigma_N, true);  // Nutrient likelihood
    nll -= dnorm(P_dat(i), P_pred(i), sigma_P, true);  // Phytoplankton likelihood
    nll -= dnorm(Z_dat(i), Z_pred(i), sigma_Z, true);  // Zooplankton likelihood
  }
  
  // Add parameter penalties
  nll += penalty;
  
  // Report predictions and parameters
  REPORT(N_pred);                       // Report predicted nutrient concentrations
  REPORT(P_pred);                       // Report predicted phytoplankton concentrations
  REPORT(Z_pred);                       // Report predicted zooplankton concentrations
  REPORT(r);                           // Report phytoplankton growth rate
  REPORT(K);                           // Report nutrient half-saturation constant
  REPORT(a);                           // Report zooplankton grazing rate
  REPORT(b);                           // Report grazing half-saturation constant
  REPORT(e);                           // Report assimilation efficiency
  REPORT(m);                           // Report zooplankton mortality rate
  REPORT(gamma);                       // Report recycling efficiency
  REPORT(d);                           // Report phytoplankton mortality rate
  REPORT(s);                           // Report nutrient supply rate
  REPORT(sigma_N);                     // Report nutrient observation error
  REPORT(sigma_P);                     // Report phytoplankton observation error
  REPORT(sigma_Z);                     // Report zooplankton observation error
  
  return nll;
}
