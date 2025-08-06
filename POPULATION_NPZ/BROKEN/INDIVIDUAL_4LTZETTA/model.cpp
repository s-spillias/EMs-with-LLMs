#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data section
  DATA_VECTOR(Time);                  // Time points (days)
  DATA_VECTOR(N_dat);                 // Nutrient observations (g C m^-3)
  DATA_VECTOR(P_dat);                 // Phytoplankton observations (g C m^-3)
  DATA_VECTOR(Z_dat);                 // Zooplankton observations (g C m^-3)
  
  // Parameter section
  PARAMETER(log_vmax);               // Maximum nutrient uptake rate (day^-1)
  PARAMETER(log_km);                 // Half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(log_gamma);              // Maximum grazing rate (day^-1)
  PARAMETER(log_ks);                 // Half-saturation constant for grazing (g C m^-3)
  PARAMETER(log_mort_p);             // Phytoplankton mortality rate (day^-1)
  PARAMETER(log_mort_z);             // Zooplankton mortality rate (day^-1)
  PARAMETER(logit_alpha);            // Zooplankton assimilation efficiency (dimensionless)
  PARAMETER(log_beta);               // Nutrient recycling fraction (dimensionless)
  PARAMETER(log_sigma_N);            // Observation error SD for nutrients
  PARAMETER(log_sigma_P);            // Observation error SD for phytoplankton
  PARAMETER(log_sigma_Z);            // Observation error SD for zooplankton
  PARAMETER(log_E_a);                // Activation energy (eV)
  PARAMETER(mean_temp);              // Mean temperature (Celsius)
  PARAMETER(temp_amplitude);         // Temperature amplitude (Celsius)

  // Transform parameters
  Type E_a = exp(log_E_a);          // Activation energy
  Type k_B = Type(8.617333262145e-5);  // Boltzmann constant (eV/K)
  Type T_ref = Type(293.15);        // Reference temperature (K)
  Type vmax = exp(log_vmax);
  Type km = exp(log_km);
  Type gamma = exp(log_gamma);
  Type ks = exp(log_ks);
  Type mort_p = exp(log_mort_p);
  Type mort_z = exp(log_mort_z);
  Type alpha = 1/(1 + exp(-logit_alpha));  // Transform to (0,1)
  Type beta = exp(log_beta);
  Type sigma_N = exp(log_sigma_N);
  Type sigma_P = exp(log_sigma_P);
  Type sigma_Z = exp(log_sigma_Z);
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Add smooth penalties to keep parameters in biological ranges
  nll -= dnorm(log_vmax, Type(0.0), Type(1.0), true);     // Prior on max uptake rate
  nll -= dnorm(log_km, Type(-1.6), Type(1.0), true);      // Prior on half-saturation
  nll -= dnorm(log_gamma, Type(-0.7), Type(1.0), true);   // Prior on max grazing
  nll -= dnorm(log_ks, Type(-2.3), Type(1.0), true);      // Prior on grazing half-saturation
  nll -= dnorm(log_mort_p, Type(-2.3), Type(1.0), true);  // Prior on phyto mortality
  nll -= dnorm(log_mort_z, Type(-2.3), Type(1.0), true);  // Prior on zoo mortality
  nll -= dnorm(logit_alpha, Type(1.4), Type(1.0), true);  // Prior on assimilation
  nll -= dnorm(log_beta, Type(-0.7), Type(1.0), true);    // Prior on recycling
  
  // Vectors for model predictions
  vector<Type> N_pred(Time.size());
  vector<Type> P_pred(Time.size());
  vector<Type> Z_pred(Time.size());
  
  // Initial conditions
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);
  
  // Integrate model using Euler method
  for(int t = 1; t < Time.size(); t++) {
    Type dt = Time(t) - Time(t-1);
    
    // Current state
    Type N = N_pred(t-1);
    Type P = P_pred(t-1);
    Type Z = Z_pred(t-1);
    
    // Simple sinusoidal temperature scaling
    Type temp_scale = Type(1.0) + Type(0.2) * sin(2.0 * M_PI * Time(t) / 365.0);
    
    // 1. Nutrient uptake by phytoplankton (Michaelis-Menten with temperature scaling)
    Type uptake = vmax * temp_scale * N * P / (km + N + eps);
    
    // 2. Zooplankton grazing (Holling Type II with temperature scaling)
    Type grazing = gamma * temp_scale * P * Z / (ks + P + eps);
    
    // 3. Calculate derivatives
    Type dN = -uptake + beta * mort_p * P + beta * mort_z * Z + (1-alpha) * grazing;
    Type dP = uptake - grazing - mort_p * P;
    Type dZ = alpha * grazing - mort_z * Z;
    
    // 4. Update state variables
    N_pred(t) = N + dN * dt;
    P_pred(t) = P + dP * dt;
    Z_pred(t) = Z + dZ * dt;
    
    // 5. Ensure positive concentrations and reasonable bounds
    N_pred(t) = N_pred(t) < eps ? eps : (N_pred(t) > Type(2.0) ? Type(2.0) : N_pred(t));
    P_pred(t) = P_pred(t) < eps ? eps : (P_pred(t) > Type(1.0) ? Type(1.0) : P_pred(t));
    Z_pred(t) = Z_pred(t) < eps ? eps : (Z_pred(t) > Type(0.5) ? Type(0.5) : Z_pred(t));
  }
  
  // Calculate likelihood using normal distribution on log scale
  for(int t = 0; t < Time.size(); t++) {
    // Skip zero/near-zero values in likelihood calculation
    if(N_dat(t) > Type(1e-4) && N_pred(t) > Type(1e-4)) {
      nll -= dnorm(log(N_dat(t)), log(N_pred(t)), sigma_N, true);
    }
    if(P_dat(t) > Type(1e-4) && P_pred(t) > Type(1e-4)) {
      nll -= dnorm(log(P_dat(t)), log(P_pred(t)), sigma_P, true);
    }
    if(Z_dat(t) > Type(1e-4) && Z_pred(t) > Type(1e-4)) {
      nll -= dnorm(log(Z_dat(t)), log(Z_pred(t)), sigma_Z, true);
    }
  }
  
  // Report predictions
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  REPORT(vmax);
  REPORT(km);
  REPORT(gamma);
  REPORT(ks);
  REPORT(mort_p);
  REPORT(mort_z);
  REPORT(alpha);
  REPORT(beta);
  
  return nll;
}
