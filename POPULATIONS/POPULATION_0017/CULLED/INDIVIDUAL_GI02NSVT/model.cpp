#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS
  DATA_VECTOR(Time);        // Time in days
  DATA_VECTOR(N_dat);       // Observed nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);       // Observed phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);       // Observed zooplankton concentration (g C m^-3)
  
  // PARAMETERS
  PARAMETER(r);             // Maximum phytoplankton growth rate (day^-1)
  PARAMETER(K_N);           // Half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(g_max);         // Maximum zooplankton grazing rate (day^-1)
  PARAMETER(K_P);           // Half-saturation constant for grazing (g C m^-3)
  PARAMETER(e);             // Zooplankton assimilation efficiency (dimensionless, 0-1)
  PARAMETER(m_P);           // Phytoplankton mortality rate (day^-1)
  PARAMETER(m_Z);           // Zooplankton mortality rate (day^-1)
  PARAMETER(log_sigma_N);   // Log-scale observation error for nutrients
  PARAMETER(log_sigma_P);   // Log-scale observation error for phytoplankton
  PARAMETER(log_sigma_Z);   // Log-scale observation error for zooplankton
  
  // Transform log-scale parameters to natural scale
  Type sigma_N = exp(log_sigma_N);  // Standard deviation for nutrient observations
  Type sigma_P = exp(log_sigma_P);  // Standard deviation for phytoplankton observations
  Type sigma_Z = exp(log_sigma_Z);  // Standard deviation for zooplankton observations
  
  // Get number of time steps
  int n = Time.size();
  
  // Initialize prediction vectors
  vector<Type> N_pred(n);   // Predicted nutrient concentration
  vector<Type> P_pred(n);   // Predicted phytoplankton concentration
  vector<Type> Z_pred(n);   // Predicted zooplankton concentration
  
  // Set initial conditions from first observation
  N_pred(0) = N_dat(0);     // Initial nutrient concentration from data
  P_pred(0) = P_dat(0);     // Initial phytoplankton concentration from data
  Z_pred(0) = Z_dat(0);     // Initial zooplankton concentration from data
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);    // Numerical stability constant
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(1e-6);  // Minimum observation error
  Type sigma_N_use = sigma_N + min_sigma;  // Effective nutrient observation error
  Type sigma_P_use = sigma_P + min_sigma;  // Effective phytoplankton observation error
  Type sigma_Z_use = sigma_Z + min_sigma;  // Effective zooplankton observation error
  
  // ECOLOGICAL PROCESS EQUATIONS (numbered for reference):
  // 1. Phytoplankton growth: Monod kinetics for nutrient-limited growth
  // 2. Zooplankton grazing: Holling Type II functional response
  // 3. Nutrient uptake: Consumed by phytoplankton growth
  // 4. Nutrient regeneration: From phytoplankton and zooplankton mortality
  // 5. Phytoplankton loss: Grazing by zooplankton plus natural mortality
  // 6. Zooplankton growth: Assimilation of grazed phytoplankton
  // 7. Zooplankton loss: Natural mortality
  
  // Forward simulation using discrete-time approximation
  for(int i = 1; i < n; i++) {
    Type dt = Time(i) - Time(i-1);  // Time step size (days)
    
    // Get previous state (avoiding data leakage)
    Type N_prev = N_pred(i-1);      // Nutrient at previous time step
    Type P_prev = P_pred(i-1);      // Phytoplankton at previous time step
    Type Z_prev = Z_pred(i-1);      // Zooplankton at previous time step
    
    // Ensure non-negative concentrations with smooth lower bound
    N_prev = N_prev + eps;          // Prevent negative nutrients
    P_prev = P_prev + eps;          // Prevent negative phytoplankton
    Z_prev = Z_prev + eps;          // Prevent negative zooplankton
    
    // Equation 1: Nutrient-limited phytoplankton growth rate (Monod kinetics)
    Type growth_rate = r * N_prev / (K_N + N_prev);  // day^-1, saturating function of nutrient availability
    
    // Equation 2: Zooplankton grazing rate (Holling Type II functional response)
    Type grazing_rate = g_max * P_prev / (K_P + P_prev);  // day^-1, saturating function of phytoplankton density
    
    // Equation 3: Phytoplankton biomass flux from growth
    Type P_growth = growth_rate * P_prev * dt;  // g C m^-3, nutrient uptake converted to phytoplankton
    
    // Equation 4: Phytoplankton biomass flux from grazing
    Type P_grazed = grazing_rate * Z_prev * dt;  // g C m^-3, phytoplankton consumed by zooplankton
    
    // Equation 5: Phytoplankton biomass flux from mortality
    Type P_mortality = m_P * P_prev * dt;  // g C m^-3, natural phytoplankton death
    
    // Equation 6: Zooplankton biomass flux from assimilated grazing
    Type Z_growth = e * P_grazed;  // g C m^-3, fraction of grazed phytoplankton converted to zooplankton
    
    // Equation 7: Zooplankton biomass flux from mortality
    Type Z_mortality = m_Z * Z_prev * dt;  // g C m^-3, natural zooplankton death
    
    // Update state variables
    // Nutrient dynamics: loss from phytoplankton uptake, gain from mortality and inefficient grazing
    N_pred(i) = N_prev - P_growth + P_mortality + Z_mortality + (Type(1.0) - e) * P_grazed;
    
    // Phytoplankton dynamics: gain from growth, loss from grazing and mortality
    P_pred(i) = P_prev + P_growth - P_grazed - P_mortality;
    
    // Zooplankton dynamics: gain from assimilated grazing, loss from mortality
    Z_pred(i) = Z_prev + Z_growth - Z_mortality;
    
    // Apply smooth lower bounds to prevent negative values
    N_pred(i) = N_pred(i) + eps;  // Ensure nutrients remain positive
    P_pred(i) = P_pred(i) + eps;  // Ensure phytoplankton remain positive
    Z_pred(i) = Z_pred(i) + eps;  // Ensure zooplankton remain positive
  }
  
  // LIKELIHOOD CALCULATION
  Type nll = 0.0;  // Initialize negative log-likelihood
  
  // Add observation likelihoods for all time points (normal distribution)
  for(int i = 0; i < n; i++) {
    nll -= dnorm(N_dat(i), N_pred(i), sigma_N_use, true);  // Nutrient observation likelihood
    nll -= dnorm(P_dat(i), P_pred(i), sigma_P_use, true);  // Phytoplankton observation likelihood
    nll -= dnorm(Z_dat(i), Z_pred(i), sigma_Z_use, true);  // Zooplankton observation likelihood
  }
  
  // PARAMETER CONSTRAINTS (soft penalties for biological realism)
  // Growth rate should be positive and reasonable (0 to 2 day^-1)
  if(r < Type(0.0)) nll += Type(100.0) * (r - Type(0.0)) * (r - Type(0.0));
  if(r > Type(2.0)) nll += Type(100.0) * (r - Type(2.0)) * (r - Type(2.0));
  
  // Half-saturation constants should be positive
  if(K_N < Type(0.0)) nll += Type(100.0) * K_N * K_N;
  if(K_P < Type(0.0)) nll += Type(100.0) * K_P * K_P;
  
  // Grazing rate should be positive and reasonable (0 to 1 day^-1)
  if(g_max < Type(0.0)) nll += Type(100.0) * g_max * g_max;
  if(g_max > Type(1.0)) nll += Type(100.0) * (g_max - Type(1.0)) * (g_max - Type(1.0));
  
  // Assimilation efficiency should be between 0 and 1
  if(e < Type(0.0)) nll += Type(100.0) * e * e;
  if(e > Type(1.0)) nll += Type(100.0) * (e - Type(1.0)) * (e - Type(1.0));
  
  // Mortality rates should be positive and reasonable (0 to 0.5 day^-1)
  if(m_P < Type(0.0)) nll += Type(100.0) * m_P * m_P;
  if(m_P > Type(0.5)) nll += Type(100.0) * (m_P - Type(0.5)) * (m_P - Type(0.5));
  if(m_Z < Type(0.0)) nll += Type(100.0) * m_Z * m_Z;
  if(m_Z > Type(0.5)) nll += Type(100.0) * (m_Z - Type(0.5)) * (m_Z - Type(0.5));
  
  // REPORTING
  REPORT(N_pred);      // Report predicted nutrient concentrations
  REPORT(P_pred);      // Report predicted phytoplankton concentrations
  REPORT(Z_pred);      // Report predicted zooplankton concentrations
  REPORT(sigma_N);     // Report nutrient observation error
  REPORT(sigma_P);     // Report phytoplankton observation error
  REPORT(sigma_Z);     // Report zooplankton observation error
  
  return nll;          // Return negative log-likelihood for minimization
}
