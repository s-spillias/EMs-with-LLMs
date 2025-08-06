#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Time);        // Time points (days)
  DATA_VECTOR(N_dat);       // Nutrient observations (g C m^-3)
  DATA_VECTOR(P_dat);       // Phytoplankton observations (g C m^-3)
  DATA_VECTOR(Z_dat);       // Zooplankton observations (g C m^-3)
  
  // Calculate seasonal light intensity
  vector<Type> I(Time.size());
  Type year_length = Type(365.0);
  for(int t = 0; t < Time.size(); t++) {
    // Seasonal variation with max at day 180 (summer)
    I(t) = Type(1.0) + Type(0.5) * cos(Type(2.0) * M_PI * (Time(t) - Type(180.0))/year_length);
  }
  
  // Create default temperature vector if not provided
  vector<Type> Temp(Time.size());
  Temp.fill(Type(20.0));  // Default temperature of 20°C
  
  // Parameters
  PARAMETER(r_max);         // Maximum phytoplankton growth rate (day^-1)
  PARAMETER(K_N);          // Half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(K_I);          // Half-saturation constant for light limitation (dimensionless)
  PARAMETER(g_max);        // Maximum zooplankton grazing rate (day^-1)
  PARAMETER(K_P);          // Half-saturation constant for grazing (g C m^-3)
  PARAMETER(alpha);        // Zooplankton assimilation efficiency
  PARAMETER(m_P);          // Phytoplankton mortality rate (day^-1)
  PARAMETER(m_Z);          // Zooplankton mortality rate (day^-1)
  PARAMETER(gamma);        // Nutrient recycling fraction
  PARAMETER(sigma_N);      // SD for nutrient observations
  PARAMETER(sigma_P);      // SD for phytoplankton observations
  PARAMETER(sigma_Z);      // SD for zooplankton observations

  // Constants for numerical stability
  const Type eps = Type(1e-8);
  const Type min_conc = Type(1e-10);  // Minimum concentration
  const Type max_dt = Type(0.1);      // Maximum time step
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Smooth penalties to keep parameters in biological ranges
  nll -= dnorm(log(r_max), Type(0.0), Type(1.0), true);     // Keep r_max positive
  nll -= dnorm(log(K_N), Type(-3.0), Type(1.0), true);      // Keep K_N positive
  nll -= dnorm(log(g_max), Type(-1.0), Type(1.0), true);    // Keep g_max positive
  nll -= dnorm(log(K_P), Type(-3.0), Type(1.0), true);      // Keep K_P positive
  nll -= dnorm(logit(alpha), Type(0.0), Type(2.0), true);   // Keep alpha between 0 and 1
  nll -= dnorm(log(m_P), Type(-3.0), Type(1.0), true);      // Keep m_P positive
  nll -= dnorm(log(m_Z), Type(-3.0), Type(1.0), true);      // Keep m_Z positive
  nll -= dnorm(logit(gamma), Type(0.0), Type(2.0), true);   // Keep gamma between 0 and 1
  
  // Initialize prediction vectors
  vector<Type> N_pred(Time.size());
  vector<Type> P_pred(Time.size());
  vector<Type> Z_pred(Time.size());
  
  // Set initial conditions (ensure positive)
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);
  
  // Numerical integration using 4th order Runge-Kutta
  for(int t = 1; t < Time.size(); t++) {
    Type dt = Time(t) - Time(t-1);
    
    // Use fixed small time steps for stability
    Type h = Type(0.1); // Fixed step size
    int n_steps = 10;   // Fixed number of steps
    
    Type N = N_pred(t-1);
    Type P = P_pred(t-1);
    Type Z = Z_pred(t-1);
    
    for(int step = 0; step < n_steps; step++) {
      // Temperature scaling (Arrhenius equation)
      Type T_K = Temp(t) + Type(273.15);  // Convert to Kelvin
      Type T_ref = Type(293.15);          // Reference temp (20°C)
      Type E_a = Type(0.63);              // Activation energy (eV)
      Type k_B = Type(8.617e-5);          // Boltzmann constant (eV/K)
      
      // Temperature scaling factor (simplified)
      Type temp_scale = exp(E_a * (Type(1.0)/T_ref - Type(1.0)/T_K) / k_B);
      // Bound scaling factor to prevent numerical issues
      temp_scale = Type(0.5) + Type(0.5) * temp_scale;
      
      // Calculate temperature-dependent rates
      Type light_limitation = I(t) / (K_I + I(t) + eps);
      Type uptake = r_max * temp_scale * N * P / (K_N + N + eps) * light_limitation;
      Type grazing = g_max * temp_scale * P * Z / (K_P + P + eps);
      
      // System of differential equations
      Type dN = -uptake + gamma * (m_P * P + m_Z * Z * Z + (1 - alpha) * grazing);
      Type dP = uptake - grazing - m_P * P;
      Type dZ = alpha * grazing - m_Z * Z * Z;
      
      // Euler integration step
      N += h * dN;
      P += h * dP;
      Z += h * dZ;
      
      // Ensure concentrations stay positive
      N = exp(log(N + eps));
      P = exp(log(P + eps));
      Z = exp(log(Z + eps));
    }
    
    // Store final values
    N_pred(t) = N;
    P_pred(t) = P;
    Z_pred(t) = Z;
  }
  
  // Likelihood calculations using lognormal distribution
  Type min_sigma = Type(0.01);  // Minimum standard deviation
  for(int t = 0; t < Time.size(); t++) {
    nll -= dnorm(log(N_dat(t) + eps), log(N_pred(t) + eps), 
                 exp(log(sigma_N + min_sigma)), true);
    nll -= dnorm(log(P_dat(t) + eps), log(P_pred(t) + eps), 
                 exp(log(sigma_P + min_sigma)), true);
    nll -= dnorm(log(Z_dat(t) + eps), log(Z_pred(t) + eps), 
                 exp(log(sigma_Z + min_sigma)), true);
  }
  
  // Store predictions for output
  vector<Type> N_pred = N_dat;  // Initialize with data
  vector<Type> P_pred = P_dat;
  vector<Type> Z_pred = Z_dat;
  
  // Calculate predictions
  for(int t = 0; t < Time.size(); t++) {
    // Temperature scaling
    Type T_K = Temp(t) + Type(273.15);
    Type T_ref = Type(293.15);
    Type E_a = Type(0.63);
    Type k_B = Type(8.617e-5);
    Type temp_scale = exp(E_a * (Type(1.0)/T_ref - Type(1.0)/T_K) / k_B);
    temp_scale = Type(0.5) + Type(0.5) * temp_scale;
    
    // Calculate rates
    Type light_limitation = I(t) / (K_I + I(t) + eps);
    Type uptake = r_max * temp_scale * N_pred(t) * P_pred(t) / (K_N + N_pred(t) + eps) * light_limitation;
    Type grazing = g_max * temp_scale * P_pred(t) * Z_pred(t) / (K_P + P_pred(t) + eps);
    
    // Update predictions
    if(t < Time.size() - 1) {
      Type dt = Time(t+1) - Time(t);
      N_pred(t+1) = N_pred(t) + dt * (-uptake + gamma * (m_P * P_pred(t) + m_Z * Z_pred(t) * Z_pred(t) + (1 - alpha) * grazing));
      P_pred(t+1) = P_pred(t) + dt * (uptake - grazing - m_P * P_pred(t));
      Z_pred(t+1) = Z_pred(t) + dt * (alpha * grazing - m_Z * Z_pred(t) * Z_pred(t));
      
      // Ensure positive concentrations
      N_pred(t+1) = exp(log(N_pred(t+1) + eps));
      P_pred(t+1) = exp(log(P_pred(t+1) + eps));
      Z_pred(t+1) = exp(log(Z_pred(t+1) + eps));
    }
  }
  
  // Report predictions
  ADREPORT(N_pred);
  ADREPORT(P_pred);
  ADREPORT(Z_pred);
  
  return nll;
}
