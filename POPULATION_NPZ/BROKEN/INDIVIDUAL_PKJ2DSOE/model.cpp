#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Time);        // Time points (days)
  DATA_VECTOR(N_dat);       // Nutrient observations (g C m^-3)
  DATA_VECTOR(P_dat);       // Phytoplankton observations (g C m^-3)
  DATA_VECTOR(Z_dat);       // Zooplankton observations (g C m^-3)
  
  // Calculate seasonal light limitation
  vector<Type> I_rel(Time.size());
  for(int t = 0; t < Time.size(); t++) {
    // Assume annual cycle with peak at day 180
    Type day_of_year = fmod(Time(t), Type(365.0));
    I_rel(t) = Type(0.5) + Type(0.4) * cos((day_of_year - Type(180.0)) * Type(2.0 * M_PI) / Type(365.0));
  }
  
  // Create default temperature vector if not provided
  vector<Type> Temp(Time.size());
  Temp.fill(Type(20.0));  // Default temperature of 20°C
  
  // Parameters
  PARAMETER(r_max);         // Maximum phytoplankton growth rate (day^-1)
  PARAMETER(K_N);          // Half-saturation constant for nutrient uptake (g C m^-3)
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
  
  // Declare prediction vectors
  vector<Type> N_pred(Time.size());
  vector<Type> P_pred(Time.size());
  vector<Type> Z_pred(Time.size());
  
  // Set initial conditions from data
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);
  
  // Calculate predictions for all time points
  for(int t = 1; t < Time.size(); t++) {
    Type dt = Time(t) - Time(t-1);
    
    // Calculate rates
    Type temp_scale = Type(1.0); // Default temperature scaling
    Type light_effect = I_rel(t);
    
    // Update state variables
    Type uptake = r_max * temp_scale * light_effect * N_pred(t-1) * P_pred(t-1) / (K_N + N_pred(t-1));
    Type grazing = g_max * temp_scale * P_pred(t-1) * Z_pred(t-1) / (K_P + P_pred(t-1));
    
    // Calculate changes
    Type dN = -uptake + gamma * (m_P * P_pred(t-1) + m_Z * Z_pred(t-1) * Z_pred(t-1) + (1 - alpha) * grazing);
    Type dP = uptake - grazing - m_P * P_pred(t-1);
    Type dZ = alpha * grazing - m_Z * Z_pred(t-1) * Z_pred(t-1);
    
    // Update predictions
    N_pred(t) = N_pred(t-1) + dt * dN;
    P_pred(t) = P_pred(t-1) + dt * dP;
    Z_pred(t) = Z_pred(t-1) + dt * dZ;
    
    // Ensure positive concentrations
    N_pred(t) = exp(log(N_pred(t) + eps));
    P_pred(t) = exp(log(P_pred(t) + eps));
    Z_pred(t) = exp(log(Z_pred(t) + eps));
  }
  
  // Report predictions
  ADREPORT(N_pred);
  ADREPORT(P_pred);
  ADREPORT(Z_pred);
  
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
      
      // Calculate temperature and light dependent rates
      Type light_limitation = I_rel(t) * N / (K_N + N + eps);
      Type uptake = r_max * temp_scale * light_limitation * P;
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
  
  // Report predictions
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  
  return nll;
}
