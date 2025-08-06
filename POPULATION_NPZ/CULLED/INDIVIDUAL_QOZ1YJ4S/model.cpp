#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Time);        // Time points (days)
  DATA_VECTOR(N_dat);       // Nutrient observations (g C m^-3)
  DATA_VECTOR(P_dat);       // Phytoplankton observations (g C m^-3)
  DATA_VECTOR(Z_dat);       // Zooplankton observations (g C m^-3)
  
  // Create default temperature vector if not provided
  vector<Type> Temp(Time.size());
  Temp.fill(Type(20.0));  // Default temperature of 20°C
  
  // Parameters
  PARAMETER(r_max);         // Maximum phytoplankton growth rate (day^-1)
  PARAMETER(K_N);          // Half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(g_max);        // Maximum zooplankton grazing rate (day^-1)
  PARAMETER(K_P);          // Half-saturation constant for grazing (g C m^-3)
  PARAMETER(alpha_base);   // Baseline zooplankton assimilation efficiency
  PARAMETER(alpha_max);    // Maximum additional assimilation efficiency
  PARAMETER(K_alpha);      // Half-saturation for nutrient-dependent efficiency
  PARAMETER(m_P);          // Phytoplankton mortality rate (day^-1)
  PARAMETER(m_Z);          // Zooplankton mortality rate (day^-1)
  PARAMETER(r_D);          // Detritus remineralization rate (day^-1)
  PARAMETER(sigma_N);      // SD for nutrient observations
  PARAMETER(sigma_P);      // SD for phytoplankton observations
  PARAMETER(sigma_Z);      // SD for zooplankton observations
  PARAMETER(I_opt);        // Optimal light intensity
  PARAMETER(beta);         // Light attenuation coefficient
  PARAMETER(m_P_N);       // Nutrient-dependent mortality coefficient
  PARAMETER(E_grazing);   // Activation energy for grazing efficiency (eV)
  PARAMETER(eta_max);     // Maximum nutrient utilization efficiency

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
  nll -= dnorm(logit(alpha_base), Type(0.0), Type(2.0), true);   // Keep alpha_base between 0 and 1
  nll -= dnorm(logit(alpha_max), Type(0.0), Type(2.0), true);    // Keep alpha_max between 0 and 1
  nll -= dnorm(log(K_alpha), Type(-3.0), Type(1.0), true);       // Keep K_alpha positive
  nll -= dnorm(log(m_P), Type(-3.0), Type(1.0), true);      // Keep m_P positive
  nll -= dnorm(log(m_Z), Type(-3.0), Type(1.0), true);      // Keep m_Z positive
  nll -= dnorm(log(r_D), Type(-3.0), Type(1.0), true);      // Keep r_D positive
  
  // Vectors to store predictions
  vector<Type> N_pred(Time.size());
  vector<Type> P_pred(Time.size());
  vector<Type> Z_pred(Time.size());
  vector<Type> D_pred(Time.size());
  
  // Initial conditions (ensure positive)
  N_pred(0) = exp(log(N_dat(0) + eps));
  D_pred(0) = Type(0.1); // Initial detritus concentration
  P_pred(0) = exp(log(P_dat(0) + eps));
  Z_pred(0) = exp(log(Z_dat(0) + eps));
  
  // Numerical integration using 4th order Runge-Kutta
  for(int t = 1; t < Time.size(); t++) {
    Type dt = Time(t) - Time(t-1);
    
    // Use fixed small time steps for stability
    Type h = Type(0.1); // Fixed step size
    int n_steps = 10;   // Fixed number of steps
    
    Type N = N_pred(t-1);
    Type P = P_pred(t-1);
    Type Z = Z_pred(t-1);
    Type D = D_pred(t-1);
    
    for(int step = 0; step < n_steps; step++) {
      // Temperature scaling (Arrhenius equation)
      Type T_K = Temp(t) + Type(273.15);  // Convert to Kelvin
      Type T_ref = Type(293.15);          // Reference temp (20°C)
      Type E_a = Type(0.63);              // Activation energy (eV)
      Type k_B = Type(8.617e-5);          // Boltzmann constant (eV/K)
      
      // Temperature scaling for general metabolism
      Type temp_scale = exp(E_a * (Type(1.0)/T_ref - Type(1.0)/T_K) / k_B);
      // Bound scaling factor to prevent numerical issues
      temp_scale = Type(0.5) + Type(0.5) * temp_scale;
      
      // Temperature-dependent grazing efficiency
      Type E_g = E_grazing; // Different activation energy for grazing
      Type graze_eff = exp(E_g * (Type(1.0)/T_ref - Type(1.0)/T_K) / k_B);
      graze_eff = Type(0.6) + Type(0.4) * graze_eff; // Bound between 0.6 and 1.0
      
      // Calculate seasonal light intensity 
      Type season = Type(0.6) * sin(Type(2.0) * M_PI * Time(t) / Type(365.0));
      Type I = I_opt * (Type(1.0) + season);
      
      // Light limitation factor (simplified Steele formulation)
      Type light_limitation = (I/I_opt) * exp(Type(1.0) - I/I_opt);
      
      // Calculate nutrient-dependent growth efficiency
      Type eta = eta_max * (N * N) / ((K_N + N) * (K_N + N) + eps);
      
      // Calculate temperature, light, and efficiency dependent rates
      Type uptake = r_max * temp_scale * light_limitation * eta * N * P / (K_N + N + eps);
      Type grazing = g_max * temp_scale * graze_eff * P * Z / (K_P + P + eps);
      
      // Detritus remineralization (temperature dependent)
      Type remin = r_D * temp_scale * D_pred(t-1);
      
      // Calculate nutrient-dependent mortality
      Type mort_P = m_P * P + m_P_N * P * (K_N/(N + K_N + eps));
      
      // System of differential equations
      Type dN = -uptake + remin;
      Type dP = uptake - grazing - mort_P;
      // Calculate nutrient-dependent assimilation efficiency
      Type alpha_N = alpha_base + alpha_max * (N / (N + K_alpha + eps));
      Type dZ = alpha_N * grazing - m_Z * Z * Z;
      Type dD = m_P * P + m_Z * Z * Z + (1 - alpha_N) * grazing - remin;
      
      // Euler integration step
      N += h * dN;
      P += h * dP;
      Z += h * dZ;
      
      // Ensure concentrations stay positive
      N = exp(log(N + eps));
      P = exp(log(P + eps));
      Z = exp(log(Z + eps));
      D += h * dD;
      D = exp(log(D + eps));
    }
    
    // Store final values
    N_pred(t) = N;
    P_pred(t) = P;
    Z_pred(t) = Z;
    D_pred(t) = D;
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
  REPORT(D_pred);

  
  return nll;
}
