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
  PARAMETER(v_max);         // Maximum nutrient uptake rate (day^-1) 
  PARAMETER(K_N);          // Half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(Q_min);        // Minimum internal nutrient quota (g N/g C)
  PARAMETER(Q_max);        // Maximum internal nutrient quota (g N/g C)
  PARAMETER(g_max);        // Maximum zooplankton grazing rate (day^-1)
  PARAMETER(K_P);          // Half-saturation constant for grazing (g C m^-3)
  PARAMETER(alpha_base);   // Baseline zooplankton assimilation efficiency
  PARAMETER(alpha_max);    // Maximum additional assimilation efficiency
  PARAMETER(K_alpha);      // Half-saturation for nutrient-dependent efficiency
  PARAMETER(m_P);          // Base phytoplankton mortality rate (day^-1)
  PARAMETER(m_P_N);        // Nutrient-dependent phytoplankton mortality (day^-1)
  PARAMETER(s_P);          // Base phytoplankton sinking rate (day^-1)
  PARAMETER(s_P_max);      // Maximum additional nutrient-stress sinking rate (day^-1)
  PARAMETER(m_Z);          // Base zooplankton mortality rate (day^-1)
  PARAMETER(m_Z_N);        // Nutrient-dependent zooplankton mortality (day^-1)
  PARAMETER(r_D);          // Detritus remineralization rate (day^-1)
  PARAMETER(sigma_N);      // SD for nutrient observations
  PARAMETER(sigma_P);      // SD for phytoplankton observations
  PARAMETER(sigma_Z);      // SD for zooplankton observations
  PARAMETER(I_opt);        // Optimal light intensity
  PARAMETER(beta);         // Light attenuation coefficient
  PARAMETER(k_w);         // Light attenuation coefficient due to phytoplankton self-shading
  PARAMETER(E_p);         // Activation energy for photosynthetic efficiency (eV)
  PARAMETER(theta_P);     // Temperature sensitivity of grazing selectivity
  PARAMETER(eta_max);     // Maximum nutrient uptake efficiency multiplier
  PARAMETER(k_eta);       // Steepness of uptake efficiency response
  PARAMETER(N_crit);      // Critical nutrient concentration for efficiency switch
  PARAMETER(eta_base);    // Baseline nutrient uptake efficiency

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
  vector<Type> Q_pred(Time.size());  // Internal nutrient quota
  Type init_quota = (Q_min + Q_max) / Type(2.0);
  for(int i = 0; i < Time.size(); i++) {
    Q_pred(i) = init_quota;
  }
  
  // Initial conditions (ensure positive)
  N_pred(0) = exp(log(N_dat(0) + eps));
  D_pred(0) = Type(0.1); // Initial detritus concentration
  P_pred(0) = exp(log(P_dat(0) + eps));
  Z_pred(0) = exp(log(Z_dat(0) + eps));
  Q_pred(0) = (Q_min + Q_max) / Type(2.0); // Start at middle of quota range
  
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
      
      // Temperature scaling factor (simplified)
      // General metabolic temperature scaling
      Type temp_scale = exp(E_a * (Type(1.0)/T_ref - Type(1.0)/T_K) / k_B);
      // Photosynthesis-specific temperature scaling
      Type photo_eff = exp(E_p * (Type(1.0)/T_ref - Type(1.0)/T_K) / k_B);
      // Bound scaling factors to prevent numerical issues
      temp_scale = Type(0.5) + Type(0.5) * temp_scale;
      photo_eff = Type(0.5) + Type(0.5) * photo_eff;
      
      // Calculate seasonal light intensity 
      Type season = Type(0.6) * sin(Type(2.0) * M_PI * Time(t) / Type(365.0));
      Type I = I_opt * (Type(1.0) + season);
      
      // Light limitation factor with self-shading
      Type I_effective = I * exp(-k_w * P);  // Reduce light based on phytoplankton density
      Type light_limitation = (I_effective/I_opt) * exp(Type(1.0) - I_effective/I_opt);
      
      // Temperature-dependent grazing selectivity
      Type K_P_T = K_P * (Type(1.0) + theta_P * (temp_scale - Type(1.0)));
      
      // Get current quota with safe bounds
      Type Q_current = Q_pred(t-1);
      Q_current = Q_min + (Q_max - Q_min) / (Type(1.0) + exp(-Type(10.0) * (Q_current - (Q_min + Q_max)/Type(2.0))));
      
      // Uptake regulation with smooth sigmoid bounds
      Type uptake_regulation = Type(1.0) / (Type(1.0) + exp(Type(10.0) * (Q_current - Q_max)/(Q_max - Q_min)));
      
      // Calculate uptake with numerical safeguards
      Type uptake = v_max * temp_scale * uptake_regulation * N * P / (K_N + N + eps);
      uptake = CppAD::CondExpLe(uptake, Type(0.0), Type(0.0), uptake);
      
      // Growth limitation based on quota
      Type quota_limitation = (Q_current - Q_min) / (Q_current + eps);
      quota_limitation = CppAD::CondExpGe(quota_limitation, Type(1.0), Type(1.0),
                        CppAD::CondExpLe(quota_limitation, Type(0.0), Type(0.0), quota_limitation));
      
      // Calculate growth with safeguards
      Type growth = r_max * temp_scale * photo_eff * light_limitation * quota_limitation * P;
      growth = CppAD::CondExpLe(growth, Type(0.0), Type(0.0), growth);
      Type grazing = g_max * temp_scale * P * Z / (K_P_T + P + eps);
      
      // Detritus remineralization (temperature dependent)
      Type remin = r_D * temp_scale * D_pred(t-1);
      
      // System of differential equations
      Type dN = -uptake + remin;
      
      // Update quota with continuous safeguards
      Type dQ = Type(0.0);
      Type P_safe = P + eps;
      dQ = (uptake - Q_current * growth) / P_safe;
      // Smooth rate limiting using tanh
      dQ = Type(2.0) * tanh(dQ / Type(2.0));
      
      // Calculate stress based on quota
      Type nutrient_stress = m_P_N * CppAD::CondExpGe((Q_max - Q_current) / (Q_max - Q_min + eps), 
                            Type(1.0), Type(1.0), (Q_max - Q_current) / (Q_max - Q_min + eps));
      Type sinking = (s_P + s_P_max * (Q_max - Q_pred(t-1)) / (Q_max - Q_min + eps)) * P;
      Type dP = growth - grazing - (m_P + nutrient_stress) * P - sinking;
      // Calculate nutrient-dependent assimilation efficiency
      Type alpha_N = alpha_base + alpha_max * (N / (N + K_alpha + eps));
      // Enhanced zooplankton mortality under nutrient stress
      Type Z_nutrient_stress = m_Z_N * K_N / (N + K_N + eps);
      Type dZ = alpha_N * grazing - (m_Z * Z + Z_nutrient_stress) * Z;
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
      
      // Update quota with smooth sigmoid bounds
      Q_pred(t) = Q_current + h * dQ;
      Q_pred(t) = Q_min + (Q_max - Q_min) / (Type(1.0) + exp(-Type(10.0) * (Q_pred(t) - (Q_min + Q_max)/Type(2.0))));
    }
    
    // Store predictions for this timestep
    N_pred(t) = N;
    P_pred(t) = P;
    Z_pred(t) = Z;
    D_pred(t) = D;
    
    // Calculate predictions for comparison with data
    Type N_prediction = N_pred(t);
    Type P_prediction = P_pred(t);
    Type Z_prediction = Z_pred(t);
  }
  
  // Likelihood calculations using lognormal distribution
  Type min_sigma = Type(0.01);  // Minimum standard deviation
  for(int t = 0; t < Time.size(); t++) {
    if(!R_IsNA(asDouble(N_dat(t)))) {
      nll -= dnorm(log(N_dat(t) + eps), log(N_pred(t) + eps), 
                   exp(log(sigma_N + min_sigma)), true);
    }
    if(!R_IsNA(asDouble(P_dat(t)))) {
      nll -= dnorm(log(P_dat(t) + eps), log(P_pred(t) + eps), 
                   exp(log(sigma_P + min_sigma)), true);
    }
    if(!R_IsNA(asDouble(Z_dat(t)))) {
      nll -= dnorm(log(Z_dat(t) + eps), log(Z_pred(t) + eps), 
                   exp(log(sigma_Z + min_sigma)), true);
    }
  }
  
  // Report predictions
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  REPORT(D_pred);
  REPORT(Q_pred);

  
  return nll;
}
