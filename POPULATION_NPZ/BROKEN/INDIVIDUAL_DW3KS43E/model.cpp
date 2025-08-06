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
  PARAMETER(eta_base);    // Baseline phytoplankton growth efficiency
  PARAMETER(eta_max);     // Maximum additional efficiency under nutrient limitation

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
  nll -= dnorm(log(m_P_N + eps), Type(-3.0), Type(1.0), true);   // Keep m_P_N positive
  nll -= dnorm(log(s_P + eps), Type(-3.0), Type(1.0), true);     // Keep s_P positive
  nll -= dnorm(log(eta_base + eps), Type(0.0), Type(1.0), true); // Keep eta_base positive
  
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
      
      // Calculate nutrient-dependent growth efficiency with bounds
      Type eta_N = (eta_base + eps) + eta_max * K_N / (N + K_N + eps);
      eta_N = CppAD::CondExpGe(eta_N, Type(2.0), Type(2.0), eta_N); // Cap maximum efficiency
      
      // Calculate temperature and light dependent rates with bounded efficiency
      Type uptake = eta_N * r_max * temp_scale * photo_eff * light_limitation * N * P / (K_N + N + eps);
      Type grazing = g_max * temp_scale * P * Z / (K_P_T + P + eps);
      
      // Detritus remineralization (temperature dependent)
      Type remin = r_D * temp_scale * D_pred(t-1);
      
      // System of differential equations
      Type dN = -uptake + remin;
      
      // Enhanced mortality and sinking under nutrient stress with bounds
      Type nutrient_stress = CppAD::CondExpGe((m_P_N + eps) * K_N / (N + K_N + eps), Type(0.5), Type(0.5), (m_P_N + eps) * K_N / (N + K_N + eps));
      // Ensure positive sinking rate 
      Type s_P_effective = s_P + eps;
      Type sinking = (s_P_effective + s_P_max * K_N / (N + K_N + eps)) * P;
      Type dP = uptake - grazing - (m_P + nutrient_stress) * P - sinking;
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
