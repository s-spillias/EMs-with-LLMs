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
  PARAMETER(log_K_P);                   // Log half-saturation constant for zooplankton grazing (g C m^-3)
  PARAMETER(log_m_P);                   // Log phytoplankton mortality rate (day^-1)
  PARAMETER(log_m_Z);                   // Log zooplankton mortality rate (day^-1)
  PARAMETER(logit_e_P);                 // Logit zooplankton assimilation efficiency (dimensionless)
  PARAMETER(logit_gamma);               // Logit nutrient recycling efficiency (dimensionless)
  PARAMETER(log_N_in);                  // Log external nutrient input rate (g C m^-3 day^-1)
  PARAMETER(log_sigma_N);               // Log observation error for nutrients
  PARAMETER(log_sigma_P);               // Log observation error for phytoplankton
  PARAMETER(log_sigma_Z);               // Log observation error for zooplankton
  
  // Transform parameters to natural scale with biological bounds
  Type r = exp(log_r);                  // Maximum phytoplankton growth rate (day^-1), from literature on marine phytoplankton
  Type K_N = exp(log_K_N);              // Half-saturation for nutrient uptake (g C m^-3), typical oceanic values
  Type g_max = exp(log_g_max);          // Maximum zooplankton grazing rate (day^-1), from copepod feeding studies
  Type K_P = exp(log_K_P);              // Half-saturation for grazing (g C m^-3), based on prey density thresholds
  Type m_P = exp(log_m_P);              // Phytoplankton mortality rate (day^-1), includes senescence and viral lysis
  Type m_Z = exp(log_m_Z);              // Zooplankton mortality rate (day^-1), includes predation and natural death
  Type e_P = Type(1.0) / (Type(1.0) + exp(-logit_e_P)); // Zooplankton assimilation efficiency (0-1), conversion of food to biomass
  Type gamma = Type(1.0) / (Type(1.0) + exp(-logit_gamma)); // Nutrient recycling efficiency (0-1), fraction of dead matter remineralized
  Type N_in = exp(log_N_in);            // External nutrient input (g C m^-3 day^-1), from upwelling or atmospheric deposition
  Type sigma_N = exp(log_sigma_N);      // Observation error standard deviation for nutrients
  Type sigma_P = exp(log_sigma_P);      // Observation error standard deviation for phytoplankton  
  Type sigma_Z = exp(log_sigma_Z);      // Observation error standard deviation for zooplankton
  
  // Add smooth penalties to keep parameters within biological ranges
  Type nll = Type(0.0);                 // Initialize negative log-likelihood
  
  // Soft bounds using quadratic penalties
  nll -= dnorm(log_r, log(Type(1.0)), Type(1.0), true);           // Penalize r far from ~1 day^-1
  nll -= dnorm(log_K_N, log(Type(0.1)), Type(1.0), true);         // Penalize K_N far from ~0.1 g C m^-3
  nll -= dnorm(log_g_max, log(Type(0.5)), Type(1.0), true);       // Penalize g_max far from ~0.5 day^-1
  nll -= dnorm(log_K_P, log(Type(0.1)), Type(1.0), true);         // Penalize K_P far from ~0.1 g C m^-3
  nll -= dnorm(log_m_P, log(Type(0.1)), Type(1.0), true);         // Penalize m_P far from ~0.1 day^-1
  nll -= dnorm(log_m_Z, log(Type(0.05)), Type(1.0), true);        // Penalize m_Z far from ~0.05 day^-1
  nll -= dnorm(logit_e_P, Type(0.0), Type(2.0), true);            // Penalize e_P far from ~0.5
  nll -= dnorm(logit_gamma, Type(1.0), Type(2.0), true);          // Penalize gamma far from ~0.73
  nll -= dnorm(log_N_in, log(Type(0.01)), Type(1.0), true);       // Penalize N_in far from ~0.01 g C m^-3 day^-1
  
  int n_obs = Time.size();              // Number of observations
  
  // Initialize state variables with first observations
  vector<Type> N_pred(n_obs);           // Predicted nutrient concentrations
  vector<Type> P_pred(n_obs);           // Predicted phytoplankton concentrations
  vector<Type> Z_pred(n_obs);           // Predicted zooplankton concentrations
  
  // Set initial conditions from first observation
  N_pred(0) = N_dat(0);                 // Initial nutrient concentration from data
  P_pred(0) = P_dat(0);                 // Initial phytoplankton concentration from data
  Z_pred(0) = Z_dat(0);                 // Initial zooplankton concentration from data
  
  // Numerical integration using Euler method
  for(int i = 1; i < n_obs; i++) {
    Type dt = Time(i) - Time(i-1);      // Time step size (days)
    
    // Previous time step values (avoid data leakage)
    Type N_prev = N_pred(i-1);          // Nutrient concentration at previous time step
    Type P_prev = P_pred(i-1);          // Phytoplankton concentration at previous time step  
    Type Z_prev = Z_pred(i-1);          // Zooplankton concentration at previous time step
    
    // Add small constants to prevent division by zero
    Type N_safe = N_prev + Type(1e-8);  // Numerically stable nutrient concentration
    Type P_safe = P_prev + Type(1e-8);  // Numerically stable phytoplankton concentration
    Type Z_safe = Z_prev + Type(1e-8);  // Numerically stable zooplankton concentration
    
    // Equation 1: Nutrient-limited phytoplankton growth (Michaelis-Menten kinetics)
    Type phyto_growth = r * (N_safe / (K_N + N_safe)) * P_safe;
    
    // Equation 2: Zooplankton grazing on phytoplankton (Type II functional response)
    Type grazing = g_max * (P_safe / (K_P + P_safe)) * Z_safe;
    
    // Equation 3: Phytoplankton natural mortality
    Type phyto_mortality = m_P * P_safe;
    
    // Equation 4: Zooplankton natural mortality  
    Type zoo_mortality = m_Z * Z_safe;
    
    // Equation 5: Nutrient recycling from dead organic matter
    Type nutrient_recycling = gamma * (phyto_mortality + zoo_mortality);
    
    // Equation 6: Zooplankton growth from assimilated phytoplankton
    Type zoo_growth = e_P * grazing;
    
    // System of differential equations:
    // dN/dt = -phyto_growth + nutrient_recycling + N_in
    Type dN_dt = -phyto_growth + nutrient_recycling + N_in;
    
    // dP/dt = phyto_growth - grazing - phyto_mortality  
    Type dP_dt = phyto_growth - grazing - phyto_mortality;
    
    // dZ/dt = zoo_growth - zoo_mortality
    Type dZ_dt = zoo_growth - zoo_mortality;
    
    // Update state variables using Euler integration
    Type N_new = N_prev + dt * dN_dt;    // New nutrient concentration
    Type P_new = P_prev + dt * dP_dt;    // New phytoplankton concentration
    Type Z_new = Z_prev + dt * dZ_dt;    // New zooplankton concentration
    
    // Use conditional expressions to ensure non-negative values
    Type min_val = Type(1e-8);           // Minimum allowed concentration
    N_pred(i) = CppAD::CondExpGe(N_new, min_val, N_new, min_val);  // Ensure non-negative nutrients
    P_pred(i) = CppAD::CondExpGe(P_new, min_val, P_new, min_val);  // Ensure non-negative phytoplankton
    Z_pred(i) = CppAD::CondExpGe(Z_new, min_val, Z_new, min_val);  // Ensure non-negative zooplankton
  }
  
  // Calculate likelihood for all observations
  Type min_sigma = Type(0.001);         // Minimum observation error to prevent numerical issues
  
  // Ensure minimum sigma values using conditional expressions
  Type sigma_N_safe = CppAD::CondExpGe(sigma_N, min_sigma, sigma_N, min_sigma);  // Numerically stable error for nutrients
  Type sigma_P_safe = CppAD::CondExpGe(sigma_P, min_sigma, sigma_P, min_sigma);  // Numerically stable error for phytoplankton
  Type sigma_Z_safe = CppAD::CondExpGe(sigma_Z, min_sigma, sigma_Z, min_sigma);  // Numerically stable error for zooplankton
  
  // Add observation likelihoods for all data points
  for(int i = 0; i < n_obs; i++) {
    // Use lognormal distribution for strictly positive concentrations
    Type log_N_obs = log(N_dat(i) + Type(1e-8));     // Log observed nutrients with small constant
    Type log_P_obs = log(P_dat(i) + Type(1e-8));     // Log observed phytoplankton with small constant
    Type log_Z_obs = log(Z_dat(i) + Type(1e-8));     // Log observed zooplankton with small constant
    
    Type log_N_pred_safe = log(N_pred(i) + Type(1e-8));  // Log predicted nutrients with small constant
    Type log_P_pred_safe = log(P_pred(i) + Type(1e-8));  // Log predicted phytoplankton with small constant
    Type log_Z_pred_safe = log(Z_pred(i) + Type(1e-8));  // Log predicted zooplankton with small constant
    
    // Check for valid values before adding to likelihood
    Type nll_N = dnorm(log_N_obs, log_N_pred_safe, sigma_N_safe, true);
    Type nll_P = dnorm(log_P_obs, log_P_pred_safe, sigma_P_safe, true);
    Type nll_Z = dnorm(log_Z_obs, log_Z_pred_safe, sigma_Z_safe, true);
    
    // Only add finite likelihood contributions
    nll -= CppAD::CondExpGe(nll_N, Type(-1e10), nll_N, Type(0.0));
    nll -= CppAD::CondExpGe(nll_P, Type(-1e10), nll_P, Type(0.0));
    nll -= CppAD::CondExpGe(nll_Z, Type(-1e10), nll_Z, Type(0.0));
  }
  
  // Add penalty if nll becomes infinite or NaN
  nll = CppAD::CondExpGe(nll, Type(1e10), Type(1e10), nll);
  
  // Report predicted values for output
  REPORT(N_pred);                       // Report predicted nutrient concentrations
  REPORT(P_pred);                       // Report predicted phytoplankton concentrations
  REPORT(Z_pred);                       // Report predicted zooplankton concentrations
  
  return nll;                           // Return negative log-likelihood for minimization
}
