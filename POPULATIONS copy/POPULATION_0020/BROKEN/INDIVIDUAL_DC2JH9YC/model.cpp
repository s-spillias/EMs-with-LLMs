#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------
  // DATA INPUTS
  // ------------------------------------------------------------------------
  
  // Time vector from the data file.
  // The C++ variable 'time_days' is mapped via parameters.json to the data column 'Time (days)'.
  DATA_VECTOR(time_days);
  
  // Observed Nutrient concentration (g C m^-3).
  // The C++ variable 'N_dat' is mapped via parameters.json to the data column 'N_dat (...)'.
  DATA_VECTOR(N_dat);
  
  // Observed Phytoplankton concentration (g C m^-3).
  // The C++ variable 'P_dat' is mapped via parameters.json to the data column 'P_dat (...)'.
  DATA_VECTOR(P_dat);
  
  // Observed Zooplankton concentration (g C m^-3).
  // The C++ variable 'Z_dat' is mapped via parameters.json to the data column 'Z_dat (...)'.
  DATA_VECTOR(Z_dat);
  
  // ------------------------------------------------------------------------
  // MODEL PARAMETERS
  // ------------------------------------------------------------------------
  
  // Parameters are optimized on a log or logit scale to ensure positivity or bounds (0-1).
  
  PARAMETER(log_V_max);   // Log of maximum phytoplankton growth rate (day^-1)
  PARAMETER(log_K_N);     // Log of half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(log_g_max);   // Log of maximum zooplankton grazing rate (day^-1)
  PARAMETER(log_K_P);     // Log of half-saturation constant for grazing (g C m^-3)
  PARAMETER(logit_beta);  // Logit of zooplankton assimilation efficiency (dimensionless)
  PARAMETER(log_m_P);     // Log of phytoplankton linear mortality rate (day^-1)
  PARAMETER(log_l_Z);     // Log of zooplankton metabolic loss rate (day^-1)
  PARAMETER(log_m_Z);     // Log of zooplankton quadratic mortality rate ((g C m^-3)^-1 day^-1)
  
  // Standard deviations for the lognormal observation error model
  PARAMETER(log_sigma_N); // Log of standard deviation for Nutrient observations
  PARAMETER(log_sigma_P); // Log of standard deviation for Phytoplankton observations
  PARAMETER(log_sigma_Z); // Log of standard deviation for Zooplankton observations
  
  // ------------------------------------------------------------------------
  // TRANSFORM PARAMETERS
  // ------------------------------------------------------------------------
  
  // Transform parameters from log/logit scale to their natural scale.
  Type V_max = exp(log_V_max);
  Type K_N = exp(log_K_N);
  Type g_max = exp(log_g_max);
  Type K_P = exp(log_K_P);
  Type beta = Type(1.0) / (Type(1.0) + exp(-logit_beta)); // Inverse logit
  Type m_P = exp(log_m_P);
  Type l_Z = exp(log_l_Z);
  Type m_Z = exp(log_m_Z);
  Type sigma_N = exp(log_sigma_N);
  Type sigma_P = exp(log_sigma_P);
  Type sigma_Z = exp(log_sigma_Z);
  
  // ------------------------------------------------------------------------
  // MODEL EQUATIONS
  // ------------------------------------------------------------------------
  
  // The model is a set of ordinary differential equations (ODEs) describing the rate of change
  // for Nutrient (N), Phytoplankton (P), and Zooplankton (Z) concentrations.
  
  // 1. dN/dt = -Uptake + NutrientRecycling
  //    Nutrient concentration changes based on consumption by phytoplankton (Uptake) and
  //    replenishment from zooplankton excretion, and mortality of both plankton groups.
  
  // 2. dP/dt = Uptake - Grazing - PhytoplanktonMortality
  //    Phytoplankton concentration increases with nutrient uptake and decreases due to
  //    zooplankton grazing and natural mortality.
  
  // 3. dZ/dt = AssimilatedGrazing - ZooplanktonLosses
  //    Zooplankton concentration increases by assimilating a fraction of the grazed
  //    phytoplankton, and decreases due to metabolic losses and mortality.
  
  // ------------------------------------------------------------------------
  // MODEL PREDICTIONS (FORWARD SIMULATION)
  // ------------------------------------------------------------------------
  
  int n_obs = time_days.size(); // Number of observations
  
  // Vectors to store model predictions
  vector<Type> N_pred(n_obs);
  vector<Type> P_pred(n_obs);
  vector<Type> Z_pred(n_obs);
  
  // Initialize predictions with the first data point
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);
  
  // Use a forward Euler method to integrate the ODEs over time
  for (int i = 1; i < n_obs; ++i) {
    Type dt = time_days(i) - time_days(i-1); // Time step
    
    // State variables from the previous time step
    Type N_prev = N_pred(i-1);
    Type P_prev = P_pred(i-1);
    Type Z_prev = Z_pred(i-1);
    
    // Ecological process rates
    // Add a small constant (1e-8) to denominators to prevent division by zero.
    
    // Phytoplankton nutrient uptake (Michaelis-Menten kinetics)
    Type uptake = V_max * (N_prev / (K_N + N_prev + Type(1e-8))) * P_prev;
    
    // Zooplankton grazing on phytoplankton (Holling Type III functional response)
    Type grazing = g_max * (pow(P_prev, 2.0) / (pow(K_P, 2.0) + pow(P_prev, 2.0) + Type(1e-8))) * Z_prev;
    
    // Calculate the change (dN, dP, dZ) for each state variable
    Type dN = -uptake + (Type(1.0) - beta) * grazing + l_Z * Z_prev + m_Z * pow(Z_prev, 2.0) + m_P * P_prev;
    Type dP = uptake - grazing - m_P * P_prev;
    Type dZ = beta * grazing - l_Z * Z_prev - m_Z * pow(Z_prev, 2.0);
    
    // Update predictions using the forward Euler step
    N_pred(i) = N_prev + dN * dt;
    P_pred(i) = P_prev + dP * dt;
    Z_pred(i) = Z_prev + dZ * dt;
    
    // Ensure predictions remain positive using a smooth approximation of max(0, x)
    // This avoids hard cutoffs and improves numerical stability.
    N_pred(i) = CppAD::CondExpGt(N_pred(i), Type(0.0), N_pred(i), Type(1e-8));
    P_pred(i) = CppAD::CondExpGt(P_pred(i), Type(0.0), P_pred(i), Type(1e-8));
    Z_pred(i) = CppAD::CondExpGt(Z_pred(i), Type(0.0), Z_pred(i), Type(1e-8));
  }
  
  // ------------------------------------------------------------------------
  // LIKELIHOOD CALCULATION
  // ------------------------------------------------------------------------
  
  Type nll = 0.0; // Initialize negative log-likelihood
  
  // Use a lognormal distribution for observation errors, as concentrations are strictly positive.
  // Add a fixed minimum standard deviation to prevent issues with very small data values.
  Type sigma_N_eff = sqrt(pow(sigma_N, 2.0) + pow(Type(0.01), 2.0));
  Type sigma_P_eff = sqrt(pow(sigma_P, 2.0) + pow(Type(0.01), 2.0));
  Type sigma_Z_eff = sqrt(pow(sigma_Z, 2.0) + pow(Type(0.01), 2.0));
  
  for (int i = 0; i < n_obs; ++i) {
    // The 'true' argument specifies that dnorm should return the log-probability.
    nll -= dnorm(log(N_dat(i)), log(N_pred(i)), sigma_N_eff, true);
    nll -= dnorm(log(P_dat(i)), log(P_pred(i)), sigma_P_eff, true);
    nll -= dnorm(log(Z_dat(i)), log(Z_pred(i)), sigma_Z_eff, true);
  }
  
  // ------------------------------------------------------------------------
  // REPORTING SECTION
  // ------------------------------------------------------------------------
  
  // Report transformed parameters
  REPORT(V_max);
  REPORT(K_N);
  REPORT(g_max);
  REPORT(K_P);
  REPORT(beta);
  REPORT(m_P);
  REPORT(l_Z);
  REPORT(m_Z);
  REPORT(sigma_N);
  REPORT(sigma_P);
  REPORT(sigma_Z);
  
  // Report predicted time series
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  
  return nll;
}
