#include <TMB.hpp>

// TMB template for a Nutrient-Phytoplankton-Zooplankton (NPZ) model
template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------
  // DATA INPUTS
  // ------------------------------------------------------------------------
  
  // Time vector from the data
  DATA_VECTOR(Time); // Time points of observations (days)
  
  // Observed state variables
  DATA_VECTOR(N_dat); // Nutrient concentration observations (g C m^-3)
  DATA_VECTOR(P_dat); // Phytoplankton concentration observations (g C m^-3)
  DATA_VECTOR(Z_dat); // Zooplankton concentration observations (g C m^-3)

  // ------------------------------------------------------------------------
  // PARAMETERS
  // ------------------------------------------------------------------------
  
  // All parameters are estimated in log-space to ensure positivity, or logit-space for [0,1] bounds.
  
  // Phytoplankton growth parameters
  PARAMETER(log_V_max);   // Log of maximum phytoplankton uptake rate (day^-1)
  PARAMETER(log_K_n);     // Log of half-saturation constant for nutrient uptake (g C m^-3)
  
  // Zooplankton grazing parameters
  PARAMETER(log_g_max);   // Log of maximum zooplankton grazing rate (day^-1)
  PARAMETER(log_K_p);     // Log of half-saturation constant for phytoplankton grazing (g C m^-3)
  PARAMETER(logit_beta);  // Logit of zooplankton assimilation efficiency (dimensionless, 0-1)
  
  // Mortality parameters
  PARAMETER(log_m_p);     // Log of phytoplankton mortality rate (day^-1)
  PARAMETER(log_m_z);     // Log of zooplankton mortality rate (day^-1)
  
  // Observation error parameters (log-space standard deviation)
  PARAMETER(log_sd_N);    // Log of standard deviation for Nutrient observations
  PARAMETER(log_sd_P);    // Log of standard deviation for Phytoplankton observations
  PARAMETER(log_sd_Z);    // Log of standard deviation for Zooplankton observations

  // ------------------------------------------------------------------------
  // TRANSFORM PARAMETERS
  // ------------------------------------------------------------------------
  
  // Back-transform parameters from log/logit space to their natural scale
  Type V_max = exp(log_V_max);   // Maximum phytoplankton uptake rate (day^-1). Determined from max growth experiments.
  Type K_n = exp(log_K_n);       // Half-saturation constant for N uptake (g C m^-3). Determined from nutrient uptake experiments.
  Type g_max = exp(log_g_max);   // Maximum zooplankton grazing rate (day^-1). Determined from feeding experiments.
  Type K_p = exp(log_K_p);       // Half-saturation constant for P grazing (g C m^-3). Determined from functional response experiments.
  Type beta = invlogit(logit_beta); // Zooplankton assimilation efficiency (dimensionless). Determined from metabolic studies.
  Type m_p = exp(log_m_p);       // Phytoplankton mortality rate (day^-1). Estimated from population decay in dark.
  Type m_z = exp(log_m_z);       // Zooplankton mortality rate (day^-1). Estimated from cohort survival studies.
  
  Type sd_N = exp(log_sd_N);     // Observation error SD for N. Estimated during model fitting.
  Type sd_P = exp(log_sd_P);     // Observation error SD for P. Estimated during model fitting.
  Type sd_Z = exp(log_sd_Z);     // Observation error SD for Z. Estimated during model fitting.

  // ------------------------------------------------------------------------
  // MODEL SETUP
  // ------------------------------------------------------------------------
  
  int n_obs = Time.size(); // Number of observation time points
  
  // Create vectors to store model predictions
  vector<Type> N_pred(n_obs);
  vector<Type> P_pred(n_obs);
  vector<Type> Z_pred(n_obs);
  
  // Initialize predictions with the first data point
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);

  // ------------------------------------------------------------------------
  // ECOLOGICAL DYNAMICS & PREDICTION
  // ------------------------------------------------------------------------
  
  // This model uses a forward Euler method to integrate the differential equations over time.
  // Loop from the second observation to the end
  for (int i = 1; i < n_obs; ++i) {
    
    // Time step (dt) calculated as the difference between consecutive time points
    Type dt = Time(i) - Time(i-1);
    
    // State variables from the previous time step
    Type N_prev = N_pred(i-1);
    Type P_prev = P_pred(i-1);
    Type Z_prev = Z_pred(i-1);
    
    // --- ECOLOGICAL PROCESS RATES ---
    // These equations describe the flows of carbon between compartments.
    
    // 1. Phytoplankton nutrient uptake (Michaelis-Menten kinetics)
    Type uptake = V_max * (N_prev / (K_n + N_prev + Type(1e-8))) * P_prev;
    
    // 2. Zooplankton grazing on phytoplankton (Holling Type II functional response)
    Type grazing = g_max * (P_prev / (K_p + P_prev + Type(1e-8))) * Z_prev;
    
    // 3. Portion of grazed phytoplankton assimilated by zooplankton
    Type assimilated_grazing = beta * grazing;
    
    // 4. Portion of grazed phytoplankton lost to sloppy feeding/excretion, returned to nutrient pool
    Type sloppy_grazing = (Type(1.0) - beta) * grazing;
    
    // 5. Phytoplankton mortality
    Type phytoplankton_mortality = m_p * P_prev;
    
    // 6. Zooplankton mortality
    Type zooplankton_mortality = m_z * Z_prev;
    
    // --- SYSTEM OF DIFFERENTIAL EQUATIONS ---
    // These equations define the rate of change for each state variable.
    // 1. dN/dt: Nutrient concentration change
    Type dN_dt = -uptake + sloppy_grazing + phytoplankton_mortality + zooplankton_mortality;
    
    // 2. dP/dt: Phytoplankton concentration change
    Type dP_dt = uptake - grazing - phytoplankton_mortality;
    
    // 3. dZ/dt: Zooplankton concentration change
    Type dZ_dt = assimilated_grazing - zooplankton_mortality;
    
    // --- EULER INTEGRATION STEP ---
    // Update the predictions for the current time step `i`.
    N_pred(i) = N_prev + dN_dt * dt;
    P_pred(i) = P_prev + dP_dt * dt;
    Z_pred(i) = Z_prev + dZ_dt * dt;
    
    // --- NUMERICAL STABILITY ---
    // Prevent negative concentrations, which are biologically impossible.
    N_pred(i) = CppAD::CondExpGt(N_pred(i), Type(0.0), N_pred(i), Type(1e-8));
    P_pred(i) = CppAD::CondExpGt(P_pred(i), Type(0.0), P_pred(i), Type(1e-8));
    Z_pred(i) = CppAD::CondExpGt(Z_pred(i), Type(0.0), Z_pred(i), Type(1e-8));
  }

  // ------------------------------------------------------------------------
  // LIKELIHOOD CALCULATION
  // ------------------------------------------------------------------------
  
  Type nll = 0.0; // Initialize negative log-likelihood
  
  // Lognormal likelihood for strictly positive concentration data.
  // This is robust to values spanning multiple orders of magnitude.
  // The 'true' argument specifies that the calculation is on the log scale.
  // The results of log() are explicitly stored in vectors to avoid issues with
  // Eigen expression templates, which can cause compilation errors with dnorm.
  vector<Type> log_N_pred = log(N_pred + Type(1e-8));
  vector<Type> log_N_dat = log(N_dat + Type(1e-8));
  nll -= dnorm(log_N_pred, log_N_dat, sd_N, true).sum();
  
  vector<Type> log_P_pred = log(P_pred + Type(1e-8));
  vector<Type> log_P_dat = log(P_dat + Type(1e-8));
  nll -= dnorm(log_P_pred, log_P_dat, sd_P, true).sum();
  
  vector<Type> log_Z_pred = log(Z_pred + Type(1e-8));
  vector<Type> log_Z_dat = log(Z_dat + Type(1e-8));
  nll -= dnorm(log_Z_pred, log_Z_dat, sd_Z, true).sum();

  // ------------------------------------------------------------------------
  // REPORTING SECTION
  // ------------------------------------------------------------------------
  
  // Report model predictions
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  
  // Report transformed parameters for interpretation
  REPORT(V_max);
  REPORT(K_n);
  REPORT(g_max);
  REPORT(K_p);
  REPORT(beta);
  REPORT(m_p);
  REPORT(m_z);
  REPORT(sd_N);
  REPORT(sd_P);
  REPORT(sd_Z);
  
  return nll;
}
