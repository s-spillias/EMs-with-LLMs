#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------
  // DATA
  // ------------------------------------------------------------------------
  
  // Observed data vectors
  DATA_VECTOR(Time); // Time vector of observations
  DATA_VECTOR(N_dat); // Observed Nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat); // Observed Phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat); // Observed Zooplankton concentration (g C m^-3)

  // ------------------------------------------------------------------------
  // PARAMETERS
  // ------------------------------------------------------------------------

  // Phytoplankton growth parameters
  PARAMETER(V_max);   // Maximum phytoplankton uptake rate (day^-1)
  PARAMETER(K_N);     // Half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(alpha_shading); // Self-shading coefficient for phytoplankton ((g C m^-3)^-1)

  // Zooplankton grazing parameters
  PARAMETER(g_max);   // Maximum grazing rate of zooplankton (day^-1)
  PARAMETER(K_P);     // Half-saturation constant for grazing (g C m^-3)
  PARAMETER(beta);    // Zooplankton assimilation efficiency (dimensionless)

  // Mortality and regeneration parameters
  PARAMETER(m_P);       // Phytoplankton linear mortality rate (day^-1)
  PARAMETER(m_Z);       // Zooplankton quadratic mortality rate ((g C m^-3)^-1 day^-1)
  PARAMETER(gamma);     // Zooplankton excretion rate (day^-1)
  PARAMETER(epsilon);   // Remineralization rate of detritus (day^-1)
  PARAMETER(w_D);       // Detritus sinking rate (day^-1)

  // Observation error parameters (log scale to ensure positivity)
  PARAMETER(log_sigma_N); // Log standard deviation for Nutrient observations
  PARAMETER(log_sigma_P); // Log standard deviation for Phytoplankton observations
  PARAMETER(log_sigma_Z); // Log standard deviation for Zooplankton observations

  // Transform log standard deviations to natural scale
  Type sigma_N = exp(log_sigma_N);
  Type sigma_P = exp(log_sigma_P);
  Type sigma_Z = exp(log_sigma_Z);

  // ------------------------------------------------------------------------
  // MODEL DEFINITION
  // ------------------------------------------------------------------------

  int n_obs = Time.size(); // Number of observation time points

  // Create vectors to store model predictions
  vector<Type> N_pred(n_obs);
  vector<Type> P_pred(n_obs);
  vector<Type> Z_pred(n_obs);
  vector<Type> D_pred(n_obs); // Detritus state variable

  // Initialize predictions at time 0 with the first data point
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);
  D_pred(0) = Type(0.0); // Initialize detritus at zero

  // --- System of Ordinary Differential Equations (ODEs) ---
  // 1. dN/dt = -Uptake + Unassimilated_Grazing + Excretion + Remineralization
  // 2. dP/dt =  Uptake - Grazing - P_Mortality
  // 3. dZ/dt =  Assimilated_Grazing - Z_Mortality - Excretion
  // 4. dD/dt =  P_Mortality + Z_Mortality - Remineralization - Detritus_Sinking

  // Time-stepping loop to solve the ODEs using the Euler method
  for (int i = 1; i < n_obs; ++i) {
    Type dt = Time(i) - Time(i-1); // Time step duration

    // Concentrations from the previous time step
    Type N_prev = N_pred(i-1);
    Type P_prev = P_pred(i-1);
    Type Z_prev = Z_pred(i-1);
    Type D_prev = D_pred(i-1);

    // --- Ecological Process Rates ---
    
    // Nutrient uptake by phytoplankton (Michaelis-Menten with self-shading)
    Type nutrient_limitation = N_prev / (K_N + N_prev + Type(1e-8));
    Type self_shading_limitation = Type(1.0) / (Type(1.0) + alpha_shading * P_prev);
    Type uptake = V_max * nutrient_limitation * self_shading_limitation * P_prev;

    // Grazing of phytoplankton by zooplankton (Holling Type III functional response)
    Type grazing = g_max * (P_prev * P_prev) / ((K_P * K_P) + (P_prev * P_prev) + Type(1e-8)) * Z_prev;

    // Phytoplankton mortality (linear)
    Type p_mortality = m_P * P_prev;

    // Zooplankton mortality (quadratic, representing predation)
    Type z_mortality = m_Z * Z_prev * Z_prev;

    // Zooplankton excretion/respiration
    Type excretion = gamma * Z_prev;

    // Grazing assimilated by zooplankton
    Type assimilated_grazing = beta * grazing;

    // Unassimilated grazing (sloppy feeding), recycled to nutrient pool
    Type unassimilated_grazing = (Type(1.0) - beta) * grazing;

    // Remineralization of detritus pool
    Type remineralization = epsilon * D_prev;

    // Sinking of detritus out of the mixed layer
    Type detritus_sinking = w_D * D_prev;

    // --- Update State Variables ---
    
    // Change in Nutrient concentration
    Type dN = -uptake + unassimilated_grazing + excretion + remineralization;
    N_pred(i) = N_prev + dN * dt;

    // Change in Phytoplankton concentration
    Type dP = uptake - grazing - p_mortality;
    P_pred(i) = P_prev + dP * dt;

    // Change in Zooplankton concentration
    Type dZ = assimilated_grazing - z_mortality - excretion;
    Z_pred(i) = Z_prev + dZ * dt;

    // Change in Detritus concentration
    Type dD = p_mortality + z_mortality - remineralization - detritus_sinking;
    D_pred(i) = D_prev + dD * dt;

    // --- Numerical Stability ---
    // Ensure predictions do not fall below a small positive value
    N_pred(i) = CppAD::CondExpGt(N_pred(i), Type(0.0), N_pred(i), Type(1e-8));
    P_pred(i) = CppAD::CondExpGt(P_pred(i), Type(0.0), P_pred(i), Type(1e-8));
    Z_pred(i) = CppAD::CondExpGt(Z_pred(i), Type(0.0), Z_pred(i), Type(1e-8));
    D_pred(i) = CppAD::CondExpGt(D_pred(i), Type(0.0), D_pred(i), Type(1e-8));
  }

  // ------------------------------------------------------------------------
  // LIKELIHOOD CALCULATION
  // ------------------------------------------------------------------------

  Type nll = 0.0; // Initialize negative log-likelihood

  // Loop through all observations to calculate the likelihood
  for (int i = 0; i < n_obs; ++i) {
    // Lognormal distribution is appropriate for strictly positive data like concentrations
    // The 'true' argument specifies that the log-probability is returned
    nll -= dnorm(log(N_dat(i)), log(N_pred(i)), sigma_N, true);
    nll -= dnorm(log(P_dat(i)), log(P_pred(i)), sigma_P, true);
    nll -= dnorm(log(Z_dat(i)), log(Z_pred(i)), sigma_Z, true);
  }

  // ------------------------------------------------------------------------
  // PARAMETER BOUNDS (Smooth Penalties)
  // ------------------------------------------------------------------------
  
  // Penalize parameters that stray outside their biologically plausible ranges.
  // This acts as a "soft" constraint during optimization.
  Type penalty_sd = 0.1;
  if (V_max < 0.0)     { nll -= dnorm(V_max, Type(0.0), penalty_sd, true); }
  if (K_N < 0.0)      { nll -= dnorm(K_N, Type(0.0), penalty_sd, true); }
  if (alpha_shading < 0.0) { nll -= dnorm(alpha_shading, Type(0.0), penalty_sd, true); }
  if (g_max < 0.0)     { nll -= dnorm(g_max, Type(0.0), penalty_sd, true); }
  if (K_P < 0.0)      { nll -= dnorm(K_P, Type(0.0), penalty_sd, true); }
  if (beta < 0.0)     { nll -= dnorm(beta, Type(0.0), penalty_sd, true); }
  if (beta > 1.0)     { nll -= dnorm(beta, Type(1.0), penalty_sd, true); }
  if (m_P < 0.0)      { nll -= dnorm(m_P, Type(0.0), penalty_sd, true); }
  if (m_Z < 0.0)      { nll -= dnorm(m_Z, Type(0.0), penalty_sd, true); }
  if (gamma < 0.0)     { nll -= dnorm(gamma, Type(0.0), penalty_sd, true); }
  if (epsilon < 0.0)   { nll -= dnorm(epsilon, Type(0.0), penalty_sd, true); }
  if (w_D < 0.0)       { nll -= dnorm(w_D, Type(0.0), penalty_sd, true); }

  // ------------------------------------------------------------------------
  // REPORTING
  // ------------------------------------------------------------------------

  // Report predicted time series for plotting and evaluation
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  REPORT(D_pred);

  // Report the final negative log-likelihood value
  REPORT(nll);

  // Report predictions for standard error calculation
  ADREPORT(N_pred);
  ADREPORT(P_pred);
  ADREPORT(Z_pred);
  ADREPORT(D_pred);

  return nll;
}
