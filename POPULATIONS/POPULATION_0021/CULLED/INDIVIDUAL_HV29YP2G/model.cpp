#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------
  // DATA
  // ------------------------------------------------------------------------
  
  // Observed data vectors
  DATA_VECTOR(Time);      // Time points of observations (days)
  DATA_VECTOR(N_dat);     // Observed Nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);     // Observed Phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);     // Observed Zooplankton concentration (g C m^-3)

  // ------------------------------------------------------------------------
  // PARAMETERS
  // ------------------------------------------------------------------------
  
  // Initial conditions (treated as fixed parameters, set from data)
  PARAMETER(N0);          // Initial Nutrient concentration (g C m^-3)
  PARAMETER(P0);          // Initial Phytoplankton concentration (g C m^-3)
  PARAMETER(Z0);          // Initial Zooplankton concentration (g C m^-3)

  // Process parameters (transformed to ensure biological constraints)
  PARAMETER(log_V_max);   // Log of max phytoplankton growth rate (day^-1)
  PARAMETER(log_K_N);     // Log of half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(log_g_max);   // Log of max zooplankton grazing rate (day^-1)
  PARAMETER(log_K_P);     // Log of half-saturation constant for grazing (g C m^-3)
  PARAMETER(log_N_in);    // Log of external nutrient concentration for mixing (g C m^-3)
  PARAMETER(log_omega);   // Log of physical mixing rate (day^-1)
  PARAMETER(logit_beta);  // Logit of zooplankton assimilation efficiency (dimensionless)
  PARAMETER(log_m_P);     // Log of phytoplankton mortality rate (day^-1)
  PARAMETER(log_m_Z);     // Log of zooplankton linear mortality rate (day^-1)
  PARAMETER(log_m_Z_sq);  // Log of zooplankton quadratic mortality rate ((g C m^-3)^-1 day^-1)
  
  // Observation error parameters (log-scale standard deviations)
  PARAMETER(log_sd_N);    // Log standard deviation for Nutrient observations
  PARAMETER(log_sd_P);    // Log standard deviation for Phytoplankton observations
  PARAMETER(log_sd_Z);    // Log standard deviation for Zooplankton observations

  // Back-transform parameters to their natural scale
  Type V_max = exp(log_V_max);     // Max phytoplankton growth rate (day^-1)
  Type K_N = exp(log_K_N);         // Half-saturation constant for nutrient uptake (g C m^-3)
  Type g_max = exp(log_g_max);     // Max zooplankton grazing rate (day^-1)
  Type K_P = exp(log_K_P);         // Half-saturation constant for grazing (g C m^-3)
  Type N_in = exp(log_N_in);       // External nutrient concentration (g C m^-3)
  Type omega = exp(log_omega);     // Physical mixing rate (day^-1)
  Type beta = invlogit(logit_beta); // Zooplankton assimilation efficiency (dimensionless, 0-1)
  Type m_P = exp(log_m_P);         // Phytoplankton mortality rate (day^-1)
  Type m_Z = exp(log_m_Z);         // Zooplankton linear mortality rate (day^-1)
  Type m_Z_sq = exp(log_m_Z_sq);   // Zooplankton quadratic mortality rate ((g C m^-3)^-1 day^-1)

  // Back-transform observation error parameters
  Type sd_N = exp(log_sd_N);       // Standard deviation for N (log scale)
  Type sd_P = exp(log_sd_P);       // Standard deviation for P (log scale)
  Type sd_Z = exp(log_sd_Z);       // Standard deviation for Z (log scale)

  // ------------------------------------------------------------------------
  // MODEL EQUATIONS
  // ------------------------------------------------------------------------
  
  // The model is a set of ordinary differential equations (ODEs) describing the
  // rate of change of each component.
  // 1. dN/dt = -Uptake + Nutrient_Recycling + Nutrient_Supply
  // 2. dP/dt =  Uptake - Grazing - Phytoplankton_Mortality
  // 3. dZ/dt =  Assimilated_Grazing - Zooplankton_Mortality

  int n_steps = Time.size();
  vector<Type> N_pred(n_steps);
  vector<Type> P_pred(n_steps);
  vector<Type> Z_pred(n_steps);

  // Initialize predictions with the initial condition parameters
  N_pred(0) = N0;
  P_pred(0) = P0;
  Z_pred(0) = Z0;

  // Small constant to prevent division by zero in rate calculations
  Type epsilon = 1e-8;

  // Use forward Euler method to integrate the ODEs over time
  for (int i = 1; i < n_steps; ++i) {
    Type dt = Time(i) - Time(i-1);
    
    // State variables from the previous time step
    Type N_prev = N_pred(i-1);
    Type P_prev = P_pred(i-1);
    Type Z_prev = Z_pred(i-1);

    // Ecological process rates
    Type uptake_term = (V_max * N_prev) / (K_N + N_prev + epsilon);
    // Changed to a Type III functional response for grazing
    Type grazing_term = (g_max * P_prev * P_prev) / (K_P * K_P + P_prev * P_prev + epsilon);

    // Total fluxes
    Type total_uptake = uptake_term * P_prev;
    Type total_grazing = grazing_term * Z_prev;
    Type assimilated_grazing = beta * total_grazing;
    Type unassimilated_grazing = (Type(1.0) - beta) * total_grazing;
    Type p_mortality_flux = m_P * P_prev;
    Type z_mortality_flux = m_Z * Z_prev + m_Z_sq * Z_prev * Z_prev;
    Type nutrient_supply = omega * (N_in - N_prev);

    // Differentials (dN/dt, dP/dt, dZ/dt)
    Type dN_dt = -total_uptake + unassimilated_grazing + p_mortality_flux + z_mortality_flux + nutrient_supply;
    Type dP_dt = total_uptake - total_grazing - p_mortality_flux;
    Type dZ_dt = assimilated_grazing - z_mortality_flux;

    // Update predictions using forward Euler step
    N_pred(i) = N_prev + dN_dt * dt;
    P_pred(i) = P_prev + dP_dt * dt;
    Z_pred(i) = Z_prev + dZ_dt * dt;

    // Ensure predictions are non-negative to maintain biological realism and numerical stability
    N_pred(i) = CppAD::CondExpGe(N_pred(i), Type(0.0), N_pred(i), epsilon);
    P_pred(i) = CppAD::CondExpGe(P_pred(i), Type(0.0), P_pred(i), epsilon);
    Z_pred(i) = CppAD::CondExpGe(Z_pred(i), Type(0.0), Z_pred(i), epsilon);
  }

  // ------------------------------------------------------------------------
  // LIKELIHOOD CALCULATION
  // ------------------------------------------------------------------------
  
  Type nll = 0.0; // Initialize negative log-likelihood

  // Assume lognormal error distribution for all observations.
  // This is appropriate for positive-only data like concentrations.
  for (int i = 0; i < n_steps; ++i) {
    // The data has no zeros, so log(dat) is safe. Predictions are floored at epsilon.
    nll -= dnorm(log(N_dat(i)), log(N_pred(i)), sd_N, true);
    nll -= dnorm(log(P_dat(i)), log(P_pred(i)), sd_P, true);
    nll -= dnorm(log(Z_dat(i)), log(Z_pred(i)), sd_Z, true);
  }

  // ------------------------------------------------------------------------
  // REPORTING
  // ------------------------------------------------------------------------
  
  // Report predicted time series for plotting and analysis
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  
  // Report back-transformed parameters for interpretation
  ADREPORT(V_max);
  ADREPORT(K_N);
  ADREPORT(g_max);
  ADREPORT(K_P);
  ADREPORT(N_in);
  ADREPORT(omega);
  ADREPORT(beta);
  ADREPORT(m_P);
  ADREPORT(m_Z);
  ADREPORT(m_Z_sq);
  ADREPORT(sd_N);
  ADREPORT(sd_P);
  ADREPORT(sd_Z);

  return nll;
}
