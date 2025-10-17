#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  /* 
  ================================================================================
  MODEL DESCRIPTION
  ================================================================================
  This TMB model implements a Nutrient-Phytoplankton-Zooplankton (NPZ) ecosystem model.
  It simulates the change in concentrations of N, P, and Z over time using a set of 
  ordinary differential equations (ODEs), solved here with a simple Euler integration scheme.

  Model Equations:
  1. dN/dt = delta*(N0 - N) - Vm*(N/(Ks + N))*P + (1-beta)*gmax*(P^2/(K_g^2 + P^2))*Z + m_p*P + m_z*Z^2
     - Nutrient concentration changes due to:
       - Mixing with deep water (+delta*(N0 - N))
       - Uptake by phytoplankton (-Vm*...)
       - Sloppy feeding/excretion by zooplankton (+(1-beta)*gmax*...)
       - Remineralization of dead phytoplankton (+m_p*P)
       - Remineralization of dead zooplankton (+m_z*Z^2)

  2. dP/dt = Vm*(N/(Ks + N))*P - gmax*(P^2/(K_g^2 + P^2))*Z - m_p*P
     - Phytoplankton concentration changes due to:
       - Growth via nutrient uptake (+Vm*...)
       - Grazing by zooplankton (-gmax*...)
       - Natural mortality (-m_p*P)

  3. dZ/dt = beta*gmax*(P^2/(K_g^2 + P^2))*Z - m_z*Z^2
     - Zooplankton concentration changes due to:
       - Growth from assimilated phytoplankton (+beta*gmax*...)
       - Quadratic mortality (-m_z*Z^2)
  ================================================================================
  */

  // --- 1. DATA INPUTS ---
  // These are the time-series data the model will be fitted to.
  DATA_VECTOR(Time_days); // Time vector from data file, sanitized from "Time (days)".
  DATA_VECTOR(N_dat);     // Observed Nutrient concentration (g C m^-3).
  DATA_VECTOR(P_dat);     // Observed Phytoplankton concentration (g C m^-3).
  DATA_VECTOR(Z_dat);     // Observed Zooplankton concentration (g C m^-3).

  // --- 2. PARAMETERS ---
  // These are the values the model will estimate.
  PARAMETER(Vm);      // Maximum phytoplankton growth rate (day^-1).
  PARAMETER(Ks);      // Nutrient half-saturation constant (g C m^-3).
  PARAMETER(gmax);    // Maximum zooplankton grazing rate (day^-1).
  PARAMETER(K_g);     // Grazing half-saturation constant (g C m^-3).
  PARAMETER(beta);    // Zooplankton assimilation efficiency (dimensionless).
  PARAMETER(m_p);     // Phytoplankton linear mortality rate (day^-1).
  PARAMETER(m_z);     // Zooplankton quadratic mortality rate ((g C m^-3)^-1 day^-1).
  PARAMETER(delta);   // Mixing rate with deep water (day^-1).
  PARAMETER(N0);      // Deep water nutrient concentration (g C m^-3).
  PARAMETER(log_sd_N); // Log of standard deviation for Nutrient observation error.
  PARAMETER(log_sd_P); // Log of standard deviation for Phytoplankton observation error.
  PARAMETER(log_sd_Z); // Log of standard deviation for Zooplankton observation error.

  // --- 3. MODEL SETUP ---
  // Transform log-standard deviations to positive standard deviations for likelihood calculation.
  Type sd_N = exp(log_sd_N);
  Type sd_P = exp(log_sd_P);
  Type sd_Z = exp(log_sd_Z);

  int n_obs = Time_days.size(); // Get the number of observations.

  // Create vectors to store the model's predictions.
  vector<Type> N_pred(n_obs);
  vector<Type> P_pred(n_obs);
  vector<Type> Z_pred(n_obs);

  // Initialize the prediction vectors with the first observed data point.
  // This sets the initial conditions of the model run.
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);

  // Initialize the negative log-likelihood. This is the value to be minimized.
  Type nll = 0.0;

  // --- 4. DYNAMIC MODEL (TIME-STEPPING LOOP) ---
  for (int i = 1; i < n_obs; ++i) {
    Type dt = Time_days(i) - Time_days(i-1); // Calculate time step duration.

    // Get predicted values from the *previous* time step for calculations.
    Type N_prev = N_pred(i-1);
    Type P_prev = P_pred(i-1);
    Type Z_prev = Z_pred(i-1);

    // --- Ecological Process Rates ---
    // 1. Phytoplankton nutrient uptake (Michaelis-Menten kinetics).
    Type nutrient_uptake = Vm * (N_prev / (Ks + N_prev + Type(1e-8))) * P_prev;

    // 2. Zooplankton grazing on phytoplankton (Holling Type III functional response).
    Type grazing = gmax * (P_prev * P_prev / (K_g*K_g + P_prev*P_prev + Type(1e-8))) * Z_prev;

    // 3. Phytoplankton mortality (linear).
    Type p_mortality = m_p * P_prev;

    // 4. Zooplankton mortality (quadratic, for stability).
    Type z_mortality = m_z * Z_prev * Z_prev;

    // 5. Nutrient remineralization from dead plankton (assumes 100% recycling).
    Type remineralization = p_mortality + z_mortality;

    // 6. Zooplankton excretion (unassimilated grazing).
    Type excretion = (Type(1.0) - beta) * grazing;

    // 7. Nutrient mixing from deep water (relaxation to N0).
    Type mixing = delta * (N0 - N_prev);

    // --- Euler Integration Step ---
    // Calculate the change in each state variable over the time step dt.
    Type dN = (mixing - nutrient_uptake + remineralization + excretion) * dt;
    Type dP = (nutrient_uptake - grazing - p_mortality) * dt;
    Type dZ = (beta * grazing - z_mortality) * dt;

    // Update predicted values for the current time step.
    N_pred(i) = N_prev + dN;
    P_pred(i) = P_prev + dP;
    Z_pred(i) = Z_prev + dZ;

    // Ensure predictions remain positive to avoid numerical errors (e.g., log(0)).
    // CppAD::CondExpGt is a smooth conditional, preferable to a hard 'if' statement.
    N_pred(i) = CppAD::CondExpGt(N_pred(i), Type(0.0), N_pred(i), Type(1e-8));
    P_pred(i) = CppAD::CondExpGt(P_pred(i), Type(0.0), P_pred(i), Type(1e-8));
    Z_pred(i) = CppAD::CondExpGt(Z_pred(i), Type(0.0), Z_pred(i), Type(1e-8));
  }

  // --- 5. LIKELIHOOD CALCULATION ---
  // Compare model predictions with observed data to calculate the likelihood.
  // A lognormal error distribution is used, which is equivalent to a normal
  // distribution on the log-transformed data and predictions. This is suitable for
  // strictly positive data like concentrations.
  for (int i = 0; i < n_obs; ++i) {
    nll -= dnorm(log(N_dat(i)), log(N_pred(i)), sd_N, true);
    nll -= dnorm(log(P_dat(i)), log(P_pred(i)), sd_P, true);
    nll -= dnorm(log(Z_dat(i)), log(Z_pred(i)), sd_Z, true);
  }

  // --- 6. REPORTING SECTION ---
  // Report parameters and predictions for output and analysis.
  REPORT(Vm);
  REPORT(Ks);
  REPORT(gmax);
  REPORT(K_g);
  REPORT(beta);
  REPORT(m_p);
  REPORT(m_z);
  REPORT(delta);
  REPORT(N0);
  REPORT(sd_N);
  REPORT(sd_P);
  REPORT(sd_Z);

  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);

  return nll;
}
