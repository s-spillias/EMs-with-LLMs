#include <TMB.hpp>

// ===============================================================================
//   NPZ (Nutrient-Phytoplankton-Zooplankton) Ecosystem Model
// ===============================================================================
//   This TMB model simulates the dynamics of a simple marine food web.
//
//   The model is described by a system of ordinary differential equations (ODEs):
//   1. dN/dt = -Uptake + Sloppy_Feeding + Excretion + Remineralization
//   2. dP/dt =  Uptake - Grazing - P_Mortality
//   3. dZ/dt =  Assimilated_Grazing - Z_Mortality - Excretion
//
//   These equations are solved numerically using a forward Euler method.
//   Parameters are estimated by maximizing the log-likelihood of the observed data.
// ===============================================================================

template<class Type>
Type objective_function<Type>::operator() ()
{
  // SECTION 1: DATA AND PREDICTION VECTORS
  // ======================================

  // --- Data Inputs ---
  // These macros declare and load data vectors.
  DATA_VECTOR(Time);      // Time points of observations (days)
  DATA_VECTOR(N_dat);     // Observed Nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);     // Observed Phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);     // Observed Zooplankton concentration (g C m^-3)

  // --- Prediction Vectors ---
  // These vectors will store the model's predictions over time.
  int n_obs = Time.size();
  vector<Type> N_pred(n_obs);
  vector<Type> P_pred(n_obs);
  vector<Type> Z_pred(n_obs);


  // SECTION 2: PARAMETERS
  // =====================

  // --- Model Parameters ---
  // These macros declare parameters to be estimated by the model.
  // Phytoplankton dynamics parameters
  PARAMETER(log_V_max);   // log of maximum phytoplankton growth rate (day^-1)
  PARAMETER(log_K_N);     // log of half-saturation constant for nutrient uptake (g C m^-3)

  // Zooplankton dynamics parameters
  PARAMETER(log_I_max);   // log of maximum zooplankton ingestion rate (day^-1)
  PARAMETER(log_K_P);     // log of half-saturation constant for grazing (g C m^-3)
  PARAMETER(logit_beta);  // logit of zooplankton assimilation efficiency (dimensionless, 0-1)
  PARAMETER(log_epsilon); // log of zooplankton excretion rate (day^-1)

  // Mortality and remineralization parameters
  PARAMETER(log_mu_P);    // log of phytoplankton linear mortality rate (day^-1)
  PARAMETER(log_mu_Z);    // log of zooplankton quadratic mortality rate ((g C m^-3)^-1 day^-1)
  PARAMETER(logit_lambda);  // logit of remineralization efficiency (dimensionless, 0-1)

  // Observation error parameters (log-scale)
  PARAMETER(log_sigma_N); // log of standard deviation for Nutrient observations
  PARAMETER(log_sigma_P); // log of standard deviation for Phytoplankton observations
  PARAMETER(log_sigma_Z); // log of standard deviation for Zooplankton observations

  // --- Parameter Transformations ---
  // Transform parameters from log/logit scale to natural scale for use in equations.
  // This enforces constraints (e.g., positivity, bounds between 0 and 1).
  Type V_max = exp(log_V_max);     // Max phytoplankton growth rate (day^-1). Determined from literature or model fitting.
  Type K_N = exp(log_K_N);         // Half-saturation constant for N uptake (g C m^-3). Nutrient level for half-max growth. Literature or fitting.
  Type I_max = exp(log_I_max);     // Max zooplankton ingestion rate (day^-1). Max consumption rate. Literature or fitting.
  Type K_P = exp(log_K_P);         // Half-saturation constant for grazing (g C m^-3). Phytoplankton density for half-max grazing. Literature or fitting.
  Type beta = invlogit(logit_beta); // Zooplankton assimilation efficiency (0-1). Fraction of grazing converted to biomass. Literature or fitting.
  Type epsilon = exp(log_epsilon); // Zooplankton excretion rate (day^-1). Rate of nutrient release. Literature or fitting.
  Type mu_P = exp(log_mu_P);       // Phytoplankton mortality rate (day^-1). Natural death rate. Literature or fitting.
  Type mu_Z = exp(log_mu_Z);       // Zooplankton quadratic mortality rate ((g C m^-3)^-1 day^-1). Represents predation. Literature or fitting.
  Type lambda = invlogit(logit_lambda);   // Remineralization efficiency (0-1). Fraction of dead organic matter recycled to N. Literature or fitting.

  // SECTION 3: MODEL DYNAMICS
  // ===========================

  // Initialize model predictions with the first observation.
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);

  // Integrate ODEs over time using a forward Euler method
  for (int i = 0; i < n_obs - 1; ++i) {
    Type dt = Time(i+1) - Time(i);

    // State variables at current time step 'i'. These are predictions from the previous step, not observed data.
    Type N_t = N_pred(i);
    Type P_t = P_pred(i);
    Type Z_t = Z_pred(i);

    // --- Ecological Process Rates ---
    // 1. Phytoplankton nutrient uptake (Michaelis-Menten kinetics)
    Type uptake = V_max * (N_t / (K_N + N_t + Type(1e-8))) * P_t;
    // 2. Zooplankton grazing on phytoplankton (Holling Type II functional response)
    Type grazing = I_max * (P_t / (K_P + P_t + Type(1e-8))) * Z_t;

    // --- System of Ordinary Differential Equations (ODEs) ---
    // 3. Change in Nutrient concentration
    Type dNdt = -uptake                                     // N loss to P
                + (Type(1.0) - beta) * grazing              // N gain from sloppy feeding
                + epsilon * Z_t                             // N gain from Z excretion
                + lambda * (mu_P * P_t + mu_Z * Z_t * Z_t); // N gain from remineralization
    // 4. Change in Phytoplankton concentration
    Type dPdt = uptake - grazing - mu_P * P_t;
    // 5. Change in Zooplankton concentration
    Type dZdt = beta * grazing - mu_Z * Z_t * Z_t - epsilon * Z_t;

    // --- Update state variables for the next time step 'i+1' ---
    N_pred(i+1) = N_pred(i) + dNdt * dt;
    P_pred(i+1) = P_pred(i) + dPdt * dt;
    Z_pred(i+1) = Z_pred(i) + dZdt * dt;

    // Ensure concentrations remain non-negative using a differentiable conditional
    N_pred(i+1) = CppAD::CondExpGe(N_pred(i+1), Type(0.0), N_pred(i+1), Type(1e-8));
    P_pred(i+1) = CppAD::CondExpGe(P_pred(i+1), Type(0.0), P_pred(i+1), Type(1e-8));
    Z_pred(i+1) = CppAD::CondExpGe(Z_pred(i+1), Type(0.0), Z_pred(i+1), Type(1e-8));
  }

  // SECTION 4: LIKELIHOOD CALCULATION
  // ===================================

  Type nll = 0.0; // Initialize negative log-likelihood

  // Transform observation error SDs from log scale and add a small constant for stability
  Type sigma_N = exp(log_sigma_N) + Type(1e-8);
  Type sigma_P = exp(log_sigma_P) + Type(1e-8);
  Type sigma_Z = exp(log_sigma_Z) + Type(1e-8);

  // Calculate likelihood for all observations
  for (int i = 0; i < n_obs; ++i) {
    // Lognormal likelihood for strictly positive data. Add small constant to avoid log(0).
    nll -= dnorm(log(N_dat(i) + Type(1e-8)), log(N_pred(i) + Type(1e-8)), sigma_N, true);
    nll -= dnorm(log(P_dat(i) + Type(1e-8)), log(P_pred(i) + Type(1e-8)), sigma_P, true);
    nll -= dnorm(log(Z_dat(i) + Type(1e-8)), log(Z_pred(i) + Type(1e-8)), sigma_Z, true);
  }

  // SECTION 5: REPORTING
  // ======================

  // Report predicted time series
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);

  // Report parameters on their natural scale for interpretation
  ADREPORT(V_max);
  ADREPORT(K_N);
  ADREPORT(I_max);
  ADREPORT(K_P);
  ADREPORT(beta);
  ADREPORT(epsilon);
  ADREPORT(mu_P);
  ADREPORT(mu_Z);
  ADREPORT(lambda);
  ADREPORT(sigma_N);
  ADREPORT(sigma_P);
  ADREPORT(sigma_Z);

  return nll;
}
