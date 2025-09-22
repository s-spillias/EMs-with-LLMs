#include <TMB.hpp>
template<class Type>
Type dlnorm_custom(Type x, Type meanlog, Type sdlog) {
  // Returns the log density of the lognormal distribution for x.
  return -log(x + Type(1e-8)) - log(sdlog) - 0.5 * log(2.0 * 3.14159265358979323846)
         - pow(log(x + Type(1e-8)) - meanlog, 2) / (Type(2) * sdlog * sdlog);
}

// 1. Data inputs and model predictions:
//    (i) time: time in days.
//    (ii) N_dat, P_dat, Z_dat: observed nutrient, phytoplankton, and zooplankton concentrations (g C m^-3).
// 2. Parameters:
//    (i) init_N, init_P, init_Z: initial concentrations of nutrient, phytoplankton, and zooplankton.
//    (ii) growth_rate: intrinsic growth rate of phytoplankton (day^-1).
//    (iii) nutrient_halfsat: nutrient half-saturation constant (g C m^-3).
//    (iv) efficiency_NtoP: conversion efficiency of nutrient uptake to phytoplankton biomass (dimensionless).
//    (v) grazing_rate: zooplankton grazing rate on phytoplankton (day^-1).
//    (vi) grazing_halfsat: half-saturation constant for grazing (g C m^-3).
//    (vii) efficiency_PtoZ: assimilation efficiency of grazed phytoplankton to zooplankton biomass (dimensionless).
//    (viii) log_sigma_N, log_sigma_P, log_sigma_Z: logarithms of standard deviations for lognormal likelihood (ensuring strictly positive error values).
//
// Numbered Equations:
//   1. dN/dt = - efficiency_NtoP * growth_rate * [N/(N + nutrient_halfsat)] * P
//   2. dP/dt = growth_rate * [N/(N + nutrient_halfsat)] * P - grazing_rate * [P/(P + grazing_halfsat)] * Z
//   3. dZ/dt = efficiency_PtoZ * (grazing_rate * [P/(P + grazing_halfsat)] * Z) - mortality_rate * Z
//
// Numerical stability is maintained by adding small constants (e.g., 1e-8) to denominators and time step differences.
// The integration is performed using an Euler method, utilizing previous time step predictions only.
//
// TMB Model:
template<class Type>
Type objective_function<Type>::operator()() {
  // DATA: Input time series and observed data (all vectors of length n)
  DATA_VECTOR(time);              // Time in days
  DATA_VECTOR(N_dat);             // Nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);             // Phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);             // Zooplankton concentration (g C m^-3)

  // PARAMETERS: Initial conditions and process rates
  PARAMETER(init_N);              // Initial nutrient concentration (g C m^-3)
  PARAMETER(init_P);              // Initial phytoplankton concentration (g C m^-3)
  PARAMETER(init_Z);              // Initial zooplankton concentration (g C m^-3)
  
  PARAMETER(growth_rate);         // Intrinsic growth rate of phytoplankton (day^-1)
  PARAMETER(nutrient_halfsat);    // Nutrient half-saturation constant (g C m^-3)
  PARAMETER(efficiency_NtoP);     // Nutrient to phytoplankton conversion efficiency (dimensionless)
  PARAMETER(grazing_rate);        // Zooplankton grazing rate (day^-1)
  PARAMETER(grazing_halfsat);     // Grazing half-saturation constant (g C m^-3)
  PARAMETER(efficiency_PtoZ);     // Conversion efficiency from phytoplankton to zooplankton (dimensionless)

  // Error standard deviations (log-transformed for positivity)
  PARAMETER(log_sigma_N);         // log standard deviation of nutrient observations
  PARAMETER(log_sigma_P);         // log standard deviation of phytoplankton observations
  PARAMETER(log_sigma_Z);         // log standard deviation of zooplankton observations
  Type sigma_N = exp(log_sigma_N) + Type(1e-8);
  Type sigma_P = exp(log_sigma_P) + Type(1e-8);
  Type sigma_Z = exp(log_sigma_Z) + Type(1e-8);

  int n = time.size();            // Number of time steps

  // Vectors to store model predictions for each dynamic variable
  vector<Type> N_pred(n);         // Predicted nutrient concentrations
  vector<Type> P_pred(n);         // Predicted phytoplankton concentrations
  vector<Type> Z_pred(n);         // Predicted zooplankton concentrations

  // Setting initial conditions using supplied parameters
  N_pred(0) = init_N;
  P_pred(0) = init_P;
  Z_pred(0) = init_Z;

  // Initialize negative log-likelihood
  Type nll = 0.0;

  // Euler integration over time steps (t > 0) using previous state values only
  for (int t = 1; t < n; t++) {
    // Compute time step, ensuring numerical stability by adding a small constant
    Type dt = time(t) - time(t-1) + Type(1e-8);

    // Equation 1 (Nutrient dynamics):
    // Uptake by phytoplankton modeled as a saturating function
    Type uptake = efficiency_NtoP * growth_rate * (N_pred(t-1) / (N_pred(t-1) + nutrient_halfsat + Type(1e-8))) * P_pred(t-1);
    // Update nutrient value
    N_pred(t) = N_pred(t-1) - dt * uptake;

    // Equation 2 (Phytoplankton dynamics):
    // Growth driven by nutrient uptake minus losses due to grazing (Type II functional response)
    Type grazing = grazing_rate * (P_pred(t-1) / (P_pred(t-1) + grazing_halfsat + Type(1e-8))) * Z_pred(t-1);
    P_pred(t) = P_pred(t-1) + dt * (growth_rate * (N_pred(t-1) / (N_pred(t-1) + nutrient_halfsat + Type(1e-8))) * P_pred(t-1) - grazing);

    // Equation 3 (Zooplankton dynamics):
    // Increase from assimilated grazing minus constant mortality (assumed 0.1 day^-1)
    Type assimilation = efficiency_PtoZ * grazing;
    Type mortality_rate = Type(0.1); // Constant mortality rate for zooplankton (day^-1)
    Z_pred(t) = Z_pred(t-1) + dt * (assimilation - mortality_rate * Z_pred(t-1));

    // Ensure all predictions remain positive using smooth conditional expressions
    N_pred(t) = CppAD::CondExpGt(N_pred(t), Type(1e-8), N_pred(t), Type(1e-8));
    P_pred(t) = CppAD::CondExpGt(P_pred(t), Type(1e-8), P_pred(t), Type(1e-8));
    Z_pred(t) = CppAD::CondExpGt(Z_pred(t), Type(1e-8), Z_pred(t), Type(1e-8));

    // Likelihood calculation (using lognormal error distribution to account for multiplicative error)
    // Observations (_dat) are compared with log-transformed predictions (_pred)
    nll -= dlnorm_custom(N_dat(t), log(N_pred(t) + Type(1e-8)), sigma_N);
    nll -= dlnorm_custom(P_dat(t), log(P_pred(t) + Type(1e-8)), sigma_P);
    nll -= dlnorm_custom(Z_dat(t), log(Z_pred(t) + Type(1e-8)), sigma_Z);
  }

  // REPORT predicted trajectories for N, P, and Z (with _pred suffix matching observations _dat)
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);

  ADREPORT(nll);
  return nll;
}
