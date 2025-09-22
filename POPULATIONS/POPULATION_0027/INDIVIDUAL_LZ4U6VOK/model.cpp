#include <TMB.hpp>

//  1. Data and parameter explanations:
//     (1) Nutrient (N): g C m^-3 (observed as N_dat)
//     (2) Phytoplankton (P): g C m^-3 (observed as P_dat)
//     (3) Zooplankton (Z): g C m^-3 (observed as Z_dat)

//  2. Model state variables predictions (_pred) are computed recursively using Euler integration.
//  3. Equations number:
//     Equation 1: dN/dt = - Uptake by phytoplankton + regeneration from mortality of P and Z.
//     Equation 2: dP/dt = (efficiency * uptake) - grazing - natural mortality of phytoplankton.
//     Equation 3: dZ/dt = (assimilation efficiency * grazing) - natural mortality of zooplankton.

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;

  // DATA: Time and observations
  DATA_VECTOR(time);               // Time vector (days), from the first column of the data file.
  DATA_VECTOR(N_dat);              // Observed nutrient concentrations (g C m^-3)
  DATA_VECTOR(P_dat);              // Observed phytoplankton concentrations (g C m^-3)
  DATA_VECTOR(Z_dat);              // Observed zooplankton concentrations (g C m^-3)

  // PARAMETERS:
  PARAMETER(r_upt);                // Maximum nutrient uptake rate (day^-1); from expert opinion.
  // r_upt: Determines the maximum rate of nutrient consumption.
  PARAMETER(K_N);                  // Half-saturation constant for nutrient uptake (g C m^-3); literature suggested (lower_bound: 1e-8).
  // K_N: Concentration at which nutrient uptake is half its maximum.
  PARAMETER(eff_P);                // Conversion efficiency for nutrient uptake to phytoplankton growth (dimensionless); estimated.
  // eff_P: Fraction of uptaken nutrient converted into phytoplankton biomass.
  PARAMETER(graze_Z);              // Maximum grazing rate by zooplankton (day^-1); initial estimate.
  // graze_Z: Governs the rate at which zooplankton graze on phytoplankton.
  PARAMETER(K_P);                  // Half-saturation constant for grazing (g C m^-3); literature suggested (lower_bound: 1e-8).
  // K_P: Phytoplankton concentration at which grazing is half its maximum.
  PARAMETER(eff_Z);                // Zooplankton assimilation efficiency (dimensionless); expert opinion.
  // eff_Z: Fraction of grazed phytoplankton biomass converted into zooplankton biomass.
  PARAMETER(mort_P);               // Phytoplankton mortality rate (day^-1); expert estimate.
  // mort_P: Natural mortality rate of phytoplankton.
  PARAMETER(mort_Z);               // Zooplankton mortality rate (day^-1); initial estimate.
  // mort_Z: Natural mortality rate of zooplankton.

  // Observation error standard deviations (log scale to ensure positivity)
  PARAMETER(log_sd_N);             // Log standard deviation for nutrient observations.
  PARAMETER(log_sd_P);             // Log standard deviation for phytoplankton observations.
  PARAMETER(log_sd_Z);             // Log standard deviation for zooplankton observations.
  Type sd_N = exp(log_sd_N);
  Type sd_P = exp(log_sd_P);
  Type sd_Z = exp(log_sd_Z);
  // Initial conditions (log transformed to ensure positivity)
  PARAMETER(log_N0); // log initial nutrient concentration (g C m^-3)
  PARAMETER(log_P0); // log initial phytoplankton concentration (g C m^-3)
  PARAMETER(log_Z0); // log initial zooplankton concentration (g C m^-3)

  int n = time.size();

  // Initialize predictions with initial condition parameters
  vector<Type> N_pred(n);
  vector<Type> P_pred(n);
  vector<Type> Z_pred(n);
  N_pred(0) = exp(log_N0); // initial nutrient concentration
  P_pred(0) = exp(log_P0); // initial phytoplankton concentration
  Z_pred(0) = exp(log_Z0); // initial zooplankton concentration

  // Small constant to ensure numerical stability
  Type eps = Type(1e-8);

  // Compute time step differences (dt) ensuring dt > 0
  vector<Type> dt(n);
  dt(0) = eps;
  for(int i = 1; i < n; i++){
    dt(i) = time(i) - time(i-1) + eps;
  }

  // Model simulation using Euler integration
  for(int t = 1; t < n; t++){
    // Compute nutrient uptake using Michaelis-Menten kinetics:
    // uptake = r_upt * (N_pred(t-1))/(K_N + N_pred(t-1) + eps) * P_pred(t-1)
    Type uptake = r_upt * N_pred(t-1)/(K_N + N_pred(t-1) + eps) * P_pred(t-1);

    // Compute grazing using Holling type II functional response:
    // grazing = graze_Z * (P_pred(t-1))/(K_P + P_pred(t-1) + eps) * Z_pred(t-1)
    Type grazing = graze_Z * P_pred(t-1)/(K_P + P_pred(t-1) + eps) * Z_pred(t-1);

    // Equation 1: Nutrient dynamics
    // dN/dt = - uptake + recycling from natural mortalities of P and Z
    N_pred(t) = N_pred(t-1) - dt(t)*uptake + dt(t)*(mort_P * P_pred(t-1) + mort_Z * Z_pred(t-1));

    // Equation 2: Phytoplankton dynamics
    // dP/dt = (eff_P * uptake) - grazing - (mort_P * P)
    P_pred(t) = P_pred(t-1) + dt(t)*(eff_P * uptake - grazing - mort_P * P_pred(t-1));

    // Equation 3: Zooplankton dynamics
    // dZ/dt = (eff_Z * grazing) - (mort_Z * Z)
    Z_pred(t) = Z_pred(t-1) + dt(t)*(eff_Z * grazing - mort_Z * Z_pred(t-1));
  }

  // Likelihood calculation using lognormal error distributions
  // Include all time steps to ensure every observation has a corresponding prediction.
  Type nll = 0.0;
  for(int t = 0; t < n; t++){
    nll -= dlnorm(N_dat(t), log(N_pred(t) + eps), sd_N, true);
    nll -= dlnorm(P_dat(t), log(P_pred(t) + eps), sd_P, true);
    nll -= dlnorm(Z_dat(t), log(Z_pred(t) + eps), sd_Z, true);
  }

  // REPORT model predictions for diagnostics (_pred variables)
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);

  return nll;
}
