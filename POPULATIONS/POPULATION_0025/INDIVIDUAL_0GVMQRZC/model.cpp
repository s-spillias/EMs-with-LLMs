#include <TMB.hpp>

// 1. Model equations (numbered):
// 1) Uptake = uptake_rate * N * P / (half_saturation_N + N)
// 2) Grazing = grazing_rate * P * Z / (grazing_half_saturation + P)
// 3) dN/dt = -Uptake + egestion_efficiency * Grazing + respiration_efficiency * Uptake * assimilation_efficiency
// 4) dP/dt = Uptake * assimilation_efficiency - Grazing - mortality_rate_P * P
// 5) dZ/dt = Grazing * assimilation_efficiency - mortality_rate_Z * Z
// 6) Observations are modeled as log‑normal: log(dat) ~ Normal(log(pred), sigma)

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  vector<Type> Time = data["Time"];          // Time (days)
  vector<Type> N_dat = data["N_dat"];        // Observed nutrient concentrations
  vector<Type> P_dat = data["P_dat"];        // Observed phytoplankton concentrations
  vector<Type> Z_dat = data["Z_dat"];        // Observed zooplankton concentrations

  // Parameters
  Type uptake_rate = parameters["uptake_rate"];                     // Max nutrient uptake rate (day^-1)
  Type half_saturation_N = parameters["half_saturation_N"];         // Half‑saturation for nutrient uptake (g C m^-3)
  Type grazing_rate = parameters["grazing_rate"];                   // Max grazing rate (day^-1)
  Type grazing_half_saturation = parameters["grazing_half_saturation"]; // Half‑saturation for grazing (g C m^-3)
  Type mortality_rate_N = parameters["mortality_rate_N"];           // Background mortality of nutrients (day^-1)
  Type mortality_rate_P = parameters["mortality_rate_P"];           // Background mortality of phytoplankton (day^-1)
  Type mortality_rate_Z = parameters["mortality_rate_Z"];           // Background mortality of zooplankton (day^-1)
  Type assimilation_efficiency = parameters["assimilation_efficiency"]; // Efficiency of converting consumed phytoplankton into zooplankton (dimensionless)
  Type egestion_efficiency = parameters["egestion_efficiency"];     // Fraction of grazed phytoplankton returned to nutrients (dimensionless)
  Type respiration_efficiency = parameters["respiration_efficiency"]; // Fraction of assimilated phytoplankton lost to respiration (dimensionless)
  Type sigma_N = parameters["sigma_N"];                             // Log‑normal SD for nutrient observations
  Type sigma_P = parameters["sigma_P"];                             // Log‑normal SD for phytoplankton observations
  Type sigma_Z = parameters["sigma_Z"];                             // Log‑normal SD for zooplankton observations

  // Small constant to avoid division by zero
  const Type eps = Type(1e-8);

  // Prediction vectors
  vector<Type> N_pred(Time.size());
  vector<Type> P_pred(Time.size());
  vector<Type> Z_pred(Time.size());

  // Initial conditions set to first observation
  N_pred[0] = N_dat[0];
  P_pred[0] = P_dat[0];
  Z_pred[0] = Z_dat[0];

  // Numerical integration (Euler)
  for (int i = 1; i < Time.size(); ++i) {
    Type dt = Time[i] - Time[i-1];

    // Smooth saturating uptake
    Type uptake = uptake_rate * N_pred[i-1] * P_pred[i-1] /
                  (half_saturation_N + N_pred[i-1] + eps);

    // Smooth grazing response
    Type grazing = grazing_rate * P_pred[i-1] * Z_pred[i-1] /
                   (grazing_half_saturation + P_pred[i-1] + eps);

    // Differential changes
    Type dN = -uptake
              + egestion_efficiency * grazing
              + respiration_efficiency * uptake * assimilation_efficiency
              - mortality_rate_N * N_pred[i-1];

    Type dP = uptake * assimilation_efficiency
              - grazing
              - mortality_rate_P * P_pred[i-1];

    Type dZ = grazing * assimilation_efficiency
              - mortality_rate_Z * Z_pred[i-1];

    // Update predictions
    N_pred[i] = N_pred[i-1] + dN * dt;
    P_pred[i] = P_pred[i-1] + dP * dt;
    Z_pred[i] = Z_pred[i-1] + dZ * dt;

    // Enforce positivity
    N_pred[i] = max(N_pred[i], eps);
    P_pred[i] = max(P_pred[i], eps);
    Z_pred[i] = max(Z_pred[i], eps);
  }

  // Likelihood calculation (log‑normal)
  Type nll = Type(0.0);
  for (int i = 0; i < Time.size(); ++i) {
    nll -= R::dnorm(log(N_dat[i]), log(N_pred[i]), sigma_N, true);
    nll -= R::dnorm(log(P_dat[i]), log(P_pred[i]), sigma_P, true);
    nll -= R::dnorm(log(Z_dat[i]), log(Z_pred[i]), sigma_Z, true);
  }

  // Smooth penalty for parameter bounds
  auto penalty = [&](Type param, Type lower, Type upper) {
    Type p = Type(0.0);
    if (param < lower) {
      p += Type(0.5) * pow((param - lower) / lower, Type(2));
    } else if (param > upper) {
      p += Type(0.5) * pow((param - upper) / upper, Type(2));
    }
    return p;
  };

  nll += penalty(uptake_rate, Type(0.0), Type(5.0));
  nll += penalty(half_saturation_N, Type(0.0), Type(1.0));
  nll += penalty(grazing_rate, Type(0.0), Type(5.0));
  nll += penalty(grazing_half_saturation, Type(0.0), Type(1.0));
  nll += penalty(mortality_rate_N, Type(0.0), Type(0.1));
  nll += penalty(mortality_rate_P, Type(0.0), Type(0.1));
  nll += penalty(mortality_rate_Z, Type(0.0), Type(0.1));
  nll += penalty(assimilation_efficiency, Type(0.0), Type(1.0));
  nll += penalty(egestion_efficiency, Type(0.0), Type(1.0));
  nll += penalty(respiration_efficiency, Type(0.0), Type(1.0));
  nll += penalty(sigma_N, Type(0.001), Type(0.1));
  nll += penalty(sigma_P, Type(0.001), Type(0.1));
  nll += penalty(sigma_Z, Type(0.001), Type(0.1));

  // Reporting predictions
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);

  return nll;
}
