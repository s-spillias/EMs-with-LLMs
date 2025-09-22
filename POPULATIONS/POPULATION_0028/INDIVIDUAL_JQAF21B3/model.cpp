#include <TMB.hpp>

//--------------------------------------------------------
// 1. Model Overview:
//    (1) Nutrient dynamics: nutrient concentration is reduced by phytoplankton uptake and replenished by environmental inputs.
//    (2) Phytoplankton dynamics: growth depends on nutrient availability (via a saturating function), reduced by mortality and grazing.
//    (3) Zooplankton dynamics: growth is driven by grazing on phytoplankton with conversion efficiency, and reduced by mortality.
//--------------------------------------------------------

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density; // for dnorm, etc.

  // DATA: Observations for each compartment, along with time vector.
  DATA_VECTOR(time);                             // Time (days)
  DATA_VECTOR(N_dat);                            // Observed nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);                            // Observed phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);                            // Observed zooplankton concentration (g C m^-3)

  // PARAMETERS: Observation error (log-scale) and model process parameters.
  PARAMETER(log_sigma_N);   // Log of observation error std. dev. for nutrient (g C m^-3)
  PARAMETER(log_sigma_P);   // Log of observation error std. dev. for phytoplankton (g C m^-3)
  PARAMETER(log_sigma_Z);   // Log of observation error std. dev. for zooplankton (g C m^-3)
  
  PARAMETER(growth_rate);   // (1) Intrinsic growth rate of phytoplankton (d^-1)
  PARAMETER(nutrient_uptake); // (2) Maximum nutrient uptake rate (g C m^-3 d^-1)
  PARAMETER(half_sat);      // (3) Half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(zoop_efficiency); // (4) Zooplankton consumption efficiency (0-1, unitless)
  PARAMETER(mortality_P);   // (5) Phytoplankton mortality rate (d^-1)
  PARAMETER(mortality_Z);   // (6) Zooplankton mortality rate (d^-1)
  PARAMETER(env_modifier);  // (7) Environmental modifier coefficient (unitless) affecting baseline rates

  // Numerical stability constant to avoid division by zero.
  Type eps = Type(1e-8);

  int n = time.size();
  // State variables predictions for each compartment:
  vector<Type> N_pred(n);   // Predicted nutrient concentration (g C m^-3)
  vector<Type> P_pred(n);   // Predicted phytoplankton concentration (g C m^-3)
  vector<Type> Z_pred(n);   // Predicted zooplankton concentration (g C m^-3)

  // Set initial conditions using the first observed timepoint.
  N_pred(0) = N_dat(0); // set initial nutrient concentration
  P_pred(0) = P_dat(0); // set initial phytoplankton concentration
  Z_pred(0) = Z_dat(0); // set initial zooplankton concentration

  //--------------------------------------------------------
  // 2. Equations Description:
  //    1) Nutrient dynamics: N_pred(t) = N_pred(t-1) + env_input - uptake
  //    2) Phytoplankton dynamics: P_pred(t) = P_pred(t-1) + growth - mortality - grazing
  //    3) Zooplankton dynamics: Z_pred(t) = Z_pred(t-1) + conversion of grazing - mortality
  //--------------------------------------------------------
  for(int t = 1; t < n; t++){
    // Compute nutrient uptake using a Michaelis-Menten type saturating function.
    // uptake = nutrient_uptake * P(t-1) * (N(t-1) / (half_sat + N(t-1) + eps))
    Type uptake = nutrient_uptake * P_pred(t-1) * (N_pred(t-1) / (half_sat + N_pred(t-1) + eps));  // (g C m^-3 d^-1)

    // Phytoplankton growth based on nutrient availability.
    // growth = growth_rate * P(t-1) * (N(t-1) / (half_sat + N(t-1) + eps))
    Type growth = growth_rate * P_pred(t-1) * (N_pred(t-1) / (half_sat + N_pred(t-1) + eps));  // (g C m^-3 d^-1)
    
    // Zooplankton grazing using a smooth Holling type II functional response.
    // grazing = zoop_efficiency * Z(t-1) * (P(t-1) / (half_sat + P(t-1) + eps))
    Type grazing = zoop_efficiency * Z_pred(t-1) * (P_pred(t-1) / (half_sat + P_pred(t-1) + eps));  // (g C m^-3 d^-1)
    
    // Update nutrient concentration:
    // Nutrient input is assumed constant baseline (0.1) modulated by env_modifier.
    N_pred(t) = N_pred(t-1) + (env_modifier * 0.1 * N_pred(t-1) - uptake);
    
    // Update phytoplankton concentration:
    // Phytoplankton gains from growth minus losses from mortality and grazing.
    P_pred(t) = P_pred(t-1) + (growth - mortality_P * P_pred(t-1) - grazing);
    
    // Update zooplankton concentration:
    // Zooplankton increases based on the assimilated fraction of grazed phytoplankton, reduced by mortality.
    Z_pred(t) = Z_pred(t-1) + (grazing * zoop_efficiency - mortality_Z * Z_pred(t-1));
  }

  //--------------------------------------------------------
  // 3. Likelihood Calculation:
  //    Each observation is assumed to follow a lognormal distribution.
  //    Log-likelihood contributions are computed for N, P, and Z.
  //--------------------------------------------------------
  Type nll = 0.0;
  Type sigma_N = exp(log_sigma_N) + Type(1e-8);
  Type sigma_P = exp(log_sigma_P) + Type(1e-8);
  Type sigma_Z = exp(log_sigma_Z) + Type(1e-8);
  
  for(int t = 0; t < n; t++){
    // Use log-transform to handle data spanning multiple orders of magnitude.
    nll -= dnorm(log(N_dat(t) + eps), log(N_pred(t) + eps), sigma_N, true);
    nll -= dnorm(log(P_dat(t) + eps), log(P_pred(t) + eps), sigma_P, true);
    nll -= dnorm(log(Z_dat(t) + eps), log(Z_pred(t) + eps), sigma_Z, true);
  }

  // REPORT all predicted states
  REPORT(N_pred);  // Nutrient predictions (g C m^-3)
  REPORT(P_pred);  // Phytoplankton predictions (g C m^-3)
  REPORT(Z_pred);  // Zooplankton predictions (g C m^-3)
  
  return nll;
}
