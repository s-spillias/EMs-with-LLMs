#include <TMB.hpp>

// Template Model Builder model for a simple plankton ecosystem
// Ecological equations:
// 1. Nutrient (N) dynamics: decreases via uptake by P and increases via remineralization from P mortality and inefficient assimilation from grazing.
// 2. Phytoplankton (P) dynamics: grows via nutrient uptake (saturating function) and declines due to grazing by Z and natural mortality.
// 3. Zooplankton (Z) dynamics: increases via assimilation (with efficiency e) of grazed P and declines by its mortality.
//
// Note: Predictions at time t are based solely on states at time t-1 to avoid data leakage.
template<class Type>
Type objective_function<Type>::operator()() {
  // Data: Time series and observations for each state variable
  DATA_VECTOR(Time);           // Time (days) vector
  DATA_VECTOR(N_dat);          // Observed Nutrient concentrations (g C m^-3)
  DATA_VECTOR(P_dat);          // Observed Phytoplankton concentrations (g C m^-3)
  DATA_VECTOR(Z_dat);          // Observed Zooplankton concentrations (g C m^-3)

  // Parameters (log-transformed to enforce positivity)
  // log_r: intrinsic rate of nutrient-limited growth for phytoplankton (day^-1)
  PARAMETER(log_r);
  // log_K: half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(log_K);
  // log_g_max: maximum grazing rate by zooplankton (day^-1)
  PARAMETER(log_g_max);
  // log_K_P: half-saturation constant for grazing functional response (g C m^-3)
  PARAMETER(log_K_P);
  // log_e: assimilation efficiency of grazed phytoplankton into zooplankton biomass (dimensionless)
  PARAMETER(log_e);
  // log_m_P: mortality rate for phytoplankton (day^-1)
  PARAMETER(log_m_P);
  // log_m_Z: mortality rate for zooplankton (day^-1)
  PARAMETER(log_m_Z);
  // log_sigma_N: log-standard deviation of observation error for Nutrient (log-scale)
  PARAMETER(log_sigma_N);
  // log_sigma_P: log-standard deviation of observation error for Phytoplankton (log-scale)
  PARAMETER(log_sigma_P);
  // log_sigma_Z: log-standard deviation of observation error for Zooplankton (log-scale)
  PARAMETER(log_sigma_Z);

  // Transform parameters to their natural scale and add small constant for numerical stability
  Type r = exp(log_r);                     // intrinsic growth rate (day^-1)
  Type K = exp(log_K) + Type(1e-8);          // half-saturation constant (g C m^-3)
  Type g_max = exp(log_g_max);               // max grazing rate (day^-1)
  Type K_P = exp(log_K_P) + Type(1e-8);        // grazing half-saturation (g C m^-3)
  Type e = exp(log_e);                       // assimilation efficiency (dimensionless)
  Type m_P = exp(log_m_P);                   // phytoplankton mortality (day^-1)
  Type m_Z = exp(log_m_Z);                   // zooplankton mortality (day^-1)
  Type sigma_N = exp(log_sigma_N) + Type(1e-8);// error sd for Nutrient observations
  Type sigma_P = exp(log_sigma_P) + Type(1e-8);// error sd for Phytoplankton observations
  Type sigma_Z = exp(log_sigma_Z) + Type(1e-8);// error sd for Zooplankton observations

  int n = Time.size();
  // Vectors to store predictions for N, P, and Z at each time step
  vector<Type> N_pred(n), P_pred(n), Z_pred(n);

  // Initialize predictions with the first (assumed error-free) observations
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);

  // Compute timestep duration (using a small constant to avoid division by zero)
  Type dt = (Time(1) - Time(0)) + Type(1e-8);

  // Loop over time steps to update predictions based on ecological dynamics
  for(int t = 1; t < n; t++){
    // 1. Nutrient uptake by phytoplankton via saturating Michaelis-Menten kinetics:
    //    uptake = r * (N_pred/(K+N_pred)) * P_pred
    Type uptake = r * N_pred(t-1) / (K + N_pred(t-1) + Type(1e-8)) * P_pred(t-1);

    // 2. Grazing on phytoplankton by zooplankton using a quadratic functional response:
    //    grazing = g_max * (P_pred^2/(K_P^2 + P_pred^2)) * Z_pred
    Type grazing = g_max * (P_pred(t-1) * P_pred(t-1)) / (K_P * K_P + P_pred(t-1) * P_pred(t-1) + Type(1e-8)) * Z_pred(t-1);

    // 3. Nutrient remineralization from phytoplankton mortality:
    Type remin = m_P * P_pred(t-1);

    // Update equations:
    // Equation 1: Nutrient dynamics
    N_pred(t) = N_pred(t-1) - dt * uptake + dt * ((Type(1.0) - e) * grazing + remin);
    // Equation 2: Phytoplankton dynamics: Growth via uptake minus losses to grazing and mortality
    P_pred(t) = P_pred(t-1) + dt * (uptake - grazing - m_P * P_pred(t-1));
    // Equation 3: Zooplankton dynamics: Gains from grazing assimilation minus mortality
    Z_pred(t) = Z_pred(t-1) + dt * (e * grazing - m_Z * Z_pred(t-1));

    // Ensure concentrations remain positive (using a small constant)
    N_pred(t) = max(N_pred(t), Type(1e-8));
    P_pred(t) = max(P_pred(t), Type(1e-8));
    Z_pred(t) = max(Z_pred(t), Type(1e-8));
  }

  // Likelihood calculation (using log-normal error distributions)
  // The likelihood is computed on the log-scale to account for data spanning orders of magnitude.
  // A fixed minimum standard deviation is used to prevent numerical issues.
  Type nll = 0.0;
  for(int t = 1; t < n; t++){
      nll -= dnorm(log(N_dat(t) + Type(1e-8)), log(N_pred(t) + Type(1e-8)), sigma_N, true);
      nll -= dnorm(log(P_dat(t) + Type(1e-8)), log(P_pred(t) + Type(1e-8)), sigma_P, true);
      nll -= dnorm(log(Z_dat(t) + Type(1e-8)), log(Z_pred(t) + Type(1e-8)), sigma_Z, true);
  }

  // Report predicted trajectories for post-estimation analysis
  REPORT(N_pred); // Nutrient predictions (g C m^-3)
  REPORT(P_pred); // Phytoplankton predictions (g C m^-3)
  REPORT(Z_pred); // Zooplankton predictions (g C m^-3)

  return nll;
}
