#include <TMB.hpp>

// TMB model for plankton ecosystem dynamics
template<class Type>
Type objective_function<Type>::operator() ()
{
  // 1. Data: Time series for nutrient (N), phytoplankton (P) and zooplankton (Z) observed concentrations.
  DATA_VECTOR(time);               // Time points (days)
  DATA_VECTOR(N_dat);              // Observed nutrient concentrations (g C m^-3)
  DATA_VECTOR(P_dat);              // Observed phytoplankton concentrations (g C m^-3)
  DATA_VECTOR(Z_dat);              // Observed zooplankton concentrations (g C m^-3)

  // 2. Parameters: individual parameters declared separately.
  PARAMETER(nutrient_input);  // Nutrient input rate (g C m^-3 day^-1)
  PARAMETER(kN);              // Half-saturation constant (g C m^-3)
  PARAMETER(mu);              // Maximum phytoplankton growth rate (day^-1)
  PARAMETER(mP);              // Phytoplankton mortality rate (day^-1)
  PARAMETER(g);               // Grazing rate (m^3 gC^-1 day^-1)
  PARAMETER(mZ);              // Zooplankton mortality rate (day^-1)
  PARAMETER(epsilon);         // Assimilation efficiency (unitless)
  PARAMETER(sigma);           // Minimum observation error (ensures numerical stability)

  Type nll = 0.0; // Negative log-likelihood

  int n = N_dat.size(); // Number of time steps

  // Model predictions for N, P, Z (using lagged values to avoid data leakage)
  vector<Type> N_pred(n);  // Predicted nutrient concentrations
  vector<Type> P_pred(n);  // Predicted phytoplankton concentrations
  vector<Type> Z_pred(n);  // Predicted zooplankton concentrations

  // 3. Initialization: Set first predictions equal to observed initial conditions.
  N_pred[0] = N_dat[0];  // (g C m^-3)
  P_pred[0] = P_dat[0];  // (g C m^-3)
  Z_pred[0] = Z_dat[0];  // (g C m^-3)

  // 4. Time-stepping simulation using lagged state values for predictions.
  for(int t = 1; t < n; t++){
    // Retrieve previous state values
    Type N_prev = N_pred[t-1];
    Type P_prev = P_pred[t-1];
    Type Z_prev = Z_pred[t-1];

    // Equation 1: Nutrient uptake by phytoplankton (Michaelis-Menten kinetics).
    Type uptake = mu * N_prev / (kN + N_prev + Type(1e-8)); // units: day^-1

    // Equation 2: Grazing on phytoplankton modeled with a saturating response.
    Type grazing = g * P_prev * Z_prev / (1 + g * P_prev + Type(1e-8)); // units: g C m^-3 day^-1

    // Equation 3: Recycling by phytoplankton mortality provides a nutrient feedback.
    Type recycling = mP * P_prev; // (g C m^-3 day^-1)

    // Equation 4: Zooplankton growth driven by grazing and modulated by assimilation efficiency.
    Type zooplankton_growth = epsilon * grazing; // (g C m^-3 day^-1)

    // Equation 5: Nutrient balance accounts for input, uptake loss, and recycling.
    N_pred[t] = N_prev + nutrient_input - uptake * P_prev + recycling; // (g C m^-3)

    // Equation 6: Phytoplankton dynamics include nutrient-driven growth, losses from grazing and mortality.
    P_pred[t] = P_prev + uptake * P_prev - grazing - mP * P_prev; // (g C m^-3)

    // Equation 7: Zooplankton dynamics based on growth from grazing and natural mortality.
    Z_pred[t] = Z_prev + zooplankton_growth - mZ * Z_prev; // (g C m^-3)
  }

  // 5. Likelihood Calculation: Compare model predictions with data using a lognormal error distribution.
  for(int t = 0; t < n; t++){
    Type this_sigma = sigma + Type(1e-8); // Ensuring a fixed minimum error for stability
    nll -= dnorm(log(N_dat[t] + Type(1e-8)), log(N_pred[t] + Type(1e-8)), this_sigma, true) - log(N_dat[t] + Type(1e-8)); // Nutrient likelihood (lognormal)
    nll -= dnorm(log(P_dat[t] + Type(1e-8)), log(P_pred[t] + Type(1e-8)), this_sigma, true) - log(P_dat[t] + Type(1e-8)); // Phytoplankton likelihood (lognormal)
    nll -= dnorm(log(Z_dat[t] + Type(1e-8)), log(Z_pred[t] + Type(1e-8)), this_sigma, true) - log(Z_dat[t] + Type(1e-8)); // Zooplankton likelihood (lognormal)
  }

  // 6. Reporting: Output predicted concentrations to be analyzed externally.
  REPORT(N_pred); // Report nutrient predictions (g C m^-3)
  REPORT(P_pred); // Report phytoplankton predictions (g C m^-3)
  REPORT(Z_pred); // Report zooplankton predictions (g C m^-3)

  return nll; // Return the negative log-likelihood
}
