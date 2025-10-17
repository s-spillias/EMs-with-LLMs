#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------
  // DATA INPUTS
  // ------------------------------------------------------------------------
  
  DATA_VECTOR(Time); // Time vector of observations (days)
  DATA_VECTOR(N_dat);     // Observed Nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);     // Observed Phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);     // Observed Zooplankton concentration (g C m^-3)

  // ------------------------------------------------------------------------
  // PARAMETER DECLARATIONS
  // ------------------------------------------------------------------------

  // Phytoplankton parameters
  PARAMETER(v_max);   // Maximum phytoplankton growth rate (day^-1)
  PARAMETER(k_n);     // Half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(m_p);     // Phytoplankton linear mortality rate (day^-1)

  // Zooplankton parameters
  PARAMETER(g_max);   // Maximum zooplankton grazing rate (day^-1)
  PARAMETER(k_p);     // Half-saturation constant for grazing (g C m^-3)
  PARAMETER(m_z);     // Zooplankton linear mortality rate (day^-1)
  PARAMETER(beta);    // Zooplankton assimilation efficiency (dimensionless)

  // System and cycling parameters
  PARAMETER(omega);   // Remineralization rate (day^-1)
  PARAMETER(n0);      // Deep nutrient concentration (g C m^-3)
  PARAMETER(m);       // Mixed layer exchange rate (day^-1)

  // Observation error parameters
  PARAMETER(log_sigma_N); // Log of standard deviation for Nutrient observations
  PARAMETER(log_sigma_P); // Log of standard deviation for Phytoplankton observations
  PARAMETER(log_sigma_Z); // Log of standard deviation for Zooplankton observations

  // ------------------------------------------------------------------------
  // TRANSFORM PARAMETERS AND INITIALIZE STATE VARIABLES
  // ------------------------------------------------------------------------

  // Exponentiate log-transformed standard deviations
  Type sigma_N = exp(log_sigma_N);
  Type sigma_P = exp(log_sigma_P);
  Type sigma_Z = exp(log_sigma_Z);

  int n_obs = N_dat.size(); // Number of observations

  // Create vectors to store model predictions
  vector<Type> N_pred(n_obs);
  vector<Type> P_pred(n_obs);
  vector<Type> Z_pred(n_obs);

  // Initialize predictions with the first data point
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);

  // ------------------------------------------------------------------------
  // MODEL EQUATIONS
  //
  // 1. dN/dt = -Uptake + Excretion + Remineralization + Mixing
  // 2. dP/dt = Uptake - Grazing - P_Mortality
  // 3. dZ/dt = Assimilation - Z_Mortality
  // ------------------------------------------------------------------------

  for (int i = 1; i < n_obs; ++i) {
    Type dt = Time(i) - Time(i-1); // Time step duration

    // Use previous time step's predictions for calculations
    Type N_prev = N_pred(i-1);
    Type P_prev = P_pred(i-1);
    Type Z_prev = Z_pred(i-1);

    // Ecological process calculations
    Type uptake = v_max * N_prev / (k_n + N_prev + Type(1e-8)) * P_prev; // Phytoplankton nutrient uptake (Michaelis-Menten)
    Type grazing = g_max * P_prev / (k_p + P_prev + Type(1e-8)) * Z_prev; // Zooplankton grazing on phytoplankton (Holling Type II)
    Type p_mortality = m_p * P_prev; // Phytoplankton mortality
    Type z_mortality = m_z * Z_prev; // Zooplankton mortality
    Type assimilation = beta * grazing; // Zooplankton assimilation of grazed material
    Type excretion = (Type(1.0) - beta) * grazing; // Zooplankton egestion/excretion
    Type remineralization = omega * (p_mortality + z_mortality); // Remineralization of dead organic matter
    Type mixing = m * (n0 - N_prev); // Nutrient exchange with deep water

    // Differential equations using Euler integration
    Type dN = -uptake + excretion + remineralization + mixing;
    Type dP = uptake - grazing - p_mortality;
    Type dZ = assimilation - z_mortality;

    // Update state variables for the current time step
    N_pred(i) = N_prev + dN * dt;
    P_pred(i) = P_prev + dP * dt;
    Z_pred(i) = Z_prev + dZ * dt;

    // Ensure predictions remain positive to maintain biological realism
    if (N_pred(i) < Type(1e-8)) N_pred(i) = Type(1e-8);
    if (P_pred(i) < Type(1e-8)) P_pred(i) = Type(1e-8);
    if (Z_pred(i) < Type(1e-8)) Z_pred(i) = Type(1e-8);
  }

  // ------------------------------------------------------------------------
  // LIKELIHOOD CALCULATION
  // ------------------------------------------------------------------------

  Type nll = 0.0; // Initialize negative log-likelihood

  // Use a lognormal distribution for strictly positive concentration data
  // Add a fixed minimum standard deviation to prevent numerical issues
  Type sigma_N_eff = sigma_N + Type(0.01);
  Type sigma_P_eff = sigma_P + Type(0.01);
  Type sigma_Z_eff = sigma_Z + Type(0.01);

  for (int i = 0; i < n_obs; ++i) {
    nll -= dnorm(log(N_dat(i)), log(N_pred(i)), sigma_N_eff, true);
    nll -= dnorm(log(P_dat(i)), log(P_pred(i)), sigma_P_eff, true);
    nll -= dnorm(log(Z_dat(i)), log(Z_pred(i)), sigma_Z_eff, true);
  }

  // ------------------------------------------------------------------------
  // PARAMETER BOUNDS (SOFT PENALTIES)
  // ------------------------------------------------------------------------
  
  // Penalize parameters if they move outside biologically plausible ranges
  // This helps guide the optimization process.
  if (v_max < 0.0)   nll -= dnorm(v_max, Type(0.0), Type(1.0), true);
  if (k_n < 0.0)     nll -= dnorm(k_n, Type(0.0), Type(1.0), true);
  if (m_p < 0.0)     nll -= dnorm(m_p, Type(0.0), Type(1.0), true);
  if (g_max < 0.0)   nll -= dnorm(g_max, Type(0.0), Type(1.0), true);
  if (k_p < 0.0)     nll -= dnorm(k_p, Type(0.0), Type(1.0), true);
  if (m_z < 0.0)     nll -= dnorm(m_z, Type(0.0), Type(1.0), true);
  if (beta < 0.0 || beta > 1.0) nll -= dnorm(beta, Type(0.5), Type(1.0), true);
  if (omega < 0.0)   nll -= dnorm(omega, Type(0.0), Type(1.0), true);
  if (n0 < 0.0)      nll -= dnorm(n0, Type(0.0), Type(1.0), true);
  if (m < 0.0)       nll -= dnorm(m, Type(0.0), Type(1.0), true);

  // ------------------------------------------------------------------------
  // REPORTING SECTION
  // ------------------------------------------------------------------------

  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);

  ADREPORT(v_max);
  ADREPORT(k_n);
  ADREPORT(m_p);
  ADREPORT(g_max);
  ADREPORT(k_p);
  ADREPORT(m_z);
  ADREPORT(beta);
  ADREPORT(omega);
  ADREPORT(n0);
  ADREPORT(m);
  ADREPORT(sigma_N);
  ADREPORT(sigma_P);
  ADREPORT(sigma_Z);

  return nll;
}
