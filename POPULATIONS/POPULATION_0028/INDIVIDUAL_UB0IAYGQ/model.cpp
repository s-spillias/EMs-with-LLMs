#include <TMB.hpp>

// 1. Data Inputs and Parameters:
//    - time: vector of time points (days)
//    - N_dat, P_dat, Z_dat: observed concentrations (g C m^-3)
//    - Parameters control rates (day^-1) and saturation constants (g C m^-3)
// 2. Equations:
//    Equation 1: U = v_N * N / (K_N + N + eps)
//    Equation 2: dP/dt = eps_P * U * P - g_Z * P^2/(K_P + P + eps) - d_P * P
//    Equation 3: dZ/dt = grazing - d_Z * Z
//    Equation 4: dN/dt = - U * P + r*(P + Z)
//    Equation 5: Use lognormal likelihood for data (log-scale)
// 3. Implementation Details:
//    - Euler integration with dt = time[t] - time[t-1]
//    - Small constant (eps) added to denominators to ensure numerical stability
template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;
  
  // Data vectors
  DATA_VECTOR(time);        // Time steps (days)
  DATA_VECTOR(N_dat);       // Nutrient observations (g C m^-3)
  DATA_VECTOR(P_dat);       // Phytoplankton observations (g C m^-3)
  DATA_VECTOR(Z_dat);       // Zooplankton observations (g C m^-3)
  DATA_VECTOR(temperature_aux); // Ambient temperature (°C) if provided

  int n = time.size();

  // Parameters with biological meanings as documented in parameters.json
  PARAMETER(v_N);           // Maximum nutrient uptake rate (day^-1)
  PARAMETER(K_N);           // Half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(eps_P);         // Phytoplankton growth efficiency (dimensionless)
  PARAMETER(g_Z);           // Zooplankton grazing rate (day^-1)
  PARAMETER(K_P);           // Half-saturation constant for grazing (g C m^-3)
  PARAMETER(d_P);           // Phytoplankton mortality rate (day^-1)
  PARAMETER(d_Z);           // Zooplankton mortality rate (day^-1)
  PARAMETER(eps_Z);         // Zooplankton conversion efficiency from grazed phytoplankton biomass (dimensionless)
  PARAMETER(gamma);         // Exponent for zooplankton grazing capturing predator interference (dimensionless)
  PARAMETER(r);             // Remineralization rate (day^-1)
  PARAMETER(alpha);         // Self-shading coefficient for phytoplankton growth (dimensionless)
  PARAMETER(beta);          // Saturation coefficient for nutrient recycling
  PARAMETER(I_L);         // Light intensity modifier scaling nutrient uptake
  PARAMETER(T_opt);       // Optimal temperature for phytoplankton growth (°C)
  PARAMETER(T_sigma);     // Temperature sensitivity standard deviation (°C)
  vector<Type> temperature(n);
  if(temperature_aux.size() == 0) {
    for(int i = 0; i < n; i++){
      temperature(i) = T_opt;
    }
  } else {
    temperature = temperature_aux;
  }

  // Initialize predicted state vectors
  vector<Type> N_pred(n), P_pred(n), Z_pred(n);
  
  // Set initial conditions to the observed first data points
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);
  
  Type nll = 0.0;         // Negative log-likelihood accumulator
  Type eps = Type(1e-8);  // Small constant for numerical stability
  
  // Time integration using Euler's method, ensuring predictions use only past values
  for(int t = 1; t < n; t++){
    // Equation 1: Nutrient uptake (Michaelis-Menten form)
    // U = v_N * N_pred(t-1) / (K_N + N_pred(t-1) + eps)
    Type U = v_N * I_L * N_pred(t-1) / (K_N + N_pred(t-1) + eps);
    
    // Equation 2: Phytoplankton growth, grazing, and mortality
    Type T_effect = exp(- pow(temperature(t-1) - T_opt, Type(2)) / (Type(2) * T_sigma * T_sigma));
    Type growth_P = eps_P * U * P_pred(t-1) * exp(-alpha * P_pred(t-1)) * T_effect; // Growth contribution, including self-shading and temperature effect
    Type grazing = g_Z * pow(Z_pred(t-1), gamma) * P_pred(t-1) / (K_P + P_pred(t-1) + eps); // Grazing loss with predator interference
    Type mortality_P = d_P * P_pred(t-1);    // Mortality loss
    Type dP = growth_P - grazing - mortality_P;
    
    // Equation 3: Zooplankton dynamics (growth via grazing, mortality)
    Type dZ = eps_Z * grazing - d_Z * Z_pred(t-1);
    
    // Equation 4: Nutrient recycling and uptake with saturating recycling efficiency
    // dN/dt = - U * P_pred(t-1) + r*(P_pred(t-1) + Z_pred(t-1))/(1 + beta*(P_pred(t-1) + Z_pred(t-1)))
    Type dN = - U * P_pred(t-1) + r*(P_pred(t-1) + Z_pred(t-1))/(1 + beta*(P_pred(t-1) + Z_pred(t-1)));
    
    // Use time step difference (dt) for integration
    Type dt = time(t) - time(t-1);
    N_pred(t) = N_pred(t-1) + dN * dt;
    P_pred(t) = P_pred(t-1) + dP * dt;
    Z_pred(t) = Z_pred(t-1) + dZ * dt;
    
    // Equation 5: Likelihood calculations for each observation using a lognormal error distribution.
    // Using a fixed minimum standard deviation (0.1) for numerical stability.
    nll -= dnorm(log(N_dat(t)), log((N_pred(t) + eps) > eps ? (N_pred(t) + eps) : eps), Type(0.1), true);
    nll -= dnorm(log(P_dat(t)), log((P_pred(t) + eps) > eps ? (P_pred(t) + eps) : eps), Type(0.1), true);
    nll -= dnorm(log(Z_dat(t)), log((Z_pred(t) + eps) > eps ? (Z_pred(t) + eps) : eps), Type(0.1), true);
  }
  
  // Reporting predicted state variables for further analysis
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);

  return nll;
}
