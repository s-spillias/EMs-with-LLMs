#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Time);        // Time points (days)
  DATA_VECTOR(N_dat);       // Nutrient observations (g C m^-3)
  DATA_VECTOR(P_dat);       // Phytoplankton observations (g C m^-3)
  DATA_VECTOR(Z_dat);       // Zooplankton observations (g C m^-3)
  
  // Parameters
  PARAMETER(mu_max);        // Maximum phytoplankton growth rate (day^-1)
  PARAMETER(k_N);          // Half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(g_max);        // Maximum zooplankton grazing rate (day^-1)
  PARAMETER(k_P);          // Half-saturation constant for grazing (g C m^-3)
  PARAMETER(alpha);        // Zooplankton assimilation efficiency (dimensionless)
  PARAMETER(m_P);          // Phytoplankton mortality rate (day^-1)
  PARAMETER(m_Z);          // Zooplankton mortality rate (day^-1)
  PARAMETER(sigma_N);      // Standard deviation for nutrient observations
  PARAMETER(sigma_P);      // Standard deviation for phytoplankton observations
  PARAMETER(sigma_Z);      // Standard deviation for zooplankton observations

  // Small constant to prevent division by zero
  const Type eps = Type(1e-8);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Parameter bounds using informative priors
  nll -= dnorm(mu_max, Type(1.0), Type(0.2), true);    // Growth rate ~ N(1.0, 0.2)
  nll -= dnorm(k_N, Type(0.1), Type(0.02), true);      // Half-saturation ~ N(0.1, 0.02)
  nll -= dnorm(g_max, Type(0.4), Type(0.1), true);     // Grazing rate ~ N(0.4, 0.1)
  nll -= dnorm(k_P, Type(0.1), Type(0.02), true);      // Grazing half-sat ~ N(0.1, 0.02)
  nll -= dbeta(alpha, Type(3), Type(7), true);         // Assimilation efficiency ~ Beta(3,7)
  nll -= dnorm(m_P, Type(0.1), Type(0.02), true);      // Mortality rates ~ N(0.1, 0.02)
  nll -= dnorm(m_Z, Type(0.1), Type(0.02), true);
  
  // Vectors for model predictions
  vector<Type> N_pred(Time.size());
  vector<Type> P_pred(Time.size());
  vector<Type> Z_pred(Time.size());
  
  // Initial conditions
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);
  
  // Numerical integration using Euler method
  for(int t = 1; t < Time.size(); t++) {
    Type dt = Time(t) - Time(t-1);
    
    // 1. Nutrient uptake by phytoplankton (Michaelis-Menten)
    Type uptake = mu_max * N_pred(t-1)/(k_N + N_pred(t-1)) * P_pred(t-1);
    
    // 2. Zooplankton grazing (Holling Type II)
    Type grazing = g_max * P_pred(t-1)/(k_P + P_pred(t-1)) * Z_pred(t-1);
    
    // 3. System of differential equations
    Type dN = -uptake + m_P*P_pred(t-1) + m_Z*Z_pred(t-1);
    Type dP = uptake - grazing - m_P*P_pred(t-1);
    Type dZ = alpha*grazing - m_Z*Z_pred(t-1);
    
    // 4. Update state variables
    N_pred(t) = N_pred(t-1) + dN*dt;
    P_pred(t) = P_pred(t-1) + dP*dt;
    Z_pred(t) = Z_pred(t-1) + dZ*dt;
    
    // 5. Ensure positive values and numerical stability
    N_pred(t) = N_pred(t) < eps ? eps : N_pred(t);
    P_pred(t) = P_pred(t) < eps ? eps : P_pred(t);
    Z_pred(t) = Z_pred(t) < eps ? eps : Z_pred(t);
  }
  
  // Likelihood calculations using lognormal distribution
  for(int t = 0; t < Time.size(); t++) {
    if(N_dat(t) > eps && N_pred(t) > eps) {
      nll -= dnorm(log(N_dat(t)), log(N_pred(t)), sigma_N, true);
    }
    if(P_dat(t) > eps && P_pred(t) > eps) {
      nll -= dnorm(log(P_dat(t)), log(P_pred(t)), sigma_P, true);
    }
    if(Z_dat(t) > eps && Z_pred(t) > eps) {
      nll -= dnorm(log(Z_dat(t)), log(Z_pred(t)), sigma_Z, true);
    }
  }
  
  // Report predictions
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  
  return nll;
}
