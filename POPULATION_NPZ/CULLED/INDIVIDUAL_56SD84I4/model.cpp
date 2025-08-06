#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Time);            // Time points (days)
  DATA_VECTOR(N_dat);           // Nutrient observations (g C m^-3)
  DATA_VECTOR(P_dat);           // Phytoplankton observations (g C m^-3)
  DATA_VECTOR(Z_dat);           // Zooplankton observations (g C m^-3)
  
  // Parameters
  PARAMETER(log_r);             // Log maximum phytoplankton growth rate (day^-1)
  PARAMETER(log_Kn);            // Log half-saturation for nutrient uptake (g C m^-3)
  PARAMETER(log_g);             // Log maximum grazing rate (day^-1)
  PARAMETER(log_Kp);            // Log half-saturation for grazing (g C m^-3)
  PARAMETER(log_mp);            // Log phytoplankton mortality rate (day^-1)
  PARAMETER(log_mz);            // Log zooplankton mortality rate (day^-1)
  PARAMETER(logit_gamma);       // Logit recycling efficiency (dimensionless)
  PARAMETER(log_sigma);         // Log observation error SD

  // Transform parameters with bounds checking
  Type r = exp(log_r) + Type(1e-8);
  Type Kn = exp(log_Kn) + Type(1e-8);
  Type g = exp(log_g) + Type(1e-8);
  Type Kp = exp(log_Kp) + Type(1e-8);
  Type mp = exp(log_mp) + Type(1e-8);
  Type mz = exp(log_mz) + Type(1e-8);
  Type gamma = Type(0.99)/(Type(1.0) + exp(-logit_gamma)) + Type(0.01); // Bound between 0.01 and 1
  Type sigma = exp(log_sigma) + Type(1e-8);
  
  // Small constant to prevent division by zero
  const Type eps = Type(1e-6);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;

  // Vectors to store predictions
  vector<Type> N_pred(Time.size());
  vector<Type> P_pred(Time.size());
  vector<Type> Z_pred(Time.size());
  
  // Initial conditions
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);
  
  // Time integration using Euler method
  for(int t = 1; t < Time.size(); t++) {
    Type dt = Time(t) - Time(t-1);
    
    // Previous state
    Type N = N_pred(t-1);
    Type P = P_pred(t-1);
    Type Z = Z_pred(t-1);
    
    // 1. Nutrient uptake by phytoplankton (Michaelis-Menten)
    Type uptake = r * N/(N + Kn + eps) * P;
    
    // 2. Zooplankton grazing (Holling Type II)
    Type grazing = g * P/(P + Kp + eps) * Z;
    
    // 3. Mortality and recycling
    Type P_mort = mp * P;
    Type Z_mort = mz * Z;
    
    // 4. State updates with bounds
    N_pred(t) = N + dt * (-uptake + gamma*(P_mort + Z_mort));
    N_pred(t) = N_pred(t) > Type(0.0) ? N_pred(t) : Type(eps);
    
    P_pred(t) = P + dt * (uptake - grazing - P_mort);
    P_pred(t) = P_pred(t) > Type(0.0) ? P_pred(t) : Type(eps);
    
    Z_pred(t) = Z + dt * (grazing - Z_mort);
    Z_pred(t) = Z_pred(t) > Type(0.0) ? Z_pred(t) : Type(eps);
  }
  
  // Observation model using lognormal distribution with robust error handling
  for(int t = 0; t < Time.size(); t++) {
    Type N_obs = N_dat(t) + eps;
    Type P_obs = P_dat(t) + eps;
    Type Z_obs = Z_dat(t) + eps;
    
    Type N_model = N_pred(t) + eps;
    Type P_model = P_pred(t) + eps;
    Type Z_model = Z_pred(t) + eps;
    
    nll -= dnorm(log(N_obs), log(N_model), sigma, true);
    nll -= dnorm(log(P_obs), log(P_model), sigma, true);
    nll -= dnorm(log(Z_obs), log(Z_model), sigma, true);
  }
  
  // Penalties to keep parameters in biologically reasonable ranges
  nll += 0.5 * pow(log_r - log(2.0), 2);     // Prior: r ~ 2.0 day^-1
  nll += 0.5 * pow(log_Kn - log(0.1), 2);    // Prior: Kn ~ 0.1 g C m^-3
  nll += 0.5 * pow(log_g - log(1.0), 2);     // Prior: g ~ 1.0 day^-1
  nll += 0.5 * pow(log_Kp - log(0.1), 2);    // Prior: Kp ~ 0.1 g C m^-3
  nll += 0.5 * pow(log_mp - log(0.1), 2);    // Prior: mp ~ 0.1 day^-1
  nll += 0.5 * pow(log_mz - log(0.1), 2);    // Prior: mz ~ 0.1 day^-1
  
  // Report predictions
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  REPORT(r);
  REPORT(Kn);
  REPORT(g);
  REPORT(Kp);
  REPORT(mp);
  REPORT(mz);
  REPORT(gamma);
  REPORT(sigma);
  
  return nll;
}
