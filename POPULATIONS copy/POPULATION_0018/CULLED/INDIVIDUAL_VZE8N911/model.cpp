#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS
  DATA_VECTOR(Time);  // Time in days
  DATA_VECTOR(N_dat);  // Observed nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);  // Observed phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);  // Observed zooplankton concentration (g C m^-3)
  
  // PARAMETERS - Phytoplankton dynamics
  PARAMETER(r_max);  // Maximum phytoplankton growth rate (day^-1) - determined from laboratory culture experiments
  PARAMETER(K_N);  // Half-saturation constant for nutrient uptake (g C m^-3) - from nutrient addition experiments
  PARAMETER(m_P);  // Phytoplankton natural mortality rate (day^-1) - from dilution experiments and literature
  
  // PARAMETERS - Zooplankton dynamics
  PARAMETER(g_max);  // Maximum zooplankton grazing rate (day^-1) - from feeding experiments
  PARAMETER(K_P);  // Half-saturation constant for grazing (g C m^-3) - from functional response experiments
  PARAMETER(epsilon);  // Zooplankton assimilation efficiency (dimensionless, 0-1) - from growth efficiency studies
  PARAMETER(m_Z);  // Zooplankton linear mortality rate (day^-1) - from population decline studies
  PARAMETER(m_Z2);  // Zooplankton quadratic mortality rate (day^-1 (g C m^-3)^-1) - represents predation pressure
  
  // PARAMETERS - Nutrient cycling
  PARAMETER(gamma_P);  // Nutrient recycling efficiency from phytoplankton mortality (dimensionless, 0-1) - from decomposition studies
  PARAMETER(gamma_Z);  // Nutrient recycling efficiency from zooplankton mortality and excretion (dimensionless, 0-1) - from excretion rate measurements
  PARAMETER(N_input);  // External nutrient input rate (g C m^-3 day^-1) - from mixing/upwelling estimates
  
  // PARAMETERS - Light limitation (NEW)
  PARAMETER(I_0);  // Surface light intensity (dimensionless, relative units) - from irradiance measurements
  PARAMETER(K_I);  // Half-saturation constant for light limitation (dimensionless) - from photosynthesis-irradiance curves
  PARAMETER(k_w);  // Light attenuation coefficient due to phytoplankton self-shading (m^2 (g C)^-1) - from optical measurements
  
  // PARAMETERS - Observation error
  PARAMETER(log_sigma_N);  // Log-scale standard deviation for nutrient observations
  PARAMETER(log_sigma_P);  // Log-scale standard deviation for phytoplankton observations
  PARAMETER(log_sigma_Z);  // Log-scale standard deviation for zooplankton observations
  
  // Transform log-scale parameters to natural scale
  Type sigma_N = exp(log_sigma_N);  // Standard deviation for nutrient concentration observations
  Type sigma_P = exp(log_sigma_P);  // Standard deviation for phytoplankton concentration observations
  Type sigma_Z = exp(log_sigma_Z);  // Standard deviation for zooplankton concentration observations
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(1e-4);  // Minimum allowed standard deviation (g C m^-3)
  sigma_N = sigma_N + min_sigma;  // Ensure sigma_N is bounded away from zero
  sigma_P = sigma_P + min_sigma;  // Ensure sigma_P is bounded away from zero
  sigma_Z = sigma_Z + min_sigma;  // Ensure sigma_Z is bounded away from zero
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);  // Small constant for numerical stability
  
  // INITIALIZE PREDICTION VECTORS
  int n_obs = Time.size();  // Number of time observations
  vector<Type> N_pred(n_obs);  // Predicted nutrient concentrations
  vector<Type> P_pred(n_obs);  // Predicted phytoplankton concentrations
  vector<Type> Z_pred(n_obs);  // Predicted zooplankton concentrations
  
  // Set initial conditions from first observation
  N_pred(0) = N_dat(0);  // Initialize nutrient from first data point
  P_pred(0) = P_dat(0);  // Initialize phytoplankton from first data point
  Z_pred(0) = Z_dat(0);  // Initialize zooplankton from first data point
  
  // FORWARD SIMULATION using Euler integration
  for(int i = 1; i < n_obs; i++) {
    Type dt = Time(i) - Time(i-1);  // Time step size (days)
    
    // Get previous time step values (avoid data leakage)
    Type N_prev = N_pred(i-1);  // Nutrient concentration at previous time step
    Type P_prev = P_pred(i-1);  // Phytoplankton concentration at previous time step
    Type Z_prev = Z_pred(i-1);  // Zooplankton concentration at previous time step
    
    // Ensure non-negative concentrations with smooth lower bound
    N_prev = N_prev + eps;  // Prevent negative nutrient values
    P_prev = P_prev + eps;  // Prevent negative phytoplankton values
    Z_prev = Z_prev + eps;  // Prevent negative zooplankton values
    
    // EQUATION 1: Light availability with self-shading (Beer-Lambert law)
    Type I_available = I_0 * exp(-k_w * P_prev);  // Light intensity after attenuation by phytoplankton (dimensionless)
    
    // EQUATION 2: Light limitation factor (Monod kinetics)
    Type light_limitation = I_available / (K_I + I_available + eps);  // Fraction of maximum growth achievable given light (dimensionless, 0-1)
    
    // EQUATION 3: Nutrient limitation factor (Michaelis-Menten kinetics)
    Type nutrient_limitation = N_prev / (K_N + N_prev + eps);  // Fraction of maximum growth achievable given nutrients (dimensionless, 0-1)
    
    // EQUATION 4: Nutrient uptake rate by phytoplankton (co-limited by light and nutrients)
    Type nutrient_uptake = r_max * nutrient_limitation * light_limitation * P_prev;  // Light and nutrient co-limited phytoplankton growth (g C m^-3 day^-1)
    
    // EQUATION 5: Phytoplankton mortality
    Type phyto_mortality = m_P * P_prev;  // Natural phytoplankton death rate (g C m^-3 day^-1)
    
    // EQUATION 6: Zooplankton grazing rate (Holling Type II functional response)
    Type grazing = g_max * (P_prev / (K_P + P_prev + eps)) * Z_prev;  // Zooplankton consumption of phytoplankton (g C m^-3 day^-1)
    
    // EQUATION 7: Zooplankton mortality (linear + quadratic density dependence)
    Type zoo_mortality = (m_Z * Z_prev) + (m_Z2 * Z_prev * Z_prev);  // Combined zooplankton mortality (g C m^-3 day^-1)
    
    // EQUATION 8: Nutrient recycling from phytoplankton mortality
    Type nutrient_from_phyto = gamma_P * phyto_mortality;  // Nutrients released from dead phytoplankton (g C m^-3 day^-1)
    
    // EQUATION 9: Nutrient recycling from zooplankton (excretion + mortality)
    Type nutrient_from_zoo = gamma_Z * zoo_mortality + (Type(1.0) - epsilon) * grazing;  // Nutrients from zooplankton waste and inefficient feeding (g C m^-3 day^-1)
    
    // EQUATION 10: Rate of change of nutrient concentration
    Type dN_dt = N_input - nutrient_uptake + nutrient_from_phyto + nutrient_from_zoo;  // Net nutrient change (g C m^-3 day^-1)
    
    // EQUATION 11: Rate of change of phytoplankton concentration
    Type dP_dt = nutrient_uptake - phyto_mortality - grazing;  // Net phytoplankton change (g C m^-3 day^-1)
    
    // EQUATION 12: Rate of change of zooplankton concentration
    Type dZ_dt = epsilon * grazing - zoo_mortality;  // Net zooplankton change (g C m^-3 day^-1)
    
    // Update predictions using Euler method
    N_pred(i) = N_prev + dt * dN_dt;  // Forward Euler step for nutrients
    P_pred(i) = P_prev + dt * dP_dt;  // Forward Euler step for phytoplankton
    Z_pred(i) = Z_prev + dt * dZ_dt;  // Forward Euler step for zooplankton
    
    // Apply soft lower bounds to prevent negative values
    N_pred(i) = N_pred(i) + eps;  // Ensure nutrient stays positive
    P_pred(i) = P_pred(i) + eps;  // Ensure phytoplankton stays positive
    Z_pred(i) = Z_pred(i) + eps;  // Ensure zooplankton stays positive
  }
  
  // LIKELIHOOD CALCULATION
  Type nll = Type(0.0);  // Initialize negative log-likelihood
  
  // Add observation likelihoods for all time points
  for(int i = 0; i < n_obs; i++) {
    // Nutrient observations (normal distribution)
    nll -= dnorm(N_dat(i), N_pred(i), sigma_N, true);  // Negative log-likelihood contribution from nutrient data
    
    // Phytoplankton observations (normal distribution)
    nll -= dnorm(P_dat(i), P_pred(i), sigma_P, true);  // Negative log-likelihood contribution from phytoplankton data
    
    // Zooplankton observations (normal distribution)
    nll -= dnorm(Z_dat(i), Z_pred(i), sigma_Z, true);  // Negative log-likelihood contribution from zooplankton data
  }
  
  // PARAMETER CONSTRAINTS using smooth penalties
  // Ensure parameters stay within biologically reasonable ranges
  
  // Growth and mortality rates should be positive
  nll -= dnorm(r_max, Type(0.5), Type(2.0), true);  // Soft prior: phytoplankton growth rate centered at 0.5 day^-1
  nll -= dnorm(g_max, Type(0.5), Type(2.0), true);  // Soft prior: zooplankton grazing rate centered at 0.5 day^-1
  nll -= dnorm(m_P, Type(0.05), Type(0.5), true);  // Soft prior: phytoplankton mortality centered at 0.05 day^-1
  nll -= dnorm(m_Z, Type(0.05), Type(0.5), true);  // Soft prior: zooplankton linear mortality centered at 0.05 day^-1
  
  // Half-saturation constants should be positive and reasonable
  nll -= dnorm(K_N, Type(0.1), Type(1.0), true);  // Soft prior: nutrient half-saturation centered at 0.1 g C m^-3
  nll -= dnorm(K_P, Type(0.1), Type(1.0), true);  // Soft prior: grazing half-saturation centered at 0.1 g C m^-3
  
  // Light parameters should be positive and reasonable
  nll -= dnorm(I_0, Type(1.0), Type(1.0), true);  // Soft prior: surface light centered at 1.0 (average conditions)
  nll -= dnorm(K_I, Type(0.2), Type(0.5), true);  // Soft prior: light half-saturation centered at 0.2
  nll -= dnorm(k_w, Type(0.1), Type(0.5), true);  // Soft prior: self-shading coefficient centered at 0.1
  
  // Efficiencies should be between 0 and 1 (using logit-like penalty)
  Type epsilon_penalty = -log(epsilon + eps) - log(Type(1.0) - epsilon + eps);  // Penalty to keep epsilon in (0,1)
  Type gamma_P_penalty = -log(gamma_P + eps) - log(Type(1.0) - gamma_P + eps);  // Penalty to keep gamma_P in (0,1)
  Type gamma_Z_penalty = -log(gamma_Z + eps) - log(Type(1.0) - gamma_Z + eps);  // Penalty to keep gamma_Z in (0,1)
  nll += Type(0.1) * (epsilon_penalty + gamma_P_penalty + gamma_Z_penalty);  // Add efficiency penalties with small weight
  
  // Quadratic mortality should be small and positive
  nll -= dnorm(m_Z2, Type(0.1), Type(1.0), true);  // Soft prior: quadratic mortality centered at 0.1
  
  // Nutrient input should be positive
  nll -= dnorm(N_input, Type(0.01), Type(0.5), true);  // Soft prior: nutrient input centered at 0.01 g C m^-3 day^-1
  
  // REPORT PREDICTIONS
  REPORT(N_pred);  // Report predicted nutrient concentrations
  REPORT(P_pred);  // Report predicted phytoplankton concentrations
  REPORT(Z_pred);  // Report predicted zooplankton concentrations
  
  // REPORT INTERMEDIATE CALCULATIONS (for diagnostics)
  vector<Type> I_available_vec(n_obs);  // Vector to store light availability at each time step
  vector<Type> light_limitation_vec(n_obs);  // Vector to store light limitation factor at each time step
  for(int i = 0; i < n_obs; i++) {
    Type P_temp = P_pred(i) + eps;  // Get phytoplankton concentration
    I_available_vec(i) = I_0 * exp(-k_w * P_temp);  // Calculate available light
    light_limitation_vec(i) = I_available_vec(i) / (K_I + I_available_vec(i) + eps);  // Calculate light limitation
  }
  REPORT(I_available_vec);  // Report light availability over time
  REPORT(light_limitation_vec);  // Report light limitation factor over time
  
  // REPORT PARAMETERS
  REPORT(r_max);  // Report maximum phytoplankton growth rate
  REPORT(K_N);  // Report nutrient half-saturation constant
  REPORT(m_P);  // Report phytoplankton mortality rate
  REPORT(g_max);  // Report maximum grazing rate
  REPORT(K_P);  // Report grazing half-saturation constant
  REPORT(epsilon);  // Report assimilation efficiency
  REPORT(m_Z);  // Report zooplankton linear mortality
  REPORT(m_Z2);  // Report zooplankton quadratic mortality
  REPORT(gamma_P);  // Report phytoplankton recycling efficiency
  REPORT(gamma_Z);  // Report zooplankton recycling efficiency
  REPORT(N_input);  // Report nutrient input rate
  REPORT(I_0);  // Report surface light intensity
  REPORT(K_I);  // Report light half-saturation constant
  REPORT(k_w);  // Report self-shading coefficient
  REPORT(sigma_N);  // Report nutrient observation error
  REPORT(sigma_P);  // Report phytoplankton observation error
  REPORT(sigma_Z);  // Report zooplankton observation error
  
  return nll;  // Return total negative log-likelihood
}
