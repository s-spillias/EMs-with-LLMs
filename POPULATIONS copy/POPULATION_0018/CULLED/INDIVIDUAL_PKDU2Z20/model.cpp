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
  PARAMETER(r_max);  // Maximum phytoplankton growth rate at reference temperature (day^-1)
  PARAMETER(K_N);  // Half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(m_P);  // Phytoplankton natural mortality rate at reference temperature (day^-1)
  
  // PARAMETERS - Zooplankton dynamics
  PARAMETER(g_max);  // Maximum zooplankton grazing rate at reference temperature (day^-1)
  PARAMETER(K_P);  // Half-saturation constant for grazing (g C m^-3)
  PARAMETER(epsilon);  // Zooplankton assimilation efficiency (dimensionless, 0-1)
  PARAMETER(m_Z);  // Zooplankton linear mortality rate at reference temperature (day^-1)
  PARAMETER(m_Z2);  // Zooplankton quadratic mortality rate at reference temperature (day^-1 (g C m^-3)^-1)
  
  // PARAMETERS - Nutrient cycling
  PARAMETER(gamma_P);  // Nutrient recycling efficiency from phytoplankton mortality (dimensionless, 0-1)
  PARAMETER(gamma_Z);  // Nutrient recycling efficiency from zooplankton mortality and excretion (dimensionless, 0-1)
  PARAMETER(N_input);  // External nutrient input rate (g C m^-3 day^-1)
  
  // PARAMETERS - Light limitation
  PARAMETER(I_0);  // Surface light intensity (μmol photons m^-2 s^-1)
  PARAMETER(K_I);  // Half-saturation constant for light limitation (μmol photons m^-2 s^-1)
  PARAMETER(k_w);  // Background water attenuation coefficient (m^-1)
  PARAMETER(k_c);  // Phytoplankton self-shading coefficient (m^2 (g C)^-1)
  PARAMETER(H);  // Mixed layer depth (m)
  
  // PARAMETERS - Temperature dependence (NEW)
  PARAMETER(Q10);  // Temperature coefficient for biological rates (dimensionless, typically 2-3)
  PARAMETER(T_ref);  // Reference temperature for rate parameters (°C, typically 15-20°C)
  
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
    
    // Use reference temperature as constant (temperature forcing not yet implemented in data)
    Type T_current = T_ref;  // Current mixed layer temperature (°C) - defaults to reference temperature
    
    // Ensure non-negative concentrations with smooth lower bound
    N_prev = N_prev + eps;  // Prevent negative nutrient values
    P_prev = P_prev + eps;  // Prevent negative phytoplankton values
    Z_prev = Z_prev + eps;  // Prevent negative zooplankton values
    
    // EQUATION 0: Calculate temperature effect on biological rates using Q10 formulation
    // f_temp = Q10^((T - T_ref)/10)
    Type temp_diff = (T_current - T_ref) / Type(10.0);  // Temperature difference in units of 10°C
    Type f_temp = pow(Q10, temp_diff);  // Temperature multiplier for biological rates (dimensionless)
    
    // Apply temperature correction to all biological rate parameters
    Type r_max_temp = r_max * f_temp;  // Temperature-corrected maximum phytoplankton growth rate (day^-1)
    Type g_max_temp = g_max * f_temp;  // Temperature-corrected maximum grazing rate (day^-1)
    Type m_P_temp = m_P * f_temp;  // Temperature-corrected phytoplankton mortality rate (day^-1)
    Type m_Z_temp = m_Z * f_temp;  // Temperature-corrected zooplankton linear mortality rate (day^-1)
    Type m_Z2_temp = m_Z2 * f_temp;  // Temperature-corrected zooplankton quadratic mortality rate (day^-1 (g C m^-3)^-1)
    
    // EQUATION 1a: Calculate light limitation with self-shading
    // Total light attenuation coefficient (background + phytoplankton self-shading)
    Type k_total = k_w + k_c * P_prev;  // Total attenuation (m^-1)
    
    // Average light intensity in mixed layer using exponential attenuation
    // Integrated from 0 to H and divided by H
    Type I_avg = I_0 * (Type(1.0) - exp(-k_total * H)) / (k_total * H + eps);  // Average PAR in mixed layer (μmol photons m^-2 s^-1)
    
    // Light limitation factor (Michaelis-Menten form)
    Type f_light = I_avg / (I_avg + K_I + eps);  // Light limitation factor (dimensionless, 0-1)
    
    // EQUATION 1b: Nutrient limitation factor
    Type f_nutrient = N_prev / (K_N + N_prev + eps);  // Nutrient limitation factor (dimensionless, 0-1)
    
    // EQUATION 1c: Combined nutrient uptake rate with light, nutrient, AND temperature limitation
    Type nutrient_uptake = r_max_temp * f_nutrient * f_light * P_prev;  // Temperature, light and nutrient co-limited phytoplankton growth (g C m^-3 day^-1)
    
    // EQUATION 2: Phytoplankton mortality (temperature-dependent)
    Type phyto_mortality = m_P_temp * P_prev;  // Temperature-corrected natural phytoplankton death rate (g C m^-3 day^-1)
    
    // EQUATION 3: Zooplankton grazing rate (Holling Type II functional response, temperature-dependent)
    Type grazing = g_max_temp * (P_prev / (K_P + P_prev + eps)) * Z_prev;  // Temperature-corrected zooplankton consumption of phytoplankton (g C m^-3 day^-1)
    
    // EQUATION 4: Zooplankton mortality (linear + quadratic density dependence, temperature-dependent)
    Type zoo_mortality = (m_Z_temp * Z_prev) + (m_Z2_temp * Z_prev * Z_prev);  // Temperature-corrected combined zooplankton mortality (g C m^-3 day^-1)
    
    // EQUATION 5: Nutrient recycling from phytoplankton mortality
    Type nutrient_from_phyto = gamma_P * phyto_mortality;  // Nutrients released from dead phytoplankton (g C m^-3 day^-1)
    
    // EQUATION 6: Nutrient recycling from zooplankton (excretion + mortality)
    Type nutrient_from_zoo = gamma_Z * zoo_mortality + (Type(1.0) - epsilon) * grazing;  // Nutrients from zooplankton waste and inefficient feeding (g C m^-3 day^-1)
    
    // EQUATION 7: Rate of change of nutrient concentration
    Type dN_dt = N_input - nutrient_uptake + nutrient_from_phyto + nutrient_from_zoo;  // Net nutrient change (g C m^-3 day^-1)
    
    // EQUATION 8: Rate of change of phytoplankton concentration
    Type dP_dt = nutrient_uptake - phyto_mortality - grazing;  // Net phytoplankton change (g C m^-3 day^-1)
    
    // EQUATION 9: Rate of change of zooplankton concentration
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
  nll -= dnorm(I_0, Type(400.0), Type(500.0), true);  // Soft prior: surface light centered at 400 μmol photons m^-2 s^-1
  nll -= dnorm(K_I, Type(50.0), Type(100.0), true);  // Soft prior: light half-saturation centered at 50 μmol photons m^-2 s^-1
  nll -= dnorm(k_w, Type(0.04), Type(0.1), true);  // Soft prior: water attenuation centered at 0.04 m^-1
  nll -= dnorm(k_c, Type(0.03), Type(0.05), true);  // Soft prior: phytoplankton attenuation centered at 0.03 m^2 (g C)^-1
  nll -= dnorm(H, Type(50.0), Type(50.0), true);  // Soft prior: mixed layer depth centered at 50 m
  
  // Temperature parameters should be reasonable
  nll -= dnorm(Q10, Type(2.0), Type(1.0), true);  // Soft prior: Q10 centered at 2.0 (typical for marine organisms)
  nll -= dnorm(T_ref, Type(15.0), Type(10.0), true);  // Soft prior: reference temperature centered at 15°C (typical oceanic)
  
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
  REPORT(k_w);  // Report water attenuation coefficient
  REPORT(k_c);  // Report phytoplankton self-shading coefficient
  REPORT(H);  // Report mixed layer depth
  REPORT(Q10);  // Report temperature coefficient
  REPORT(T_ref);  // Report reference temperature
  REPORT(sigma_N);  // Report nutrient observation error
  REPORT(sigma_P);  // Report phytoplankton observation error
  REPORT(sigma_Z);  // Report zooplankton observation error
  
  return nll;  // Return total negative log-likelihood
}
