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
  PARAMETER(epsilon_min);  // Minimum zooplankton assimilation efficiency under nutrient limitation (dimensionless, 0-1)
  PARAMETER(epsilon_max);  // Maximum zooplankton assimilation efficiency under nutrient-replete conditions (dimensionless, 0-1)
  PARAMETER(m_Z);  // Zooplankton linear mortality rate at reference temperature (day^-1)
  PARAMETER(m_Z2);  // Zooplankton quadratic mortality rate (day^-1 (g C m^-3)^-1)
  
  // PARAMETERS - Nutrient cycling
  PARAMETER(gamma_P);  // Nutrient recycling efficiency from phytoplankton mortality (dimensionless, 0-1)
  PARAMETER(gamma_Z);  // Nutrient recycling efficiency from zooplankton mortality and excretion (dimensionless, 0-1)
  PARAMETER(N_input);  // External nutrient input rate (g C m^-3 day^-1)
  
  // PARAMETERS - Light limitation with seasonal variation
  PARAMETER(I_0_mean);  // Annual mean surface light intensity (μmol photons m^-2 s^-1)
  PARAMETER(I_0_amplitude);  // Fractional amplitude of seasonal light variation (dimensionless, 0-1)
  PARAMETER(t_phase);  // Day of year when light peaks (days, 0-365)
  PARAMETER(K_I);  // Half-saturation constant for light-limited growth (μmol photons m^-2 s^-1)
  PARAMETER(k_w);  // Background light attenuation coefficient (m^-1)
  PARAMETER(k_c);  // Phytoplankton self-shading coefficient (m^2 (g C)^-1)
  PARAMETER(z_mix);  // Mixed layer depth (m)
  
  // PARAMETERS - Temperature dependence
  PARAMETER(Q10_phyto);  // Q10 coefficient for phytoplankton growth (dimensionless)
  PARAMETER(Q10_zoo);  // Q10 coefficient for zooplankton grazing and metabolism (dimensionless)
  PARAMETER(Q10_remin);  // Q10 coefficient for remineralization processes (dimensionless)
  PARAMETER(T_ref);  // Reference temperature for rate normalization (°C)
  PARAMETER(T_mean);  // Annual mean mixed layer temperature (°C)
  PARAMETER(T_amplitude);  // Amplitude of seasonal temperature variation (°C)
  PARAMETER(t_phase_temp);  // Day of year when temperature peaks (days, 0-365)
  
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
  
  // Mathematical constants
  Type pi = Type(3.14159265358979323846);  // Pi for seasonal calculations
  
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
    Type t_prev = Time(i-1);  // Time at previous step (for seasonal calculations)
    
    // Ensure non-negative concentrations with smooth lower bound
    N_prev = N_prev + eps;  // Prevent negative nutrient values
    P_prev = P_prev + eps;  // Prevent negative phytoplankton values
    Z_prev = Z_prev + eps;  // Prevent negative zooplankton values
    
    // EQUATION 0: Seasonal temperature variation (NEW)
    // T(t) = T_mean + T_amplitude * sin(2π * (t - t_phase_temp) / 365)
    // This creates an annual temperature cycle with peak at t_phase_temp
    // Typically t_phase_temp is 30-60 days after t_phase due to ocean thermal inertia
    Type year_length = Type(365.0);  // Days in a year
    Type temp_seasonal_angle = Type(2.0) * pi * (t_prev - t_phase_temp) / year_length;  // Phase-shifted angle for temperature (radians)
    Type T_prev = T_mean + T_amplitude * sin(temp_seasonal_angle);  // Time-varying mixed layer temperature (°C)
    
    // EQUATION 1: Temperature factors using Q10 formulation
    // Q10 formulation: rate(T) = rate(T_ref) * Q10^((T - T_ref)/10)
    // Now T_prev varies seasonally, so temperature factors will vary throughout the year
    Type temp_diff = (T_prev - T_ref) / Type(10.0);  // Temperature difference in units of 10°C
    Type f_temp_phyto = pow(Q10_phyto, temp_diff);  // Temperature factor for phytoplankton growth
    Type f_temp_zoo = pow(Q10_zoo, temp_diff);  // Temperature factor for zooplankton processes
    Type f_temp_remin = pow(Q10_remin, temp_diff);  // Temperature factor for remineralization
    
    // EQUATION 2: Seasonal surface light intensity (sinusoidal variation)
    // I_0(t) = I_0_mean * (1 + I_0_amplitude * sin(2π * (t - t_phase) / 365))
    // This creates an annual cycle with peak at t_phase and trough 182.5 days later
    Type light_seasonal_angle = Type(2.0) * pi * (t_prev - t_phase) / year_length;  // Phase-shifted angle for light (radians)
    Type I_0 = I_0_mean * (Type(1.0) + I_0_amplitude * sin(light_seasonal_angle));  // Time-varying surface light intensity (μmol photons m^-2 s^-1)
    
    // Ensure I_0 remains positive (should be guaranteed by 0 <= I_0_amplitude <= 1, but add safety)
    I_0 = I_0 + eps;  // Prevent negative light values
    
    // EQUATION 3: Light availability in the mixed layer (exponential attenuation with self-shading)
    Type attenuation_coef = k_w + k_c * P_prev;  // Total light attenuation coefficient (m^-1)
    Type light_at_depth = I_0 * exp(-attenuation_coef * z_mix);  // Light at bottom of mixed layer (μmol photons m^-2 s^-1)
    Type I_avg = I_0 * (Type(1.0) - exp(-attenuation_coef * z_mix)) / (attenuation_coef * z_mix + eps);  // Depth-averaged light intensity (μmol photons m^-2 s^-1)
    
    // EQUATION 4: Light limitation factor (Monod/Michaelis-Menten type saturation)
    Type light_limitation = I_avg / (K_I + I_avg + eps);  // Light limitation factor (dimensionless, 0-1)
    
    // EQUATION 5: Nutrient limitation factor (Michaelis-Menten kinetics)
    Type nutrient_limitation = N_prev / (K_N + N_prev + eps);  // Nutrient limitation factor (dimensionless, 0-1)
    
    // EQUATION 6: Temperature-dependent phytoplankton growth rate
    // Growth is co-limited by light, nutrients, AND temperature
    Type phyto_growth = r_max * f_temp_phyto * nutrient_limitation * light_limitation * P_prev;  // Temperature-modified phytoplankton growth (g C m^-3 day^-1)
    
    // EQUATION 7: Nutrient uptake rate by phytoplankton (equals growth rate in carbon units)
    Type nutrient_uptake = phyto_growth;  // Nutrient consumed equals phytoplankton growth (g C m^-3 day^-1)
    
    // EQUATION 8: Temperature-dependent phytoplankton mortality
    Type phyto_mortality = m_P * f_temp_remin * P_prev;  // Temperature-modified phytoplankton death rate (g C m^-3 day^-1)
    
    // EQUATION 9: Temperature-dependent zooplankton grazing rate (Holling Type II functional response)
    Type grazing = g_max * f_temp_zoo * (P_prev / (K_P + P_prev + eps)) * Z_prev;  // Temperature-modified zooplankton consumption (g C m^-3 day^-1)
    
    // EQUATION 10: Variable assimilation efficiency based on phytoplankton nutrient status
    // When phytoplankton are nutrient-limited, they have higher C:N ratios (lower quality food)
    // This reduces zooplankton assimilation efficiency
    // epsilon_effective = epsilon_min + (epsilon_max - epsilon_min) * nutrient_limitation
    Type epsilon_effective = epsilon_min + (epsilon_max - epsilon_min) * nutrient_limitation;  // Food quality-dependent assimilation efficiency (dimensionless, epsilon_min to epsilon_max)
    
    // EQUATION 11: Temperature-dependent zooplankton mortality (linear + quadratic density dependence)
    Type zoo_mortality = (m_Z * f_temp_zoo * Z_prev) + (m_Z2 * Z_prev * Z_prev);  // Temperature-modified zooplankton mortality (g C m^-3 day^-1)
    
    // EQUATION 12: Temperature-dependent nutrient recycling from phytoplankton mortality
    Type nutrient_from_phyto = gamma_P * f_temp_remin * phyto_mortality;  // Temperature-modified nutrients from dead phytoplankton (g C m^-3 day^-1)
    
    // EQUATION 13: Temperature-dependent nutrient recycling from zooplankton
    // Note: epsilon_effective is now variable, so unassimilated fraction (1 - epsilon_effective) varies with food quality
    Type nutrient_from_zoo = gamma_Z * f_temp_remin * zoo_mortality + (Type(1.0) - epsilon_effective) * grazing;  // Temperature-modified nutrients from zooplankton (g C m^-3 day^-1)
    
    // EQUATION 14: Rate of change of nutrient concentration
    Type dN_dt = N_input - nutrient_uptake + nutrient_from_phyto + nutrient_from_zoo;  // Net nutrient change (g C m^-3 day^-1)
    
    // EQUATION 15: Rate of change of phytoplankton concentration
    Type dP_dt = phyto_growth - phyto_mortality - grazing;  // Net phytoplankton change (g C m^-3 day^-1)
    
    // EQUATION 16: Rate of change of zooplankton concentration
    // Now uses variable epsilon_effective instead of constant epsilon
    Type dZ_dt = epsilon_effective * grazing - zoo_mortality;  // Net zooplankton change (g C m^-3 day^-1)
    
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
  
  // Growth and mortality rates should be positive
  nll -= dnorm(r_max, Type(0.5), Type(2.0), true);  // Soft prior: phytoplankton growth rate centered at 0.5 day^-1
  nll -= dnorm(g_max, Type(0.5), Type(2.0), true);  // Soft prior: zooplankton grazing rate centered at 0.5 day^-1
  nll -= dnorm(m_P, Type(0.05), Type(0.5), true);  // Soft prior: phytoplankton mortality centered at 0.05 day^-1
  nll -= dnorm(m_Z, Type(0.05), Type(0.5), true);  // Soft prior: zooplankton linear mortality centered at 0.05 day^-1
  
  // Half-saturation constants should be positive and reasonable
  nll -= dnorm(K_N, Type(0.1), Type(1.0), true);  // Soft prior: nutrient half-saturation centered at 0.1 g C m^-3
  nll -= dnorm(K_P, Type(0.1), Type(1.0), true);  // Soft prior: grazing half-saturation centered at 0.1 g C m^-3
  nll -= dnorm(K_I, Type(50.0), Type(100.0), true);  // Soft prior: light half-saturation centered at 50 μmol photons m^-2 s^-1
  
  // Light parameters should be positive and reasonable
  nll -= dnorm(I_0_mean, Type(400.0), Type(500.0), true);  // Soft prior: mean surface light intensity centered at 400 μmol photons m^-2 s^-1
  nll -= dnorm(k_w, Type(0.04), Type(0.1), true);  // Soft prior: water attenuation centered at 0.04 m^-1
  nll -= dnorm(k_c, Type(0.03), Type(0.1), true);  // Soft prior: phytoplankton attenuation centered at 0.03 m^2 (g C)^-1
  nll -= dnorm(z_mix, Type(20.0), Type(50.0), true);  // Soft prior: mixed layer depth centered at 20 m
  
  // Seasonal light variation parameters
  nll -= dnorm(I_0_amplitude, Type(0.4), Type(0.3), true);  // Soft prior: seasonal amplitude centered at 0.4 (±40% variation)
  nll -= dnorm(t_phase, Type(172.0), Type(100.0), true);  // Soft prior: light peak centered at day 172 (June 21, Northern Hemisphere)
  
  // Temperature parameters (Q10 values should be between 1 and 5, typically 1.5-4)
  nll -= dnorm(Q10_phyto, Type(2.0), Type(1.0), true);  // Soft prior: phytoplankton Q10 centered at 2.0
  nll -= dnorm(Q10_zoo, Type(2.5), Type(1.0), true);  // Soft prior: zooplankton Q10 centered at 2.5
  nll -= dnorm(Q10_remin, Type(3.0), Type(1.5), true);  // Soft prior: remineralization Q10 centered at 3.0
  nll -= dnorm(T_ref, Type(15.0), Type(10.0), true);  // Soft prior: reference temperature centered at 15°C
  
  // Seasonal temperature variation parameters (NEW)
  nll -= dnorm(T_mean, Type(12.0), Type(10.0), true);  // Soft prior: annual mean temperature centered at 12°C (temperate ocean)
  nll -= dnorm(T_amplitude, Type(6.0), Type(5.0), true);  // Soft prior: temperature amplitude centered at 6°C (temperate seasonality)
  nll -= dnorm(t_phase_temp, Type(210.0), Type(100.0), true);  // Soft prior: temperature peak centered at day 210 (late July, ~38 days after light peak)
  
  // Temperature amplitude should be positive (using log-normal-like penalty)
  Type T_amplitude_penalty = -log(T_amplitude + eps);  // Penalty to keep T_amplitude positive
  nll += Type(0.1) * T_amplitude_penalty;  // Add amplitude penalty with small weight
  
  // Variable assimilation efficiency parameters
  nll -= dnorm(epsilon_min, Type(0.15), Type(0.1), true);  // Soft prior: minimum assimilation efficiency centered at 0.15
  nll -= dnorm(epsilon_max, Type(0.45), Type(0.15), true);  // Soft prior: maximum assimilation efficiency centered at 0.45
  
  // Efficiencies should be between 0 and 1 (using logit-like penalty)
  Type epsilon_min_penalty = -log(epsilon_min + eps) - log(Type(1.0) - epsilon_min + eps);  // Penalty to keep epsilon_min in (0,1)
  Type epsilon_max_penalty = -log(epsilon_max + eps) - log(Type(1.0) - epsilon_max + eps);  // Penalty to keep epsilon_max in (0,1)
  Type gamma_P_penalty = -log(gamma_P + eps) - log(Type(1.0) - gamma_P + eps);  // Penalty to keep gamma_P in (0,1)
  Type gamma_Z_penalty = -log(gamma_Z + eps) - log(Type(1.0) - gamma_Z + eps);  // Penalty to keep gamma_Z in (0,1)
  Type I_0_amplitude_penalty = -log(I_0_amplitude + eps) - log(Type(1.0) - I_0_amplitude + eps);  // Penalty to keep I_0_amplitude in (0,1)
  nll += Type(0.1) * (epsilon_min_penalty + epsilon_max_penalty + gamma_P_penalty + gamma_Z_penalty + I_0_amplitude_penalty);  // Add efficiency penalties with small weight
  
  // Constraint: epsilon_max should be greater than epsilon_min
  Type epsilon_ordering_penalty = Type(0.0);
  if(epsilon_max <= epsilon_min) {
    epsilon_ordering_penalty = Type(1000.0) * pow(epsilon_min - epsilon_max + Type(0.01), Type(2.0));  // Large penalty if ordering is violated
  }
  nll += epsilon_ordering_penalty;  // Add ordering constraint penalty
  
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
  REPORT(epsilon_min);  // Report minimum assimilation efficiency
  REPORT(epsilon_max);  // Report maximum assimilation efficiency
  REPORT(m_Z);  // Report zooplankton linear mortality
  REPORT(m_Z2);  // Report zooplankton quadratic mortality
  REPORT(gamma_P);  // Report phytoplankton recycling efficiency
  REPORT(gamma_Z);  // Report zooplankton recycling efficiency
  REPORT(N_input);  // Report nutrient input rate
  REPORT(I_0_mean);  // Report annual mean surface light intensity
  REPORT(I_0_amplitude);  // Report seasonal light variation amplitude
  REPORT(t_phase);  // Report day of year when light peaks
  REPORT(K_I);  // Report light half-saturation constant
  REPORT(k_w);  // Report water attenuation coefficient
  REPORT(k_c);  // Report phytoplankton attenuation coefficient
  REPORT(z_mix);  // Report mixed layer depth
  REPORT(Q10_phyto);  // Report phytoplankton Q10 coefficient
  REPORT(Q10_zoo);  // Report zooplankton Q10 coefficient
  REPORT(Q10_remin);  // Report remineralization Q10 coefficient
  REPORT(T_ref);  // Report reference temperature
  REPORT(T_mean);  // Report annual mean temperature
  REPORT(T_amplitude);  // Report seasonal temperature amplitude
  REPORT(t_phase_temp);  // Report day of year when temperature peaks
  REPORT(sigma_N);  // Report nutrient observation error
  REPORT(sigma_P);  // Report phytoplankton observation error
  REPORT(sigma_Z);  // Report zooplankton observation error
  
  return nll;  // Return total negative log-likelihood
}
