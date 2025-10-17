#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs
  DATA_VECTOR(Time);  // Time in days
  DATA_VECTOR(N_dat);  // Nutrient concentration observations (g C m^-3)
  DATA_VECTOR(P_dat);  // Phytoplankton concentration observations (g C m^-3)
  DATA_VECTOR(Z_dat);  // Zooplankton concentration observations (g C m^-3)
  
  // Parameters for phytoplankton dynamics
  PARAMETER(r);  // Maximum phytoplankton growth rate at reference temperature (day^-1)
  PARAMETER(K_N);  // Baseline half-saturation constant for nutrient uptake under optimal light (g C m^-3)
  PARAMETER(alpha_LN);  // Light-nutrient interaction coefficient for photoacclimation (dimensionless)
  PARAMETER(m_P);  // Phytoplankton mortality rate at reference temperature (day^-1)
  PARAMETER(s_P);  // Phytoplankton aggregation and sinking coefficient (m^3 (g C)^-1 day^-1)
  PARAMETER(xi);  // Fraction of sinking phytoplankton recycled as nutrients (dimensionless, 0-1)
  
  // Parameters for zooplankton dynamics
  PARAMETER(g_max);  // Maximum zooplankton grazing rate at reference temperature (day^-1)
  PARAMETER(K_P);  // Half-saturation constant for grazing (g C m^-3)
  PARAMETER(e_max);  // Maximum zooplankton assimilation efficiency at high food concentrations (dimensionless, 0-1)
  PARAMETER(e_min);  // Minimum zooplankton assimilation efficiency at low food concentrations (dimensionless, 0-1)
  PARAMETER(K_e);  // Half-saturation constant for food-dependent assimilation efficiency (g C m^-3)
  PARAMETER(m_Z);  // Zooplankton mortality rate coefficient at reference temperature (day^-1 (g C m^-3)^-1)
  
  // Nutrient recycling parameters
  PARAMETER(gamma);  // Fraction of phytoplankton mortality recycled to nutrients (dimensionless, 0-1)
  PARAMETER(delta);  // Fraction of zooplankton mortality recycled to nutrients (dimensionless, 0-1)
  
  // Light limitation parameters
  PARAMETER(I_0);  // Surface light intensity (W m^-2)
  PARAMETER(K_I);  // Half-saturation constant for light-limited growth (W m^-2)
  PARAMETER(k_w);  // Background light attenuation coefficient (m^-1)
  PARAMETER(k_p);  // Specific light attenuation coefficient due to phytoplankton (m^2 (g C)^-1)
  PARAMETER(z_mix);  // Mixed layer depth (m)
  
  // Temperature dependency parameters (Q10 formulation)
  PARAMETER(Q10_P);  // Q10 coefficient for phytoplankton growth rate (dimensionless)
  PARAMETER(Q10_Z);  // Q10 coefficient for zooplankton grazing rate (dimensionless)
  PARAMETER(Q10_M);  // Q10 coefficient for mortality and decomposition rates (dimensionless)
  PARAMETER(Temperature);  // Water temperature (degrees C) - constant or mean value
  PARAMETER(T_ref);  // Reference temperature for Q10 calculations (degrees C)
  
  // Observation error parameters
  PARAMETER(log_sigma_N);  // Log-scale standard deviation for nutrient observations
  PARAMETER(log_sigma_P);  // Log-scale standard deviation for phytoplankton observations
  PARAMETER(log_sigma_Z);  // Log-scale standard deviation for zooplankton observations
  
  // Transform log-scale parameters to natural scale
  Type sigma_N = exp(log_sigma_N);  // Standard deviation for nutrient observation error
  Type sigma_P = exp(log_sigma_P);  // Standard deviation for phytoplankton observation error
  Type sigma_Z = exp(log_sigma_Z);  // Standard deviation for zooplankton observation error
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(1e-4);  // Minimum observation error to ensure numerical stability
  sigma_N = sigma_N + min_sigma;  // Add minimum to nutrient error
  sigma_P = sigma_P + min_sigma;  // Add minimum to phytoplankton error
  sigma_Z = sigma_Z + min_sigma;  // Add minimum to zooplankton error
  
  // Initialize prediction vectors
  int n = Time.size();  // Number of time points in the dataset
  vector<Type> N_pred(n);  // Predicted nutrient concentrations
  vector<Type> P_pred(n);  // Predicted phytoplankton concentrations
  vector<Type> Z_pred(n);  // Predicted zooplankton concentrations
  
  // Set initial conditions from first observation
  N_pred(0) = N_dat(0);  // Initialize nutrient from first data point
  P_pred(0) = P_dat(0);  // Initialize phytoplankton from first data point
  Z_pred(0) = Z_dat(0);  // Initialize zooplankton from first data point
  
  // Small constant to prevent division by zero
  Type epsilon = Type(1e-8);  // Small value added to denominators for numerical stability
  
  // Soft constraints on parameters using smooth penalties
  Type nll = Type(0.0);  // Initialize negative log-likelihood
  
  // Soft bounds for biological realism (using quadratic penalties outside reasonable ranges)
  Type penalty_weight = Type(10.0);  // Weight for soft constraint penalties
  
  // Growth rate should be positive and reasonable (0.1 to 3.0 day^-1)
  if(r < Type(0.1)) nll += penalty_weight * pow(Type(0.1) - r, 2);  // Penalize if growth rate too low
  if(r > Type(3.0)) nll += penalty_weight * pow(r - Type(3.0), 2);  // Penalize if growth rate too high
  
  // Half-saturation constants should be positive and reasonable (0.001 to 1.0 g C m^-3)
  if(K_N < Type(0.001)) nll += penalty_weight * pow(Type(0.001) - K_N, 2);  // Penalize if K_N too low
  if(K_N > Type(1.0)) nll += penalty_weight * pow(K_N - Type(1.0), 2);  // Penalize if K_N too high
  if(K_P < Type(0.001)) nll += penalty_weight * pow(Type(0.001) - K_P, 2);  // Penalize if K_P too low
  if(K_P > Type(1.0)) nll += penalty_weight * pow(K_P - Type(1.0), 2);  // Penalize if K_P too high
  
  // Light-nutrient interaction coefficient should be non-negative and reasonable (0.0 to 2.0)
  if(alpha_LN < Type(0.0)) nll += penalty_weight * pow(alpha_LN, 2);  // Penalize if alpha_LN negative
  if(alpha_LN > Type(2.0)) nll += penalty_weight * pow(alpha_LN - Type(2.0), 2);  // Penalize if alpha_LN too high
  
  // Mortality rates should be positive and reasonable (0.01 to 1.0 day^-1)
  if(m_P < Type(0.01)) nll += penalty_weight * pow(Type(0.01) - m_P, 2);  // Penalize if phytoplankton mortality too low
  if(m_P > Type(1.0)) nll += penalty_weight * pow(m_P - Type(1.0), 2);  // Penalize if phytoplankton mortality too high
  if(m_Z < Type(0.01)) nll += penalty_weight * pow(Type(0.01) - m_Z, 2);  // Penalize if zooplankton mortality coefficient too low
  if(m_Z > Type(5.0)) nll += penalty_weight * pow(m_Z - Type(5.0), 2);  // Penalize if zooplankton mortality coefficient too high
  
  // Aggregation/sinking coefficient should be positive and reasonable (0.01 to 0.5 m^3 (g C)^-1 day^-1)
  if(s_P < Type(0.01)) nll += penalty_weight * pow(Type(0.01) - s_P, 2);  // Penalize if aggregation rate too low
  if(s_P > Type(0.5)) nll += penalty_weight * pow(s_P - Type(0.5), 2);  // Penalize if aggregation rate too high
  
  // Grazing rate should be positive and reasonable (0.1 to 2.0 day^-1)
  if(g_max < Type(0.1)) nll += penalty_weight * pow(Type(0.1) - g_max, 2);  // Penalize if grazing rate too low
  if(g_max > Type(2.0)) nll += penalty_weight * pow(g_max - Type(2.0), 2);  // Penalize if grazing rate too high
  
  // Assimilation efficiency parameters should be between 0 and 1, with e_min < e_max
  if(e_max < Type(0.0)) nll += penalty_weight * pow(e_max, 2);  // Penalize negative maximum assimilation efficiency
  if(e_max > Type(1.0)) nll += penalty_weight * pow(e_max - Type(1.0), 2);  // Penalize maximum assimilation efficiency > 1
  if(e_min < Type(0.0)) nll += penalty_weight * pow(e_min, 2);  // Penalize negative minimum assimilation efficiency
  if(e_min > Type(1.0)) nll += penalty_weight * pow(e_min - Type(1.0), 2);  // Penalize minimum assimilation efficiency > 1
  if(e_min > e_max) nll += penalty_weight * pow(e_min - e_max, 2);  // Penalize if minimum exceeds maximum
  
  // Half-saturation constant for assimilation efficiency should be positive and reasonable
  if(K_e < Type(0.001)) nll += penalty_weight * pow(Type(0.001) - K_e, 2);  // Penalize if K_e too low
  if(K_e > Type(1.0)) nll += penalty_weight * pow(K_e - Type(1.0), 2);  // Penalize if K_e too high
  
  // Recycling fractions should be between 0 and 1
  if(gamma < Type(0.0)) nll += penalty_weight * pow(gamma, 2);  // Penalize negative recycling fraction
  if(gamma > Type(1.0)) nll += penalty_weight * pow(gamma - Type(1.0), 2);  // Penalize recycling fraction > 1
  if(delta < Type(0.0)) nll += penalty_weight * pow(delta, 2);  // Penalize negative recycling fraction
  if(delta > Type(1.0)) nll += penalty_weight * pow(delta - Type(1.0), 2);  // Penalize recycling fraction > 1
  if(xi < Type(0.0)) nll += penalty_weight * pow(xi, 2);  // Penalize negative sinking recycling fraction
  if(xi > Type(1.0)) nll += penalty_weight * pow(xi - Type(1.0), 2);  // Penalize sinking recycling fraction > 1
  
  // Light parameters should be positive and reasonable
  if(I_0 < Type(50.0)) nll += penalty_weight * pow(Type(50.0) - I_0, 2);  // Penalize if surface light too low
  if(I_0 > Type(400.0)) nll += penalty_weight * pow(I_0 - Type(400.0), 2);  // Penalize if surface light too high
  if(K_I < Type(10.0)) nll += penalty_weight * pow(Type(10.0) - K_I, 2);  // Penalize if light half-saturation too low
  if(K_I > Type(100.0)) nll += penalty_weight * pow(K_I - Type(100.0), 2);  // Penalize if light half-saturation too high
  if(k_w < Type(0.02)) nll += penalty_weight * pow(Type(0.02) - k_w, 2);  // Penalize if background attenuation too low
  if(k_w > Type(0.2)) nll += penalty_weight * pow(k_w - Type(0.2), 2);  // Penalize if background attenuation too high
  if(k_p < Type(0.01)) nll += penalty_weight * pow(Type(0.01) - k_p, 2);  // Penalize if phytoplankton attenuation too low
  if(k_p > Type(0.1)) nll += penalty_weight * pow(k_p - Type(0.1), 2);  // Penalize if phytoplankton attenuation too high
  if(z_mix < Type(5.0)) nll += penalty_weight * pow(Type(5.0) - z_mix, 2);  // Penalize if mixed layer depth too shallow
  if(z_mix > Type(100.0)) nll += penalty_weight * pow(z_mix - Type(100.0), 2);  // Penalize if mixed layer depth too deep
  
  // Q10 parameters should be within biologically realistic ranges
  if(Q10_P < Type(1.5)) nll += penalty_weight * pow(Type(1.5) - Q10_P, 2);  // Penalize if phytoplankton Q10 too low
  if(Q10_P > Type(2.5)) nll += penalty_weight * pow(Q10_P - Type(2.5), 2);  // Penalize if phytoplankton Q10 too high
  if(Q10_Z < Type(2.0)) nll += penalty_weight * pow(Type(2.0) - Q10_Z, 2);  // Penalize if zooplankton Q10 too low
  if(Q10_Z > Type(3.0)) nll += penalty_weight * pow(Q10_Z - Type(3.0), 2);  // Penalize if zooplankton Q10 too high
  if(Q10_M < Type(1.5)) nll += penalty_weight * pow(Type(1.5) - Q10_M, 2);  // Penalize if mortality Q10 too low
  if(Q10_M > Type(3.0)) nll += penalty_weight * pow(Q10_M - Type(3.0), 2);  // Penalize if mortality Q10 too high
  
  // Temperature should be reasonable for ocean systems
  if(Temperature < Type(5.0)) nll += penalty_weight * pow(Type(5.0) - Temperature, 2);  // Penalize if temperature too low
  if(Temperature > Type(25.0)) nll += penalty_weight * pow(Temperature - Type(25.0), 2);  // Penalize if temperature too high
  
  // Reference temperature should be reasonable for ocean systems
  if(T_ref < Type(10.0)) nll += penalty_weight * pow(Type(10.0) - T_ref, 2);  // Penalize if reference temperature too low
  if(T_ref > Type(20.0)) nll += penalty_weight * pow(T_ref - Type(20.0), 2);  // Penalize if reference temperature too high
  
  // Calculate Q10 temperature scaling factors (constant for all time steps)
  // Q10 formula: Rate(T) = Rate(T_ref) × Q10^((T - T_ref)/10)
  Type temp_diff = (Temperature - T_ref) / Type(10.0);  // Temperature difference in units of 10°C
  Type f_temp_P = pow(Q10_P, temp_diff);  // Temperature scaling factor for phytoplankton growth
  Type f_temp_Z = pow(Q10_Z, temp_diff);  // Temperature scaling factor for zooplankton grazing
  Type f_temp_M = pow(Q10_M, temp_diff);  // Temperature scaling factor for mortality processes
  
  // Apply temperature-adjusted rates (constant for all time steps)
  Type r_temp = r * f_temp_P;  // Temperature-adjusted phytoplankton growth rate (day^-1)
  Type g_max_temp = g_max * f_temp_Z;  // Temperature-adjusted zooplankton grazing rate (day^-1)
  Type m_P_temp = m_P * f_temp_M;  // Temperature-adjusted phytoplankton mortality rate (day^-1)
  Type m_Z_temp = m_Z * f_temp_M;  // Temperature-adjusted zooplankton mortality coefficient (day^-1 (g C m^-3)^-1)
  Type s_P_temp = s_P * f_temp_M;  // Temperature-adjusted aggregation/sinking coefficient (m^3 (g C)^-1 day^-1)
  
  // Forward simulation using Euler integration
  for(int i = 1; i < n; i++) {
    Type dt = Time(i) - Time(i-1);  // Time step size (days)
    
    // Get previous time step values (avoid data leakage)
    Type N_prev = N_pred(i-1);  // Nutrient concentration at previous time step
    Type P_prev = P_pred(i-1);  // Phytoplankton concentration at previous time step
    Type Z_prev = Z_pred(i-1);  // Zooplankton concentration at previous time step
    
    // Ensure non-negative concentrations with smooth transition
    N_prev = N_prev + epsilon;  // Add small constant to prevent negative values
    P_prev = P_prev + epsilon;  // Add small constant to prevent negative values
    Z_prev = Z_prev + epsilon;  // Add small constant to prevent negative values
    
    // Calculate food-dependent assimilation efficiency
    // e_effective = e_min + (e_max - e_min) * P / (K_e + P)
    // At low P: e_effective approaches e_min (inefficient feeding on poor quality/scarce food)
    // At high P: e_effective approaches e_max (efficient feeding with selective grazing)
    Type e_effective = e_min + (e_max - e_min) * P_prev / (K_e + P_prev);  // Food-dependent assimilation efficiency (dimensionless, 0-1)
    
    // Calculate light availability using Beer-Lambert law with self-shading
    // Light intensity at depth z: I(z) = I_0 * exp(-(k_w + k_p * P) * z)
    // For mixed layer, use average light over depth z_mix
    Type total_attenuation = k_w + k_p * P_prev;  // Total light attenuation coefficient (m^-1)
    Type light_at_depth = I_0 * exp(-total_attenuation * z_mix);  // Light at bottom of mixed layer (W m^-2)
    Type avg_light = I_0 * (Type(1.0) - exp(-total_attenuation * z_mix)) / (total_attenuation * z_mix + epsilon);  // Average light in mixed layer (W m^-2)
    
    // Light limitation term (Michaelis-Menten/Monod kinetics)
    Type light_limitation = avg_light / (K_I + avg_light);  // Dimensionless light limitation factor (0-1)
    
    // Calculate light-dependent nutrient half-saturation constant (photoacclimation)
    // K_N_effective = K_N * (1 + alpha_LN * K_I / avg_light)
    // Under high light (avg_light >> K_I): K_N_effective ≈ K_N (baseline nutrient requirement)
    // Under low light (avg_light << K_I): K_N_effective increases (higher nutrient demand for photoacclimation)
    Type K_N_effective = K_N * (Type(1.0) + alpha_LN * K_I / (avg_light + epsilon));  // Light-dependent nutrient half-saturation (g C m^-3)
    
    // Calculate process rates
    // Equation 1: Phytoplankton nutrient-limited growth with light, temperature, and photoacclimation
    Type nutrient_limitation = N_prev / (K_N_effective + N_prev);  // Nutrient limitation factor with light-dependent K_N (0-1)
    Type uptake = r_temp * nutrient_limitation * light_limitation * P_prev;  // Nutrient uptake by phytoplankton with light, nutrient, temperature, and photoacclimation effects (g C m^-3 day^-1)
    
    // Equation 2: Zooplankton grazing on phytoplankton (Holling Type II functional response with temperature)
    Type grazing = g_max_temp * (P_prev / (K_P + P_prev)) * Z_prev;  // Phytoplankton consumption by zooplankton (g C m^-3 day^-1)
    
    // Equation 3: Phytoplankton natural mortality (temperature-dependent)
    Type P_mortality = m_P_temp * P_prev;  // Phytoplankton death rate (g C m^-3 day^-1)
    
    // Equation 4: Phytoplankton aggregation and sinking (density-dependent, temperature-dependent)
    Type P_aggregation = s_P_temp * P_prev * P_prev;  // Phytoplankton loss via aggregation and sinking (g C m^-3 day^-1)
    
    // Equation 5: Zooplankton mortality (density-dependent, temperature-dependent)
    Type Z_mortality = m_Z_temp * Z_prev * Z_prev;  // Zooplankton death rate, quadratic to represent density dependence (g C m^-3 day^-1)
    
    // Equation 6: Nutrient recycling from phytoplankton mortality
    Type N_recycling_P_mortality = gamma * P_mortality;  // Nutrients returned from dead phytoplankton (g C m^-3 day^-1)
    
    // Equation 7: Nutrient recycling from phytoplankton aggregation/sinking
    Type N_recycling_P_sinking = xi * P_aggregation;  // Nutrients recycled from sinking aggregates before export (g C m^-3 day^-1)
    
    // Equation 8: Nutrient recycling from zooplankton mortality and excretion
    // Now uses food-dependent assimilation efficiency: more efficient feeding = less sloppy feeding/excretion
    Type N_recycling_Z = delta * Z_mortality + (Type(1.0) - e_effective) * grazing;  // Nutrients from zooplankton waste and inefficient assimilation (g C m^-3 day^-1)
    
    // Equation 9: Rate of change for nutrients (dN/dt)
    Type dN_dt = -uptake + N_recycling_P_mortality + N_recycling_P_sinking + N_recycling_Z;  // Net change in nutrient concentration (g C m^-3 day^-1)
    
    // Equation 10: Rate of change for phytoplankton (dP/dt)
    Type dP_dt = uptake - grazing - P_mortality - P_aggregation;  // Net change in phytoplankton concentration (g C m^-3 day^-1)
    
    // Equation 11: Rate of change for zooplankton (dZ/dt)
    // Now uses food-dependent assimilation efficiency: higher efficiency at high food = more growth
    Type dZ_dt = e_effective * grazing - Z_mortality;  // Net change in zooplankton concentration (g C m^-3 day^-1)
    
    // Update predictions using Euler method
    N_pred(i) = N_prev + dN_dt * dt;  // Update nutrient concentration
    P_pred(i) = P_prev + dP_dt * dt;  // Update phytoplankton concentration
    Z_pred(i) = Z_prev + dZ_dt * dt;  // Update zooplankton concentration
    
    // Ensure predictions remain non-negative
    N_pred(i) = CppAD::CondExpGt(N_pred(i), Type(0.0), N_pred(i), Type(epsilon));  // Set to epsilon if negative
    P_pred(i) = CppAD::CondExpGt(P_pred(i), Type(0.0), P_pred(i), Type(epsilon));  // Set to epsilon if negative
    Z_pred(i) = CppAD::CondExpGt(Z_pred(i), Type(0.0), Z_pred(i), Type(epsilon));  // Set to epsilon if negative
  }
  
  // Calculate likelihood for all observations
  for(int i = 0; i < n; i++) {
    // Normal likelihood for nutrient observations
    nll -= dnorm(N_dat(i), N_pred(i), sigma_N, true);  // Add negative log-likelihood for nutrient data
    
    // Normal likelihood for phytoplankton observations
    nll -= dnorm(P_dat(i), P_pred(i), sigma_P, true);  // Add negative log-likelihood for phytoplankton data
    
    // Normal likelihood for zooplankton observations
    nll -= dnorm(Z_dat(i), Z_pred(i), sigma_Z, true);  // Add negative log-likelihood for zooplankton data
  }
  
  // Report predictions and parameters
  REPORT(N_pred);  // Report predicted nutrient concentrations
  REPORT(P_pred);  // Report predicted phytoplankton concentrations
  REPORT(Z_pred);  // Report predicted zooplankton concentrations
  REPORT(sigma_N);  // Report nutrient observation error
  REPORT(sigma_P);  // Report phytoplankton observation error
  REPORT(sigma_Z);  // Report zooplankton observation error
  REPORT(r);  // Report phytoplankton growth rate at reference temperature
  REPORT(K_N);  // Report baseline nutrient half-saturation constant
  REPORT(alpha_LN);  // Report light-nutrient interaction coefficient
  REPORT(m_P);  // Report phytoplankton mortality rate at reference temperature
  REPORT(s_P);  // Report phytoplankton aggregation/sinking coefficient
  REPORT(xi);  // Report sinking recycling fraction
  REPORT(g_max);  // Report maximum grazing rate at reference temperature
  REPORT(K_P);  // Report grazing half-saturation constant
  REPORT(e_max);  // Report maximum assimilation efficiency
  REPORT(e_min);  // Report minimum assimilation efficiency
  REPORT(K_e);  // Report half-saturation constant for assimilation efficiency
  REPORT(m_Z);  // Report zooplankton mortality coefficient at reference temperature
  REPORT(gamma);  // Report phytoplankton mortality recycling fraction
  REPORT(delta);  // Report zooplankton mortality recycling fraction
  REPORT(I_0);  // Report surface light intensity
  REPORT(K_I);  // Report light half-saturation constant
  REPORT(k_w);  // Report background light attenuation coefficient
  REPORT(k_p);  // Report phytoplankton-specific light attenuation coefficient
  REPORT(z_mix);  // Report mixed layer depth
  REPORT(Q10_P);  // Report phytoplankton Q10 coefficient
  REPORT(Q10_Z);  // Report zooplankton Q10 coefficient
  REPORT(Q10_M);  // Report mortality Q10 coefficient
  REPORT(Temperature);  // Report water temperature
  REPORT(T_ref);  // Report reference temperature
  
  return nll;  // Return total negative log-likelihood
}
