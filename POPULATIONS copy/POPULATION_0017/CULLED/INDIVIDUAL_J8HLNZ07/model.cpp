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
  PARAMETER(r);  // Maximum phytoplankton growth rate (day^-1)
  PARAMETER(K_N);  // Half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(m_P);  // Phytoplankton linear mortality rate (day^-1)
  PARAMETER(m_P2);  // Phytoplankton density-dependent mortality rate coefficient (day^-1 (g C m^-3)^-1)
  
  // Parameters for zooplankton dynamics
  PARAMETER(g_max);  // Maximum zooplankton grazing rate (day^-1)
  PARAMETER(K_P);  // Half-saturation constant for grazing (g C m^-3)
  PARAMETER(e);  // Zooplankton assimilation efficiency (dimensionless, 0-1)
  PARAMETER(m_Z);  // Zooplankton mortality rate (day^-1)
  
  // Nutrient recycling parameters
  PARAMETER(gamma);  // Fraction of phytoplankton mortality recycled to nutrients (dimensionless, 0-1)
  PARAMETER(delta);  // Fraction of zooplankton mortality recycled to nutrients (dimensionless, 0-1)
  
  // Light limitation parameters
  PARAMETER(I_0);  // Surface light intensity (W m^-2)
  PARAMETER(K_I);  // Half-saturation constant for light-limited growth (W m^-2)
  PARAMETER(k_w);  // Background light attenuation coefficient (m^-1)
  PARAMETER(k_chl);  // Chlorophyll-specific light attenuation coefficient (m^2 (g Chl)^-1)
  PARAMETER(z_mix);  // Mixed layer depth (m)
  
  // Photoacclimation parameters
  PARAMETER(theta_max);  // Maximum chlorophyll-to-carbon ratio (g Chl (g C)^-1)
  PARAMETER(theta_min);  // Minimum chlorophyll-to-carbon ratio (g Chl (g C)^-1)
  PARAMETER(rho);  // Photoacclimation rate (day^-1)
  
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
  vector<Type> theta_pred(n);  // Predicted chlorophyll-to-carbon ratio
  
  // Set initial conditions from first observation
  N_pred(0) = N_dat(0);  // Initialize nutrient from first data point
  P_pred(0) = P_dat(0);  // Initialize phytoplankton from first data point
  Z_pred(0) = Z_dat(0);  // Initialize zooplankton from first data point
  theta_pred(0) = (theta_max + theta_min) / Type(2.0);  // Initialize chlorophyll ratio at midpoint
  
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
  
  // Linear mortality rate should be positive and reasonable (0.01 to 1.0 day^-1)
  if(m_P < Type(0.01)) nll += penalty_weight * pow(Type(0.01) - m_P, 2);  // Penalize if phytoplankton linear mortality too low
  if(m_P > Type(1.0)) nll += penalty_weight * pow(m_P - Type(1.0), 2);  // Penalize if phytoplankton linear mortality too high
  
  // Density-dependent mortality coefficient should be positive and reasonable (0.01 to 0.5 day^-1 (g C m^-3)^-1)
  if(m_P2 < Type(0.01)) nll += penalty_weight * pow(Type(0.01) - m_P2, 2);  // Penalize if phytoplankton quadratic mortality too low
  if(m_P2 > Type(0.5)) nll += penalty_weight * pow(m_P2 - Type(0.5), 2);  // Penalize if phytoplankton quadratic mortality too high
  
  // Zooplankton mortality rate should be positive and reasonable (0.01 to 1.0 day^-1)
  if(m_Z < Type(0.01)) nll += penalty_weight * pow(Type(0.01) - m_Z, 2);  // Penalize if zooplankton mortality too low
  if(m_Z > Type(1.0)) nll += penalty_weight * pow(m_Z - Type(1.0), 2);  // Penalize if zooplankton mortality too high
  
  // Grazing rate should be positive and reasonable (0.1 to 2.0 day^-1)
  if(g_max < Type(0.1)) nll += penalty_weight * pow(Type(0.1) - g_max, 2);  // Penalize if grazing rate too low
  if(g_max > Type(2.0)) nll += penalty_weight * pow(g_max - Type(2.0), 2);  // Penalize if grazing rate too high
  
  // Efficiency and recycling fractions should be between 0 and 1
  if(e < Type(0.0)) nll += penalty_weight * pow(e, 2);  // Penalize negative assimilation efficiency
  if(e > Type(1.0)) nll += penalty_weight * pow(e - Type(1.0), 2);  // Penalize assimilation efficiency > 1
  if(gamma < Type(0.0)) nll += penalty_weight * pow(gamma, 2);  // Penalize negative recycling fraction
  if(gamma > Type(1.0)) nll += penalty_weight * pow(gamma - Type(1.0), 2);  // Penalize recycling fraction > 1
  if(delta < Type(0.0)) nll += penalty_weight * pow(delta, 2);  // Penalize negative recycling fraction
  if(delta > Type(1.0)) nll += penalty_weight * pow(delta - Type(1.0), 2);  // Penalize recycling fraction > 1
  
  // Light parameters should be positive and reasonable
  if(I_0 < Type(50.0)) nll += penalty_weight * pow(Type(50.0) - I_0, 2);  // Penalize if surface light too low
  if(I_0 > Type(400.0)) nll += penalty_weight * pow(I_0 - Type(400.0), 2);  // Penalize if surface light too high
  if(K_I < Type(10.0)) nll += penalty_weight * pow(Type(10.0) - K_I, 2);  // Penalize if light half-saturation too low
  if(K_I > Type(100.0)) nll += penalty_weight * pow(K_I - Type(100.0), 2);  // Penalize if light half-saturation too high
  if(k_w < Type(0.02)) nll += penalty_weight * pow(Type(0.02) - k_w, 2);  // Penalize if background attenuation too low
  if(k_w > Type(0.2)) nll += penalty_weight * pow(k_w - Type(0.2), 2);  // Penalize if background attenuation too high
  if(k_chl < Type(0.02)) nll += penalty_weight * pow(Type(0.02) - k_chl, 2);  // Penalize if chlorophyll attenuation too low
  if(k_chl > Type(0.05)) nll += penalty_weight * pow(k_chl - Type(0.05), 2);  // Penalize if chlorophyll attenuation too high
  if(z_mix < Type(5.0)) nll += penalty_weight * pow(Type(5.0) - z_mix, 2);  // Penalize if mixed layer depth too shallow
  if(z_mix > Type(100.0)) nll += penalty_weight * pow(z_mix - Type(100.0), 2);  // Penalize if mixed layer depth too deep
  
  // Photoacclimation parameters should be positive and reasonable
  if(theta_max < Type(0.03)) nll += penalty_weight * pow(Type(0.03) - theta_max, 2);  // Penalize if max Chl:C too low
  if(theta_max > Type(0.08)) nll += penalty_weight * pow(theta_max - Type(0.08), 2);  // Penalize if max Chl:C too high
  if(theta_min < Type(0.005)) nll += penalty_weight * pow(Type(0.005) - theta_min, 2);  // Penalize if min Chl:C too low
  if(theta_min > Type(0.02)) nll += penalty_weight * pow(theta_min - Type(0.02), 2);  // Penalize if min Chl:C too high
  if(theta_min > theta_max) nll += penalty_weight * pow(theta_min - theta_max, 2);  // Penalize if min > max
  if(rho < Type(0.05)) nll += penalty_weight * pow(Type(0.05) - rho, 2);  // Penalize if acclimation rate too slow
  if(rho > Type(0.5)) nll += penalty_weight * pow(rho - Type(0.5), 2);  // Penalize if acclimation rate too fast
  
  // Forward simulation using Euler integration
  for(int i = 1; i < n; i++) {
    Type dt = Time(i) - Time(i-1);  // Time step size (days)
    
    // Get previous time step values (avoid data leakage)
    Type N_prev = N_pred(i-1);  // Nutrient concentration at previous time step
    Type P_prev = P_pred(i-1);  // Phytoplankton concentration at previous time step
    Type Z_prev = Z_pred(i-1);  // Zooplankton concentration at previous time step
    Type theta_prev = theta_pred(i-1);  // Chlorophyll-to-carbon ratio at previous time step
    
    // Ensure non-negative concentrations with smooth transition
    N_prev = N_prev + epsilon;  // Add small constant to prevent negative values
    P_prev = P_prev + epsilon;  // Add small constant to prevent negative values
    Z_prev = Z_prev + epsilon;  // Add small constant to prevent negative values
    
    // Constrain theta to biologically realistic range
    theta_prev = CppAD::CondExpGt(theta_prev, theta_min, theta_prev, theta_min);  // Ensure theta >= theta_min
    theta_prev = CppAD::CondExpLt(theta_prev, theta_max, theta_prev, theta_max);  // Ensure theta <= theta_max
    
    // Calculate light availability using Beer-Lambert law with dynamic self-shading
    // Light intensity at depth z: I(z) = I_0 * exp(-(k_w + k_chl * theta * P) * z)
    // For mixed layer, use average light over depth z_mix
    Type total_attenuation = k_w + k_chl * theta_prev * P_prev;  // Total light attenuation coefficient (m^-1)
    Type light_at_depth = I_0 * exp(-total_attenuation * z_mix);  // Light at bottom of mixed layer (W m^-2)
    Type avg_light = I_0 * (Type(1.0) - exp(-total_attenuation * z_mix)) / (total_attenuation * z_mix + epsilon);  // Average light in mixed layer (W m^-2)
    
    // Light limitation term (Michaelis-Menten/Monod kinetics)
    Type light_limitation = avg_light / (K_I + avg_light);  // Dimensionless light limitation factor (0-1)
    
    // Calculate optimal chlorophyll-to-carbon ratio based on light availability
    // Under low light: theta_opt approaches theta_max (maximize light harvesting)
    // Under high light: theta_opt approaches theta_min (reduce photoinhibition, minimize self-shading)
    Type theta_opt = theta_max * (K_I / (K_I + avg_light));  // Optimal Chl:C ratio for current light (g Chl (g C)^-1)
    
    // Ensure theta_opt stays within bounds
    theta_opt = CppAD::CondExpGt(theta_opt, theta_min, theta_opt, theta_min);  // Ensure theta_opt >= theta_min
    theta_opt = CppAD::CondExpLt(theta_opt, theta_max, theta_opt, theta_max);  // Ensure theta_opt <= theta_max
    
    // Calculate process rates
    // Equation 1: Phytoplankton nutrient-limited growth with light co-limitation
    Type nutrient_limitation = N_prev / (K_N + N_prev);  // Nutrient limitation factor (0-1)
    Type uptake = r * nutrient_limitation * light_limitation * P_prev;  // Nutrient uptake by phytoplankton with light and nutrient co-limitation (g C m^-3 day^-1)
    
    // Equation 2: Zooplankton grazing on phytoplankton (Holling Type II functional response)
    Type grazing = g_max * (P_prev / (K_P + P_prev)) * Z_prev;  // Phytoplankton consumption by zooplankton (g C m^-3 day^-1)
    
    // Equation 3: Phytoplankton mortality with density-dependent component
    // Mixed linear-quadratic form: m_P * P + m_P2 * P^2
    // Linear term represents baseline mortality (senescence, background viral lysis)
    // Quadratic term represents density-dependent processes (epidemic viral lysis, allelopathy, aggregation)
    Type P_mortality = m_P * P_prev + m_P2 * P_prev * P_prev;  // Phytoplankton death rate with density dependence (g C m^-3 day^-1)
    
    // Equation 4: Zooplankton mortality (density-dependent)
    Type Z_mortality = m_Z * Z_prev * Z_prev;  // Zooplankton death rate, quadratic to represent density dependence (g C m^-3 day^-1)
    
    // Equation 5: Nutrient recycling from phytoplankton mortality
    Type N_recycling_P = gamma * P_mortality;  // Nutrients returned from dead phytoplankton (g C m^-3 day^-1)
    
    // Equation 6: Nutrient recycling from zooplankton mortality and excretion
    Type N_recycling_Z = delta * Z_mortality + (Type(1.0) - e) * grazing;  // Nutrients from zooplankton waste and inefficient assimilation (g C m^-3 day^-1)
    
    // Equation 7: Rate of change for nutrients (dN/dt)
    Type dN_dt = -uptake + N_recycling_P + N_recycling_Z;  // Net change in nutrient concentration (g C m^-3 day^-1)
    
    // Equation 8: Rate of change for phytoplankton (dP/dt)
    Type dP_dt = uptake - grazing - P_mortality;  // Net change in phytoplankton concentration (g C m^-3 day^-1)
    
    // Equation 9: Rate of change for zooplankton (dZ/dt)
    Type dZ_dt = e * grazing - Z_mortality;  // Net change in zooplankton concentration (g C m^-3 day^-1)
    
    // Equation 10: Rate of change for chlorophyll-to-carbon ratio (dtheta/dt) - photoacclimation
    Type dtheta_dt = rho * (theta_opt - theta_prev);  // Adjustment toward optimal Chl:C ratio (g Chl (g C)^-1 day^-1)
    
    // Update predictions using Euler method
    N_pred(i) = N_prev + dN_dt * dt;  // Update nutrient concentration
    P_pred(i) = P_prev + dP_dt * dt;  // Update phytoplankton concentration
    Z_pred(i) = Z_prev + dZ_dt * dt;  // Update zooplankton concentration
    theta_pred(i) = theta_prev + dtheta_dt * dt;  // Update chlorophyll-to-carbon ratio
    
    // Ensure predictions remain non-negative
    N_pred(i) = CppAD::CondExpGt(N_pred(i), Type(0.0), N_pred(i), Type(epsilon));  // Set to epsilon if negative
    P_pred(i) = CppAD::CondExpGt(P_pred(i), Type(0.0), P_pred(i), Type(epsilon));  // Set to epsilon if negative
    Z_pred(i) = CppAD::CondExpGt(Z_pred(i), Type(0.0), Z_pred(i), Type(epsilon));  // Set to epsilon if negative
    
    // Ensure theta stays within bounds
    theta_pred(i) = CppAD::CondExpGt(theta_pred(i), theta_min, theta_pred(i), theta_min);  // Ensure theta >= theta_min
    theta_pred(i) = CppAD::CondExpLt(theta_pred(i), theta_max, theta_pred(i), theta_max);  // Ensure theta <= theta_max
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
  REPORT(theta_pred);  // Report predicted chlorophyll-to-carbon ratio
  REPORT(sigma_N);  // Report nutrient observation error
  REPORT(sigma_P);  // Report phytoplankton observation error
  REPORT(sigma_Z);  // Report zooplankton observation error
  REPORT(r);  // Report phytoplankton growth rate
  REPORT(K_N);  // Report nutrient half-saturation constant
  REPORT(m_P);  // Report phytoplankton linear mortality rate
  REPORT(m_P2);  // Report phytoplankton density-dependent mortality coefficient
  REPORT(g_max);  // Report maximum grazing rate
  REPORT(K_P);  // Report grazing half-saturation constant
  REPORT(e);  // Report assimilation efficiency
  REPORT(m_Z);  // Report zooplankton mortality rate
  REPORT(gamma);  // Report phytoplankton recycling fraction
  REPORT(delta);  // Report zooplankton recycling fraction
  REPORT(I_0);  // Report surface light intensity
  REPORT(K_I);  // Report light half-saturation constant
  REPORT(k_w);  // Report background light attenuation
  REPORT(k_chl);  // Report chlorophyll-specific light attenuation
  REPORT(z_mix);  // Report mixed layer depth
  REPORT(theta_max);  // Report maximum chlorophyll-to-carbon ratio
  REPORT(theta_min);  // Report minimum chlorophyll-to-carbon ratio
  REPORT(rho);  // Report photoacclimation rate
  
  return nll;  // Return total negative log-likelihood
}
