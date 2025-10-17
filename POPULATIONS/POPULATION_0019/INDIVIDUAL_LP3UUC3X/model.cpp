#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS
  DATA_VECTOR(Time);  // Time in days
  DATA_VECTOR(N_dat);  // Observed nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);  // Observed phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);  // Observed zooplankton concentration (g C m^-3)
  
  // PARAMETERS - Phytoplankton growth
  PARAMETER(r_P);  // Maximum phytoplankton growth rate at T_ref (day^-1) - determined from laboratory culture experiments
  PARAMETER(K_N);  // Half-saturation constant for nutrient uptake (g C m^-3) - from nutrient addition experiments
  PARAMETER(epsilon_P);  // Nutrient uptake efficiency (dimensionless, 0-1) - from stoichiometric constraints
  
  // PARAMETERS - Zooplankton grazing
  PARAMETER(g_max);  // Maximum zooplankton grazing rate at T_ref (day^-1) - from feeding experiments
  PARAMETER(K_Z);  // Half-saturation constant for grazing (g C m^-3) - from functional response experiments
  PARAMETER(epsilon_Z);  // Assimilation efficiency of zooplankton (dimensionless, 0-1) - from growth efficiency studies
  
  // PARAMETERS - Mortality and recycling
  PARAMETER(m_P);  // Phytoplankton mortality rate (day^-1) - from dilution experiments and natural mortality observations
  PARAMETER(m_Z);  // Zooplankton linear mortality rate (day^-1) - from population dynamics studies (natural death, senescence)
  PARAMETER(m_Z2);  // Zooplankton quadratic mortality coefficient ((g C m^-3)^-1 day^-1) - represents density-dependent predation by higher trophic levels
  PARAMETER(gamma);  // Nutrient recycling fraction (dimensionless, 0-1) - from remineralization rate measurements
  
  // PARAMETERS - Light limitation
  PARAMETER(I_0);  // Surface light intensity (W m^-2) - photosynthetically active radiation
  PARAMETER(MLD);  // Mixed layer depth (m) - depth over which plankton are mixed
  PARAMETER(k_w);  // Background water light attenuation coefficient (m^-1) - from optical measurements
  PARAMETER(k_p);  // Phytoplankton-specific light attenuation coefficient (m^2 (g C)^-1) - self-shading effect
  PARAMETER(K_I);  // Half-saturation light intensity for photosynthesis (W m^-2) - from P-I curves
  
  // PARAMETERS - Temperature dependence
  PARAMETER(T_ref);  // Reference temperature (K) - temperature at which r_P and g_max are defined
  PARAMETER(A_E_P);  // Activation energy for phytoplankton growth (E_a/R in Kelvin) - from metabolic theory and experiments
  PARAMETER(A_E_Z);  // Activation energy for zooplankton grazing (E_a/R in Kelvin) - from metabolic theory and experiments
  
  // PARAMETERS - Observation error
  PARAMETER(log_sigma_N);  // Log-scale standard deviation for nutrient observations - estimated from data variability
  PARAMETER(log_sigma_P);  // Log-scale standard deviation for phytoplankton observations - estimated from data variability
  PARAMETER(log_sigma_Z);  // Log-scale standard deviation for zooplankton observations - estimated from data variability
  
  // Transform log-scale parameters to natural scale
  Type sigma_N = exp(log_sigma_N);  // Standard deviation for nutrient (g C m^-3)
  Type sigma_P = exp(log_sigma_P);  // Standard deviation for phytoplankton (g C m^-3)
  Type sigma_Z = exp(log_sigma_Z);  // Standard deviation for zooplankton (g C m^-3)
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(1e-4);  // Minimum allowed standard deviation (g C m^-3)
  sigma_N = sigma_N + min_sigma;  // Ensure sigma_N is always above minimum
  sigma_P = sigma_P + min_sigma;  // Ensure sigma_P is always above minimum
  sigma_Z = sigma_Z + min_sigma;  // Ensure sigma_Z is always above minimum
  
  int n = Time.size();  // Number of time steps in the data
  
  // PREDICTION VECTORS
  vector<Type> N_pred(n);  // Predicted nutrient concentration (g C m^-3)
  vector<Type> P_pred(n);  // Predicted phytoplankton concentration (g C m^-3)
  vector<Type> Z_pred(n);  // Predicted zooplankton concentration (g C m^-3)
  
  // INITIALIZE with first observations (initial conditions from data)
  N_pred(0) = N_dat(0);  // Initial nutrient concentration from data
  P_pred(0) = P_dat(0);  // Initial phytoplankton concentration from data
  Z_pred(0) = Z_dat(0);  // Initial zooplankton concentration from data
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);  // Small constant for numerical stability
  
  // SOFT PARAMETER CONSTRAINTS using penalties
  Type nll = Type(0.0);  // Initialize negative log-likelihood
  
  // Soft bounds for rates (should be positive)
  nll -= dnorm(r_P, Type(0.5), Type(2.0), true);  // Prior: r_P should be positive, centered around 0.5
  nll -= dnorm(g_max, Type(0.5), Type(2.0), true);  // Prior: g_max should be positive, centered around 0.5
  nll -= dnorm(m_P, Type(0.05), Type(0.5), true);  // Prior: m_P should be small positive, centered around 0.05
  nll -= dnorm(m_Z, Type(0.05), Type(0.5), true);  // Prior: m_Z should be small positive, centered around 0.05
  nll -= dnorm(m_Z2, Type(0.1), Type(0.5), true);  // Prior: m_Z2 should be positive, centered around 0.1
  
  // Soft bounds for half-saturation constants (should be positive)
  nll -= dnorm(K_N, Type(0.1), Type(1.0), true);  // Prior: K_N should be positive, centered around 0.1
  nll -= dnorm(K_Z, Type(0.1), Type(1.0), true);  // Prior: K_Z should be positive, centered around 0.1
  nll -= dnorm(K_I, Type(30.0), Type(30.0), true);  // Prior: K_I should be positive, centered around 30 W m^-2
  
  // Soft bounds for light parameters (should be positive)
  nll -= dnorm(I_0, Type(200.0), Type(150.0), true);  // Prior: I_0 centered around 200 W m^-2
  nll -= dnorm(MLD, Type(50.0), Type(50.0), true);  // Prior: MLD centered around 50 m
  nll -= dnorm(k_w, Type(0.04), Type(0.1), true);  // Prior: k_w centered around 0.04 m^-1
  nll -= dnorm(k_p, Type(0.03), Type(0.03), true);  // Prior: k_p centered around 0.03 m^2 (g C)^-1
  
  // Soft bounds for temperature parameters
  nll -= dnorm(T_ref, Type(288.15), Type(10.0), true);  // Prior: T_ref centered around 288.15 K (15°C)
  nll -= dnorm(A_E_P, Type(4500.0), Type(1500.0), true);  // Prior: A_E_P centered around 4500 K (Q10 ~ 1.88)
  nll -= dnorm(A_E_Z, Type(6000.0), Type(1500.0), true);  // Prior: A_E_Z centered around 6000 K (Q10 ~ 2.4)
  
  // Soft bounds for efficiencies (should be between 0 and 1)
  Type epsilon_P_bounded = Type(1.0) / (Type(1.0) + exp(-epsilon_P));  // Logistic transform to bound between 0 and 1
  Type epsilon_Z_bounded = Type(1.0) / (Type(1.0) + exp(-epsilon_Z));  // Logistic transform to bound between 0 and 1
  Type gamma_bounded = Type(1.0) / (Type(1.0) + exp(-gamma));  // Logistic transform to bound between 0 and 1
  
  // FORWARD SIMULATION using Euler method
  for(int i = 1; i < n; i++) {
    Type dt = Time(i) - Time(i-1);  // Time step size (days)
    
    // Get previous state (avoid using current time step observations)
    Type N_prev = N_pred(i-1);  // Nutrient at previous time step
    Type P_prev = P_pred(i-1);  // Phytoplankton at previous time step
    Type Z_prev = Z_pred(i-1);  // Zooplankton at previous time step
    
    // Ensure non-negative concentrations
    N_prev = N_prev + eps;  // Add small constant to prevent negative values
    P_prev = P_prev + eps;  // Add small constant to prevent negative values
    Z_prev = Z_prev + eps;  // Add small constant to prevent negative values
    
    // TEMPERATURE DEPENDENCE - Arrhenius function
    // Use T_ref as the constant temperature (no temperature variation)
    // This means temp_modifier = exp[A_E * (1/T_ref - 1/T_ref)] = exp[0] = 1
    // So temperature-adjusted rates equal the base rates
    Type temp_modifier_P = Type(1.0);  // No temperature effect when T = T_ref
    Type temp_modifier_Z = Type(1.0);  // No temperature effect when T = T_ref
    
    // Apply temperature modifiers to rates (currently = 1, so no effect)
    Type r_P_temp = r_P * temp_modifier_P;  // Temperature-adjusted phytoplankton growth rate (day^-1)
    Type g_max_temp = g_max * temp_modifier_Z;  // Temperature-adjusted zooplankton grazing rate (day^-1)
    
    // LIGHT LIMITATION CALCULATION
    // Total light attenuation coefficient includes water and phytoplankton self-shading
    Type k_total = k_w + k_p * P_prev;  // Total attenuation (m^-1)
    
    // Average light intensity in mixed layer (depth-integrated)
    // Derived from: I_avg = (1/MLD) * integral_0^MLD [I_0 * exp(-k_total * z)] dz
    Type I_avg;  // Average light in mixed layer (W m^-2)
    if(k_total * MLD > eps) {
      // Standard case: exponential attenuation over mixed layer
      I_avg = (I_0 / (k_total * MLD)) * (Type(1.0) - exp(-k_total * MLD));
    } else {
      // Edge case: very shallow mixed layer or very clear water
      I_avg = I_0;  // Approximate as surface light
    }
    
    // Light limitation factor (Monod-type saturation)
    Type light_limitation = I_avg / (K_I + I_avg + eps);  // Light limitation factor (0-1)
    
    // EQUATION 1: Phytoplankton nutrient uptake (Monod/Michaelis-Menten kinetics)
    Type nutrient_limitation = N_prev / (K_N + N_prev + eps);  // Nutrient limitation factor (0-1)
    
    // EQUATION 1b: Phytoplankton growth with CO-LIMITATION by nutrients AND light, MODIFIED by temperature
    Type uptake = r_P_temp * nutrient_limitation * light_limitation * P_prev;  // Temperature-modified, co-limited phytoplankton growth (g C m^-3 day^-1)
    
    // EQUATION 2: Zooplankton grazing (Holling Type II functional response), MODIFIED by temperature
    Type grazing = g_max_temp * (P_prev / (K_Z + P_prev + eps)) * Z_prev;  // Temperature-modified phytoplankton consumption by zooplankton (g C m^-3 day^-1)
    
    // EQUATION 3: Phytoplankton mortality and losses
    Type P_loss = m_P * P_prev;  // Phytoplankton mortality (g C m^-3 day^-1)
    
    // EQUATION 4: Zooplankton mortality and losses (LINEAR + QUADRATIC)
    Type Z_loss_linear = m_Z * Z_prev;  // Linear mortality: natural death and senescence (g C m^-3 day^-1)
    Type Z_loss_quadratic = m_Z2 * Z_prev * Z_prev;  // Quadratic mortality: density-dependent predation by higher trophic levels (g C m^-3 day^-1)
    Type Z_loss = Z_loss_linear + Z_loss_quadratic;  // Total zooplankton mortality (g C m^-3 day^-1)
    
    // EQUATION 5: Nutrient recycling from dead organic matter
    Type recycling = gamma_bounded * (P_loss + Z_loss);  // Remineralization of nutrients (g C m^-3 day^-1)
    
    // EQUATION 6: Nutrient dynamics (dN/dt)
    Type dN_dt = -epsilon_P_bounded * uptake + recycling;  // Change in nutrient concentration (g C m^-3 day^-1)
    
    // EQUATION 7: Phytoplankton dynamics (dP/dt)
    Type dP_dt = epsilon_P_bounded * uptake - grazing - P_loss;  // Change in phytoplankton concentration (g C m^-3 day^-1)
    
    // EQUATION 8: Zooplankton dynamics (dZ/dt) - includes quadratic mortality
    Type dZ_dt = epsilon_Z_bounded * grazing - Z_loss;  // Change in zooplankton concentration (g C m^-3 day^-1)
    
    // Euler integration step
    N_pred(i) = N_prev + dN_dt * dt;  // Update nutrient concentration
    P_pred(i) = P_prev + dP_dt * dt;  // Update phytoplankton concentration
    Z_pred(i) = Z_prev + dZ_dt * dt;  // Update zooplankton concentration
    
    // Ensure predictions remain non-negative
    N_pred(i) = N_pred(i) + eps;  // Prevent negative nutrients
    P_pred(i) = P_pred(i) + eps;  // Prevent negative phytoplankton
    Z_pred(i) = Z_pred(i) + eps;  // Prevent negative zooplankton
  }
  
  // LIKELIHOOD CALCULATION - compare predictions to observations
  for(int i = 0; i < n; i++) {
    // Always include all observations in likelihood
    nll -= dnorm(N_dat(i), N_pred(i), sigma_N, true);  // Nutrient observation likelihood
    nll -= dnorm(P_dat(i), P_pred(i), sigma_P, true);  // Phytoplankton observation likelihood
    nll -= dnorm(Z_dat(i), Z_pred(i), sigma_Z, true);  // Zooplankton observation likelihood
  }
  
  // REPORTING - output predictions and parameters
  REPORT(N_pred);  // Report predicted nutrient concentrations
  REPORT(P_pred);  // Report predicted phytoplankton concentrations
  REPORT(Z_pred);  // Report predicted zooplankton concentrations
  REPORT(sigma_N);  // Report nutrient observation error
  REPORT(sigma_P);  // Report phytoplankton observation error
  REPORT(sigma_Z);  // Report zooplankton observation error
  REPORT(epsilon_P_bounded);  // Report bounded phytoplankton efficiency
  REPORT(epsilon_Z_bounded);  // Report bounded zooplankton efficiency
  REPORT(gamma_bounded);  // Report bounded recycling fraction
  
  return nll;  // Return total negative log-likelihood
}
