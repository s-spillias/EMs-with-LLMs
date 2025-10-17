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
  PARAMETER(epsilon_P_max);  // Maximum nutrient uptake efficiency at low nutrients (dimensionless, 0-1) - from luxury uptake studies
  PARAMETER(K_eff);  // Half-saturation constant for efficiency response (g C m^-3) - controls efficiency variation with nutrients
  
  // PARAMETERS - Light limitation
  PARAMETER(I_0);  // Surface light intensity (μmol photons m^-2 s^-1) - incoming solar radiation
  PARAMETER(K_I);  // Half-saturation constant for light (μmol photons m^-2 s^-1) - phytoplankton light requirement
  PARAMETER(k_bg);  // Background light attenuation coefficient (m^-1) - water clarity/turbidity
  PARAMETER(k_c);  // Phytoplankton self-shading coefficient (m^2 (g C)^-1) - chlorophyll-specific attenuation
  PARAMETER(H);  // Mixed layer depth (m) - depth over which phytoplankton are mixed
  
  // PARAMETERS - Temperature dependence (NEW)
  PARAMETER(T);  // Water temperature (°C) - ambient temperature in mixed layer
  PARAMETER(T_ref);  // Reference temperature (°C) - temperature at which base rates are defined (typically 20°C)
  PARAMETER(k_T_growth);  // Temperature coefficient for phytoplankton growth (°C^-1) - from Eppley 1972
  PARAMETER(k_T_grazing);  // Temperature coefficient for zooplankton grazing (°C^-1) - heterotroph temperature sensitivity
  PARAMETER(k_T_mortality);  // Temperature coefficient for mortality/remineralization (°C^-1) - bacterial processes
  
  // PARAMETERS - Zooplankton grazing
  PARAMETER(g_max);  // Maximum zooplankton grazing rate at T_ref (day^-1) - from feeding experiments
  PARAMETER(K_Z);  // Half-saturation constant for grazing (g C m^-3) - from functional response experiments
  PARAMETER(epsilon_Z);  // Assimilation efficiency of zooplankton (dimensionless, 0-1) - from growth efficiency studies
  
  // PARAMETERS - Mortality and recycling
  PARAMETER(m_P);  // Phytoplankton mortality rate at T_ref (day^-1) - from dilution experiments and natural mortality observations
  PARAMETER(m_Z);  // Zooplankton linear mortality rate at T_ref (day^-1) - from population dynamics studies (natural death, senescence)
  PARAMETER(m_Z2);  // Zooplankton quadratic mortality coefficient ((g C m^-3)^-1 day^-1) - represents density-dependent predation by higher trophic levels
  PARAMETER(gamma);  // Nutrient recycling fraction (dimensionless, 0-1) - from remineralization rate measurements
  
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
  
  // TEMPERATURE MODULATION OF RATES (Eppley 1972 formulation)
  // All biological rates scale exponentially with temperature: rate(T) = rate(T_ref) × exp(k_T × (T - T_ref))
  Type temp_diff = T - T_ref;  // Temperature difference from reference (°C)
  Type f_T_growth = exp(k_T_growth * temp_diff);  // Temperature factor for phytoplankton growth (dimensionless)
  Type f_T_grazing = exp(k_T_grazing * temp_diff);  // Temperature factor for zooplankton grazing (dimensionless)
  Type f_T_mortality = exp(k_T_mortality * temp_diff);  // Temperature factor for mortality and remineralization (dimensionless)
  
  // Temperature-adjusted rates
  Type r_P_T = r_P * f_T_growth;  // Temperature-adjusted phytoplankton growth rate (day^-1)
  Type g_max_T = g_max * f_T_grazing;  // Temperature-adjusted zooplankton grazing rate (day^-1)
  Type m_P_T = m_P * f_T_mortality;  // Temperature-adjusted phytoplankton mortality rate (day^-1)
  Type m_Z_T = m_Z * f_T_mortality;  // Temperature-adjusted zooplankton mortality rate (day^-1)
  
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
  nll -= dnorm(K_eff, Type(0.5), Type(1.0), true);  // Prior: K_eff should be positive, centered around 0.5
  
  // Soft bounds for light parameters
  nll -= dnorm(I_0, Type(400.0), Type(500.0), true);  // Prior: I_0 should be positive, centered around 400
  nll -= dnorm(K_I, Type(50.0), Type(100.0), true);  // Prior: K_I should be positive, centered around 50
  nll -= dnorm(k_bg, Type(0.04), Type(0.1), true);  // Prior: k_bg should be small positive, centered around 0.04
  nll -= dnorm(k_c, Type(0.03), Type(0.05), true);  // Prior: k_c should be small positive, centered around 0.03
  nll -= dnorm(H, Type(50.0), Type(50.0), true);  // Prior: H should be positive, centered around 50
  
  // Soft bounds for temperature parameters (NEW)
  nll -= dnorm(T, Type(15.0), Type(10.0), true);  // Prior: T should be reasonable oceanic temperature, centered around 15°C
  nll -= dnorm(T_ref, Type(20.0), Type(5.0), true);  // Prior: T_ref should be near standard 20°C
  nll -= dnorm(k_T_growth, Type(0.063), Type(0.02), true);  // Prior: k_T_growth centered on Eppley 1972 value
  nll -= dnorm(k_T_grazing, Type(0.069), Type(0.02), true);  // Prior: k_T_grazing slightly higher than growth
  nll -= dnorm(k_T_mortality, Type(0.069), Type(0.02), true);  // Prior: k_T_mortality similar to grazing
  
  // Soft bounds for efficiencies (should be between 0 and 1)
  Type epsilon_P_max_bounded = Type(1.0) / (Type(1.0) + exp(-epsilon_P_max));  // Logistic transform to bound between 0 and 1
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
    
    // EQUATION 1: Light limitation with self-shading
    // Calculate total light attenuation coefficient (background + phytoplankton self-shading)
    Type k_total = k_bg + k_c * P_prev;  // Total attenuation (m^-1)
    
    // Calculate average light intensity in mixed layer using Beer-Lambert law
    // I_avg = I_0 * (1 - exp(-k_total * H)) / (k_total * H)
    // This represents the depth-averaged light available to phytoplankton
    Type k_H = k_total * H + eps;  // Total attenuation over mixed layer depth
    Type exp_term = exp(-k_H);  // Exponential decay term
    Type I_avg = I_0 * (Type(1.0) - exp_term) / k_H;  // Average light intensity (μmol photons m^-2 s^-1)
    
    // Light limitation factor (Monod kinetics for light)
    Type light_limitation = I_avg / (K_I + I_avg + eps);  // Light limitation factor (0-1)
    
    // EQUATION 2: Nutrient limitation
    Type nutrient_limitation = N_prev / (K_N + N_prev + eps);  // Nutrient limitation factor (0-1)
    
    // EQUATION 3: Variable nutrient uptake efficiency
    // Efficiency decreases with increasing nutrient availability
    // At low N: efficiency approaches epsilon_P_max (luxury uptake)
    // At high N: efficiency approaches 0 (saturation of uptake machinery)
    Type epsilon_P_variable = epsilon_P_max_bounded * K_eff / (K_eff + N_prev + eps);  // Variable efficiency (0-1)
    
    // EQUATION 4: Phytoplankton growth with Liebig's Law of the Minimum and TEMPERATURE DEPENDENCE
    // Growth is limited by the MOST LIMITING resource (light OR nutrients)
    // This implements Liebig's Law: growth is controlled by the scarcest resource
    // More ecologically realistic than multiplicative co-limitation
    // NOW MODULATED BY TEMPERATURE using r_P_T instead of r_P
    Type limitation_factor = CppAD::CondExpLt(light_limitation, nutrient_limitation, 
                                               light_limitation, nutrient_limitation);  // min(light, nutrient)
    Type uptake = r_P_T * limitation_factor * P_prev;  // Temperature-adjusted resource-limited phytoplankton growth (g C m^-3 day^-1)
    
    // EQUATION 5: Zooplankton grazing with TEMPERATURE DEPENDENCE (Holling Type II functional response)
    // NOW MODULATED BY TEMPERATURE using g_max_T instead of g_max
    Type grazing = g_max_T * (P_prev / (K_Z + P_prev + eps)) * Z_prev;  // Temperature-adjusted phytoplankton consumption by zooplankton (g C m^-3 day^-1)
    
    // EQUATION 6: Phytoplankton mortality with TEMPERATURE DEPENDENCE
    // NOW MODULATED BY TEMPERATURE using m_P_T instead of m_P
    Type P_loss = m_P_T * P_prev;  // Temperature-adjusted phytoplankton mortality (g C m^-3 day^-1)
    
    // EQUATION 7: Zooplankton mortality and losses with TEMPERATURE DEPENDENCE (LINEAR + QUADRATIC)
    // Linear mortality NOW MODULATED BY TEMPERATURE using m_Z_T instead of m_Z
    Type Z_loss_linear = m_Z_T * Z_prev;  // Temperature-adjusted linear mortality: natural death and senescence (g C m^-3 day^-1)
    Type Z_loss_quadratic = m_Z2 * Z_prev * Z_prev;  // Quadratic mortality: density-dependent predation by higher trophic levels (g C m^-3 day^-1)
    Type Z_loss = Z_loss_linear + Z_loss_quadratic;  // Total zooplankton mortality (g C m^-3 day^-1)
    
    // EQUATION 8: Nutrient recycling from dead organic matter
    // Recycling rate is implicitly temperature-dependent through temperature-adjusted mortality rates
    Type recycling = gamma_bounded * (P_loss + Z_loss);  // Remineralization of nutrients (g C m^-3 day^-1)
    
    // EQUATION 9: Nutrient dynamics (dN/dt)
    // Now uses variable efficiency that depends on current nutrient concentration
    Type dN_dt = -epsilon_P_variable * uptake + recycling;  // Change in nutrient concentration (g C m^-3 day^-1)
    
    // EQUATION 10: Phytoplankton dynamics (dP/dt)
    // Now uses variable efficiency that depends on current nutrient concentration
    Type dP_dt = epsilon_P_variable * uptake - grazing - P_loss;  // Change in phytoplankton concentration (g C m^-3 day^-1)
    
    // EQUATION 11: Zooplankton dynamics (dZ/dt)
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
  REPORT(epsilon_P_max_bounded);  // Report bounded maximum phytoplankton efficiency
  REPORT(epsilon_Z_bounded);  // Report bounded zooplankton efficiency
  REPORT(gamma_bounded);  // Report bounded recycling fraction
  REPORT(f_T_growth);  // Report temperature factor for growth
  REPORT(f_T_grazing);  // Report temperature factor for grazing
  REPORT(f_T_mortality);  // Report temperature factor for mortality
  REPORT(r_P_T);  // Report temperature-adjusted phytoplankton growth rate
  REPORT(g_max_T);  // Report temperature-adjusted grazing rate
  REPORT(m_P_T);  // Report temperature-adjusted phytoplankton mortality
  REPORT(m_Z_T);  // Report temperature-adjusted zooplankton mortality
  
  return nll;  // Return total negative log-likelihood
}
