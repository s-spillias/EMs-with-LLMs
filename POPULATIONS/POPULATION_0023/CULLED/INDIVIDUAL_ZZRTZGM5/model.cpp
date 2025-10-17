#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS
  DATA_VECTOR(Year);                    // Time vector (years)
  DATA_VECTOR(cots_dat);               // Observed COTS abundance (individuals/m²)
  DATA_VECTOR(fast_dat);               // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);               // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);                // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);            // COTS larval immigration rate (individuals/m²/year)
  
  // COTS POPULATION PARAMETERS
  PARAMETER(log_r_cots);                // Log intrinsic recruitment rate of COTS (year⁻¹)
  PARAMETER(log_K_cots);                // Log carrying capacity for COTS density-dependent mortality (individuals/m²)
  PARAMETER(log_m_cots);                // Log baseline natural mortality rate of COTS (year⁻¹)
  PARAMETER(log_allee_threshold);       // Log Allee effect threshold density (individuals/m²)
  PARAMETER(log_temp_opt);              // Log optimal temperature for COTS recruitment (°C)
  PARAMETER(log_temp_width);            // Log temperature tolerance width (°C)
  PARAMETER(log_dd_mort_strength);      // Log strength of density-dependent mortality (m²/individuals)
  
  // CORAL GROWTH PARAMETERS
  PARAMETER(log_r_fast);                // Log intrinsic growth rate of fast-growing coral (year⁻¹)
  PARAMETER(log_r_slow);                // Log intrinsic growth rate of slow-growing coral (year⁻¹)
  PARAMETER(log_K_coral);               // Log total coral carrying capacity (% cover)
  PARAMETER(log_temp_bleach_threshold); // Log temperature threshold for bleaching stress (°C)
  PARAMETER(log_bleach_mort_rate);      // Log bleaching mortality rate (year⁻¹)
  
  // PREDATION PARAMETERS
  PARAMETER(log_attack_fast);           // Log attack rate on fast-growing coral (m²/individuals/year)
  PARAMETER(log_attack_slow);           // Log attack rate on slow-growing coral (m²/individuals/year)
  PARAMETER(log_handling_fast);         // Log handling time for fast-growing coral (year)
  PARAMETER(log_handling_slow);         // Log handling time for slow-growing coral (year)
  PARAMETER(log_predation_efficiency);  // Log conversion efficiency of coral to COTS biomass support (dimensionless)
  
  // OBSERVATION ERROR PARAMETERS
  PARAMETER(log_sigma_cots);            // Log observation error SD for COTS (individuals/m²)
  PARAMETER(log_sigma_fast);            // Log observation error SD for fast coral (%)
  PARAMETER(log_sigma_slow);            // Log observation error SD for slow coral (%)
  
  // Transform parameters from log scale
  Type r_cots = exp(log_r_cots);                          // COTS recruitment rate (year⁻¹)
  Type K_cots = exp(log_K_cots);                          // COTS carrying capacity (individuals/m²)
  Type m_cots = exp(log_m_cots);                          // COTS baseline mortality (year⁻¹)
  Type allee_threshold = exp(log_allee_threshold);        // Allee threshold (individuals/m²)
  Type temp_opt = exp(log_temp_opt);                      // Optimal temperature (°C)
  Type temp_width = exp(log_temp_width);                  // Temperature tolerance (°C)
  Type dd_mort_strength = exp(log_dd_mort_strength);      // Density-dependent mortality strength (m²/individuals)
  Type r_fast = exp(log_r_fast);                          // Fast coral growth rate (year⁻¹)
  Type r_slow = exp(log_r_slow);                          // Slow coral growth rate (year⁻¹)
  Type K_coral = exp(log_K_coral);                        // Coral carrying capacity (%)
  Type temp_bleach_threshold = exp(log_temp_bleach_threshold); // Bleaching threshold (°C)
  Type bleach_mort_rate = exp(log_bleach_mort_rate);      // Bleaching mortality (year⁻¹)
  Type attack_fast = exp(log_attack_fast);                // Attack rate on fast coral (m²/individuals/year)
  Type attack_slow = exp(log_attack_slow);                // Attack rate on slow coral (m²/individuals/year)
  Type handling_fast = exp(log_handling_fast);            // Handling time fast coral (year)
  Type handling_slow = exp(log_handling_slow);            // Handling time slow coral (year)
  Type predation_efficiency = exp(log_predation_efficiency); // Predation efficiency (dimensionless)
  Type sigma_cots = exp(log_sigma_cots);                  // COTS observation error (individuals/m²)
  Type sigma_fast = exp(log_sigma_fast);                  // Fast coral observation error (%)
  Type sigma_slow = exp(log_sigma_slow);                  // Slow coral observation error (%)
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(0.01);                            // Minimum SD for all observations
  sigma_cots = sigma_cots + min_sigma;                    // Ensure minimum COTS SD
  sigma_fast = sigma_fast + min_sigma;                    // Ensure minimum fast coral SD
  sigma_slow = sigma_slow + min_sigma;                    // Ensure minimum slow coral SD
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);                                  // Small constant for numerical stability
  
  // Initialize prediction vectors
  int n = Year.size();                                    // Number of time steps
  vector<Type> cots_pred(n);                             // Predicted COTS abundance (individuals/m²)
  vector<Type> fast_pred(n);                             // Predicted fast coral cover (%)
  vector<Type> slow_pred(n);                             // Predicted slow coral cover (%)
  
  // Set initial conditions from first observation
  cots_pred(0) = cots_dat(0);                            // Initialize COTS from data
  fast_pred(0) = fast_dat(0);                            // Initialize fast coral from data
  slow_pred(0) = slow_dat(0);                            // Initialize slow coral from data
  
  // Time step (assuming annual data)
  Type dt = Type(1.0);                                    // Time step (years)
  
  // PROCESS MODEL: Iterate through time steps
  for(int t = 1; t < n; t++) {
    
    // Get previous time step values (avoid data leakage)
    Type cots_prev = cots_pred(t-1);                     // Previous COTS abundance (individuals/m²)
    Type fast_prev = fast_pred(t-1);                     // Previous fast coral cover (%)
    Type slow_prev = slow_pred(t-1);                     // Previous slow coral cover (%)
    Type sst = sst_dat(t);                               // Current sea surface temperature (°C)
    Type cots_immigration = cotsimm_dat(t);              // Current larval immigration (individuals/m²/year)
    
    // Equation 1: Temperature effect on COTS recruitment (Gaussian response)
    Type temp_effect = exp(-pow(sst - temp_opt, 2) / (2.0 * pow(temp_width, 2) + eps)); // Temperature effect on recruitment (dimensionless, 0-1)
    
    // Equation 2: Allee effect on COTS recruitment (sigmoid function)
    Type allee_effect = cots_prev / (allee_threshold + cots_prev + eps); // Allee effect (dimensionless, 0-1)
    
    // Equation 3: Type II functional response for COTS predation on fast-growing coral
    Type predation_fast = (attack_fast * fast_prev * cots_prev) / (1.0 + attack_fast * handling_fast * fast_prev + eps); // Predation on fast coral (%/year)
    
    // Equation 4: Type II functional response for COTS predation on slow-growing coral
    Type predation_slow = (attack_slow * slow_prev * cots_prev) / (1.0 + attack_slow * handling_slow * slow_prev + eps); // Predation on slow coral (%/year)
    
    // Equation 5: Total coral availability for COTS (food resource)
    Type total_coral = fast_prev + slow_prev + eps;      // Total coral cover (%)
    
    // Equation 6: Resource-dependent COTS recruitment boost
    Type resource_effect = total_coral / (K_coral + eps); // Resource availability effect (dimensionless, 0-1)
    
    // Equation 7: COTS recruitment (temperature, Allee, immigration, resource effects)
    Type cots_recruitment = r_cots * cots_prev * temp_effect * allee_effect * resource_effect + cots_immigration; // COTS recruitment (individuals/m²/year)
    
    // Equation 8: Density-dependent COTS mortality
    Type cots_mortality = (m_cots + dd_mort_strength * cots_prev) * cots_prev; // COTS mortality (individuals/m²/year)
    
    // Equation 9: COTS population change
    Type dcots_dt = cots_recruitment - cots_mortality;   // COTS population change (individuals/m²/year)
    
    // Equation 10: Total coral cover (for competition)
    Type total_coral_cover = fast_prev + slow_prev;      // Total coral cover (%)
    
    // Equation 11: Space limitation for coral growth (smooth max to ensure non-negative)
    Type space_deficit = K_coral - total_coral_cover;    // Space deficit (%)
    Type space_available = (space_deficit + sqrt(pow(space_deficit, 2) + Type(4.0) * eps)) / (Type(2.0) * (K_coral + eps)); // Available space (dimensionless, 0-1) using smooth approximation
    
    // Equation 12: Temperature bleaching stress (smooth threshold)
    Type bleaching_stress = Type(1.0) / (Type(1.0) + exp(-Type(5.0) * (sst - temp_bleach_threshold))); // Bleaching stress (dimensionless, 0-1)
    
    // Equation 13: Fast-growing coral growth
    Type fast_growth = r_fast * fast_prev * space_available; // Fast coral growth (%/year)
    
    // Equation 14: Fast-growing coral mortality (predation + bleaching)
    Type fast_mortality = predation_fast + bleaching_stress * bleach_mort_rate * fast_prev; // Fast coral mortality (%/year)
    
    // Equation 15: Fast-growing coral population change
    Type dfast_dt = fast_growth - fast_mortality;        // Fast coral change (%/year)
    
    // Equation 16: Slow-growing coral growth
    Type slow_growth = r_slow * slow_prev * space_available; // Slow coral growth (%/year)
    
    // Equation 17: Slow-growing coral mortality (predation + bleaching, more resistant)
    Type slow_mortality = predation_slow + Type(0.5) * bleaching_stress * bleach_mort_rate * slow_prev; // Slow coral mortality (%/year)
    
    // Equation 18: Slow-growing coral population change
    Type dslow_dt = slow_growth - slow_mortality;        // Slow coral change (%/year)
    
    // Update predictions with smooth bounds to ensure biological realism
    // Equation 19: Update COTS (smooth lower bound at 0)
    Type cots_new = cots_prev + dt * dcots_dt;           // Tentative new COTS value
    cots_pred(t) = (cots_new + sqrt(pow(cots_new, 2) + Type(4.0) * eps)) / Type(2.0); // Smooth max with 0
    
    // Equation 20: Update fast coral (smooth bounds at 0 and 100)
    Type fast_new = fast_prev + dt * dfast_dt;           // Tentative new fast coral value
    Type fast_lower = (fast_new + sqrt(pow(fast_new, 2) + Type(4.0) * eps)) / Type(2.0); // Smooth max with 0
    Type fast_excess = fast_lower - Type(100.0);         // Excess above 100
    fast_pred(t) = fast_lower - (fast_excess + sqrt(pow(fast_excess, 2) + Type(4.0) * eps)) / Type(2.0); // Smooth min with 100
    
    // Equation 21: Update slow coral (smooth bounds at 0 and 100)
    Type slow_new = slow_prev + dt * dslow_dt;           // Tentative new slow coral value
    Type slow_lower = (slow_new + sqrt(pow(slow_new, 2) + Type(4.0) * eps)) / Type(2.0); // Smooth max with 0
    Type slow_excess = slow_lower - Type(100.0);         // Excess above 100
    slow_pred(t) = slow_lower - (slow_excess + sqrt(pow(slow_excess, 2) + Type(4.0) * eps)) / Type(2.0); // Smooth min with 100
  }
  
  // LIKELIHOOD CALCULATION
  Type nll = Type(0.0);                                  // Initialize negative log-likelihood
  
  // Equation 22: Likelihood for COTS observations (normal distribution)
  for(int t = 0; t < n; t++) {
    nll -= dnorm(cots_dat(t), cots_pred(t), sigma_cots, true); // COTS observation likelihood
  }
  
  // Equation 23: Likelihood for fast-growing coral observations (normal distribution)
  for(int t = 0; t < n; t++) {
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_fast, true); // Fast coral observation likelihood
  }
  
  // Equation 24: Likelihood for slow-growing coral observations (normal distribution)
  for(int t = 0; t < n; t++) {
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_slow, true); // Slow coral observation likelihood
  }
  
  // REPORT PREDICTIONS
  REPORT(cots_pred);                                     // Report predicted COTS abundance
  REPORT(fast_pred);                                     // Report predicted fast coral cover
  REPORT(slow_pred);                                     // Report predicted slow coral cover
  
  // REPORT PARAMETERS
  REPORT(r_cots);                                        // Report COTS recruitment rate
  REPORT(K_cots);                                        // Report COTS carrying capacity
  REPORT(m_cots);                                        // Report COTS mortality rate
  REPORT(allee_threshold);                               // Report Allee threshold
  REPORT(temp_opt);                                      // Report optimal temperature
  REPORT(temp_width);                                    // Report temperature tolerance
  REPORT(dd_mort_strength);                              // Report density-dependent mortality
  REPORT(r_fast);                                        // Report fast coral growth rate
  REPORT(r_slow);                                        // Report slow coral growth rate
  REPORT(K_coral);                                       // Report coral carrying capacity
  REPORT(temp_bleach_threshold);                         // Report bleaching threshold
  REPORT(bleach_mort_rate);                              // Report bleaching mortality
  REPORT(attack_fast);                                   // Report attack rate on fast coral
  REPORT(attack_slow);                                   // Report attack rate on slow coral
  REPORT(handling_fast);                                 // Report handling time fast coral
  REPORT(handling_slow);                                 // Report handling time slow coral
  REPORT(predation_efficiency);                          // Report predation efficiency
  REPORT(sigma_cots);                                    // Report COTS observation error
  REPORT(sigma_fast);                                    // Report fast coral observation error
  REPORT(sigma_slow);                                    // Report slow coral observation error
  
  return nll;                                            // Return negative log-likelihood
}
