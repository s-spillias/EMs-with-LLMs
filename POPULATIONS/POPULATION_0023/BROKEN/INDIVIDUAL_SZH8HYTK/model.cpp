#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS
  DATA_VECTOR(Year);                    // Time vector (years)
  DATA_VECTOR(cots_dat);               // Observed total COTS abundance (juveniles + adults, individuals/m2)
  DATA_VECTOR(fast_dat);               // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);               // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);                // Sea surface temperature forcing (Celsius)
  DATA_VECTOR(cotsimm_dat);            // COTS larval immigration forcing (individuals/m2/year)
  
  // COTS POPULATION PARAMETERS - AGE STRUCTURED
  PARAMETER(log_r_cots);                // Log intrinsic reproductive rate of adult COTS (year^-1)
  PARAMETER(log_K_cots_base);           // Log baseline carrying capacity of adult COTS (individuals/m2)
  PARAMETER(log_m_cots);                // Log natural mortality rate of adult COTS (year^-1)
  PARAMETER(log_m_cots_juvenile);       // Log baseline natural mortality rate of juvenile COTS (year^-1)
  PARAMETER(log_predation_mortality_base); // Log baseline predation mortality rate on juveniles (year^-1)
  PARAMETER(log_predator_saturation);   // Log juvenile density at which predators become 50% saturated (individuals/m2)
  PARAMETER(predator_swamping_exponent); // Steepness of predator saturation response (Hill coefficient)
  PARAMETER(predation_refuge_interaction); // Strength of coral refuge effect on predation mortality (0-1)
  PARAMETER(log_maturation_rate);       // Log rate of maturation from juvenile to adult (year^-1)
  PARAMETER(juvenile_feeding_efficiency); // Relative feeding efficiency of juveniles vs adults (dimensionless, 0-1)
  PARAMETER(log_allee_threshold);       // Log Allee threshold density for adults (individuals/m2)
  PARAMETER(allee_strength);            // Allee effect strength (dimensionless, 0-1)
  PARAMETER(log_dd_mortality);          // Log density-dependent mortality coefficient for adults (m2/individuals/year)
  
  // CORAL REFUGE PARAMETERS FOR JUVENILE COTS
  PARAMETER(log_K_refuge);              // Log coral cover providing 50% of maximum refuge benefit (%)
  PARAMETER(refuge_strength);           // Maximum proportional reduction in juvenile natural mortality (dimensionless, 0-1)
  PARAMETER(refuge_steepness);          // Steepness of refuge response curve (Hill coefficient, dimensionless)
  
  // TEMPERATURE EFFECTS ON COTS
  PARAMETER(temp_opt);                  // Optimal temperature for COTS recruitment (Celsius)
  PARAMETER(log_temp_width);            // Log temperature tolerance width (Celsius)
  PARAMETER(log_temp_effect_max);       // Log maximum temperature effect multiplier (dimensionless)
  
  // LARVAL SURVIVAL PARAMETERS
  PARAMETER(log_larval_survival_efficiency);  // Log efficiency of larval survival under optimal conditions (dimensionless)
  PARAMETER(log_nutrient_half_sat);     // Log half-saturation constant for nutrient effect (nutrient units)
  PARAMETER(log_nutrient_outbreak_threshold);  // Log nutrient threshold for outbreak enhancement (nutrient units)
  PARAMETER(nutrient_outbreak_multiplier);     // Additional boost to larval survival during high nutrients (dimensionless)
  PARAMETER(nutrient_response_steepness);      // Steepness of sigmoid response at threshold (dimensionless)
  
  // CORAL DYNAMICS PARAMETERS
  PARAMETER(log_r_fast);                // Log growth rate of fast-growing coral (year^-1)
  PARAMETER(log_r_slow);                // Log growth rate of slow-growing coral (year^-1)
  PARAMETER(log_K_coral);               // Log carrying capacity for total coral cover (%)
  PARAMETER(competition_strength);      // Competition coefficient between coral types (dimensionless)
  
  // COTS FEEDING PARAMETERS
  PARAMETER(log_attack_fast);           // Log attack rate on fast-growing coral (m2/individuals/year)
  PARAMETER(log_attack_slow);           // Log attack rate on slow-growing coral (m2/individuals/year)
  PARAMETER(log_handling_time);         // Log handling time for coral consumption (year)
  PARAMETER(feeding_preference);        // Preference for fast vs slow coral (dimensionless, >1 prefers fast)
  PARAMETER(log_coral_to_cots);         // Log conversion efficiency of coral to COTS biomass (dimensionless)
  
  // OBSERVATION ERROR PARAMETERS
  PARAMETER(log_sigma_cots);            // Log observation error SD for total COTS (individuals/m2)
  PARAMETER(log_sigma_fast);            // Log observation error SD for fast coral (%)
  PARAMETER(log_sigma_slow);            // Log observation error SD for slow coral (%)
  
  // Transform parameters from log scale
  Type r_cots = exp(log_r_cots);
  Type K_cots_base = exp(log_K_cots_base);
  Type m_cots = exp(log_m_cots);
  Type m_cots_juvenile = exp(log_m_cots_juvenile);
  Type predation_mortality_base = exp(log_predation_mortality_base);
  Type predator_saturation = exp(log_predator_saturation);
  Type maturation_rate = exp(log_maturation_rate);
  Type allee_threshold = exp(log_allee_threshold);
  Type dd_mortality = exp(log_dd_mortality);
  Type K_refuge = exp(log_K_refuge);
  Type temp_width = exp(log_temp_width);
  Type temp_effect_max = exp(log_temp_effect_max);
  Type larval_survival_efficiency = exp(log_larval_survival_efficiency);
  Type nutrient_half_sat = exp(log_nutrient_half_sat);
  Type nutrient_outbreak_threshold = exp(log_nutrient_outbreak_threshold);
  Type r_fast = exp(log_r_fast);
  Type r_slow = exp(log_r_slow);
  Type K_coral = exp(log_K_coral);
  Type attack_fast = exp(log_attack_fast);
  Type attack_slow = exp(log_attack_slow);
  Type handling_time = exp(log_handling_time);
  Type coral_to_cots = exp(log_coral_to_cots);
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_fast = exp(log_sigma_fast);
  Type sigma_slow = exp(log_sigma_slow);
  
  // Add minimum sigma values for numerical stability
  Type min_sigma = Type(0.01);
  sigma_cots = sigma_cots + min_sigma;
  sigma_fast = sigma_fast + min_sigma;
  sigma_slow = sigma_slow + min_sigma;
  
  // Initialize prediction vectors - now tracking juveniles and adults separately
  int n = Year.size();
  vector<Type> juvenile_pred(n);  // Juvenile COTS (age 0-2 years)
  vector<Type> adult_pred(n);     // Adult COTS (age 2+ years)
  vector<Type> cots_pred(n);      // Total COTS (for comparison with data)
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Set initial conditions from data
  // Assume initial COTS are split 30% juvenile, 70% adult (reasonable for non-outbreak conditions)
  juvenile_pred(0) = cots_dat(0) * Type(0.3);
  adult_pred(0) = cots_dat(0) * Type(0.7);
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // TIME LOOP: Simulate dynamics forward in time
  for(int t = 0; t < n-1; t++) {
    
    // Current state (using only previous time step values)
    Type juvenile_curr = juvenile_pred(t);
    Type adult_curr = adult_pred(t);
    Type fast_curr = fast_pred(t);
    Type slow_curr = slow_pred(t);
    Type immigration_curr = cotsimm_dat(t);
    
    // Ensure positive values with simple max
    if(asDouble(juvenile_curr) < 0.001) juvenile_curr = Type(0.001);
    if(asDouble(adult_curr) < 0.001) adult_curr = Type(0.001);
    if(asDouble(fast_curr) < 0.5) fast_curr = Type(0.5);
    if(asDouble(slow_curr) < 0.5) slow_curr = Type(0.5);
    
    // ULTRA-SIMPLIFIED DYNAMICS
    Type total_coral = fast_curr + slow_curr;
    
    // Simple juvenile mortality (constant)
    Type juvenile_mortality = m_cots_juvenile + predation_mortality_base;
    if(asDouble(juvenile_mortality) < 0.1) juvenile_mortality = Type(0.1);
    if(asDouble(juvenile_mortality) > 2.0) juvenile_mortality = Type(2.0);
    
    // Simple recruitment
    Type recruitment = immigration_curr * larval_survival_efficiency;
    if(asDouble(recruitment) < 0.0) recruitment = Type(0.0);
    if(asDouble(recruitment) > immigration_curr * 3.0) recruitment = immigration_curr * Type(3.0);
    
    // Maturation
    Type maturation_flux = maturation_rate * juvenile_curr;
    
    // Update JUVENILE COTS - simple Euler
    juvenile_pred(t+1) = juvenile_curr + recruitment - maturation_flux - juvenile_mortality * juvenile_curr;
    if(asDouble(juvenile_pred(t+1)) < 0.001) juvenile_pred(t+1) = Type(0.001);
    if(asDouble(juvenile_pred(t+1)) > 20.0) juvenile_pred(t+1) = Type(20.0);
    
    // Update ADULT COTS - simple mortality
    adult_pred(t+1) = adult_curr + maturation_flux - m_cots * adult_curr;
    if(asDouble(adult_pred(t+1)) < 0.001) adult_pred(t+1) = Type(0.001);
    if(asDouble(adult_pred(t+1)) > 20.0) adult_pred(t+1) = Type(20.0);
    
    // Total COTS
    cots_pred(t+1) = juvenile_pred(t+1) + adult_pred(t+1);
    
    // SIMPLIFIED CORAL DYNAMICS
    Type total_cots = adult_curr + juvenile_curr * Type(0.3);
    
    // Simple linear consumption
    Type consumption_fast = attack_fast * fast_curr * total_cots * Type(0.1);
    Type consumption_slow = attack_slow * slow_curr * total_cots * Type(0.1);
    
    if(asDouble(consumption_fast) > fast_curr * 0.5) consumption_fast = fast_curr * Type(0.5);
    if(asDouble(consumption_slow) > slow_curr * 0.5) consumption_slow = slow_curr * Type(0.5);
    
    // Update corals - simple logistic
    fast_pred(t+1) = fast_curr + r_fast * fast_curr * (Type(1.0) - total_coral / K_coral) - consumption_fast;
    if(asDouble(fast_pred(t+1)) < 0.5) fast_pred(t+1) = Type(0.5);
    if(asDouble(fast_pred(t+1)) > K_coral * 0.8) fast_pred(t+1) = K_coral * Type(0.8);
    
    slow_pred(t+1) = slow_curr + r_slow * slow_curr * (Type(1.0) - total_coral / K_coral) - consumption_slow;
    if(asDouble(slow_pred(t+1)) < 0.5) slow_pred(t+1) = Type(0.5);
    if(asDouble(slow_pred(t+1)) > K_coral * 0.8) slow_pred(t+1) = K_coral * Type(0.8);
  }
  
  // LIKELIHOOD CALCULATION
  Type nll = Type(0.0);
  
  for(int t = 0; t < n; t++) {
    nll -= dnorm(cots_dat(t), cots_pred(t), sigma_cots, true);
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_fast, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_slow, true);
  }
  
  // REPORT PREDICTIONS
  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);
  ADREPORT(juvenile_pred);
  ADREPORT(adult_pred);
  
  return nll;
}
