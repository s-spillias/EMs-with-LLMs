#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS
  DATA_VECTOR(Year);                    // Time vector (years)
  DATA_VECTOR(cots_dat);                // Observed COTS abundance (individuals/m²)
  DATA_VECTOR(fast_dat);                // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);                // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);                 // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);             // COTS larval immigration rate (individuals/m²/year)
  
  // COTS POPULATION PARAMETERS
  PARAMETER(log_r_cots);                // Log intrinsic recruitment rate of COTS (year⁻¹)
  PARAMETER(log_K_cots);                // Log carrying capacity for COTS (individuals/m²)
  PARAMETER(log_allee_threshold);       // Log Allee threshold density (individuals/m²)
  PARAMETER(allee_strength);            // Allee effect strength (dimensionless, 0-1)
  PARAMETER(log_mort_cots);             // Log baseline mortality rate of COTS (year⁻¹)
  PARAMETER(log_dd_mort);               // Log density-dependent mortality coefficient (m²/individuals/year)
  PARAMETER(temp_effect_cots);          // Temperature effect on COTS recruitment (°C⁻¹)
  PARAMETER(log_temp_opt);              // Log optimal temperature for COTS recruitment (°C)
  
  // LARVAL SETTLEMENT PARAMETERS
  PARAMETER(gregarious_strength);       // Conspecific attraction strength for settlement (dimensionless)
  PARAMETER(log_gregarious_threshold);  // Log half-saturation density for gregarious effect (individuals/m²)
  PARAMETER(temp_effect_settlement);    // Temperature sensitivity for settlement (°C⁻²)
  
  // CORAL PREDATION PARAMETERS
  PARAMETER(log_attack_fast);           // Log attack rate on fast-growing coral (m²/individuals/year)
  PARAMETER(log_attack_slow);           // Log attack rate on slow-growing coral (m²/individuals/year)
  PARAMETER(log_handling_fast);         // Log handling time for fast-growing coral (year)
  PARAMETER(log_handling_slow);         // Log handling time for slow-growing coral (year)
  PARAMETER(pred_preference);           // Predation preference for fast vs slow coral (dimensionless, 0-1)
  PARAMETER(log_interference);          // Log interference competition coefficient (m²/individuals)
  
  // CORAL GROWTH PARAMETERS
  PARAMETER(log_r_fast);                // Log intrinsic growth rate of fast coral (year⁻¹)
  PARAMETER(log_r_slow);                // Log intrinsic growth rate of slow coral (year⁻¹)
  PARAMETER(log_K_coral);               // Log total coral carrying capacity (%)
  PARAMETER(competition_coef);          // Competition coefficient between coral types (dimensionless)
  PARAMETER(log_temp_stress);           // Log temperature stress coefficient (°C⁻²)
  PARAMETER(temp_stress_threshold);     // Temperature threshold for stress (°C)
  
  // OBSERVATION ERROR PARAMETERS
  PARAMETER(log_sigma_cots);            // Log observation error SD for COTS
  PARAMETER(log_sigma_fast);            // Log observation error SD for fast coral
  PARAMETER(log_sigma_slow);            // Log observation error SD for slow coral
  
  // Transform parameters to natural scale
  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type allee_threshold = exp(log_allee_threshold);
  Type mort_cots = exp(log_mort_cots);
  Type dd_mort = exp(log_dd_mort);
  Type temp_opt = exp(log_temp_opt);
  Type gregarious_threshold = exp(log_gregarious_threshold);
  Type attack_fast = exp(log_attack_fast);
  Type attack_slow = exp(log_attack_slow);
  Type handling_fast = exp(log_handling_fast);
  Type handling_slow = exp(log_handling_slow);
  Type interference = exp(log_interference);
  Type r_fast = exp(log_r_fast);
  Type r_slow = exp(log_r_slow);
  Type K_coral = exp(log_K_coral);
  Type temp_stress_coef = exp(log_temp_stress);
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_fast = exp(log_sigma_fast);
  Type sigma_slow = exp(log_sigma_slow);
  
  // Minimum SD to prevent numerical issues
  Type min_sigma = Type(0.01);
  sigma_cots = sigma_cots + min_sigma;
  sigma_fast = sigma_fast + min_sigma;
  sigma_slow = sigma_slow + min_sigma;
  
  // Small constant for numerical stability
  Type eps = Type(1e-8);
  
  int n = Year.size();
  
  // Initialize prediction vectors
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Set initial conditions from data
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Initialize negative log-likelihood
  Type nll = Type(0.0);
  
  // Time step loop (starting from t=1)
  for(int t = 1; t < n; t++) {
    
    // Previous time step values
    Type cots_prev = cots_pred(t-1);
    Type fast_prev = fast_pred(t-1);
    Type slow_prev = slow_pred(t-1);
    Type sst_prev = sst_dat(t-1);
    Type immigration = cotsimm_dat(t-1);
    
    // Ensure non-negative values
    cots_prev = cots_prev + eps;
    fast_prev = fast_prev + eps;
    slow_prev = slow_prev + eps;
    
    // EQUATION 1: Allee effect function for local reproduction
    // Allee effect reduces recruitment at low densities due to reduced mating success
    Type allee_effect = Type(1.0) - allee_strength * exp(-cots_prev / (allee_threshold + eps));
    
    // EQUATION 2: Temperature effect on COTS local recruitment
    // Gaussian function centered on optimal temperature
    Type temp_deviation = sst_prev - temp_opt;
    Type temp_effect = exp(-temp_effect_cots * temp_deviation * temp_deviation);
    
    // EQUATION 3: Local density-dependent recruitment (from resident population)
    // Logistic growth modified by Allee effect and temperature
    Type local_recruitment = r_cots * cots_prev * (Type(1.0) - cots_prev / (K_cots + eps)) * allee_effect * temp_effect;
    
    // EQUATION 4: Substrate availability for larval settlement
    // Available space decreases as coral is lost and COTS density increases
    Type total_coral = fast_prev + slow_prev;
    Type coral_substrate = (total_coral + eps) / (K_coral + eps);  // Proportion of available coral
    Type cots_space = Type(1.0) - cots_prev / (K_cots + eps);      // Remaining space for COTS
    Type substrate_availability = coral_substrate * cots_space;
    substrate_availability = substrate_availability > eps ? substrate_availability : eps;
    
    // EQUATION 5: Gregarious settlement factor
    // Michaelis-Menten function: larvae attracted to conspecifics
    // Increases settlement success at low-moderate densities
    Type gregarious_factor = Type(1.0) + gregarious_strength * (cots_prev / (cots_prev + gregarious_threshold + eps));
    
    // EQUATION 6: Temperature suitability for settlement
    // Larvae more sensitive to temperature than adults
    Type temp_dev_settlement = sst_prev - temp_opt;
    Type temp_suitability = exp(-temp_effect_settlement * temp_dev_settlement * temp_dev_settlement);
    
    // EQUATION 7: Overall larval settlement success
    // Combines substrate, gregarious attraction, and temperature effects
    Type settlement_success = substrate_availability * gregarious_factor * temp_suitability;
    
    // EQUATION 8: Larval recruitment
    // Immigration × settlement success
    Type larval_settlement = immigration * settlement_success;
    
    // EQUATION 9: Total recruitment
    // Sum of local reproduction and external larval settlement
    Type total_recruitment = local_recruitment + larval_settlement;
    
    // EQUATION 10: Density-dependent mortality
    // Mortality increases with crowding (disease, resource competition)
    Type mortality = (mort_cots + dd_mort * cots_prev) * cots_prev;
    
    // EQUATION 11: Total coral availability for predation
    Type total_coral_prey = fast_prev + slow_prev + eps;
    
    // EQUATION 12: Interference competition among COTS
    // Reduces foraging efficiency at high COTS densities
    Type interference_effect = Type(1.0) / (Type(1.0) + interference * cots_prev);
    
    // EQUATION 13: Type II functional response for fast-growing coral
    // Predation rate saturates with coral availability
    Type pred_rate_fast = (attack_fast * fast_prev * interference_effect) / 
                          (Type(1.0) + attack_fast * handling_fast * total_coral_prey + eps);
    
    // EQUATION 14: Type II functional response for slow-growing coral
    Type pred_rate_slow = (attack_slow * slow_prev * interference_effect) / 
                          (Type(1.0) + attack_slow * handling_slow * total_coral_prey + eps);
    
    // EQUATION 15: Weighted predation preference
    // COTS preferentially feed on fast-growing corals
    Type weighted_pred_fast = pred_preference * pred_rate_fast;
    Type weighted_pred_slow = (Type(1.0) - pred_preference) * pred_rate_slow;
    
    // EQUATION 16: Total predation impact on COTS energy budget
    Type total_predation = weighted_pred_fast + weighted_pred_slow;
    
    // EQUATION 17: COTS population change
    // Recruitment (local + settlement) - mortality + predation energy gain
    Type dcots = total_recruitment - mortality + Type(0.1) * total_predation * cots_prev;
    cots_pred(t) = cots_prev + dcots;
    cots_pred(t) = cots_pred(t) > eps ? cots_pred(t) : eps;
    
    // EQUATION 18: Temperature stress on corals
    // Stress increases when temperature exceeds threshold
    Type temp_stress = Type(0.0);
    if(sst_prev > temp_stress_threshold) {
      Type stress_deviation = sst_prev - temp_stress_threshold;
      temp_stress = temp_stress_coef * stress_deviation * stress_deviation;
    }
    
    // EQUATION 19: Fast coral logistic growth
    // Growth limited by space and competition with slow corals
    Type fast_growth = r_fast * fast_prev * (Type(1.0) - (fast_prev + competition_coef * slow_prev) / (K_coral + eps));
    
    // EQUATION 20: Fast coral predation loss
    Type fast_loss = pred_rate_fast * cots_prev;
    
    // EQUATION 21: Fast coral temperature mortality
    Type fast_temp_mort = temp_stress * fast_prev;
    
    // EQUATION 22: Fast coral population change
    Type dfast = fast_growth - fast_loss - fast_temp_mort;
    fast_pred(t) = fast_prev + dfast;
    fast_pred(t) = fast_pred(t) > eps ? fast_pred(t) : eps;
    fast_pred(t) = fast_pred(t) < K_coral ? fast_pred(t) : K_coral;
    
    // EQUATION 23: Slow coral logistic growth
    // Slower growth rate but more resistant to stress
    Type slow_growth = r_slow * slow_prev * (Type(1.0) - (slow_prev + competition_coef * fast_prev) / (K_coral + eps));
    
    // EQUATION 24: Slow coral predation loss
    Type slow_loss = pred_rate_slow * cots_prev;
    
    // EQUATION 25: Slow coral temperature mortality (reduced compared to fast coral)
    Type slow_temp_mort = Type(0.5) * temp_stress * slow_prev;
    
    // EQUATION 26: Slow coral population change
    Type dslow = slow_growth - slow_loss - slow_temp_mort;
    slow_pred(t) = slow_prev + dslow;
    slow_pred(t) = slow_pred(t) > eps ? slow_pred(t) : eps;
    slow_pred(t) = slow_pred(t) < K_coral ? slow_pred(t) : K_coral;
  }
  
  // LIKELIHOOD CALCULATION
  // Use lognormal distribution for strictly positive data
  for(int t = 0; t < n; t++) {
    // COTS observations (lognormal)
    Type log_cots_pred = log(cots_pred(t) + eps);
    Type log_cots_obs = log(cots_dat(t) + eps);
    nll -= dnorm(log_cots_obs, log_cots_pred, sigma_cots, true);
    
    // Fast coral observations (lognormal)
    Type log_fast_pred = log(fast_pred(t) + eps);
    Type log_fast_obs = log(fast_dat(t) + eps);
    nll -= dnorm(log_fast_obs, log_fast_pred, sigma_fast, true);
    
    // Slow coral observations (lognormal)
    Type log_slow_pred = log(slow_pred(t) + eps);
    Type log_slow_obs = log(slow_dat(t) + eps);
    nll -= dnorm(log_slow_obs, log_slow_pred, sigma_slow, true);
  }
  
  // Soft penalties to keep parameters in biologically reasonable ranges
  // COTS parameters
  nll += Type(0.01) * pow(log_r_cots - log(Type(0.5)), 2);           // Penalize far from r=0.5
  nll += Type(0.01) * pow(log_K_cots - log(Type(5.0)), 2);           // Penalize far from K=5
  nll += Type(0.01) * pow(log_allee_threshold - log(Type(0.1)), 2);  // Penalize far from 0.1
  nll += Type(0.01) * pow(allee_strength - Type(0.5), 2);            // Penalize far from 0.5
  nll += Type(0.01) * pow(temp_effect_cots - Type(0.1), 2);          // Penalize far from 0.1
  
  // Settlement parameters
  nll += Type(0.01) * pow(gregarious_strength - Type(2.0), 2);       // Penalize far from 2.0
  nll += Type(0.01) * pow(log_gregarious_threshold - log(Type(0.05)), 2);  // Penalize far from 0.05
  nll += Type(0.01) * pow(temp_effect_settlement - Type(0.15), 2);   // Penalize far from 0.15
  
  // Coral parameters
  nll += Type(0.01) * pow(log_r_fast - log(Type(0.3)), 2);           // Penalize far from r=0.3
  nll += Type(0.01) * pow(log_r_slow - log(Type(0.1)), 2);           // Penalize far from r=0.1
  nll += Type(0.01) * pow(log_K_coral - log(Type(50.0)), 2);         // Penalize far from K=50%
  
  // REPORTING
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(r_cots);
  REPORT(K_cots);
  REPORT(allee_threshold);
  REPORT(allee_strength);
  REPORT(mort_cots);
  REPORT(dd_mort);
  REPORT(temp_effect_cots);
  REPORT(temp_opt);
  REPORT(gregarious_strength);
  REPORT(gregarious_threshold);
  REPORT(temp_effect_settlement);
  REPORT(attack_fast);
  REPORT(attack_slow);
  REPORT(handling_fast);
  REPORT(handling_slow);
  REPORT(pred_preference);
  REPORT(interference);
  REPORT(r_fast);
  REPORT(r_slow);
  REPORT(K_coral);
  REPORT(competition_coef);
  REPORT(temp_stress_coef);
  REPORT(temp_stress_threshold);
  REPORT(sigma_cots);
  REPORT(sigma_fast);
  REPORT(sigma_slow);
  
  return nll;
}
