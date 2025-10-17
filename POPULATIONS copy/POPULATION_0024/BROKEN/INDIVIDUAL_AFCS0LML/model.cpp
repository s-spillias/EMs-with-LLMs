#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs
  DATA_VECTOR(Year);                    // Time vector (years)
  DATA_VECTOR(cots_dat);                // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);                // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);                // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);                 // Sea surface temperature forcing (Celsius)
  DATA_VECTOR(cotsimm_dat);             // COTS larval immigration forcing (individuals/m2/year)
  
  // COTS population parameters
  PARAMETER(log_r_cots);                // Log intrinsic recruitment rate of COTS (year^-1)
  PARAMETER(log_K_cots);                // Log carrying capacity for COTS (individuals/m2)
  PARAMETER(log_allee_threshold);       // Log Allee threshold density (individuals/m2)
  PARAMETER(allee_strength);            // Allee effect strength (dimensionless, 0-1)
  PARAMETER(log_mort_cots);             // Log baseline natural mortality rate (year^-1)
  PARAMETER(log_dd_mort);               // Log density-dependent mortality coefficient (m2/individuals/year)
  
  // Temperature effects on COTS
  PARAMETER(temp_opt);                  // Optimal temperature for COTS recruitment (Celsius)
  PARAMETER(log_temp_width);            // Log temperature tolerance width (Celsius)
  PARAMETER(temp_effect_strength);      // Strength of temperature effect on recruitment (dimensionless)
  
  // Coral growth parameters
  PARAMETER(log_r_fast);                // Log intrinsic growth rate of fast coral (year^-1)
  PARAMETER(log_r_slow);                // Log intrinsic growth rate of slow coral (year^-1)
  PARAMETER(log_K_coral);               // Log total coral carrying capacity (% cover)
  PARAMETER(competition_fast);          // Competition coefficient of fast coral on slow (dimensionless)
  PARAMETER(competition_slow);          // Competition coefficient of slow coral on fast (dimensionless)
  
  // COTS feeding parameters
  PARAMETER(log_attack_fast);           // Log attack rate on fast coral (m2/individuals/year)
  PARAMETER(log_attack_slow);           // Log attack rate on slow coral (m2/individuals/year)
  PARAMETER(log_handling_fast);         // Log handling time for fast coral (year)
  PARAMETER(log_handling_slow);         // Log handling time for slow coral (year)
  PARAMETER(log_preference_ratio);      // Log preference ratio (fast/slow when equal abundance)
  
  // Coral bleaching parameters
  PARAMETER(bleach_threshold);          // Temperature threshold for bleaching (Celsius)
  PARAMETER(log_bleach_mort);           // Log bleaching mortality rate above threshold (year^-1)
  
  // Observation error parameters
  PARAMETER(log_sigma_cots);            // Log observation error SD for COTS
  PARAMETER(log_sigma_fast);            // Log observation error SD for fast coral
  PARAMETER(log_sigma_slow);            // Log observation error SD for slow coral
  
  // Transform parameters to natural scale
  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type allee_threshold = exp(log_allee_threshold);
  Type mort_cots = exp(log_mort_cots);
  Type dd_mort = exp(log_dd_mort);
  Type temp_width = exp(log_temp_width);
  Type r_fast = exp(log_r_fast);
  Type r_slow = exp(log_r_slow);
  Type K_coral = exp(log_K_coral);
  Type attack_fast = exp(log_attack_fast);
  Type attack_slow = exp(log_attack_slow);
  Type handling_fast = exp(log_handling_fast);
  Type handling_slow = exp(log_handling_slow);
  Type preference_ratio = exp(log_preference_ratio);
  Type bleach_mort = exp(log_bleach_mort);
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_fast = exp(log_sigma_fast);
  Type sigma_slow = exp(log_sigma_slow);
  
  // Minimum standard deviations for numerical stability
  Type min_sigma = Type(0.01);
  sigma_cots = sigma_cots + min_sigma;
  sigma_fast = sigma_fast + min_sigma;
  sigma_slow = sigma_slow + min_sigma;
  
  // Small constant for numerical stability
  Type eps = Type(1e-8);
  
  // Get number of time steps
  int n = Year.size();
  
  // Initialize prediction vectors
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Set initial conditions from first observation
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Initialize negative log-likelihood
  Type nll = Type(0.0);
  
  // Time loop - start from index 1 since initial conditions set at index 0
  for(int t = 1; t < n; t++) {
    
    // Get previous time step values (avoid data leakage)
    Type cots_prev = cots_pred(t-1);
    Type fast_prev = fast_pred(t-1);
    Type slow_prev = slow_pred(t-1);
    Type sst = sst_dat(t);
    Type immigration = cotsimm_dat(t);
    
    // Ensure non-negative values
    cots_prev = cots_prev + eps;
    fast_prev = fast_prev + eps;
    slow_prev = slow_prev + eps;
    
    // === EQUATION 1: Temperature effect on COTS recruitment ===
    // Gaussian function centered at optimal temperature
    Type temp_deviation = sst - temp_opt;
    Type temp_effect = exp(-0.5 * pow(temp_deviation / (temp_width + eps), 2));
    Type temp_modifier = Type(1.0) + temp_effect_strength * (temp_effect - Type(1.0));
    temp_modifier = temp_modifier + eps;
    
    // === EQUATION 2: Allee effect on COTS recruitment ===
    // Sigmoid function that reduces recruitment at low densities
    Type allee_ratio = cots_prev / (allee_threshold + eps);
    Type allee_effect = pow(allee_ratio, allee_strength) / (Type(1.0) + pow(allee_ratio, allee_strength));
    
    // === EQUATION 3: COTS recruitment rate ===
    // Combines intrinsic rate, temperature effect, Allee effect, and density dependence
    Type recruitment = r_cots * temp_modifier * allee_effect * (Type(1.0) - cots_prev / (K_cots + eps));
    recruitment = recruitment + eps;
    
    // === EQUATION 4: COTS mortality ===
    // Baseline mortality plus density-dependent component
    Type mortality = mort_cots + dd_mort * cots_prev;
    
    // === EQUATION 5: Total coral cover ===
    // Sum of both coral types for space competition calculations
    Type total_coral = fast_prev + slow_prev + eps;
    
    // === EQUATION 6: Type II functional response for fast coral consumption ===
    // Accounts for handling time and preference
    Type effective_attack_fast = attack_fast * preference_ratio;
    Type consumption_fast = (effective_attack_fast * fast_prev * cots_prev) / 
                           (Type(1.0) + effective_attack_fast * handling_fast * fast_prev + 
                            attack_slow * handling_slow * slow_prev + eps);
    
    // === EQUATION 7: Type II functional response for slow coral consumption ===
    // Accounts for handling time and lower preference
    Type consumption_slow = (attack_slow * slow_prev * cots_prev) / 
                           (Type(1.0) + effective_attack_fast * handling_fast * fast_prev + 
                            attack_slow * handling_slow * slow_prev + eps);
    
    // === EQUATION 8: Bleaching mortality ===
    // Activated when temperature exceeds threshold
    Type bleach_stress = sst - bleach_threshold;
    Type bleach_effect = Type(0.0);
    if(bleach_stress > Type(0.0)) {
      bleach_effect = bleach_mort * bleach_stress;
    }
    
    // === EQUATION 9: Fast coral growth ===
    // Logistic growth with competition and COTS predation
    Type space_available_fast = Type(1.0) - (fast_prev + competition_slow * slow_prev) / (K_coral + eps);
    Type fast_growth = r_fast * fast_prev * space_available_fast;
    Type fast_loss = consumption_fast + bleach_effect * fast_prev;
    
    // === EQUATION 10: Slow coral growth ===
    // Logistic growth with competition and COTS predation
    Type space_available_slow = Type(1.0) - (slow_prev + competition_fast * fast_prev) / (K_coral + eps);
    Type slow_growth = r_slow * slow_prev * space_available_slow;
    Type slow_loss = consumption_slow + bleach_effect * slow_prev;
    
    // === EQUATION 11: COTS population update ===
    // Net change from recruitment, mortality, and immigration
    cots_pred(t) = cots_prev + recruitment * cots_prev - mortality * cots_prev + immigration;
    cots_pred(t) = cots_pred(t) + eps;
    
    // === EQUATION 12: Fast coral update ===
    // Net change from growth and losses
    fast_pred(t) = fast_prev + fast_growth - fast_loss;
    fast_pred(t) = fast_pred(t) + eps;
    
    // === EQUATION 13: Slow coral update ===
    // Net change from growth and losses
    slow_pred(t) = slow_prev + slow_growth - slow_loss;
    slow_pred(t) = slow_pred(t) + eps;
    
    // === LIKELIHOOD CONTRIBUTIONS ===
    // Lognormal observation error for all state variables
    
    // COTS likelihood
    Type log_cots_pred = log(cots_pred(t) + eps);
    Type log_cots_obs = log(cots_dat(t) + eps);
    nll -= dnorm(log_cots_obs, log_cots_pred, sigma_cots, true);
    
    // Fast coral likelihood
    Type log_fast_pred = log(fast_pred(t) + eps);
    Type log_fast_obs = log(fast_dat(t) + eps);
    nll -= dnorm(log_fast_obs, log_fast_pred, sigma_fast, true);
    
    // Slow coral likelihood
    Type log_slow_pred = log(slow_pred(t) + eps);
    Type log_slow_obs = log(slow_dat(t) + eps);
    nll -= dnorm(log_slow_obs, log_slow_pred, sigma_slow, true);
  }
  
  // Add likelihood for initial conditions
  Type log_cots_pred0 = log(cots_pred(0) + eps);
  Type log_cots_obs0 = log(cots_dat(0) + eps);
  nll -= dnorm(log_cots_obs0, log_cots_pred0, sigma_cots, true);
  
  Type log_fast_pred0 = log(fast_pred(0) + eps);
  Type log_fast_obs0 = log(fast_dat(0) + eps);
  nll -= dnorm(log_fast_obs0, log_fast_pred0, sigma_fast, true);
  
  Type log_slow_pred0 = log(slow_pred(0) + eps);
  Type log_slow_obs0 = log(slow_dat(0) + eps);
  nll -= dnorm(log_slow_obs0, log_slow_pred0, sigma_slow, true);
  
  // Soft parameter bounds using penalties
  // COTS parameters
  if(r_cots < Type(0.0)) nll += Type(100.0) * pow(r_cots, 2);
  if(r_cots > Type(5.0)) nll += Type(100.0) * pow(r_cots - Type(5.0), 2);
  if(K_cots < Type(0.1)) nll += Type(100.0) * pow(K_cots - Type(0.1), 2);
  if(K_cots > Type(50.0)) nll += Type(100.0) * pow(K_cots - Type(50.0), 2);
  if(allee_threshold < Type(0.01)) nll += Type(100.0) * pow(allee_threshold - Type(0.01), 2);
  if(allee_strength < Type(0.0)) nll += Type(100.0) * pow(allee_strength, 2);
  if(allee_strength > Type(5.0)) nll += Type(100.0) * pow(allee_strength - Type(5.0), 2);
  if(mort_cots < Type(0.0)) nll += Type(100.0) * pow(mort_cots, 2);
  if(mort_cots > Type(2.0)) nll += Type(100.0) * pow(mort_cots - Type(2.0), 2);
  
  // Temperature parameters
  if(temp_opt < Type(20.0)) nll += Type(100.0) * pow(temp_opt - Type(20.0), 2);
  if(temp_opt > Type(32.0)) nll += Type(100.0) * pow(temp_opt - Type(32.0), 2);
  if(temp_width < Type(0.5)) nll += Type(100.0) * pow(temp_width - Type(0.5), 2);
  if(temp_width > Type(10.0)) nll += Type(100.0) * pow(temp_width - Type(10.0), 2);
  
  // Coral parameters
  if(r_fast < Type(0.0)) nll += Type(100.0) * pow(r_fast, 2);
  if(r_fast > Type(1.0)) nll += Type(100.0) * pow(r_fast - Type(1.0), 2);
  if(r_slow < Type(0.0)) nll += Type(100.0) * pow(r_slow, 2);
  if(r_slow > Type(0.5)) nll += Type(100.0) * pow(r_slow - Type(0.5), 2);
  if(K_coral < Type(50.0)) nll += Type(100.0) * pow(K_coral - Type(50.0), 2);
  if(K_coral > Type(100.0)) nll += Type(100.0) * pow(K_coral - Type(100.0), 2);
  
  // Feeding parameters
  if(attack_fast < Type(0.0)) nll += Type(100.0) * pow(attack_fast, 2);
  if(attack_slow < Type(0.0)) nll += Type(100.0) * pow(attack_slow, 2);
  if(handling_fast < Type(0.0)) nll += Type(100.0) * pow(handling_fast, 2);
  if(handling_slow < Type(0.0)) nll += Type(100.0) * pow(handling_slow, 2);
  if(preference_ratio < Type(0.1)) nll += Type(100.0) * pow(preference_ratio - Type(0.1), 2);
  if(preference_ratio > Type(10.0)) nll += Type(100.0) * pow(preference_ratio - Type(10.0), 2);
  
  // Bleaching parameters
  if(bleach_threshold < Type(28.0)) nll += Type(100.0) * pow(bleach_threshold - Type(28.0), 2);
  if(bleach_threshold > Type(32.0)) nll += Type(100.0) * pow(bleach_threshold - Type(32.0), 2);
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  // Report transformed parameters
  REPORT(r_cots);
  REPORT(K_cots);
  REPORT(allee_threshold);
  REPORT(mort_cots);
  REPORT(dd_mort);
  REPORT(temp_width);
  REPORT(r_fast);
  REPORT(r_slow);
  REPORT(K_coral);
  REPORT(attack_fast);
  REPORT(attack_slow);
  REPORT(handling_fast);
  REPORT(handling_slow);
  REPORT(preference_ratio);
  REPORT(bleach_mort);
  REPORT(sigma_cots);
  REPORT(sigma_fast);
  REPORT(sigma_slow);
  
  return nll;
}
