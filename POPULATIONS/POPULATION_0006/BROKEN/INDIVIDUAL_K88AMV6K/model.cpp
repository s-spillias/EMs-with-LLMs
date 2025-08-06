#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA SECTION
  DATA_VECTOR(Year);                  // Vector of years for time series data
  DATA_VECTOR(sst_dat);               // Sea surface temperature data (Celsius)
  DATA_VECTOR(cotsimm_dat);           // COTS larval immigration rate (individuals/m^2/year)
  DATA_VECTOR(cots_dat);              // Observed COTS abundance (individuals/m^2)
  DATA_VECTOR(fast_dat);              // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);              // Observed slow-growing coral cover (%)
  
  // PARAMETER SECTION
  PARAMETER(r_cots);                  // Intrinsic growth rate of COTS (year^-1)
  PARAMETER(K_cots);                  // Carrying capacity of COTS (individuals/m^2)
  PARAMETER(m_cots);                  // Natural mortality rate of COTS (year^-1)
  PARAMETER(alpha_fast);              // Attack rate on fast-growing coral (m^2/individual/year)
  PARAMETER(alpha_slow);              // Attack rate on slow-growing coral (m^2/individual/year)
  PARAMETER(h_fast);                  // Handling time for fast-growing coral (year/% cover)
  PARAMETER(h_slow);                  // Handling time for slow-growing coral (year/% cover)
  PARAMETER(r_fast);                  // Intrinsic growth rate of fast-growing coral (year^-1)
  PARAMETER(r_slow);                  // Intrinsic growth rate of slow-growing coral (year^-1)
  PARAMETER(K_fast);                  // Carrying capacity of fast-growing coral (% cover)
  PARAMETER(K_slow);                  // Carrying capacity of slow-growing coral (% cover)
  PARAMETER(beta_sst);                // Effect of SST on COTS reproduction (dimensionless)
  PARAMETER(sst_opt);                 // Optimal SST for COTS reproduction (Celsius)
  PARAMETER(sst_width);               // Width parameter for temperature response (Celsius)
  PARAMETER(imm_effect);              // Effect of larval immigration (dimensionless)
  PARAMETER(coral_threshold);         // Coral threshold for COTS mortality (% cover)
  PARAMETER(sigma_cots);              // Observation error SD for COTS (log scale)
  PARAMETER(sigma_fast);              // Observation error SD for fast coral (log scale)
  PARAMETER(sigma_slow);              // Observation error SD for slow coral (log scale)
  PARAMETER(allee_threshold);         // Population threshold for Allee effect
  PARAMETER(outbreak_threshold);      // Population threshold for outbreak dynamics
  PARAMETER(allee_strength);          // Strength of Allee effect
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize vectors for model predictions
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Set initial values (first year)
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Run the model for each time step
  for (int t = 1; t < n; t++) {
    // 1. Calculate temperature effect on COTS reproduction using a Gaussian response curve
    Type temp_effect = exp(-pow(sst_dat(t-1) - sst_opt, 2) / (2 * pow(sst_width + 0.1, 2)));
    
    // 2. Calculate total coral cover (fast + slow) for density dependence
    Type total_coral = fast_pred(t-1) + slow_pred(t-1) + eps;
    
    // 3. Calculate food-dependent mortality modifier (increases when coral is scarce)
    Type mort_modifier = 1.0 + 1.0 / (1.0 + exp((total_coral - (coral_threshold + 0.1)) / 0.5));
    
    // 4. Calculate functional responses for COTS feeding on corals (Type II)
    Type denom = 1.0 + alpha_fast * h_fast * fast_pred(t-1) + alpha_slow * h_slow * slow_pred(t-1) + eps;
    Type consumption_fast = (alpha_fast * fast_pred(t-1) * cots_pred(t-1)) / denom;
    Type consumption_slow = (alpha_slow * slow_pred(t-1) * cots_pred(t-1)) / denom;
    
    // 5. Calculate density-dependent reproductive success with Allee effect
    // Using a simplified smooth function for Allee effect
    Type allee_effect = pow(cots_pred(t-1), 2) / (pow(allee_threshold + 0.1, 2) + pow(cots_pred(t-1), 2));
    
    // Smooth function for outbreak effect
    Type outbreak_effect = 1.0 + 0.5 * (total_coral / (K_fast + K_slow + eps)) * 
                          (1.0 / (1.0 + exp(-(cots_pred(t-1) - outbreak_threshold) * 2.0)));
    
    // Combined reproductive modifier
    Type repro_modifier = allee_effect * outbreak_effect;
    
    // 6. Calculate COTS population dynamics
    // Use CppAD::CondExpGt for conditional expressions instead of fmax
    Type r_cots_safe = CppAD::CondExpGt(r_cots, Type(0.0), r_cots, Type(0.0));
    Type K_cots_safe = CppAD::CondExpGt(K_cots, Type(0.01), K_cots, Type(0.01));
    
    Type cots_growth = r_cots_safe * temp_effect * repro_modifier * cots_pred(t-1) * 
                      (1.0 - cots_pred(t-1) / (K_cots_safe * (total_coral / (K_fast + K_slow + eps)) + eps));
    
    Type m_cots_safe = CppAD::CondExpGt(m_cots, Type(0.0), m_cots, Type(0.0));
    Type cots_mortality = m_cots_safe * mort_modifier * cots_pred(t-1);
    
    Type imm_effect_safe = CppAD::CondExpGt(imm_effect, Type(0.0), imm_effect, Type(0.0));
    Type cots_immigration = imm_effect_safe * cotsimm_dat(t-1);
    
    // 7. Update COTS abundance
    cots_pred(t) = cots_pred(t-1) + cots_growth - cots_mortality + cots_immigration;
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), eps, cots_pred(t), eps);
    
    // 8. Calculate coral dynamics with logistic growth and COTS predation
    Type r_fast_safe = CppAD::CondExpGt(r_fast, Type(0.0), r_fast, Type(0.0));
    Type r_slow_safe = CppAD::CondExpGt(r_slow, Type(0.0), r_slow, Type(0.0));
    Type K_fast_safe = CppAD::CondExpGt(K_fast, Type(1.0), K_fast, Type(1.0));
    Type K_slow_safe = CppAD::CondExpGt(K_slow, Type(1.0), K_slow, Type(1.0));
    
    Type fast_growth = r_fast_safe * fast_pred(t-1) * 
                      (1.0 - (fast_pred(t-1) + 0.5 * slow_pred(t-1)) / K_fast_safe);
    
    Type slow_growth = r_slow_safe * slow_pred(t-1) * 
                      (1.0 - (slow_pred(t-1) + 0.3 * fast_pred(t-1)) / K_slow_safe);
    
    // 9. Update coral cover
    fast_pred(t) = fast_pred(t-1) + fast_growth - consumption_fast;
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), eps, fast_pred(t), eps);
    
    slow_pred(t) = slow_pred(t-1) + slow_growth - consumption_slow;
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), eps, slow_pred(t), eps);
  }
  
  // Calculate negative log-likelihood using lognormal distribution
  // Add a small constant to observations and predictions to handle zeros
  Type const_obs = Type(1e-4);
  
  for (int t = 0; t < n; t++) {
    // 11. COTS abundance likelihood
    Type cots_obs = cots_dat(t) + const_obs;
    Type cots_model = cots_pred(t) + const_obs;
    nll -= dnorm(log(cots_obs), log(cots_model), sigma_cots, true);
    
    // 12. Fast-growing coral cover likelihood
    Type fast_obs = fast_dat(t) + const_obs;
    Type fast_model = fast_pred(t) + const_obs;
    nll -= dnorm(log(fast_obs), log(fast_model), sigma_fast, true);
    
    // 13. Slow-growing coral cover likelihood
    Type slow_obs = slow_dat(t) + const_obs;
    Type slow_model = slow_pred(t) + const_obs;
    nll -= dnorm(log(slow_obs), log(slow_model), sigma_slow, true);
  }
  
  // Add smooth penalties to constrain parameters within biologically meaningful ranges
  // 14. Penalty to keep r_cots positive but not too large
  nll += 0.01 * pow(r_cots - 1.0, 2) * (r_cots > 1.0);
  
  // 15. Penalty to keep attack rates in reasonable range
  nll += 0.01 * pow(alpha_fast - 0.5, 2) * (alpha_fast > 0.5);
  nll += 0.01 * pow(alpha_slow - 0.5, 2) * (alpha_slow > 0.5);
  
  // 16. Penalty to ensure allee_threshold < outbreak_threshold < K_cots
  nll += 0.1 * pow(allee_threshold - outbreak_threshold, 2) * (allee_threshold > outbreak_threshold);
  nll += 0.1 * pow(outbreak_threshold - K_cots, 2) * (outbreak_threshold > K_cots);
  
  // REPORT SECTION
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  ADREPORT(r_cots);
  ADREPORT(K_cots);
  ADREPORT(alpha_fast);
  ADREPORT(alpha_slow);
  ADREPORT(r_fast);
  ADREPORT(r_slow);
  ADREPORT(beta_sst);
  ADREPORT(imm_effect);
  ADREPORT(allee_threshold);
  ADREPORT(outbreak_threshold);
  ADREPORT(allee_strength);
  
  return nll;
}
