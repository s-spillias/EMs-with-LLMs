#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA SECTION
  DATA_VECTOR(Year);                  // Years of observation
  DATA_VECTOR(cots_dat);              // Adult COTS abundance (individuals/m2)
  DATA_VECTOR(slow_dat);              // Slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);              // Fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);               // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);           // COTS immigration rate (individuals/m2/year)
  
  // PARAMETER SECTION
  PARAMETER(r_cots);                  // COTS intrinsic growth rate (year^-1)
  PARAMETER(K_cots);                  // COTS carrying capacity (individuals/m2)
  PARAMETER(m_cots);                  // COTS natural mortality rate (year^-1)
  
  PARAMETER(r_slow);                  // Intrinsic growth rate of slow-growing corals (year^-1)
  PARAMETER(r_fast);                  // Intrinsic growth rate of fast-growing corals (year^-1)
  PARAMETER(K_coral);                 // Combined carrying capacity for all corals (%)
  PARAMETER(comp_coef);               // Competition coefficient between coral types
  
  PARAMETER(a_slow);                  // Attack rate on slow-growing corals (m2/individual/year)
  PARAMETER(a_fast);                  // Attack rate on fast-growing corals (m2/individual/year)
  PARAMETER(h_slow);                  // Handling time for slow-growing corals (year/%)
  PARAMETER(h_fast);                  // Handling time for fast-growing corals (year/%)
  PARAMETER(q_cots);                  // Conversion efficiency of coral to COTS (individuals/%)
  
  PARAMETER(temp_opt);                // Optimal temperature for coral growth (°C)
  PARAMETER(temp_tol);                // Temperature tolerance range (°C)
  PARAMETER(temp_mort);               // Temperature mortality coefficient
  
  PARAMETER(log_sigma_cots);          // Log of observation error SD for COTS
  PARAMETER(log_sigma_slow);          // Log of observation error SD for slow-growing coral
  PARAMETER(log_sigma_fast);          // Log of observation error SD for fast-growing coral
  
  // Transform parameters to ensure they're positive
  Type r_cots_pos = exp(r_cots);
  Type K_cots_pos = exp(K_cots);
  Type m_cots_pos = exp(m_cots);
  
  Type r_slow_pos = exp(r_slow);
  Type r_fast_pos = exp(r_fast);
  Type K_coral_pos = exp(K_coral);
  Type comp_coef_pos = exp(comp_coef);
  
  Type a_slow_pos = exp(a_slow);
  Type a_fast_pos = exp(a_fast);
  Type h_slow_pos = exp(h_slow);
  Type h_fast_pos = exp(h_fast);
  Type q_cots_pos = exp(q_cots);
  
  Type temp_tol_pos = exp(temp_tol);
  Type temp_mort_pos = exp(temp_mort);
  
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_slow = exp(log_sigma_slow);
  Type sigma_fast = exp(log_sigma_fast);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize vectors for predictions
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  
  // Set initial values for first time step
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-6);
  
  // Process model: predict state variables through time
  for(int t = 1; t < n; t++) {
    // 1. Temperature effect on coral growth (Gaussian response)
    Type temp_diff = sst_dat(t-1) - temp_opt;
    Type temp_effect = exp(-0.5 * pow(temp_diff / (temp_tol_pos + eps), 2));
    
    // 2. COTS predation on corals (simplified functional response)
    Type pred_slow = a_slow_pos * cots_pred(t-1) * slow_pred(t-1) / (1.0 + eps);
    Type pred_fast = a_fast_pos * cots_pred(t-1) * fast_pred(t-1) / (1.0 + eps);
    
    // Limit predation to available coral using smooth transitions
    pred_slow = pred_slow / (1.0 + exp(5.0 * (pred_slow - 0.5 * slow_pred(t-1)))) * slow_pred(t-1);
    pred_fast = pred_fast / (1.0 + exp(5.0 * (pred_fast - 0.5 * fast_pred(t-1)))) * fast_pred(t-1);
    
    // 3. Coral growth with competition
    Type slow_growth = r_slow_pos * slow_pred(t-1) * (1.0 - (slow_pred(t-1) + comp_coef_pos * fast_pred(t-1)) / (K_coral_pos + eps)) * temp_effect;
    Type fast_growth = r_fast_pos * fast_pred(t-1) * (1.0 - (fast_pred(t-1) + comp_coef_pos * slow_pred(t-1)) / (K_coral_pos + eps)) * temp_effect;
    
    // Ensure growth is not negative using smooth function
    slow_growth = slow_growth / (1.0 + exp(-10.0 * slow_growth)) * slow_growth;
    fast_growth = fast_growth / (1.0 + exp(-10.0 * fast_growth)) * fast_growth;
    
    // 4. Temperature mortality effect (only above optimal temperature)
    Type temp_above = CppAD::CondExpGt(sst_dat(t-1), temp_opt, sst_dat(t-1) - temp_opt, Type(0));
    Type slow_mortality = temp_mort_pos * pow(temp_above, 2) * slow_pred(t-1);
    Type fast_mortality = temp_mort_pos * pow(temp_above, 2) * fast_pred(t-1) * 1.5; // Fast corals more sensitive
    
    // Limit mortality to available coral using smooth transitions
    slow_mortality = slow_mortality / (1.0 + exp(5.0 * (slow_mortality - 0.5 * slow_pred(t-1)))) * slow_pred(t-1);
    fast_mortality = fast_mortality / (1.0 + exp(5.0 * (fast_mortality - 0.5 * fast_pred(t-1)))) * fast_pred(t-1);
    
    // 5. Update coral cover
    slow_pred(t) = slow_pred(t-1) + slow_growth - pred_slow - slow_mortality;
    fast_pred(t) = fast_pred(t-1) + fast_growth - pred_fast - fast_mortality;
    
    // Ensure coral cover doesn't go below minimum using smooth function
    slow_pred(t) = 0.01 + (slow_pred(t) - 0.01) / (1.0 + exp(-10.0 * (slow_pred(t) - 0.01)));
    fast_pred(t) = 0.01 + (fast_pred(t) - 0.01) / (1.0 + exp(-10.0 * (fast_pred(t) - 0.01)));
    
    // 6. Update COTS population
    // Food limitation based on available coral
    Type food_limitation = (slow_pred(t-1) + fast_pred(t-1)) / (K_coral_pos + eps);
    food_limitation = food_limitation / (1.0 + exp(10.0 * (food_limitation - 1.0))); // Smooth cap at 1.0
    
    // Logistic growth with food limitation
    Type cots_growth = r_cots_pos * cots_pred(t-1) * (1.0 - cots_pred(t-1) / (K_cots_pos + eps)) * food_limitation;
    
    // Conversion of consumed coral to COTS biomass
    Type cots_conversion = q_cots_pos * (pred_slow + pred_fast);
    
    // Update COTS population
    cots_pred(t) = cots_pred(t-1) + cots_growth + cots_conversion - m_cots_pos * cots_pred(t-1) + cotsimm_dat(t-1);
    
    // Ensure COTS abundance doesn't go below minimum using smooth function
    cots_pred(t) = 0.01 + (cots_pred(t) - 0.01) / (1.0 + exp(-10.0 * (cots_pred(t) - 0.01)));
  }
  
  // Observation model: calculate negative log-likelihood
  for(int t = 0; t < n; t++) {
    // Minimum standard deviations to prevent numerical issues
    Type min_sd_cots = 0.1;
    Type min_sd_coral = 1.0;
    
    // Effective standard deviations
    Type sd_cots_eff = sqrt(pow(sigma_cots, 2) + pow(min_sd_cots, 2));
    Type sd_slow_eff = sqrt(pow(sigma_slow, 2) + pow(min_sd_coral, 2));
    Type sd_fast_eff = sqrt(pow(sigma_fast, 2) + pow(min_sd_coral, 2));
    
    // COTS abundance (lognormal error)
    Type cots_dat_pos = CppAD::CondExpLt(cots_dat(t), Type(0.01), Type(0.01), cots_dat(t));
    
    nll -= dnorm(log(cots_dat_pos), log(cots_pred(t)), sd_cots_eff, true);
    
    // Slow-growing coral cover (normal error)
    nll -= dnorm(slow_dat(t), slow_pred(t), sd_slow_eff, true);
    
    // Fast-growing coral cover (normal error)
    nll -= dnorm(fast_dat(t), fast_pred(t), sd_fast_eff, true);
  }
  
  // Report predictions and parameters
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(r_cots_pos);
  REPORT(K_cots_pos);
  REPORT(m_cots_pos);
  REPORT(r_slow_pos);
  REPORT(r_fast_pos);
  REPORT(K_coral_pos);
  REPORT(comp_coef_pos);
  REPORT(a_slow_pos);
  REPORT(a_fast_pos);
  REPORT(h_slow_pos);
  REPORT(h_fast_pos);
  REPORT(q_cots_pos);
  REPORT(temp_tol_pos);
  REPORT(temp_mort_pos);
  ADREPORT(sigma_cots);
  ADREPORT(sigma_slow);
  ADREPORT(sigma_fast);
  
  return nll;
}
