#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA SECTION
  DATA_VECTOR(Year);                  // Years of observation
  DATA_VECTOR(cots_dat);              // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);              // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);              // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);               // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);           // COTS larval immigration rate (individuals/m2/year)
  
  // PARAMETER SECTION
  PARAMETER(log_r_cots);              // Log of COTS intrinsic growth rate (year^-1)
  PARAMETER(log_K_cots);              // Log of COTS carrying capacity (individuals/m2)
  PARAMETER(log_m_cots);              // Log of COTS natural mortality rate (year^-1)
  
  PARAMETER(log_r_fast);              // Log of fast-growing coral intrinsic growth rate (year^-1)
  PARAMETER(log_K_fast);              // Log of fast-growing coral carrying capacity (%)
  
  PARAMETER(log_r_slow);              // Log of slow-growing coral intrinsic growth rate (year^-1)
  PARAMETER(log_K_slow);              // Log of slow-growing coral carrying capacity (%)
  
  PARAMETER(log_a_fast);              // Log of COTS attack rate on fast-growing coral (m2/individual/year)
  PARAMETER(log_a_slow);              // Log of COTS attack rate on slow-growing coral (m2/individual/year)
  PARAMETER(log_h_fast);              // Log of handling time for fast-growing coral (year/%)
  PARAMETER(log_h_slow);              // Log of handling time for slow-growing coral (year/%)
  PARAMETER(log_pref);                // Log of preference factor for fast-growing coral (dimensionless)
  
  PARAMETER(log_temp_opt);            // Log of optimal temperature for COTS reproduction (°C)
  PARAMETER(log_temp_width);          // Log of temperature response width (°C)
  
  PARAMETER(log_coral_threshold);     // Log of coral threshold for COTS reproduction (%)
  PARAMETER(log_coral_steepness);     // Log of steepness of coral limitation function (dimensionless)
  
  PARAMETER(log_obs_sd_cots);         // Log of observation error SD for COTS
  PARAMETER(log_obs_sd_fast);         // Log of observation error SD for fast-growing coral
  PARAMETER(log_obs_sd_slow);         // Log of observation error SD for slow-growing coral
  
  PARAMETER(log_proc_sd_cots);        // Log of process error SD for COTS
  PARAMETER(log_proc_sd_fast);        // Log of process error SD for fast-growing coral
  PARAMETER(log_proc_sd_slow);        // Log of process error SD for slow-growing coral
  
  // Transform parameters to natural scale
  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type m_cots = exp(log_m_cots);
  
  Type r_fast = exp(log_r_fast);
  Type K_fast = exp(log_K_fast);
  
  Type r_slow = exp(log_r_slow);
  Type K_slow = exp(log_K_slow);
  
  Type a_fast = exp(log_a_fast);
  Type a_slow = exp(log_a_slow);
  Type h_fast = exp(log_h_fast);
  Type h_slow = exp(log_h_slow);
  Type pref = exp(log_pref);
  
  Type temp_opt = exp(log_temp_opt);
  Type temp_width = exp(log_temp_width);
  
  Type coral_threshold = exp(log_coral_threshold);
  Type coral_steepness = exp(log_coral_steepness);
  
  // Set minimum standard deviations
  Type min_sd = Type(0.1);
  Type obs_sd_cots = exp(log_obs_sd_cots) < min_sd ? min_sd : exp(log_obs_sd_cots);
  Type obs_sd_fast = exp(log_obs_sd_fast) < min_sd ? min_sd : exp(log_obs_sd_fast);
  Type obs_sd_slow = exp(log_obs_sd_slow) < min_sd ? min_sd : exp(log_obs_sd_slow);
  Type proc_sd_cots = exp(log_proc_sd_cots) < min_sd ? min_sd : exp(log_proc_sd_cots);
  Type proc_sd_fast = exp(log_proc_sd_fast) < min_sd ? min_sd : exp(log_proc_sd_fast);
  Type proc_sd_slow = exp(log_proc_sd_slow) < min_sd ? min_sd : exp(log_proc_sd_slow);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Vectors to store predictions
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Small constant to prevent division by zero or log(0)
  Type eps = Type(0.01);
  
  // Initialize with first observation
  cots_pred(0) = cots_dat(0) + eps;
  fast_pred(0) = fast_dat(0) + eps;
  slow_pred(0) = slow_dat(0) + eps;
  
  // Simple process model without process error
  for(int t = 1; t < n; t++) {
    // Simple temperature effect (bounded)
    Type temp_effect = Type(1.0) - pow((sst_dat(t-1) - temp_opt) / (temp_width + eps), 2);
    temp_effect = temp_effect < Type(0.1) ? Type(0.1) : temp_effect;
    temp_effect = temp_effect > Type(1.0) ? Type(1.0) : temp_effect;
    
    // Simple coral effect (bounded)
    Type total_coral = fast_pred(t-1) + slow_pred(t-1);
    Type coral_effect = Type(1.0) / (Type(1.0) + exp(-coral_steepness * (total_coral - coral_threshold)));
    coral_effect = coral_effect < Type(0.1) ? Type(0.1) : coral_effect;
    coral_effect = coral_effect > Type(1.0) ? Type(1.0) : coral_effect;
    
    // Simple predation rates
    Type pred_fast = a_fast * cots_pred(t-1) * fast_pred(t-1);
    Type pred_slow = a_slow * cots_pred(t-1) * slow_pred(t-1);
    
    // Limit predation to available coral
    pred_fast = pred_fast > (Type(0.5) * fast_pred(t-1)) ? (Type(0.5) * fast_pred(t-1)) : pred_fast;
    pred_slow = pred_slow > (Type(0.5) * slow_pred(t-1)) ? (Type(0.5) * slow_pred(t-1)) : pred_slow;
    
    // Update COTS population
    Type cots_growth = r_cots * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1) / K_cots) * temp_effect * coral_effect;
    Type cots_mortality = m_cots * cots_pred(t-1);
    cots_pred(t) = cots_pred(t-1) + cots_growth - cots_mortality + cotsimm_dat(t-1);
    cots_pred(t) = cots_pred(t) < eps ? eps : cots_pred(t);
    
    // Update fast-growing coral
    Type fast_growth = r_fast * fast_pred(t-1) * (Type(1.0) - fast_pred(t-1) / K_fast);
    fast_pred(t) = fast_pred(t-1) + fast_growth - pred_fast;
    fast_pred(t) = fast_pred(t) < eps ? eps : fast_pred(t);
    
    // Update slow-growing coral
    Type slow_growth = r_slow * slow_pred(t-1) * (Type(1.0) - slow_pred(t-1) / K_slow);
    slow_pred(t) = slow_pred(t-1) + slow_growth - pred_slow;
    slow_pred(t) = slow_pred(t) < eps ? eps : slow_pred(t);
  }
  
  // Simple observation model
  for(int t = 0; t < n; t++) {
    // Add small constant to data and predictions to avoid log(0)
    Type cots_obs = cots_dat(t) + eps;
    Type fast_obs = fast_dat(t) + eps;
    Type slow_obs = slow_dat(t) + eps;
    
    // Use normal distribution on log scale
    nll -= dnorm(log(cots_obs), log(cots_pred(t)), obs_sd_cots, true);
    nll -= dnorm(log(fast_obs), log(fast_pred(t)), obs_sd_fast, true);
    nll -= dnorm(log(slow_obs), log(slow_pred(t)), obs_sd_slow, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  // Report transformed parameters
  REPORT(r_cots);
  REPORT(K_cots);
  REPORT(m_cots);
  REPORT(r_fast);
  REPORT(K_fast);
  REPORT(r_slow);
  REPORT(K_slow);
  REPORT(a_fast);
  REPORT(a_slow);
  REPORT(h_fast);
  REPORT(h_slow);
  REPORT(pref);
  REPORT(temp_opt);
  REPORT(temp_width);
  REPORT(coral_threshold);
  REPORT(coral_steepness);
  REPORT(obs_sd_cots);
  REPORT(obs_sd_fast);
  REPORT(obs_sd_slow);
  REPORT(proc_sd_cots);
  REPORT(proc_sd_fast);
  REPORT(proc_sd_slow);
  
  return nll;
}
