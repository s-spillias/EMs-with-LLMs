#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs
  DATA_VECTOR(Year);           // Time vector (years)
  DATA_VECTOR(cots_dat);       // Observed COTS density (individuals/m²)
  DATA_VECTOR(slow_dat);       // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);       // Observed fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);        // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);    // COTS immigration rate (individuals/m²/year)
  
  // Parameters
  PARAMETER(log_r_slow);       // Log of slow coral intrinsic growth rate
  PARAMETER(log_r_fast);       // Log of fast coral intrinsic growth rate
  PARAMETER(log_K_slow);       // Log of slow coral carrying capacity
  PARAMETER(log_K_fast);       // Log of fast coral carrying capacity
  PARAMETER(log_alpha_slow);   // Log of COTS attack rate on slow coral
  PARAMETER(log_alpha_fast);   // Log of COTS attack rate on fast coral
  PARAMETER(log_h_slow);       // Log of handling time for slow coral
  PARAMETER(log_h_fast);       // Log of handling time for fast coral
  PARAMETER(log_m);            // Log of COTS density-dependent mortality
  PARAMETER(log_T_opt);        // Log of optimal temperature for COTS
  PARAMETER(log_sigma_T);      // Log of temperature tolerance width
  PARAMETER(log_obs_sd);       // Log of observation error SD
  PARAMETER(log_c_fast_on_slow); // Log of competitive effect of fast coral on slow coral
  PARAMETER(log_c_slow_on_fast); // Log of competitive effect of slow coral on fast coral
  PARAMETER(log_q_attack);       // Log of temperature sensitivity of COTS attack rates
  PARAMETER(log_v_max);          // Log of maximum predation vulnerability
  PARAMETER(log_beta);           // Log of vulnerability decay rate with coral size
  PARAMETER(log_r_nutr);         // Log of nutrient effect on COTS recruitment
  PARAMETER(log_K_nutr);         // Log of half-saturation constant for nutrient effect
  
  // Transform parameters
  Type r_slow = exp(log_r_slow);
  Type r_fast = exp(log_r_fast);
  Type K_slow = exp(log_K_slow);
  Type K_fast = exp(log_K_fast);
  Type alpha_slow = exp(log_alpha_slow);
  Type alpha_fast = exp(log_alpha_fast);
  Type h_slow = exp(log_h_slow);
  Type h_fast = exp(log_h_fast);
  Type m = exp(log_m);
  Type T_opt = exp(log_T_opt);
  Type sigma_T = exp(log_sigma_T);
  Type obs_sd = exp(log_obs_sd);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize vectors for predictions
  vector<Type> cots_pred(Year.size());
  vector<Type> slow_pred(Year.size());
  vector<Type> fast_pred(Year.size());
  vector<Type> cotsimm_pred(Year.size());
  vector<Type> nutr_pred(Year.size());
  
  // Initialize nutrient predictions with baseline value
  Type nutr_base = Type(0.5);
  nutr_pred(0) = nutr_base;
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Calculate initial predictions from first data point
  cots_pred(0) = std::max(eps, cots_dat(0));
  slow_pred(0) = std::max(eps, slow_dat(0));
  fast_pred(0) = std::max(eps, fast_dat(0));
  
  // Calculate initial immigration prediction
  Type temp_effect_cots_0 = exp(-0.5 * pow((sst_dat(0) - exp(log_T_opt)) / exp(log_sigma_T), 2));
  Type baseline_immigration = Type(0.1);
  cotsimm_pred(0) = baseline_immigration * temp_effect_cots_0;
  
  // Process model
  for(int t = 1; t < Year.size(); t++) {
    // Temperature stress effects - different for each coral type
    Type temp_stress_fast = exp(-2.0 * pow((sst_dat(t) - T_opt) / sigma_T, 2));
    Type temp_stress_slow = exp(-0.5 * pow((sst_dat(t) - T_opt) / sigma_T, 2));
    
    // Competition coefficients
    Type c_fast_on_slow = exp(log_c_fast_on_slow);
    Type c_slow_on_fast = exp(log_c_slow_on_fast);
    
    // Asymmetric competition effects
    Type comp_effect_slow = slow_pred(t-1) + c_fast_on_slow * fast_pred(t-1);
    Type comp_effect_fast = fast_pred(t-1) + c_slow_on_fast * slow_pred(t-1);
    
    // Temperature-dependent attack rates
    Type q_attack = exp(log_q_attack);
    Type temp_effect_attack = exp(q_attack * (sst_dat(t) - T_opt) / T_opt);
    Type alpha_slow_T = alpha_slow * temp_effect_attack;
    Type alpha_fast_T = alpha_fast * temp_effect_attack;
    
    // Size-dependent vulnerability with minimum threshold
    Type v_max = exp(log_v_max);
    Type beta = exp(log_beta);
    Type v_slow = v_max * (Type(0.2) + Type(0.8) * exp(-beta * slow_pred(t-1)));
    Type v_fast = v_max * (Type(0.2) + Type(0.8) * exp(-beta * fast_pred(t-1)));
    
    // Modified Holling Type II functional responses
    Type f_slow = (alpha_slow_T * v_slow * slow_pred(t-1)) / 
                 (Type(1.0) + alpha_slow_T * h_slow * v_slow * slow_pred(t-1) + 
                  alpha_fast_T * h_fast * v_fast * fast_pred(t-1));
    Type f_fast = (alpha_fast_T * v_fast * fast_pred(t-1)) / 
                 (Type(1.0) + alpha_slow_T * h_slow * v_slow * slow_pred(t-1) + 
                  alpha_fast_T * h_fast * v_fast * fast_pred(t-1));
    
    // COTS dynamics with enhanced temperature response
    Type temp_effect_cots = exp(-0.5 * pow((sst_dat(t) - T_opt) / sigma_T, 2));
    
    // Model immigration as temperature-dependent process with baseline rate
    Type baseline_immigration = Type(0.1); // Constant baseline immigration rate
    cotsimm_pred(t) = baseline_immigration * temp_effect_cots * 
                      (Type(1.0) + Type(0.2) * (f_slow + f_fast));
    
    // Population growth depends on available coral food resources
    Type food_availability = (f_slow + f_fast) / (Type(1.0) + f_slow + f_fast);
    
    // Simple AR(1) process for nutrients with mean reversion
    Type nutr_mean = nutr_base;
    Type phi = Type(0.7);       // Autocorrelation coefficient
    nutr_pred(t) = nutr_mean + phi * (nutr_pred(t-1) - nutr_mean);
    
    // Nutrient-enhanced recruitment
    Type r_nutr = exp(log_r_nutr);
    Type K_nutr = exp(log_K_nutr);
    Type nutr_effect = (nutr_pred(t) * r_nutr) / (K_nutr + nutr_pred(t));
    
    // Modified COTS dynamics with nutrient-enhanced recruitment
    cots_pred(t) = cots_pred(t-1) + 
                   temp_effect_cots * food_availability * cots_pred(t-1) * (1 + nutr_effect) -
                   m * pow(cots_pred(t-1), 2) +
                   cotsimm_pred(t) * food_availability;
    cots_pred(t) = std::max(cots_pred(t), eps);
    
    // Coral dynamics with space limitation
    slow_pred(t) = slow_pred(t-1) + 
                   r_slow * slow_pred(t-1) * (1 - comp_effect_slow/K_slow) * temp_stress_slow -
                   f_slow * cots_pred(t-1);
    slow_pred(t) = std::max(slow_pred(t), eps);
    
    fast_pred(t) = fast_pred(t-1) + 
                   r_fast * fast_pred(t-1) * (1 - comp_effect_fast/K_fast) * temp_stress_fast -
                   f_fast * cots_pred(t-1);
    fast_pred(t) = std::max(fast_pred(t), eps);
  }
  
  // Observation model (lognormal)
  for(int t = 0; t < Year.size(); t++) {
    nll -= dnorm(log(cots_dat(t)), log(cots_pred(t)), obs_sd, true);
    nll -= dnorm(log(slow_dat(t)), log(slow_pred(t)), obs_sd, true);
    nll -= dnorm(log(fast_dat(t)), log(fast_pred(t)), obs_sd, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(cotsimm_pred);
  REPORT(nutr_pred);
  
  // Ensure nll is finite
  if (!R_FINITE(asDouble(nll))) {
    nll = Type(1e10); // Large penalty for non-finite values
  }
  
  // Report objective function value
  REPORT(nll);
  
  return nll;
}
