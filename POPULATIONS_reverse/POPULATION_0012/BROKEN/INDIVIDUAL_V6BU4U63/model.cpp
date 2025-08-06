#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Year);           // Time vector (years)
  DATA_VECTOR(sst_dat);        // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);    // COTS immigration rate (individuals/m2/year)
  DATA_VECTOR(cots_dat);       // Observed COTS density (individuals/m2)
  DATA_VECTOR(slow_dat);       // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);       // Observed fast-growing coral cover (%)
  
  // Parameters
  PARAMETER(r_cots);           // COTS population growth rate (year^-1)
  PARAMETER(K_cots);           // COTS carrying capacity (individuals/m2)
  PARAMETER(temp_opt);         // Optimal temperature for COTS survival (°C)
  PARAMETER(temp_tol);         // Temperature tolerance range (°C)
  PARAMETER(alpha_slow);       // Attack rate on slow corals (m2/individual/year)
  PARAMETER(alpha_fast);       // Attack rate on fast corals (m2/individual/year)
  PARAMETER(r_slow);           // Growth rate of slow corals (year^-1)
  PARAMETER(r_fast);           // Growth rate of fast corals (year^-1)
  PARAMETER(K_coral);          // Combined coral carrying capacity (%)
  PARAMETER(obs_sd_cots);      // Observation SD for COTS
  PARAMETER(obs_sd_coral);     // Observation SD for coral cover
  PARAMETER(h_coral);          // Half-saturation constant for coral-dependent predation efficiency (%)
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize vectors for predictions
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  
  // Set initial conditions with small positive values
  Type eps = Type(1e-8);
  cots_pred(0) = cots_dat(0) + eps;
  slow_pred(0) = slow_dat(0) + eps;
  fast_pred(0) = fast_dat(0) + eps;
  
  // Model predictions
  for(int t = 1; t < n; t++) {
    // 1. Temperature effect on COTS (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t-1) - temp_opt) / temp_tol, 2));
    
    // 2. Total coral cover (food availability)
    Type total_coral = slow_pred(t-1) + fast_pred(t-1);
    
    // 3. COTS dynamics with temperature effect and immigration
    Type cots_growth = r_cots * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1)/K_cots) * temp_effect;
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t-1);
    
    // 4. Coral predation rates with density-dependent efficiency
    Type pred_efficiency = total_coral / (total_coral + h_coral);
    Type denominator = Type(1.0) + alpha_slow * slow_pred(t-1) + alpha_fast * fast_pred(t-1);
    Type slow_consumed = pred_efficiency * alpha_slow * cots_pred(t-1) * slow_pred(t-1) / denominator;
    Type fast_consumed = pred_efficiency * alpha_fast * cots_pred(t-1) * fast_pred(t-1) / denominator;
    
    // 5. Coral dynamics with logistic growth and predation
    Type available_space = (K_coral - total_coral) / K_coral;
    slow_pred(t) = slow_pred(t-1) + r_slow * slow_pred(t-1) * available_space - slow_consumed + eps;
    fast_pred(t) = fast_pred(t-1) + r_fast * fast_pred(t-1) * available_space - fast_consumed + eps;
  }
  
  // Observation model using log-normal distribution
  for(int t = 0; t < n; t++) {
    if(cots_dat(t) > 0 && cots_pred(t) > 0) {
      nll -= dnorm(log(cots_dat(t)), log(cots_pred(t)), obs_sd_cots, true);
    }
    if(slow_dat(t) > 0 && slow_pred(t) > 0) {
      nll -= dnorm(log(slow_dat(t)), log(slow_pred(t)), obs_sd_coral, true);
    }
    if(fast_dat(t) > 0 && fast_pred(t) > 0) {
      nll -= dnorm(log(fast_dat(t)), log(fast_pred(t)), obs_sd_coral, true);
    }
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
