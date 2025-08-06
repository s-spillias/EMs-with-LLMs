#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(cots_dat);    // Observed COTS abundance (ind/m2)
  DATA_VECTOR(slow_dat);    // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);    // Observed fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);     // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat); // COTS immigration rate (ind/m2/year)
  
  // Parameters
  PARAMETER(r_slow);        // Growth rate of slow-growing corals
  PARAMETER(r_fast);        // Growth rate of fast-growing corals
  PARAMETER(K_slow);        // Carrying capacity of slow-growing corals
  PARAMETER(K_fast);        // Carrying capacity of fast-growing corals
  PARAMETER(a_slow);        // COTS attack rate on slow corals
  PARAMETER(a_fast);        // COTS attack rate on fast corals
  PARAMETER(m_cots);        // Natural mortality rate of COTS
  PARAMETER(temp_opt);      // Optimal temperature for coral growth
  PARAMETER(temp_range);    // Temperature tolerance range
  PARAMETER(log_sigma_cots);    // Log of COTS observation error SD
  PARAMETER(log_sigma_coral);   // Log of coral observation error SD
  PARAMETER(agg_strength);      // COTS aggregation strength coefficient

  // Derived values
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_coral = exp(log_sigma_coral);
  Type small_const = Type(1e-8);  // Small constant to prevent division by zero
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Vectors for predictions
  vector<Type> cots_pred(cots_dat.size());
  vector<Type> slow_pred(slow_dat.size());
  vector<Type> fast_pred(fast_dat.size());
  
  // Initial values
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Process model
  for(int t = 1; t < cots_dat.size(); t++) {
    // 1. Temperature effect on coral growth (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t) - temp_opt)/temp_range, 2));
    
    // 2. Coral growth with temperature effect and density dependence
    Type slow_growth = r_slow * temp_effect * slow_pred(t-1) * (1 - slow_pred(t-1)/(K_slow + small_const));
    Type fast_growth = r_fast * temp_effect * fast_pred(t-1) * (1 - fast_pred(t-1)/(K_fast + small_const));
    
    // 3. COTS predation rates
    Type pred_slow = a_slow * cots_pred(t-1) * slow_pred(t-1) / (Type(1.0) + small_const);
    Type pred_fast = a_fast * cots_pred(t-1) * fast_pred(t-1) / (Type(1.0) + small_const);
    
    // Apply density-dependent effect
    pred_slow *= Type(1.0) + agg_strength * cots_pred(t-1) / (Type(1.0) + cots_pred(t-1));
    pred_fast *= Type(1.0) + agg_strength * cots_pred(t-1) / (Type(1.0) + cots_pred(t-1));
    
    // 4. COTS population dynamics with density dependence and immigration
    Type cots_mort = m_cots * cots_pred(t-1) / (Type(1.0) + cots_pred(t-1));  // Bounded mortality
    cots_pred(t) = cots_pred(t-1) + cotsimm_dat(t) - cots_mort;
    
    // 5. Update coral cover with bounded changes
    slow_pred(t) = slow_pred(t-1) * exp(slow_growth/(slow_pred(t-1) + small_const) - pred_slow/(slow_pred(t-1) + small_const));
    fast_pred(t) = fast_pred(t-1) * exp(fast_growth/(fast_pred(t-1) + small_const) - pred_fast/(fast_pred(t-1) + small_const));
    
    // Bound predictions to reasonable ranges
    cots_pred(t) = Type(0.001) + Type(0.998) * cots_pred(t)/(Type(1.0) + cots_pred(t));
    slow_pred(t) = Type(0.001) + Type(99.998) * slow_pred(t)/(Type(100.0) + slow_pred(t));
    fast_pred(t) = Type(0.001) + Type(99.998) * fast_pred(t)/(Type(100.0) + fast_pred(t));
  }
  
  // Observation model using lognormal distribution
  for(int t = 0; t < cots_dat.size(); t++) {
    // COTS abundance
    nll -= dnorm(log(cots_dat(t) + small_const), 
                 log(cots_pred(t) + small_const), 
                 sigma_cots, true);
    
    // Coral cover
    nll -= dnorm(log(slow_dat(t) + small_const),
                 log(slow_pred(t) + small_const),
                 sigma_coral, true);
    nll -= dnorm(log(fast_dat(t) + small_const),
                 log(fast_pred(t) + small_const),
                 sigma_coral, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(sigma_cots);
  REPORT(sigma_coral);
  
  return nll;
}
