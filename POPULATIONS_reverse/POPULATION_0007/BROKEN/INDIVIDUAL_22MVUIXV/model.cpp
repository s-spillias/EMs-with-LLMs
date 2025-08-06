#include <TMB.hpp>  // TMB header for model framework

// 4. Model structure and equations:
//    Equation Descriptions:
//    (1) slow_pred[t] = slow_pred[t-1] * exp(slow_growth) * (1 - predation_slow * cots_pred[t-1])
//        - Models slow coral cover dynamics with exponential growth and impact from starfish.
//    (2) fast_pred[t] = fast_pred[t-1] * exp(fast_growth) * (1 - predation_fast * cots_pred[t-1])
//        - Models fast coral cover dynamics with exponential growth and impact from starfish.
//    (3) cots_pred[t] = cotsimm_dat[t] * cots_effectiveness + 0.3 * slow_pred[t-1] + 0.2 * fast_pred[t-1]
//        - Computes the effective impact of starfish feeding on corals, combining immigration and consumption.
template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;
  
  // 1. Data inputs:
  //    slow_dat: observed slow-growing coral cover (%) for Faviidae/Porites
  //    fast_dat: observed fast-growing coral cover (%) for Acropora
  //    cots_dat: observed Crown-of-Thorns starfish abundance (individuals/m^2)
  //    sst_dat: external forcing: sea surface temperature (Celsius)
  //    cotsimm_dat: external forcing: Crown-of-Thorns immigration rate (individuals/m^2/year)
  DATA_VECTOR(slow_dat);     // observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);     // observed fast-growing coral cover (%)
  DATA_VECTOR(cots_dat);     // observed Crown-of-Thorns abundance (individuals/m^2)
  DATA_VECTOR(sst_dat);      // sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);  // Crown-of-Thorns immigration rate (individuals/m^2/year)
  
  // 2. Parameters to be estimated:
  //    slow_growth: intrinsic growth rate for slow-growing coral (year^-1)
  //    fast_growth: intrinsic growth rate for fast-growing coral (year^-1)
  //    predation_slow: impact rate of starfish on slow-growing corals (unitless)
  //    predation_fast: impact rate of starfish on fast-growing corals (unitless)
  //    cots_effectiveness: feeding effectiveness multiplier (unitless)
  //    log_sd: log standard deviation for observation error (for lognormal likelihood)
  //    slow_init: initial condition for slow-growing coral cover (%)
  //    fast_init: initial condition for fast-growing coral cover (%)
  //    cots_init: initial condition for Crown-of-Thorns impact measure
  PARAMETER(slow_growth);
  PARAMETER(fast_growth);
  PARAMETER(predation_slow);
  PARAMETER(predation_fast);
  PARAMETER(cots_effectiveness);
  PARAMETER(log_sd);
  PARAMETER(slow_init);
  PARAMETER(fast_init);
  PARAMETER(cots_init);
  
  int n = slow_dat.size();  // assuming all data vectors are of equal length
  Type nll = 0;
  Type eps = Type(1e-8);
  
  // Initialize prediction vectors
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> cots_pred(n);
  
  // Set initial conditions from parameters to avoid data leakage
  slow_pred[0] = slow_init;
  fast_pred[0] = fast_init;
  cots_pred[0] = cots_init;
  
  // Recursive predictions using ecological relationships
  for(int t = 1; t < n; t++){
    slow_pred[t] = slow_pred[t-1] * exp(slow_growth) * (1 - predation_slow * cots_pred[t-1]);
    fast_pred[t] = fast_pred[t-1] * exp(fast_growth) * (1 - predation_fast * cots_pred[t-1]);
    cots_pred[t] = cotsimm_dat[t] * cots_effectiveness + Type(0.3) * slow_pred[t-1] + Type(0.2) * fast_pred[t-1];
    
    Type sigma = exp(log_sd);
    if(sigma < Type(0.1)) sigma = Type(0.1);
    
    // Likelihood: lognormal error for observations
    nll -= (-log(slow_dat[t] + eps) + dnorm(log(slow_dat[t] + eps), log(slow_pred[t] + eps), sigma, true));
    nll -= (-log(fast_dat[t] + eps) + dnorm(log(fast_dat[t] + eps), log(fast_pred[t] + eps), sigma, true));
    nll -= (-log(cots_dat[t] + eps) + dnorm(log(cots_dat[t] + eps), log(cots_pred[t] + eps), sigma, true));
  }
  
  ADREPORT(slow_pred);
  ADREPORT(fast_pred);
  ADREPORT(cots_pred);
  
  return nll;
}
