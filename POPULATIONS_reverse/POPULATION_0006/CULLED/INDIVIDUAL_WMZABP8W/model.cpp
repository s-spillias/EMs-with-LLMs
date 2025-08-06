#include <TMB.hpp>

// 1. Data section: observed values for COTS and corals (units in individuals/m2, %, etc.)
template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;
  
  // DATA: response observations
  DATA_VECTOR(cots_dat);      // Adult COTS abundance (individuals/m2)
  DATA_VECTOR(slow_dat);      // Slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);      // Fast-growing coral cover (%)
  
  // DATA: forcing variables
  DATA_VECTOR(sst_dat);       // Sea-surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);   // COTS immigration rate (individuals/m2/year)
  
  // 2. Parameters (in log-scale for positivity)
  // log_growth_rate: Log intrinsic growth rate of COTS (year^-1)
  // log_feeding_rate_slow: Log feeding rate on slow-growing corals (m2/(individual·year))
  // log_feeding_rate_fast: Log feeding rate on fast-growing corals (m2/(individual·year))
  // log_coral_loss_rate: Log coral loss rate due to predation (unitless scaling factor)
  PARAMETER(log_growth_rate);
  PARAMETER(log_feeding_rate_slow);
  PARAMETER(log_feeding_rate_fast);
  PARAMETER(log_coral_loss_rate);
  
  // Transform parameters from log-scale for numerical stability and positivity
  Type growth_rate       = exp(log_growth_rate);        // (year^-1)
  Type feeding_rate_slow = exp(log_feeding_rate_slow);    // (m2/(individual·year))
  Type feeding_rate_fast = exp(log_feeding_rate_fast);    // (m2/(individual·year))
  Type coral_loss_rate   = exp(log_coral_loss_rate);      // (unitless)
  
  // 3. Define a small constant to avoid numerical issues
  Type eps = Type(1e-8);
  
  // 4. Initialize the negative log likelihood (nll)
  Type nll = 0;
  
  int n_obs = cots_dat.size();
  vector<Type> cots_pred(n_obs);  // Predicted COTS abundance (_pred)
  vector<Type> slow_pred(n_obs);  // Predicted slow coral cover (_pred)
  vector<Type> fast_pred(n_obs);  // Predicted fast coral cover (_pred)
  
  // 5. Model Equations:
  // Equation 1: COTS dynamics
  //   Predicted COTS = growth_rate * (observed COTS + eps) + immigration 
  //                    - (feeding_rate_slow * slow coral cover + feeding_rate_fast * fast coral cover) * (observed COTS + eps)
  // Equation 2: Slow coral dynamics
  //   Predicted slow coral = observed slow coral - coral_loss_rate * feeding_rate_slow * slow coral * (observed COTS + eps)
  // Equation 3: Fast coral dynamics
  //   Predicted fast coral = observed fast coral - coral_loss_rate * feeding_rate_fast * fast coral * (observed COTS + eps)
  // Set initial conditions using first observation values.
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);

  // Likelihood contribution for initial conditions using lognormal density reparameterization
  nll -= (dnorm(log(cots_dat(0) + eps), log(cots_pred(0) + eps), Type(0.1), true) - log(cots_dat(0) + eps));
  nll -= (dnorm(log(slow_dat(0) + eps), log(slow_pred(0) + eps), Type(0.1), true) - log(slow_dat(0) + eps));
  nll -= (dnorm(log(fast_dat(0) + eps), log(fast_pred(0) + eps), Type(0.1), true) - log(fast_dat(0) + eps));

  // Recurrence for subsequent time steps using previous predictions.
  for(int i = 1; i < n_obs; i++){
    // Equation 1: COTS dynamics predicted solely by previous predictions and external forcing.
    {
      Type tmp = cots_pred(i-1) * growth_rate + cotsimm_dat(i)
                   - (feeding_rate_slow * slow_pred(i-1) + feeding_rate_fast * fast_pred(i-1)) * cots_pred(i-1);
      cots_pred(i) = tmp > eps ? tmp : eps;
    }
    
    // Equation 2: Slow-growing coral dynamics with smooth exponential decline 
    //         due to predation by COTS.
    {
      Type tmp = slow_pred(i-1) * exp(- coral_loss_rate * feeding_rate_slow * cots_pred(i-1));
      slow_pred(i) = tmp > eps ? tmp : eps;
    }
    
    // Equation 3: Fast-growing coral dynamics with smooth exponential decline 
    //         due to predation by COTS.
    {
      Type tmp = fast_pred(i-1) * exp(- coral_loss_rate * feeding_rate_fast * cots_pred(i-1));
      fast_pred(i) = tmp > eps ? tmp : eps;
    }
    
    // Likelihood Calculation:
    nll -= (dnorm(log(cots_dat(i) + eps), log(cots_pred(i) + eps), Type(0.1), true) - log(cots_dat(i) + eps));
    nll -= (dnorm(log(slow_dat(i) + eps), log(slow_pred(i) + eps), Type(0.1), true) - log(slow_dat(i) + eps));
    nll -= (dnorm(log(fast_dat(i) + eps), log(fast_pred(i) + eps), Type(0.1), true) - log(fast_dat(i) + eps));
  }
  
  // 7. Reporting section: include all important variables
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(growth_rate);         // Intrinsic growth rate (year^-1)
  REPORT(feeding_rate_slow);   // Feeding rate on slow corals (m2/(individual·year))
  REPORT(feeding_rate_fast);   // Feeding rate on fast corals (m2/(individual·year))
  REPORT(coral_loss_rate);     // Coral loss rate (unitless)
  
  return nll;
}
