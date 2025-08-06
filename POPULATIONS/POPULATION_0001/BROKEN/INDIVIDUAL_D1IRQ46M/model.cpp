#include <TMB.hpp>  // TMB library for automatic differentiation and optimization

template<class Type>
Type objective_function<Type>::operator()(){
  
  /***** Data *****/
  DATA_VECTOR(cots_dat);      // 1. Observed COTS abundance (individuals/m2; log-scale likely)
  DATA_VECTOR(fast_dat);      // 2. Observed fast-growing coral cover (%) for Acropora spp.
  DATA_VECTOR(slow_dat);      // 3. Observed slow-growing coral cover (%) for Faviidae/Porites spp.
  
  /***** Parameters *****/
  // 1. COTS dynamics parameters:
  PARAMETER(lambda);          // (year^-1) Intrinsic growth rate of COTS during outbreak conditions [Literature/Expert Opinion]
  PARAMETER(mu);              // (year^-1) Mortality/decline rate of COTS post-outbreak [Literature/Expert Opinion]
  PARAMETER(threshold);       // (individuals/m2) Outbreak threshold for triggering population explosion [Expert Opinion]
  
  // 2. Interaction parameters:
  PARAMETER(epsilon);         // (unitless) Efficiency of COTS predation on coral cover [Expert Opinion]
  PARAMETER(phi_fast);        // (year^-1 per %) Effect of fast coral on modifying COTS reproduction [Expert Opinion]
  PARAMETER(phi_slow);        // (year^-1 per %) Effect of slow coral on modifying COTS reproduction [Expert Opinion]
  PARAMETER(alpha_coral);     // Saturation constant for the effect of coral cover on COTS reproduction
  
  // 3. Coral regeneration parameters:
  PARAMETER(coral_regen_fast); // (year^-1) Regeneration rate for fast-growing coral (Acropora spp.) [Literature/Expert Opinion]
  PARAMETER(coral_regen_slow); // (year^-1) Regeneration rate for slow-growing coral (Faviidae/Porites spp.) [Literature/Expert Opinion]
  
  // 4. Process error parameters (log-scale, then exponentiated):
  PARAMETER(log_sigma_cots);  // log(process error sd for COTS dynamics)
  PARAMETER(log_sigma_fast);  // log(process error sd for fast coral dynamics)
  PARAMETER(log_sigma_slow);  // log(process error sd for slow coral dynamics)
  
  // Small constant added for numerical stability to prevent division by zero
  Type delta = Type(1e-8);
  Type sigma_cots = exp(log_sigma_cots) + delta;
  Type sigma_fast = exp(log_sigma_fast) + delta;
  Type sigma_slow = exp(log_sigma_slow) + delta;
  
  int n = cots_dat.size();  // number of time steps/time series length
  
  // Vectors to hold model predictions (must follow naming convention: _pred for each _dat)
  vector<Type> cots_pred(n);  // predicted COTS abundance
  vector<Type> fast_pred(n);  // predicted fast coral cover
  vector<Type> slow_pred(n);  // predicted slow coral cover
  
  // Set initial conditions using first observation to avoid leakage of current time step values
  cots_pred[0] = cots_dat[0];   // Initial COTS abundance
  fast_pred[0] = fast_dat[0];   // Initial fast coral cover
  slow_pred[0] = slow_dat[0];   // Initial slow coral cover

  Type nll = 0.0;  // negative log likelihood

  // Loop over time steps using previous time step’s state for predictions
  for(int t = 1; t < n; t++){
    // Equation 1: COTS dynamics (boom-bust cycle) with saturating coral influence
    // Description:
    //   (1) A smooth outbreak trigger is applied via a logistic function.
    //   (2) COTS reproduction is modified by the outbreak trigger and saturating coral cover effects.
    //   (3) An exponential growth/decline is applied.
    Type trigger = Type(1) / (Type(1) + exp(-(cots_pred[t-1] - threshold)));  // Smooth outbreak trigger response
    cots_pred[t] = cots_pred[t-1] * exp(lambda * trigger - mu + 
                   phi_fast * fast_pred[t-1] / (Type(1) + alpha_coral * fast_pred[t-1]) +
                   phi_slow * slow_pred[t-1] / (Type(1) + alpha_coral * slow_pred[t-1])) + delta;
    
    // Equation 2: Fast coral dynamics (Acropora spp.)
    // Description:
    //   (1) Coral cover regenerates towards full cover (scaled 0 to 1).
    //   (2) Losses occur via predation represented by a saturating functional response.
    fast_pred[t] = fast_pred[t-1] + coral_regen_fast * (Type(1) - fast_pred[t-1])
                   - epsilon * cots_pred[t-1] * fast_pred[t-1] / (fast_pred[t-1] + delta) + delta;
    
    // Equation 3: Slow coral dynamics (Faviidae/Porites spp.)
    // Description:
    //   (1) Similar regenerative dynamics with its own rate.
    //   (2) Losses from predation by COTS using a saturating functional response.
    slow_pred[t] = slow_pred[t-1] + coral_regen_slow * (Type(1) - slow_pred[t-1])
                   - epsilon * cots_pred[t-1] * slow_pred[t-1] / (slow_pred[t-1] + delta) + delta;
    // Ensure predictions are bounded below by delta to avoid log(0) issues
    if(cots_pred[t] < delta || std::isnan(Value(cots_pred[t]))) cots_pred[t] = delta;
    if(fast_pred[t] < delta || std::isnan(Value(fast_pred[t]))) fast_pred[t] = delta;
    if(slow_pred[t] < delta || std::isnan(Value(slow_pred[t]))) slow_pred[t] = delta;
    
    // Likelihood contributions using lognormal error distributions:
    // For a lognormal distribution, if log(x) ~ Normal(log(pred), sigma),
    // then the density on x is given by dnorm(log(x), log(pred), sigma, true) - log(x).
    nll -= dnorm(log(cots_dat[t] + delta), log(cots_pred[t]), sigma_cots, true) - log(cots_dat[t] + delta);
    nll -= dnorm(log(fast_dat[t] + delta), log(fast_pred[t]), sigma_fast, true) - log(fast_dat[t] + delta);
    nll -= dnorm(log(slow_dat[t] + delta), log(slow_pred[t]), sigma_slow, true) - log(slow_dat[t] + delta);
  }
  
  // Reporting model predictions for post-estimation diagnostics
  REPORT(cots_pred); // 1. Predicted COTS abundance over time
  REPORT(fast_pred); // 2. Predicted fast-growing coral cover over time
  REPORT(slow_pred); // 3. Predicted slow-growing coral cover over time
  return nll;
}
