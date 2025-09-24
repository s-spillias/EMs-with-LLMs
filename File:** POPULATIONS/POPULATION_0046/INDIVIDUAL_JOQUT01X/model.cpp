#include <TMB.hpp>


// 2. Model parameters:
//    a. growth_rate: Intrinsic growth rate for COTS (year^-1)
//    b. cots_mortality: Mortality rate for COTS (year^-1)
//    c. coral_mortality_fast: Mortality rate for fast-growing coral due to predation (year^-1)
//    d. coral_mortality_slow: Mortality rate for slow-growing coral due to predation (year^-1)
//    e. threshold: COTS density threshold (individuals/m2) triggering outbreak events
//    f. empirical_efficiency: Efficiency scaling for outbreak triggering (unitless)
//    g. process_noise_log_sd: Standard deviation for lognormal observation error (log-scale)
//    h. beta: Coefficient for the saturating functional response in coral predation
PARAMETER(growth_rate);           // e.g., initial value ~0.2
PARAMETER(cots_mortality);        // e.g., initial value ~0.1
PARAMETER(coral_mortality_fast);  // e.g., initial value ~0.05
PARAMETER(coral_mortality_slow);  // e.g., initial value ~0.03
PARAMETER(threshold);            // e.g., initial value ~50.0
PARAMETER(empirical_efficiency); // e.g., initial value ~0.1
PARAMETER(process_noise_log_sd); // e.g., initial value ~0.2
PARAMETER(beta);                 // e.g., initial value ~1e-3

// 3. Template Model Builder (TMB) model function:
//    Equation descriptions:
//      (1) COTS dynamics: cots_pred[t] = cots_pred[t-1] + growth_rate * cots_pred[t-1] * outbreak_factor - cots_mortality * cots_pred[t-1] + cotsimm_dat[t-1]
//          where outbreak_factor = 1/(1+exp(-empirical_efficiency*(cots_pred[t-1]-threshold)))
//      (2) Fast coral dynamics: fast_pred[t] = fast_pred[t-1] - coral_mortality_fast * fast_pred[t-1] * (cots_pred[t-1] / (fast_pred[t-1] + eps))
//      (3) Slow coral dynamics: slow_pred[t] = slow_pred[t-1] - coral_mortality_slow * slow_pred[t-1] * (cots_pred[t-1] / (slow_pred[t-1] + eps))
//      (4) Likelihood: lognormal error applied to each observed data series with process_noise_log_sd
template<class Type>
Type objective_function<Type>::operator() ()
{
  int n = cots_dat.size();                  // Number of time steps
  vector<Type> nll(n);                      // Negative log-likelihood contributions per time step
  
  // State predictions for COTS and coral covers
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Initialize predictions using first observed values to avoid data leakage
  cots_pred[0] = cots_dat[0];   // Initial COTS density (individuals/m2)
  fast_pred[0] = fast_dat[0];   // Initial fast-growing coral cover (%)
  slow_pred[0] = slow_dat[0];   // Initial slow-growing coral cover (%)

  // Small constant to prevent division by zero
  Type eps = Type(1e-8);

  // Loop over time steps (starting at 1 to use previous time step's predictions)
  for(int t = 1; t < n; t++){
    // (1) Outbreak dynamics for COTS:
    //     Smooth outbreak triggering via a sigmoidal function based on the threshold.
    Type outbreak_factor = Type(1.0) / (Type(1.0) + exp(-empirical_efficiency * (cots_pred[t-1] - threshold)));
    cots_pred[t] = cots_pred[t-1] 
                   + growth_rate * cots_pred[t-1] * outbreak_factor   // density-dependent growth with outbreak modifier
                   - cots_mortality * cots_pred[t-1]                     // natural mortality
                   + cotsimm_dat[t-1];                                 // environmental larval immigration effect

    // (2) Coral dynamics for fast-growing species:
    //     Loss due to COTS predation modeled with a saturating functional response.
    fast_pred[t] = fast_pred[t-1] 
                   - coral_mortality_fast * fast_pred[t-1] * (cots_pred[t-1] / (fast_pred[t-1] + eps));

    // (3) Coral dynamics for slow-growing species:
    slow_pred[t] = slow_pred[t-1] 
                   - coral_mortality_slow * slow_pred[t-1] * (cots_pred[t-1] / (slow_pred[t-1] + eps));

    // (4) Likelihood using lognormal errors for all observed data:
    //     Log-transform observations to accommodate wide data range with fixed sd = process_noise_log_sd.
    nll[t] = -dnorm(log(cots_dat[t] + eps), log(cots_pred[t] + eps), process_noise_log_sd, true)
             - dnorm(log(fast_dat[t] + eps), log(fast_pred[t] + eps), process_noise_log_sd, true)
             - dnorm(log(slow_dat[t] + eps), log(slow_pred[t] + eps), process_noise_log_sd, true);
  }
  
  // Aggregate negative log-likelihood
  Type jnll = -accumulate(nll);

  // Reporting model predictions for diagnostic output (_pred values)
  REPORT(cots_pred);  // 1: Predicted COTS density (individuals/m2)
  REPORT(fast_pred);  // 2: Predicted fast-growing coral cover (%)
  REPORT(slow_pred);  // 3: Predicted slow-growing coral cover (%)

  return jnll;
}
