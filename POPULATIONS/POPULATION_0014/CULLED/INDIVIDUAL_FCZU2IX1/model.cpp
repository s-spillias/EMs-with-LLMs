#include <TMB.hpp>

// 1. Data declarations:
//    - Year: Time variable (Year)
//    - cots_dat: Observed COTS abundance (individuals/m2)
//    - fast_dat: Observed fast-growing coral cover (%) (Acropora spp.)
//    - slow_dat: Observed slow-growing coral cover (%) (Faviidae/Porites spp.)
template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA
  DATA_VECTOR(Year);            // Time series in years
  DATA_VECTOR(cots_dat);        // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);        // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);        // Observed slow-growing coral cover (%)

  // PARAMETERS
  // growth_rate: Intrinsic growth rate of COTS (year^-1)
  PARAMETER(growth_rate);       
  // carrying_capacity: Environmental carrying capacity for COTS (individuals/m2)
  PARAMETER(carrying_capacity); 
  // predation_rate_fast: Predation rate on fast-growing coral cover per COTS (unitless)
  PARAMETER(predation_rate_fast);
  // predation_rate_slow: Predation rate on slow-growing coral cover per COTS (unitless)
  PARAMETER(predation_rate_slow);
  // outbreak_trigger_threshold: COTS density threshold to trigger an outbreak (individuals/m2)
  PARAMETER(outbreak_trigger_threshold);
  // outbreak_magnitude: Magnitude multiplier applied during outbreak events (unitless)
  PARAMETER(outbreak_magnitude);
  // log_sd_cots: Log-standard deviation for COTS observation noise (log-scale)
  PARAMETER(log_sd_cots);
  Type sd_cots = exp(log_sd_cots) + Type(1e-8);  // Ensure minimum SD for numerical stability

  int n = cots_dat.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Initial conditions: set predictions equal to initial observed values
  cots_pred(0) = cots_dat(0);  // Initial COTS density
  fast_pred(0) = fast_dat(0);  // Initial fast-growing coral cover
  slow_pred(0) = slow_dat(0);  // Initial slow-growing coral cover

  // Model Equations:
  // (1) COTS dynamics:
  //     cots_pred[t] = cots_pred[t-1] + growth_rate * cots_pred[t-1] * (1 - cots_pred[t-1]/carrying_capacity)
  //                    + outbreak
  //     where outbreak = outbreak_magnitude / (1 + exp(-100 * (cots_pred[t-1] - outbreak_trigger_threshold)))
  //
  // (2) Fast-growing coral dynamics:
  //     fast_pred[t] = fast_pred[t-1] - predation_rate_fast * cots_pred[t-1] * fast_pred[t-1] / (fast_pred[t-1] + 1e-8)
  //
  // (3) Slow-growing coral dynamics:
  //     slow_pred[t] = slow_pred[t-1] - predation_rate_slow * cots_pred[t-1] * slow_pred[t-1] / (slow_pred[t-1] + 1e-8)

  for(int t = 1; t < n; t++){
    // Smooth sigmoid to capture outbreak trigger, avoids hard threshold
    Type outbreak = outbreak_magnitude / (Type(1.0) + exp(Type(-100.0) * (cots_pred(t-1) - outbreak_trigger_threshold)));
    cots_pred(t) = cots_pred(t-1)
                   + growth_rate * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1) / carrying_capacity)
                   + outbreak;
    fast_pred(t) = fast_pred(t-1)
                   - predation_rate_fast * cots_pred(t-1) * fast_pred(t-1) / (fast_pred(t-1) + Type(1e-8));
    slow_pred(t) = slow_pred(t-1)
                   - predation_rate_slow * cots_pred(t-1) * slow_pred(t-1) / (slow_pred(t-1) + Type(1e-8));
    // Note: Only the previous time step's values are used in the predictions to avoid data leakage.
  }

  // Likelihood Calculation:
  // Use a lognormal likelihood for COTS abundance observations, ensuring all observations contribute.
  // A minimum standard deviation (sd_cots) is enforced to prevent numerical issues.
  Type nll = 0.0;
  for(int t = 1; t < n; t++){
    Type logcots = log(cots_dat(t) + Type(1e-8)); // log of observed COTS abundance
    Type logpred = log(cots_pred(t) + Type(1e-8));  // log of predicted COTS abundance
    Type log2pi = log(Type(2.0) * Type(3.14159265358979323846)); // constant term for log(2*pi)
    nll += log(cots_dat(t) + Type(1e-8)) + log(sd_cots) + 0.5 * log2pi +
           0.5 * pow((logcots - logpred)/sd_cots, 2);
  }

  // REPORT predictions for diagnostic purposes. These will be available as *_pred values.
  REPORT(cots_pred);  // Predicted COTS densities
  REPORT(fast_pred);  // Predicted fast-growing coral cover
  REPORT(slow_pred);  // Predicted slow-growing coral cover

  return nll;
}
