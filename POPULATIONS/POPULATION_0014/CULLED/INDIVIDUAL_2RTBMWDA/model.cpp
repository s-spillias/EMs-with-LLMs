#include <TMB.hpp>

// TMB Model for modeling episodic outbreaks of Crown-of-Thorns Starfish (COTS)
// and their impact on coral communities on the Great Barrier Reef.
//
// Equations:
// 1. COTS Dynamics:
//    cots_pred[t] = cots_pred[t-1] + growth_rate * cots_pred[t-1] * (1/(1 + resource_limitation_coeff * cots_pred[t-1] + 1e-8))
//                   + burst_efficiency * exp( - (cots_pred[t-1] - outbreak_threshold)^2 )
//                   - predation_efficiency * cots_pred[t-1]
// 2. Fast-growing Coral Dynamics:
//    fast_pred[t] = fast_pred[t-1] + fast_growth_rate * (fast_max_cover - fast_pred[t-1])
//                   - coral_pred_rate * cots_pred[t-1] * fast_pred[t-1]
// 3. Slow-growing Coral Dynamics:
//    slow_pred[t] = slow_pred[t-1] + slow_growth_rate * (slow_max_cover - slow_pred[t-1])
//                   - coral_pred_rate * cots_pred[t-1] * slow_pred[t-1]
//
// Note: All predictions use lagged values to prevent data leakage.
// A small constant (1e-8) is added in denominators to ensure numerical stability.

template<class Type>
Type objective_function<Type>::operator() () {
  
  // Data vectors: 'time' must match the first column of the data file.
  DATA_VECTOR(Year);                            // Time (year)
  DATA_VECTOR(cots_dat);                        // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);                        // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);                        // Observed slow-growing coral cover (%)
  
  // Parameters for COTS dynamics:
  PARAMETER(growth_rate);         // Intrinsic growth rate of COTS (year^-1)
  PARAMETER(predation_efficiency); // Reduction rate of COTS due to predation (unitless)
  PARAMETER(resource_limitation_coeff); // Coefficient for resource saturation (per unit)
  PARAMETER(outbreak_threshold);  // Threshold value triggering an outbreak (individuals/m2)
  PARAMETER(burst_efficiency);    // Efficiency converting resource surplus to outbreak (unitless)
  
  // Parameters for coral dynamics:
  PARAMETER(fast_growth_rate);    // Intrinsic growth rate for fast-growing coral (year^-1)
  PARAMETER(slow_growth_rate);    // Intrinsic growth rate for slow-growing coral (year^-1)
  PARAMETER(fast_max_cover);      // Maximum percent cover for fast-growing coral (%)
  PARAMETER(slow_max_cover);      // Maximum percent cover for slow-growing coral (%)
  PARAMETER(coral_pred_rate);     // Rate at which COTS reduce coral cover (per individual per percent cover)
  
  // Observation error parameter:
  PARAMETER(log_sigma);  // Log standard deviation for observational error
  Type sigma = exp(log_sigma) + Type(1e-8);  // Ensure sigma is positive
  
  int n = Year.size();
  // Vectors to store model predictions:
  vector<Type> cots_pred(n);   // Predicted COTS abundance
  vector<Type> fast_pred(n);   // Predicted fast-growing coral cover
  vector<Type> slow_pred(n);   // Predicted slow-growing coral cover
  
  // Initialize predictions with first observation (lagged initialization)
  cots_pred[0] = cots_dat[0];
  fast_pred[0] = fast_dat[0];
  slow_pred[0] = slow_dat[0];

  Type nll = 0.0; // Negative Log Likelihood
  
  // Loop over time steps (starting from t=1 so that we use lagged values)
  for (int t = 1; t < n; t++) {
    // ---- COTS Dynamics ----
    // Calculate resource limitation using a saturating function:
    Type resource_limit = Type(1.0) / (Type(1.0) + resource_limitation_coeff * cots_pred[t-1] + Type(1e-8));
    // Outbreak effect: smooth trigger using an exponential decay from the threshold, scaled for stability:
    Type outbreak_effect = burst_efficiency * exp( - pow(cots_pred[t-1] - outbreak_threshold, 2) / Type(1e4) );
    // Update COTS abundance using previous value and process efficiencies:
    cots_pred[t] = cots_pred[t-1] + growth_rate * cots_pred[t-1] * resource_limit + outbreak_effect - predation_efficiency * cots_pred[t-1];
    // Force positive predictions for stability:
    if(cots_pred[t] < Type(1e-8)) cots_pred[t] = Type(1e-8);
    
    // ---- Fast-growing Coral Dynamics ----
    fast_pred[t] = fast_pred[t-1] + fast_growth_rate * (fast_max_cover - fast_pred[t-1])
                   - coral_pred_rate * cots_pred[t-1] * fast_pred[t-1];
    if(fast_pred[t] < Type(1e-8)) fast_pred[t] = Type(1e-8);
    
    // ---- Slow-growing Coral Dynamics ----
    slow_pred[t] = slow_pred[t-1] + slow_growth_rate * (slow_max_cover - slow_pred[t-1])
                   - coral_pred_rate * cots_pred[t-1] * slow_pred[t-1];
    if(slow_pred[t] < Type(1e-8)) slow_pred[t] = Type(1e-8);
    
    // ---- Likelihood Calculation ----
    // Incorporate likelihood for COTS and both coral observations simultaneously.
    nll -= dnorm(cots_dat[t], cots_pred[t], sigma, true);
    nll -= dnorm(fast_dat[t], fast_pred[t], sigma, true);
    nll -= dnorm(slow_dat[t], slow_pred[t], sigma, true);
  }
  
  // Report predictions so they can be inspected:
  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
