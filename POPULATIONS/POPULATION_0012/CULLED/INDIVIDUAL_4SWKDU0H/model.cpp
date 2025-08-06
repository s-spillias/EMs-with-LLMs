#include <TMB.hpp>

// Model Description:
// 1. COTS Dynamics: Employs a logistic function for outbreak triggering. When lagged COTS abundance exceeds a threshold,
//    growth is boosted by larval immigration and declines otherwise, representing boom-bust cycles.
// 2. Fast Coral Dynamics: Fast-growing corals (Acropora spp.) decline due to COTS predation modeled with a saturating functional response.
// 3. Slow Coral Dynamics: Slow-growing corals (Faviidae spp. and Porites spp.) decline similarly under COTS predation.
// Each parameter is annotated with units and its source; equations are designed to avoid numerical instability via small constants.

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;
  
  // DATA INPUTS (units as in data files)
  DATA_VECTOR(Year);                      // Year (time unit: year)
  DATA_VECTOR(sst_dat);                   // Sea-Surface Temperature (°C)
  DATA_VECTOR(cotsimm_dat);               // COTS larval immigration rate (individuals/m2/year)
  DATA_VECTOR(cots_dat);                  // Observed adult COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);                  // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);                  // Observed slow-growing coral cover (%)
  // Using fixed minimum standard deviation for likelihood to prevent numerical issues
  Type min_sd = Type(1e-2);

  // PARAMETERS (all provided in log-scale to ensure positivity)
  PARAMETER(log_growth_rate);             // Log intrinsic growth rate for COTS (log(year^-1)); from expert opinion
  PARAMETER(log_decline_rate);            // Log decline rate for COTS post-outbreak (log(year^-1)); from literature
  PARAMETER(log_coral_predation_eff_fast);  // Log predation efficiency on fast coral (log(m2/(individuals*%))); initial estimate
  PARAMETER(log_coral_predation_eff_slow);  // Log predation efficiency on slow coral (log(m2/(individuals*%))); initial estimate
  PARAMETER(log_threshold);               // Log outbreak threshold for COTS abundance (log(individuals/m2)); expert assessment
  PARAMETER(sst_effect);                // Effect of sea-surface temperature anomalies on outbreak threshold; modulates threshold based on (sst_dat - 27.0°C)

  // Transform parameters from log-scale
  Type growth_rate = exp(log_growth_rate);           // Intrinsic growth rate (year^-1)
  Type decline_rate = exp(log_decline_rate);           // Decline rate (year^-1)
  Type pred_eff_fast = exp(log_coral_predation_eff_fast); // Predation efficiency on fast coral (m2/(individuals*%))
  Type pred_eff_slow = exp(log_coral_predation_eff_slow); // Predation efficiency on slow coral (m2/(individuals*%))
  Type threshold = exp(log_threshold);                 // Outbreak threshold (individuals/m2)

  // Likelihood accumulator
  Type nll = 0.0;
  int n = Year.size();
  
  // Predicted state vectors for reporting (indexed by time)
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Initialize predictions with first observed values (to avoid data leakage)
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Small constant to prevent division by zero and ensure numerical stability
  Type eps = Type(1e-8);

  // Loop over time steps (using lagged state variables only)
  for(int t = 1; t < n; t++){
    // Equation 1: COTS Dynamics
    // (1) Outbreak triggering is modeled by a smooth logistic function based on lagged COTS abundance versus threshold.
    // (2) Growth is boosted by larval immigration and scaled by the outbreak trigger.
    // (3) Decline is applied when not in outbreak.
    Type outbreak_trigger = Type(1.0) / (Type(1.0) + exp(-(cots_pred(t-1) - (threshold + sst_effect * (sst_dat(t-1) - 27.0)))));
    cots_pred(t) = cots_pred(t-1) + (growth_rate * outbreak_trigger * (cotsimm_dat(t-1) + eps)
                      - decline_rate * (Type(1.0) - outbreak_trigger) * cots_pred(t-1));
    if(cots_pred(t) < eps) cots_pred(t) = eps;

    // Equation 2: Fast Coral Dynamics
    // Decline in fast-growing coral cover due to saturating predation by COTS.
    fast_pred(t) = fast_pred(t-1) - pred_eff_fast * cots_pred(t-1) * fast_pred(t-1) / (fast_pred(t-1) + eps);
    if(fast_pred(t) < eps) fast_pred(t) = eps;

    // Equation 3: Slow Coral Dynamics
    // Decline in slow-growing coral cover due to COTS predation with a similar saturating response.
    slow_pred(t) = slow_pred(t-1) - pred_eff_slow * cots_pred(t-1) * slow_pred(t-1) / (slow_pred(t-1) + eps);
    if(slow_pred(t) < eps) slow_pred(t) = eps;
  }

  // Likelihood Calculation:
  // Model observations using a lognormal error structure to handle multiple orders of magnitude.
  // A fixed minimum standard deviation (min_sd) prevents numerical issues.
  for(int t = 0; t < n; t++){
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), min_sd, true); // (1) COTS data
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), min_sd, true); // (2) Fast coral data
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), min_sd, true); // (3) Slow coral data
  }

  // REPORT predicted values for model diagnostics and further analysis
  REPORT(cots_pred);    // Predicted COTS abundance (individuals/m2)
  REPORT(fast_pred);    // Predicted fast-growing coral cover (%)
  REPORT(slow_pred);    // Predicted slow-growing coral cover (%)
  
  return nll;
}
