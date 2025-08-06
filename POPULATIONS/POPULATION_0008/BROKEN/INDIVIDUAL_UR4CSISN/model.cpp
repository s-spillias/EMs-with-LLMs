#include <TMB.hpp>
template <class Type>
Type softplus(Type x) {
  return (x > Type(20)) ? (x + log(Type(1) + exp(-x))) : log(Type(1) + exp(x));
}

// TMB model for simulating COTS outbreaks and their impact on coral communities.
template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;
  
  // DATA INPUTS
  DATA_VECTOR(time);                         // Time (Year) from the observation dataset.
  DATA_VECTOR(cots_dat);                       // Observed COTS abundance (individuals/m^2).
  DATA_VECTOR(fast_dat);                       // Observed cover of fast-growing coral (Acropora in %).
  DATA_VECTOR(slow_dat);                       // Observed cover of slow-growing coral (Faviidae/Porites in %).
  DATA_VECTOR(sst_dat);                        // Sea Surface Temperature (°C) influencing outbreak dynamics.
  DATA_VECTOR(cotsimm_dat);                    // COTS larval immigration rate (individuals/m^2/year).
  
  // MODEL PARAMETERS
  // Log-transformed intrinsic growth rate for COTS outbreak (year^-1).
  PARAMETER(log_growth_rate_cots);
  // Log-transformed decline rate post-outbreak (year^-1).
  PARAMETER(log_decline_rate_cots);
  // Log-transformed intrinsic growth rate of corals (year^-1).
  PARAMETER(log_intrinsic_coral);
  // Efficiency of COTS predation on fast-growing corals (unitless, 0-1).
  PARAMETER(efficiency_predation_fast);
  // Efficiency of COTS predation on slow-growing corals (unitless, 0-1).
  PARAMETER(efficiency_predation_slow);
  // Threshold for resource limitation effect (in % cover).
  PARAMETER(threshold_coral);
  // Effect of Sea Surface Temperature on COTS outbreak trigger (linear, unitless).
  PARAMETER(sst_effect);
  // Additional quadratic effect of Sea Surface Temperature on COTS outbreak trigger (unitless).
  PARAMETER(sst_effect2);
  
  // Small constant to ensure numerical stability.
  Type eps = Type(1e-8);
  
  // Check for NA parameters and set defaults if necessary.
  if(CppAD::isnan(log_growth_rate_cots)) log_growth_rate_cots = Type(0.0);
  if(CppAD::isnan(log_decline_rate_cots)) log_decline_rate_cots = Type(0.0);
  
  // Transformation from log-space for growth parameters.
  Type growth_rate_cots = exp(log_growth_rate_cots);      // year^-1
  Type decline_rate_cots = exp(log_decline_rate_cots);      // year^-1
  Type intrinsic_coral = exp(log_intrinsic_coral);          // year^-1
  
  // Number of time steps.
  int n = time.size();
  
  // Vectors to store model predictions (_pred) for each variable.
  vector<Type> cots_pred(n);     // Predicted COTS abundance (individuals/m^2)
  vector<Type> fast_pred(n);     // Predicted fast-growing coral cover (%)
  vector<Type> slow_pred(n);     // Predicted slow-growing coral cover (%)
  
  // Initialize predictions with the first observed values.
  cots_pred(0) = cots_dat(0);     // Initialization from data.
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Negative log-likelihood accumulator.
  Type nll = 0.0;
  
  // Iterate over each time step (t >= 1) using previous time step values.
  for(int t = 1; t < n; t++){
    // Equation 1: COTS Dynamics
    // [1] cots_pred[t] = cots_pred[t-1] + growth - decline
    // Growth: growth_rate_cots * cots[t-1] modulated by SST effect, plus larval immigration.
    // Decline: density-dependent decline scaled by decline_rate_cots.
    Type cots_growth = growth_rate_cots * cots_pred(t-1) * (Type(1) + sst_effect * (sst_dat(t-1) - Type(26.0)) + sst_effect2 * pow((sst_dat(t-1) - Type(26.0)), 2)) + cotsimm_dat(t-1); // year^-1 effect
    Type cots_decline = decline_rate_cots * cots_pred(t-1) * cots_pred(t-1);  // density-dependent mortality
    Type cots_pred_raw = cots_pred(t-1) + (cots_growth - cots_decline);
    cots_pred(t) = softplus(cots_pred_raw) + eps;
    
    // Equation 2: Fast Coral Dynamics
    // [2] fast_pred[t] = fast_pred[t-1] + growth - predation loss
    // Growth is logistic with intrinsic_coral and carrying capacity assumed to be 100% cover.
    // Predation loss uses a saturating functional response.
    Type predation_fast = efficiency_predation_fast * cots_pred(t-1) / (threshold_coral + fast_pred(t-1) + eps);
    Type fast_pred_raw = fast_pred(t-1) + intrinsic_coral * fast_pred(t-1) * (Type(1) - fast_pred(t-1) / Type(100.0)) - predation_fast;
    fast_pred(t) = softplus(fast_pred_raw) + eps;
    
    // Equation 3: Slow Coral Dynamics
    // [3] slow_pred[t] = slow_pred[t-1] + growth - predation loss
    // Similar process to fast coral but with a different efficiency parameter.
    Type predation_slow = efficiency_predation_slow * cots_pred(t-1) / (threshold_coral + slow_pred(t-1) + eps);
    Type slow_pred_raw = slow_pred(t-1) + intrinsic_coral * slow_pred(t-1) * (Type(1) - slow_pred(t-1) / Type(100.0)) - predation_slow;
    slow_pred(t) = softplus(slow_pred_raw) + eps;
    
    // Likelihood Calculation using a lognormal error distribution (observations are strictly positive)
    // A fixed minimum standard deviation is used for numerical stability.
    Type sd_cots = Type(1.0);
    Type sd_fast = Type(1.0);
    Type sd_slow = Type(1.0);
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sd_cots, true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sd_fast, true);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sd_slow, true);
  }
  
  // Smooth penalties to bound predation efficiency parameters around 0.5 (with penalty width 0.25).
  nll += pow((efficiency_predation_fast - Type(0.5)) / Type(0.25), 2);
  nll += pow((efficiency_predation_slow - Type(0.5)) / Type(0.25), 2);
  
  // REPORT model predictions for later inspection.
  REPORT(cots_pred);   // Report predicted series for COTS abundance.
  REPORT(fast_pred);   // Report predicted series for fast-growing coral cover.
  REPORT(slow_pred);   // Report predicted series for slow-growing coral cover.
  
  return nll;
}
