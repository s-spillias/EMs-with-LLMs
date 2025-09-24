#include <TMB.hpp>
template<class Type>
Type logistic(Type x, Type x0, Type k) {
  // Logistic function for smooth threshold; x0 = threshold, k = steepness
  return Type(1.0) / (Type(1.0) + exp(-k * (x - x0)));
}

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;
  // Data vectors (observations)
  DATA_VECTOR(Year); // Years corresponding to observations
  DATA_VECTOR(cots_dat); // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat); // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat); // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat); // Sea-surface temperature (°C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration rate (individuals/m2/year)

  int n = cots_dat.size();   // number of time steps
  Type eps = Type(1e-8);     // Numerical stability constant

  // Parameters for COTS dynamics
  PARAMETER(cots_growth_rate);   // Intrinsic growth rate of COTS (year^-1)
  PARAMETER(cots_decline_rate);  // Decline rate of COTS (year^-1) post outbreak
  PARAMETER(cots_threshold);     // COTS density threshold triggering outbreak (individuals/m2)
  PARAMETER(outbreak_k);         // Steepness of outbreak trigger (unitless)
  PARAMETER(K);                  // Carrying capacity for COTS (individuals/m2)
  PARAMETER(sst_effect);         // Effect of SST deviation on growth rate (per °C)
  PARAMETER(sst_baseline);       // Baseline SST (°C)

  // Parameters for coral predation impacts
  PARAMETER(predation_eff_fast); // Efficiency of COTS predation on fast-growing coral (unitless)
  PARAMETER(predation_eff_slow); // Efficiency of COTS predation on slow-growing coral (unitless)
  PARAMETER(res_half_sat);       // Half-saturation constant for coral (in % cover)

  // Observation error standard deviations (log-transformed data assumed)
  PARAMETER(sigma_cots);         // Standard deviation for COTS abundance likelihood (log-scale)
  PARAMETER(sigma_fast);         // Standard deviation for fast coral likelihood (log-scale)
  PARAMETER(sigma_slow);         // Standard deviation for slow coral likelihood (log-scale)

  // Vectors to store predicted states
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // 1. Initialization using first observations
  cots_pred(0) = cots_dat(0) + eps;
  fast_pred(0) = fast_dat(0) + eps;
  slow_pred(0) = slow_dat(0) + eps;

  // Negative log-likelihood
  Type nll = 0.0;

  // Initial likelihood contributions using initial predicted states at t = 0
  nll -= dlnorm(cots_dat(0), log(cots_pred(0)), sigma_cots, true);
  nll -= dlnorm(fast_dat(0), log(fast_pred(0)), sigma_fast, true);
  nll -= dlnorm(slow_dat(0), log(slow_pred(0)), sigma_slow, true);

  // Numbered list of equations:
  // (1) COTS dynamics:
  //     cots_pred[t] = cots_pred[t-1] + (growth + forcing - decline) where:
  //         growth = cots_growth_rate * cots_pred[t-1] * (1 - cots_pred[t-1] / K) * (1 + sst_effect * (sst_dat[t-1] - sst_baseline))
  //         outbreak = logistic(cots_pred[t-1], cots_threshold, outbreak_k)
  //         forcing = cotsimm_dat[t-1]
  //         decline = cots_decline_rate * cots_pred[t-1] * outbreak
  // (2) Coral dynamics:
  //     fast_pred[t] = fast_pred[t-1] - predation_eff_fast * cots_pred[t-1] * fast_pred[t-1] / (res_half_sat + fast_pred[t-1])
  //     slow_pred[t] = slow_pred[t-1] - predation_eff_slow * cots_pred[t-1] * slow_pred[t-1] / (res_half_sat + slow_pred[t-1])
  //   Note: Only previous states are used for prediction to avoid data leakage.
  for(int t = 1; t < n; t++){
    // Compute outbreak effect using previous predicted COTS value
    Type outbreak_effect = logistic(cots_pred(t-1), cots_threshold, outbreak_k);
    // (1) Update COTS dynamics using prediction from t-1
    Type growth = cots_growth_rate * cots_pred(t-1) * (1 - cots_pred(t-1) / K) * (1 + sst_effect * (sst_dat(t-1) - sst_baseline));
    Type forcing = cotsimm_dat(t-1);
    Type decline = cots_decline_rate * cots_pred(t-1) * outbreak_effect;
    cots_pred(t) = cots_pred(t-1) + growth + forcing - decline;
    cots_pred(t) = fmax(cots_pred(t), eps);

    // (2) Update Coral dynamics using predictions from t-1
    fast_pred(t) = fast_pred(t-1) - predation_eff_fast * cots_pred(t-1) * fast_pred(t-1) / (res_half_sat + fast_pred(t-1));
    slow_pred(t) = slow_pred(t-1) - predation_eff_slow * cots_pred(t-1) * slow_pred(t-1) / (res_half_sat + slow_pred(t-1));
    fast_pred(t) = fmax(fast_pred(t), eps);
    slow_pred(t) = fmax(slow_pred(t), eps);

    // Likelihood contributions using lognormal likelihoods for predictions at time t
    nll -= dlnorm(cots_dat(t), log(cots_pred(t)), sigma_cots, true);
    nll -= dlnorm(fast_dat(t), log(fast_pred(t)), sigma_fast, true);
    nll -= dlnorm(slow_dat(t), log(slow_pred(t)), sigma_slow, true);
  }

  // Reporting predicted trajectories
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
