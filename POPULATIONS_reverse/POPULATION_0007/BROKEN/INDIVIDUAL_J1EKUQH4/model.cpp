#include <TMB.hpp>
using namespace density;

template<class Type>
Type dlnorm(Type x, Type meanlog, Type sd, int give_log=1) {
    // Prevent x from being zero or negative to avoid log domain errors.
    Type x_adj = (x < Type(1e-8)) ? Type(1e-8) : x;
    Type logpdf = -log(x_adj) + dnorm(log(x_adj), meanlog, sd, true);
    return (give_log ? logpdf : exp(logpdf));
}
template<class Type>
Type my_max(Type a, Type b) {
  return a > b ? a : b;
}

// 1. This function computes the negative log-likelihood (nll) for the Crown-of-Thorns starfish and coral model.
// 2. The model accounts for COTS dynamics and coral cover changes by combining baseline levels, growth processes,
//    and predation effects with smooth transitions to ensure numerical stability.
// 3. Likelihoods for observed data (cots_dat, slow_dat, fast_dat) are computed using lognormal error distributions.
template<class Type>
Type objective_function<Type>::operator() () {
  // DATA input observations
  DATA_VECTOR(cots_dat);         // Observed Crown-of-Thorns starfish abundance (individuals/m2)
  DATA_VECTOR(slow_dat);         // Observed slow-growing coral cover (Faviidae & Porites, %)
  DATA_VECTOR(fast_dat);         // Observed fast-growing coral cover (Acropora, %)
  DATA_VECTOR(sst_dat);          // Sea Surface Temperature data (°C)
  DATA_VECTOR(cotsimm_dat);      // COTS immigration rate (individuals/m2/year)

  // PARAMETERS:
  PARAMETER(alpha_cots);         // (individuals/m2) Baseline COTS abundance
  PARAMETER(conversion_rate);    // (unitless) Conversion efficiency from coral consumption to COTS growth
  PARAMETER(consumption_slow);   // (per % cover per individual) Consumption rate of slow-growing corals
  PARAMETER(consumption_fast);   // (per % cover per individual) Consumption rate of fast-growing corals
  PARAMETER(mortality_cots);     // (year^-1) Mortality rate of COTS
  PARAMETER(alpha_slow);         // (%) Baseline slow-growing coral cover
  PARAMETER(growth_slow);        // (year^-1) Growth rate of slow-growing corals
  PARAMETER(alpha_fast);         // (%) Baseline fast-growing coral cover
  PARAMETER(growth_fast);        // (year^-1) Growth rate of fast-growing corals
  PARAMETER(handling);           // (dimensionless) Handling time constant for saturating coral consumption
  PARAMETER(log_sigma_cots);     // Log-scale standard deviation for COTS observations
  PARAMETER(log_sigma_coral);    // Log-scale standard deviation for coral observations

  // Transform error parameters with a small constant for numerical stability
  Type sigma_cots = exp(log_sigma_cots) + Type(1e-8);
  Type sigma_coral = exp(log_sigma_coral) + Type(1e-8);

  int n = cots_dat.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  Type nll = 0.0;

  // Equations:
  // (1) COTS prediction:
  //     cots_pred = alpha_cots + conversion_rate * (consumption_slow * slow_dat + consumption_fast * fast_dat)
  //                 - mortality_cots * cots_dat
  //     => Balances baseline abundance, gains from coral consumption, and losses due to mortality.
  // (2) Slow-growing coral prediction:
  //     slow_pred = alpha_slow + growth_slow * slow_dat - (consumption_slow * cots_dat * slow_dat) / (Type(1) + handling * slow_dat + Type(1e-8))
  //     => Represents baseline cover, growth, and predation losses with a saturating response.
  // (3) Fast-growing coral prediction:
  //     fast_pred = alpha_fast + growth_fast * fast_dat - (consumption_fast * cots_dat * fast_dat) / (Type(1) + handling * fast_dat + Type(1e-8))
  //     => Similar to slow coral with parameters adjusted for fast-growing species.

  for(int i = 0; i < n; i++){
    if(i == 0){
      cots_pred(i) = my_max(alpha_cots + conversion_rate * cotsimm_dat(i), Type(1e-8));  // Initial COTS prediction based on immigration forcing
      slow_pred(i) = my_max(alpha_slow, Type(1e-8));                                     // Baseline slow-growing coral cover
      fast_pred(i) = my_max(alpha_fast, Type(1e-8));                                     // Baseline fast-growing coral cover
    } else {
      cots_pred(i) = my_max(cots_pred(i-1) + conversion_rate * cotsimm_dat(i) - mortality_cots * cots_pred(i-1), Type(1e-8));
      slow_pred(i) = my_max(slow_pred(i-1) * growth_slow * (1 - consumption_slow * cots_pred(i-1) + Type(1e-8)), Type(1e-8));
      fast_pred(i) = my_max(fast_pred(i-1) * growth_fast * (1 - consumption_fast * cots_pred(i-1) + Type(1e-8)), Type(1e-8));
    }
    // Likelihood calculation using lognormal distributions:
    //    - dlnorm(obs, log(pred + 1e-8), sigma, true) ensures strictly positive predictions.
    nll -= dlnorm(cots_dat(i), log(cots_pred(i) + Type(1e-8)), sigma_cots, true);
    nll -= dlnorm(slow_dat(i), log(slow_pred(i) + Type(1e-8)), sigma_coral, true);
    nll -= dlnorm(fast_dat(i), log(fast_pred(i) + Type(1e-8)), sigma_coral, true);
  }

  // Reporting model predictions for comparison with observed data
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);

  return nll;
}
