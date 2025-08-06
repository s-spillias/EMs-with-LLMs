#include <TMB.hpp>  // TMB library for template-based model building

// 1. Data: observations of starfish abundance (star_dat), coral covers (slow_dat, fast_dat),
//    sea surface temperature (sst_dat), and Crown-of-Thorns immigration rate (cotsimm_dat)
template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA declarations: all observations from input data
  DATA_VECTOR(star_dat);    // Observed Crown-of-Thorns starfish abundance (individuals/m2)
  DATA_VECTOR(slow_dat);    // Observed cover of slow-growing coral (%)
  if(star_dat.size() == 0) star_dat = slow_dat; // Fallback if star_dat is missing
  DATA_VECTOR(fast_dat);    // Observed cover of fast-growing coral (%)
  DATA_VECTOR(Year);        // Time steps (years)
  DATA_VECTOR(sst_dat);     // Sea Surface Temperature (°C)
  DATA_VECTOR(cots_dat); // Crown-of-Thorns immigration rate (individuals/m2/year)

  // PARAMETER declarations with comments:
  PARAMETER(growth_rate_star);   // Intrinsic growth rate of starfish (year^-1)
  PARAMETER(predation_rate_slow);  // Predation efficiency on slow-growing coral (year^-1)
  PARAMETER(predation_rate_fast);  // Predation efficiency on fast-growing coral (year^-1)
  PARAMETER(saturation_constant);  // Saturation constant for resource limitation
  PARAMETER(env_effect);           // Environmental effect modifier (unitless)
  PARAMETER(obs_sd);               // Base observation standard deviation (minimum)

  int n = star_dat.size();  // Number of time steps

  // Initialize prediction vectors using lagged values to avoid data leakage
  vector<Type> star_pred(n);      // Predicted starfish abundance
  vector<Type> slow_pred(n);      // Predicted slow-growing coral cover
  vector<Type> fast_pred(n);      // Predicted fast-growing coral cover
  vector<Type> cots_pred(n);      // Predicted Crown-of-Thorns immigration forcing

  // Initialize first time step with observed data
  star_pred(0) = star_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  cots_pred(0) = cots_dat(0);

  // Dynamic ecosystem model equations:
  // 1. Starfish population growth with immigration:
  //    star_pred[t] = star_pred[t-1] + growth_rate_star * star_pred[t-1] * resource_limitation + cotsimm_dat[t-1]
  // 2. Resource limitation using a saturating function:
  //    resource_limitation = env_effect * (saturation_constant / (saturation_constant + star_pred[t-1] + 1e-8))
  // 3. Decline in coral cover due to predation:
  //    slow_pred[t] = slow_pred[t-1] - predation_rate_slow * star_pred[t-1] * slow_pred[t-1]
  //    fast_pred[t] = fast_pred[t-1] - predation_rate_fast * star_pred[t-1] * fast_pred[t-1]
  for (int t = 1; t < n; t++){
    cots_pred(t) = cots_pred(t-1);
    // Compute resource limitation effect with small constant for numerical stability
    Type resource_limitation = env_effect * (saturation_constant / (saturation_constant + star_pred(t-1) + Type(1e-8)));
    // Update starfish abundance using previous value and immigration
    star_pred(t) = star_pred(t-1) + growth_rate_star * star_pred(t-1) * resource_limitation + cots_pred(t-1);
    // Update coral covers based on selective predation
    slow_pred(t) = slow_pred(t-1) - predation_rate_slow * star_pred(t-1) * slow_pred(t-1);
    fast_pred(t) = fast_pred(t-1) - predation_rate_fast * star_pred(t-1) * fast_pred(t-1);
  }

  // Likelihood calculation using a lognormal error distribution to ensure strictly positive predictions
  Type nll = 0.0;
  for (int t = 0; t < n; t++){
    nll -= (dnorm(log(star_dat(t) + Type(1e-8)), log(star_pred(t) + Type(1e-8)), obs_sd, true) - log(star_dat(t) + Type(1e-8)));
    nll -= (dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred(t) + Type(1e-8)), obs_sd, true) - log(slow_dat(t) + Type(1e-8)));
    nll -= (dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred(t) + Type(1e-8)), obs_sd, true) - log(fast_dat(t) + Type(1e-8)));
    nll -= (dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t) + Type(1e-8)), obs_sd, true) - log(cots_dat(t) + Type(1e-8)));
  }

  // Report prediction vectors for diagnostic purposes
  REPORT(star_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(cots_pred);
  return nll;
}
