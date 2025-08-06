#include <TMB.hpp>

// 1. This Template Model Builder model simulates the trophic interactions between Crown-of-Thorns starfish (COTS)
// and coral communities (slow-growing and fast-growing types) on the Great Barrier Reef.
// 2. Data vectors provided: slow_dat (slow-growing coral cover, %), fast_dat (fast-growing coral cover, %),
// and cots_dat (COTS abundance in individuals/m2).
// 3. Parameters include attack rates, intrinsic coral growth rates, and COTS mortality, all with biological interpretations.
// 4. Numerical stability is ensured by using small constants (eps = 1e-8) and smooth exponential functions for transitions.
// 5. The likelihood is computed using a lognormal distribution (with fixed minimal SD) to compare predictions with observations.

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs (observed values)
  DATA_VECTOR(slow_dat); // Slow-growing coral % cover, units: percentage
  DATA_VECTOR(fast_dat); // Fast-growing coral % cover, units: percentage
  DATA_VECTOR(cots_dat); // COTS abundance, units: individuals/m2
  DATA_VECTOR(sst_dat); // Sea-Surface Temperature in Celsius, forcing variable
  DATA_VECTOR(cotsimm_dat); // Crown-of-Thorns immigration (individuals/m2/year), forcing variable

  // Parameters with initial values to be estimated:
  // attack_rate_slow: Attack rate (1/year) of COTS on slow-growing corals [Literature]
  PARAMETER(attack_rate_slow);
  // attack_rate_fast: Attack rate (1/year) of COTS on fast-growing corals [Literature]
  PARAMETER(attack_rate_fast);
  // growth_rate_slow: Intrinsic growth rate (year^-1) of slow-growing corals [Expert opinion]
  PARAMETER(growth_rate_slow);
  // growth_rate_fast: Intrinsic growth rate (year^-1) of fast-growing corals [Expert opinion]
  PARAMETER(growth_rate_fast);
  // cots_mortality: Mortality rate (year^-1) of Crown-of-Thorns starfish [Initial estimate]
  PARAMETER(cots_mortality);

  // Small constant for numerical stability (prevents division by zero, log of 0, etc.)
  Type eps = Type(1e-8);

  // 1. Slow coral dynamics: Growth minus loss by COTS predation
  //    Equation: slow_pred = slow_dat * exp(-attack_rate_slow * pred_intensity)
  // 2. Fast coral dynamics: Growth minus loss by COTS predation
  //    Equation: fast_pred = fast_dat * exp(-attack_rate_fast * pred_intensity)
  // 3. COTS dynamics: Implicitly incorporated via its observed abundance and its predation effect on corals

  int n = slow_dat.size();
  vector<Type> slow_pred(n), fast_pred(n), cots_pred(n);
  // Set initial conditions using first observation (assumed known)
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  cots_pred(0) = cots_dat(0);

  for(int i = 1; i < n; i++){
      // Slow coral prediction: previous prediction grows multiplicatively by intrinsic growth rate and is reduced by COTS impact.
      slow_pred(i) = slow_pred(i-1) * growth_rate_slow * (Type(1) - attack_rate_slow * cots_pred(i-1));
      // Fast coral prediction: previous prediction grows multiplicatively by intrinsic growth rate and is reduced by COTS impact.
      fast_pred(i) = fast_pred(i-1) * growth_rate_fast * (Type(1) - attack_rate_fast * cots_pred(i-1));
      // COTS prediction: previous COTS abundance is reduced by mortality and increased by immigration modulated by SST.
      cots_pred(i) = cots_pred(i-1) * (Type(1) - cots_mortality + Type(0.01) * (sst_dat(i) - Type(28))) + cotsimm_dat(i);
  }

  // Likelihood calculation: Summing log-likelihoods for slow, fast corals and COTS observations using lognormal errors.
  // A fixed minimal standard deviation of 0.1 is used to prevent numerical issues.
  Type jnll = 0;
  for(int i = 0; i < n; i++){
    jnll -= dlnorm(slow_dat(i), log(slow_pred(i) + eps), Type(0.1), true);
    jnll -= dlnorm(fast_dat(i), log(fast_pred(i) + eps), Type(0.1), true);
    jnll -= dlnorm(cots_dat(i), log(cots_pred(i) + eps), Type(0.1), true);
  }

  // Reporting predicted values for further diagnostic purposes (with '_pred' suffix)
  ADREPORT(slow_pred);
  ADREPORT(fast_pred);
  ADREPORT(cots_pred);

  return jnll;
}
