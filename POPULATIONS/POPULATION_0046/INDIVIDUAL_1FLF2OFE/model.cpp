#include <TMB.hpp>

// TMB model for simulating outbreak dynamics of COTS on the Great Barrier Reef.
// Equations description:
// 1. COTS growth follows a saturating function to capture resource limitation.
// 2. Mortality is proportional to current COTS abundance.
// 3. Outbreak triggers use a smooth threshold based on SST and larval immigration.
// 4. Coral cover declines due to COTS predation using saturating functional responses.
// 5. Likelihood is based on lognormal error models for all observations,
//    using predictions from the previous time step to avoid data leakage.
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs:
  DATA_VECTOR(time);               // Time (years)
  DATA_VECTOR(cots_dat);           // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);           // Observed fast-growing coral cover (%) (e.g., Acropora spp.)
  DATA_VECTOR(slow_dat);           // Observed slow-growing coral cover (%) (e.g., Faviidae spp. and Porites spp.)
  DATA_VECTOR(sst_dat);            // Sea Surface Temperature (°C)
  DATA_VECTOR(cotsimm_dat);        // COTS larval immigration rate (individuals/m2/year)

  // Parameter definitions:
  PARAMETER(log_growth_rate);         // Log intrinsic growth rate (year^-1) for COTS
  PARAMETER(log_mortality_rate);      // Log mortality rate (year^-1) for COTS
  PARAMETER(log_predation_efficiency); // Log predation efficiency affecting coral cover (unitless)
  PARAMETER(b0);                      // Baseline threshold for outbreak trigger (unitless)
  PARAMETER(b1);                      // Sensitivity to SST in outbreak trigger (unitless)

  // Transform parameters to their natural scale
  Type growth_rate = exp(log_growth_rate);
  Type mortality_rate = exp(log_mortality_rate);
  Type pred_eff = exp(log_predation_efficiency);
  Type one = Type(1.0);
  Type epsilon = Type(1e-8); // Small constant for numerical stability

  int n = time.size();
  // Prediction arrays
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Initialize predictions with first observed values
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Negative log-likelihood
  Type nll = Type(0.0);

  // Loop through time steps (using values from previous steps only)
  for(int t = 0; t < n - 1; t++){
    // (1) Compute environmental effect using a smooth threshold (logistic function)
    Type env_effect = one / (one + exp(-(b0 + b1 * sst_dat[t]))); // Range (0,1)
    // (2) Outbreak trigger scaled by larval immigration rate
    Type outbreak_trigger = env_effect * cotsimm_dat[t];

    // (3) Saturating growth for COTS using a Michaelis-Menten-like function
    Type growth = growth_rate * cots_pred(t) / (one + epsilon * cots_pred(t));
    // (4) Mortality is proportional to current abundance
    Type mortality = mortality_rate * cots_pred(t);

    // (5) Update COTS abundance incorporating growth, outbreak trigger, and mortality
    cots_pred(t+1) = cots_pred(t) + growth + outbreak_trigger - mortality;

    // (6) Coral dynamics: reduction in coral cover due to predation by COTS with a saturating response
    fast_pred(t+1) = fast_pred(t) - pred_eff * cots_pred(t) * fast_pred(t) / (one + epsilon * fast_pred(t));
    slow_pred(t+1) = slow_pred(t) - pred_eff * cots_pred(t) * slow_pred(t) / (one + epsilon * slow_pred(t));

    // (7) Likelihood contributions using lognormal error distributions; sigma fixed to 1.0 for numerical stability
    Type sigma = one;
    nll -= dlnorm(cots_dat[t+1], log(cots_pred(t+1) + epsilon), sigma, true);
    nll -= dlnorm(fast_dat[t+1], log(fast_pred(t+1) + epsilon), sigma, true);
    nll -= dlnorm(slow_dat[t+1], log(slow_pred(t+1) + epsilon), sigma, true);
  }

  // Reporting predicted values for diagnostics
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}
