#include <TMB.hpp>

// Template Model Builder model for Crown-of-Thorns Starfish dynamics
// on the Great Barrier Reef affecting slow-growing (Faviidae/Porites) and fast-growing (Acropora) corals.
template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS:
  // slow_dat: observed slow-growing coral cover (%) [data]
  // fast_dat: observed fast-growing coral cover (%) [data]
  // cots_dat: observed crown-of-thorns starfish density (ind/m2) [data]
  // sst_dat: observed Sea-Surface Temperature (°C) (provided for potential environmental effects)
  DATA_VECTOR(slow_dat);
  DATA_VECTOR(fast_dat);
  DATA_VECTOR(cots_dat);
  DATA_VECTOR(sst_dat);
  DATA_VECTOR(cotsimm_dat);

  // PARAMETERS:
  // growth_rate_COTS (year^-1): intrinsic growth rate of COTS
  // consumption_rate_slow (per individual per year): feeding rate on slow-growing corals
  // consumption_rate_fast (per individual per year): feeding rate on fast-growing corals
  // mortality_rate_COTS (year^-1): mortality rate of COTS
  // carrying_capacity (ind/m2): maximum sustainable COTS density
  // log_sigma_slow: logarithm of the observational standard deviation for slow coral cover
  // log_sigma_fast: logarithm of the observational standard deviation for fast coral cover
  PARAMETER(growth_rate_COTS);
  PARAMETER(consumption_rate_slow);
  PARAMETER(consumption_rate_fast);
  PARAMETER(mortality_rate_COTS);
  PARAMETER(carrying_capacity);
  PARAMETER(log_sigma_slow);
  PARAMETER(log_sigma_fast);
  PARAMETER(cots_init); // initial COTS density (ind/m2)
  PARAMETER(slow_init); // initial slow-growing coral cover (%) 
  PARAMETER(fast_init); // initial fast-growing coral cover (%)
  PARAMETER(growth_rate_slow); // intrinsic growth rate of slow-growing corals (year^-1)
  PARAMETER(growth_rate_fast); // intrinsic growth rate of fast-growing corals (year^-1)
  PARAMETER(theta_sst); // effect of sea surface temperature (°C) on COTS growth

  // Transform observation error standard deviations ensuring a fixed minimum value for stability.
  Type sigma_slow = exp(log_sigma_slow) + Type(1e-8);
  Type sigma_fast = exp(log_sigma_fast) + Type(1e-8);

  // Initialize the negative log likelihood.
  Type nll = 0.0;

  // 1. Equation: Smooth penalties to bound parameters within biologically meaningful ranges.
  //    (A quadratic penalty is added if parameters deviate from expected bounds.)
  nll += square(max(Type(0), fabs(growth_rate_COTS) - Type(2))) * Type(1e-4); // growth_rate_COTS: expected within [-2, 2] year^-1
  nll += square(max(Type(0), consumption_rate_slow - Type(1.0))) * Type(1e-4);  // consumption_rate_slow: expected < 1 per year
  nll += square(max(Type(0), consumption_rate_fast - Type(1.0))) * Type(1e-4);  // consumption_rate_fast: expected < 1 per year
  nll += square(max(Type(0), mortality_rate_COTS - Type(1.0))) * Type(1e-4);    // mortality_rate_COTS: expected < 1 year^-1
  nll += square(max(Type(0), carrying_capacity - Type(5.0))) * Type(1e-4);      // carrying_capacity: expected < 5 ind/m2

  // Number of observed data points.
  int n = slow_dat.size();

  // Vectors to store model predictions for response variables.
  vector<Type> slow_pred(n), fast_pred(n), cots_pred(n);

  // 2. Equations: Recursive model predictions based on external forcing and previous predictions.
  // Initialize predictions using parameter initial conditions:
  slow_pred(0) = slow_init;   // initial slow-growing coral cover (%)
  fast_pred(0) = fast_init;   // initial fast-growing coral cover (%)
  cots_pred(0) = cots_init;   // initial COTS density (ind/m2)

  // For i >= 1, update predictions using previous predictions and external forcing.
  for(int i = 1; i < n; i++){
    // Equation 1: COTS dynamics: multiplicative update using previous prediction and external forcing.
    cots_pred(i) = cots_pred(i-1) * (1 + growth_rate_COTS * (1 - cots_pred(i-1) / carrying_capacity) - mortality_rate_COTS 
                    + theta_sst * sst_dat(i) + cotsimm_dat(i) / (cots_pred(i-1) + Type(1e-8)));
    
    // Equation 2: Slow-growing coral cover dynamics: multiplicative update incorporating intrinsic growth and predation.
    slow_pred(i) = slow_pred(i-1) * ((1 + growth_rate_slow) * (1 - consumption_rate_slow * cots_pred(i-1)));
    
    // Equation 3: Fast-growing coral cover dynamics: multiplicative update incorporating intrinsic growth and predation.
    fast_pred(i) = fast_pred(i-1) * ((1 + growth_rate_fast) * (1 - consumption_rate_fast * cots_pred(i-1)));
  }

  // 3. Likelihood Calculation:
  //    Use lognormal likelihoods to account for the strictly positive coral cover measurements.
  //    A fixed minimum standard deviation is ensured by sigma_slow and sigma_fast.
  for(int i = 0; i < n; i++){
    nll -= dlnorm(slow_dat(i) + Type(1e-8), log(slow_pred(i) + Type(1e-8)), sigma_slow, true);
    nll -= dlnorm(fast_dat(i) + Type(1e-8), log(fast_pred(i) + Type(1e-8)), sigma_fast, true);
  }

  // 4. Reporting variables for further diagnostics and inspection.
  REPORT(slow_pred);  // Predicted slow-growing coral cover for each observation.
  REPORT(fast_pred);  // Predicted fast-growing coral cover for each observation.
  REPORT(cots_pred);  // Predicted COTS density for each observation.
  REPORT(sigma_slow); // Transformed observation error for slow coral data.
  REPORT(sigma_fast); // Transformed observation error for fast coral data.

  /*
    Equation descriptions:
    1. Logistic mitigation of coral cover reduction as COTS density approaches carrying capacity.
    2. Observation likelihood modeled using lognormal distributions with minimal standard deviations to avoid numerical issues.
    3. Smooth penalty functions ensure parameters remain biologically plausible without hard constraints.
  */

  return nll;
}
