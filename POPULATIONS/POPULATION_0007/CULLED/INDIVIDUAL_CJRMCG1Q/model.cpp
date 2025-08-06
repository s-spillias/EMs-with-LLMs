#include <TMB.hpp>

// Template Model Builder model for modeling episodic outbreaks of Crown-of-Thorns Starfish (COTS)
// on the Great Barrier Reef with selective predation on coral communities.
// Equation Overview:
// 1. COTS density update: Exponential growth modulated by environmental forcing and an outbreak boost when density exceeds a threshold.
// 2. Fast-growing coral dynamics: Decline due to saturating predation by COTS.
// 3. Slow-growing coral dynamics: Decline simulated using threshold-like functional responses.
// 4. Likelihood: Lognormal error structure for all observations, employing small constants for numerical stability.

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs
  DATA_VECTOR(time);               // Time variable (e.g., year)
  DATA_VECTOR(cots_dat);           // Observed COTS density (individuals/m2)
  DATA_VECTOR(fast_dat);           // Observed fast-growing coral cover (%; Acropora)
  DATA_VECTOR(slow_dat);           // Observed slow-growing coral cover (%; Faviidae/Porites)
  DATA_VECTOR(sst_dat);            // Sea-surface temperature (°C), environmental driver
  DATA_VECTOR(cotsimm_dat);        // Observed larval immigration rate (individuals/m2/year)

  // Model parameters (with comments on units and origin)
  PARAMETER(growth_rate_COTS);           // Intrinsic growth rate of COTS (year^-1)
  PARAMETER(outbreak_trigger_threshold); // COTS density threshold for outbreak initiation (individuals/m2)
  PARAMETER(coral_consumption_rate);     // Rate of coral loss due to COTS predation (year^-1)
  PARAMETER(environment_modulation);     // Environmental modulation factor for COTS growth (unitless)
  PARAMETER(small_constant);             // Small constant to prevent division by zero (unitless)

  int n = time.size();
  vector<Type> cots_pred(n);  // Predicted COTS density (individuals/m2)
  vector<Type> fast_pred(n);  // Predicted fast-growing coral cover (%)
  vector<Type> slow_pred(n);  // Predicted slow-growing coral cover (%)
  
  Type nll = 0.0;  // Negative log likelihood

  // Initialize predictions at time=0 using initial observations
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Time series loop: predictions based solely on previous time step values to avoid data leakage.
  for (int t = 1; t < n; t++){
    // Equation 1: COTS density update
    // 1. Calculate effective growth rate with an outbreak boost if previous density > outbreak_trigger_threshold.
    Type effective_growth = growth_rate_COTS * environment_modulation;
    effective_growth += (cots_pred(t-1) > outbreak_trigger_threshold ? Type(0.5) : Type(0.0));
    // 2. Update COTS density with additional impact from coral cover (both fast and slow) and numerical stabilization.
    cots_pred(t) = cots_pred(t-1) * exp(effective_growth - coral_consumption_rate * (fast_pred(t-1) + slow_pred(t-1)) + small_constant);

    // Equation 2: Fast-growing coral update
    // Modeled with a saturating functional response to COTS predation.
    fast_pred(t) = fast_pred(t-1) * exp(-coral_consumption_rate * cots_pred(t-1) / (Type(1) + fast_pred(t-1) + small_constant));

    // Equation 3: Slow-growing coral update
    // Modeled with threshold effects in the predation impact.
    slow_pred(t) = slow_pred(t-1) * exp(-coral_consumption_rate * cots_pred(t-1) / (Type(1) + slow_pred(t-1) + small_constant));

    // Equation 4-6: Likelihood contributions using a lognormal error model.
    nll -= dnorm(log(cots_dat(t) + small_constant), log(cots_pred(t) + small_constant), Type(0.1), true);
    nll -= dnorm(log(fast_dat(t) + small_constant), log(fast_pred(t) + small_constant), Type(0.1), true);
    nll -= dnorm(log(slow_dat(t) + small_constant), log(slow_pred(t) + small_constant), Type(0.1), true);
  }

  // Report predicted time series for downstream analysis with the '_pred' suffix.
  REPORT(cots_pred);  // (1) COTS density prediction time series.
  REPORT(fast_pred);  // (2) Fast-growing coral prediction time series.
  REPORT(slow_pred);  // (3) Slow-growing coral prediction time series.

  return nll;
}
