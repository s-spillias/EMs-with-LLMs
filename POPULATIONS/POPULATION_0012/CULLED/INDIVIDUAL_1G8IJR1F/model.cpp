#include <TMB.hpp>  // TMB header: Provides automatic differentiation and statistical routines

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs:
  DATA_VECTOR(Year);            // Time (year) extracted from the forcing file
  DATA_VECTOR(sst_dat);         // Sea-surface temperature (°C) affecting ecological rates
  DATA_VECTOR(cotsimm_dat);     // Immigration rates of COTS (ind/m^2/year)
  DATA_VECTOR(cots_dat);        // Observed adult COTS abundance (ind/m^2)
  DATA_VECTOR(fast_dat);        // Observed fast-growing coral cover (%) (Acropora spp.)
  DATA_VECTOR(slow_dat);        // Observed slow-growing coral cover (%) (Faviidae/Porites spp.)

  // Parameters to be estimated:
  PARAMETER(growth_rate);       // Intrinsic COTS growth rate (year^-1)
  PARAMETER(carrying_capacity); // Carrying capacity for COTS (ind/m^2)
  PARAMETER(predation_rate);    // Rate of COTS decline due to coral predation (year^-1)
  PARAMETER(efficiency);        // Feeding efficiency modifying predation pressure (dimensionless)
  PARAMETER(log_sigma);         // Log of observation error SD (for all likelihood components)
  PARAMETER(log_initial_cots);  // Log initial COTS abundance (ind/m^2)
  PARAMETER(log_initial_fast);  // Log initial fast-growing coral cover (%) 
  PARAMETER(log_initial_slow);  // Log initial slow-growing coral cover (%) 

  // Transform parameters:
  Type sigma = exp(log_sigma) + Type(1e-4);  // Observation error standard deviation (ensures non-zero)

  int n = Year.size();

  // State predictions vector (all indices correspond to time step t)
  vector<Type> cots_pred(n), fast_pred(n), slow_pred(n);

  // Initialize states using log-transformed initial conditions:
  cots_pred[0] = exp(log_initial_cots); // (1) COTS initial state
  fast_pred[0] = exp(log_initial_fast); // (2) Initial fast-growing coral cover (%)
  slow_pred[0] = exp(log_initial_slow); // (3) Initial slow-growing coral cover (%)

  // Initialize negative log likelihood:
  Type jnll = 0.0;

  // -------------------------------------------------------------------------
  // Model Equations:
  // 1. COTS dynamics: Logistic growth modified by predation.
  //    cots[t+1] = cots[t] + growth_rate * cots[t] * (1 - cots[t]/carrying_capacity)
  //                - predation_rate * cots[t] * ( (fast[t]+slow[t])/(1+efficiency*(fast[t]+slow[t])) )
  // 2. Fast-growing coral dynamics: Decline due to COTS predation and regrowth.
  //    fast[t+1] = fast[t] - 0.1 * cots[t] * fast[t] / (fast[t]+1e-8) + 0.05*(100-fast[t])
  // 3. Slow-growing coral dynamics: Similar structure with lower predation sensitivity.
  //    slow[t+1] = slow[t] - 0.05 * cots[t] * slow[t] / (slow[t]+1e-8) + 0.03*(100-slow[t])
  // -------------------------------------------------------------------------
  for(int t = 1; t < n; t++) {
    // COTS update equation:
    cots_pred[t] = cots_pred[t-1]
                     + growth_rate * cots_pred[t-1] * (1 - cots_pred[t-1] / (carrying_capacity + Type(1e-8)))
                     - predation_rate * cots_pred[t-1]
                       * ((fast_pred[t-1] + slow_pred[t-1]) / (1 + efficiency * (fast_pred[t-1] + slow_pred[t-1]) + Type(1e-8)));

    // Fast-growing coral update:
    fast_pred[t] = fast_pred[t-1]
                     - 0.1 * cots_pred[t-1] * fast_pred[t-1] / (fast_pred[t-1] + Type(1e-8))
                     + 0.05 * (Type(100) - fast_pred[t-1]); // Regrowth towards 100% cover

    // Slow-growing coral update:
    slow_pred[t] = slow_pred[t-1]
                     - 0.05 * cots_pred[t-1] * slow_pred[t-1] / (slow_pred[t-1] + Type(1e-8))
                     + 0.03 * (Type(100) - slow_pred[t-1]); // Regrowth towards 100% cover

    // Likelihood contributions:
    // Using log-transform of data to accommodate wide range and ensure positivity.
    // Fixed minimum standard deviation 'sigma' ensures numerical stability.
    jnll -= dnorm(log(cots_dat[t] + Type(1e-8)),
                  log(cots_pred[t] + Type(1e-8)),
                  sigma, true);
    jnll -= dnorm(log(fast_dat[t] + Type(1e-8)),
                  log(fast_pred[t] + Type(1e-8)),
                  sigma, true);
    jnll -= dnorm(log(slow_dat[t] + Type(1e-8)),
                  log(slow_pred[t] + Type(1e-8)),
                  sigma, true);
  }

  // Reporting model predictions matching observation names (_dat):
  REPORT(cots_pred);  // Report predicted COTS (ind/m^2)
  REPORT(fast_pred);  // Report predicted fast-growing coral cover (%) for Acropora spp.
  REPORT(slow_pred);  // Report predicted slow-growing coral cover (%) for Faviidae/Porites spp.

  return jnll;  // Return total negative log likelihood
}
