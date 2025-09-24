#include <TMB.hpp>

// TMB model for simulating Crown-of-Thorns starfish outbreaks on the Great Barrier Reef.
// The model uses time series dynamics with a one-year timestep and includes the following equations:
// 1. COTS dynamics:
//    cots[t] = cots[t-1] + dt * { (r_cots + sst_effect * sst[t-1]) * cots[t-1] * [1 - cots[t-1] / (K_cots + coral_available)] - m_cots * cots[t-1] + cotsimm[t] }
//    where coral_available = (fast[t-1] + slow[t-1])/(h + 1e-8).
// 2. Fast coral dynamics:
//    fast[t] = fast[t-1] + dt * { regen_fast * (max_fast - fast[t-1]) - coral_eff_fast * cots[t-1] * fast[t-1] / (fast[t-1] + 1e-8) }
// 3. Slow coral dynamics:
//    slow[t] = slow[t-1] + dt * { regen_slow * (max_slow - slow[t-1]) - coral_eff_slow * cots[t-1] * slow[t-1] / (slow[t-1] + 1e-8) }

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA inputs (observations)
  DATA_VECTOR(cots_dat);      // COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);      // Fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);      // Slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);       // Sea-surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);   // Larval immigration rate of COTS (individuals/m2/year)

  int n = cots_dat.size();  // number of time steps

  // PARAMETERS for the ecosystem model
  PARAMETER(r_cots);           // Intrinsic growth rate of COTS (year^-1)
  PARAMETER(logK_cots);        // Log of carrying capacity for COTS (log(individuals/m2))
  PARAMETER(m_cots);           // Natural mortality rate of COTS (year^-1)
  PARAMETER(h);                // Half-saturation constant for combined coral availability (unitless)
  PARAMETER(sst_effect);       // Effect of SST on COTS growth (per °C)
  PARAMETER(coral_eff_fast);   // Efficiency of predation on fast-growing coral (unitless)
  PARAMETER(coral_eff_slow);   // Efficiency of predation on slow-growing coral (unitless)
  PARAMETER(regen_fast);       // Regeneration rate for fast-growing coral (%/year)
  PARAMETER(regen_slow);       // Regeneration rate for slow-growing coral (%/year)
  PARAMETER(max_fast);         // Maximum possible fast coral cover (%)
  PARAMETER(max_slow);         // Maximum possible slow coral cover (%)
  PARAMETER(log_sigma_cots);   // Log standard deviation for COTS likelihood
  PARAMETER(log_sigma_fast);   // Log standard deviation for fast coral likelihood
  PARAMETER(log_sigma_slow);   // Log standard deviation for slow coral likelihood

  // Convert log parameters to natural scale
  Type K_cots = exp(logK_cots);  // Carrying capacity for COTS

  // Vectors to hold predictions
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Set initial conditions equal to the first observation to avoid data leakage
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Small constant to avoid division by zero
  Type onee = Type(1e-8);
  // Timestep (1 year)
  Type dt = 1.0;

  // Dynamic model loop: use previous timestep values for predictions
  for(int t = 1; t < n; t++){
    // Compute coral availability using a saturating function
    Type coral_available = (fast_pred(t-1) + slow_pred(t-1)) / (h + onee);

    // Adjusted intrinsic growth rate influenced by SST from previous timestep
    Type r_adj = r_cots + sst_effect * sst_dat(t-1);

    // Equation 1: COTS dynamics including growth, density dependence, natural mortality, and immigration
    cots_pred(t) = cots_pred(t-1) +
      dt * ( r_adj * cots_pred(t-1) * ( 1 - cots_pred(t-1) / (K_cots + coral_available + onee) )
           - m_cots * cots_pred(t-1)
           + cotsimm_dat(t) );

    // Equation 2: Fast coral dynamics with regeneration and COTS predation
    fast_pred(t) = fast_pred(t-1) +
      dt * ( regen_fast * (max_fast - fast_pred(t-1))
           - coral_eff_fast * cots_pred(t-1) * fast_pred(t-1)/(fast_pred(t-1) + onee) );

    // Equation 3: Slow coral dynamics with regeneration and COTS predation
    slow_pred(t) = slow_pred(t-1) +
      dt * ( regen_slow * (max_slow - slow_pred(t-1))
           - coral_eff_slow * cots_pred(t-1) * slow_pred(t-1)/(slow_pred(t-1) + onee) );
  }

  // Likelihood calculation using lognormal errors to accommodate data spanning multiple orders of magnitude
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_fast = exp(log_sigma_fast);
  Type sigma_slow = exp(log_sigma_slow);

  Type nll = 0.0;
  for(int t = 1; t < n; t++){
    nll -= dlnorm(cots_dat(t), log(cots_pred(t) + onee), sigma_cots, true); // (1) COTS likelihood
    nll -= dlnorm(fast_dat(t), log(fast_pred(t) + onee), sigma_fast, true); // (2) Fast coral likelihood
    nll -= dlnorm(slow_dat(t), log(slow_pred(t) + onee), sigma_slow, true); // (3) Slow coral likelihood
  }

  // Reporting model predictions for diagnostic purposes
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}
