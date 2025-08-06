#include <TMB.hpp>

// 1. Data inputs
//    - Year: time variable (years)
//    - cots_dat: observed COTS abundance (individuals/m^2)
//    - fast_dat: observed fast-growing coral cover (%)
//    - slow_dat: observed slow-growing coral cover (%)
//    - sst_dat: Sea-Surface Temperature (°C)
//    - cotsimm_dat: Crown-of-Thorns larval immigration rate (individuals/m^2/year)

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data vectors
  DATA_VECTOR(Year);                // Time (year)
  DATA_VECTOR(cots_dat);            // Observed COTS abundance
  DATA_VECTOR(fast_dat);            // Observed fast-growing Acropora cover
  DATA_VECTOR(slow_dat);            // Observed slow-growing coral cover (Faviidae/Porites)
  DATA_VECTOR(sst_dat);             // Environmental forcing: sea-surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);         // Larval immigration rate

  const int n = cots_dat.size();    // Number of time steps
  
  // 2. Parameters
  PARAMETER(growth_rate);           // Intrinsic growth rate for COTS (year^-1)
  PARAMETER(threshold_outbreak);    // Larval immigration threshold triggering outbreak (individuals/m^2/year)
  PARAMETER(outbreak_efficiency);   // Efficiency factor scaling outbreak initiation (unitless)
  PARAMETER(predation_coeff_fast);  // Predation coefficient on fast-growing coral (unitless)
  PARAMETER(predation_coeff_slow);  // Predation coefficient on slow-growing coral (unitless)
  PARAMETER(coral_regeneration_fast); // Regeneration rate for fast-growing coral (% year^-1)
  PARAMETER(coral_regeneration_slow); // Regeneration rate for slow-growing coral (% year^-1)
  PARAMETER(mortality_rate);        // Natural mortality rate of COTS (year^-1)
  PARAMETER(sigma_cots);            // Observation error standard deviation for COTS (log-scale)

  // 3. Numerical Stability: small constant to avoid division by zero
  Type eps = Type(1e-8);
  
  // 4. Initialize Negative Log Likelihood
  Type nll = 0.0;
  
  // 5. Initialize state variables: predictions for COTS and coral communities
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Use first observation with numeric smoothing
  cots_pred[0] = cots_dat[0] + eps;
  fast_pred[0] = fast_dat[0] + eps;
  slow_pred[0] = slow_dat[0] + eps;
  
  // 6. Temporal dynamics loop (using previous time step values only)
  for(int t = 1; t < n; t++){
    
    // Equation 1:
    // Compute outbreak effect as a smooth function of larval immigration relative to a threshold.
    // outbreak_effect = outbreak_efficiency * (cotsimm_dat[t-1] - threshold_outbreak) / (1 + |cotsimm_dat[t-1] - threshold_outbreak|)
    Type outbreak_effect = outbreak_efficiency * (cotsimm_dat[t-1] - threshold_outbreak) / (Type(1) + fabs(cotsimm_dat[t-1] - threshold_outbreak));

    // Equation 2:
    // COTS dynamics combine intrinsic growth modified by the outbreak effect, natural mortality,
    // and an additive environmental forcing term proportional to sea-surface temperature.
    cots_pred[t] = cots_pred[t-1] 
                   + growth_rate * cots_pred[t-1] * outbreak_effect 
                   - mortality_rate * cots_pred[t-1] 
                   + Type(0.1) * sst_dat[t-1]; // 0.1: environmental efficiency constant

    // Equation 3:
    // Fast-growing coral dynamics: regeneration towards 100% cover, reduced by predation pressure from COTS.
    fast_pred[t] = fast_pred[t-1] 
                   + coral_regeneration_fast * (Type(100) - fast_pred[t-1])
                   - predation_coeff_fast * cots_pred[t-1] * fast_pred[t-1] / (Type(1) + fast_pred[t-1]);

    // Equation 4:
    // Slow-growing coral dynamics: similar regeneration dynamics and predation loss.
    slow_pred[t] = slow_pred[t-1] 
                   + coral_regeneration_slow * (Type(100) - slow_pred[t-1])
                   - predation_coeff_slow * cots_pred[t-1] * slow_pred[t-1] / (Type(1) + slow_pred[t-1]);

    // Report predictions for diagnostics
    REPORT(cots_pred[t]);
    REPORT(fast_pred[t]);
    REPORT(slow_pred[t]);

    // 7. Likelihood calculation:
    // Lognormal likelihood for strictly positive COTS data.
    Type mu = log(cots_pred[t] + eps);
    nll -= ( dnorm(log(cots_dat[t] + eps), mu, sigma_cots, true) - log(cots_dat[t] + eps) );
  }
  
  // 8. Equation Descriptions:
  // (1) outbreak_effect computes a smooth relative difference between larval supply and outbreak threshold.
  // (2) COTS dynamic: integrates growth, outbreak boost, mortality, and SST-induced process efficiency.
  // (3) Fast coral: recovers toward full cover but is reduced by COTS predation.
  // (4) Slow coral: similar recovery dynamics with a separate predation coefficient.

  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  return nll;
}
