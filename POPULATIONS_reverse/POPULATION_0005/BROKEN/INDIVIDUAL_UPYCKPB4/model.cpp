#include <TMB.hpp>

// Template Model Builder (TMB) model for Crown-of-Thorns starfish and coral dynamics.
//
// Equations:
// 1. Slow-growing coral dynamics: logistic growth with predation pressure.
// 2. Fast-growing coral dynamics: logistic growth with predation pressure.
// 3. Crown-of-Thorns starfish dynamics: change in abundance based on coral food availability and natural mortality.
//
// Notes:
// - Small constants (e.g., 1e-8) are used for numerical stability.
// - Parameters are log-transformed and then exponentiated to ensure positivity.
// - Smooth penalties bound parameters within biologically meaningful ranges.
// - Likelihood uses a lognormal-inspired error expression via dnorm with log-transformation and Jacobian adjustment.
// - Predictions (_pred) are reported alongside ADREPORTs for key derived parameters.
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs (observed values)
  DATA_VECTOR(slow_dat);   // Observed slow-growing coral cover (%) 
  DATA_VECTOR(fast_dat);   // Observed fast-growing coral cover (%) 
  DATA_VECTOR(cots_dat);   // Observed Crown-of-Thorns starfish abundance (individuals/m2)
  DATA_VECTOR(sst_dat);    // External forcing: Sea Surface Temperature (Celsius)
  DATA_VECTOR(cotsimm_dat); // External forcing: Crown-of-Thorns immigration rate (individuals/m2/year)

  // Parameters (log-transformed to ensure positivity)
  PARAMETER(log_growth_rate);          // Log intrinsic coral growth rate (log(year^-1))
  PARAMETER(log_consumption_rate_slow);  // Log consumption rate on slow-growing corals (log(m2/(individual*year)))
  PARAMETER(log_consumption_rate_fast);  // Log consumption rate on fast-growing corals (log(m2/(individual*year)))
  PARAMETER(log_mortality_coral);        // Log natural coral mortality rate (log(year^-1))
  PARAMETER(log_mortality_star);         // Log Crown-of-Thorns starfish mortality rate (log(year^-1))
  PARAMETER(log_init_slow);              // Log initial slow-growing coral cover
  PARAMETER(log_init_fast);              // Log initial fast-growing coral cover
  PARAMETER(log_init_cots);              // Log initial Crown-of-Thorns starfish abundance

  // Transform parameters to natural scale
  Type growth_rate = exp(log_growth_rate);                      // Intrinsic growth rate (year^-1)
  Type consumption_rate_slow = exp(log_consumption_rate_slow);    // Consumption rate on slow corals (m2/(individual*year))
  Type consumption_rate_fast = exp(log_consumption_rate_fast);    // Consumption rate on fast corals (m2/(individual*year))
  Type mortality_coral = exp(log_mortality_coral);                // Coral mortality rate (year^-1)
  Type mortality_star = exp(log_mortality_star);                  // Starfish mortality rate (year^-1)
  Type init_slow = exp(log_init_slow);                            // Initial slow-growing coral cover
  Type init_fast = exp(log_init_fast);                            // Initial fast-growing coral cover
  Type init_cots = exp(log_init_cots);                            // Initial Crown-of-Thorns starfish abundance

  // Small constant for numerical stability
  Type eps = Type(1e-8);

  // Number of observations
  int n = slow_dat.size();

  // Predicted values for each observation
  vector<Type> slow_pred(n), fast_pred(n), cots_pred(n);

  // Negative log likelihood
  Type nll = 0.0;

  // Loop over observations with recursive predictions and external forcing.
  // Initial conditions are set from parameters, and predictions depend solely on previous predictions and external forcings.
  for(int i = 0; i < n; i++){
    if(i == 0){
      slow_pred(i) = init_slow;    // initial condition from parameter
      fast_pred(i) = init_fast;
      cots_pred(i) = init_cots;
    } else {
      slow_pred(i) = slow_pred(i-1) * ( growth_rate * (1 - consumption_rate_slow * cots_pred(i-1)) ) * exp(Type(0.01) * (sst_dat(i) - Type(28)));
      fast_pred(i) = fast_pred(i-1) * ( growth_rate * (1 - consumption_rate_fast * cots_pred(i-1)) ) * exp(Type(0.01) * (sst_dat(i) - Type(28)));
      cots_pred(i) = cots_pred(i-1)
          + ( consumption_rate_slow * slow_pred(i-1)
            + consumption_rate_fast * fast_pred(i-1) )
            / ( slow_pred(i-1) + fast_pred(i-1) + eps )
          - mortality_star * cots_pred(i-1)
          + cotsimm_dat(i);
    }
    Type sigma = Type(0.1) + eps;
    nll -= ( dnorm(log(slow_dat(i) + eps), log(slow_pred(i) + eps), sigma, true) - log(slow_dat(i) + eps) );
    nll -= ( dnorm(log(fast_dat(i) + eps), log(fast_pred(i) + eps), sigma, true) - log(fast_dat(i) + eps) );
    nll -= ( dnorm(log(cots_dat(i) + eps), log(cots_pred(i) + eps), sigma, true) - log(cots_dat(i) + eps) );
  }

  // Smooth penalty: penalize growth_rate values exceeding 2.0 year^-1
  nll += pow(max(Type(0.0), growth_rate - Type(2.0)), 2);

  // Reporting predictions and key derived parameters
  REPORT(slow_pred);              // Predicted slow-growing coral cover (%)
  REPORT(fast_pred);              // Predicted fast-growing coral cover (%)
  REPORT(cots_pred);              // Predicted Crown-of-Thorns starfish abundance (individuals/m2)
  ADREPORT(growth_rate);          // Report intrinsic growth rate (year^-1)
  ADREPORT(consumption_rate_slow); // Report consumption rate on slow corals (m2/(individual*year))
  ADREPORT(consumption_rate_fast); // Report consumption rate on fast corals (m2/(individual*year))
  ADREPORT(mortality_coral);       // Report coral mortality rate (year^-1)
  ADREPORT(mortality_star);        // Report starfish mortality rate (year^-1)

  return nll;
}
