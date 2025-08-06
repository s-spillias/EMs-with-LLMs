#include <TMB.hpp>

// Template Model Builder model for episodic COTS outbreaks on the Great Barrier Reef.
// 
// Equations and key considerations:
// 1. COTS Dynamics Equation:
//    - Growth: logistic growth modulated by intrinsic growth rate and carrying capacity.
//    - Environmental modification: multiplicative factor based on sea-surface temperature (sst_dat) and larval immigration (cotsimm_dat)
//    - Predation: loss term via a saturating functional response on combined coral cover (fast_dat + slow_dat)
// 2. Numerical Stability:
//    - Small constant (1e-8) is added to denominators to avoid division by zero.
//    - Smooth transitions are implemented via logistic functions rather than hard thresholds.
// 3. Likelihood:
//    - Observations (cots_dat) are modeled using a lognormal error distribution.
//    - Standard deviation is computed on the log-scale with a minimum value imposed.
//
// Note: All predictions (_pred variables) use values from the previous time step to avoid data leakage.

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA input vectors (observations for each time step)
  DATA_VECTOR(cots_dat);         // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);         // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);         // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);          // Sea Surface Temperature (°C)
  DATA_VECTOR(cotsimm_dat);      // Crown-of-thorns larval immigration rate (individuals/m2/year)

  // PARAMETERS
  PARAMETER(growth_rate);           // Intrinsic growth rate for COTS (year^-1)
  PARAMETER(carrying_capacity);     // Carrying capacity for outbreak levels (individuals/m2)
  PARAMETER(predation_rate);        // Rate at which COTS predate on coral (per year)
  PARAMETER(predation_efficiency);  // Efficiency of converting coral predation into COTS growth (unitless)
  PARAMETER(environment_sensitivity); // Modifier for environmental conditions on growth (unitless)
  PARAMETER(log_sigma);             // Log standard deviation for observation error
  PARAMETER(fast_growth_rate);         // Intrinsic growth rate for fast-growing coral (% per year)
  PARAMETER(fast_carrying_capacity);   // Carrying capacity for fast-growing coral (%)
  PARAMETER(slow_growth_rate);         // Intrinsic growth rate for slow-growing coral (% per year)
  PARAMETER(slow_carrying_capacity);   // Carrying capacity for slow-growing coral (%)
  PARAMETER(fast_predation_rate);      // Rate at which COTS predation reduces fast-growing coral (% loss per individual/m2)
  PARAMETER(slow_predation_rate);      // Rate at which COTS predation reduces slow-growing coral (% loss per individual/m2)
  PARAMETER(log_sigma_fast);           // Log standard deviation for fast coral observation error
  PARAMETER(log_sigma_slow);           // Log standard deviation for slow coral observation error

  int n = cots_dat.size();  // number of time steps
  vector<Type> cots_pred(n);  // predicted COTS abundance at each time step
  vector<Type> fast_pred(n);  // predicted fast-growing coral cover at each time step
  vector<Type> slow_pred(n);  // predicted slow-growing coral cover at each time step

  Type sigma = exp(log_sigma) + Type(1e-8); // observation error, stabilized by small constant

  // Initialize the population prediction
  cots_pred(0) = cots_dat(0);  // Starting at observed equilibrium
  fast_pred(0) = fast_dat(0);  // Initialize fast-growing coral prediction with observed value
  slow_pred(0) = slow_dat(0);  // Initialize slow-growing coral prediction with observed value

  // Loop through time steps. Predictions use the previous time step's values.
  for(int t = 1; t < n; t++){
    // (1) Logistic growth modulated by growth_rate and carrying_capacity
    Type growth = growth_rate * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1)/carrying_capacity);

    // (2) Environmental effect: a smooth additive modifier combining SST and larval immigration
    Type env_effect = environment_sensitivity * ( sst_dat(t-1) / (sst_dat(t-1) + Type(1e-8))
                      + cotsimm_dat(t-1) / (cotsimm_dat(t-1) + Type(1e-8)) );

    // (3) Predation loss: modeled with a saturating response depending on total coral cover
    Type total_coral = fast_dat(t-1) + slow_dat(t-1) + Type(1e-8);    
    Type predation_loss = predation_rate * cots_pred(t-1) * ( Type(1.0) / (Type(1.0) + exp(-total_coral)) );

    // Update prediction using the previous time step only (avoid data leakage)
    cots_pred(t) = cots_pred(t-1) + (growth * env_effect - predation_efficiency * predation_loss);
    
    // Fast-growing coral dynamics: logistic growth with predation loss from COTS
    fast_pred(t) = fast_pred(t-1) 
                   + fast_growth_rate * fast_pred(t-1) * (Type(1.0) - fast_pred(t-1)/fast_carrying_capacity)
                   - fast_predation_rate * cots_pred(t-1) * fast_pred(t-1) / (fast_pred(t-1) + Type(1e-8));
    
    // Slow-growing coral dynamics: logistic growth with predation loss from COTS
    slow_pred(t) = slow_pred(t-1) 
                   + slow_growth_rate * slow_pred(t-1) * (Type(1.0) - slow_pred(t-1)/slow_carrying_capacity)
                   - slow_predation_rate * cots_pred(t-1) * slow_pred(t-1) / (slow_pred(t-1) + Type(1e-8));
  }

  // Likelihood: assume lognormal error distribution for positive observations.
  Type nll = 0.0;
  for(int t = 0; t < n; t++){
    nll -= (dnorm(log(cots_dat(t)), log(cots_pred(t) + Type(1e-8)), sigma, true) - log(cots_dat(t) + Type(1e-8)));
    nll -= (dnorm(log(fast_dat(t)), log(fast_pred(t) + Type(1e-8)), exp(log_sigma_fast) + Type(1e-8), true) - log(fast_dat(t) + Type(1e-8)));
    nll -= (dnorm(log(slow_dat(t)), log(slow_pred(t) + Type(1e-8)), exp(log_sigma_slow) + Type(1e-8), true) - log(slow_dat(t) + Type(1e-8)));
  }

  // REPORT all model predictions for inspection.
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}
