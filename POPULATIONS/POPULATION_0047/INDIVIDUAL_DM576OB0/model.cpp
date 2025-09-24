#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() () {
  using namespace density;
  
  // Data inputs (note: use t-1 values for process predictions)
  DATA_VECTOR(time);             // Time variable (years)
  DATA_VECTOR(cots_dat);         // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);         // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);         // Observed slow-growing coral cover (%)

  // Parameters (all units in appropriate rates or proportions)
  PARAMETER(growth_rate);               // (year^-1) Intrinsic outbreak growth rate [literature/expert opinion]
  PARAMETER(decay_rate);                // (year^-1) Post-outbreak decay rate [expert opinion]
  PARAMETER(coral_predation_fast);      // (m2/%/year) Impact of fast-growing coral on predation pressure [initial estimate]
  PARAMETER(coral_predation_slow);      // (m2/%/year) Impact of slow-growing coral on predation pressure [initial estimate]
  PARAMETER(ecological_efficiency);     // (dimensionless) Efficiency factor translating coral cover to suppression of outbreaks [expert opinion]

  // Initialization for predictions
  int n = cots_dat.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  cots_pred(0) = cots_dat(0);  // initial condition from data
  fast_pred(0) = fast_dat(0);  // initial condition from data
  slow_pred(0) = slow_dat(0);  // initial condition from data

  // Negative log-likelihood accumulator
  Type nll = 0.0;
  Type epsilon = Type(1e-8); // small constant for numerical stability

  // Loop over time steps using only previous time step values
  for(int t = 1; t < n; t++){
    // 1. Outbreak growth dynamics:
    //    Equation 1: Logistic growth with saturating function,
    //    carrying capacity assumed to be 1e5 individuals/m2.
    Type growth = growth_rate * cots_pred(t-1) * (1 - cots_pred(t-1) / (Type(1e5) + epsilon));

    // 2. Exponential decay capturing outbreak decline.
    Type decay = decay_rate * cots_pred(t-1);

    // 3. Predation feedback effect:
    //    Equation 3: Coral cover modifies outbreak dynamics through selective predation.
    //    Fast-growing and slow-growing coral data (from previous time step) are weighted by their impacts.
    Type predation_effect = ecological_efficiency * (coral_predation_fast * fast_dat(t-1) + coral_predation_slow * slow_dat(t-1));

    // Update the predicted COTS abundance ensuring smooth transition by bounding below with epsilon.
    cots_pred(t) = cots_pred(t-1) + growth - decay - predation_effect;
    cots_pred(t) = (cots_pred(t) > 0 ? cots_pred(t) : epsilon);

    // Update coral predictions with a persistence model (no change over time)
    fast_pred(t) = fast_pred(t-1);
    slow_pred(t) = slow_pred(t-1);
    
    // Likelihood calculation:
    //    Equation 4: Lognormal likelihood for COTS data with a small fixed standard deviation.
    nll -= dnorm(log(cots_dat(t)), log(cots_pred(t)), Type(0.1), true);
    //    Equation 5: Lognormal likelihood for fast-growing coral cover observation.
    nll -= dnorm(log(fast_dat(t)), log(fast_pred(t)), Type(0.1), true);
    //    Equation 6: Lognormal likelihood for slow-growing coral cover observation.
    nll -= dnorm(log(slow_dat(t)), log(slow_pred(t)), Type(0.1), true);
  }

  // Reporting predicted values (for post-processing and diagnostics)
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  // Numbered Equations description:
  //   (1) Logistic growth with saturation.
  //   (2) Exponential decay for outbreak decline.
  //   (3) Coral cover mediated predation feedback.
  //   (4) Lognormal likelihood to match observed COTS data.
  
  return nll;
}
