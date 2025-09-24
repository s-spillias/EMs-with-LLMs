#include <TMB.hpp>

// 1. Data and parameter definitions:
//    [1] year: time index/data
//    [2] cots_dat: observed COTS abundance (ind/m^2)
//    [3] fast_dat: observed fast-coral cover (%)
//    [4] slow_dat: observed slow-coral cover (%)
//    [5] sst_dat: Sea-Surface Temperature (°C)
//    [6] cotsimm_dat: larval immigration rate (ind/m^2/year)
//
// 2. Parameters (with suggested bounds embedded in comments):
//    growth_rate (year^-1): intrinsic COTS growth rate
//    mortality_rate (year^-1): COTS mortality rate
//    outbreak_threshold (ind/m^2): threshold triggering outbreak dynamics
//    prey_efficiency (unitless): efficiency of coral recovery rate
//    predation_rate_fast (m^2/(ind*year)): predation rate on fast-growing corals
//    predation_rate_slow (m^2/(ind*year)): predation rate on slow-growing corals
//
// Equations description:
// (1) COTS dynamics: Logistic-type growth modified by an outbreak effect determined by a smooth threshold function.
// (2) Coral dynamics: Growth with saturating logistic recovery and reduction due to COTS predation modeled with a Michaelis-Menten-like response.
// (3) Likelihood: Lognormal error structure for the strictly positive COTS data, ensuring a fixed minimum variance.
template<class Type>
Type objective_function<Type>::operator()()
{
  // DATA inputs
  DATA_VECTOR(year);                // Time variable from the data file (year)
  DATA_VECTOR(cots_dat);            // Observed COTS abundance (ind/m^2)
  DATA_VECTOR(fast_dat);            // Observed fast-coral cover (%)
  DATA_VECTOR(slow_dat);            // Observed slow-coral cover (%)
  DATA_VECTOR(sst_dat);             // Sea-Surface Temperature (°C)
  DATA_VECTOR(cotsimm_dat);         // COTS larval immigration rate (ind/m^2/year)

  // PARAMETERS of the model
  PARAMETER(growth_rate);           // Intrinsic growth rate of COTS, suggested bounds: [0, 5]
  PARAMETER(mortality_rate);        // Mortality rate of COTS, suggested bounds: [0, 5]
  PARAMETER(outbreak_threshold);    // Outbreak threshold for COTS density, suggested bounds: [0, 100]
  PARAMETER(prey_efficiency);       // Efficiency of coral recovery, suggested bounds: [0, 1]
  PARAMETER(predation_rate_fast);   // Predation rate on fast-growing corals, suggested bounds: [0, 1]
  PARAMETER(predation_rate_slow);   // Predation rate on slow-growing corals, suggested bounds: [0, 1]

  int n = year.size();              // Total number of time steps

  // Initialize vectors for model predictions
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Set initial predictions equal to the observed initial conditions
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Negative log-likelihood
  Type nll = 0.0;
  Type eps = Type(1e-8); // small constant for numerical stability

  // Loop over time steps (using t-1 predictions to compute t predictions)
  for(int t = 1; t < n; t++){
    // Equation 1: COTS dynamics
    // Outbreak effect: smooth transition using a logistic function
    Type outbreak_effect = 1.0 / (1.0 + exp(-(cots_pred(t-1) - outbreak_threshold)));
    cots_pred(t) = cots_pred(t-1)
                   + growth_rate * outbreak_effect * cots_pred(t-1) * (1.0 - cots_pred(t-1) / (Type(100.0) + eps))
                   - mortality_rate * cots_pred(t-1);
    
    // Equation 2: Coral dynamics
    // Fast-growing coral dynamics with predation
    fast_pred(t) = fast_pred(t-1)
                   + prey_efficiency * fast_pred(t-1) * (1.0 - fast_pred(t-1) / Type(100.0))
                   - predation_rate_fast * cots_pred(t-1) * fast_pred(t-1) / (Type(1.0) + fast_pred(t-1));
    // Slow-growing coral dynamics with predation
    slow_pred(t) = slow_pred(t-1)
                   + prey_efficiency * slow_pred(t-1) * (1.0 - slow_pred(t-1) / Type(50.0))
                   - predation_rate_slow * cots_pred(t-1) * slow_pred(t-1) / (Type(1.0) + slow_pred(t-1));
    
    // Equation 3: Likelihood for COTS data using lognormal error:
    // We incorporate environmental forcing implicitly via the state variables.
    Type sd = 1.0; // fixed minimum standard deviation; could be modified to include sst_dat or cotsimm_dat effects
    if(cots_dat(t) > 0){
      Type log_pred = log(cots_pred(t) + eps);
      Type log_obs  = log(cots_dat(t) + eps);
      nll -= dnorm(log_obs, log_pred, sd, true);
    }
  }

  // REPORT model predictions so they can be inspected in the output:
  REPORT(cots_pred);  // Predicted COTS abundance time series
  REPORT(fast_pred);  // Predicted fast-coral cover time series
  REPORT(slow_pred);  // Predicted slow-coral cover time series

  return nll;
}
