#include <TMB.hpp>  // TMB library for fast statistical and template-based modeling

// Template Model Builder (TMB) model for COTS outbreak dynamics and coral interactions.
// The equations incorporate:
// 1. COTS population growth modulated by environmental factors and resource limitation.
//    - intrinsic_growth_rate_COTS (year^-1)
//    - resource_limitation_threshold (unitless)
//    - environmental_modifier (unitless)
// 2. Selective predation on fast-growing (Acropora spp.) and slow-growing (Faviidae/Porities spp.) coral communities.
//    - max_predation_rate (% reduction per year)
//    - process_efficiency (unitless)
// 3. Coral regeneration.
//    - coral_regeneration_rate_fast (year^-1)
//    - coral_regeneration_rate_slow (year^-1)
// 4. Likelihood calculation based on lognormal error distributions to account for the strictly positive data values.
//    - minimum_std: Small constant (usually 1e-8) for numerical stability.

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;
  
  // Data inputs: (all vectors have length n corresponding to time steps)
  DATA_VECTOR(Year);               // Year (time steps)
  DATA_VECTOR(cots_dat);           // Observed COTS abundance (boom-bust outbreak levels, individuals/m2)
  DATA_VECTOR(fast_dat);           // Observed fast-growing coral cover (Acropora spp., %)
  DATA_VECTOR(slow_dat);           // Observed slow-growing coral cover (Faviidae/Porities spp., %)
  
  // Parameters to be estimated:
  PARAMETER(intrinsic_growth_rate_COTS);      // Intrinsic growth rate of COTS (year^-1)
  PARAMETER(max_predation_rate);                // Maximum rate of predation on fast-growing corals (% reduction per year)
  PARAMETER(coral_regeneration_rate_fast);      // Regeneration rate of fast-growing coral (year^-1)
  PARAMETER(coral_regeneration_rate_slow);      // Regeneration rate of slow-growing coral (year^-1)
  PARAMETER(resource_limitation_threshold);     // Threshold for resource limitation effects (unitless)
  PARAMETER(environmental_modifier);            // Environmental modifier (unitless) affecting growth rates
  PARAMETER(self_limitation_rate);              // New parameter: density-dependent self limitation on growth
  PARAMETER(process_efficiency);                // Efficiency factor for predation processes (unitless)
  PARAMETER(minimum_std);                       // Small constant to avoid division-by-zero in likelihood
  
  int n = cots_dat.size();  // Total number of time steps
  
  // Vectors to store model predictions for each compartment:
  vector<Type> cots_pred(n);  // Predicted COTS abundance (individuals/m2)
  vector<Type> fast_pred(n);  // Predicted fast-growing coral cover (%) 
  vector<Type> slow_pred(n);  // Predicted slow-growing coral cover (%) 
  
  // Initialize predictions at time 0 using observations (to avoid data leakage, only first time step is set)
  cots_pred(0) = cots_dat(0);  // Initial COTS abundance (from data)
  fast_pred(0) = fast_dat(0);  // Initial fast-growing coral cover (from data)
  slow_pred(0) = slow_dat(0);  // Initial slow-growing coral cover (from data)
  
  Type nll = 0.0;  // Initialize negative log likelihood
  
  // Equations description:
  // 1. COTS Dynamics: COTS_pred[t] = COTS[t-1] + growth - predation
  //    - growth term: intrinsic_growth_rate_COTS * COTS[t-1] * environmental_modifier * (1 - (COTS[t-1] / (resource_limitation_threshold + COTS[t-1] + minimum_std)))
  //    - predation term: max_predation_rate * fast_pred[t-1] * process_efficiency
  //
  // 2. Fast-growing Coral Dynamics: fast_pred[t] = fast_pred[t-1] + regeneration - loss due to predation
  //    - regeneration: coral_regeneration_rate_fast * (100 - fast_pred[t-1])
  //    - loss: max_predation_rate * COTS[t-1] / (1 + COTS[t-1] + minimum_std)
  //
  // 3. Slow-growing Coral Dynamics: slow_pred[t] = slow_pred[t-1] + regeneration - loss due to predation
  //    - regeneration: coral_regeneration_rate_slow * (100 - slow_pred[t-1])
  //    - loss: (max_predation_rate * COTS[t-1] / (1 + COTS[t-1] + minimum_std)) * 0.5
  
  // Loop over time steps (starting at t = 1)
  for (int t = 1; t < n; t++){
    // COTS Dynamics
    Type previous_cots = cots_pred(t-1);
    Type growth = intrinsic_growth_rate_COTS * previous_cots * environmental_modifier - self_limitation_rate * previous_cots * previous_cots; // Growth with density-dependent negative feedback
    Type limitation = previous_cots / (resource_limitation_threshold + previous_cots + minimum_std); // Saturating resource limitation
    Type predation_effect = max_predation_rate * fast_pred(t-1) * process_efficiency; // Loss due to predation on fast coral
    cots_pred(t) = previous_cots + growth * (1 - limitation) - predation_effect;
    
    // Fast-growing Coral Dynamics
    Type previous_fast = fast_pred(t-1);
    Type fast_regen = coral_regeneration_rate_fast * (100 - previous_fast);  // Regeneration assuming maximum cover is 100%
    Type fast_loss = max_predation_rate * previous_cots / (1 + previous_cots + minimum_std);  // Predation loss rate
    fast_pred(t) = previous_fast + fast_regen - fast_loss;
    
    // Slow-growing Coral Dynamics
    Type previous_slow = slow_pred(t-1);
    Type slow_regen = coral_regeneration_rate_slow * (100 - previous_slow);  // Regeneration for slow-growing corals
    Type slow_loss = (max_predation_rate * previous_cots / (1 + previous_cots + minimum_std)) * 0.5;  // Lower impact due to selective predation
    slow_pred(t) = previous_slow + slow_regen - slow_loss;
    
    // Likelihood: use lognormal error distributions for strictly positive data
    // 4. Likelihood for COTS abundance (lognormal: safe prediction)
    Type safe_cots_pred = (cots_pred(t) > minimum_std ? cots_pred(t) : minimum_std);
    Type sd_cots = (0.1 * safe_cots_pred > minimum_std ? 0.1 * safe_cots_pred : minimum_std);
    nll -= (dnorm(log(cots_dat(t) + minimum_std), log(safe_cots_pred), sd_cots, true) - log(cots_dat(t) + minimum_std));
    
    // 5. Likelihood for fast-growing coral cover (lognormal: safe prediction)
    Type safe_fast_pred = (fast_pred(t) > minimum_std ? fast_pred(t) : minimum_std);
    Type sd_fast = (0.1 * safe_fast_pred > minimum_std ? 0.1 * safe_fast_pred : minimum_std);
    nll -= (dnorm(log(fast_dat(t) + minimum_std), log(safe_fast_pred), sd_fast, true) - log(fast_dat(t) + minimum_std));
    
    // 6. Likelihood for slow-growing coral cover (lognormal: safe prediction)
    Type safe_slow_pred = (slow_pred(t) > minimum_std ? slow_pred(t) : minimum_std);
    Type sd_slow = (0.1 * safe_slow_pred > minimum_std ? 0.1 * safe_slow_pred : minimum_std);
    nll -= (dnorm(log(slow_dat(t) + minimum_std), log(safe_slow_pred), sd_slow, true) - log(slow_dat(t) + minimum_std));
  }
  
  // Report predicted values for further diagnostics (all _pred variables are included)
  REPORT(cots_pred);    // Predicted COTS abundance time-series
  REPORT(fast_pred);    // Predicted fast-growing coral cover time-series
  REPORT(slow_pred);    // Predicted slow-growing coral cover time-series
  
  return nll;
}
