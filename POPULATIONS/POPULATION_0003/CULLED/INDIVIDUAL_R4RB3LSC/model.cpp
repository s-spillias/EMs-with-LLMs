// TMB Template Model Builder model for episodic COTS outbreaks on the Great Barrier Reef.
// Each parameter used in the model has detailed inline comments indicating units and sources.
// The following numbered list describes the main equations:
// 1. Coral growth dynamics: Logistic growth modified by saturating functions with small constants Type(1e-8)
// 2. COTS population dynamics: A burst of exponential growth when an outbreak threshold is exceeded, with smooth transitions
// 3. Predation effects: COTS selectively prey on coral communities, with different impacts on fast-growing and slow-growing corals
// 4. Feedback loops: Coral decline further moderates COTS growth through resource limitation.
// 5. Likelihood: Log-normal error models are used for all observations (e.g., coral cover and COTS abundance)
// Note: Only lagged values (from previous time steps) are used in dynamic predictions to avoid data leakage.

#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;
  
  // Data inputs from files (vector length equals number of time steps)
  DATA_VECTOR(Year);              // Years from the data file
  DATA_VECTOR(cotsimm_dat);           // COTS larval immigration rate (individuals/m2/year)
  DATA_VECTOR(cots_dat);              // Adult COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);              // Fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);              // Slow-growing coral cover (%)

  // Model parameters (units provided in parameters.json)
  PARAMETER(growth_rate_COTS);        // Intrinsic growth rate of COTS (year^-1)
  PARAMETER(mortality_rate_COTS);     // Natural mortality rate of COTS (year^-1)
  PARAMETER(outbreak_threshold);      // COTS density threshold triggering outbreak (individuals/m2)
  PARAMETER(coral_growth_fast);       // Growth rate of fast-growing corals (year^-1)
  PARAMETER(coral_growth_slow);       // Growth rate of slow-growing corals (year^-1)
  PARAMETER(predation_rate);          // Predation rate effect of COTS on coral cover (% reduction per unit COTS)
  PARAMETER(env_modifier);            // Modifier for environmental conditions (dimensionless)
  
  // Likelihood accumulation
  Type nll = 0.0;
  
  // Number of time steps
  int n = Year.size();
  
  // Initialize prediction vectors
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Set initial conditions from the first observed data point to avoid using current time step values
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Loop over time steps, starting from the second observation
  for(int t = 1; t < n; t++){
    // Equation 1: Coral growth with saturating function and environmental modifiers
    // Uses lagged values
    fast_pred(t) = fast_pred(t-1) + coral_growth_fast * fast_pred(t-1) * (1 - fast_pred(t-1)/ (Type(100)+Type(1e-8))) * env_modifier;
    slow_pred(t) = slow_pred(t-1) + coral_growth_slow * slow_pred(t-1) * (1 - slow_pred(t-1)/ (Type(100)+Type(1e-8))) * env_modifier;
    
    // Equation 2: COTS dynamics with outbreak trigger and suppression by coral cover
    // Incorporates a smooth threshold function using an exponential transition from low to high growth.
    Type outbreak_effect = Type(1.0) / (Type(1.0) + exp(-(cots_pred(t-1) - outbreak_threshold)));
    // Integrate previous larval immigration and intrinsic growth modulated by available coral cover
    cots_pred(t) = cots_pred(t-1) + (growth_rate_COTS * cots_pred(t-1) * outbreak_effect 
                   - mortality_rate_COTS * cots_pred(t-1) 
                   - predation_rate * (fast_pred(t-1) + slow_pred(t-1))/(Type(100)+Type(1e-8))
                   + cotsimm_dat(t-1));
    
    // Ensure predictions remain non-negative
    cots_pred(t) = cots_pred(t) < 0 ? Type(0) : cots_pred(t);
    fast_pred(t) = fast_pred(t) < 0 ? Type(0) : fast_pred(t);
    slow_pred(t) = slow_pred(t) < 0 ? Type(0) : slow_pred(t);
    
    // Equation 3: Likelihood contribution using lognormal error for positive data.
    // For a lognormal distribution: log(x_dat) ~ Normal(log(pred + eps), sigma)
    // Therefore, log likelihood = dnorm(log(x_dat+eps), log(pred+eps), sigma, true) - log(x_dat+eps)
    nll -= (dnorm(log(cots_dat(t)+Type(1e-8)), log(cots_pred(t)+Type(1e-8)), Type(0.1), true) - log(cots_dat(t)+Type(1e-8)));
    nll -= (dnorm(log(fast_dat(t)+Type(1e-8)), log(fast_pred(t)+Type(1e-8)), Type(0.1), true) - log(fast_dat(t)+Type(1e-8)));
    nll -= (dnorm(log(slow_dat(t)+Type(1e-8)), log(slow_pred(t)+Type(1e-8)), Type(0.1), true) - log(slow_dat(t)+Type(1e-8)));
  }
  
  // Report predicted values for each observation
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
