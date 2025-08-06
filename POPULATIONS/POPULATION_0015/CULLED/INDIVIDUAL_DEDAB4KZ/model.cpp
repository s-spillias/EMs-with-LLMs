#include <TMB.hpp>

// TMB model for episodic outbreaks of COTS on the Great Barrier Reef
// Numbered equations description:
// 1. COTS dynamics: cots[t+1] = cots[t] + dt*( r_cots * cots[t]*(1 - cots[t] / K_cots) + theta * log(1 + fast[t] + slow[t]) )
// 2. Fast-growing coral dynamics: fast[t+1] = fast[t] + dt*( r_fast * fast[t]*(1 - fast[t] / K_fast) - pred_eff_fast * cots[t] * fast[t]/(fast[t] + Type(1e-8)) )
// 3. Slow-growing coral dynamics: slow[t+1] = slow[t] + dt*( r_slow * slow[t]*(1 - slow[t] / K_slow) - pred_eff_slow * cots[t] * slow[t]/(slow[t] + Type(1e-8)) )

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;
  
  // ** DATA **
  DATA_VECTOR(cots_dat);       // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);       // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);       // Observed slow-growing coral cover (%)
  int n = cots_dat.size();     // Number of time steps (assumes equal length for all data vectors)
  Type dt = 1.0;               // Time step (year); fixed value
  
  // ** PARAMETERS **
  // COTS parameters
  PARAMETER(r_cots);           // Intrinsic growth rate of COTS (year^-1); source: literature
  PARAMETER(K_cots);           // Carrying capacity for COTS (individuals/m2); source: expert opinion
  PARAMETER(theta);            // Outbreak threshold effect parameter (unitless); modulates outbreak trigger based on coral cover
  
  // Fast-growing coral parameters
  PARAMETER(r_fast);           // Intrinsic growth rate for fast-growing coral (% per year)
  PARAMETER(K_fast);           // Carrying capacity for fast-growing coral (% cover)
  PARAMETER(pred_eff_fast);    // Predation efficiency of COTS on fast-growing coral (m2/(individual*year))
  PARAMETER(h_fast);           // Handling time parameter for predation on fast-growing coral (dimensionless)
  
  // Slow-growing coral parameters
  PARAMETER(r_slow);           // Intrinsic growth rate for slow-growing coral (% per year)
  PARAMETER(K_slow);           // Carrying capacity for slow-growing coral (% cover)
  PARAMETER(pred_eff_slow);    // Predation efficiency of COTS on slow-growing coral (m2/(individual*year))
  PARAMETER(h_slow);           // Handling time parameter for predation on slow-growing coral (dimensionless)
  
  // Initial conditions (state variables)
  PARAMETER(cots0);            // Initial COTS abundance (individuals/m2)
  PARAMETER(fast0);            // Initial fast-growing coral cover (%)
  PARAMETER(slow0);            // Initial slow-growing coral cover (%)
  
  // Observation error parameters
  PARAMETER(sigma_cots);       // Observation standard deviation for COTS (individuals/m2)
  PARAMETER(sigma_fast);       // Observation standard deviation for fast-growing coral cover (%)
  PARAMETER(sigma_slow);       // Observation standard deviation for slow-growing coral cover (%)
  
  // Convert sigma parameters to be positive, with a fixed minimum to ensure numerical stability
  sigma_cots = sigma_cots < Type(1e-8) ? Type(1e-8) : sigma_cots;
  sigma_fast = sigma_fast < Type(1e-8) ? Type(1e-8) : sigma_fast;
  sigma_slow = sigma_slow < Type(1e-8) ? Type(1e-8) : sigma_slow;
  
  // ** STATE VECTORS **
  vector<Type> cots_pred(n), fast_pred(n), slow_pred(n);
  cots_pred(0) = cots0;  // set initial state from parameter
  fast_pred(0) = fast0;
  slow_pred(0) = slow0;
  
  Type nll = 0.0;  // negative log likelihood
  // Include likelihood for initial conditions (time 0)
  nll -= dnorm(cots_dat(0), cots_pred(0), sigma_cots, true);
  nll -= dnorm(fast_dat(0), fast_pred(0), sigma_fast, true);
  nll -= dnorm(slow_dat(0), slow_pred(0), sigma_slow, true);
  
  // ** DYNAMIC MODEL **
  for(int t=1; t < n; t++){
    Type fast_prev = fast_pred(t-1);
    if(fast_prev < Type(1e-8)) fast_prev = Type(1e-8);
    Type slow_prev = slow_pred(t-1);
    if(slow_prev < Type(1e-8)) slow_prev = Type(1e-8);
    // Equation 1: COTS dynamics
    {
      Type fast_for_log = fast_pred(t-1) < Type(1e-8) ? Type(1e-8) : fast_pred(t-1);
      Type slow_for_log = slow_pred(t-1) < Type(1e-8) ? Type(1e-8) : slow_pred(t-1);
      cots_pred(t) = cots_pred(t-1) + 
        dt * (r_cots * cots_pred(t-1) * (1 - cots_pred(t-1)/K_cots) + 
              theta * log(1 + fast_for_log + slow_for_log));
    }
    
    // Equation 2: Fast-growing coral dynamics
    fast_pred(t) = fast_pred(t-1) +
      dt * (r_fast * fast_pred(t-1) * (1 - fast_pred(t-1)/K_fast) -
            (pred_eff_fast * cots_pred(t-1) * fast_pred(t-1))/(Type(1) + exp(h_fast) * fast_prev + Type(1e-8)));
    
    // Equation 3: Slow-growing coral dynamics
    slow_pred(t) = slow_pred(t-1) +
      dt * (r_slow * slow_pred(t-1) * (1 - slow_pred(t-1)/K_slow) -
            (pred_eff_slow * cots_pred(t-1) * slow_pred(t-1))/(Type(1) + exp(h_slow) * slow_prev + Type(1e-8)));
    
    // ** LIKELIHOOD CALCULATION **
    nll -= dnorm(cots_dat(t), cots_pred(t), sigma_cots, true);
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_fast, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_slow, true);
  }
  
  // ** REPORTING **
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
