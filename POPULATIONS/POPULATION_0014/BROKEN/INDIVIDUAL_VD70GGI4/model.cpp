#include <TMB.hpp>

// TMB model for episodic outbreaks of Crown-of-Thorns starfish (COTS) on the Great Barrier Reef
// 1. COTS dynamics: exponential growth modulated by an outbreak trigger based on coral cover.
// 2. Coral dynamics: recovery towards full cover minus predation by COTS.
// 3. Smooth saturating functions and small constants (1e-8) are used for numerical stability.
// 4. Likelihood: Lognormal error distributions for strictly positive observations.
// 5. Predictions (_pred) are computed using previous time step states to avoid data leakage.

template<class Type>
Type objective_function<Type>::operator()() {
  using namespace density;
  
  // DATA: Time (year) and observations for COTS and coral covers (%)
  DATA_VECTOR(Year);         // Years (from data file)
  DATA_VECTOR(cots_dat);       // Observed COTS density (individuals/m^2)
  DATA_VECTOR(fast_dat);       // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);       // Observed slow-growing coral cover (%)
  
  // PARAMETERS (log-transformed where necessary for positivity):
  PARAMETER(log_growth_rate);   // [year^-1] Log intrinsic growth rate of COTS (Equation 4)
  PARAMETER(log_predation_fast);  // [m^2/individual/year] Log predation efficiency on fast-growing coral (Equation 5)
  PARAMETER(log_predation_slow);  // [m^2/individual/year] Log predation efficiency on slow-growing coral (Equation 6)
  PARAMETER(log_recovery);      // [year^-1] Log recovery rate of coral cover (Equations 5 & 6)
  PARAMETER(cots_init);         // [individuals/m^2] Initial density of COTS (Equation 1)
  PARAMETER(fast_init);         // [% cover] Initial fast-growing coral cover (Equation 2)
  PARAMETER(slow_init);         // [% cover] Initial slow-growing coral cover (Equation 3)
  PARAMETER(dbeta);             // Outbreak trigger parameter; influences sensitivity to coral cover declines (Equation 4)
  
  // Transform parameters to natural scale:
  Type growth_rate    = exp(log_growth_rate);
  Type predation_fast = exp(log_predation_fast);
  Type predation_slow = exp(log_predation_slow);
  Type recovery       = exp(log_recovery);
  
  // Number of time steps:
  int n = Year.size();
  
  // Vectors for predicted states:
  vector<Type> cots_pred(n);  // Predicted COTS density
  vector<Type> fast_pred(n);  // Predicted fast-growing coral cover
  vector<Type> slow_pred(n);  // Predicted slow-growing coral cover
  
  // Initial conditions (Equations 1-3)
  cots_pred[0] = cots_init;
  fast_pred[0] = fast_init;
  slow_pred[0] = slow_init;
  
  // Negative log-likelihood:
  Type nll = 0.0;
  
  // Loop over time steps (using previous states for predictions)
  for(int t = 1; t < n; t++){
      Type dt = Year[t] - Year[t-1];  // Time step duration
      
      // Equation 4: Update COTS density using multiplicative exponential growth modulated by an outbreak trigger.
      // The trigger function is smooth and increases outbreak potential as coral cover declines.
      Type trigger = Type(1.0) / (Type(1.0) + exp(-dbeta * (fast_pred[t-1] + slow_pred[t-1]))); 
      cots_pred[t] = cots_pred[t-1] * (1 + dt * (growth_rate * trigger));
      
      // Equation 5: Fast-growing coral dynamics with recovery and predation loss.
      fast_pred[t] = fast_pred[t-1] + dt * (recovery * (Type(100.0) - fast_pred[t-1])
                       - predation_fast * cots_pred[t-1] * fast_pred[t-1] / (fast_pred[t-1] + Type(1e-8)));
      
      // Equation 6: Slow-growing coral dynamics with recovery and predation loss.
      slow_pred[t] = slow_pred[t-1] + dt * (recovery * (Type(100.0) - slow_pred[t-1])
                       - predation_slow * cots_pred[t-1] * slow_pred[t-1] / (slow_pred[t-1] + Type(1e-8)));
      
      // Ensure predictions remain strictly positive (smoothly enforced):
      cots_pred[t] = (cots_pred[t] > Type(0)) ? cots_pred[t] : Type(1e-8);
      fast_pred[t] = (fast_pred[t] > Type(0)) ? fast_pred[t] : Type(1e-8);
      slow_pred[t] = (slow_pred[t] > Type(0)) ? slow_pred[t] : Type(1e-8);
      
      // Equation 7-9: Likelihood contributions using lognormal distributions for each observation.
      // For a lognormal distribution, the log-density is computed as:
      // dnorm(log(x), log(pred), sd, true) - log(x)
      nll -= (dnorm(log(cots_dat[t] + Type(1e-8)), log(cots_pred[t]), Type(0.1), true) - log(cots_dat[t] + Type(1e-8)));
      nll -= (dnorm(log(fast_dat[t] + Type(1e-8)), log(fast_pred[t]), Type(0.1), true) - log(fast_dat[t] + Type(1e-8)));
      nll -= (dnorm(log(slow_dat[t] + Type(1e-8)), log(slow_pred[t]), Type(0.1), true) - log(slow_dat[t] + Type(1e-8)));
  }
  
  // REPORT model predictions for further analysis.
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
