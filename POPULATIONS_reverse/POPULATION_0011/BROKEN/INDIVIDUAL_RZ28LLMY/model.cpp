#include <TMB.hpp>
template<class Type>
Type smooth_max(Type x, Type m) {
  return m + log1p(exp(x - m));
}
// TMB model for Crown-of-Thorns starfish (COTS) outbreak dynamics and coral predation impacts.
// Equations:
// 1. COTS dynamics: 
//    cots_pred[t] = cots_pred[t-1] + growth_rate_cots * cots_pred[t-1]*(1 - cots_pred[t-1]/(carrying_capacity + 1e-8))
//                  - predation_rate * (slow_pred[t-1] + fast_pred[t-1]) * efficiency
//    (Logistic growth modulated by predation impact)
// 2. Slow coral dynamics:
//    slow_pred[t] = slow_pred[t-1] + coral_regrow_slow*(100 - slow_pred[t-1])
//                   - predation_rate * cots_pred[t-1] * slow_pred[t-1]/(slow_pred[t-1]+1e-8)
//    (Regrowth limited by maximum cover, reduced by predation)
// 3. Fast coral dynamics:
//    fast_pred[t] = fast_pred[t-1] + coral_regrow_fast*(100 - fast_pred[t-1])
//                   - predation_rate * cots_pred[t-1] * fast_pred[t-1]/(fast_pred[t-1]+1e-8)
//    (Analogous dynamics for fast-growing corals)

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data vectors (observations)
  DATA_VECTOR(time);         // Time variable (years)
  DATA_VECTOR(cots_dat);       // Observed COTS density (ind/m^2)
  DATA_VECTOR(slow_dat);       // Observed slow-growing coral cover (%) 
  DATA_VECTOR(fast_dat);       // Observed fast-growing coral cover (%)

  // Parameters with their units and descriptions:
  PARAMETER(growth_rate_cots);   // Intrinsic growth rate of COTS (year^-1)
  PARAMETER(carrying_capacity);  // Maximum sustainable COTS density (ind/m^2)
  PARAMETER(predation_rate);     // Selective predation rate on corals (controls impact per unit coral cover, unit: m^2/(ind*year))
  PARAMETER(efficiency);         // Efficiency of predation impact conversion (unitless)
  PARAMETER(coral_regrow_slow);  // Regrowth rate of slow coral (% cover per year)
  PARAMETER(coral_regrow_fast);  // Regrowth rate of fast coral (% cover per year)
  PARAMETER(log_sigma);          // log standard deviation for observation error
  Type sigma = exp(log_sigma) + Type(1e-8); // Ensuring a minimum standard deviation for numerical stability

  int n = time.size();
  vector<Type> cots_pred(n);   // Predicted COTS density
  vector<Type> slow_pred(n);   // Predicted slow coral cover
  vector<Type> fast_pred(n);   // Predicted fast coral cover

  // Initial conditions: using the first observations to seed predictions
  cots_pred[0] = cots_dat[0];   // Initial COTS
  slow_pred[0] = slow_dat[0];   // Initial slow coral
  fast_pred[0] = fast_dat[0];   // Initial fast coral

  Type nll = 0.0;  // Negative log likelihood

  // Iterate over time steps (note: predictions use only previous time step values)
  for(int t = 1; t < n; t++){
      // Equation 1: COTS dynamics (logistic growth minus impact of predation on corals)
      cots_pred[t] = cots_pred[t-1] 
                    + growth_rate_cots * cots_pred[t-1] * (1 - cots_pred[t-1]/(carrying_capacity + Type(1e-8)))
                    - predation_rate * (slow_pred[t-1] + fast_pred[t-1]) * efficiency;
      cots_pred[t] = smooth_max(cots_pred[t], Type(1e-8));

      // Equation 2: Slow coral dynamics with competition (regrowth limited by total coral cover)
      slow_pred[t] = slow_pred[t-1] 
                    + coral_regrow_slow * (100 - (slow_pred[t-1] + fast_pred[t-1]))
                    - predation_rate * cots_pred[t-1] * slow_pred[t-1] / (slow_pred[t-1] + Type(1e-8));
      slow_pred[t] = smooth_max(slow_pred[t], Type(1e-8));

      // Equation 3: Fast coral dynamics with competition (regrowth limited by total coral cover)
      fast_pred[t] = fast_pred[t-1] 
                    + coral_regrow_fast * (100 - (slow_pred[t-1] + fast_pred[t-1]))
                    - predation_rate * cots_pred[t-1] * fast_pred[t-1] / (fast_pred[t-1] + Type(1e-8));
      fast_pred[t] = smooth_max(fast_pred[t], Type(1e-8));

      // Likelihood: using lognormal error distributions to stabilize variability over orders of magnitude
      nll -= dnorm(log(cots_dat[t] + Type(1e-8)), log(cots_pred[t]), sigma, true);
      nll -= dnorm(log(slow_dat[t] + Type(1e-8)), log(slow_pred[t] + Type(1e-8)), sigma, true);
      nll -= dnorm(log(fast_dat[t] + Type(1e-8)), log(fast_pred[t] + Type(1e-8)), sigma, true);
  }

  // Reporting predictions for all time steps (_pred suffix to match _dat observations)
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);

  return nll;
}
