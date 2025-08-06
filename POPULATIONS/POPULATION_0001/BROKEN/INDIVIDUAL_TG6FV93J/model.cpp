#include <TMB.hpp> // Include TMB header

template<class Type>
Type objective_function<Type>::operator() ()
{
  // 1. DATA DECLARATION:
  // The following data vectors are provided in the observation data.
  DATA_VECTOR(Year);           // Year of observation (integer values)
  DATA_VECTOR(cots_dat);         // Observed adult Crown-of-Thorns starfish abundance (individuals/m2)
  DATA_VECTOR(fast_dat);         // Observed cover (%) for fast-growing coral (Acropora spp.)
  DATA_VECTOR(slow_dat);         // Observed cover (%) for slow-growing corals (Faviidae spp. & Porities spp.)

  // 2. PARAMETER DECLARATION:
  // Parameters controlling COTS dynamics and coral interactions.
  PARAMETER(p0_initial);       // [individuals/m2] initial COTS abundance (from expert estimate)
  PARAMETER(log_lambda);       // [unitless] outbreak magnitude modifier (log scale)
  PARAMETER(p1_growth);        // [year^-1] log intrinsic growth rate of COTS (literature-based)
  PARAMETER(p3_decay);         // [year^-1] natural decay rate of COTS population (literature-based)
  PARAMETER(p4_effc);          // [m2/(individual*year)] efficiency of predation on coral communities
  PARAMETER(p5_threshold);     // [individuals/m2] threshold COTS density triggering outbreak mechanisms
  PARAMETER(p7_fast_recovery); // [year^-1] recovery rate of fast-growing corals (Acropora spp.)
  PARAMETER(p8_slow_recovery); // [year^-1] recovery rate of slow-growing corals (Faviidae spp. & Porities spp.)
  PARAMETER(sigma_cots);       // [log scale error] minimum std. deviation for COTS lognormal likelihood
  PARAMETER(sigma_fast);       // [log scale error] minimum std. deviation for fast coral lognormal likelihood
  PARAMETER(sigma_slow);       // [log scale error] minimum std. deviation for slow coral lognormal likelihood
  PARAMETER(k_fast);           // Saturation constant for fast coral predation (unitless)
  PARAMETER(k_slow);           // Saturation constant for slow coral predation (unitless)


  // 3. INITIALIZATION OF STATE VARIABLES:
  int n = Year.size(); // Number of time steps
  vector<Type> cots_pred(n); // Predicted COTS abundance (individuals/m2)
  vector<Type> fast_pred(n); // Predicted fast coral cover (%)
  vector<Type> slow_pred(n); // Predicted slow coral cover (%)

  // Set initial conditions (we use provided data for coral as starting points)
  cots_pred(0) = p0_initial;             // (1) Initial COTS density
  fast_pred(0) = fast_dat(0);            // (2) Initial fast coral cover
  slow_pred(0) = slow_dat(0);            // (3) Initial slow coral cover

  // 4. MODEL DYNAMICS (Equations):
  // Equation descriptions:
  // [1] COTS dynamics: Outbreak magnitude is scaled by an outbreak effect function.
  //     The outbreak effect is computed as an exponential function modulated by a sigmoid threshold.
  // [2] Fast coral dynamics: Decline via predation proportional to COTS density with a recovery towards 100%.
  // [3] Slow coral dynamics: Similar to fast corals but with lower predation sensitivity.
  for(int t = 1; t < n; t++){
    // (1) Compute outbreak effect using a smooth sigmoid function (avoid hard cutoffs)
    Type outbreak_effect = exp(log_lambda) / (Type(1) + exp(-(cots_pred(t-1) - p5_threshold))) + Type(1e-8); // small constant for stability
    if(outbreak_effect < Type(1e-8)) outbreak_effect = Type(1e-8);

    // (2) COTS population dynamics: 
    // New COTS density = previous density + growth (amplified during outbreak) - natural decay.
    cots_pred(t) = cots_pred(t-1) + exp(p1_growth) * cots_pred(t-1) * outbreak_effect - p3_decay * cots_pred(t-1);
    if(cots_pred(t) < Type(1e-8)) cots_pred(t) = Type(1e-8);

    // (3) Fast coral dynamics:
    // Fast coral cover decreases due to predation with a saturating response (Michaelis–Menten form) and increases by recovery toward 100%.
    fast_pred(t) = fast_pred(t-1) - (p4_effc * cots_pred(t-1) * fast_pred(t-1))/(Type(1) + k_fast * fast_pred(t-1))
                   + p7_fast_recovery * (Type(100) - fast_pred(t-1));
    if(fast_pred(t) < Type(1e-8)) fast_pred(t) = Type(1e-8);

    // (4) Slow coral dynamics:
    // Slow coral cover experiences less predation pressure (scaled by 0.5) with a saturating predation term and recovers toward 100%.
    slow_pred(t) = slow_pred(t-1) - (p4_effc * Type(0.5) * cots_pred(t-1) * slow_pred(t-1))/(Type(1) + k_slow * slow_pred(t-1))
                   + p8_slow_recovery * (Type(100) - slow_pred(t-1));
    if(slow_pred(t) < Type(1e-8)) slow_pred(t) = Type(1e-8);
  }

  // 5. LIKELIHOOD CALCULATION:
  // For each time step, compute the negative log-likelihood using a lognormal distribution.
  // Data are log-transformed and a fixed sigma (with a minimum standard deviation) is included.
  Type nll = 0.0;
  for(int i = 0; i < n; i++){
      // (5) Likelihood for COTS observations:
      nll -= dnorm(log(cots_dat(i)+Type(1e-8)), log(cots_pred(i)+Type(1e-8)), sigma_cots, true);
      // (6) Likelihood for fast coral observations:
      nll -= dnorm(log(fast_dat(i)+Type(1e-8)), log(fast_pred(i)+Type(1e-8)), sigma_fast, true);
      // (7) Likelihood for slow coral observations:
      nll -= dnorm(log(slow_dat(i)+Type(1e-8)), log(slow_pred(i)+Type(1e-8)), sigma_slow, true);
  }

  // 6. REPORTING:
  // Report predictions so they can be compared to observed data.
  REPORT(cots_pred);  // Predicted COTS abundance over time
  REPORT(fast_pred);  // Predicted fast coral cover over time
  REPORT(slow_pred);  // Predicted slow coral cover over time

  return nll;
}
