#include <TMB.hpp>
#include <algorithm>

// 1. DATA section: Observations for coral cover (in percent)
template<class Type>
Type objective_function<Type>::operator() (void) {
  using namespace density;
  
  // Observed data vectors
  DATA_VECTOR(slow_dat);  // Slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);  // Fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);   // Sea Surface Temperature forcing factor
  DATA_VECTOR(cotsimm_dat); // COTS immigration forcing (individuals/m2/year)

  // 2. PARAMETERS section:
  // Parameter 1: r_slow (Intrinsic growth rate for slow-growing corals, year^-1)
  PARAMETER(r_slow);
  // Parameter 2: K_slow (Carrying capacity for slow-growing corals, percent cover)
  PARAMETER(K_slow);
  // Parameter 3: alpha_slow (Feeding consumption rate on slow corals, per unit biomass)
  PARAMETER(alpha_slow);
  // Parameter 4: r_fast (Intrinsic growth rate for fast-growing corals, year^-1)
  PARAMETER(r_fast);
  // Parameter 5: K_fast (Carrying capacity for fast-growing corals, percent cover)
  PARAMETER(K_fast);
  // Parameter 6: alpha_fast (Feeding consumption rate on fast corals, per unit biomass)
  PARAMETER(alpha_fast);
  // Parameter 7: cots_effect (Effect parameter for COTS predation intensity)
  PARAMETER(cots_effect);
  // Parameter X: r_cots (Intrinsic growth rate for COTS, year^-1)
  PARAMETER(r_cots);
  // Parameter 8: log_sigma_slow (Log-standard deviation for observation error in slow coral cover)
  PARAMETER(log_sigma_slow);
  // Parameter 9: log_sigma_fast (Log-standard deviation for observation error in fast coral cover)
  PARAMETER(log_sigma_fast);

  // 3. Transform parameters and enforce numerical stability for standard deviations:
  Type sigma_slow_tmp = exp(log_sigma_slow);
  Type sigma_slow = sigma_slow_tmp < Type(1e-8) ? Type(1e-8) : sigma_slow_tmp;
  Type sigma_fast_tmp = exp(log_sigma_fast);
  Type sigma_fast = sigma_fast_tmp < Type(1e-8) ? Type(1e-8) : sigma_fast_tmp;

  // 4. Initialize negative log likelihood (nll)
  Type nll = 0;

  // 5. Model equations:
  // Equation 1: slow_pred[t] = slow_dat[t-1] + r_slow*slow_dat[t-1]*(1 - slow_dat[t-1]/K_slow)
  //                - alpha_slow*cots_effect*slow_dat[t-1] / (1 + alpha_slow*slow_dat[t-1] + 1e-8)
  // Equation 2: fast_pred[t] = fast_dat[t-1] + r_fast*fast_dat[t-1]*(1 - fast_dat[t-1]/K_fast)
  //                - alpha_fast*cots_effect*fast_dat[t-1] / (1 + alpha_fast*fast_dat[t-1] + 1e-8)
  // Equation 3: Likelihood is calculated using a lognormal error distribution:
  //             log(observation + 1e-8) ~ N(log(prediction + 1e-8), sigma)

  int n = slow_dat.size();
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> cots_pred(n);
  
  // Initialize predictions with the first observation for continuity
  slow_pred[0] = slow_dat[0];
  fast_pred[0] = fast_dat[0];
  // Initialize COTS prediction with the first immigration forcing value
  cots_pred[0] = cotsimm_dat[0];

  for(int t = 1; t < n; t++){
    // Update COTS prediction: exponential growth plus forcing immigration
    cots_pred[t] = cots_pred[t-1] * (Type(1) + r_cots) + cotsimm_dat[t-1];

    // Update slow-growing coral prediction using previous prediction, SST forcing, and COTS impact
    slow_pred[t] = slow_pred[t-1] * (Type(1) + r_slow * sst_dat[t-1] * (Type(1) - slow_pred[t-1] / (K_slow + Type(1e-8))))
                  * (Type(1) - alpha_slow * cots_pred[t-1]);

    // Update fast-growing coral prediction using previous prediction, SST forcing, and COTS impact
    fast_pred[t] = fast_pred[t-1] * (Type(1) + r_fast * sst_dat[t-1] * (Type(1) - fast_pred[t-1] / (K_fast + Type(1e-8))))
                  * (Type(1) - alpha_fast * cots_pred[t-1]);

    // Likelihood: model observation on log scale using predictions
    nll -= dnorm(log(slow_dat[t] + Type(1e-8)), log(slow_pred[t] + Type(1e-8)), sigma_slow, true);
    nll -= dnorm(log(fast_dat[t] + Type(1e-8)), log(fast_pred[t] + Type(1e-8)), sigma_fast, true);
  }

  // 6. Reporting important model variables:
  REPORT(slow_pred);       // Predicted slow-growing coral cover (%)
  REPORT(fast_pred);       // Predicted fast-growing coral cover (%)
  REPORT(cots_pred);       // Predicted COTS population (individuals/m2)
  REPORT(r_slow);          // Intrinsic growth rate for slow corals (year^-1)
  REPORT(K_slow);          // Carrying capacity for slow corals (%)
  REPORT(alpha_slow);      // Feeding consumption rate on slow corals
  REPORT(r_fast);          // Intrinsic growth rate for fast corals (year^-1)
  REPORT(K_fast);          // Carrying capacity for fast corals (%)
  REPORT(alpha_fast);      // Feeding consumption rate on fast corals
  REPORT(cots_effect);     // COTS predation effect parameter
  REPORT(r_cots);          // Intrinsic growth rate for COTS (year^-1)
  
  return nll;
}
