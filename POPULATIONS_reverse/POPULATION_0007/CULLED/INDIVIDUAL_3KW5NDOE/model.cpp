#include <TMB.hpp>
#include <cmath>
  
 // Helper function to compute log-density of a lognormal distribution
 template<class Type>
 Type log_lnorm(const Type& x, const Type& meanlog, const Type& sdlog) {
   // Log-density for a lognormal: log[f(x)] = -log(x) - log(sdlog) - 0.5*log(2*pi) - (log(x)-meanlog)^2/(2*sdlog^2)
   constexpr double PI_val = 3.14159265358979323846;
   return -log(x) - log(sdlog) - 0.5*log(2.0 * PI_val) - pow(log(x) - meanlog, 2.0) / (2.0 * sdlog * sdlog);
 }

// TMB model for Crown-of-Thorns starfish feeding on corals on the Great Barrier Reef
// Updated Equations:
// We use recursive relationships based on external forcing and previous predictions:
// 1. slow_pred: Predicted slow-growing coral cover dynamics using previous state, intrinsic growth,
//    impact from COTS, and SST effects:
//    slow_pred[0] = slow_dat[0] (initial condition)
//    slow_pred[i] = slow_pred[i-1] * exp(growth_slow * (1 - impact_slow * cots_pred[i-1]) + effect_sst_slow * sst_dat[i-1])
// 2. fast_pred: Predicted fast-growing coral cover dynamics using previous state, intrinsic growth,
//    impact from COTS, and SST effects:
//    fast_pred[0] = fast_dat[0] (initial condition)
//    fast_pred[i] = fast_pred[i-1] * exp(growth_fast * (1 - impact_fast * cots_pred[i-1]) + effect_sst_fast * sst_dat[i-1])
// 3. cots_pred: Predicted Crown-of-Thorns dynamics using previous abundance with mortality and
//    immigration forcing effects:
//    cots_pred[0] = cots_dat[0] (initial condition)
//    cots_pred[i] = cots_pred[i-1] * exp(-mortality + cotsimm_effect * cotsimm_dat[i-1])
// 
// Each parameter has comments explaining its role, units, and determination method.

template<class Type>
Type objective_function<Type>::operator() () {
  // ------------------------ Data ------------------------
  DATA_VECTOR(slow_dat);   // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);   // Observed fast-growing coral cover (%)
  DATA_VECTOR(cots_dat);   // Observed Crown-of-Thorns starfish abundance (individuals/m2)
  DATA_VECTOR(sst_dat);    // Sea-Surface Temperature in Celsius (external forcing)
  DATA_VECTOR(cotsimm_dat); // Crown-of-Thorns immigration rate (individuals/m2/year, external forcing)
  
  // --------------------- Parameters ---------------------
  PARAMETER(alpha);          // Consumption rate on slow corals (per starfish per time unit; estimated from feeding observations)
  PARAMETER(beta);           // Consumption rate on fast corals (per starfish per time unit; estimated from feeding observations)
  PARAMETER(mortality);      // Mortality rate of Crown-of-Thorns (per year; based on expert opinion)
  PARAMETER(log_sigma_coral); // Log standard deviation for coral cover observations (log(%), ensures positive stdev)
  PARAMETER(log_sigma_cots);  // Log standard deviation for Crown-of-Thorns observations (log(individuals/m2), ensures positive stdev)
  
  // New dynamic model parameters
  PARAMETER(growth_slow);      // Intrinsic growth rate of slow corals (year^-1)
  PARAMETER(impact_slow);      // Impact of COTS on slow corals (per individual)
  PARAMETER(effect_sst_slow);  // Effect of SST on slow corals (per degree Celsius)
  PARAMETER(growth_fast);      // Intrinsic growth rate of fast corals (year^-1)
  PARAMETER(impact_fast);      // Impact of COTS on fast corals (per individual)
  PARAMETER(effect_sst_fast);  // Effect of SST on fast corals (per degree Celsius)
  PARAMETER(cotsimm_effect);   // Effect of immigration forcing on COTS population (per unit rate)
  PARAMETER(handling_slow);    // Handling time parameter for saturating predation on slow corals (per individual)
  PARAMETER(handling_fast);    // Handling time parameter for saturating predation on fast corals (per individual)
  PARAMETER(K_slow);           // Carrying capacity for slow-growing corals (%)
  PARAMETER(K_fast);           // Carrying capacity for fast-growing corals (%)
  
  // ------------------ Numerical Stability ------------------
  Type sigma_coral = exp(log_sigma_coral) + Type(1e-8); // Minimum sigma to avoid division by zero
  Type sigma_cots = exp(log_sigma_cots) + Type(1e-8);
  
  // -------------------- Initialize Likelihood --------------------
  Type nll = 0.0;
  int n = slow_dat.size();
  
  // Predicted variables corresponding to observations (_pred)
  vector<Type> slow_pred(n);   // Predicted slow-growing coral cover (%)
  vector<Type> fast_pred(n);   // Predicted fast-growing coral cover (%)
  vector<Type> cots_pred(n);   // Predicted Crown-of-Thorns abundance (individuals/m2)
  
  // ------------------- Model Equations & Likelihood -------------------
  // Initialize predictions with the first observation as initial condition
  slow_pred[0] = slow_dat[0];
  fast_pred[0] = fast_dat[0];
  cots_pred[0] = cots_dat[0];
  
  // Iterate over time steps (starting from 1) using external forcing and previous predictions:
  for(int i = 1; i < n; i++){
    // 1. Slow coral dynamics: previous state modulated by intrinsic growth, reduced by prior COTS impact,
    //    and influenced by SST forcing.
    slow_pred[i] = slow_pred[i-1] * exp(growth_slow * (1 - slow_pred[i-1] / K_slow) - (impact_slow * cots_pred[i-1])/(1 + handling_slow * cots_pred[i-1]) + effect_sst_slow * sst_dat[i-1]);
    
    // 2. Fast coral dynamics: previous state modulated by intrinsic growth, reduced by prior COTS impact,
    //    and influenced by SST forcing.
    fast_pred[i] = fast_pred[i-1] * exp(growth_fast * (1 - fast_pred[i-1] / K_fast) - (impact_fast * cots_pred[i-1])/(1 + handling_fast * cots_pred[i-1]) + effect_sst_fast * sst_dat[i-1]);
    
    // 3. COTS dynamics: previous abundance affected by mortality and boosted by immigration forcing.
    cots_pred[i] = cots_pred[i-1] * exp(-mortality + cotsimm_effect * cotsimm_dat[i-1]);
    
    // Likelihood contributions using a lognormal error distribution:
    nll -= log_lnorm(slow_dat[i] + Type(1e-8), log(slow_pred[i] + Type(1e-8)), sigma_coral);
    nll -= log_lnorm(fast_dat[i] + Type(1e-8), log(fast_pred[i] + Type(1e-8)), sigma_coral);
    nll -= log_lnorm(cots_dat[i] + Type(1e-8), log(cots_pred[i] + Type(1e-8)), sigma_cots);
  }
  
  // --------------------- Reporting ---------------------
  REPORT(slow_pred);    // Predicted slow-growing coral cover (%)
  REPORT(fast_pred);    // Predicted fast-growing coral cover (%)
  REPORT(cots_pred);    // Predicted Crown-of-Thorns abundance (individuals/m2)
  REPORT(alpha);        // Consumption rate on slow corals (per starfish per time unit)
  REPORT(beta);         // Consumption rate on fast corals (per starfish per time unit)
  REPORT(mortality);    // Mortality rate of Crown-of-Thorns (per year)
  REPORT(sigma_coral);// Fixed standard deviation for coral cover observations (%)
  REPORT(sigma_cots); // Fixed standard deviation for Crown-of-Thorns observations (individuals/m2)
  
  return nll;
}
