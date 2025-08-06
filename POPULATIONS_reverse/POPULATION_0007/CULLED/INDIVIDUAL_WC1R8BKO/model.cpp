#include <TMB.hpp>

// Template Model Builder model for COTS-coral dynamics on the Great Barrier Reef.
// The model equations are described below:
// 1. cots_pred = exp(growth_cots * ( rate_slow * slow_dat + rate_fast * fast_dat ) - mort)
//    (Starfish abundance increases with available coral food, reduced by mortality)
// 2. slow_pred = slow_dat * exp(- rate_slow * cots_dat)
//    (Slow-growing coral cover declines with increasing COTS abundance)
// 3. fast_pred = fast_dat * exp(- rate_fast * cots_dat)
//    (Fast-growing coral cover declines with increasing COTS abundance)

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA: Observed data vectors (units in comments)
  DATA_VECTOR(slow_dat);      // Slow-growing coral cover (%) - data from field surveys
  DATA_VECTOR(fast_dat);      // Fast-growing coral cover (%) - data from field surveys
  DATA_VECTOR(cots_dat);      // Adult Crown-of-Thorns starfish abundance (individuals/m2) - survey counts
  DATA_VECTOR(sst_dat);       // Sea surface temperature (°C) - external forcing variable
  DATA_VECTOR(cotsimm_dat);   // COTS immigration forcing (individuals/m2/year) - external forcing variable

  // PARAMETERS:
  PARAMETER(growth_cots);     // Intrinsic growth rate of COTS (year^-1)
  PARAMETER(rate_slow);       // Feeding rate coefficient on slow-growing corals (per % per year)
  PARAMETER(rate_fast);       // Feeding rate coefficient on fast-growing corals (per % per year)
  PARAMETER(mort);            // Mortality rate of COTS (dimensionless, bounded between 0 and 1)
  PARAMETER(log_sigma_cots);  // Log of observational error standard deviation for COTS (log-scale)
  PARAMETER(log_sigma_slow);  // Log of observational error standard deviation for slow coral (% cover) (log-scale)
  PARAMETER(log_sigma_fast);  // Log of observational error standard deviation for fast coral (% cover) (log-scale)
  PARAMETER(growth_slow);     // Intrinsic growth rate for slow-growing corals (year^-1) for initial state
  PARAMETER(impact_slow);     // Impact rate of COTS on slow-growing corals (m2/individual per year)
  PARAMETER(growth_fast);     // Intrinsic growth rate for fast-growing corals (year^-1) for initial state
  PARAMETER(impact_fast);     // Impact rate of COTS on fast-growing corals (m2/individual per year)

  // Transform sigma values ensuring a fixed minimum to avoid numerical issues.
  Type sigma_cots = (exp(log_sigma_cots) > Type(1e-4) ? exp(log_sigma_cots) : Type(1e-4));   // sigma for COTS data
  Type sigma_slow = (exp(log_sigma_slow) > Type(1e-4) ? exp(log_sigma_slow) : Type(1e-4));   // sigma for slow-growing coral data
  Type sigma_fast = (exp(log_sigma_fast) > Type(1e-4) ? exp(log_sigma_fast) : Type(1e-4));   // sigma for fast-growing coral data

  int n = slow_dat.size();  // Number of observations (assumed equal across data vectors)
  vector<Type> cots_pred(n);  // [Prediction 1] Predicted COTS abundance (individuals/m2)
  vector<Type> slow_pred(n);  // [Prediction 2] Predicted slow-growing coral cover (%)
  vector<Type> fast_pred(n);  // [Prediction 3] Predicted fast-growing coral cover (%)
  
  Type nll = 0.0;  // Initialize negative log-likelihood

  // Loop through each observation to compute predictions and likelihood contributions
  for(int i = 0; i < n; i++){
    if(i == 0){
      // Initial conditions using forcing variables:
      // [1] COTS initial prediction from SST and immigration forcing.
      cots_pred(i) = exp(growth_cots * sst_dat(i) + cotsimm_dat(i) - mort + Type(1e-8));
      // [2] Initial slow-growing coral cover prediction from its intrinsic growth parameter.
      slow_pred(i) = exp(growth_slow);
      // [3] Initial fast-growing coral cover prediction from its intrinsic growth parameter.
      fast_pred(i) = exp(growth_fast);
    } else {
      // Recursive predictions using previous time step predictions and forcing variables:
      // [1] COTS prediction: previous COTS adjusted by current forcing and mortality.
      cots_pred(i) = cots_pred(i-1) * exp(growth_cots * sst_dat(i) + cotsimm_dat(i) - mort + Type(1e-8));
      // [2] Slow-growing coral prediction: previous value modified by intrinsic growth and impact from COTS.
      slow_pred(i) = slow_pred(i-1) * (1 + growth_slow - impact_slow * cots_pred(i-1) + Type(1e-8));
      slow_pred(i) = (slow_pred(i) > Type(1e-8)) ? slow_pred(i) : Type(1e-8);
      // [3] Fast-growing coral prediction: previous value modified by intrinsic growth and impact from COTS.
      fast_pred(i) = fast_pred(i-1) * (1 + growth_fast - impact_fast * cots_pred(i-1) + Type(1e-8));
      fast_pred(i) = (fast_pred(i) > Type(1e-8)) ? fast_pred(i) : Type(1e-8);
    }
    
    // Likelihood calculation using lognormal error distributions
    // (Data are log-transformed to accommodate variability over orders of magnitude.)
    nll -= dnorm(log(cots_dat(i) + Type(1e-8)), log(cots_pred(i) + Type(1e-8)), sigma_cots, true);  // COTS data
    nll -= dnorm(log(slow_dat(i) + Type(1e-8)), log(slow_pred(i) + Type(1e-8)), sigma_slow, true);  // Slow coral data
    nll -= dnorm(log(fast_dat(i) + Type(1e-8)), log(fast_pred(i) + Type(1e-8)), sigma_fast, true);  // Fast coral data
  }
  
  // Smooth penalties for biologically meaningful parameter ranges:
  // 1. growth_cots, rate_slow, and rate_fast must be non-negative.
  if(growth_cots < Type(0)){
    nll += pow(growth_cots, 2) * 1e6;  // Heavy penalty for negative growth rate
  }
  if(rate_slow < Type(0)){
    nll += pow(rate_slow, 2) * 1e6;    // Heavy penalty for negative feeding rate on slow corals
  }
  if(rate_fast < Type(0)){
    nll += pow(rate_fast, 2) * 1e6;    // Heavy penalty for negative feeding rate on fast corals
  }
  // 2. mort should be between 0 and 1.
  if(mort < Type(0)){
    nll += pow(mort, 2) * 1e6;         // Heavy penalty for negative mortality rate
  }
  if(mort > Type(1)){
    nll += pow(mort - Type(1), 2) * 1e6; // Heavy penalty for mortality rate above 1
  }
  
  // Reporting predicted values and key parameters for diagnostic purposes:
  REPORT(cots_pred);     // [Report] Predicted COTS abundance (individuals/m2)
  REPORT(slow_pred);     // [Report] Predicted slow-growing coral cover (%)
  REPORT(fast_pred);     // [Report] Predicted fast-growing coral cover (%)
  ADREPORT(growth_cots); // [ADREPORT] Intrinsic growth rate of COTS (year^-1)
  ADREPORT(rate_slow);   // [ADREPORT] Feeding rate on slow-growing corals (/ % per year)
  ADREPORT(rate_fast);   // [ADREPORT] Feeding rate on fast-growing corals (/ % per year)
  ADREPORT(mort);        // [ADREPORT] Mortality rate of COTS (dimensionless)
  ADREPORT(sigma_cots);  // [ADREPORT] Observational error for COTS data
  ADREPORT(sigma_slow);  // [ADREPORT] Observational error for slow-growing coral data
  ADREPORT(sigma_fast);  // [ADREPORT] Observational error for fast-growing coral data
  
  return nll;
}
