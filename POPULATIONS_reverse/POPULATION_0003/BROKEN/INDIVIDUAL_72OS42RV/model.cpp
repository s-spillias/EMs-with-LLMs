#include <TMB.hpp>

template<class Type>
Type my_dnorm(Type x, Type mean, Type sd){
    return -0.5 * log(2.0 * M_PI) - log(sd) - 0.5 * pow((x - mean)/sd, 2);
}

// 1. DATA INPUTS
// slow_dat: Observed slow-growing coral cover (%) [DATA_VECTOR]
// fast_dat: Observed fast-growing coral cover (%) [DATA_VECTOR]
// cots_abundance: Abundance of crown-of-thorns starfish (individuals/m2) [DATA_SCALAR]

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;
  
  // Data variables
  DATA_VECTOR(slow_dat);      // Observations for slow coral cover (%)
  DATA_VECTOR(fast_dat);      // Observations for fast coral cover (%)
  DATA_VECTOR(sst_dat);       // Sea surface temperature (°C) [external forcing]
  DATA_VECTOR(cotsimm_dat);   // COTS immigration rate (individuals/m2/year) [external forcing]
  DATA_SCALAR(cots_abundance); // Baseline COTS abundance (individuals/m2) [data scalar]
  Type sst_dummy = sst_dat.sum(); // Dummy variable to use sst_dat and avoid unused variable issue
  
  // 2. MODEL PARAMETERS
  // intercept_slow: Baseline slow coral cover (%) [PARAMETER, estimated from initial percent cover]
  // growth_slow: Intrinsic growth rate for slow corals (year^-1, bounded between 0 and 1)
  // cslow: Consumption rate of COTS on slow corals (per individual/m2 per year)
  PARAMETER(intercept_slow);
  PARAMETER(growth_slow);
  PARAMETER(cslow);
  
  // intercept_fast: Baseline fast coral cover (%) [PARAMETER]
  // growth_fast: Intrinsic growth rate for fast corals (year^-1, bounded between 0 and 1)
  // cfast: Consumption rate of COTS on fast corals (per individual/m2 per year)
  PARAMETER(intercept_fast);
  PARAMETER(growth_fast);
  PARAMETER(cfast);
  // growth_cots: Intrinsic growth rate for COTS (year^-1)
  // benefit_coral: Benefit to COTS from coral availability (unitless scaling factor)
  PARAMETER(growth_cots);
  PARAMETER(benefit_coral);
  
  // log_sigma variables for observation errors, transformed to ensure positivity
  // log_sigma_slow: Log standard deviation for slow coral observation error
  // log_sigma_fast: Log standard deviation for fast coral observation error
  PARAMETER(log_sigma_slow);
  PARAMETER(log_sigma_fast);
  
  Type sigma_slow = exp(log_sigma_slow) + Type(1e-8); // prevent sigma from being too close to zero
  Type sigma_fast = exp(log_sigma_fast) + Type(1e-8);
  
  // 3. PENALTY TERMS FOR BIOLOGICAL BOUNDS (smooth constraints)
  Type penalty = 0;
  
  // (1) Bound growth_slow between 0 and 1 (year^-1)
  if(growth_slow < Type(0)) penalty += pow(growth_slow - Type(0), 2);
  if(growth_slow > Type(1)) penalty += pow(growth_slow - Type(1), 2);
  
  // (2) Bound growth_fast between 0 and 1 (year^-1)
  if(growth_fast < Type(0)) penalty += pow(growth_fast - Type(0), 2);
  if(growth_fast > Type(1)) penalty += pow(growth_fast - Type(1), 2);
  
  // (3) Consumption rates must be non-negative.
  if(cslow < Type(0)) penalty += pow(cslow - Type(0), 2);
  if(cfast < Type(0)) penalty += pow(cfast - Type(0), 2);
  
  // 4. PREDICTED DYNAMICS FOR CORALS AND COTS USING RECURSIVE EQUATIONS
  // For i=0: initial conditions are given by intercept parameters for corals
  // and cots_abundance for COTS.
  int n = slow_dat.size();
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> cots_pred(n);
  slow_pred(0) = intercept_slow;
  fast_pred(0) = intercept_fast;
  cots_pred(0) = cots_abundance;
  for(int i = 1; i < n; i++){
      slow_pred(i) = slow_pred(i-1) * (1 + growth_slow) * exp(- cslow * cots_pred(i-1));
      slow_pred(i) = (slow_pred(i) > Type(1e-8)) ? slow_pred(i) : Type(1e-8);
      fast_pred(i) = fast_pred(i-1) * (1 + growth_fast) * exp(- cfast * cots_pred(i-1));
      fast_pred(i) = (fast_pred(i) > Type(1e-8)) ? fast_pred(i) : Type(1e-8);
      cots_pred(i) = cots_pred(i-1) * (1 + growth_cots) * (1 + benefit_coral * ((slow_pred(i-1) + fast_pred(i-1)) / 2)) + cotsimm_dat(i-1);
      cots_pred(i) = (cots_pred(i) > Type(1e-8)) ? cots_pred(i) : Type(1e-8);
  }
  
  // 5. LIKELIHOOD CALCULATION (using lognormal error distributions)
  // Equation 3: Likelihood contributions from observed coral cover vs. predictions
  Type nll = penalty; // Start with penalty terms
  
  // Loop over slow coral observations
  for(int i = 0; i < slow_dat.size(); i++){
      nll -= (my_dnorm(log(slow_dat[i] + Type(1e-8)), log(slow_pred(i)), sigma_slow) - log(slow_dat[i] + Type(1e-8)));
  }
  
  // Loop over fast coral observations
  for(int i = 0; i < fast_dat.size(); i++){
      nll -= (my_dnorm(log(fast_dat[i] + Type(1e-8)), log(fast_pred(i)), sigma_fast) - log(fast_dat[i] + Type(1e-8)));
  }
  
  // 6. REPORT MODEL PREDICTIONS
  ADREPORT(slow_pred); // Report predicted slow coral cover (%)
  ADREPORT(fast_pred); // Report predicted fast coral cover (%)
  
  return nll;
}
