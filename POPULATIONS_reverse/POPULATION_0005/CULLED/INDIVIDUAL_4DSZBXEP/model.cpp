#include <TMB.hpp>
  
// Custom lognormal density function
template<class Type>
Type dlnorm(Type x, Type meanlog, Type sd, int give_log = 1){
  Type logdensity = -0.5 * log(2.0 * M_PI) - log(sd) - log(x) - 0.5 * pow((log(x) - meanlog)/sd, 2);
  return give_log ? logdensity : exp(logdensity);
}

// TMB model for Crown-of-Thorns starfish feeding dynamics
// Equation descriptions:
// 1. The model predicts COTS abundance using a lognormal likelihood.
// 2. Intrinsic growth (exp(log_growth_rate)) scales the predation effects.
// 3. Predation on slow (exp(log_predation_rate_slow)) and fast (exp(log_predation_rate_fast)) corals is blended via a smooth sigmoid function.
// 4. A small constant (1e-8) is added to predictions to ensure numerical stability.

template<class Type>
Type objective_function<Type>::operator() ()
{
    // Data inputs: observed values
    DATA_VECTOR(slow_dat);  // Slow-growing coral cover (%) - observation data
    DATA_VECTOR(fast_dat);  // Fast-growing coral cover (%) - observation data
    DATA_VECTOR(cots_dat);  // COTS abundance (ind./m^2) - observation data
    DATA_VECTOR(sst_dat);   // Sea-surface temperature (°C) from forcing data
    DATA_VECTOR(cotsimm_dat); // COTS immigration rate (ind./m^2/year) from forcing data

    // Parameters (to be estimated)
    // log_growth_rate: Log intrinsic growth rate of COTS (year^-1)
    // log_predation_rate_slow: Log impact factor from slow-growing corals (per individual per % cover)
    // log_predation_rate_fast: Log impact factor from fast-growing corals (per individual per % cover)
    // log_sd: Log of the observation error standard deviation (unitless, log scale)
    // cots0: Initial COTS abundance (ind./m^2)
    // slow0: Initial slow-growing coral cover (%)
    // fast0: Initial fast-growing coral cover (%)
    // log_growth_rate_slow: Log intrinsic growth rate of slow-growing corals (year^-1)
    // log_growth_rate_fast: Log intrinsic growth rate of fast-growing corals (year^-1)
    // impact_slow: Impact parameter of COTS on slow-growing corals (per ind. per % cover)
    // impact_fast: Impact parameter of COTS on fast-growing corals (per ind. per % cover)
    // gamma: Modifier for sea-surface temperature effects on COTS predation
    PARAMETER(log_growth_rate);
    PARAMETER(log_predation_rate_slow);
    PARAMETER(log_predation_rate_fast);
    PARAMETER(log_sd);
    PARAMETER(cots0);
    PARAMETER(slow0);
    PARAMETER(fast0);
    PARAMETER(log_growth_rate_slow);
    PARAMETER(log_growth_rate_fast);
    PARAMETER(impact_slow);
    PARAMETER(impact_fast);
    PARAMETER(gamma);

    // Transform parameters to natural scale
    Type growth_rate = exp(log_growth_rate);  // COTS intrinsic growth rate (year^-1)
    Type predation_rate_slow = exp(log_predation_rate_slow); // Impact factor from slow-growing corals (per individual per % cover)
    Type predation_rate_fast = exp(log_predation_rate_fast); // Impact factor from fast-growing corals (per individual per % cover)
    Type growth_rate_slow = exp(log_growth_rate_slow); // Slow-growing coral intrinsic growth rate (year^-1)
    Type growth_rate_fast = exp(log_growth_rate_fast); // Fast-growing coral intrinsic growth rate (year^-1)
    Type sd = exp(log_sd) + Type(1e-8); // observation error standard deviation with small constant for stability

    // Initialize prediction vectors (assume all data vectors are of the same length)
    int n = cots_dat.size();
    vector<Type> cots_pred(n);
    vector<Type> slow_pred(n);
    vector<Type> fast_pred(n);
    Type nll = Type(0);

    // Set initial conditions from parameters
    cots_pred(0) = cots0;
    slow_pred(0) = slow0;
    fast_pred(0) = fast0;

    // Recursive prediction equations (for i >= 1) using only external forcing and previous predictions:
    // (1) slow_pred(i) = slow_pred(i-1) * (1 + growth_rate_slow) * exp(- impact_slow * cots_pred(i-1)) + 1e-8;
    // (2) fast_pred(i) = fast_pred(i-1) * (1 + growth_rate_fast) * exp(- impact_fast * cots_pred(i-1)) + 1e-8;
    // (3) cots_pred(i) = cots_pred(i-1) * (1 + growth_rate * (1 + gamma*(sst_dat(i-1) - Type(28))) *
    //                     ( slow_pred(i-1)*predation_rate_slow + fast_pred(i-1)*predation_rate_fast )) + cotsimm_dat(i) + 1e-8;
    for(int i = 1; i < n; i++){
        slow_pred(i) = slow_pred(i-1) * (Type(1) + growth_rate_slow) * exp(- impact_slow * cots_pred(i-1)) + Type(1e-8);
        fast_pred(i) = fast_pred(i-1) * (Type(1) + growth_rate_fast) * exp(- impact_fast * cots_pred(i-1)) + Type(1e-8);
        cots_pred(i) = cots_pred(i-1) * (Type(1) + growth_rate * (Type(1) + gamma*(sst_dat(i-1) - Type(28))) *
                        ( slow_pred(i-1)*predation_rate_slow + fast_pred(i-1)*predation_rate_fast )) + cotsimm_dat(i) + Type(1e-8);
    }

    // Likelihood: Compare predicted values with observed data using lognormal error structure
    // (Assuming observations start from time step 1; initial conditions are set by parameters)
    for(int i = 1; i < n; i++){
        nll -= dlnorm(slow_dat(i), log(slow_pred(i)), sd, true);
        nll -= dlnorm(fast_dat(i), log(fast_pred(i)), sd, true);
        nll -= dlnorm(cots_dat(i), log(cots_pred(i)), sd, true);
    }

    // Report outputs for further analysis in TMB
    REPORT(cots_pred);
    REPORT(growth_rate);
    REPORT(predation_rate_slow);
    REPORT(predation_rate_fast);
    
    return nll;
}
