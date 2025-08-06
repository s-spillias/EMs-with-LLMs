#include <TMB.hpp>

// TMB model for Crown of Thorns starfish dynamics
// This model simulates the impact of starfish consumption on slow- and fast-growing corals.
// Each model step includes smooth transitions and numerical stability measures (using small constants)
// to ensure the likelihood is robustly computed using lognormal error distributions.

template<class Type>
Type objective_function<Type>::operator() ()
{
    // DATA:
    DATA_VECTOR(slow_dat);   // Experimental slow-growing coral cover data (% cover)
    DATA_VECTOR(fast_dat);   // Experimental fast-growing coral cover data (% cover)
    DATA_VECTOR(cots_dat);   // Observed Crown-of-Thorns starfish abundance (individuals/m2)
    DATA_VECTOR(sst_dat);    // Sea-surface temperature data (°C) - external forcing
    DATA_VECTOR(cotsimm_dat); // Crown-of-Thorns immigration rate (individuals/m2/year) - external forcing

    // PARAMETERS:
    // consumption_slow: Consumption rate on slow-growing corals (per individual impact)
    // consumption_fast: Consumption rate on fast-growing corals (per individual impact)
    // growth_rate: Intrinsic growth rate of starfish (year^-1)
    // log_sigma: Log of standard deviation of observation error (ensures sigma > 0)
    PARAMETER(consumption_slow);
    PARAMETER(consumption_fast);
    PARAMETER(growth_rate);
    PARAMETER(log_sigma);

    // CONSTANTS:
    Type epsilon = Type(1e-8);  // Small constant to prevent division by zero

    // Transform log_sigma ensuring sigma never falls below epsilon
    Type sigma = exp(log_sigma) + epsilon;  

    int n = slow_dat.size();  // assuming all data vectors have the same length
    vector<Type> nll(1);
    nll(0) = 0;
    vector<Type> cots_pred(n);   // Predicted Crown-of-Thorns starfish abundance (individuals/m2)
    vector<Type> slow_pred(n);   // Predicted slow-growing coral cover (% cover)
    vector<Type> fast_pred(n);   // Predicted fast-growing coral cover (% cover)

    // Initial conditions: initialize predictions with observed data at time 0
    cots_pred(0) = cots_dat(0);
    slow_pred(0) = slow_dat(0);
    fast_pred(0) = fast_dat(0);

    // Recurrence equations:
    for(int i = 1; i < n; i++){
        // Equation 1: Starfish dynamics governed by intrinsic growth and immigration forcing
        // cots_pred(i) = previous abundance times growth_rate plus external immigration forcing
        cots_pred(i) = cots_pred(i-1) * growth_rate + cotsimm_dat(i);
        // Equation 2: Slow-growing coral dynamics affected by predation from starfish
        // Predicted by exponential decay based on previous predicted starfish abundance
        slow_pred(i) = slow_pred(i-1) * exp(-consumption_slow * cots_pred(i-1));
        // Equation 3: Fast-growing coral dynamics affected by predation from starfish
        fast_pred(i) = fast_pred(i-1) * exp(-consumption_fast * cots_pred(i-1));
    }

    // Likelihood calculation: compare predictions against observed data using lognormal likelihood
    for(int i = 0; i < n; i++){
        nll(0) -= dnorm(log(cots_dat(i) + epsilon), log(cots_pred(i) + epsilon), sigma, true)
               +  dnorm(log(slow_dat(i) + epsilon), log(slow_pred(i) + epsilon), sigma, true)
               +  dnorm(log(fast_dat(i) + epsilon), log(fast_pred(i) + epsilon), sigma, true);
    }

    // Reporting predicted values for further diagnostics
    REPORT(cots_pred);    // Predicted Crown-of-Thorns starfish abundance
    REPORT(slow_pred);    // Predicted slow-growing coral cover
    REPORT(fast_pred);    // Predicted fast-growing coral cover

    Type nll_total = nll(0);
    return nll_total;
}
