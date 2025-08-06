#include <TMB.hpp>

// Template Model Builder implementation for predicting Crown-of-Thorns starfish dynamics
// and their ecological impacts on coral communities.
template<class Type>
Type objective_function<Type>::operator() ()
{
    // Data section:
    DATA_VECTOR(Year); // Year (time variable in years)
    DATA_VECTOR(cots_dat); // Observed Crown-of-Thorns starfish density (individuals/m^2)
    DATA_VECTOR(slow_dat); // Observed slow-growing coral cover (%) (Faviidae and Porites)
    DATA_VECTOR(fast_dat); // Observed fast-growing coral cover (%) (Acropora)

    // Parameters for starfish dynamics:
    PARAMETER(r); // (year^-1) Intrinsic growth rate of COTS, estimated from literature/expert opinion
    PARAMETER(K); // (individuals/m^2) Carrying capacity of COTS, based on reef conditions
    PARAMETER(a1); // (year^-1) Predation/damage coefficient representing effect of slow coral on COTS dynamics

    // Parameters for slow coral dynamics:
    PARAMETER(g_s);    // (year^-1) Intrinsic growth rate of slow-growing corals, from field estimates
    PARAMETER(slow_K); // (%) Maximum cover for slow-growing coral species
    PARAMETER(a2);     // (year^-1) Damage rate of slow coral due to COTS predation
    PARAMETER(b);      // (unitless) Saturation parameter in COTS consumption of slow corals
    PARAMETER(alpha);  // (unitless) Competition coefficient representing inhibition of slow-growing coral growth by fast-growing corals
    PARAMETER(delta);  // (year^-1) Additional mortality rate in COTS due to indirect fast coral predation effects
    PARAMETER(M);  // (unitless) Half-saturation constant for total coral cover modulating COTS reproduction efficiency
    PARAMETER(n_eff);  // (unitless) Hill exponent for the efficiency term in COTS reproduction, capturing non-linear resource uptake

    // Parameters for fast coral dynamics:
    PARAMETER(g_f);     // (year^-1) Intrinsic growth rate of fast-growing corals, from reef monitoring data
    PARAMETER(fast_K);  // (%) Maximum cover for fast-growing coral species
    PARAMETER(a3);      // (year^-1) Damage rate of fast coral due to COTS predation
    PARAMETER(c);       // (unitless) Saturation parameter in COTS consumption of fast corals

    // Likelihood observation error standard deviations:
    PARAMETER(sigma_cots); // (log-scale SD) Observation error for COTS density (lognormal likelihood)
    PARAMETER(sigma_slow); // (log-scale SD) Observation error for slow coral cover (lognormal likelihood)
    PARAMETER(sigma_fast); // (log-scale SD) Observation error for fast coral cover (lognormal likelihood)
    PARAMETER(E); // (unitless) Environmental quality modifier affecting coral growth rates

    // Number of time steps
    int n = Year.size();

    vector<Type> cots_pred(n); // Predicted COTS density at each time step
    vector<Type> slow_pred(n); // Predicted slow coral cover (%) at each time step
    vector<Type> fast_pred(n); // Predicted fast coral cover (%) at each time step

    // Initialize with the first observed values to avoid data leakage
    cots_pred(0) = cots_dat(0);
    slow_pred(0) = slow_dat(0);
    fast_pred(0) = fast_dat(0);

    // Model equations for time steps t = 1 .. n-1.
    for(int t = 1; t < n; t++){
        // Equation 1: COTS dynamics (numbered description 1)
        // 1. Logistic growth modulated by carrying capacity K.
        // 2. Reduction due to interaction with slow coral (saturating predation effect).
        // 3. Additional mortality due to fast-coral associated predation (indirect pathway).
        {
            Type total_coral = slow_pred(t-1) + fast_pred(t-1);
            Type efficiency = pow(total_coral, n_eff) / (pow(M, n_eff) + pow(total_coral, n_eff) + Type(1e-8));
            cots_pred(t) = cots_pred(t-1) 
                      + ( r * efficiency * cots_pred(t-1) * (1 - cots_pred(t-1) / (K + Type(1e-8))) 
                          - a1 * cots_pred(t-1) * slow_pred(t-1) / (1 + b * slow_pred(t-1) + Type(1e-8))
                          - delta * cots_pred(t-1) * fast_pred(t-1) / (1 + c * fast_pred(t-1) + Type(1e-8)) );
        }

        // Equation 2: Slow coral dynamics (numbered description 2)
        // 1. Logistic growth with maximum cover slow_K modified by competition from fast-growing corals.
        //    This term incorporates competitive inhibition using the new parameter "alpha" and is scaled by environmental quality.
        // 2. Reduction due to COTS predation using a saturating function.
        slow_pred(t) = slow_pred(t-1) 
                      + ( g_s * E * slow_pred(t-1) * (1 - (slow_pred(t-1) + alpha * fast_pred(t-1)) / (slow_K + Type(1e-8))) 
                          - a2 * cots_pred(t-1) * slow_pred(t-1) / (1 + b * slow_pred(t-1) + Type(1e-8)) );

        // Equation 3: Fast coral dynamics (numbered description 3)
        // 1. Logistic growth with carrying capacity fast_K scaled by environmental conditions.
        // 2. Reduction due to COTS predation with a separate saturation parameter.
        fast_pred(t) = fast_pred(t-1) 
                      + ( g_f * E * fast_pred(t-1) * (1 - fast_pred(t-1) / (fast_K + Type(1e-8))) 
                          - a3 * cots_pred(t-1) * fast_pred(t-1) / (1 + c * fast_pred(t-1) + Type(1e-8)) );
    }

    // Log-likelihood calculation (using lognormal likelihood for strictly positive data)
    Type jnll = 0.0;
    for(int t = 1; t < n; t++){
        // Likelihood for COTS density
        jnll -= ( dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t) + Type(1e-8)), sigma_cots, true)
                  - log(cots_dat(t) + Type(1e-8)) );
        // Likelihood for slow coral cover
        jnll -= ( dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred(t) + Type(1e-8)), sigma_slow, true)
                  - log(slow_dat(t) + Type(1e-8)) );
        // Likelihood for fast coral cover
        jnll -= ( dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred(t) + Type(1e-8)), sigma_fast, true)
                  - log(fast_dat(t) + Type(1e-8)) );
    }

    // Report predicted time-series for observations (_pred variables)
    REPORT(cots_pred);
    REPORT(slow_pred);
    REPORT(fast_pred);

    /*
      Equation descriptions:
      1. cots_pred(t): Predicted Crown-of-Thorns starfish density using logistic growth with predation
         reduction based on slow coral availability.
      2. slow_pred(t): Predicted slow-growing coral cover evolving via logistic growth diminished by COTS grazing.
      3. fast_pred(t): Predicted fast-growing coral cover evolving via logistic growth diminished by COTS grazing.
    */

    return jnll;
}
