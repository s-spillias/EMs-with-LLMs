#include <TMB.hpp>

// Template Model Builder (TMB) model for Crown-of-Thorns starfish and coral dynamics
// 1. MODEL EQUATIONS:
//    (1) slow_pred: Predicted slow-growing coral cover equals the observed value (placeholder model).
//    (2) fast_pred: Predicted fast-growing coral cover equals the observed value (placeholder model).
//    (3) cots_pred: Predicted crown-of-thorns starfish abundance is modeled as an exponential function of coral covers.
 // 2. PARAMETERS:
 //    - log_sd: Log of the observation standard deviation to ensure positive values (units: log-scale).
 //    - slow_init: Initial slow-growing coral cover (%)
 //    - fast_init: Initial fast-growing coral cover (%)
 //    - cots_init: Initial crown-of-thorns starfish abundance (individuals/m^2)
 //    - growth_rate_slow: Growth rate multiplier for slow-growing corals, influenced by SST.
 //    - impact_rate_slow: Impact sensitivity of slow-growing corals to starfish abundance.
 //    - growth_rate_fast: Growth rate multiplier for fast-growing corals, influenced by SST.
 //    - impact_rate_fast: Impact sensitivity of fast-growing corals to starfish abundance.
 //    - growth_rate_cots: Growth rate multiplier for crown-of-thorns starfish, influenced by immigration forcing.
 //    - impact_coral: Impact sensitivity of starfish abundance to coral cover.
//
// IMPORTANT CONSIDERATIONS:
//    * Numerical stability is ensured via a small constant (epsilon = 1e-8).
//    * Smooth transitions are applied using divisions that include epsilon.
//    * All observations are included in the likelihood with lognormal distributions after log-transformation.
//    * Smooth penalties (Gaussian priors) are applied to bound parameters within meaningful ranges.
template<class Type>
Type objective_function<Type>::operator() ()
{
    using namespace density;
    
    // DATA
    DATA_VECTOR(slow_dat);      // Observed slow-growing coral cover (%) (for likelihood)
    DATA_VECTOR(fast_dat);      // Observed fast-growing coral cover (%) (for likelihood)
    DATA_VECTOR(cots_dat);      // Observed crown-of-thorns starfish abundance (individuals/m^2) (for likelihood)
    DATA_VECTOR(sst_dat);       // Sea-surface temperature (°C) used as external forcing for coral growth
    DATA_VECTOR(cotsimm_dat);   // Crown-of-thorns immigration rate (individuals/m2/year) used as external forcing for starfish dynamics

    // PARAMETERS 
    PARAMETER(log_sd);            // Log observation standard deviation (log-scale)
    PARAMETER(slow_init);         // Initial slow-growing coral cover (%)
    PARAMETER(fast_init);         // Initial fast-growing coral cover (%)
    PARAMETER(cots_init);         // Initial crown-of-thorns starfish abundance (individuals/m^2)
    PARAMETER(growth_rate_slow);  // Growth rate multiplier for slow-growing corals, influenced by SST
    PARAMETER(impact_rate_slow);  // Impact sensitivity of slow-growing corals to starfish abundance
    PARAMETER(growth_rate_fast);  // Growth rate multiplier for fast-growing corals, influenced by SST
    PARAMETER(impact_rate_fast);  // Impact sensitivity of fast-growing corals to starfish abundance
    PARAMETER(growth_rate_cots);  // Growth rate multiplier for crown-of-thorns starfish, influenced by immigration forcing
    PARAMETER(impact_coral);      // Impact sensitivity of starfish abundance to coral cover
    PARAMETER(carr_cap);          // Carrying capacity for coral cover (%) used in logistic growth limiting
    PARAMETER(sst_effect);        // Effect of sea surface temperature on reducing effective coral carrying capacity (thermal stress)

    // Small constant for numerical stability
    Type epsilon = Type(1e-8);

    // Initialize negative log-likelihood
    Type nll = 0.0;

    // Numbered list of equations:
    // (1) slow_pred: Predicted slow-growing coral cover dynamics using SST and impact from starfish.
    // (2) fast_pred: Predicted fast-growing coral cover dynamics using SST and impact from starfish.
    // (3) cots_pred: Predicted crown-of-thorns starfish abundance dynamics using immigration forcing and coral cover feedback.
    int n = sst_dat.size();
    vector<Type> slow_pred(n);   // slow-growing coral cover prediction (%)
    vector<Type> fast_pred(n);   // fast-growing coral cover prediction (%)
    vector<Type> cots_pred(n);   // crown-of-thorns starfish abundance prediction (individuals/m^2)

    // Initialize predictions with model parameters
    slow_pred(0) = slow_init;
    fast_pred(0) = fast_init;
    cots_pred(0) = cots_init;

    for(int i = 1; i < n; i++){
        Type sst_val = sst_dat(i);
        Type carr_eff = carr_cap * exp(-sst_effect * sst_val);
        slow_pred(i) = slow_pred(i-1) * (1 + growth_rate_slow * sst_val) * (1 - slow_pred(i-1)/carr_eff) * exp(-impact_rate_slow * cots_pred(i-1));
        fast_pred(i) = fast_pred(i-1) * (1 + growth_rate_fast * sst_val) * (1 - fast_pred(i-1)/carr_eff) * exp(-impact_rate_fast * cots_pred(i-1));
        cots_pred(i) = cots_pred(i-1) * (1 + growth_rate_cots * cotsimm_dat(i)) * exp(-impact_coral * ((slow_pred(i-1) + fast_pred(i-1))/2));
    }
    
    // Likelihood calculation using lognormal distributions with fixed minimum standard deviations.
    Type sd = exp(log_sd);  // Ensure standard deviation is positive
    for(int i = 0; i < n; i++){
        // Using log-transformation to handle data spanning multiple orders of magnitude.
        Type log_slow_obs = log(slow_dat(i) + epsilon);
        Type log_slow_pred = log(slow_pred(i) + epsilon);
        nll -= dnorm(log_slow_obs, log_slow_pred, sd, true);  // slow coral cover likelihood

        Type log_fast_obs = log(fast_dat(i) + epsilon);
        Type log_fast_pred = log(fast_pred(i) + epsilon);
        nll -= dnorm(log_fast_obs, log_fast_pred, sd, true);  // fast coral cover likelihood

        Type log_cots_obs = log(cots_dat(i) + epsilon);
        Type log_cots_pred = log(cots_pred(i) + epsilon);
        nll -= dnorm(log_cots_obs, log_cots_pred, sd, true);  // crown-of-thorns starfish likelihood
    }


    // REPORT predictions for further analysis (suffixed with _pred)
    REPORT(slow_pred); // Reporting slow_pred: predicted slow-growing coral cover (%) 
    REPORT(fast_pred); // Reporting fast_pred: predicted fast-growing coral cover (%)
    REPORT(cots_pred); // Reporting cots_pred: predicted crown-of-thorns starfish abundance (individuals/m^2)

    return nll;
}
