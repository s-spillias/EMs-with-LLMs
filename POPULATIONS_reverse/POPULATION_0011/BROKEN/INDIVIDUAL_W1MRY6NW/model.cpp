#include <TMB.hpp>

// 1. Equation List:
//    1. Starfish (COTS) dynamics: Logistic growth with environmental forcing
//       Equation: cots_pred[t] = cots_pred[t-1] + growth_star * cots_pred[t-1] * (1 - cots_pred[t-1]/K_star)
//                   + growth_factor_env * sst_dat[t-1] * cotsimm_dat[t]
//    2. Slow-growing coral dynamics (Faviidae/Porites): Logistic growth reduced by starfish predation 
//       using a saturating functional response.
//       Equation: slow_pred[t] = slow_pred[t-1] + growth_slow * slow_pred[t-1] * (1 - slow_pred[t-1]/K_slow)
//                   - efficiency_slow * cots_pred[t-1] * slow_pred[t-1] / (slow_pred[t-1] + half_saturation_slow + Type(1e-8))
//    3. Fast-growing coral dynamics (Acropora): Logistic growth reduced by starfish predation 
//       using a saturating functional response.
//       Equation: fast_pred[t] = fast_pred[t-1] + growth_fast * fast_pred[t-1] * (1 - fast_pred[t-1]/K_fast)
//                   - efficiency_fast * cots_pred[t-1] * fast_pred[t-1] / (fast_pred[t-1] + half_saturation_fast + Type(1e-8))
//
// All parameters have attached inline comments explaining their units and estimated values.
// A small constant (Type(1e-8)) is used in divisions to ensure numerical stability.
template<class Type>
Type objective_function<Type>::operator() ()
{
    using namespace density;
    
    // Data inputs (observed time series; all units are as in the provided datasets)
    DATA_VECTOR(Year);         // Year of observation
    DATA_VECTOR(cots_dat);     // Observed COTS abundance (individuals/m2)
    DATA_VECTOR(slow_dat);     // Observed slow-growing coral cover (%)
    DATA_VECTOR(fast_dat);     // Observed fast-growing coral cover (%)
    DATA_VECTOR(sst_dat);      // Sea-surface temperature (°C)
    DATA_VECTOR(cotsimm_dat);  // COTS immigration rate (individuals/m2/year)
    
    int n = cots_dat.size();   // number of time steps
    
    // === COTS Parameters ===
    PARAMETER(log_growth_star); // Log intrinsic growth rate for COTS (year^-1)
    PARAMETER(log_K_star);      // Log carrying capacity for COTS (individuals/m2)
    PARAMETER(growth_factor_env); // Environmental forcing factor (unitless) modifying COTS growth due to sst and immigration
    
    // === Slow-growing Coral Parameters ===
    PARAMETER(growth_slow);        // Intrinsic growth rate for slow-growing coral (year^-1)
    PARAMETER(K_slow);             // Carrying capacity for slow-growing coral (% cover)
    PARAMETER(efficiency_slow);    // Predation efficiency on slow-growing coral by COTS (unitless)
    PARAMETER(half_saturation_slow); // Half-saturation constant for slow-growing coral predation (% cover)
    PARAMETER(competition_slow_fast); // Competition coefficient for fast-growing coral's effect on slow-growing coral (unitless)
    
    // === Fast-growing Coral Parameters ===
    PARAMETER(growth_fast);        // Intrinsic growth rate for fast-growing coral (year^-1)
    PARAMETER(K_fast);             // Carrying capacity for fast-growing coral (% cover)
    PARAMETER(efficiency_fast);    // Predation efficiency on fast-growing coral by COTS (unitless)
    PARAMETER(half_saturation_fast); // Half-saturation constant for fast-growing coral predation (% cover)
    
    // === Observation Error Parameters (log scale for positivity) ===
    PARAMETER(log_sd_cots);  // Log standard deviation for COTS observations
    PARAMETER(log_sd_slow);  // Log standard deviation for slow-growing coral observations
    PARAMETER(log_sd_fast);  // Log standard deviation for fast-growing coral observations

    // Transform log-parameters to natural scale, adding a small constant to standard deviations for numerical stability.
    Type growth_star = exp(log_growth_star);  // Intrinsic growth rate (year^-1)
    if(!(growth_star==growth_star)) growth_star = 0.1;
    Type K_star = exp(log_K_star);              // Carrying capacity (individuals/m2)
    Type sd_cots = exp(log_sd_cots) + Type(1e-8);
    Type sd_slow = exp(log_sd_slow) + Type(1e-8);
    Type sd_fast = exp(log_sd_fast) + Type(1e-8);
    if(!(growth_slow==growth_slow)) growth_slow = 0.05;
    if(!(growth_fast==growth_fast)) growth_fast = 0.1;
    
    // Initialize predictions for each state variable.
    vector<Type> cots_pred(n);  // Predicted COTS abundance (individuals/m2)
    vector<Type> slow_pred(n);  // Predicted slow-growing coral cover (%)
    vector<Type> fast_pred(n);  // Predicted fast-growing coral cover (%)
    
    // Set initial values equal to the first observed values (to avoid using current time step data in prediction)
    cots_pred(0) = cots_dat(0);
    slow_pred(0) = slow_dat(0);
    fast_pred(0) = fast_dat(0);
    
    // Time loop: predictions for t>=1 depend solely on state at t-1
    for (int t = 1; t < n; t++) {
        // --- Equation 1: COTS Dynamics ---
        // Logistic growth with environmental forcing (influence of sst and immigration).
        cots_pred(t) = cots_pred(t-1) 
                        + growth_star * cots_pred(t-1) * (1 - cots_pred(t-1) / K_star)
                        + growth_factor_env * sst_dat(t-1) * cotsimm_dat(t-1);
        if(cots_pred(t) < Type(1e-8)) cots_pred(t) = Type(1e-8);
        
        // --- Equation 2: Slow-growing Coral Dynamics ---
        // Logistic growth; reduced by starfish predation with a saturating functional response and interspecific competition with fast-growing coral.
        slow_pred(t) = slow_pred(t-1) 
                        + growth_slow * slow_pred(t-1) * (1 - (slow_pred(t-1) + competition_slow_fast * fast_pred(t-1)) / K_slow)
                        - efficiency_slow * cots_pred(t-1) * slow_pred(t-1) / (slow_pred(t-1) + half_saturation_slow + Type(1e-8));
        if(slow_pred(t) < Type(1e-8)) slow_pred(t) = Type(1e-8);
                        
        // --- Equation 3: Fast-growing Coral Dynamics ---
        // Logistic growth; reduced by starfish predation with a saturating functional response.
        fast_pred(t) = fast_pred(t-1) 
                        + growth_fast * fast_pred(t-1) * (1 - fast_pred(t-1) / K_fast)
                        - efficiency_fast * cots_pred(t-1) * fast_pred(t-1) / (fast_pred(t-1) + half_saturation_fast + Type(1e-8));
        if(fast_pred(t) < Type(1e-8)) fast_pred(t) = Type(1e-8);
    }
    
    // Calculate negative log likelihood (nll) using lognormal likelihoods.
    Type nll = 0;
    for (int t = 1; t < n; t++) {
        // Add likelihood contributions for each observation (log-transform avoids issues with multi-order magnitude data).
        nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t) + Type(1e-8)), sd_cots, true);
        nll -= dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred(t) + Type(1e-8)), sd_slow, true);
        nll -= dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred(t) + Type(1e-8)), sd_fast, true);
    }
    
    // Report all prediction variables (with '_pred' suffix) in the output.
    REPORT(cots_pred);
    REPORT(slow_pred);
    REPORT(fast_pred);
    
    return nll;
}
