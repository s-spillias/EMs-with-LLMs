#include <TMB.hpp>  // TMB header for template model builder

// This TMB model simulates the dynamics of Crown of Thorns starfish (COTS) outbreaks 
// and their interaction with coral communities (fast-growing Acropora and slow-growing 
// Faviidae/Porites). 
//
// Equations used in the model:
// 1. COTS dynamics: Logistic growth modulated by coral-mediated predation:
//    cots_pred[t] = cots_pred[t-1] + growth_rate * cots_pred[t-1] * (1 - cots_pred[t-1]/K)
//                  - predation_rate * cots_pred[t-1] * coral_effect
// 2. Coral predation effect: A saturating function of coral cover:
//    coral_effect = beta * (fast_pred[t-1] + slow_pred[t-1]) / (1 + beta * (fast_pred[t-1] + slow_pred[t-1]) + 1e-8)
// 3. Fast-growing coral dynamics: Recovery towards 100% cover minus loss due to COTS predation:
//    fast_pred[t] = fast_pred[t-1] + fast_coral_recov * (100 - fast_pred[t-1])
//                   - predation_rate * cots_pred[t-1] * (fast_pred[t-1]/(100 + 1e-8))
// 4. Slow-growing coral dynamics: Similar to fast coral dynamics:
//    slow_pred[t] = slow_pred[t-1] + slow_coral_recov * (100 - slow_pred[t-1])
//                   - predation_rate * cots_pred[t-1] * (slow_pred[t-1]/(100 + 1e-8))
//
// All predictions (_pred) for time step t use values from t-1 to prevent data leakage.
// Small constants (1e-8) are added to denominators and sigma values to ensure numerical stability.
template<class Type>
Type objective_function<Type>::operator() ()
{
    using namespace density;

    // DATA INPUTS:
    DATA_VECTOR(time);         // Time vector (year)
    DATA_VECTOR(cots_dat);       // Observed COTS abundance (individuals/m2)
    DATA_VECTOR(fast_dat);       // Observed fast-growing coral cover (%)
    DATA_VECTOR(slow_dat);       // Observed slow-growing coral cover (%)

    int n = time.size();       // Number of time steps

    // MODEL PARAMETERS (log-transformed where applicable)
    PARAMETER(log_growth_rate);       // Log intrinsic growth rate for COTS (year^-1)
    PARAMETER(log_K);                 // Log carrying capacity for COTS (individuals/m2)
    PARAMETER(log_predation_rate);    // Log predation rate coefficient (m2/(individual*year))
    PARAMETER(log_fast_coral_recov);  // Log recovery rate for fast-growing coral (%/year)
    PARAMETER(log_slow_coral_recov);  // Log recovery rate for slow-growing coral (%/year)
    PARAMETER(beta);                  // Efficiency parameter for coral effect on predation (unitless)

    // Log observation error standard deviations:
    PARAMETER(log_sigma_cots);        // Log standard deviation for COTS observations (log scale)
    PARAMETER(log_sigma_fast);        // Log standard deviation for fast coral observations (log scale)
    PARAMETER(log_sigma_slow);        // Log standard deviation for slow coral observations (log scale)

    // Exponentiate to transform parameters to their natural scale and add small constant for stability
    Type growth_rate = exp(log_growth_rate);      // COTS intrinsic growth rate (year^-1)
    Type K = exp(log_K);                          // COTS carrying capacity (individuals/m2)
    Type predation_rate = exp(log_predation_rate);  // Predation rate coefficient (m2/(individual*year))
    Type fast_coral_recov = exp(log_fast_coral_recov); // Recovery rate fast coral (%/year)
    Type slow_coral_recov = exp(log_slow_coral_recov); // Recovery rate slow coral (%/year)
    Type sigma_cots = exp(log_sigma_cots) + Type(1e-8); // Observation error for COTS
    Type sigma_fast = exp(log_sigma_fast) + Type(1e-8); // Observation error for fast coral
    Type sigma_slow = exp(log_sigma_slow) + Type(1e-8); // Observation error for slow coral

    // STATE VECTORS: Predictions for each time step
    vector<Type> cots_pred(n);     // Predicted COTS abundance
    vector<Type> fast_pred(n);     // Predicted fast-growing coral cover
    vector<Type> slow_pred(n);     // Predicted slow-growing coral cover

    // INITIAL CONDITIONS (using first observed values)
    cots_pred(0) = cots_dat(0);     // Initial COTS level (individuals/m2)
    fast_pred(0) = fast_dat(0);     // Initial fast coral cover (%)
    slow_pred(0) = slow_dat(0);     // Initial slow coral cover (%)

    Type nll = 0.0;  // Negative log-likelihood accumulator

    // Loop through time steps to update predictions based on past state (avoiding current response data)
    for(int t = 1; t < n; t++){
        // 1. Compute coral effect on predation with a saturating function 
        Type coral_effect = beta * (fast_pred(t-1) + slow_pred(t-1))
                             / (Type(1.0) + beta * (fast_pred(t-1) + slow_pred(t-1)) + Type(1e-8));  // Unitless

        // 2. Update COTS with logistic growth and subtract predation loss:
        // Equation 1: COTS dynamic update [year^-1]
        cots_pred(t) = cots_pred(t-1)
                      + growth_rate * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1)/K)
                      - predation_rate * cots_pred(t-1) * coral_effect;

        // 3. Update fast-growing coral: recovery to 100% cover minus predation losses:
        // Equation 3: Fast coral dynamics (% per year)
        fast_pred(t) = fast_pred(t-1)
                       + fast_coral_recov * (Type(100.0) - fast_pred(t-1))
                       - predation_rate * cots_pred(t-1) * (fast_pred(t-1)/(Type(100.0)+Type(1e-8)));

        // 4. Update slow-growing coral dynamics similarly:
        // Equation 4: Slow coral dynamics (% per year)
        slow_pred(t) = slow_pred(t-1)
                       + slow_coral_recov * (Type(100.0) - slow_pred(t-1))
                       - predation_rate * cots_pred(t-1) * (slow_pred(t-1)/(Type(100.0)+Type(1e-8)));

        // 5. Likelihood contributions (lognormal error distribution for strictly positive data)
        nll -= dnorm(log(cots_dat(t)+Type(1e-8)), log(cots_pred(t)+Type(1e-8)), sigma_cots, true) - log(cots_dat(t)+Type(1e-8));
        nll -= dnorm(log(fast_dat(t)+Type(1e-8)), log(fast_pred(t)+Type(1e-8)), sigma_fast, true) - log(fast_dat(t)+Type(1e-8));
        nll -= dnorm(log(slow_dat(t)+Type(1e-8)), log(slow_pred(t)+Type(1e-8)), sigma_slow, true) - log(slow_dat(t)+Type(1e-8));
    }

    // REPORT the predicted trajectories for further diagnostic and plotting purposes
    REPORT(cots_pred);    // Predicted COTS abundance over time
    REPORT(fast_pred);    // Predicted fast-growing coral cover over time
    REPORT(slow_pred);    // Predicted slow-growing coral cover over time

    return nll;    // Return the negative log-likelihood
}
