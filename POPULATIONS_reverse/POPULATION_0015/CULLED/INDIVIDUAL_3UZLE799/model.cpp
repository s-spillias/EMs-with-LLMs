#include <TMB.hpp>

// TMB model for COTS starfish and coral dynamics
// 1) Equation for starfish population growth with resource limitation.
// 2) Equations for coral community dynamics under selective predation pressure.
// 3) Smooth transitions are implemented to ensure numerical stability.
template<class Type>
Type objective_function<Type>::operator() ()
{
    //---- DATA SECTION ----
    // Data: Year is provided in the input CSV's first column.
    DATA_VECTOR(Year);                      // Year (time, unit: year)
    DATA_VECTOR(cots_dat);                  // Observed COTS abundance (individuals/m2)
    DATA_VECTOR(slow_dat);                  // Observed slow-growing coral cover (%)
    DATA_VECTOR(fast_dat);                  // Observed fast-growing coral cover (%)

    //---- PARAMETER SECTION ----
    // Log-transformed parameters to ensure positivity
    PARAMETER(log_growth_rate);             // Log intrinsic growth rate of COTS (year^-1). Determined from literature.
    PARAMETER(log_rate_slow);               // Log predation rate on slow coral (rate, unit: year^-1).
    PARAMETER(log_rate_fast);               // Log predation rate on fast coral (rate, unit: year^-1).
    PARAMETER(log_eff_coral);               // Log efficiency parameter modifying predation effect (unitless).
    PARAMETER(sd_obs);                      // Observation error standard deviation.

    //---- TRANSFORMATIONS ----
    Type growth_rate = exp(log_growth_rate);  // Intrinsic growth rate (year^-1)
    Type rate_slow = exp(log_rate_slow);        // Predation rate for slow coral (year^-1)
    Type rate_fast = exp(log_rate_fast);        // Predation rate for fast coral (year^-1)
    Type eff_coral = exp(log_eff_coral);        // Efficiency of coral modifying predation (unitless)
    Type min_sd = Type(1e-8);                   // Small constant to prevent division by zero

    //---- INITIALISATION & PREDICTIONS ----
    int n = cots_dat.size();                  // Number of time steps
    vector<Type> cots_pred(n);                // Predicted COTS values
    vector<Type> slow_pred(n);                // Predicted slow coral values
    vector<Type> fast_pred(n);                // Predicted fast coral values

    // Initialize predictions with the first observation to start the recurrence.
    cots_pred[0] = cots_dat[0];
    slow_pred[0] = slow_dat[0];
    fast_pred[0] = fast_dat[0];

    //---- LIKELIHOOD CALCULATION ----
    Type nll = Type(0);                       // Initialize negative log likelihood

    /* 
      Equations:
      1) COTS dynamics: Logistic growth modulated by effective coral cover.
         - Functional response: 1 / (min_sd + eff_coral*(slow + fast))
      2) Slow coral dynamics: Decline due to predation on slow coral.
         - Saturating removal: (COTS * coral) / (1 + coral)
      3) Fast coral dynamics: Decline due to predation on fast coral.
         - Similar functional response as slow coral.
    */
    for (int t = 1; t < n; t++) {
        // Equation 1: Starfish population dynamics (using previous time step values)
        Type predation_effect = Type(1.0) / (min_sd + eff_coral * (slow_pred[t-1] + fast_pred[t-1])); // Saturating response function
        cots_pred[t] = cots_pred[t-1] + growth_rate * cots_pred[t-1] * (Type(1.0) - cots_pred[t-1] * predation_effect); // Logistic growth with feedback

        // Equation 2: Slow-growing coral predation dynamics
        slow_pred[t] = slow_pred[t-1] - rate_slow * (cots_pred[t-1] * slow_pred[t-1] / (Type(1.0) + slow_pred[t-1] + min_sd)); // Smooth saturating decline

        // Equation 3: Fast-growing coral predation dynamics
        fast_pred[t] = fast_pred[t-1] - rate_fast * (cots_pred[t-1] * fast_pred[t-1] / (Type(1.0) + fast_pred[t-1] + min_sd)); // Smooth saturating decline
        if(cots_pred[t] < min_sd) cots_pred[t] = min_sd; // enforce minimum COTS pred value for stability
        if(slow_pred[t] < min_sd) slow_pred[t] = min_sd; // enforce minimum slow coral value for stability
        if(fast_pred[t] < min_sd) fast_pred[t] = min_sd; // enforce minimum fast coral value for stability

        // Likelihood: Lognormal error distribution comparing predictions to observed data
        nll -= dnorm(log(cots_dat[t] + min_sd), log(cots_pred[t] + min_sd), sd_obs, true);
        nll -= dnorm(log(slow_dat[t] + min_sd), log(slow_pred[t] + min_sd), sd_obs, true);
        nll -= dnorm(log(fast_dat[t] + min_sd), log(fast_pred[t] + min_sd), sd_obs, true);
    }

    //---- REPORTING ----
    REPORT(cots_pred);     // Report predicted COTS values (_pred)
    REPORT(slow_pred);     // Report predicted slow coral values (_pred)
    REPORT(fast_pred);     // Report predicted fast coral values (_pred)

    return nll;
}
