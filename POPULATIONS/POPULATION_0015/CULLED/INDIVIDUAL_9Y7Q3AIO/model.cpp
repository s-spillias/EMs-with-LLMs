#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator()() {
// 1. DATA SECTION:
//    Observe time series data. Time is provided as 'Year' (integer or real).
//    cots_dat: Observed adult COTS abundance (individuals/m2)
//    fast_dat: Observed fast-growing Acropora coral cover (%)
//    slow_dat: Observed slow-growing coral (Faviidae and Porites) cover (%)
DATA_VECTOR(Year);
DATA_VECTOR(cots_dat);
DATA_VECTOR(fast_dat);
DATA_VECTOR(slow_dat);

// 2. PARAMETER SECTION:
//    log_growth_rate: Log intrinsic growth rate (year^-1) for COTS [From literature]
//    log_decline_rate: Log decline rate (year^-1) during outbreak bust [From literature]
//    log_coral_effect: Log effect of coral predation on COTS abundance (unitless multiplier) [Expert opinion]
//    log_threshold: Log threshold for outbreak trigger (individuals/m2) [Initial estimate]
//    attack_efficiency: Efficiency of outbreak triggering (0-1) [Expert opinion]
PARAMETER(log_growth_rate);      // log intrinsic growth rate (year^-1)
PARAMETER(log_decline_rate);     // log decline rate (year^-1)
PARAMETER(log_coral_effect);     // log coral predation effect (unitless)
PARAMETER(log_threshold);        // log threshold for outbreak trigger (individuals/m2)
PARAMETER(attack_efficiency);    // outbreak trigger efficiency (0-1)
PARAMETER(gamma);                // coefficient for quadratic term in outbreak trigger logistic function


// 3. TRANSFORMATION SECTION:
//    Transform log-parameters to natural scale.
Type growth_rate = exp(log_growth_rate);    // Intrinsic growth rate (year^-1)
Type decline_rate = exp(log_decline_rate);    // Outbreak decline rate (year^-1)
Type coral_effect = exp(log_coral_effect);    // Coral predation effect multiplier (unitless)
Type threshold = exp(log_threshold);          // Outbreak trigger threshold (individuals/m2)

int n = Year.size(); // number of time steps
Type jnll = 0.0;     // joint negative log likelihood

// 4. PREDICTION ARRAYS:
//    Arrays to store model predictions for each time step.
vector<Type> cots_pred(n);  // predicted COTS abundance (individuals/m2)
vector<Type> fast_pred(n);  // predicted fast-growing coral cover (%)
vector<Type> slow_pred(n);  // predicted slow-growing coral cover (%)

// Set initial conditions using first data point (avoid using current observations in predictions)
cots_pred(0) = cots_dat(0);  // initial COTS abundance
fast_pred(0) = fast_dat(0);  // initial fast-growing coral cover
slow_pred(0) = slow_dat(0);  // initial slow-growing coral cover

// 5. MODEL DYNAMICS:
//    Loop over time steps t = 1,..., n-1 to predict population trajectories.
//    Equations used (each step explained below):
//
//    Equation 1:
//      cots_pred(t) = cots_pred(t-1) + growth_rate * saturating_function(cots_pred(t-1))
//                     - decline_rate * outbreak_trigger(cots_pred(t-1)) * cots_pred(t-1)
//                     - coral_effect * (fast_pred(t-1) + slow_pred(t-1)) * cots_pred(t-1)
//      - Saturating function: cots_pred(t-1)/(cots_pred(t-1) + threshold + 1e-8)
//      - Outbreak trigger: smooth logistic indicator based on difference (cots_pred(t-1) - threshold)
//    Equation 2 & 3 (Dummy coral dynamics):
//      fast_pred(t)   = fast_pred(t-1) - coral_effect * outbreak_trigger * fast_pred(t-1)
//                       + growth_rate * ( (100 - fast_pred(t-1)) * 1e-8 )
//      slow_pred(t)   = slow_pred(t-1) - coral_effect * outbreak_trigger * slow_pred(t-1)
//                       + growth_rate * ( (100 - slow_pred(t-1)) * 1e-8 )
//
//    Note: A small constant (1e-8) is added to prevent division by zero.
for(int t = 1; t < n; t++){
    // Calculate a saturating function for growth limitation.
    Type saturating = cots_pred(t-1) / (cots_pred(t-1) + threshold + Type(1e-8));
    
    // Calculate a smooth outbreak trigger using a logistic function.
    Type diff = cots_pred(t-1) - threshold;
    Type outbreak_trigger = 1 / (1 + exp(-attack_efficiency * diff - gamma * diff * diff));
    
    // Equation 1: COTS dynamics.
    cots_pred(t) = cots_pred(t-1)
                   + growth_rate * saturating * cots_pred(t-1)                     // growth component [1]
                   - decline_rate * outbreak_trigger * cots_pred(t-1)                // outbreak-related decline [2]
                   - coral_effect * (fast_pred(t-1) + slow_pred(t-1)) * cots_pred(t-1); // coral predation effect [3]
    cots_pred(t) = (cots_pred(t) < Type(1e-8)) ? Type(1e-8) : cots_pred(t);
    
    // Equation 2: Fast-growing coral dynamics (dummy formulation).
    fast_pred(t) = fast_pred(t-1)
                   - coral_effect * outbreak_trigger * fast_pred(t-1)                // predation loss [4]
                   + growth_rate * ((Type(100) - fast_pred(t-1)) * Type(1e-8));        // regrowth component [5]
    fast_pred(t) = (fast_pred(t) < Type(1e-8)) ? Type(1e-8) : fast_pred(t);
    
    // Equation 3: Slow-growing coral dynamics (dummy formulation).
    slow_pred(t) = slow_pred(t-1)
                   - coral_effect * outbreak_trigger * slow_pred(t-1)                // predation loss [6]
                   + growth_rate * ((Type(100) - slow_pred(t-1)) * Type(1e-8));        // regrowth component [7]
    slow_pred(t) = (slow_pred(t) < Type(1e-8)) ? Type(1e-8) : slow_pred(t);
    
    // 6. LIKELIHOOD CALCULATION:
    //    Use a lognormal likelihood for strictly positive data.
    //    Fixed sigma with a small constant ensures numerical stability.
    Type sigma = 0.1 + Type(1e-8); // minimum standard deviation
    
    // Add likelihood contributions by comparing the log-transformed observed and predicted values.
    jnll += dnorm(log(cots_dat(t)), log(cots_pred(t)), sigma, true);
    jnll += dnorm(log(fast_dat(t)), log(fast_pred(t)), sigma, true);
    jnll += dnorm(log(slow_dat(t)), log(slow_pred(t)), sigma, true);
}

// 7. REPORT PREDICTIONS:
//    Return all predicted time series to facilitate diagnostics.
REPORT(cots_pred);
REPORT(fast_pred);
REPORT(slow_pred);

return jnll;
}
