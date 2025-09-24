#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() () {

// 1. Data objects
//    - years: Time index (year)
//    - cots_dat: Observed COTS densities (individuals per m^2)
//    - fast_dat: Observed fast-growing coral cover (%)
//    - slow_dat: Observed slow-growing coral cover (%)
//    - sst_dat: Sea Surface Temperature (°C)
//    - cotsimm_dat: Larval immigration rate (individuals per m^2/year)
DATA_VECTOR(years);
DATA_VECTOR(cots_dat);
DATA_VECTOR(fast_dat);
DATA_VECTOR(slow_dat);
DATA_VECTOR(sst_dat);
DATA_VECTOR(cotsimm_dat);

// 2. Parameter objects (log-scale where noted)
//    - log_r: Log of intrinsic growth rate of COTS outbreak (year^-1)
//    - log_m: Log of mortality rate for COTS (year^-1)
//    - log_alpha: Log of predation efficiency on fast coral (unitless)
//    - log_beta: Log of predation efficiency on slow coral (unitless)
//    - threshold: COTS abundance threshold triggering outbreak (individuals per m^2)
//    - smoothing: Smoothing parameter for outbreak function (population units)
//    - log_coral_recovery: Log of coral recovery rate (year^-1)
PARAMETER(log_r);
PARAMETER(log_m);
PARAMETER(log_alpha);
PARAMETER(log_beta);
PARAMETER(threshold);
PARAMETER(smoothing);
PARAMETER(log_coral_recovery);

// 3. Parameter transformations to natural scale
Type r = exp(log_r);               // (1) Intrinsic growth rate of COTS outbreak
Type m = exp(log_m);               // (2) Mortality rate of COTS
Type alpha = exp(log_alpha);       // (3) Predation efficiency on fast-growing coral
Type beta = exp(log_beta);         // (4) Predation efficiency on slow-growing coral
Type coral_recovery = exp(log_coral_recovery); // (5) Coral recovery rate

// Number of time steps
int n = cots_dat.size();

// Vectors for predictions (_pred variables)
vector<Type> cots_pred(n);      // Predicted COTS density
vector<Type> fast_pred(n);   // Predicted fast coral cover (%)
vector<Type> slow_pred(n);   // Predicted slow coral cover (%)

// Negative log-likelihood to be minimized
Type nll = 0.0;

// Set initial conditions using data from time step 0
cots_pred(0) = cots_dat(0);
fast_pred(0) = fast_dat(0);
slow_pred(0) = slow_dat(0);

// Loop through time steps (using only past values for predictions to avoid data leakage)
for (int t = 1; t < n; t++) {
    // 1. Outbreak trigger function: a smooth saturating function transitioning at the threshold.
    //    Equation: outbreak_factor = 1/(1 + exp(-(cots_pred(t-1) - threshold)/smoothing))
    Type outbreak_factor = 1.0 / (1.0 + exp(-(cots_pred(t-1) - threshold) / (smoothing + Type(1e-8))));
    
    // 2. COTS dynamics:
    //    Equation: n_pred(t) = n_pred(t-1) + (r*outbreak_factor - m) * n_pred(t-1) + cotsimm_dat(t-1)
    cots_pred(t) = cots_pred(t-1) + r * outbreak_factor * cots_pred(t-1) - m * cots_pred(t-1) + cotsimm_dat(t-1);
    
    // 3. Fast coral dynamics:
    //    Equation: fast_pred(t) = fast_pred(t-1) - alpha*outbreak_factor*cots_pred(t-1)*fast_pred(t-1)
    //              + coral_recovery*(100 - fast_pred(t-1))
    fast_pred(t) = fast_pred(t-1) - alpha * outbreak_factor * cots_pred(t-1) * fast_pred(t-1)
                   + coral_recovery * (Type(100.0) - fast_pred(t-1));
    
    // 4. Slow coral dynamics:
    //    Equation: slow_pred(t) = slow_pred(t-1) - beta*outbreak_factor*cots_pred(t-1)*slow_pred(t-1)
    //              + coral_recovery*(100 - slow_pred(t-1))
    slow_pred(t) = slow_pred(t-1) - beta * outbreak_factor * cots_pred(t-1) * slow_pred(t-1)
                   + coral_recovery * (Type(100.0) - slow_pred(t-1));
    
    // 5. Likelihood: Using lognormal likelihood for observational data (ensuring positivity via small constant)
    //    A fixed standard deviation of 0.1 (minimum observation error) is assumed.
    nll += density::ldnorm(cots_dat(t), log(cots_pred(t) + Type(1e-8)), Type(0.1));
    nll += density::ldnorm(fast_dat(t), log(fast_pred(t) + Type(1e-8)), Type(0.1));
    nll += density::ldnorm(slow_dat(t), log(slow_pred(t) + Type(1e-8)), Type(0.1));
}

// Reporting predicted values for diagnostic purposes (the _pred variables)
REPORT(cots_pred);     // Report of COTS predictions
REPORT(fast_pred);  // Report of fast coral predictions
REPORT(slow_pred);  // Report of slow coral predictions

return nll;
}
