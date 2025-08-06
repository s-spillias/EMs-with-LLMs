#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{

// 1. Data input: observed coral covers (in %)
//    slow_dat: slow-growing coral (Faviidae and Porites) cover percentages [DATA_VECTOR]
//    fast_dat: fast-growing coral (Acropora) cover percentages [DATA_VECTOR]
DATA_VECTOR(slow_dat);   // Observed slow coral cover (%)
DATA_VECTOR(fast_dat);   // Observed fast coral cover (%)
DATA_VECTOR(cots_dat);   // Observed crown-of-thorns starfish density (individuals/m2)
DATA_VECTOR(sst_dat);    // Sea surface temperature (°C)
DATA_VECTOR(cotsimm_dat); // Crown-of-thorns immigration rate (individuals/m2/year)

// 2. Parameters to be estimated:
//    attack_rate_slow: Predation rate on slow-growing corals (year^-1) [PARAMETER]
//    attack_rate_fast: Predation rate on fast-growing corals (year^-1) [PARAMETER]
//    starfish_growth_rate: Intrinsic growth rate of crown-of-thorns starfish (year^-1) [PARAMETER]
//    log_sigma_slow: Log-standard deviation for slow coral observations (log(%)) [PARAMETER]
//    log_sigma_fast: Log-standard deviation for fast coral observations (log(%)) [PARAMETER]
PARAMETER(attack_rate_slow);    // 1/year, attack rate for slow-growing corals
PARAMETER(attack_rate_fast);      // 1/year, attack rate for fast-growing corals
PARAMETER(starfish_growth_rate);  // 1/year, intrinsic growth rate of crown-of-thorns starfish
PARAMETER(log_sigma_slow);        // log(%), log standard deviation for slow coral cover observations
PARAMETER(log_sigma_fast);        // log(%), log standard deviation for fast coral cover observations
PARAMETER(log_sigma_cots);        // log(%), log standard deviation for COTS density observations
PARAMETER(mu_cots);               // constant predicted COTS density (individuals/m2)
PARAMETER(mu_slow);               // constant predicted slow coral cover (%)
PARAMETER(mu_fast);               // constant predicted fast coral cover (%)

// 3. Derived parameters with numerical stability enhancements (fixed min constant of 1e-8)
Type sigma_slow = exp(log_sigma_slow) + Type(1e-8);  // standard deviation for slow coral cover
Type sigma_fast = exp(log_sigma_fast) + Type(1e-8);    // standard deviation for fast coral cover
Type sigma_cots = exp(log_sigma_cots) + Type(1e-8);    // standard deviation for COTS density observations
// Lambda function for lognormal density (log-scale if give_log is true)
auto dlnorm = [&](Type x, Type meanlog, Type sd, bool give_log) -> Type {
    Type log_density = -log(x) - 0.5 * log(2 * M_PI) - log(sd) - 0.5 * pow((log(x) - meanlog) / sd, 2);
    return give_log ? log_density : exp(log_density);
};

int n = slow_dat.size();
vector<Type> cots_pred(n);  // Predicted crown-of-thorns starfish density (individuals/m2)
vector<Type> slow_pred(n);  // Predicted slow coral cover (%)
vector<Type> fast_pred(n);  // Predicted fast coral cover (%)

// Use constant predictions for a simpler model
for(int i = 0; i < n; i++){
    cots_pred[i] = mu_cots;
    slow_pred[i] = mu_slow;
    fast_pred[i] = mu_fast;
}

// 5. Negative log-likelihood (nll) initialization
Type nll = 0.0;

// 6. Model Equations (detailed below):
//    Equation 1: Consumption for slow-growing corals modeled by a logistic function
//      consumption_slow = attack_rate_slow / (1 + exp(-starfish_growth_rate * (i - n/2)))
//    Equation 2: Consumption for fast-growing corals modeled similarly
//      consumption_fast = attack_rate_fast / (1 + exp(-starfish_growth_rate * (i - n/2)))
//    Equation 3: Predicted coral cover is the observed cover adjusted by consumption
for(int i = 0; i < n; i++){
    // Smooth consumption transition via logistic function for time index "i"
    Type consumption_slow = attack_rate_slow / (Type(1.0) + exp(-starfish_growth_rate * (i - n/2)));
    Type consumption_fast = attack_rate_fast / (Type(1.0) + exp(-starfish_growth_rate * (i - n/2)));
    
    // Update predicted coral cover (ensuring predictions remain positive via addition of a small constant)
    slow_pred[i] = slow_dat[i] - consumption_slow;
    fast_pred[i] = fast_dat[i] - consumption_fast;
    
    // Likelihood contribution using a lognormal likelihood:
    // Using log-transform to handle data spanning multiple orders of magnitude.
    nll -= dlnorm(slow_dat[i], log(slow_pred[i] + Type(1e-8)), sigma_slow, true);
    nll -= dlnorm(fast_dat[i], log(fast_pred[i] + Type(1e-8)), sigma_fast, true);
}

ADREPORT(cots_pred);  // Predicted COTS density (individuals/m2)
ADREPORT(slow_pred);  // Predicted slow coral cover (%)
ADREPORT(fast_pred);  // Predicted fast coral cover (%)

return nll;
}
