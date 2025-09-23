#include <TMB.hpp>

// 1. Data Section:
//    - Year: Time (year)
//    - cots_dat: Observed adult COTS abundance (individuals/m2)
//    - fast_dat: Observed fast-growing Acropora cover (%) 
//    - slow_dat: Observed slow-growing coral cover (%) 
DATA_VECTOR(Year);
DATA_VECTOR(cots_dat);
DATA_VECTOR(fast_dat);
DATA_VECTOR(slow_dat);

// 2. Parameter Section:
//    - log_growth_cots: log intrinsic growth rate of COTS (year^-1)
//    - log_decline_cots: log decline rate of COTS (year^-1)
//    - log_predation_fast: log predation efficiency on fast-growing coral (per unit cover)
//    - log_predation_slow: log predation efficiency on slow-growing coral (per unit cover)
//    - log_thresh: log outbreak threshold that triggers rapid population growth (individuals/m2)
//    - coral_efficiency: efficiency of coral growth (dimensionless)
PARAMETER(log_growth_cots);
PARAMETER(log_decline_cots);
PARAMETER(log_predation_fast);
PARAMETER(log_predation_slow);
PARAMETER(log_thresh);
PARAMETER(coral_efficiency);

// 3. Transform Parameters
//    (Exponentiating log-parameters to ensure positivity)
Type growth_cots = exp(log_growth_cots);          // (year^-1)
Type decline_cots = exp(log_decline_cots);          // (year^-1)
Type predation_fast = exp(log_predation_fast);      // per unit coral cover
Type predation_slow = exp(log_predation_slow);      // per unit coral cover
Type outbreak_threshold = exp(log_thresh);          // individuals/m2

// 4. Initialize Predicted Vectors:
//    - _pred variables hold model predictions, using lag terms only.
vector<Type> cots_pred(cots_dat.size());
vector<Type> fast_pred(fast_dat.size());
vector<Type> slow_pred(slow_dat.size());

// Set initial conditions from data (t = 0)
cots_pred[0] = cots_dat[0];
fast_pred[0] = fast_dat[0];
slow_pred[0] = slow_dat[0];

// 5. Objective Function Calculation:
//    The negative log-likelihood (nll) accumulates likelihood contributions across time steps.
//    Equations (for each t >= 1):
//    (1) COTS Dynamics: 
//        cots_pred[t] = cots_pred[t-1] * exp( growth_cots*modifier - decline_cots*(1 - modifier) )
//        where modifier is a smooth function (1 / (1 + exp(-(lag - threshold))))
//    (2) Coral Dynamics (Fast and Slow):
//        fast_pred[t] = fast_pred[t-1] + coral_efficiency * fast_pred[t-1]*(1 - fast_pred[t-1]/100)
//                         - predation_fast * cots_pred[t-1]*fast_pred[t-1]
//        slow_pred[t] = slow_pred[t-1] + coral_efficiency * slow_pred[t-1]*(1 - slow_pred[t-1]/100)
//                         - predation_slow * cots_pred[t-1]*slow_pred[t-1]
//    Likelihood: using a lognormal error model with a fixed minimum standard deviation (0.1) to prevent numerical issues.
Type nll = 0.0;
for(int t = 1; t < cots_dat.size(); t++){
    // (1) COTS Population Dynamics
    Type modifier = Type(1.0) / (Type(1.0) + exp( -(cots_pred[t-1] - outbreak_threshold) ));
    cots_pred[t] = cots_pred[t-1] * exp(growth_cots * modifier - decline_cots * (Type(1.0) - modifier) + Type(1e-8)); // Added small constant for numerical stability
    
    // (2) Coral Dynamics with saturating growth and predation:
    fast_pred[t] = fast_pred[t-1] + coral_efficiency * fast_pred[t-1] * (Type(1.0) - fast_pred[t-1]/Type(100.0))
                   - predation_fast * cots_pred[t-1] * fast_pred[t-1] + Type(1e-8);
    slow_pred[t] = slow_pred[t-1] + coral_efficiency * slow_pred[t-1] * (Type(1.0) - slow_pred[t-1]/Type(100.0))
                   - predation_slow * cots_pred[t-1] * slow_pred[t-1] + Type(1e-8);
    
    // (3) Likelihood Calculation using lognormal error distributions
    //     (Using previous predictions and a fixed error std of 0.1)
    nll -= dlnorm(cots_dat[t], log(cots_pred[t] + Type(1e-8)), Type(0.1), true);
    nll -= dlnorm(fast_dat[t], log(fast_pred[t] + Type(1e-8)), Type(0.1), true);
    nll -= dlnorm(slow_dat[t], log(slow_pred[t] + Type(1e-8)), Type(0.1), true);
}

// 6. Reporting Predicted Values (_pred variables) for inspection
REPORT(cots_pred);
REPORT(fast_pred);
REPORT(slow_pred);

return nll;
