#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() () {

// 1. DATA DECLARATIONS:
//    - cots_dat: Observed COTS abundance (individuals/m2)
//    - fast_dat: Observed fast-growing coral cover (%)
//    - slow_dat: Observed slow-growing coral cover (%)
//    - sst_dat: Sea-Surface Temperature (°C)
//    - cotsimm_dat: Larval immigration rate of COTS (individuals/m2/year)
DATA_VECTOR(cots_dat);
DATA_VECTOR(fast_dat);
DATA_VECTOR(slow_dat);
DATA_VECTOR(sst_dat);
DATA_VECTOR(cotsimm_dat);

// 2. PARAMETER DECLARATIONS:
//    - r_cots: Intrinsic COTS outbreak growth rate (year^-1)
//    - K_cots: Baseline carrying capacity modulated by coral cover (individuals/m2)
//    - alpha_fast: Modulation effect of fast-growing coral on COTS dynamics (unitless)
//    - alpha_slow: Modulation effect of slow-growing coral on COTS dynamics (unitless)
//    - r_fast: Recovery rate of fast-growing coral (year^-1)
//    - r_slow: Recovery rate of slow-growing coral (year^-1)
//    - sigma_cots: Observation error for COTS data (log scale)
//    - sigma_fast: Observation error for fast coral cover data
//    - sigma_slow: Observation error for slow coral cover data
PARAMETER(r_cots);
PARAMETER(K_cots);
PARAMETER(alpha_fast);
PARAMETER(alpha_slow);
PARAMETER(r_fast);
PARAMETER(r_slow);
PARAMETER(sigma_cots);
PARAMETER(sigma_fast);
PARAMETER(sigma_slow);
PARAMETER(beta_fast);
PARAMETER(beta_slow);

// 3. NUMBER OF TIME STEPS (assumed to be the length of observation vectors)
int n = cots_dat.size();

// 4. DECLARE PREDICTION ARRAYS
vector<Type> cots_pred(n);   // Predicted COTS abundance
vector<Type> fast_pred(n);   // Predicted fast-growing coral cover
vector<Type> slow_pred(n);   // Predicted slow-growing coral cover

// 5. LIKELIHOOD INITIALIZATION
Type nll = 0.0;

// 6. INITIALIZE FIRST TIME STEP WITH OBSERVED VALUES (to avoid data leakage, these are fixed)
cots_pred(0) = cots_dat(0) + Type(1e-8);
fast_pred(0) = fast_dat(0) + Type(1e-8);
slow_pred(0) = slow_dat(0) + Type(1e-8);

// 7. DYNAMIC MODEL EQUATIONS (for t >= 1)
// Equations:
// (1) COTS dynamics:
//     cots_pred[t] = cots_pred[t-1] + r_cots * cots_pred[t-1] *
//                    (1 - cots_pred[t-1]/(K_cots + alpha_fast * fast_pred[t-1] + alpha_slow * slow_pred[t-1])) 
//                    + f(sst_dat[t-1], cotsimm_dat[t-1])
// (2) Fast coral dynamics:
//     fast_pred[t] = fast_pred[t-1] + r_fast*(100 - fast_pred[t-1]) - predation effect from COTS
// (3) Slow coral dynamics:
//     slow_pred[t] = slow_pred[t-1] + r_slow*(100 - slow_pred[t-1]) - predation effect from COTS
for(int t = 1; t < n; t++){
    // 7.1 Calculate environmental forcing for COTS outbreak (smooth function of sst and larval input)
    Type env_force = 1 / (1 + exp(- (sst_dat(t-1) - 27.0))) + cotsimm_dat(t-1);
    
    // 7.2 COTS equation with saturating growth, resource limitation, and environmental forcing:
    {
      Type denom = K_cots + (alpha_fast * fast_pred(t-1))/(1 + beta_fast * fast_pred(t-1))
                         + (alpha_slow * slow_pred(t-1))/(1 + beta_slow * slow_pred(t-1)) + Type(1e-8);
      denom = CppAD::CondExpLt(denom, Type(1e-8), Type(1e-8), denom);
      cots_pred(t) = cots_pred(t-1)
                     + r_cots * cots_pred(t-1) * (1 - cots_pred(t-1)/denom)
                     + env_force;
    }
    
    // 7.3 Fast coral recovery with logistic growth minus predation impact:
    fast_pred(t) = fast_pred(t-1)
                   + r_fast * (100 - fast_pred(t-1))     // Recovery towards maximum cover of 100%
                   - 0.01 * cots_pred(t-1);               // Predation effect scaled by COTS abundance
    
    // 7.4 Slow coral recovery with logistic growth minus predation impact:
    slow_pred(t) = slow_pred(t-1)
                   + r_slow * (100 - slow_pred(t-1))
                   - 0.005 * cots_pred(t-1);              // Lower predation rate on slow-growing corals
    
    // Ensure predicted values remain positive using TMB conditional expressions
    cots_pred(t) = CppAD::CondExpLt(cots_pred(t), Type(1e-8), Type(1e-8), cots_pred(t));
    fast_pred(t) = CppAD::CondExpLt(fast_pred(t), Type(1e-8), Type(1e-8), fast_pred(t));
    slow_pred(t) = CppAD::CondExpLt(slow_pred(t), Type(1e-8), Type(1e-8), slow_pred(t));
    // 7.5 Accumulate NLL with lognormal error for each observed data type:
    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t) + Type(1e-8)), sigma_cots, true) - log(cots_dat(t) + Type(1e-8));
    nll -= dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred(t) + Type(1e-8)), sigma_fast, true) - log(fast_dat(t) + Type(1e-8));
    nll -= dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred(t) + Type(1e-8)), sigma_slow, true) - log(slow_dat(t) + Type(1e-8));
}

// 8. REPORT predicted values for inspection (with _pred suffix)
REPORT(cots_pred);
REPORT(fast_pred);
REPORT(slow_pred);

return nll;
}
