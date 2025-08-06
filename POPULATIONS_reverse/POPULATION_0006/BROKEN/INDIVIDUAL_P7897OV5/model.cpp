#include <TMB.hpp>

// 1. Data Inputs:
//    - slow_dat: Observed slow-growing coral cover (%) [data]
//    - fast_dat: Observed fast-growing coral cover (%) [data]
//    - cots_dat: Observed starfish abundance (individuals/m^2) [data]
//    - sst_dat: Sea surface temperature (°C) [data]
//    - cotsimm_dat: Crown-of-thorns starfish immigration rate (individuals/m^2/year) [data]
DATA_VECTOR(slow_dat);
DATA_VECTOR(fast_dat);
DATA_VECTOR(cots_dat);
DATA_VECTOR(sst_dat);
DATA_VECTOR(cotsimm_dat);

// 2. Parameters with biological meaning and smooth penalties:
//    - growth_rate: Intrinsic growth rate of starfish (year^-1)
//    - predation_rate_slow: Predation rate impacting slow-growing corals (unitless)
//    - predation_rate_fast: Predation rate impacting fast-growing corals (unitless)
//    - smooth_penalty: Smoothing penalty coefficient to constrain parameter variation (unitless)
//    - log_sd: Log of observation standard deviation (ensures positive error SD)
PARAMETER(growth_rate);
PARAMETER(predation_rate_slow);
PARAMETER(predation_rate_fast);
PARAMETER(slow_coral_growth_rate);  // Multiplicative growth factor for slow-growing corals (unitless)
PARAMETER(fast_coral_growth_rate);  // Multiplicative growth factor for fast-growing corals (unitless)
PARAMETER(smooth_penalty);
PARAMETER(log_sd);
PARAMETER(slow_init);  // Initial slow-growing coral cover (%) [parameter]
PARAMETER(fast_init);  // Initial fast-growing coral cover (%) [parameter]
PARAMETER(cots_init);  // Initial starfish abundance (individuals/m^2) [parameter]

// Convert log_sd to actual SD with a fixed minimum for stability.
Type sd = exp(log_sd);
if(sd < Type(1e-8)) sd = Type(1e-8);

// Model predictions
// Here we generate predictions for coral cover and starfish abundance using smooth transitions.
// The predictions (_pred) are computed as a transformation of the observed data with exponential damping.
vector<Type> slow_pred(slow_dat.size());
vector<Type> fast_pred(fast_dat.size());
vector<Type> cots_pred(cots_dat.size());

// Initialize predictions using new initial parameters (do not use data directly)
slow_pred(0) = slow_init;  // Initial slow coral cover (%) 
fast_pred(0) = fast_init;  // Initial fast coral cover (%) 
cots_pred(0) = cots_init;  // Initial starfish abundance (individuals/m^2)

// Recursive prediction equations for t >= 1 using only forcing variables and previous predictions.
// Equation 1: Slow coral dynamics: growth modulated by sea surface temperature and reduced by starfish predation.
for (int i = 1; i < slow_dat.size(); i++){
  slow_pred(i) = slow_pred(i-1) * slow_coral_growth_rate * (Type(1) - predation_rate_slow * cots_pred(i-1)) + Type(1e-8);
}

// Equation 2: Fast coral dynamics: updated similarly with sst_dat and starfish impacts.
for (int i = 1; i < fast_dat.size(); i++){
  fast_pred(i) = fast_pred(i-1) * fast_coral_growth_rate * (Type(1) - predation_rate_fast * cots_pred(i-1)) + Type(1e-8);
}

// Equation 3: Starfish dynamics: evolves with intrinsic growth and external immigration.
for (int i = 1; i < cots_dat.size(); i++){
  cots_pred(i) = cots_pred(i-1) * exp(growth_rate) + cotsimm_dat(i-1) + Type(1e-8);
}

// Likelihood calculation using lognormal error distributions for strictly positive data.
// A fixed minimum standard deviation (sd) is used to prevent numerical issues.
//  - Equation 4: Log-likelihood for slow coral observations based on slow_pred.
//  - Equation 5: Log-likelihood for fast coral observations based on fast_pred.
//  - Equation 6: Log-likelihood for starfish abundance observations based on cots_pred.
Type nll = 0.0;
for(int i = 0; i < slow_dat.size(); i++){
  nll -= dlnorm(slow_dat(i) + Type(1e-8), log(slow_pred(i) + Type(1e-8)), sd, true);
  nll -= dlnorm(fast_dat(i) + Type(1e-8), log(fast_pred(i) + Type(1e-8)), sd, true);
}
for (int i = 0; i < cots_dat.size(); i++){
  nll -= dlnorm(cots_dat(i) + Type(1e-8), log(cots_pred(i) + Type(1e-8)), sd, true);
}

/*
Equation Descriptions:
1. Slow Coral Dynamics:
   slow_pred(i) = slow_pred(i-1) * slow_coral_growth_rate * (1 - predation_rate_slow * cots_pred(i-1)) + 1e-8;
   (Represents slow-growing coral cover dynamics driven by intrinsic coral growth and reduced by starfish predation)
2. Fast Coral Dynamics:
   fast_pred(i) = fast_pred(i-1) * fast_coral_growth_rate * (1 - predation_rate_fast * cots_pred(i-1)) + 1e-8;
   (Represents fast-growing coral cover dynamics driven by intrinsic coral growth and reduced by starfish predation)
3. Starfish Dynamics:
   cots_pred(i) = cots_pred(i-1) * exp(growth_rate) + cotsimm_dat(i-1) + 1e-8;
   (Represents starfish abundance dynamics driven by intrinsic growth and external immigration)
4. Likelihood Equations:
   Observations are modeled using lognormal distributions comparing observed data with predictions.
5. Note:
   All equations rely solely on external forcing variables, previous predictions, and parameters, thus avoiding data leakage.
*/

ADREPORT(slow_pred);
ADREPORT(fast_pred);
ADREPORT(cots_pred);

return nll;
