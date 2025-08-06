#include <TMB.hpp>

// TMB model for Crown-of-Thorns starfish predation dynamics on coral reefs.
// Equations description:
// 1. Consumption: Coral loss due to predation is proportional to starfish abundance and consumption rates.
// 2. Coral Recovery: Coral cover recovers logistically with intrinsic growth rates reduced by predation.
// 3. Likelihood: Lognormal likelihood is used for strictly positive coral cover data, with fixed minimum standard deviation.

template<class Type>
Type objective_function<Type>::operator()() {
    // Data:
    DATA_VECTOR(cots_dat);  // Observed Crown-of-Thorns starfish abundance (individuals/m2)
    DATA_VECTOR(slow_dat);  // Observed slow-growing coral cover (%) - Faviidae and Porites
    DATA_VECTOR(fast_dat);  // Observed fast-growing coral cover (%) - Acropora
    DATA_VECTOR(sst_dat);   // Sea surface temperature forcing (C)
    DATA_VECTOR(cotsimm_dat); // Crown-of-thorns immigration rate (individuals/m2/year)

    int n = cots_dat.size();
    Type ans = 0.0;
    
    // Parameters with comments (units in parentheses, determination notes):
    PARAMETER(log_consumption_rate_slow); // ln(consumption rate on slow corals; (1/year); estimated via lab feeding trials)
    PARAMETER(log_consumption_rate_fast); // ln(consumption rate on fast corals; (1/year); estimated via field observations)
    PARAMETER(log_growth_rate_slow);        // ln(coral intrinsic recovery rate for slow corals; (year^-1); from literature)
    PARAMETER(log_growth_rate_fast);        // ln(coral intrinsic recovery rate for fast corals; (year^-1); from literature)
    PARAMETER(carrying_capacity_slow);      // Carrying capacity for slow-growing corals (%) [expected range: 0-100]
    PARAMETER(carrying_capacity_fast);      // Carrying capacity for fast-growing corals (%) [expected range: 0-100]
    PARAMETER(log_sigma);                   // ln(observation error standard deviation; used in lognormal likelihood)
    PARAMETER(cots0);      // Initial COTS abundance (individuals/m2)
    PARAMETER(slow0);      // Initial slow-growing coral cover (%)
    PARAMETER(fast0);      // Initial fast-growing coral cover (%)
    PARAMETER(log_mortality); // ln(mortality rate for COTS; (1/year); estimated from observations)
    PARAMETER(sst_coeff); // Coefficient for effect of sea surface temperature (C) on COTS dynamics (unitless scaling factor)
    PARAMETER(cotsimm_coeff); // Coefficient for COTS immigration forcing (unitless scaling factor)
    PARAMETER(impact_rate_slow); // Impact rate of COTS on slow-growing corals (per individual)
    PARAMETER(impact_rate_fast); // Impact rate of COTS on fast-growing corals (per individual)

    // Transform log-parameters:
    Type consumption_rate_slow = exp(log_consumption_rate_slow); // (1/year)
    Type consumption_rate_fast = exp(log_consumption_rate_fast); // (1/year)
    Type growth_rate_slow = exp(log_growth_rate_slow);           // (year^-1)
    Type growth_rate_fast = exp(log_growth_rate_fast);           // (year^-1)
    Type sigma = exp(log_sigma) + Type(1e-8);                    // Ensure sigma > 0
    Type mortality = exp(log_mortality);                       // (1/year)

    // Smooth penalty for carrying capacity parameters to keep them in biologically plausible ranges:
    // (Using a normal prior centered at 50% with SD = 20)
    ans += -dnorm(carrying_capacity_slow, Type(50), Type(20), true);
    ans += -dnorm(carrying_capacity_fast, Type(50), Type(20), true);

    // Dynamic prediction equations revised to avoid data leakage:
    // Equation 1: COTS dynamics: multiplicative update using external immigration forcing and mortality.
    vector<Type> cots_pred(n), slow_pred(n), fast_pred(n);
    cots_pred(0) = cots0;
    slow_pred(0) = slow0;
    fast_pred(0) = fast0;
    for (int t = 1; t < n; t++){
        // Equation 1: Update COTS prediction using external forcing from sst_dat and cotsimm_dat.
        cots_pred[t] = cots_pred[t-1] * exp(sst_coeff * sst_dat[t] - mortality) + cotsimm_coeff * cotsimm_dat[t];
        // Equation 2: Slow-growing coral prediction updated multiplicatively using previous predicted state.
        slow_pred[t] = slow_pred[t-1] * growth_rate_slow * (1 - impact_rate_slow * cots_pred[t-1]);
        // Equation 3: Fast-growing coral prediction updated multiplicatively using previous predicted state.
        fast_pred[t] = fast_pred[t-1] * growth_rate_fast * (1 - impact_rate_fast * cots_pred[t-1]);
    }
    
    // Likelihood: Compare predictions to observations using lognormal likelihood.
    for (int i = 0; i < n; i++){
        ans -= dlnorm(cots_dat[i] + Type(1e-8), log(cots_pred[i] + Type(1e-8)), sigma, true);
        ans -= dlnorm(slow_dat[i] + Type(1e-8), log(slow_pred[i] + Type(1e-8)), sigma, true);
        ans -= dlnorm(fast_dat[i] + Type(1e-8), log(fast_pred[i] + Type(1e-8)), sigma, true);
    }

    // Reporting important variables with '_pred' suffix for model predictions.
    ADREPORT(consumption_rate_slow);
    ADREPORT(consumption_rate_fast);
    ADREPORT(growth_rate_slow);
    ADREPORT(growth_rate_fast);
    ADREPORT(carrying_capacity_slow);
    ADREPORT(carrying_capacity_fast);
    ADREPORT(sigma);

    return ans;
}
