#include <TMB.hpp>

// Crown-of-Thorns Starfish (COTS) - Coral dynamics model
// Captures episodic outbreaks, boom-bust population cycles, and selective predation on coral communities
// Indexing: predictions at time t are based ONLY on state variables from time t-1 (no data leakage)

template<class Type>
Type objective_function<Type>::operator() ()
{
    // ---- DATA INPUTS ----
    DATA_VECTOR(Year);            // Year (time variable)
    DATA_VECTOR(cots_dat);        // Observed adult COTS density (ind/m2)
    DATA_VECTOR(fast_dat);        // Observed fast coral cover (%) - Acropora
    DATA_VECTOR(slow_dat);        // Observed slow coral cover (%) - Faviidae/Porites
    DATA_VECTOR(sst_dat);         // Observed SST (Celsius)
    DATA_VECTOR(cotsimm_dat);     // Observed larval immigration rate (ind/m2/year)

    int n = Year.size();

    // ---- PARAMETERS ----
    PARAMETER(log_cots_r);        // Log intrinsic growth rate for COTS (year^-1)
    PARAMETER(log_cots_K);        // Log carrying capacity scaling for COTS relative to coral cover
    PARAMETER(log_cots_m);        // Log natural mortality rate of COTS (year^-1)
    PARAMETER(log_attack_fast);   // Log attack rate on fast-growing corals (m2/ind/year)
    PARAMETER(log_attack_slow);   // Log attack rate on slow-growing corals (m2/ind/year)
    PARAMETER(log_handling);      // Log handling time constant for predation (year)
    PARAMETER(log_regrow_fast);   // Log intrinsic regrowth rate fast corals (%/year)
    PARAMETER(log_regrow_slow);   // Log intrinsic regrowth rate slow corals (%/year)
    PARAMETER(log_env_sst);       // Coefficient modifying coral growth response to SST deviation
    PARAMETER(log_sd_obs);        // Log observation error SD for lognormal likelihood

    // Transform back to natural scale
    Type cots_r       = exp(log_cots_r);
    Type cots_K       = exp(log_cots_K);
    Type cots_m       = exp(log_cots_m);
    Type attack_fast  = exp(log_attack_fast);
    Type attack_slow  = exp(log_attack_slow);
    Type handling     = exp(log_handling);
    Type regrow_fast  = exp(log_regrow_fast);
    Type regrow_slow  = exp(log_regrow_slow);
    Type env_sst      = exp(log_env_sst);
    Type sd_obs       = exp(log_sd_obs);

    // ---- STATE VARIABLES ----
    vector<Type> cots_pred(n);
    vector<Type> fast_pred(n);
    vector<Type> slow_pred(n);

    // ---- INITIAL CONDITIONS ----
    cots_pred(0) = cots_dat(0) + Type(1e-6);
    fast_pred(0) = fast_dat(0) + Type(1e-6);
    slow_pred(0) = slow_dat(0) + Type(1e-6);

    // ---- PROCESS MODEL ----
    for(int t=1; t<n; t++){
        // Available coral resources
        Type coral_total_prev = fast_pred(t-1) + slow_pred(t-1) + Type(1e-8);
        Type free_space = fmax(Type(0.0), Type(100.0) - coral_total_prev);

        // Functional response for predation (Holling type II-like)
        Type coral_consumed_fast = (attack_fast * cots_pred(t-1) * fast_pred(t-1)) /
                                   (Type(1.0) + handling * (attack_fast * fast_pred(t-1) + attack_slow * slow_pred(t-1)) + Type(1e-8));

        Type coral_consumed_slow = (attack_slow * cots_pred(t-1) * slow_pred(t-1)) /
                                   (Type(1.0) + handling * (attack_fast * fast_pred(t-1) + attack_slow * slow_pred(t-1)) + Type(1e-8));

        // Coral growth influenced by SST
        Type sst_dev = sst_dat(t) - Type(27.0); // reference optimum ~27C
        Type growth_modifier = exp(-env_sst * sst_dev * sst_dev);

        Type fast_growth = regrow_fast * fast_pred(t-1) * (free_space/Type(100.0)) * growth_modifier;
        Type slow_growth = regrow_slow * slow_pred(t-1) * (free_space/Type(100.0)) * growth_modifier;

        // Coral updates
        fast_pred(t) = fmax(Type(0.0), fast_pred(t-1) + fast_growth - coral_consumed_fast);
        slow_pred(t) = fmax(Type(0.0), slow_pred(t-1) + slow_growth - coral_consumed_slow);

        // COTS population dynamics
        Type carrying_capacity = cots_K * (fast_pred(t-1) + Type(0.5)*slow_pred(t-1) + Type(1e-8));
        Type logistic_growth = cots_r * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1)/carrying_capacity);
        Type mortality = cots_m * cots_pred(t-1);

        cots_pred(t) = fmax(Type(1e-8),
                                   cots_pred(t-1) + logistic_growth - mortality +
                                   cotsimm_dat(t) );
    }

    // ---- LIKELIHOOD ----
    Type nll = 0.0;
    for(int t=0; t<n; t++){
        nll -= dlognorm(cots_dat(t) + Type(1e-8), log(cots_pred(t) + Type(1e-8)), sd_obs, true);
        nll -= dlognorm(fast_dat(t) + Type(1e-8), log(fast_pred(t) + Type(1e-8)), sd_obs, true);
        nll -= dlognorm(slow_dat(t) + Type(1e-8), log(slow_pred(t) + Type(1e-8)), sd_obs, true);
    }

    // ---- REPORT SECTION ----
    REPORT(cots_pred);
    REPORT(fast_pred);
    REPORT(slow_pred);

    return nll;
}

/*
EQUATION DESCRIPTIONS
1. Coral mortality from COTS predation follows a Holling type II functional response with different attack rates for fast vs slow corals.
2. Coral regrowth follows logistic colonization into available free space, modified by SST deviation.
3. COTS dynamics follow logistic growth limited by coral-dependent carrying capacity, with losses due to mortality and gains from occasional larval immigration pulses.
4. Likelihood is lognormal to reflect positive continuous abundance/cover data spanning multiple orders of magnitude.
*/
