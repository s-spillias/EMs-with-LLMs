#include <TMB.hpp>

// Crown-of-Thorns Starfish (COTS) Outbreak Model
// Predicts episodic boom-bust cycles of COTS and their effects on coral communities

template<class Type>
Type objective_function<Type>::operator() ()
{
    // ============================
    // DATA INPUTS
    // ============================
    DATA_VECTOR(Year);         // Time steps (years)
    DATA_VECTOR(cots_dat);     // Observed COTS density (indiv/m^2)
    DATA_VECTOR(fast_dat);     // Observed fast-growing coral cover (%)
    DATA_VECTOR(slow_dat);     // Observed slow-growing coral cover (%)
    DATA_VECTOR(sst_dat);      // Sea surface temperature (°C)
    DATA_VECTOR(cotsimm_dat);  // COTS larval immigration (indiv/m^2/year)

    // ============================
    // MODEL PARAMETERS
    // ============================
    PARAMETER(log_r_cots);    // Intrinsic growth rate of COTS (log scale, year^-1)
    PARAMETER(log_K_cots);    // Carrying capacity of COTS (log scale, indiv/m^2)
    PARAMETER(alpha_fast);    // Attack rate coefficient on fast coral (Holling-II)
    PARAMETER(alpha_slow);    // Attack rate coefficient on slow coral (Holling-II)
    PARAMETER(attack_fast);   // Saturation/handling coefficient for fast coral (Holling-II)
    PARAMETER(attack_slow);   // Saturation/handling coefficient for slow coral (Holling-II)
    PARAMETER(log_r_fast);    // Growth rate of fast coral (log scale, year^-1)
    PARAMETER(log_r_slow);    // Growth rate of slow coral (log scale, year^-1)
    PARAMETER(beta_fast);     // Competition coefficient on fast coral recovery
    PARAMETER(beta_slow);     // Competition coefficient on slow coral recovery
    PARAMETER(theta_sst);     // Temperature sensitivity of COTS growth (per °C deviation)
    PARAMETER(log_sd_cots);   // Obs error sd for COTS (lognormal, log scale)
    PARAMETER(log_sd_fast);   // Obs error sd for fast coral (% cover, log scale)
    PARAMETER(log_sd_slow);   // Obs error sd for slow coral (% cover, log scale)

    // ============================
    // TRANSFORM PARAMETERS
    // ============================
    Type r_cots = exp(log_r_cots);     // Ensure positivity
    Type K_cots = exp(log_K_cots);
    Type r_fast = exp(log_r_fast);
    Type r_slow = exp(log_r_slow);
    Type sd_cots = exp(log_sd_cots) + Type(1e-6);
    Type sd_fast = exp(log_sd_fast) + Type(1e-6);
    Type sd_slow = exp(log_sd_slow) + Type(1e-6);

    int n = cots_dat.size();

    // ============================
    // STATE VARIABLES
    // ============================
    vector<Type> cots_pred(n);
    vector<Type> fast_pred(n);
    vector<Type> slow_pred(n);

    // ============================
    // INITIAL CONDITIONS
    // ============================
    cots_pred(0) = cots_dat(0);
    fast_pred(0) = fast_dat(0);
    slow_pred(0) = slow_dat(0);

    // ============================
    // PROCESS MODEL
    // ============================
    Type mean_sst = sum(sst_dat) / Type(n); // manually compute mean

    for(int t=1; t<n; t++){
        // 1. Environmental modification of growth by SST (smooth effect around mean)
        Type sst_dev = sst_dat(t-1) - mean_sst; 
        Type sst_effect = exp(theta_sst * sst_dev);

        // 2. COTS dynamics (boom-bust cycles with density-dependence, immigration, and resource limitation)
        Type resource_avail = fast_pred(t-1)/100.0 + 0.3 * slow_pred(t-1)/100.0;
        // clamp to max 1.0
        resource_avail = CppAD::CondExpLt(resource_avail, Type(1.0), resource_avail, Type(1.0));
        cots_pred(t) = cots_pred(t-1) +
                       r_cots * cots_pred(t-1) * (1 - cots_pred(t-1)/K_cots) * sst_effect * resource_avail +
                       cotsimm_dat(t-1);

        // Enforce non-negativity
        cots_pred(t) = CppAD::CondExpLt(cots_pred(t), Type(1e-8), Type(1e-8), cots_pred(t));

        // 3. Coral consumption by COTS (Holling type II functional response)
        Type cons_fast = (alpha_fast * cots_pred(t-1) * fast_pred(t-1)) / (1.0 + attack_fast * fast_pred(t-1));
        Type cons_slow = (alpha_slow * cots_pred(t-1) * slow_pred(t-1)) / (1.0 + attack_slow * slow_pred(t-1));

        // 4. Coral dynamics (logistic growth with predation losses and competition)
        fast_pred(t) = fast_pred(t-1) +
                       r_fast * fast_pred(t-1) * (1 - (fast_pred(t-1) + beta_fast*slow_pred(t-1))/100.0) -
                       cons_fast;

        slow_pred(t) = slow_pred(t-1) +
                       r_slow * slow_pred(t-1) * (1 - (slow_pred(t-1) + beta_slow*fast_pred(t-1))/100.0) -
                       cons_slow;

        // Enforce bounds (0-100% cover)
        fast_pred(t) = CppAD::CondExpLt(fast_pred(t), Type(1e-8), Type(1e-8),
                          CppAD::CondExpGt(fast_pred(t), Type(100.0), Type(100.0), fast_pred(t)));
        slow_pred(t) = CppAD::CondExpLt(slow_pred(t), Type(1e-8), Type(1e-8),
                          CppAD::CondExpGt(slow_pred(t), Type(100.0), Type(100.0), slow_pred(t)));
    }

    // ============================
    // LIKELIHOOD (OBSERVATION MODEL)
    // ============================
    Type nll = 0.0;

    // COTS - lognormal error
    for(int t=0; t<n; t++){
        nll -= dnorm(log(cots_dat(t) + Type(1e-8)),
                     log(cots_pred(t) + Type(1e-8)),
                     sd_cots,
                     true);
    }

    // Fast coral - lognormal error
    for(int t=0; t<n; t++){
        nll -= dnorm(log(fast_dat(t) + Type(1e-8)),
                     log(fast_pred(t) + Type(1e-8)),
                     sd_fast,
                     true);
    }

    // Slow coral - lognormal error
    for(int t=0; t<n; t++){
        nll -= dnorm(log(slow_dat(t) + Type(1e-8)),
                     log(slow_pred(t) + Type(1e-8)),
                     sd_slow,
                     true);
    }

    // ============================
    // REPORTING
    // ============================
    REPORT(cots_pred);
    REPORT(fast_pred);
    REPORT(slow_pred);

    return nll;
}
