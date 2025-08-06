#include <TMB.hpp>

// Template Model Builder requires all code to be inside a function template<class Type>
template<class Type>
Type objective_function<Type>::operator() ()
{
    // 1. DATA SECTION
    // ---------------

    // Time vector (years)
    DATA_VECTOR(Year); // Observation years

    // Observed adult COTS abundance (individuals/m2)
    DATA_VECTOR(cots_dat);

    // Observed fast-growing coral cover (Acropora spp., %)
    DATA_VECTOR(fast_dat);

    // Observed slow-growing coral cover (Faviidae/Porites spp., %)
    DATA_VECTOR(slow_dat);

    // Environmental covariates
    DATA_VECTOR(sst_dat);      // Sea-surface temperature (deg C)
    DATA_VECTOR(cotsimm_dat);  // COTS larval immigration rate (indiv/m2/year)

    // 2. PARAMETER SECTION
    // --------------------

    // COTS population parameters
    PARAMETER(log_r_cots);      // log intrinsic growth rate of COTS (year^-1)
    PARAMETER(log_K_cots);      // log carrying capacity for COTS (indiv/m2)
    PARAMETER(log_alpha_cots);  // log predation attack rate on coral (m2/indiv/year)
    PARAMETER(log_beta_cots);   // log half-saturation coral cover for COTS predation (%, for functional response)
    PARAMETER(log_m_cots);      // log baseline COTS mortality rate (year^-1)
    PARAMETER(log_eps_cots);    // log process error SD for COTS

    // Coral parameters
    PARAMETER(log_r_fast);      // log intrinsic growth rate of fast coral (year^-1)
    PARAMETER(log_r_slow);      // log intrinsic growth rate of slow coral (year^-1)
    PARAMETER(log_K_coral);     // log total coral carrying capacity (% cover)
    PARAMETER(log_eps_fast);    // log process error SD for fast coral
    PARAMETER(log_eps_slow);    // log process error SD for slow coral

    // COTS predation selectivity
    PARAMETER(logit_sel_fast);  // logit selectivity of COTS for fast coral (proportion)
    PARAMETER(logit_sel_slow);  // logit selectivity of COTS for slow coral (proportion)

    // Outbreak threshold and environmental effects
    PARAMETER(logit_outbreak_thresh); // logit threshold for outbreak initiation (indiv/m2)
    PARAMETER(beta_sst);              // effect of SST on COTS growth (per deg C)
    PARAMETER(beta_imm);              // effect of larval immigration on COTS recruitment

    // Observation error
    PARAMETER(log_sigma_cots);   // log obs error SD for COTS
    PARAMETER(log_sigma_fast);   // log obs error SD for fast coral
    PARAMETER(log_sigma_slow);   // log obs error SD for slow coral

    // 3. TRANSFORM PARAMETERS
    // -----------------------

    Type r_cots = exp(log_r_cots);           // Intrinsic COTS growth rate (year^-1)
    Type K_cots = exp(log_K_cots);           // COTS carrying capacity (indiv/m2)
    Type alpha_cots = exp(log_alpha_cots);   // COTS attack rate (m2/indiv/year)
    Type beta_cots = exp(log_beta_cots);     // Coral cover half-saturation for COTS predation (%)
    Type m_cots = exp(log_m_cots);           // Baseline COTS mortality (year^-1)
    Type eps_cots = exp(log_eps_cots);       // COTS process error SD

    Type r_fast = exp(log_r_fast);           // Fast coral growth rate (year^-1)
    Type r_slow = exp(log_r_slow);           // Slow coral growth rate (year^-1)
    Type K_coral = exp(log_K_coral);         // Total coral carrying capacity (%)
    Type eps_fast = exp(log_eps_fast);       // Fast coral process error SD
    Type eps_slow = exp(log_eps_slow);       // Slow coral process error SD

    Type sel_fast = 1.0 / (1.0 + exp(-logit_sel_fast)); // Selectivity for fast coral (0-1)
    Type sel_slow = 1.0 / (1.0 + exp(-logit_sel_slow)); // Selectivity for slow coral (0-1)

    Type outbreak_thresh = 1.0 / (1.0 + exp(-logit_outbreak_thresh)); // Outbreak threshold (proportion of K_cots)

    Type sigma_cots = exp(log_sigma_cots);   // Obs error SD for COTS
    Type sigma_fast = exp(log_sigma_fast);   // Obs error SD for fast coral
    Type sigma_slow = exp(log_sigma_slow);   // Obs error SD for slow coral

    // Resource-dependent COTS recruitment efficiency (Hill function)
    PARAMETER(coral_cover_thresh); // critical coral cover threshold for COTS recruitment (proportion of K_coral))
    Type coral_cover_threshold = coral_cover_thresh * K_coral + Type(1e-8); // absolute threshold (% cover), avoid zero
    Type coral_cover_hill_k = 10.0; // Hill coefficient for steepness (fixed, can be parameterized if needed)

    // 4. INITIAL CONDITIONS
    // ---------------------
    int n = Year.size();
    vector<Type> cots_pred(n);
    vector<Type> fast_pred(n);
    vector<Type> slow_pred(n);

    // Set initial states to observed values at t=0 (could be parameters if desired)
    cots_pred(0) = CppAD::CondExpGt(cots_dat(0), Type(1e-8), cots_dat(0), Type(1e-8));
    fast_pred(0) = CppAD::CondExpGt(fast_dat(0), Type(1e-8), fast_dat(0), Type(1e-8));
    slow_pred(0) = CppAD::CondExpGt(slow_dat(0), Type(1e-8), slow_dat(0), Type(1e-8));

    // 5. PROCESS MODEL
    // ----------------
    // Numbered equation descriptions:
    // 1. COTS population: density-dependent growth + resource limitation + environmental forcing + process error
    // 2. Coral groups: logistic growth - COTS predation (selective) + process error
    // 3. COTS predation: Holling Type II functional response, saturating with coral cover, selectivity by coral type
    // 4. Outbreaks: smooth threshold for outbreak initiation, modulated by SST and immigration
    // 5. All rates bounded and transitions smoothed for numerical stability

    for(int t=1; t<n; t++) {
        // Resource limitation: total coral cover available
        Type coral_total_prev = fast_pred(t-1) + slow_pred(t-1) + Type(1e-8); // % cover, avoid zero

        // COTS predation rates (Holling Type II functional response)
        Type pred_fast = alpha_cots * sel_fast * cots_pred(t-1) * fast_pred(t-1) / (beta_cots + fast_pred(t-1) + Type(1e-8)); // indiv/m2/year
        Type pred_slow = alpha_cots * sel_slow * cots_pred(t-1) * slow_pred(t-1) / (beta_cots + slow_pred(t-1) + Type(1e-8)); // indiv/m2/year

        // Outbreak effect: smooth threshold on COTS recruitment
        Type outbreak_factor = 1.0 / (1.0 + exp(-20.0 * (cots_pred(t-1)/K_cots - outbreak_thresh))); // rapid transition near threshold

        // Environmental effects
        Type env_effect = exp(beta_sst * (sst_dat(t-1) - 27.0)); // SST effect, baseline at 27C
        Type imm_effect = 1.0 + beta_imm * cotsimm_dat(t-1);     // Immigration effect

        // Resource-dependent COTS recruitment efficiency (Hill function, steep drop below threshold)
        Type coral_cover_eff = pow(coral_total_prev, coral_cover_hill_k) /
                               (pow(coral_total_prev, coral_cover_hill_k) + pow(coral_cover_threshold, coral_cover_hill_k) + Type(1e-8));

        // COTS population update (Eq 1, with new feedback)
        Type cots_growth = r_cots * cots_pred(t-1) * (1.0 - cots_pred(t-1)/K_cots) * (coral_total_prev/K_coral) * env_effect * imm_effect * outbreak_factor * coral_cover_eff;
        cots_pred(t) = cots_pred(t-1) + cots_growth - m_cots * cots_pred(t-1) + eps_cots * pow(Type(1e-8) + cots_pred(t-1), 0.5) * rnorm(Type(0), Type(1));
        cots_pred(t) = CppAD::CondExpGt(cots_pred(t), Type(1e-8), cots_pred(t), Type(1e-8)); // Bound to positive

        // Fast coral update (Eq 2)
        Type fast_growth = r_fast * fast_pred(t-1) * (1.0 - (fast_pred(t-1) + slow_pred(t-1))/K_coral);
        fast_pred(t) = fast_pred(t-1) + fast_growth - pred_fast + eps_fast * pow(Type(1e-8) + fast_pred(t-1), 0.5) * rnorm(Type(0), Type(1));
        fast_pred(t) = CppAD::CondExpGt(fast_pred(t), Type(1e-8), fast_pred(t), Type(1e-8)); // Bound to positive

        // Slow coral update (Eq 2)
        Type slow_growth = r_slow * slow_pred(t-1) * (1.0 - (fast_pred(t-1) + slow_pred(t-1))/K_coral);
        slow_pred(t) = slow_pred(t-1) + slow_growth - pred_slow + eps_slow * pow(Type(1e-8) + slow_pred(t-1), 0.5) * rnorm(Type(0), Type(1));
        slow_pred(t) = CppAD::CondExpGt(slow_pred(t), Type(1e-8), slow_pred(t), Type(1e-8)); // Bound to positive
    }

    // 6. LIKELIHOOD
    // -------------
    // Use lognormal likelihood for strictly positive data, with minimum SD for stability

    Type nll = 0.0;
    Type min_sd = Type(1e-3);

    // COTS
    for(int t=0; t<n; t++) {
        Type sd = sqrt(sigma_cots*sigma_cots + min_sd*min_sd);
        nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t) + Type(1e-8)), sd, true);
    }

    // Fast coral
    for(int t=0; t<n; t++) {
        Type sd = sqrt(sigma_fast*sigma_fast + min_sd*min_sd);
        nll -= dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred(t) + Type(1e-8)), sd, true);
    }

    // Slow coral
    for(int t=0; t<n; t++) {
        Type sd = sqrt(sigma_slow*sigma_slow + min_sd*min_sd);
        nll -= dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred(t) + Type(1e-8)), sd, true);
    }

    // 7. REPORTING
    // ------------
    REPORT(cots_pred);  // Predicted COTS abundance (indiv/m2)
    REPORT(fast_pred);  // Predicted fast coral cover (%)
    REPORT(slow_pred);  // Predicted slow coral cover (%)

    return nll;
}
