#include <TMB.hpp>

// 1. Model equations describe the coupled dynamics of:
//    (1) COTS abundance (cots_pred)
//    (2) Fast-growing coral cover (fast_pred)
//    (3) Slow-growing coral cover (slow_pred)
//    All rates are per year, time is in years, and all state variables are positive and continuous.

// 2. Parameters are bounded within biologically meaningful ranges using smooth penalties.
// 3. Resource limitation is modeled with saturating (Michaelis-Menten) and threshold functions.
// 4. Environmental effects (e.g., SST) modify COTS and coral rates.
// 5. All _pred variables are reported for likelihood and output.

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA BLOCK ---
  DATA_VECTOR(Year); // Time in years
  DATA_VECTOR(sst_dat); // Sea-surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (indiv/m2/yr)
  DATA_VECTOR(cots_dat); // Observed adult COTS (indiv/m2)
  DATA_VECTOR(fast_dat); // Observed fast coral cover (%)
  DATA_VECTOR(slow_dat); // Observed slow coral cover (%)

  // --- PARAMETER BLOCK ---
  PARAMETER(log_r_cots); // log intrinsic COTS growth rate (year^-1)
  PARAMETER(log_K_cots); // log COTS carrying capacity (indiv/m2)
  PARAMETER(log_alpha_cots); // log COTS predation rate on coral (% cover^-1 yr^-1)
  PARAMETER(log_beta_fast); // log selectivity for fast coral (unitless)
  PARAMETER(log_beta_slow); // log selectivity for slow coral (unitless)
  PARAMETER(log_r_fast); // log fast coral recovery rate (yr^-1)
  PARAMETER(log_r_slow); // log slow coral recovery rate (yr^-1)
  PARAMETER(log_K_fast); // log fast coral max cover (%)
  PARAMETER(log_K_slow); // log slow coral max cover (%)
  PARAMETER(log_env_cots); // log environmental effect on COTS (per deg C)
  PARAMETER(log_env_fast); // log environmental effect on fast coral (per deg C)
  PARAMETER(log_env_slow); // log environmental effect on slow coral (per deg C)
  PARAMETER(log_sigma_cots); // log obs SD for COTS
  PARAMETER(log_sigma_fast); // log obs SD for fast coral
  PARAMETER(log_sigma_slow); // log obs SD for slow coral

  // --- TRANSFORM PARAMETERS ---
  Type r_cots = exp(log_r_cots); // COTS intrinsic growth rate (year^-1)
  Type K_cots = exp(log_K_cots); // COTS carrying capacity (indiv/m2)
  Type alpha_cots = exp(log_alpha_cots); // COTS predation rate on coral (% cover^-1 yr^-1)
  Type beta_fast = exp(log_beta_fast); // Selectivity for fast coral (unitless)
  Type beta_slow = exp(log_beta_slow); // Selectivity for slow coral (unitless)
  Type r_fast = exp(log_r_fast); // Fast coral recovery rate (yr^-1)
  Type r_slow = exp(log_r_slow); // Slow coral recovery rate (yr^-1)
  Type K_fast = exp(log_K_fast); // Fast coral max cover (%)
  Type K_slow = exp(log_K_slow); // Slow coral max cover (%)
  Type env_cots = exp(log_env_cots); // Environmental effect on COTS (per deg C)
  Type env_fast = exp(log_env_fast); // Environmental effect on fast coral (per deg C)
  Type env_slow = exp(log_env_slow); // Environmental effect on slow coral (per deg C)
  Type sigma_cots = exp(log_sigma_cots) + Type(1e-4); // SD for COTS obs
  Type sigma_fast = exp(log_sigma_fast) + Type(1e-4); // SD for fast coral obs
  Type sigma_slow = exp(log_sigma_slow) + Type(1e-4); // SD for slow coral obs

  // --- INITIAL CONDITIONS ---
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Set initial state to observed at t=0
  cots_pred(0) = CppAD::CondExpGt(cots_dat(0), Type(0.0), cots_dat(0), Type(1e-8));
  fast_pred(0) = CppAD::CondExpGt(fast_dat(0), Type(0.0), fast_dat(0), Type(1e-8));
  slow_pred(0) = CppAD::CondExpGt(slow_dat(0), Type(0.0), slow_dat(0), Type(1e-8));

  // --- PROCESS MODEL ---
  for(int t=1; t<n; t++) {
    // 1. Environmental modifiers (centered at mean SST)
    Type sst_dev = sst_dat(t-1) - Type(27.0); // deviation from reference SST (deg C)
    Type env_mod_cots = exp(env_cots * sst_dev); // effect of SST on COTS
    Type env_mod_fast = exp(env_fast * sst_dev); // effect of SST on fast coral
    Type env_mod_slow = exp(env_slow * sst_dev); // effect of SST on slow coral

    // 2. Coral resource limitation (Michaelis-Menten, saturating)
    Type coral_avail = beta_fast * fast_pred(t-1) + beta_slow * slow_pred(t-1) + Type(1e-8); // total edible coral (% cover)
    Type coral_limit = coral_avail / (coral_avail + Type(10.0)); // saturating function

    // 3. COTS population dynamics (boom-bust)
    //    Growth + immigration - mortality (resource-limited, environmentally modified)
    Type cots_growth = r_cots * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1)/K_cots) * coral_limit * env_mod_cots;
    // Defensive: ensure cotsimm_dat is not negative or NA (TMB does not support NA, but user data might have -999 etc)
    Type cots_imm = CppAD::CondExpGt(cotsimm_dat(t-1), Type(0.0), cotsimm_dat(t-1), Type(0.0)); // exogenous larval input (indiv/m2/yr)
    Type cots_mort = alpha_cots * cots_pred(t-1); // density-dependent mortality (predation, disease, etc)
    cots_pred(t) = cots_pred(t-1) + cots_growth + cots_imm - cots_mort;
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), Type(0.0), cots_pred(t), Type(1e-8)); // enforce positivity

    // 4. Coral predation losses (Type II functional response)
    Type pred_fast = alpha_cots * beta_fast * cots_pred(t-1) * fast_pred(t-1) / (fast_pred(t-1) + Type(5.0) + Type(1e-8));
    Type pred_slow = alpha_cots * beta_slow * cots_pred(t-1) * slow_pred(t-1) / (slow_pred(t-1) + Type(5.0) + Type(1e-8));

    // 5. Coral recovery (logistic, environmentally modified)
    Type fast_recov = r_fast * fast_pred(t-1) * (Type(1.0) - fast_pred(t-1)/K_fast) * env_mod_fast;
    Type slow_recov = r_slow * slow_pred(t-1) * (Type(1.0) - slow_pred(t-1)/K_slow) * env_mod_slow;

    // 6. Coral update equations
    fast_pred(t) = fast_pred(t-1) + fast_recov - pred_fast;
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), Type(0.0), fast_pred(t), Type(1e-8)); // enforce positivity

    slow_pred(t) = slow_pred(t-1) + slow_recov - pred_slow;
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), Type(0.0), slow_pred(t), Type(1e-8)); // enforce positivity
  }

  // --- LIKELIHOOD ---
  Type nll = Type(0.0);
  for(int t=0; t<n; t++) {
    // Lognormal likelihood for strictly positive data
    // Defensive: skip likelihood if any obs is negative (TMB does not support NA, but user data might have -999 etc)
    if (cots_dat(t) > Type(0.0)) {
      nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t) + Type(1e-8)), sigma_cots, true);
    }
    if (fast_dat(t) > Type(0.0)) {
      nll -= dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred(t) + Type(1e-8)), sigma_fast, true);
    }
    if (slow_dat(t) > Type(0.0)) {
      nll -= dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred(t) + Type(1e-8)), sigma_slow, true);
    }
  }

  // --- SMOOTH PENALTIES FOR PARAMETER BOUNDS ---
  // Example: penalize if COTS growth rate is outside plausible range (0.1-3.0 yr^-1)
  nll += pow(CppAD::CondExpLt(r_cots, Type(0.1), r_cots-Type(0.1), Type(0.0)), 2) * Type(10.0);
  nll += pow(CppAD::CondExpGt(r_cots, Type(3.0), r_cots-Type(3.0), Type(0.0)), 2) * Type(10.0);

  // --- REPORTING ---
  REPORT(cots_pred); // predicted COTS abundance (indiv/m2)
  REPORT(fast_pred); // predicted fast coral cover (%)
  REPORT(slow_pred); // predicted slow coral cover (%)
  return nll;
}

/*
Equation list:
1. cots_pred(t) = cots_pred(t-1) + r_cots * cots_pred(t-1) * (1 - cots_pred(t-1)/K_cots) * coral_limit * env_mod_cots + cotsimm_dat(t-1) - alpha_cots * cots_pred(t-1)
2. coral_limit = (beta_fast * fast_pred(t-1) + beta_slow * slow_pred(t-1)) / (beta_fast * fast_pred(t-1) + beta_slow * slow_pred(t-1) + 10)
3. pred_fast = alpha_cots * beta_fast * cots_pred(t-1) * fast_pred(t-1) / (fast_pred(t-1) + 5)
4. pred_slow = alpha_cots * beta_slow * cots_pred(t-1) * slow_pred(t-1) / (slow_pred(t-1) + 5)
5. fast_pred(t) = fast_pred(t-1) + r_fast * fast_pred(t-1) * (1 - fast_pred(t-1)/K_fast) * env_mod_fast - pred_fast
6. slow_pred(t) = slow_pred(t-1) + r_slow * slow_pred(t-1) * (1 - slow_pred(t-1)/K_slow) * env_mod_slow - pred_slow
7. env_mod_* = exp(env_* * (sst_dat(t-1) - 27.0))
*/
