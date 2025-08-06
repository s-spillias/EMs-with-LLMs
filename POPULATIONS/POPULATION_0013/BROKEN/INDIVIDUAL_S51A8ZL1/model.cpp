#include <TMB.hpp>

// Model for episodic outbreaks of Crown-of-Thorns starfish (COTS) and their impact on coral communities

template<class Type>
Type objective_function<Type>::operator() ()
{
  // 1. DATA INPUTS
  DATA_VECTOR(Year); // Time variable (years)
  DATA_VECTOR(cots_dat); // COTS abundance (individuals/m^2)
  DATA_VECTOR(fast_dat); // Fast-growing coral cover (Acropora spp., %)
  DATA_VECTOR(slow_dat); // Slow-growing coral cover (Faviidae/Porites, %)
  DATA_VECTOR(sst_dat); // Sea-surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration rate (individuals/m^2/year)

  int n = Year.size();

  // Defensive: check all data vectors are the same length as Year
  if ((cots_dat.size() != n) ||
      (fast_dat.size() != n) ||
      (slow_dat.size() != n) ||
      (sst_dat.size() != n) ||
      (cotsimm_dat.size() != n)) {
    error("All data vectors must have the same length as Year.");
  }

  // 2. PARAMETERS
  PARAMETER(log_r_cots); // log intrinsic growth rate of COTS (year^-1)
  PARAMETER(log_K_cots); // log carrying capacity of COTS (individuals/m^2)
  PARAMETER(log_alpha_fast); // log attack rate on fast coral (m2/individual/year)
  PARAMETER(log_alpha_slow); // log attack rate on slow coral (m2/individual/year)
  PARAMETER(log_e_fast); // log assimilation efficiency from fast coral (unitless)
  PARAMETER(log_e_slow); // log assimilation efficiency from slow coral (unitless)
  PARAMETER(log_r_fast); // log growth rate of fast coral (year^-1)
  PARAMETER(log_r_slow); // log growth rate of slow coral (year^-1)
  PARAMETER(log_K_fast); // log carrying capacity of fast coral (% cover)
  PARAMETER(log_K_slow); // log carrying capacity of slow coral (% cover)
  PARAMETER(log_sigma_cots); // log SD of COTS obs error
  PARAMETER(log_sigma_fast); // log SD of fast coral obs error
  PARAMETER(log_sigma_slow); // log SD of slow coral obs error
  PARAMETER(beta_sst); // effect of SST on COTS growth (per deg C)
  PARAMETER(log_immig_scale); // log scaling for larval immigration effect
  PARAMETER(log_thresh_fast); // log threshold coral cover for COTS outbreak (fast coral, %)
  PARAMETER(log_thresh_slow); // log threshold coral cover for COTS outbreak (slow coral, %)
  PARAMETER(log_min_cots); // log minimum COTS density (for numerical stability)
  PARAMETER(log_min_coral); // log minimum coral cover (for numerical stability)

  // 3. TRANSFORM PARAMETERS
  Type r_cots = exp(log_r_cots); // Intrinsic growth rate of COTS
  Type K_cots = exp(log_K_cots); // Carrying capacity of COTS
  Type alpha_fast = exp(log_alpha_fast); // Attack rate on fast coral
  Type alpha_slow = exp(log_alpha_slow); // Attack rate on slow coral
  Type e_fast = exp(log_e_fast); // Assimilation efficiency from fast coral
  Type e_slow = exp(log_e_slow); // Assimilation efficiency from slow coral
  Type r_fast = exp(log_r_fast); // Growth rate of fast coral
  Type r_slow = exp(log_r_slow); // Growth rate of slow coral
  Type K_fast = exp(log_K_fast); // Carrying capacity of fast coral
  Type K_slow = exp(log_K_slow); // Carrying capacity of slow coral
  Type sigma_cots = exp(log_sigma_cots); // SD of COTS obs error
  Type sigma_fast = exp(log_sigma_fast); // SD of fast coral obs error
  Type sigma_slow = exp(log_sigma_slow); // SD of slow coral obs error
  Type immig_scale = exp(log_immig_scale); // Scaling for larval immigration
  Type thresh_fast = exp(log_thresh_fast); // Threshold for fast coral cover
  Type thresh_slow = exp(log_thresh_slow); // Threshold for slow coral cover
  Type min_cots = exp(log_min_cots); // Minimum COTS density
  Type min_coral = exp(log_min_coral); // Minimum coral cover

  // 4. STATE VARIABLES
  vector<Type> cots_pred(n); // Predicted COTS abundance
  vector<Type> fast_pred(n); // Predicted fast coral cover
  vector<Type> slow_pred(n); // Predicted slow coral cover

  // 5. INITIAL CONDITIONS
  cots_pred(0) = CppAD::CondExpGt(cots_dat(0), Type(0), cots_dat(0), min_cots); // Use data or min
  fast_pred(0) = CppAD::CondExpGt(fast_dat(0), Type(0), fast_dat(0), min_coral);
  slow_pred(0) = CppAD::CondExpGt(slow_dat(0), Type(0), slow_dat(0), min_coral);

  // 6. PROCESS MODEL
  for(int t=1; t<n; t++){
    // Defensive: ensure previous values are positive (no NaN/Inf checks, just lower bound)
    Type fast_prev = CppAD::CondExpGt(fast_pred(t-1), Type(0), fast_pred(t-1), min_coral);
    Type slow_prev = CppAD::CondExpGt(slow_pred(t-1), Type(0), slow_pred(t-1), min_coral);
    Type cots_prev = CppAD::CondExpGt(cots_pred(t-1), Type(0), cots_pred(t-1), min_cots);

    // Resource limitation: saturating function of coral cover (sum of both types)
    Type coral_total_prev = fast_prev + slow_prev + Type(1e-8);

    // Outbreak trigger: smooth threshold on coral cover (fast and slow)
    Type outbreak_trigger = 1.0 / (1.0 + exp(-10.0 * ((fast_prev - thresh_fast) + (slow_prev - thresh_slow))));

    // Environmental effect: SST modifies COTS growth
    Type env_mod = exp(beta_sst * (sst_dat(t) - Type(27.0))); // 27C as reference

    // Immigration effect (larval supply)
    Type immig = immig_scale * cotsimm_dat(t);

    // COTS predation rates (functional response: Holling Type II)
    Type pred_fast = alpha_fast * cots_prev * fast_prev / (fast_prev + Type(1.0) + Type(1e-8));
    Type pred_slow = alpha_slow * cots_prev * slow_prev / (slow_prev + Type(1.0) + Type(1e-8));

    // COTS population update (boom-bust, resource-limited, outbreak-triggered)
    Type cots_growth = r_cots * cots_prev * (Type(1.0) - cots_prev/K_cots) * (coral_total_prev/(coral_total_prev + Type(10.0))) * outbreak_trigger * env_mod;
    cots_pred(t) = CppAD::CondExpGt(
      cots_prev + cots_growth + e_fast * pred_fast + e_slow * pred_slow + immig,
      min_cots, 
      cots_prev + cots_growth + e_fast * pred_fast + e_slow * pred_slow + immig,
      min_cots
    );

    // Fast coral update (logistic growth minus COTS predation)
    Type fast_growth = r_fast * fast_prev * (Type(1.0) - fast_prev/K_fast);
    fast_pred(t) = CppAD::CondExpGt(
      fast_prev + fast_growth - pred_fast,
      min_coral,
      fast_prev + fast_growth - pred_fast,
      min_coral
    );

    // Slow coral update (logistic growth minus COTS predation)
    Type slow_growth = r_slow * slow_prev * (Type(1.0) - slow_prev/K_slow);
    slow_pred(t) = CppAD::CondExpGt(
      slow_prev + slow_growth - pred_slow,
      min_coral,
      slow_prev + slow_growth - pred_slow,
      min_coral
    );
  }

  // 7. LIKELIHOOD (lognormal, robust to zeros, fixed min SD)
  Type nll = 0.0;
  Type min_sd = Type(1e-3);

  for(int t=0; t<n; t++){
    // COTS
    nll -= dnorm(log(CppAD::CondExpGt(cots_dat(t), Type(0), cots_dat(t), Type(1e-8))),
                 log(CppAD::CondExpGt(cots_pred(t), Type(0), cots_pred(t), Type(1e-8))),
                 sigma_cots + min_sd, true);
    // Fast coral
    nll -= dnorm(log(CppAD::CondExpGt(fast_dat(t), Type(0), fast_dat(t), Type(1e-8))),
                 log(CppAD::CondExpGt(fast_pred(t), Type(0), fast_pred(t), Type(1e-8))),
                 sigma_fast + min_sd, true);
    // Slow coral
    nll -= dnorm(log(CppAD::CondExpGt(slow_dat(t), Type(0), slow_dat(t), Type(1e-8))),
                 log(CppAD::CondExpGt(slow_pred(t), Type(0), slow_pred(t), Type(1e-8))),
                 sigma_slow + min_sd, true);
  }

  // 8. SMOOTH PENALTIES FOR PARAMETER BOUNDS (example: keep rates positive, K > 0, efficiencies 0-1)
  nll += pow(CppAD::CondExpLt(e_fast, Type(0.0), e_fast, Type(0.0)), 2) * 1e2;
  nll += pow(CppAD::CondExpGt(e_fast, Type(1.0), e_fast-Type(1.0), Type(0.0)), 2) * 1e2;
  nll += pow(CppAD::CondExpLt(e_slow, Type(0.0), e_slow, Type(0.0)), 2) * 1e2;
  nll += pow(CppAD::CondExpGt(e_slow, Type(1.0), e_slow-Type(1.0), Type(0.0)), 2) * 1e2;

  // 9. REPORTING
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  // 10. EQUATION DESCRIPTIONS
  /*
    1. COTS population: 
       cots_pred(t) = cots_pred(t-1) + r_cots * cots_pred(t-1) * (1 - cots_pred(t-1)/K_cots) * (coral_total_prev/(coral_total_prev + 10)) * outbreak_trigger * env_mod
                      + e_fast * pred_fast + e_slow * pred_slow + immig
    2. Fast coral: 
       fast_pred(t) = fast_pred(t-1) + r_fast * fast_pred(t-1) * (1 - fast_pred(t-1)/K_fast) - pred_fast
    3. Slow coral: 
       slow_pred(t) = slow_pred(t-1) + r_slow * slow_pred(t-1) * (1 - slow_pred(t-1)/K_slow) - pred_slow
    4. Outbreak trigger: 
       outbreak_trigger = 1 / (1 + exp(-10 * ((fast_pred(t-1) - thresh_fast) + (slow_pred(t-1) - thresh_slow))))
    5. Predation (Holling II): 
       pred_fast = alpha_fast * cots_pred(t-1) * fast_pred(t-1) / (fast_pred(t-1) + 1)
       pred_slow = alpha_slow * cots_pred(t-1) * slow_pred(t-1) / (slow_pred(t-1) + 1)
    6. Environmental effect: 
       env_mod = exp(beta_sst * (sst_dat(t) - 27))
    7. Immigration: 
       immig = immig_scale * cotsimm_dat(t)
  */

  return nll;
}
