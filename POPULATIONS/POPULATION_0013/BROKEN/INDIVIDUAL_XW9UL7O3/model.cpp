#include <TMB.hpp>

// 1. COTS = Crown-of-Thorns starfish (individuals/m2)
// 2. fast = Fast-growing coral (Acropora spp.) cover (%)
// 3. slow = Slow-growing coral (Faviidae/Porites) cover (%)
// 4. sst = Sea-surface temperature (deg C)
// 5. cotsimm = COTS larval immigration (individuals/m2/year)
// 6. All _dat are observed data, _pred are model predictions

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA BLOCK ---
  DATA_VECTOR(Year); // Observation years
  DATA_VECTOR(cots_dat); // COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat); // Fast coral cover (%)
  DATA_VECTOR(slow_dat); // Slow coral cover (%)
  DATA_VECTOR(sst_dat); // Sea-surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (ind/m2/yr)

  int n = Year.size();

  // --- PARAMETER BLOCK ---
  PARAMETER(log_r_cots); // log intrinsic COTS growth rate (year^-1)
  // Determines how quickly COTS can increase in absence of limitation

  PARAMETER(log_K_cots); // log COTS carrying capacity (ind/m2)
  // Maximum sustainable COTS density

  PARAMETER(log_alpha_fast); // log COTS attack rate on fast coral (m2/%/yr)
  // COTS predation rate on Acropora

  PARAMETER(log_alpha_slow); // log COTS attack rate on slow coral (m2/%/yr)
  // COTS predation rate on Faviidae/Porites

  PARAMETER(log_e_fast); // log assimilation efficiency from fast coral (unitless)
  // Fraction of consumed fast coral converted to COTS biomass

  PARAMETER(log_e_slow); // log assimilation efficiency from slow coral (unitless)
  // Fraction of consumed slow coral converted to COTS biomass

  PARAMETER(log_r_fast); // log intrinsic growth rate of fast coral (year^-1)
  // Maximum per-year increase in fast coral cover

  PARAMETER(log_r_slow); // log intrinsic growth rate of slow coral (year^-1)
  // Maximum per-year increase in slow coral cover

  PARAMETER(log_K_fast); // log carrying capacity of fast coral (% cover)
  // Maximum possible % cover for fast coral

  PARAMETER(log_K_slow); // log carrying capacity of slow coral (% cover)
  // Maximum possible % cover for slow coral

  PARAMETER(beta_sst); // Effect of SST on COTS growth (per deg C)
  // Modifies COTS growth rate with temperature

  PARAMETER(log_sigma_cots); // log SD of COTS obs error (lognormal)
  PARAMETER(log_sigma_fast); // log SD of fast coral obs error (lognormal)
  PARAMETER(log_sigma_slow); // log SD of slow coral obs error (lognormal)

  // --- TRANSFORM PARAMETERS ---
  Type r_cots = exp(log_r_cots); // COTS intrinsic growth rate
  Type K_cots = exp(log_K_cots); // COTS carrying capacity
  Type alpha_fast = exp(log_alpha_fast); // COTS attack rate (fast coral)
  Type alpha_slow = exp(log_alpha_slow); // COTS attack rate (slow coral)
  Type e_fast = exp(log_e_fast); // COTS assimilation efficiency (fast coral)
  Type e_slow = exp(log_e_slow); // COTS assimilation efficiency (slow coral)
  Type r_fast = exp(log_r_fast); // Fast coral growth rate
  Type r_slow = exp(log_r_slow); // Slow coral growth rate
  Type K_fast = exp(log_K_fast); // Fast coral carrying capacity
  Type K_slow = exp(log_K_slow); // Slow coral carrying capacity

  Type sigma_cots = exp(log_sigma_cots) + Type(1e-8); // COTS obs SD
  Type sigma_fast = exp(log_sigma_fast) + Type(1e-8); // Fast coral obs SD
  Type sigma_slow = exp(log_sigma_slow) + Type(1e-8); // Slow coral obs SD

  // --- INITIAL CONDITIONS ---
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  cots_pred(0) = (cots_dat(0) > Type(1e-8)) ? cots_dat(0) : Type(1e-8); // Initial COTS abundance (avoid zero/negative)
  fast_pred(0) = (fast_dat(0) > Type(1e-8)) ? fast_dat(0) : Type(1e-8); // Initial fast coral cover (avoid zero/negative)
  slow_pred(0) = (slow_dat(0) > Type(1e-8)) ? slow_dat(0) : Type(1e-8); // Initial slow coral cover (avoid zero/negative)

  // --- PROCESS MODEL ---
  for(int t=1; t<n; t++) {
    // Defensive: ensure previous values are strictly positive (avoid NaN in log, division)
    Type prev_cots = CppAD::CondExpGt(cots_pred(t-1), Type(1e-8), cots_pred(t-1), Type(1e-8));
    Type prev_fast = CppAD::CondExpGt(fast_pred(t-1), Type(1e-8), fast_pred(t-1), Type(1e-8));
    Type prev_slow = CppAD::CondExpGt(slow_pred(t-1), Type(1e-8), slow_pred(t-1), Type(1e-8));

    // 1. COTS predation on corals (Holling Type II functional response)
    Type pred_fast = alpha_fast * prev_cots * prev_fast / (prev_fast + Type(1.0) + Type(1e-8)); // predation on fast coral
    Type pred_slow = alpha_slow * prev_cots * prev_slow / (prev_slow + Type(1.0) + Type(1e-8)); // predation on slow coral

    // 2. COTS population dynamics (logistic + resource + SST + immigration)
    Type resource_term = (e_fast * pred_fast + e_slow * pred_slow); // resource assimilation
    Type sst_effect = exp(beta_sst * (sst_dat(t-1) - Type(27.0))); // SST modifies growth (27C baseline)
    Type density_term = (Type(1.0) - prev_cots/K_cots); // logistic limitation

    // Outbreak threshold: smooth sigmoid on resource assimilation
    Type outbreak_trigger = Type(1.0) / (Type(1.0) + exp(-10.0 * (resource_term - Type(0.05)))); // triggers when resource_term > 0.05

    Type cots_growth = r_cots * prev_cots * density_term * sst_effect * outbreak_trigger; // growth
    Type cots_update = prev_cots + cots_growth + resource_term + cotsimm_dat(t-1); // update

    // Prevent negative or zero COTS
    cots_pred(t) = CppAD::CondExpGt(cots_update, Type(1e-8), cots_update, Type(1e-8));

    // 3. Fast coral dynamics (logistic growth - COTS predation)
    Type fast_growth = r_fast * prev_fast * (Type(1.0) - prev_fast/K_fast);
    Type fast_update = prev_fast + fast_growth - pred_fast;

    // Prevent negative or zero coral cover
    fast_pred(t) = CppAD::CondExpGt(fast_update, Type(1e-8), fast_update, Type(1e-8));

    // 4. Slow coral dynamics (logistic growth - COTS predation)
    Type slow_growth = r_slow * prev_slow * (Type(1.0) - prev_slow/K_slow);
    Type slow_update = prev_slow + slow_growth - pred_slow;

    slow_pred(t) = CppAD::CondExpGt(slow_update, Type(1e-8), slow_update, Type(1e-8));
  }

  // --- LIKELIHOOD (lognormal, all obs included, fixed min SD) ---
  Type nll = 0.0;
  for(int t=0; t<n; t++) {
    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t) + Type(1e-8)), sigma_cots, true);
    nll -= dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred(t) + Type(1e-8)), sigma_fast, true);
    nll -= dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred(t) + Type(1e-8)), sigma_slow, true);
  }

  // --- PENALTIES FOR BIOLOGICAL RANGES (soft, smooth) ---
  // Example: discourage negative or implausible parameter values
  nll += pow(CppAD::CondExpLt(r_cots, Type(0.01), r_cots-Type(0.01), Type(0.0)), 2);
  nll += pow(CppAD::CondExpGt(r_cots, Type(5.0), r_cots-Type(5.0), Type(0.0)), 2);
  nll += pow(CppAD::CondExpLt(K_cots, Type(0.01), K_cots-Type(0.01), Type(0.0)), 2);
  nll += pow(CppAD::CondExpGt(K_cots, Type(10.0), K_cots-Type(10.0), Type(0.0)), 2);

  // --- REPORTING ---
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  // --- EQUATION DESCRIPTIONS ---
  // 1. COTS predation on corals: Holling Type II functional response
  // 2. COTS population: logistic growth, resource assimilation, SST effect, outbreak trigger, immigration
  // 3. Coral groups: logistic growth minus COTS predation
  // 4. All transitions smooth, no hard cutoffs, all obs included in likelihood

  return nll;
}
