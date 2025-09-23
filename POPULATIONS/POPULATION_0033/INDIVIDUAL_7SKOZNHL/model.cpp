#include <TMB.hpp>

// Crown-of-thorns starfish outbreak model (Great Barrier Reef)
// Implements boom-bust predator-prey cycles with coral selective predation
// All _pred outputs correspond directly to _dat observations

template<class Type>
Type objective_function<Type>::operator() ()
{
  // -------------------------
  // DATA
  // -------------------------
  DATA_VECTOR(Year);                            // Time in years
  DATA_VECTOR(cots_dat);                        // Observed adult COTS density (ind/m2)
  DATA_VECTOR(fast_dat);                        // Observed fast coral (Acropora) cover (%)
  DATA_VECTOR(slow_dat);                        // Observed slow coral (Porites, Faviidae) cover (%)
  DATA_VECTOR(sst_dat);                         // Observed sea-surface temperature (C)
  DATA_VECTOR(cotsimm_dat);                     // Observed larval immigration (ind/m2/yr)

  // -------------------------
  // PARAMETERS
  // -------------------------
  PARAMETER(r_cots);        // Intrinsic adult growth rate (yr^-1), drives local COTS recruitment
  PARAMETER(K_cots);        // Carrying capacity of COTS population (ind/m2), max density limited by food
  PARAMETER(m_cots);        // Natural mortality rate of adult COTS (yr^-1)
  PARAMETER(alpha_fast);    // Attack rate of COTS on fast-growing corals (per capita rate)
  PARAMETER(alpha_slow);    // Attack rate of COTS on slow-growing corals (per capita rate)
  PARAMETER(beta_fast);     // Growth rate of fast corals (% cover per yr)
  PARAMETER(beta_slow);     // Growth rate of slow corals (% cover per yr)
  PARAMETER(K_fast);        // Carrying capacity for fast corals (% cover)
  PARAMETER(K_slow);        // Carrying capacity for slow corals (% cover)
  PARAMETER(gamma_sst);     // Temperature sensitivity scaling that modifies COTS recruitment
  PARAMETER(sigma_cots);    // Observation error SD for COTS log scale
  PARAMETER(sigma_fast);    // Observation error SD for fast coral
  PARAMETER(sigma_slow);    // Observation error SD for slow coral

  // -------------------------
  // INITIAL CONDITIONS
  // -------------------------
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Initialize with first observed values
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // -------------------------
  // PROCESS MODEL
  // -------------------------
  for(int t=1; t<n; t++){
    // Previous state values
    Type cots_prev = cots_pred(t-1);
    Type fast_prev = fast_pred(t-1);
    Type slow_prev = slow_pred(t-1);

    // Coral availability as food resource
    Type total_coral = fast_prev + slow_prev + Type(1e-8);

    // COTS recruitment: logistic growth scaled by coral and modified by SST + immigration
    Type env_effect = exp(gamma_sst * (sst_dat(t-1) - Type(27.0))); // Smooth temp effect relative to optimal ~27°C
    Type recruitment = r_cots * cots_prev * (1 - cots_prev/K_cots) * (total_coral / (total_coral + 10.0)) * env_effect;
    Type immigration = cotsimm_dat(t-1);

    // COTS mortality
    Type mortality = m_cots * cots_prev;

    // Update COTS population
    cots_pred(t) = cots_prev + recruitment + immigration - mortality;
    cots_pred(t) = CppAD::CondExpLt(cots_pred(t), Type(1e-8), Type(1e-8), cots_pred(t)); // prevent negatives

    // Predation functional responses (Holling type II for stability)
    Type pred_fast = alpha_fast * cots_prev * fast_prev / (1.0 + alpha_fast * fast_prev);
    Type pred_slow = alpha_slow * cots_prev * slow_prev / (1.0 + alpha_slow * slow_prev);

    // Coral dynamics
    fast_pred(t) = fast_prev + beta_fast * fast_prev * (1 - fast_prev/K_fast) - pred_fast;
    slow_pred(t) = slow_prev + beta_slow * slow_prev * (1 - slow_prev/K_slow) - pred_slow;

    // Ensure non-negative coral cover
    fast_pred(t) = CppAD::CondExpLt(fast_pred(t), Type(1e-8), Type(1e-8), fast_pred(t));
    slow_pred(t) = CppAD::CondExpLt(slow_pred(t), Type(1e-8), Type(1e-8), slow_pred(t));
  }

  // -------------------------
  // LIKELIHOOD
  // -------------------------
  Type nll = 0.0;
  for(int t=0; t<n; t++){
    // Lognormal likelihood for COTS (strictly positive)
    nll -= dnorm(log(cots_dat(t) + 1e-8), log(cots_pred(t) + 1e-8), sigma_cots, true);
    // Normal likelihood for coral cover
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_fast, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_slow, true);
  }

  // -------------------------
  // REPORTING
  // -------------------------
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}

/*
Equations summary:
1. COTS(t) = COTS(t-1) + r*COTS*(1 - COTS/K)*f(coral)*f(SST) + immigration - m*COTS
2. Coral_fast(t) = Coral_fast(t-1) + beta_fast*logistic_growth - predation_by_COTS
3. Coral_slow(t) = Coral_slow(t-1) + beta_slow*logistic_growth - predation_by_COTS
4. Predation = Holling type II functional response by COTS on corals
5. Likelihood combines lognormal for starfish and Gaussian for coral covers
*/
