#include <TMB.hpp>

// Model for episodic outbreaks of Crown-of-Thorns starfish (COTS) and their impact on coral communities

template<class Type>
Type objective_function<Type>::operator() ()
{
  // 1. DATA INPUTS
  DATA_VECTOR(Year); // Year (time variable)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration rate (individuals/m2/year)
  DATA_VECTOR(sst_dat); // Sea-surface temperature (Celsius)
  DATA_VECTOR(cots_dat); // Observed adult COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat); // Observed fast-growing coral cover (Acropora spp.) (%)
  DATA_VECTOR(slow_dat); // Observed slow-growing coral cover (Faviidae/Porites spp.) (%)

  int n = Year.size();

  // 2. PARAMETERS

  // COTS population dynamics
  PARAMETER(log_r_cots); // log intrinsic growth rate of COTS (year^-1)
  PARAMETER(log_K_cots); // log carrying capacity of COTS (individuals/m2)
  PARAMETER(log_alpha_cots); // log predation rate on fast coral (m2/individual/year)
  PARAMETER(log_beta_cots); // log predation rate on slow coral (m2/individual/year)
  PARAMETER(log_m_cots); // log baseline mortality rate of COTS (year^-1)
  PARAMETER(log_imm_eff); // log efficiency of larval immigration (proportion)

  // Coral dynamics
  PARAMETER(log_r_fast); // log growth rate of fast coral (year^-1)
  PARAMETER(log_r_slow); // log growth rate of slow coral (year^-1)
  PARAMETER(log_K_coral); // log total coral carrying capacity (% cover)
  PARAMETER(log_m_fast); // log background mortality of fast coral (year^-1)
  PARAMETER(log_m_slow); // log background mortality of slow coral (year^-1)
  PARAMETER(log_rec_fast); // log recruitment rate of fast coral (%/year)
  PARAMETER(log_rec_slow); // log recruitment rate of slow coral (%/year)

  // Environmental effects
  PARAMETER(temp_cots); // effect of SST on COTS growth (per deg C)
  PARAMETER(temp_fast); // effect of SST on fast coral growth (per deg C)
  PARAMETER(temp_slow); // effect of SST on slow coral growth (per deg C)

  // Observation error (lognormal SDs)
  PARAMETER(log_sd_cots);
  PARAMETER(log_sd_fast);
  PARAMETER(log_sd_slow);

  // 3. TRANSFORM PARAMETERS TO NATURAL SCALE
  Type r_cots = exp(log_r_cots); // COTS intrinsic growth rate
  Type K_cots = exp(log_K_cots); // COTS carrying capacity
  Type alpha_cots = exp(log_alpha_cots); // COTS predation rate on fast coral
  Type beta_cots = exp(log_beta_cots); // COTS predation rate on slow coral
  Type m_cots = exp(log_m_cots); // COTS baseline mortality
  Type imm_eff = exp(log_imm_eff); // Immigration efficiency

  Type r_fast = exp(log_r_fast); // Fast coral growth rate
  Type r_slow = exp(log_r_slow); // Slow coral growth rate
  Type K_coral = exp(log_K_coral); // Total coral carrying capacity
  Type m_fast = exp(log_m_fast); // Fast coral background mortality
  Type m_slow = exp(log_m_slow); // Slow coral background mortality
  Type rec_fast = exp(log_rec_fast); // Fast coral recruitment
  Type rec_slow = exp(log_rec_slow); // Slow coral recruitment

  Type sd_cots = exp(log_sd_cots) + Type(1e-6); // COTS obs error SD
  Type sd_fast = exp(log_sd_fast) + Type(1e-6); // Fast coral obs error SD
  Type sd_slow = exp(log_sd_slow) + Type(1e-6); // Slow coral obs error SD

  // 4. INITIAL CONDITIONS (use first obs as initial state)
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Use positive minimums for initial conditions to avoid NaN in log-likelihood
  cots_pred(0) = (CppAD::Value(cots_dat(0)) <= Type(1e-8) || CppAD::isnan(cots_dat(0))) ? Type(1e-8) : cots_dat(0);
  fast_pred(0) = (CppAD::Value(fast_dat(0)) <= Type(1e-8) || CppAD::isnan(fast_dat(0))) ? Type(1e-8) : fast_dat(0);
  slow_pred(0) = (CppAD::Value(slow_dat(0)) <= Type(1e-8) || CppAD::isnan(slow_dat(0))) ? Type(1e-8) : slow_dat(0);

  // 5. PROCESS MODEL
  for(int t=1; t<n; t++) {

    // 1. COTS population growth (density-dependent, resource-limited, SST effect, immigration)
    //   - Logistic growth with resource limitation (total coral cover)
    //   - Immigration is episodic, modulated by efficiency
    //   - SST modifies growth rate
    Type coral_avail = fast_pred(t-1) + slow_pred(t-1) + Type(1e-8); // total coral cover (%)
    Type coral_lim = coral_avail / (K_coral + Type(1e-8)); // resource limitation (0-1)
    coral_lim = CppAD::CondExpLt(coral_lim, Type(1e-3), Type(1e-3), coral_lim); // smooth lower bound

    Type r_cots_eff = r_cots * (1 + temp_cots * (sst_dat(t-1) - Type(27.0))); // SST effect on COTS
    Type imm_input = imm_eff * cotsimm_dat(t-1); // effective immigration

    Type cots_growth = r_cots_eff * cots_pred(t-1) * (1 - cots_pred(t-1) / (K_cots * coral_lim + Type(1e-8)));
    Type cots_mortality = m_cots * cots_pred(t-1);
    cots_pred(t) = cots_pred(t-1) + cots_growth + imm_input - cots_mortality;
    cots_pred(t) = CppAD::CondExpLt(cots_pred(t), Type(1e-8), Type(1e-8), cots_pred(t)); // prevent negative

    // 2. Fast coral dynamics (Acropora spp.)
    //   - Logistic growth, background mortality, COTS predation (Type II functional response), SST effect
    Type fast_growth = r_fast * fast_pred(t-1) * (1 - (fast_pred(t-1) + slow_pred(t-1)) / (K_coral + Type(1e-8)));
    Type fast_rec = rec_fast * (1 - (fast_pred(t-1) + slow_pred(t-1)) / (K_coral + Type(1e-8)));
    Type fast_mortality = m_fast * fast_pred(t-1);
    Type fast_predation = alpha_cots * cots_pred(t-1) * fast_pred(t-1) / (fast_pred(t-1) + slow_pred(t-1) + Type(1e-8));
    Type fast_temp_eff = 1 + temp_fast * (sst_dat(t-1) - Type(27.0));
    fast_pred(t) = fast_pred(t-1) + (fast_growth + fast_rec - fast_mortality - fast_predation) * fast_temp_eff;
    fast_pred(t) = CppAD::CondExpLt(fast_pred(t), Type(1e-8), Type(1e-8), fast_pred(t)); // prevent negative

    // 3. Slow coral dynamics (Faviidae/Porites spp.)
    //   - Logistic growth, background mortality, COTS predation (Type II functional response), SST effect
    Type slow_growth = r_slow * slow_pred(t-1) * (1 - (fast_pred(t-1) + slow_pred(t-1)) / (K_coral + Type(1e-8)));
    Type slow_rec = rec_slow * (1 - (fast_pred(t-1) + slow_pred(t-1)) / (K_coral + Type(1e-8)));
    Type slow_mortality = m_slow * slow_pred(t-1);
    Type slow_predation = beta_cots * cots_pred(t-1) * slow_pred(t-1) / (fast_pred(t-1) + slow_pred(t-1) + Type(1e-8));
    Type slow_temp_eff = 1 + temp_slow * (sst_dat(t-1) - Type(27.0));
    slow_pred(t) = slow_pred(t-1) + (slow_growth + slow_rec - slow_mortality - slow_predation) * slow_temp_eff;
    slow_pred(t) = CppAD::CondExpLt(slow_pred(t), Type(1e-8), Type(1e-8), slow_pred(t)); // prevent negative
  }

  // 6. LIKELIHOOD (lognormal, fixed minimum SD)
  Type nll = 0.0;
  for(int t=0; t<n; t++) {
    // Log-transform for strictly positive data
    // Use positive minimums for observed data to avoid NaN in log()
    Type cots_obs = (CppAD::Value(cots_dat(t)) <= Type(1e-8) || CppAD::isnan(cots_dat(t))) ? Type(1e-8) : cots_dat(t);
    Type fast_obs = (CppAD::Value(fast_dat(t)) <= Type(1e-8) || CppAD::isnan(fast_dat(t))) ? Type(1e-8) : fast_dat(t);
    Type slow_obs = (CppAD::Value(slow_dat(t)) <= Type(1e-8) || CppAD::isnan(slow_dat(t))) ? Type(1e-8) : slow_dat(t);
    nll -= dnorm(log(cots_obs), log(cots_pred(t) + Type(1e-8)), sd_cots, true);
    nll -= dnorm(log(fast_obs), log(fast_pred(t) + Type(1e-8)), sd_fast, true);
    nll -= dnorm(log(slow_obs), log(slow_pred(t) + Type(1e-8)), sd_slow, true);
  }

  // 7. REPORTING
  REPORT(cots_pred); // predicted COTS abundance (individuals/m2)
  REPORT(fast_pred); // predicted fast coral cover (%)
  REPORT(slow_pred); // predicted slow coral cover (%)

  // 8. EQUATION DESCRIPTIONS
  /*
    1. COTS: Logistic growth with resource limitation (total coral), SST effect, episodic immigration, and mortality.
    2. Fast coral: Logistic growth, recruitment, background mortality, COTS predation (Type II), SST effect.
    3. Slow coral: Logistic growth, recruitment, background mortality, COTS predation (Type II), SST effect.
    4. All transitions are smooth and numerically stable.
    5. Likelihood: Lognormal error for all observed variables, fixed minimum SD.
  */

  return nll;
}
