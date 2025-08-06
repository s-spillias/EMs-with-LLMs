#include <TMB.hpp>

// 1. Model equations are described at the end of this file.

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA INPUTS ---
  DATA_VECTOR(Year); // Time variable (years)
  DATA_VECTOR(cots_dat); // Adult COTS abundance (indiv/m2)
  DATA_VECTOR(fast_dat); // Fast-growing coral cover (%)
  DATA_VECTOR(slow_dat); // Slow-growing coral cover (%)
  DATA_VECTOR(sst_dat); // Sea-surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (indiv/m2/year)

  int n = Year.size();

  // --- PARAMETERS ---
  PARAMETER(log_r_cots); // log intrinsic COTS recruitment rate (log(year^-1))
  PARAMETER(log_K_cots); // log COTS carrying capacity (log(indiv/m2))
  PARAMETER(log_alpha_cots); // log COTS coral predation efficiency (log(m2/indiv/year))
  PARAMETER(log_beta_cots); // log half-saturation coral cover for COTS predation (log(%))
  PARAMETER(log_m_cots); // log COTS natural mortality rate (log(year^-1))
  PARAMETER(log_phi_cots); // log immigration efficiency (log(unitless))
  PARAMETER(log_r_fast); // log fast coral intrinsic growth rate (log(%/year))
  PARAMETER(log_K_fast); // log fast coral carrying capacity (log(% cover))
  PARAMETER(log_r_slow); // log slow coral intrinsic growth rate (log(%/year))
  PARAMETER(log_K_slow); // log slow coral carrying capacity (log(% cover))
  PARAMETER(log_gamma_fast); // log COTS predation rate on fast coral (log(%/indiv/year))
  PARAMETER(log_gamma_slow); // log COTS predation rate on slow coral (log(%/indiv/year))
  PARAMETER(log_env_cots); // log environmental effect on COTS recruitment (log(unitless))
  PARAMETER(log_env_fast); // log environmental effect on fast coral growth (log(unitless))
  PARAMETER(log_env_slow); // log environmental effect on slow coral growth (log(unitless))
  PARAMETER(log_sigma_cots); // log obs SD for COTS (log(indiv/m2))
  PARAMETER(log_sigma_fast); // log obs SD for fast coral (log(%))
  PARAMETER(log_sigma_slow); // log obs SD for slow coral (log(%))

  // --- TRANSFORM PARAMETERS ---
  Type r_cots = exp(log_r_cots); // COTS recruitment rate (year^-1)
  Type K_cots = exp(log_K_cots); // COTS carrying capacity (indiv/m2)
  Type alpha_cots = exp(log_alpha_cots); // COTS coral predation efficiency (m2/indiv/year)
  Type beta_cots = exp(log_beta_cots); // Half-saturation coral cover for COTS predation (%)
  Type m_cots = exp(log_m_cots); // COTS mortality rate (year^-1)
  Type phi_cots = exp(log_phi_cots); // Immigration efficiency (unitless)
  Type r_fast = exp(log_r_fast); // Fast coral growth rate (%/year)
  Type K_fast = exp(log_K_fast); // Fast coral carrying capacity (%)
  Type r_slow = exp(log_r_slow); // Slow coral growth rate (%/year)
  Type K_slow = exp(log_K_slow); // Slow coral carrying capacity (%)
  Type gamma_fast = exp(log_gamma_fast); // COTS predation rate on fast coral (%/indiv/year)
  Type gamma_slow = exp(log_gamma_slow); // COTS predation rate on slow coral (%/indiv/year)
  Type env_cots = exp(log_env_cots); // Environmental effect on COTS recruitment (unitless)
  Type env_fast = exp(log_env_fast); // Environmental effect on fast coral growth (unitless)
  Type env_slow = exp(log_env_slow); // Environmental effect on slow coral growth (unitless)
  Type sigma_cots = exp(log_sigma_cots) + Type(1e-8); // Obs SD for COTS (indiv/m2)
  Type sigma_fast = exp(log_sigma_fast) + Type(1e-8); // Obs SD for fast coral (%)
  Type sigma_slow = exp(log_sigma_slow) + Type(1e-8); // Obs SD for slow coral (%)

  // --- INITIAL CONDITIONS ---
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  cots_pred.setZero();
  fast_pred.setZero();
  slow_pred.setZero();

  if(n > 0) {
    cots_pred(0) = CppAD::CondExpGt(cots_dat(0), Type(0), cots_dat(0), Type(1e-8)); // Initial COTS abundance (indiv/m2)
    fast_pred(0) = CppAD::CondExpGt(fast_dat(0), Type(0), fast_dat(0), Type(1e-8)); // Initial fast coral cover (%)
    slow_pred(0) = CppAD::CondExpGt(slow_dat(0), Type(0), slow_dat(0), Type(1e-8)); // Initial slow coral cover (%)
  }

  // --- PROCESS MODEL ---
  for(int t=1; t<n; t++) {
    // 1. Total coral cover at previous step
    Type coral_prev = fast_pred(t-1) + slow_pred(t-1) + Type(1e-8); // (% cover, avoid zero)

    // 2. COTS recruitment (density-dependent, resource-limited, environmental effect, immigration)
    Type recruit_cots = r_cots * cots_pred(t-1) * (coral_prev/(coral_prev + beta_cots + Type(1e-8))) * env_cots * exp(sst_dat(t-1)-Type(27.0)); // (indiv/m2/year)
    Type immigrate_cots = phi_cots * cotsimm_dat(t-1); // (indiv/m2/year)
    Type cots_growth = recruit_cots + immigrate_cots;

    // 3. COTS mortality (density-dependent, saturating)
    Type mort_cots = m_cots * cots_pred(t-1) * (K_cots/(K_cots + cots_pred(t-1) + Type(1e-8))); // (indiv/m2/year)

    // 4. Update COTS abundance
    cots_pred(t) = cots_pred(t-1) + cots_growth - mort_cots;
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), Type(0), cots_pred(t), Type(1e-8)); // Bound to >=0

    // 5. COTS predation on corals (functional response, saturating)
    Type pred_fast = gamma_fast * cots_pred(t-1) * (fast_pred(t-1)/(fast_pred(t-1) + beta_cots + Type(1e-8))); // (%/year)
    Type pred_slow = gamma_slow * cots_pred(t-1) * (slow_pred(t-1)/(slow_pred(t-1) + beta_cots + Type(1e-8))); // (%/year)

    // 6. Fast coral growth (logistic, environmental effect)
    Type grow_fast = r_fast * fast_pred(t-1) * (1.0 - (fast_pred(t-1) + slow_pred(t-1))/(K_fast + Type(1e-8))) * env_fast * exp(-alpha_cots * cots_pred(t-1));
    // 7. Slow coral growth (logistic, environmental effect)
    Type grow_slow = r_slow * slow_pred(t-1) * (1.0 - (fast_pred(t-1) + slow_pred(t-1))/(K_slow + Type(1e-8))) * env_slow * exp(-alpha_cots * cots_pred(t-1));

    // 8. Update coral covers
    fast_pred(t) = fast_pred(t-1) + grow_fast - pred_fast;
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), Type(0), fast_pred(t), Type(1e-8)); // Bound to >=0

    slow_pred(t) = slow_pred(t-1) + grow_slow - pred_slow;
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), Type(0), slow_pred(t), Type(1e-8)); // Bound to >=0
  }

  // --- LIKELIHOOD ---
  Type nll = Type(0);

  // 1. COTS abundance (lognormal likelihood)
  for(int t=0; t<n; t++) {
    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t) + Type(1e-8)), sigma_cots, true);
  }
  // 2. Fast coral cover (lognormal likelihood)
  for(int t=0; t<n; t++) {
    nll -= dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred(t) + Type(1e-8)), sigma_fast, true);
  }
  // 3. Slow coral cover (lognormal likelihood)
  for(int t=0; t<n; t++) {
    nll -= dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred(t) + Type(1e-8)), sigma_slow, true);
  }

  // --- REPORTING ---
  REPORT(cots_pred); // Predicted COTS abundance (indiv/m2)
  REPORT(fast_pred); // Predicted fast coral cover (%)
  REPORT(slow_pred); // Predicted slow coral cover (%)

  return nll;
}

/*
MODEL EQUATION DESCRIPTIONS:
1. coral_prev = fast_pred(t-1) + slow_pred(t-1) + 1e-8
   (Total coral cover at previous time step, %)
2. recruit_cots = r_cots * cots_pred(t-1) * (coral_prev/(coral_prev + beta_cots)) * env_cots * exp(sst_dat(t-1)-27.0)
   (COTS recruitment, density- and resource-dependent, environmental effect)
3. immigrate_cots = phi_cots * cotsimm_dat(t-1)
   (COTS larval immigration, scaled by efficiency)
4. cots_growth = recruit_cots + immigrate_cots
   (Total COTS population increase)
5. mort_cots = m_cots * cots_pred(t-1) * (K_cots/(K_cots + cots_pred(t-1)))
   (COTS mortality, density-dependent, saturating)
6. cots_pred(t) = cots_pred(t-1) + cots_growth - mort_cots
   (Update COTS abundance)
7. pred_fast = gamma_fast * cots_pred(t-1) * (fast_pred(t-1)/(fast_pred(t-1) + beta_cots + 1e-8))
   (COTS predation on fast coral, saturating functional response)
8. pred_slow = gamma_slow * cots_pred(t-1) * (slow_pred(t-1)/(slow_pred(t-1) + beta_cots + 1e-8))
   (COTS predation on slow coral, saturating functional response)
9. grow_fast = r_fast * fast_pred(t-1) * (1 - (fast_pred(t-1) + slow_pred(t-1))/K_fast) * env_fast * exp(-alpha_cots * cots_pred(t-1))
   (Fast coral growth, logistic, environmental and COTS suppression)
10. grow_slow = r_slow * slow_pred(t-1) * (1 - (fast_pred(t-1) + slow_pred(t-1))/K_slow) * env_slow * exp(-alpha_cots * cots_pred(t-1))
   (Slow coral growth, logistic, environmental and COTS suppression)
11. fast_pred(t) = fast_pred(t-1) + grow_fast - pred_fast
    (Update fast coral cover)
12. slow_pred(t) = slow_pred(t-1) + grow_slow - pred_slow
    (Update slow coral cover)
*/
