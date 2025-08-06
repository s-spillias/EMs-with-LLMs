#include <TMB.hpp>

// 1. COTS = Crown-of-Thorns starfish (Acanthaster spp.)
// 2. fast = Fast-growing coral (Acropora spp.)
// 3. slow = Slow-growing coral (Faviidae spp. and Porites spp.)

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA INPUTS ---
  DATA_VECTOR(Year); // Time (years)
  DATA_VECTOR(sst_dat); // Sea-surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (indiv/m2/year)
  DATA_VECTOR(cots_dat); // Observed adult COTS abundance (indiv/m2)
  DATA_VECTOR(fast_dat); // Observed fast coral cover (%)
  DATA_VECTOR(slow_dat); // Observed slow coral cover (%)

  int n = Year.size();

  // --- PARAMETERS ---
  PARAMETER(log_r_cots); // log intrinsic growth rate of COTS (year^-1)
  PARAMETER(log_K_cots); // log carrying capacity for COTS (indiv/m2)
  PARAMETER(log_alpha_pred); // log predation rate of COTS on coral (m2/indiv/year)
  PARAMETER(log_beta_fast); // log preference/efficiency for fast coral
  PARAMETER(log_beta_slow); // log preference/efficiency for slow coral
  PARAMETER(log_m_cots); // log natural mortality rate of COTS (year^-1)
  PARAMETER(log_gamma_imm); // log scaling for larval immigration
  PARAMETER(logit_theta_sst); // logit temperature sensitivity (0-1)
  PARAMETER(log_r_fast); // log regrowth rate of fast coral (%/year)
  PARAMETER(log_r_slow); // log regrowth rate of slow coral (%/year)
  PARAMETER(log_m_fast); // log mortality rate of fast coral (%/year)
  PARAMETER(log_m_slow); // log mortality rate of slow coral (%/year)
  PARAMETER(log_sigma_cots); // log SD for COTS obs (lognormal)
  PARAMETER(log_sigma_fast); // log SD for fast coral obs (lognormal)
  PARAMETER(log_sigma_slow); // log SD for slow coral obs (lognormal)

  // --- TRANSFORM PARAMETERS ---
  Type r_cots = exp(log_r_cots); // COTS intrinsic growth rate (year^-1)
  Type K_cots = exp(log_K_cots); // COTS carrying capacity (indiv/m2)
  Type alpha_pred = exp(log_alpha_pred); // COTS predation rate (m2/indiv/year)
  Type beta_fast = exp(log_beta_fast); // Preference/efficiency for fast coral
  Type beta_slow = exp(log_beta_slow); // Preference/efficiency for slow coral
  Type m_cots = exp(log_m_cots); // COTS natural mortality (year^-1)
  Type gamma_imm = exp(log_gamma_imm); // Scaling for larval immigration
  Type theta_sst = Type(1)/(Type(1)+exp(-logit_theta_sst)); // SST effect (0-1)
  Type r_fast = exp(log_r_fast); // Fast coral regrowth (%/year)
  Type r_slow = exp(log_r_slow); // Slow coral regrowth (%/year)
  Type m_fast = exp(log_m_fast); // Fast coral mortality (%/year)
  Type m_slow = exp(log_m_slow); // Slow coral mortality (%/year)
  Type sigma_cots = exp(log_sigma_cots); // SD for COTS obs
  Type sigma_fast = exp(log_sigma_fast); // SD for fast coral obs
  Type sigma_slow = exp(log_sigma_slow); // SD for slow coral obs

  // --- INITIAL STATES ---
  vector<Type> cots_pred(n); // Predicted COTS abundance (indiv/m2)
  vector<Type> fast_pred(n); // Predicted fast coral cover (%)
  vector<Type> slow_pred(n); // Predicted slow coral cover (%)

  // Defensive: check data length
  // Removed error on zero-length input to allow TMB to compile and link the model

  // Set initial conditions from first observation
  cots_pred(0) = CppAD::CondExpGt(cots_dat(0), Type(1e-12), cots_dat(0), Type(1e-12)); // indiv/m2
  fast_pred(0) = CppAD::CondExpGt(fast_dat(0), Type(1e-12), fast_dat(0), Type(1e-12)); // %
  slow_pred(0) = CppAD::CondExpGt(slow_dat(0), Type(1e-12), slow_dat(0), Type(1e-12)); // %

  // --- MODEL DYNAMICS ---
  for(int t=1; t<n; t++){
    // Defensive: check for NaN/Inf in previous state
    if(!R_finite(asDouble(fast_pred(t-1))) ||
       !R_finite(asDouble(slow_pred(t-1))) ||
       !R_finite(asDouble(cots_pred(t-1)))) {
      error("NaN or Inf detected in state at t-1 (t=%d)", t);
    }

    // 1. Resource limitation for COTS (saturating function)
    Type fast_prev = CppAD::CondExpGt(fast_pred(t-1), Type(1e-12), fast_pred(t-1), Type(1e-12));
    Type slow_prev = CppAD::CondExpGt(slow_pred(t-1), Type(1e-12), slow_pred(t-1), Type(1e-12));
    Type cots_prev = CppAD::CondExpGt(cots_pred(t-1), Type(1e-12), cots_pred(t-1), Type(1e-12));
    Type coral_avail = beta_fast * fast_prev + beta_slow * slow_prev + Type(1e-8); // weighted coral cover

    // 2. SST effect on COTS growth (smooth, 0-1 scaling)
    Type sst_effect = exp(-pow(sst_dat(t-1)-28.0,2)/Type(2.0)); // peak at 28C, smooth Gaussian

    // 3. COTS recruitment: density-dependent + immigration + SST
    // Prevent division by zero in K_cots
    Type safe_K_cots = K_cots + Type(1e-8);

    Type recruit = r_cots * cots_prev * (Type(1) - cots_prev/safe_K_cots) * coral_avail/(coral_avail+Type(10.0)) * sst_effect * theta_sst
                   + gamma_imm * cotsimm_dat(t-1); // indiv/m2/year

    // 4. COTS mortality (natural + resource limitation)
    Type mort = m_cots * cots_prev + alpha_pred * cots_prev * (Type(1) - coral_avail/(coral_avail+Type(10.0))); // indiv/m2/year

    // 5. Update COTS abundance (ensure non-negative)
    cots_pred(t) = CppAD::CondExpGt(recruit + cots_prev - mort, Type(0), recruit + cots_prev - mort, Type(1e-8));

    // 6. COTS predation on coral (Type II functional response)
    Type pred_fast = alpha_pred * cots_prev * (beta_fast * fast_prev)/(coral_avail+Type(1e-8));
    Type pred_slow = alpha_pred * cots_prev * (beta_slow * slow_prev)/(coral_avail+Type(1e-8));

    // 7. Coral regrowth (logistic, resource-limited)
    Type reg_fast = r_fast * fast_prev * (Type(100.0) - fast_prev)/Type(100.0);
    Type reg_slow = r_slow * slow_prev * (Type(100.0) - slow_prev)/Type(100.0);

    // 8. Coral mortality (background + COTS predation)
    Type mort_fast = m_fast * fast_prev + pred_fast;
    Type mort_slow = m_slow * slow_prev + pred_slow;

    // 9. Update coral cover (ensure non-negative)
    fast_pred(t) = CppAD::CondExpGt(fast_prev + reg_fast - mort_fast, Type(0), fast_prev + reg_fast - mort_fast, Type(1e-8));
    slow_pred(t) = CppAD::CondExpGt(slow_prev + reg_slow - mort_slow, Type(0), slow_prev + reg_slow - mort_slow, Type(1e-8));
  }

  // --- LIKELIHOOD ---
  Type nll = 0.0;
  Type min_sd = Type(1e-3); // minimum SD for stability

  // 10. Likelihood for COTS (lognormal, strictly positive)
  for(int t=0; t<n; t++){
    // Defensive: check for negative or zero predictions before log()
    Type cots_pred_safe = CppAD::CondExpGt(cots_pred(t), Type(1e-12), cots_pred(t), Type(1e-12));
    Type fast_pred_safe = CppAD::CondExpGt(fast_pred(t), Type(1e-12), fast_pred(t), Type(1e-12));
    Type slow_pred_safe = CppAD::CondExpGt(slow_pred(t), Type(1e-12), slow_pred(t), Type(1e-12));

    // Defensive: check for NaN/Inf in predictions
    if(!R_finite(asDouble(cots_pred_safe)) ||
       !R_finite(asDouble(fast_pred_safe)) ||
       !R_finite(asDouble(slow_pred_safe))) {
      error("NaN or Inf detected in predictions at t=%d", t);
    }

    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred_safe + Type(1e-8)), sigma_cots + min_sd, true);
    nll -= dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred_safe + Type(1e-8)), sigma_fast + min_sd, true);
    nll -= dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred_safe + Type(1e-8)), sigma_slow + min_sd, true);
  }

  // --- SMOOTH PENALTIES FOR PARAMETER BOUNDS ---
  // Example: penalize if r_cots, K_cots, alpha_pred, etc. are outside plausible ranges
  nll += pow(CppAD::CondExpLt(r_cots, Type(0.01), r_cots-Type(0.01), Type(0)), 2) * Type(1e2);
  nll += pow(CppAD::CondExpGt(r_cots, Type(5.0), r_cots-Type(5.0), Type(0)), 2) * Type(1e2);
  nll += pow(CppAD::CondExpLt(K_cots, Type(0.01), K_cots-Type(0.01), Type(0)), 2) * Type(1e2);
  nll += pow(CppAD::CondExpGt(K_cots, Type(10.0), K_cots-Type(10.0), Type(0)), 2) * Type(1e2);

  // --- REPORTING ---
  REPORT(cots_pred); // Predicted COTS abundance (indiv/m2)
  REPORT(fast_pred); // Predicted fast coral cover (%)
  REPORT(slow_pred); // Predicted slow coral cover (%)

  return nll;
}

/*
Equation descriptions:
1. coral_avail = beta_fast * fast_pred + beta_slow * slow_pred + 1e-8
   (Weighted coral cover available to COTS, prevents division by zero)
2. sst_effect = exp(-((sst-28)^2)/2)
   (Gaussian effect of SST on COTS growth, peak at 28C)
3. recruit = r_cots * COTS * (1 - COTS/K) * coral_avail/(coral_avail+10) * sst_effect * theta_sst + gamma_imm * cotsimm
   (COTS recruitment, density-dependent, resource-limited, SST-modified, plus immigration)
4. mort = m_cots * COTS + alpha_pred * COTS * (1 - coral_avail/(coral_avail+10))
   (COTS mortality, natural plus starvation/resource limitation)
5. cots_pred(t) = max(recruit + cots_pred(t-1) - mort, 1e-8)
   (Update COTS abundance, non-negative)
6. pred_fast = alpha_pred * COTS * (beta_fast * fast_pred)/(coral_avail+1e-8)
   (COTS predation on fast coral, Type II functional response)
7. pred_slow = alpha_pred * COTS * (beta_slow * slow_pred)/(coral_avail+1e-8)
   (COTS predation on slow coral, Type II functional response)
8. reg_fast = r_fast * fast_pred * (100-fast_pred)/100
   (Fast coral regrowth, logistic)
9. reg_slow = r_slow * slow_pred * (100-slow_pred)/100
   (Slow coral regrowth, logistic)
10. mort_fast = m_fast * fast_pred + pred_fast
    (Fast coral mortality, background + COTS predation)
11. mort_slow = m_slow * slow_pred + pred_slow
    (Slow coral mortality, background + COTS predation)
12. fast_pred(t) = max(fast_pred(t-1) + reg_fast - mort_fast, 1e-8)
    (Update fast coral cover, non-negative)
13. slow_pred(t) = max(slow_pred(t-1) + reg_slow - mort_slow, 1e-8)
    (Update slow coral cover, non-negative)
*/
