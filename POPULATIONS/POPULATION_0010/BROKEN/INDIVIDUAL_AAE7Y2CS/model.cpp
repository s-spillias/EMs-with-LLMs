#include <TMB.hpp>

// 1. Model equations (see below for numbered descriptions):
//    1. COTS population: boom-bust cycles via density dependence, resource limitation, and episodic larval input
//    2. Coral dynamics: selective predation by COTS, differential growth rates, and resource feedbacks
//    3. Environmental modulation: temperature effects on rates
//    4. Likelihood: lognormal errors, minimum SD, all observations included

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA INPUTS ---
  DATA_VECTOR(Year); // Year (time, in years)
  DATA_VECTOR(sst_dat); // Sea-surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (indiv/m2/yr)
  DATA_VECTOR(cots_dat); // Observed adult COTS abundance (indiv/m2)
  DATA_VECTOR(fast_dat); // Observed fast coral cover (%)
  DATA_VECTOR(slow_dat); // Observed slow coral cover (%)

  int n = Year.size();

  // --- PARAMETERS ---
  PARAMETER(log_r_cots); // log intrinsic COTS growth rate (log(yr^-1)), to be exponentiated
  PARAMETER(log_K_cots); // log COTS carrying capacity (log(indiv/m2)), to be exponentiated
  PARAMETER(log_alpha_fast); // log attack rate on fast coral (log(m2/indiv/yr)), to be exponentiated
  PARAMETER(log_alpha_slow); // log attack rate on slow coral (log(m2/indiv/yr)), to be exponentiated
  PARAMETER(log_h_fast); // log half-saturation for fast coral (log(%)), to be exponentiated
  PARAMETER(log_h_slow); // log half-saturation for slow coral (log(%)), to be exponentiated
  PARAMETER(log_r_fast); // log fast coral growth rate (log(%/yr)), to be exponentiated
  PARAMETER(log_r_slow); // log slow coral growth rate (log(%/yr)), to be exponentiated
  PARAMETER(log_K_fast); // log fast coral carrying capacity (log(%)), to be exponentiated
  PARAMETER(log_K_slow); // log slow coral carrying capacity (log(%)), to be exponentiated
  PARAMETER(log_m_cots); // log COTS natural mortality (log(yr^-1)), to be exponentiated
  PARAMETER(log_eps_imm); // log immigration efficiency (log(unitless)), to be exponentiated
  PARAMETER(beta_sst); // effect of SST on COTS growth (unitless, per deg C)
  PARAMETER(log_sigma_cots); // log obs SD for COTS (log scale)
  PARAMETER(log_sigma_fast); // log obs SD for fast coral (log scale)
  PARAMETER(log_sigma_slow); // log obs SD for slow coral (log scale)

  // --- TRANSFORM PARAMETERS ---
  Type r_cots = exp(log_r_cots); // COTS intrinsic growth rate (yr^-1)
  Type K_cots = exp(log_K_cots); // COTS carrying capacity (indiv/m2)
  Type alpha_fast = exp(log_alpha_fast); // COTS attack rate on fast coral (m2/indiv/yr)
  Type alpha_slow = exp(log_alpha_slow); // COTS attack rate on slow coral (m2/indiv/yr)
  Type h_fast = exp(log_h_fast); // Half-saturation for fast coral (%)
  Type h_slow = exp(log_h_slow); // Half-saturation for slow coral (%)
  Type r_fast = exp(log_r_fast); // Fast coral growth rate (%/yr)
  Type r_slow = exp(log_r_slow); // Slow coral growth rate (%/yr)
  Type K_fast = exp(log_K_fast); // Fast coral carrying capacity (%)
  Type K_slow = exp(log_K_slow); // Slow coral carrying capacity (%)
  Type m_cots = exp(log_m_cots); // COTS natural mortality (yr^-1)
  Type eps_imm = exp(log_eps_imm); // Immigration efficiency (unitless)
  Type sigma_cots = exp(log_sigma_cots); // Obs SD for COTS (log scale)
  Type sigma_fast = exp(log_sigma_fast); // Obs SD for fast coral (log scale)
  Type sigma_slow = exp(log_sigma_slow); // Obs SD for slow coral (log scale)

  // --- INITIAL STATES ---
  PARAMETER(log_init_cots); // log initial COTS abundance (log(indiv/m2))
  PARAMETER(log_init_fast); // log initial fast coral cover (log(%))
  PARAMETER(log_init_slow); // log initial slow coral cover (log(%))
  Type cots_prev = exp(log_init_cots); // initial COTS abundance
  Type fast_prev = exp(log_init_fast); // initial fast coral cover
  Type slow_prev = exp(log_init_slow); // initial slow coral cover

  // --- SMALL CONSTANTS FOR NUMERICAL STABILITY ---
  Type eps = Type(1e-8); // small value to avoid division by zero
  Type min_sd = Type(1e-3); // minimum SD for likelihood

  // --- STORAGE FOR PREDICTIONS ---
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // --- PROCESS MODEL ---
  // Use t=0 as initial state, predictions start at t=0
  // Set initial predictions to initial state
  cots_pred(0) = cots_prev;
  fast_pred(0) = fast_prev;
  slow_pred(0) = slow_prev;

  // Forward simulation
  for(int t=1; t<n; t++) {
    // 1. COTS functional response to coral (Type II, saturating)
    Type coral_food = alpha_fast * fast_prev/(fast_prev + h_fast + eps) + alpha_slow * slow_prev/(slow_prev + h_slow + eps); // total food intake rate (yr^-1)
    // 2. COTS population update: logistic growth + food limitation + immigration + mortality + SST effect
    Type r_cots_eff = r_cots * exp(beta_sst * (sst_dat(t-1) - Type(27.0))); // SST modifies growth (centered at 27C)
    Type immig = eps_imm * cotsimm_dat(t-1); // immigration pulse (indiv/m2/yr)
    Type cots_next = cots_prev + r_cots_eff * cots_prev * (1 - cots_prev/(K_cots + eps)) * exp(-coral_food) + immig - m_cots * cots_prev;
    if (CppAD::Value(cots_next) <= eps) cots_next = eps; // prevent negative/zero
    cots_pred(t) = cots_next;

    // 3. Coral predation losses (Type II, saturating), differential for fast/slow
    Type pred_fast = cots_prev * alpha_fast * fast_prev/(fast_prev + h_fast + eps); // predation on fast coral (%/yr)
    Type pred_slow = cots_prev * alpha_slow * slow_prev/(slow_prev + h_slow + eps); // predation on slow coral (%/yr)

    // 4. Coral updates: logistic growth - predation
    Type fast_next = fast_prev + r_fast * fast_prev * (1 - fast_prev/(K_fast + eps)) - pred_fast;
    Type slow_next = slow_prev + r_slow * slow_prev * (1 - slow_prev/(K_slow + eps)) - pred_slow;
    if (CppAD::Value(fast_next) <= eps) fast_next = eps; // prevent negative/zero
    if (CppAD::Value(slow_next) <= eps) slow_next = eps; // prevent negative/zero
    fast_pred(t) = fast_next;
    slow_pred(t) = slow_next;

    // 5. Update for next time step
    cots_prev = cots_next;
    fast_prev = fast_next;
    slow_prev = slow_next;
  }

  // --- LIKELIHOOD ---
  Type nll = 0.0;
  for(int t=0; t<n; t++) {
    // Log-transform for strictly positive data
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots + min_sd, true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast + min_sd, true);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow + min_sd, true);
  }

  // --- SMOOTH PENALTIES FOR PARAMETER BOUNDS ---
  // Example: penalize if COTS growth rate is outside plausible range (0.1-2.0 yr^-1)
  nll += pow(CppAD::CondExpLt(r_cots, Type(0.1), r_cots-Type(0.1), Type(0.0)), 2) * 10.0;
  nll += pow(CppAD::CondExpGt(r_cots, Type(2.0), r_cots-Type(2.0), Type(0.0)), 2) * 10.0;
  // Similar penalties can be added for other parameters as needed

  // --- REPORTING ---
  REPORT(cots_pred); // predicted COTS abundance (indiv/m2)
  REPORT(fast_pred); // predicted fast coral cover (%)
  REPORT(slow_pred); // predicted slow coral cover (%)

  return nll;
}

/*
Equation Descriptions:
1. COTS population: N_{t+1} = N_t + r_cots_eff * N_t * (1 - N_t/K_cots) * exp(-coral_food) + immigration - m_cots * N_t
2. Coral predation: pred_fast = N_t * alpha_fast * F_t/(F_t + h_fast); pred_slow = N_t * alpha_slow * S_t/(S_t + h_slow)
3. Coral update: F_{t+1} = F_t + r_fast * F_t * (1 - F_t/K_fast) - pred_fast; S_{t+1} = S_t + r_slow * S_t * (1 - S_t/K_slow) - pred_slow
4. SST modifies COTS growth: r_cots_eff = r_cots * exp(beta_sst * (sst - 27))
5. All predictions (_pred) use only previous time step values (no data leakage)
*/
