#include <TMB.hpp>

// 1. Model equations are described at the end of this file.

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA INPUTS ---
  DATA_VECTOR(Year); // Observation years
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (indiv/m2/yr)
  DATA_VECTOR(sst_dat); // Sea-surface temperature (deg C)
  DATA_VECTOR(cots_dat); // Observed adult COTS abundance (indiv/m2)
  DATA_VECTOR(fast_dat); // Observed fast coral cover (%)
  DATA_VECTOR(slow_dat); // Observed slow coral cover (%)

  // --- PARAMETERS ---
  PARAMETER(log_r_cots); // log intrinsic COTS growth rate (log(yr^-1))
  PARAMETER(log_K_cots); // log COTS carrying capacity (log(indiv/m2))
  PARAMETER(log_alpha_cots); // log COTS predation rate scaling (log(m2/indiv/yr))
  PARAMETER(log_beta_cots); // log COTS predation half-saturation (log(% coral cover))
  PARAMETER(log_m_cots); // log COTS natural mortality (log(yr^-1))
  PARAMETER(log_phi_imm); // log immigration efficiency (log(unitless))
  PARAMETER(log_r_fast); // log fast coral growth rate (log(%/yr))
  PARAMETER(log_r_slow); // log slow coral growth rate (log(%/yr))
  PARAMETER(log_K_fast); // log fast coral carrying capacity (log(% cover))
  PARAMETER(log_K_slow); // log slow coral carrying capacity (log(% cover))
  PARAMETER(log_gamma_fast); // log COTS predation efficiency on fast coral (log(unitless))
  PARAMETER(log_gamma_slow); // log COTS predation efficiency on slow coral (log(unitless))
  PARAMETER(log_sigma_cots); // log obs SD for COTS (log(indiv/m2))
  PARAMETER(log_sigma_fast); // log obs SD for fast coral (log(%))
  PARAMETER(log_sigma_slow); // log obs SD for slow coral (log(%))
  PARAMETER(logit_temp_effect); // logit temperature effect scaling (logit(unitless))

  // --- TRANSFORM PARAMETERS ---
  Type r_cots = exp(log_r_cots); // Intrinsic COTS growth rate (yr^-1)
  Type K_cots = exp(log_K_cots); // COTS carrying capacity (indiv/m2)
  Type alpha_cots = exp(log_alpha_cots); // COTS predation rate scaling (m2/indiv/yr)
  Type beta_cots = exp(log_beta_cots); // COTS predation half-saturation (% coral cover)
  Type m_cots = exp(log_m_cots); // COTS natural mortality (yr^-1)
  Type phi_imm = exp(log_phi_imm); // Immigration efficiency (unitless)
  Type r_fast = exp(log_r_fast); // Fast coral growth rate (%/yr)
  Type r_slow = exp(log_r_slow); // Slow coral growth rate (%/yr)
  Type K_fast = exp(log_K_fast); // Fast coral carrying capacity (%)
  Type K_slow = exp(log_K_slow); // Slow coral carrying capacity (%)
  Type gamma_fast = exp(log_gamma_fast); // COTS predation efficiency on fast coral (unitless)
  Type gamma_slow = exp(log_gamma_slow); // COTS predation efficiency on slow coral (unitless)
  Type sigma_cots = exp(log_sigma_cots) + Type(1e-6); // Obs SD for COTS (indiv/m2)
  Type sigma_fast = exp(log_sigma_fast) + Type(1e-6); // Obs SD for fast coral (%)
  Type sigma_slow = exp(log_sigma_slow) + Type(1e-6); // Obs SD for slow coral (%)
  Type temp_effect = Type(1)/(Type(1)+exp(-logit_temp_effect)); // Temperature effect scaling (0-1)

  int n = Year.size();

  // --- STATE VARIABLES ---
  vector<Type> cots_pred(n); // Predicted COTS abundance (indiv/m2)
  vector<Type> fast_pred(n); // Predicted fast coral cover (%)
  vector<Type> slow_pred(n); // Predicted slow coral cover (%)

  // --- INITIAL CONDITIONS ---
  cots_pred(0) = (cots_dat(0) > 0 ? cots_dat(0) : Type(1e-4)); // Avoid zero/negative
  fast_pred(0) = (fast_dat(0) > 0 ? fast_dat(0) : Type(1e-4));
  slow_pred(0) = (slow_dat(0) > 0 ? slow_dat(0) : Type(1e-4));

  // Defensive: ensure initial conditions are finite and positive
  // NOTE: Only use plain C++ logic on data (not AD types) for initial conditions.

  // --- PROCESS MODEL ---
  for(int t=1; t<n; t++){
    // 1. Resource limitation: total coral cover available
    Type coral_total_prev = fast_pred(t-1) + slow_pred(t-1) + Type(1e-8); // % cover, avoid zero

    // 2. Environmental effect: temperature modifies COTS growth and coral growth
    Type temp_dev = sst_dat(t-1) - Type(28.0); // Deviation from reference temp (deg C)
    Type temp_mod = exp(temp_effect * temp_dev); // Smooth effect, >1 if temp above ref

    // 3. COTS population dynamics
    // Immigration pulse (episodic outbreaks)
    Type immig = phi_imm * cotsimm_dat(t-1); // Immigration efficiency * observed immigration

    // COTS predation functional response (Type II, saturating)
    Type predation = alpha_cots * cots_pred(t-1) * coral_total_prev / (beta_cots + coral_total_prev);

    // COTS population update (boom-bust, resource-limited, temp-modified)
    Type cots_growth = r_cots * cots_pred(t-1) * (Type(1) - cots_pred(t-1)/K_cots) * temp_mod;
    cots_pred(t) = cots_pred(t-1) + cots_growth + immig - predation - m_cots * cots_pred(t-1);
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), Type(1e-8), cots_pred(t), Type(1e-8)); // Avoid negative

    // 4. Coral dynamics: fast and slow
    // Fast coral (Acropora)
    Type fast_growth = r_fast * fast_pred(t-1) * (Type(1) - (fast_pred(t-1)+slow_pred(t-1))/K_fast) * temp_mod;
    Type fast_predation = gamma_fast * predation * (fast_pred(t-1)/coral_total_prev); // Selective predation
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_predation;
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), Type(1e-8), fast_pred(t), Type(1e-8));

    // Slow coral (Faviidae/Porites)
    Type slow_growth = r_slow * slow_pred(t-1) * (Type(1) - (fast_pred(t-1)+slow_pred(t-1))/K_slow) * temp_mod;
    Type slow_predation = gamma_slow * predation * (slow_pred(t-1)/coral_total_prev);
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_predation;
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), Type(1e-8), slow_pred(t), Type(1e-8));

    // Defensive: prevent runaway or non-finite values
    // NOTE: Do not use logical checks on AD types, as this can cause segfaults in TMB.
    // The CppAD::CondExpGt above is sufficient for numerical stability.
  }

  // --- LIKELIHOOD ---
  Type nll = 0.0;
  for(int t=0; t<n; t++){
    // Lognormal likelihood for strictly positive data
    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t) + Type(1e-8)), sigma_cots, true);
    nll -= dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred(t) + Type(1e-8)), sigma_fast, true);
    nll -= dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred(t) + Type(1e-8)), sigma_slow, true);

    // NOTE: Do not use logical checks on AD types, as this can cause segfaults in TMB.
    // The CppAD::CondExpGt above is sufficient for numerical stability.
  }

  // --- REPORTING ---
  REPORT(cots_pred); // Predicted COTS abundance (indiv/m2)
  REPORT(fast_pred); // Predicted fast coral cover (%)
  REPORT(slow_pred); // Predicted slow coral cover (%)

  return nll;
}

/*
Model equations (numbered):

1. COTS immigration: immig = phi_imm * cotsimm_dat(t-1)
2. COTS predation: predation = alpha_cots * COTS * coral_total / (beta_cots + coral_total)
3. COTS growth: cots_growth = r_cots * COTS * (1 - COTS/K_cots) * temp_mod
4. COTS update: COTS(t) = COTS(t-1) + cots_growth + immig - predation - m_cots * COTS(t-1)
5. Fast coral growth: fast_growth = r_fast * fast * (1 - (fast+slow)/K_fast) * temp_mod
6. Fast coral predation: fast_predation = gamma_fast * predation * (fast/coral_total)
7. Fast coral update: fast(t) = fast(t-1) + fast_growth - fast_predation
8. Slow coral growth: slow_growth = r_slow * slow * (1 - (fast+slow)/K_slow) * temp_mod
9. Slow coral predation: slow_predation = gamma_slow * predation * (slow/coral_total)
10. Slow coral update: slow(t) = slow(t-1) + slow_growth - slow_predation

All variables with _pred are predictions, _dat are observations.
All rates are per year. All coral cover is in percent (%).
*/
