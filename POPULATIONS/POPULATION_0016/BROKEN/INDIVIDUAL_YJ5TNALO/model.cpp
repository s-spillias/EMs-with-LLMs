// Needed for NA/NaN detection with TMB/CppAD types
#include <TMB.hpp>
extern "C" {
  #include <R_ext/Arith.h>
}
template<class Type> double asDouble(Type x) { return CppAD::Value(x); }

// 1. Model for episodic COTS outbreaks and coral community dynamics
// 2. State variables: cots (Crown-of-Thorns starfish), fast (Acropora), slow (Faviidae/Porites)
// 3. Environmental drivers: sst (sea surface temperature), cotsimm (larval immigration)
// 4. All predictions use lagged (previous time step) state variables only

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA INPUTS ---
  DATA_VECTOR(Year); // Observation years
  DATA_VECTOR(cots_dat); // Observed COTS density (indiv/m2)
  DATA_VECTOR(fast_dat); // Observed fast coral cover (%)
  DATA_VECTOR(slow_dat); // Observed slow coral cover (%)
  DATA_VECTOR(sst_dat); // Sea surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (indiv/m2/year)

  int n = cots_dat.size();

  // --- PARAMETERS ---
  PARAMETER(log_r_cots); // log intrinsic COTS recruitment rate (log(year^-1))
  PARAMETER(log_K_cots); // log COTS carrying capacity (log(indiv/m2))
  PARAMETER(log_m_cots); // log COTS natural mortality rate (log(year^-1))
  PARAMETER(log_alpha_acrop); // log COTS predation rate on Acropora (log(m2/year))
  PARAMETER(log_alpha_slow); // log COTS predation rate on slow corals (log(m2/year))
  PARAMETER(log_h_acrop); // log half-saturation for Acropora predation (log(% cover))
  PARAMETER(log_h_slow); // log half-saturation for slow coral predation (log(% cover))
  PARAMETER(log_r_fast); // log Acropora growth rate (log(%/year))
  PARAMETER(log_r_slow); // log slow coral growth rate (log(%/year))
  PARAMETER(log_K_fast); // log Acropora carrying capacity (log(% cover))
  PARAMETER(log_K_slow); // log slow coral carrying capacity (log(% cover))
  PARAMETER(log_temp_cots); // log temperature effect on COTS survival (logit scale)
  PARAMETER(log_temp_coral); // log temperature effect on coral growth (logit scale)
  PARAMETER(log_sigma_cots); // log SD for COTS obs (lognormal)
  PARAMETER(log_sigma_fast); // log SD for fast coral obs (lognormal)
  PARAMETER(log_sigma_slow); // log SD for slow coral obs (lognormal)
  PARAMETER(log_h_cots_coral); // log half-saturation for Acropora effect on COTS recruitment (log(% cover))

  // --- TRANSFORM PARAMETERS ---
  Type r_cots = exp(log_r_cots); // COTS recruitment rate (year^-1)
  Type K_cots = exp(log_K_cots); // COTS carrying capacity (indiv/m2)
  Type m_cots = exp(log_m_cots); // COTS mortality (year^-1)
  Type alpha_acrop = exp(log_alpha_acrop); // COTS predation rate on Acropora (m2/year)
  Type alpha_slow = exp(log_alpha_slow); // COTS predation rate on slow corals (m2/year)
  Type h_acrop = exp(log_h_acrop); // Half-saturation for Acropora predation (% cover)
  Type h_slow = exp(log_h_slow); // Half-saturation for slow coral predation (% cover)
  Type r_fast = exp(log_r_fast); // Acropora growth rate (%/year)
  Type r_slow = exp(log_r_slow); // Slow coral growth rate (%/year)
  Type K_fast = exp(log_K_fast); // Acropora carrying capacity (% cover)
  Type K_slow = exp(log_K_slow); // Slow coral carrying capacity (% cover)
  Type temp_cots = 1.0/(1.0 + exp(-exp(log_temp_cots))); // Logistic effect of temp on COTS
  Type temp_coral = 1.0/(1.0 + exp(-exp(log_temp_coral))); // Logistic effect of temp on coral
  Type sigma_cots = exp(log_sigma_cots) + Type(1e-8); // SD for COTS obs
  Type sigma_fast = exp(log_sigma_fast) + Type(1e-8); // SD for fast coral obs
  Type sigma_slow = exp(log_sigma_slow) + Type(1e-8); // SD for slow coral obs
  Type h_cots_coral = exp(log_h_cots_coral); // Half-saturation for Acropora effect on COTS recruitment (% cover)

  // --- INITIAL STATES ---
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Set initial state to first observation
  cots_pred(0) = (CppAD::Value(cots_dat(0)) != CppAD::Value(cots_dat(0)) || cots_dat(0) <= Type(0.0)) ? Type(1e-8) : cots_dat(0);
  fast_pred(0) = (CppAD::Value(fast_dat(0)) != CppAD::Value(fast_dat(0)) || fast_dat(0) <= Type(0.0)) ? Type(1e-8) : fast_dat(0);
  slow_pred(0) = (CppAD::Value(slow_dat(0)) != CppAD::Value(slow_dat(0)) || slow_dat(0) <= Type(0.0)) ? Type(1e-8) : slow_dat(0);

  // --- PROCESS MODEL ---
  for(int t=1; t<n; t++){
    // Environmental effects (smooth, bounded)
    Type temp_eff_cots = 1.0/(1.0 + exp(-(sst_dat(t-1)-28.0)*temp_cots)); // 28C = reference
    Type temp_eff_coral = 1.0/(1.0 + exp(-(sst_dat(t-1)-28.0)*temp_coral));

    // COTS predation on corals (Michaelis-Menten, saturating)
    Type pred_fast = alpha_acrop * cots_pred(t-1) * fast_pred(t-1) / (h_acrop + fast_pred(t-1) + Type(1e-8));
    Type pred_slow = alpha_slow * cots_pred(t-1) * slow_pred(t-1) / (h_slow + slow_pred(t-1) + Type(1e-8));

    // COTS population dynamics (recruitment, immigration, mortality, resource limitation, prey feedback)
    Type coral_feedback = fast_pred(t-1) / (h_cots_coral + fast_pred(t-1) + Type(1e-8));
    Type recruit_cots = r_cots * cots_pred(t-1) * (1.0 - cots_pred(t-1)/K_cots) * temp_eff_cots * coral_feedback;
    Type immigrate_cots = cotsimm_dat(t-1); // External larval input
    Type mort_cots = m_cots * cots_pred(t-1);

    // Update COTS (ensure non-negative)
    cots_pred(t) = cots_pred(t-1) + recruit_cots + immigrate_cots - mort_cots;
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), Type(0.0), cots_pred(t), Type(1e-8)); // Smooth lower bound

    // Acropora (fast coral) dynamics (growth, predation, resource limitation)
    Type grow_fast = r_fast * fast_pred(t-1) * (1.0 - (fast_pred(t-1)+slow_pred(t-1))/K_fast) * temp_eff_coral;
    fast_pred(t) = fast_pred(t-1) + grow_fast - pred_fast;
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), Type(0.0), fast_pred(t), Type(1e-8));

    // Slow coral dynamics (growth, predation, resource limitation)
    Type grow_slow = r_slow * slow_pred(t-1) * (1.0 - (fast_pred(t-1)+slow_pred(t-1))/K_slow) * temp_eff_coral;
    slow_pred(t) = slow_pred(t-1) + grow_slow - pred_slow;
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), Type(0.0), slow_pred(t), Type(1e-8));
  }

  // --- LIKELIHOOD ---
  Type nll = 0.0;
  for(int t=0; t<n; t++){
    // Lognormal likelihood for strictly positive data
    Type cots_obs = (CppAD::Value(cots_dat(t)) != CppAD::Value(cots_dat(t)) || cots_dat(t) <= Type(0.0)) ? Type(1e-8) : cots_dat(t);
    Type cots_mod = (CppAD::Value(cots_pred(t)) != CppAD::Value(cots_pred(t)) || cots_pred(t) <= Type(0.0)) ? Type(1e-8) : cots_pred(t);
    Type fast_obs = (CppAD::Value(fast_dat(t)) != CppAD::Value(fast_dat(t)) || fast_dat(t) <= Type(0.0)) ? Type(1e-8) : fast_dat(t);
    Type fast_mod = (CppAD::Value(fast_pred(t)) != CppAD::Value(fast_pred(t)) || fast_pred(t) <= Type(0.0)) ? Type(1e-8) : fast_pred(t);
    Type slow_obs = (CppAD::Value(slow_dat(t)) != CppAD::Value(slow_dat(t)) || slow_dat(t) <= Type(0.0)) ? Type(1e-8) : slow_dat(t);
    Type slow_mod = (CppAD::Value(slow_pred(t)) != CppAD::Value(slow_pred(t)) || slow_pred(t) <= Type(0.0)) ? Type(1e-8) : slow_pred(t);

    nll -= dnorm(log(cots_obs), log(cots_mod), sigma_cots, true);
    nll -= dnorm(log(fast_obs), log(fast_mod), sigma_fast, true);
    nll -= dnorm(log(slow_obs), log(slow_mod), sigma_slow, true);
  }

  // --- REPORTING ---
  REPORT(cots_pred); // Predicted COTS density (indiv/m2)
  REPORT(fast_pred); // Predicted Acropora cover (%)
  REPORT(slow_pred); // Predicted slow coral cover (%)

  // --- EQUATION DESCRIPTIONS ---
  // 1. COTS: cots_pred(t) = cots_pred(t-1) + recruit_cots + immigrate_cots - mort_cots
  // 2. recruit_cots = r_cots * cots_pred(t-1) * (1 - cots_pred(t-1)/K_cots) * temp_eff_cots * (fast_pred(t-1)/(h_cots_coral + fast_pred(t-1)))
  // 3. mort_cots = m_cots * cots_pred(t-1)
  // 4. pred_fast = alpha_acrop * cots_pred(t-1) * fast_pred(t-1) / (h_acrop + fast_pred(t-1))
  // 5. pred_slow = alpha_slow * cots_pred(t-1) * slow_pred(t-1) / (h_slow + slow_pred(t-1))
  // 6. fast_pred(t) = fast_pred(t-1) + grow_fast - pred_fast
  // 7. grow_fast = r_fast * fast_pred(t-1) * (1 - (fast_pred(t-1)+slow_pred(t-1))/K_fast) * temp_eff_coral
  // 8. slow_pred(t) = slow_pred(t-1) + grow_slow - pred_slow
  // 9. grow_slow = r_slow * slow_pred(t-1) * (1 - (fast_pred(t-1)+slow_pred(t-1))/K_slow) * temp_eff_coral

  return nll;
}
