#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // 1. DATA SECTION
  // Data vectors (time series)
  DATA_VECTOR(Year); // Year (time variable, matches data file)
  DATA_VECTOR(sst_dat); // Sea-surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (indiv/m2/year)
  DATA_VECTOR(cots_dat); // Observed adult COTS abundance (indiv/m2)
  DATA_VECTOR(fast_dat); // Observed fast coral cover (Acropora, %)
  DATA_VECTOR(slow_dat); // Observed slow coral cover (Faviidae/Porites, %)

  // 2. PARAMETER SECTION
  // COTS population parameters
  PARAMETER(log_r_cots); // log intrinsic growth rate of COTS (year^-1)
  PARAMETER(log_K_cots); // log carrying capacity of COTS (indiv/m2)
  PARAMETER(log_m_cots); // log baseline COTS mortality rate (year^-1)
  PARAMETER(log_alpha_cots); // log COTS recruitment efficiency (unitless)
  PARAMETER(log_beta_cots); // log half-saturation coral cover for COTS recruitment (%)
  PARAMETER(log_gamma_cots); // log outbreak threshold parameter (unitless)
  PARAMETER(log_temp_cots); // log temperature sensitivity of COTS recruitment (deg C^-1)
  PARAMETER(log_phi_cots); // log effect of larval immigration on COTS recruitment (unitless)

  // Coral parameters
  PARAMETER(log_r_fast); // log growth rate of fast coral (year^-1)
  PARAMETER(log_r_slow); // log growth rate of slow coral (year^-1)
  PARAMETER(log_K_coral); // log total coral carrying capacity (% cover)
  PARAMETER(log_m_fast); // log background mortality of fast coral (year^-1)
  PARAMETER(log_m_slow); // log background mortality of slow coral (year^-1)
  PARAMETER(log_q_fast); // log COTS predation rate on fast coral (m2/indiv/year)
  PARAMETER(log_q_slow); // log COTS predation rate on slow coral (m2/indiv/year)
  PARAMETER(log_e_fast); // log assimilation efficiency of fast coral to COTS (unitless)
  PARAMETER(log_e_slow); // log assimilation efficiency of slow coral to COTS (unitless)

  // Observation error parameters
  PARAMETER(log_sigma_cots); // log SD for COTS (lognormal)
  PARAMETER(log_sigma_fast); // log SD for fast coral (lognormal)
  PARAMETER(log_sigma_slow); // log SD for slow coral (lognormal)

  // 3. TRANSFORM PARAMETERS TO NATURAL SCALE
  Type r_cots = exp(log_r_cots); // COTS intrinsic growth rate (year^-1)
  Type K_cots = exp(log_K_cots); // COTS carrying capacity (indiv/m2)
  Type m_cots = exp(log_m_cots); // COTS baseline mortality (year^-1)
  Type alpha_cots = exp(log_alpha_cots); // COTS recruitment efficiency
  Type beta_cots = exp(log_beta_cots); // COTS recruitment half-saturation (%)
  Type gamma_cots = exp(log_gamma_cots); // Outbreak threshold parameter
  Type temp_cots = exp(log_temp_cots); // Temperature sensitivity (deg C^-1)
  Type phi_cots = exp(log_phi_cots); // Larval immigration effect

  Type r_fast = exp(log_r_fast); // Fast coral growth rate (year^-1)
  Type r_slow = exp(log_r_slow); // Slow coral growth rate (year^-1)
  Type K_coral = exp(log_K_coral); // Total coral carrying capacity (%)
  Type m_fast = exp(log_m_fast); // Fast coral background mortality (year^-1)
  Type m_slow = exp(log_m_slow); // Slow coral background mortality (year^-1)
  Type q_fast = exp(log_q_fast); // COTS predation rate on fast coral (m2/indiv/year)
  Type q_slow = exp(log_q_slow); // COTS predation rate on slow coral (m2/indiv/year)
  Type e_fast = exp(log_e_fast); // Assimilation efficiency (fast coral)
  Type e_slow = exp(log_e_slow); // Assimilation efficiency (slow coral)

  Type sigma_cots = exp(log_sigma_cots); // SD for COTS (lognormal)
  Type sigma_fast = exp(log_sigma_fast); // SD for fast coral (lognormal)
  Type sigma_slow = exp(log_sigma_slow); // SD for slow coral (lognormal)

  // 4. INITIAL CONDITIONS
  int n = Year.size();
  vector<Type> cots_pred(n); // Predicted COTS abundance (indiv/m2)
  vector<Type> fast_pred(n); // Predicted fast coral cover (%)
  vector<Type> slow_pred(n); // Predicted slow coral cover (%)

  // Set initial conditions to observed values at t=0
  cots_pred(0) = CppAD::CondExpGt(cots_dat(0), Type(1e-8), cots_dat(0), Type(1e-8));
  fast_pred(0) = CppAD::CondExpGt(fast_dat(0), Type(1e-8), fast_dat(0), Type(1e-8));
  slow_pred(0) = CppAD::CondExpGt(slow_dat(0), Type(1e-8), slow_dat(0), Type(1e-8));

  // 5. PROCESS MODEL
  for(int t=1; t<n; t++) {
    // Previous time step values
    Type cots_prev = cots_pred(t-1); // indiv/m2
    Type fast_prev = fast_pred(t-1); // %
    Type slow_prev = slow_pred(t-1); // %

    // Total coral cover (for resource limitation)
    Type coral_prev = fast_prev + slow_prev + Type(1e-8); // % cover, avoid zero

    // 1. COTS recruitment (density-dependent, saturating, outbreak-triggered)
    // Recruitment is a function of available coral, temperature, larval supply, and outbreak threshold
    Type recruit_base = alpha_cots * (coral_prev / (beta_cots + coral_prev)); // Michaelis-Menten resource limitation
    Type recruit_env = exp(temp_cots * (sst_dat(t-1) - Type(28.0))); // Temperature effect (centered at 28C)
    Type recruit_imm = phi_cots * cotsimm_dat(t-1); // Larval immigration effect
    Type outbreak_trigger = Type(1.0) / (Type(1.0) + exp(-gamma_cots * (cotsimm_dat(t-1) - Type(0.5)))); // Smooth outbreak threshold
    Type cots_recruit = recruit_base * recruit_env * (Type(1.0) + recruit_imm) * outbreak_trigger;

    // 2. COTS mortality (density-dependent, resource-limited)
    Type mort_cots = m_cots + (cots_prev / (K_cots + cots_prev)); // Density-dependent mortality

    // 3. COTS predation on coral (functional response)
    Type pred_fast = q_fast * cots_prev * fast_prev / (K_coral + fast_prev + slow_prev + Type(1e-8)); // Fast coral predation
    Type pred_slow = q_slow * cots_prev * slow_prev / (K_coral + fast_prev + slow_prev + Type(1e-8)); // Slow coral predation

    // 4. COTS population update (boom-bust dynamics)
    cots_pred(t) = cots_prev + r_cots * cots_prev * (Type(1.0) - cots_prev / (K_cots + Type(1e-8)))
                   + cots_recruit * (e_fast * pred_fast + e_slow * pred_slow)
                   - mort_cots * cots_prev;
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), Type(1e-8), cots_pred(t), Type(1e-8)); // Prevent negative/zero

    // 5. Fast coral update
    fast_pred(t) = fast_prev + r_fast * fast_prev * (Type(1.0) - (fast_prev + slow_prev) / (K_coral + Type(1e-8)))
                   - pred_fast - m_fast * fast_prev;
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), Type(1e-8), fast_pred(t), Type(1e-8)); // Prevent negative/zero

    // 6. Slow coral update
    slow_pred(t) = slow_prev + r_slow * slow_prev * (Type(1.0) - (fast_prev + slow_prev) / (K_coral + Type(1e-8)))
                   - pred_slow - m_slow * slow_prev;
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), Type(1e-8), slow_pred(t), Type(1e-8)); // Prevent negative/zero
  }

  // 6. LIKELIHOOD (LOGNORMAL, FIXED MINIMUM SD)
  Type eps = Type(1e-3); // Minimum SD for numerical stability
  Type nll = 0.0;
  for(int t=0; t<n; t++) {
    // Log-transform for strictly positive data
    nll -= dnorm(log(CppAD::CondExpGt(cots_dat(t), Type(1e-8), cots_dat(t), Type(1e-8))),
                 log(CppAD::CondExpGt(cots_pred(t), Type(1e-8), cots_pred(t), Type(1e-8))),
                 sigma_cots + eps, true);
    nll -= dnorm(log(CppAD::CondExpGt(fast_dat(t), Type(1e-8), fast_dat(t), Type(1e-8))),
                 log(CppAD::CondExpGt(fast_pred(t), Type(1e-8), fast_pred(t), Type(1e-8))),
                 sigma_fast + eps, true);
    nll -= dnorm(log(CppAD::CondExpGt(slow_dat(t), Type(1e-8), slow_dat(t), Type(1e-8))),
                 log(CppAD::CondExpGt(slow_pred(t), Type(1e-8), slow_pred(t), Type(1e-8))),
                 sigma_slow + eps, true);

    // NOTE: Do not use R_finite or any double-only check on AD types.
    // If you want to penalize for non-finite values, use TMB's isNA/isInf for Type.
    // But for most models, simply using CppAD::CondExpGt to keep values positive is sufficient.
    // If you want to add a penalty for negative or zero predictions, you can do:
    // Remove the above block: do not use logical checks on AD types (not supported in TMB)
    // Instead, rely on CppAD::CondExpGt to keep values positive, as already done above.
  }

  // 7. PENALTIES FOR PARAMETER BOUNDS (SMOOTH)
  nll += pow(CppAD::CondExpLt(r_cots, Type(0.01), r_cots-Type(0.01), Type(0.0)), 2); // r_cots > 0.01
  nll += pow(CppAD::CondExpGt(r_cots, Type(5.0), r_cots-Type(5.0), Type(0.0)), 2); // r_cots < 5
  nll += pow(CppAD::CondExpLt(K_cots, Type(0.01), K_cots-Type(0.01), Type(0.0)), 2); // K_cots > 0.01
  nll += pow(CppAD::CondExpLt(K_coral, Type(1.0), K_coral-Type(1.0), Type(0.0)), 2); // K_coral > 1%

  // 8. REPORTING
  REPORT(cots_pred); // Predicted COTS abundance (indiv/m2)
  REPORT(fast_pred); // Predicted fast coral cover (%)
  REPORT(slow_pred); // Predicted slow coral cover (%)

  // 9. EQUATION DESCRIPTIONS
  // 1. COTS recruitment: Michaelis-Menten resource limitation, temperature, larval supply, smooth outbreak threshold
  // 2. COTS mortality: baseline + density-dependent
  // 3. COTS predation: saturating functional response on each coral group
  // 4. COTS update: logistic growth + recruitment + coral assimilation - mortality
  // 5. Coral update: logistic growth - COTS predation - background mortality

  return nll;
}
