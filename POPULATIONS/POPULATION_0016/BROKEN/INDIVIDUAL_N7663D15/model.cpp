#include <TMB.hpp>

// 1. Model equations describe the coupled dynamics of:
//    (1) Adult COTS abundance (cots_pred)
//    (2) Fast-growing coral cover (fast_pred)
//    (3) Slow-growing coral cover (slow_pred)
//    All predictions use only previous time step values to avoid data leakage.

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA INPUTS ---
  DATA_VECTOR(Year); // Year (time variable)
  DATA_VECTOR(cots_dat); // Adult COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat); // Fast-growing coral cover (%)
  DATA_VECTOR(slow_dat); // Slow-growing coral cover (%)
  DATA_VECTOR(sst_dat); // Sea-surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration rate (individuals/m2/year)

  int n = Year.size();

  // --- PARAMETERS ---
  PARAMETER(log_r_cots); // log intrinsic growth rate of COTS (year^-1)
  PARAMETER(log_K_cots); // log carrying capacity for COTS (individuals/m2)
  PARAMETER(log_alpha_fast); // log attack rate on fast coral (m2/individual/year)
  PARAMETER(log_alpha_slow); // log attack rate on slow coral (m2/individual/year)
  PARAMETER(log_e_fast); // log assimilation efficiency from fast coral (unitless)
  PARAMETER(log_e_slow); // log assimilation efficiency from slow coral (unitless)
  PARAMETER(log_m_cots); // log baseline mortality rate of COTS (year^-1)
  PARAMETER(log_gamma); // log coral recovery rate (year^-1)
  PARAMETER(log_K_fast); // log max fast coral cover (%)
  PARAMETER(log_K_slow); // log max slow coral cover (%)
  PARAMETER(log_env_sens); // log environmental sensitivity (unitless)
  PARAMETER(log_sigma_cots); // log SD for COTS obs (lognormal)
  PARAMETER(log_sigma_fast); // log SD for fast coral obs (lognormal)
  PARAMETER(log_sigma_slow); // log SD for slow coral obs (lognormal)

  // --- TRANSFORM PARAMETERS ---
  Type r_cots = exp(log_r_cots); // Intrinsic growth rate of COTS
  Type K_cots = exp(log_K_cots); // Carrying capacity for COTS
  Type alpha_fast = exp(log_alpha_fast); // Attack rate on fast coral
  Type alpha_slow = exp(log_alpha_slow); // Attack rate on slow coral
  Type e_fast = exp(log_e_fast); // Assimilation efficiency from fast coral
  Type e_slow = exp(log_e_slow); // Assimilation efficiency from slow coral
  Type m_cots = exp(log_m_cots); // Baseline mortality rate of COTS
  Type gamma = exp(log_gamma); // Coral recovery rate
  Type K_fast = exp(log_K_fast); // Max fast coral cover
  Type K_slow = exp(log_K_slow); // Max slow coral cover
  Type env_sens = exp(log_env_sens); // Environmental sensitivity
  Type sigma_cots = exp(log_sigma_cots) + Type(1e-8); // SD for COTS obs
  Type sigma_fast = exp(log_sigma_fast) + Type(1e-8); // SD for fast coral obs
  Type sigma_slow = exp(log_sigma_slow) + Type(1e-8); // SD for slow coral obs

  // --- INITIAL CONDITIONS ---
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Defensive: ensure input vectors are non-empty and not NaN/Inf
  cots_pred(0) = (cots_dat(0) > Type(0) && cots_dat(0) == cots_dat(0)) ? cots_dat(0) : Type(1e-8); // Initial COTS abundance
  fast_pred(0) = (fast_dat(0) > Type(0) && fast_dat(0) == fast_dat(0)) ? fast_dat(0) : Type(1e-8); // Initial fast coral cover
  slow_pred(0) = (slow_dat(0) > Type(0) && slow_dat(0) == slow_dat(0)) ? slow_dat(0) : Type(1e-8); // Initial slow coral cover

  // --- PROCESS MODEL ---
  for(int t=1; t<n; t++){
    // 1. Resource limitation for COTS (Michaelis-Menten type)
    Type food_avail = (alpha_fast * fast_pred(t-1) + alpha_slow * slow_pred(t-1)) / 
                      (Type(1.0) + alpha_fast * fast_pred(t-1) + alpha_slow * slow_pred(t-1) + Type(1e-8)); // Unitless, saturating

    // 2. Environmental modifier (e.g., SST anomaly effect, smooth)
    Type env_mod = exp(env_sens * (sst_dat(t-1) - Type(27.0))); // Sensitivity to SST, centered at 27C

    // 3. COTS population dynamics (boom-bust, with immigration)
    Type cots_growth = r_cots * cots_pred(t-1) * food_avail * env_mod; // Growth term
    Type cots_mortality = m_cots * cots_pred(t-1); // Mortality
    Type cots_density_dep = (cots_pred(t-1) / (K_cots + Type(1e-8))); // Density dependence
    Type cots_immigration = cotsimm_dat(t-1); // Immigration

    cots_pred(t) = cots_pred(t-1) + cots_growth * (Type(1.0) - cots_density_dep) - cots_mortality + cots_immigration;
    // Prevent negative/zero
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), Type(1e-8), cots_pred(t), Type(1e-8));

    // 4. Coral predation (Type II functional response, saturating)
    Type pred_fast = (alpha_fast * cots_pred(t-1) * fast_pred(t-1)) / (Type(1.0) + alpha_fast * fast_pred(t-1) + Type(1e-8));
    Type pred_slow = (alpha_slow * cots_pred(t-1) * slow_pred(t-1)) / (Type(1.0) + alpha_slow * slow_pred(t-1) + Type(1e-8));

    // 5. Coral recovery (logistic, with smooth threshold for minimum cover)
    Type fast_recovery = gamma * fast_pred(t-1) * (Type(1.0) - fast_pred(t-1)/K_fast);
    Type slow_recovery = gamma * slow_pred(t-1) * (Type(1.0) - slow_pred(t-1)/K_slow);

    // 6. Coral update equations
    fast_pred(t) = fast_pred(t-1) + fast_recovery - pred_fast;
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), Type(1e-8), fast_pred(t), Type(1e-8)); // Prevent negative/zero

    slow_pred(t) = slow_pred(t-1) + slow_recovery - pred_slow;
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), Type(1e-8), slow_pred(t), Type(1e-8)); // Prevent negative/zero
  }

  // --- LIKELIHOOD ---
  Type nll = 0.0;
  for(int t=0; t<n; t++){
    // Lognormal likelihood for strictly positive data
    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t) + Type(1e-8)), sigma_cots, true);
    nll -= dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred(t) + Type(1e-8)), sigma_fast, true);
    nll -= dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred(t) + Type(1e-8)), sigma_slow, true);
  }

  // --- SMOOTH PENALTIES FOR PARAMETER BOUNDS ---
  // Example: Bound K_cots, K_fast, K_slow to be >0 and < reasonable upper limits
  nll += pow(CppAD::CondExpGt(K_cots, Type(10.0), K_cots-Type(10.0), Type(0.0)), 2); // Penalty if K_cots > 10 ind/m2
  nll += pow(CppAD::CondExpLt(K_cots, Type(1e-4), Type(1e-4)-K_cots, Type(0.0)), 2); // Penalty if K_cots < 1e-4

  nll += pow(CppAD::CondExpGt(K_fast, Type(100.0), K_fast-Type(100.0), Type(0.0)), 2); // Penalty if K_fast > 100%
  nll += pow(CppAD::CondExpLt(K_fast, Type(1e-4), Type(1e-4)-K_fast, Type(0.0)), 2); // Penalty if K_fast < 1e-4

  nll += pow(CppAD::CondExpGt(K_slow, Type(100.0), K_slow-Type(100.0), Type(0.0)), 2); // Penalty if K_slow > 100%
  nll += pow(CppAD::CondExpLt(K_slow, Type(1e-4), Type(1e-4)-K_slow, Type(0.0)), 2); // Penalty if K_slow < 1e-4

  // --- REPORTING ---
  REPORT(cots_pred); // Predicted COTS abundance (individuals/m2)
  REPORT(fast_pred); // Predicted fast coral cover (%)
  REPORT(slow_pred); // Predicted slow coral cover (%)

  return nll;
}

/*
Equation descriptions:
1. COTS growth: r_cots * COTS * food_avail * env_mod * (1 - density/K_cots)
2. COTS mortality: m_cots * COTS
3. COTS immigration: cotsimm_dat
4. Coral predation: Type II functional response (saturating) for each coral group
5. Coral recovery: logistic growth with recovery rate gamma and carrying capacity K
6. All updates use only previous time step values (no data leakage)
7. Environmental modifier: exponential function of SST anomaly
8. Parameter bounds enforced with smooth penalties
9. Lognormal likelihood for all observed variables
*/
