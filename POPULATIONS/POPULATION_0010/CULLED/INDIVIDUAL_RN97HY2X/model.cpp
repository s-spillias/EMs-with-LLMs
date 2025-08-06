#include <TMB.hpp>

// 1. Model equations are described at the end of this file.

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
  PARAMETER(log_alpha_cots); // log COTS predation rate on coral (m2/indiv/year)
  PARAMETER(log_beta_fast); // log selectivity for fast coral (unitless)
  PARAMETER(log_beta_slow); // log selectivity for slow coral (unitless)
  PARAMETER(log_r_fast); // log intrinsic growth rate of fast coral (year^-1)
  PARAMETER(log_r_slow); // log intrinsic growth rate of slow coral (year^-1)
  PARAMETER(log_K_fast); // log carrying capacity for fast coral (%)
  PARAMETER(log_K_slow); // log carrying capacity for slow coral (%)
  PARAMETER(log_gamma_env); // log environmental effect scaling (unitless)
  PARAMETER(log_sigma_cots); // log SD for COTS obs (lognormal)
  PARAMETER(log_sigma_fast); // log SD for fast coral obs (lognormal)
  PARAMETER(log_sigma_slow); // log SD for slow coral obs (lognormal)
  PARAMETER(logit_phi_outbreak); // logit probability of outbreak trigger (unitless)
  PARAMETER(log_tau_outbreak); // log mean duration of outbreak (years)
  PARAMETER(log_epsilon); // log small constant for stability
  PARAMETER(log_h_cots); // log half-saturation constant for COTS predation (indiv/m2)
  PARAMETER(log_kappa_coral); // log half-saturation constant for coral limitation on COTS growth
  PARAMETER(log_m_cots); // log maximum COTS mortality rate when coral is absent

  // --- TRANSFORM PARAMETERS ---
  Type r_cots = exp(log_r_cots); // COTS intrinsic growth rate
  Type K_cots = exp(log_K_cots); // COTS carrying capacity
  Type alpha_cots = exp(log_alpha_cots); // COTS predation rate
  Type beta_fast = exp(log_beta_fast); // Selectivity for fast coral
  Type beta_slow = exp(log_beta_slow); // Selectivity for slow coral
  Type r_fast = exp(log_r_fast); // Fast coral growth rate
  Type r_slow = exp(log_r_slow); // Slow coral growth rate
  Type K_fast = exp(log_K_fast); // Fast coral carrying capacity
  Type K_slow = exp(log_K_slow); // Slow coral carrying capacity
  Type gamma_env = exp(log_gamma_env); // Environmental effect scaling
  Type sigma_cots = exp(log_sigma_cots); // SD for COTS obs
  Type sigma_fast = exp(log_sigma_fast); // SD for fast coral obs
  Type sigma_slow = exp(log_sigma_slow); // SD for slow coral obs
  Type phi_outbreak = Type(1)/(Type(1)+exp(-logit_phi_outbreak)); // Outbreak trigger probability
  Type tau_outbreak = exp(log_tau_outbreak); // Mean outbreak duration
  Type epsilon = exp(log_epsilon); // Small constant for stability
  Type h_cots = exp(log_h_cots); // Half-saturation constant for COTS predation
  Type kappa_coral = exp(log_kappa_coral); // Half-saturation for coral limitation on COTS growth
  Type m_cots = exp(log_m_cots); // Maximum COTS mortality rate when coral is absent

  // --- STATE VARIABLES ---
  vector<Type> cots_pred(n); // Predicted COTS abundance
  vector<Type> fast_pred(n); // Predicted fast coral cover
  vector<Type> slow_pred(n); // Predicted slow coral cover

  // --- INITIAL CONDITIONS ---
  cots_pred(0) = cots_dat(0); // Start at observed value
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // --- OUTBREAK STATE ---
  vector<Type> outbreak_state(n); // 1 if outbreak, 0 otherwise
  outbreak_state.setZero();

  // --- PROCESS MODEL ---
  for(int t=1; t<n; t++) {
    // 1. Outbreak trigger (probabilistic, smooth)
    Type env_trigger = gamma_env * cotsimm_dat(t-1) * exp(sst_dat(t-1)-28.0); // Environmental forcing
    Type p_outbreak = phi_outbreak * (Type(1) - exp(-env_trigger)); // Smooth outbreak probability
    outbreak_state(t) = outbreak_state(t-1) * exp(-Type(1)/tau_outbreak) + p_outbreak * (Type(1) - outbreak_state(t-1));

    // 2. COTS population dynamics (resource-limited, outbreak-driven)
    Type coral_avail = beta_fast * fast_pred(t-1) + beta_slow * slow_pred(t-1) + epsilon; // Weighted coral resource
    // COTS density-dependent predation efficiency (Holling Type II functional response)
    Type predation_eff = cots_pred(t-1) / (cots_pred(t-1) + h_cots + epsilon); // Efficiency saturates at high COTS
    Type predation = alpha_cots * predation_eff * coral_avail / (coral_avail + Type(1)); // Modified saturating predation
    Type immigration = cotsimm_dat(t-1); // External larval input
    // Logistic growth with saturating coral limitation
    Type coral_limitation = coral_avail / (coral_avail + kappa_coral);
    Type growth = r_cots * cots_pred(t-1) * (Type(1) - cots_pred(t-1)/K_cots) * coral_limitation; // Logistic growth limited by coral
    Type outbreak_boost = outbreak_state(t); // Outbreak amplifies growth
    // Direct coral-dependent mortality: increases as coral_avail declines
    Type coral_mortality = m_cots * cots_pred(t-1) * (Type(1) - coral_limitation);

    cots_pred(t) = cots_pred(t-1)
      + growth * outbreak_boost // Outbreak-driven growth, now limited by coral
      + immigration // Larval input
      - predation // Loss from resource limitation
      - coral_mortality // Direct mortality when coral is scarce
      ;
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), epsilon, cots_pred(t), epsilon); // Bound away from zero

    // 3. Coral dynamics (predation, growth, competition)
    Type predation_fast = alpha_cots * predation_eff * beta_fast * fast_pred(t-1) / (coral_avail + Type(1));
    Type predation_slow = alpha_cots * predation_eff * beta_slow * slow_pred(t-1) / (coral_avail + Type(1));

    fast_pred(t) = fast_pred(t-1)
      + r_fast * fast_pred(t-1) * (Type(1) - (fast_pred(t-1)+slow_pred(t-1))/K_fast) // Logistic growth
      - predation_fast // COTS predation
      ;
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), epsilon, fast_pred(t), epsilon);

    slow_pred(t) = slow_pred(t-1)
      + r_slow * slow_pred(t-1) * (Type(1) - (fast_pred(t-1)+slow_pred(t-1))/K_slow) // Logistic growth
      - predation_slow // COTS predation
      ;
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), epsilon, slow_pred(t), epsilon);
  }

  // --- LIKELIHOOD ---
  Type nll = 0.0;
  for(int t=0; t<n; t++) {
    // Lognormal likelihood for strictly positive data
    nll -= dnorm(log(cots_dat(t)+epsilon), log(cots_pred(t)+epsilon), sigma_cots + Type(1e-3), true);
    nll -= dnorm(log(fast_dat(t)+epsilon), log(fast_pred(t)+epsilon), sigma_fast + Type(1e-3), true);
    nll -= dnorm(log(slow_dat(t)+epsilon), log(slow_pred(t)+epsilon), sigma_slow + Type(1e-3), true);
  }

  // --- SMOOTH PENALTIES FOR PARAMETER BOUNDS ---
  // Example: penalize negative or unreasonably high rates
  nll += pow(CppAD::CondExpLt(r_cots, Type(1e-4), r_cots-Type(1e-4), Type(0)), 2);
  nll += pow(CppAD::CondExpGt(r_cots, Type(10), r_cots-Type(10), Type(0)), 2);
  nll += pow(CppAD::CondExpLt(K_cots, Type(1e-3), K_cots-Type(1e-3), Type(0)), 2);
  nll += pow(CppAD::CondExpGt(K_cots, Type(100), K_cots-Type(100), Type(0)), 2);

  // --- REPORTING ---
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(outbreak_state);

  return nll;
}

/*
MODEL EQUATION DESCRIPTIONS:
1. Outbreak trigger: Outbreaks are probabilistically triggered by environmental forcing (larval input, temperature), with smooth transitions and mean duration tau_outbreak.
2. COTS population: Logistic growth, resource-limited by coral, amplified during outbreaks, with external larval immigration and saturating predation loss.
3. Coral dynamics: Logistic growth for each coral group, reduced by selective COTS predation, with competition for space.
4. Likelihood: Lognormal error for all observed variables, with fixed minimum SD for stability.
5. Parameter bounds: Smooth penalties for biologically implausible parameter values.
6. Direct coral-dependent COTS mortality: COTS mortality increases as coral availability declines, capturing starvation and predation risk after outbreaks.
*/
