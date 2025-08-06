#include <TMB.hpp>

// 1. Model equations describe:
//    (1) COTS population dynamics with episodic outbreaks driven by larval immigration, density dependence, and resource limitation.
//    (2) Selective predation by COTS on fast- and slow-growing corals.
//    (3) Coral recovery via growth, subject to competition and environmental modification.
//    (4) Environmental modulation of key rates (e.g., temperature effects).
//    (5) Feedbacks between COTS and coral cover, with smooth transitions and robust numerical stability.

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA INPUTS ---
  DATA_VECTOR(Year); // Time variable (years)
  DATA_VECTOR(sst_dat); // Sea-surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (indiv/m2/yr)
  DATA_VECTOR(cots_dat); // Observed adult COTS abundance (indiv/m2)
  DATA_VECTOR(fast_dat); // Observed fast coral cover (%)
  DATA_VECTOR(slow_dat); // Observed slow coral cover (%)

  int n = Year.size();

  // --- PARAMETERS ---
  PARAMETER(log_r_cots); // log intrinsic COTS growth rate (log(yr^-1))
  // Determines the baseline rate at which COTS population increases in absence of limitation
  PARAMETER(log_K_cots); // log COTS carrying capacity (log indiv/m2)
  // Maximum sustainable COTS density, set by resource limitation
  PARAMETER(log_alpha_cots); // log COTS density-dependence strength (log m2/indiv)
  // Controls strength of density-dependent feedback on COTS
  PARAMETER(log_beta_cots); // log resource limitation half-saturation (log % coral cover)
  // Coral cover at which COTS growth is half-maximal
  PARAMETER(log_effic_pred_fast); // log predation efficiency on fast coral (log %/indiv)
  // Per capita COTS predation rate on fast coral
  PARAMETER(log_effic_pred_slow); // log predation efficiency on slow coral (log %/indiv)
  // Per capita COTS predation rate on slow coral
  PARAMETER(log_r_fast); // log intrinsic growth rate of fast coral (log %/yr)
  // Baseline growth rate of fast coral
  PARAMETER(log_r_slow); // log intrinsic growth rate of slow coral (log %/yr)
  // Baseline growth rate of slow coral
  PARAMETER(log_competition); // log competition coefficient (log unitless)
  // Strength of interspecific competition between coral types
  PARAMETER(log_temp_effect); // log temperature effect scaling (log unitless)
  // Sensitivity of rates to temperature anomalies
  PARAMETER(log_sigma_cots); // log obs error SD for COTS (log indiv/m2)
  PARAMETER(log_sigma_fast); // log obs error SD for fast coral (log %)
  PARAMETER(log_sigma_slow); // log obs error SD for slow coral (log %)

  // --- TRANSFORM PARAMETERS TO NATURAL SCALE ---
  Type r_cots = exp(log_r_cots); // yr^-1
  Type K_cots = exp(log_K_cots); // indiv/m2
  Type alpha_cots = exp(log_alpha_cots); // m2/indiv
  Type beta_cots = exp(log_beta_cots); // % coral cover
  Type effic_pred_fast = exp(log_effic_pred_fast); // %/indiv
  Type effic_pred_slow = exp(log_effic_pred_slow); // %/indiv
  Type r_fast = exp(log_r_fast); // %/yr
  Type r_slow = exp(log_r_slow); // %/yr
  Type competition = exp(log_competition); // unitless
  Type temp_effect = exp(log_temp_effect); // unitless
  Type sigma_cots = exp(log_sigma_cots); // indiv/m2
  Type sigma_fast = exp(log_sigma_fast); // %
  Type sigma_slow = exp(log_sigma_slow); // %

  // --- SMALL CONSTANTS FOR NUMERICAL STABILITY ---
  Type eps = Type(1e-8);

  // --- INITIAL CONDITIONS ---
  Type cots_prev = (cots_dat.size() > 0) ? CppAD::CondExpGt(cots_dat(0), eps, cots_dat(0), eps) : eps; // indiv/m2
  Type fast_prev = (fast_dat.size() > 0) ? CppAD::CondExpGt(fast_dat(0), eps, fast_dat(0), eps) : eps; // %
  Type slow_prev = (slow_dat.size() > 0) ? CppAD::CondExpGt(slow_dat(0), eps, slow_dat(0), eps) : eps; // %

  // --- OUTPUT VECTORS ---
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // --- PROCESS MODEL ---
  // Compute mean SST for anomaly calculation
  Type sst_sum = 0.0;
  for(int i=0; i<n; i++) sst_sum += sst_dat(i);
  Type sst_mean = sst_sum / Type(n);

  for(int t=0; t<n; t++) {
    // 1. Temperature anomaly (centered on mean)
    Type temp_anom = sst_dat(t) - sst_mean;

    // 2. Resource limitation: total coral cover
    Type coral_total_prev = fast_prev + slow_prev + eps;

    // 3. COTS population dynamics (boom-bust cycles)
    //    - Immigration, density dependence, resource limitation, temp effect
    Type resource_lim = coral_total_prev / (coral_total_prev + beta_cots); // saturating function
    Type temp_mod = 1.0 + temp_effect * temp_anom; // smooth temp effect
    Type density_feedback = 1.0 / (1.0 + alpha_cots * cots_prev); // smooth density dependence

    Type cots_growth = r_cots * cots_prev * resource_lim * density_feedback * temp_mod;
    Type cots_next = cots_prev + cots_growth + cotsimm_dat(t); // immigration as additive pulse

    // Prevent negative or zero COTS
    cots_next = CppAD::CondExpGt(cots_next, eps, cots_next, eps);

    // 4. Coral predation by COTS (selective, saturating)
    Type pred_fast = effic_pred_fast * cots_prev * fast_prev / (fast_prev + beta_cots + eps);
    Type pred_slow = effic_pred_slow * cots_prev * slow_prev / (slow_prev + beta_cots + eps);

    // 5. Fast coral dynamics: growth, competition, predation
    Type comp_fast = competition * slow_prev / (fast_prev + slow_prev + eps);
    Type fast_growth = r_fast * fast_prev * (1.0 - comp_fast);
    Type fast_next = fast_prev + fast_growth - pred_fast;
    fast_next = CppAD::CondExpGt(fast_next, eps, fast_next, eps);

    // 6. Slow coral dynamics: growth, competition, predation
    Type comp_slow = competition * fast_prev / (fast_prev + slow_prev + eps);
    Type slow_growth = r_slow * slow_prev * (1.0 - comp_slow);
    Type slow_next = slow_prev + slow_growth - pred_slow;
    slow_next = CppAD::CondExpGt(slow_next, eps, slow_next, eps);

    // --- SAVE PREDICTIONS ---
    cots_pred(t) = cots_next;
    fast_pred(t) = fast_next;
    slow_pred(t) = slow_next;

    // --- UPDATE FOR NEXT TIME STEP ---
    cots_prev = cots_next;
    fast_prev = fast_next;
    slow_prev = slow_next;
  }

  // --- LIKELIHOOD CALCULATION ---
  Type nll = 0.0;
  for(int t=0; t<n; t++) {
    // Use lognormal likelihood for strictly positive data, with minimum SD for stability
    if(t < cots_dat.size())
      nll -= dnorm(log(CppAD::CondExpGt(cots_dat(t), eps, cots_dat(t), eps)), log(cots_pred(t) + eps), sigma_cots + eps, true);
    if(t < fast_dat.size())
      nll -= dnorm(log(CppAD::CondExpGt(fast_dat(t), eps, fast_dat(t), eps)), log(fast_pred(t) + eps), sigma_fast + eps, true);
    if(t < slow_dat.size())
      nll -= dnorm(log(CppAD::CondExpGt(slow_dat(t), eps, slow_dat(t), eps)), log(slow_pred(t) + eps), sigma_slow + eps, true);
  }

  // --- REPORTING ---
  REPORT(cots_pred); // Predicted COTS abundance (indiv/m2)
  REPORT(fast_pred); // Predicted fast coral cover (%)
  REPORT(slow_pred); // Predicted slow coral cover (%)

  return nll;
}
