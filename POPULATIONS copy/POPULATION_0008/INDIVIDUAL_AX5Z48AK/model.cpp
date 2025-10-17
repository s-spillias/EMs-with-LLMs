#include <TMB.hpp>

// Template Model Builder model for COTS boom-bust dynamics and coral impacts
// Uses forcing: Year, sst_dat, cotsimm_dat
// Predicts: cots_pred (indiv m^-2), fast_pred (% cover), slow_pred (% cover)
// Observations: cots_dat, fast_dat, slow_dat (matched names, lognormal likelihood)

// Helper functions with small constants for stability
template<class Type>
Type inv_logit(Type x) { // Smooth logistic function
  return Type(1.0) / (Type(1.0) + exp(-x));
}

template<class Type>
Type square(Type x) { return x * x; }

template<class Type>
Type max_floor(Type x, Type m) { return CppAD::CondExpGt(x, m, x, m); } // Smooth enough floor via conditional

template<class Type>
Type min_ceiling(Type x, Type M) { return CppAD::CondExpLt(x, M, x, M); } // Smooth enough cap via conditional

// TMB objective function
template<class Type>
Type objective_function<Type>::operator() ()
{
  // ----------------------------
  // DATA
  // ----------------------------
  DATA_VECTOR(Year);        // Year vector (calendar year), used to align time steps
  DATA_VECTOR(sst_dat);     // Sea-surface temperature (C)
  DATA_VECTOR(cotsimm_dat); // Larval immigration (indiv m^-2 yr^-1)
  DATA_VECTOR(cots_dat);    // Observed adult COTS density (indiv m^-2)
  DATA_VECTOR(fast_dat);    // Observed fast coral cover (%) (Acropora)
  DATA_VECTOR(slow_dat);    // Observed slow coral cover (%) (Faviidae, Porites)

  // ----------------------------
  // PARAMETERS (unconstrained; soft bounds applied via penalties)
  // ----------------------------
  PARAMETER(r_fast);         // Intrinsic growth rate of fast coral (yr^-1); estimated from time series or literature priors
  PARAMETER(r_slow);         // Intrinsic growth rate of slow coral (yr^-1); estimated from time series or literature priors
  PARAMETER(K_coral);        // Coral community carrying capacity (percent cover, %); typical 60–90%
  PARAMETER(g_max);          // Max per-capita COTS grazing capacity (percent cover per indiv per yr); scales grazing intensity
  PARAMETER(K_prey);         // Half-saturation constant for prey index in COTS processes (%, cover); smooth resource limitation
  PARAMETER(pref_fast);      // Preference/weight for fast coral in prey index (dimensionless, >0); higher => more selective predation
  PARAMETER(pref_slow);      // Preference/weight for slow coral in prey index (dimensionless, >0)
  PARAMETER(s0_cots);        // Baseline annual survival probability of adult COTS in prey-replete conditions (0–1)
  PARAMETER(theta_surv);     // Shape parameter for prey effect on COTS survival (dimensionless, >=0); captures nonlinearity
  PARAMETER(r0_recruit);     // Local per-adult annual recruitment to adult class (indiv per indiv per yr); fecundity*survival to adult
  PARAMETER(alpha_imm);      // Conversion of larval immigration to adult recruitment (indiv adult per indiv larva per yr)
  PARAMETER(kc_carry);       // Scaling linking prey to COTS carrying capacity (indiv m^-2 per % prey); regulates outbreak saturation
  PARAMETER(Topt_cots);      // COTS reproduction optimal temperature (C)
  PARAMETER(sigmaT_cots);    // Width of thermal performance curve for COTS reproduction (C)
  PARAMETER(Topt_coral);     // Coral growth optimal temperature (C)
  PARAMETER(sigmaT_coral);   // Width of thermal performance curve for coral growth (C)
  PARAMETER(T_bleach);       // Bleaching onset temperature (C) for logistic bleaching response
  PARAMETER(k_bleach);       // Steepness of bleaching logistic vs temperature (C^-1)
  PARAMETER(m_bleach_fast);  // Bleaching mortality coefficient for fast coral (yr^-1 equivalent as fractional loss parameter)
  PARAMETER(m_bleach_slow);  // Bleaching mortality coefficient for slow coral (yr^-1 equivalent as fractional loss parameter)
  PARAMETER(k_allee);        // Steepness of smooth Allee threshold on COTS recruitment (indiv m^-2)^-1
  PARAMETER(c50_allee);      // Adult COTS density at 50% of Allee effect (indiv m^-2)
  PARAMETER(sd_log_cots);    // Observation SD on log-scale for COTS (dimensionless); lognormal error
  PARAMETER(sd_log_fast);    // Observation SD on log-scale for fast coral (dimensionless); lognormal error
  PARAMETER(sd_log_slow);    // Observation SD on log-scale for slow coral (dimensionless); lognormal error
  PARAMETER(m_heat_cots);    // New: Heat-stress survival penalty for adult COTS (yr^-1 equivalent coefficient)

  // ----------------------------
  // CONSTANTS AND STABILITY GUARDS
  // ----------------------------
  int n = Year.size();                                   // Number of time steps (years)
  Type eps = Type(1e-8);                                 // Small constant to avoid division by zero
  Type sd_min = Type(0.05);                              // Minimum observation SD on log scale for numerical stability
  Type sd_cots = max_floor(sd_log_cots, sd_min);         // Enforce minimum SD for COTS
  Type sd_fast = max_floor(sd_log_fast, sd_min);         // Enforce minimum SD for fast coral
  Type sd_slow = max_floor(sd_log_slow, sd_min);         // Enforce minimum SD for slow coral

  // Internal fixed penalty weight to avoid external DATA_SCALAR dependency
  Type penalty_weight = Type(1.0); // Dimensionless weight for soft parameter-range penalties

  // ----------------------------
  // SOFT PARAMETER BOUNDS (smooth penalties; zero inside range, quadratic outside)
  // These suggested plausible ranges reflect ecological priors.
  // ----------------------------
  Type nll = 0.0;                                        // Negative log-likelihood accumulator
  // Suggested ranges:
  // r_fast [0.0, 1.5], r_slow [0.0, 0.8], K_coral [30, 100], g_max [0.0, 5.0], K_prey [1.0, 60.0]
  // pref_fast [0.1, 10], pref_slow [0.1, 10], s0_cots [0.1, 0.99], theta_surv [0.0, 4.0]
  // r0_recruit [0.0, 5.0], alpha_imm [0.0, 5.0], kc_carry [0.0, 1.0]
  // Topt_cots [20, 33], sigmaT_cots [0.5, 6], Topt_coral [28, 31], sigmaT_coral [0.5, 6]
  // T_bleach [27, 34], k_bleach [0.1, 5], m_bleach_fast [0.0, 2.0], m_bleach_slow [0.0, 2.0]
  // k_allee [0.0, 20.0], c50_allee [0.0, 3.0], m_heat_cots [0.0, 2.0]
  auto penalize_range = [&](Type x, Type L, Type U) {
    Type pen = Type(0.0);
    pen += CppAD::CondExpLt(x, L, square(L - x), Type(0.0));
    pen += CppAD::CondExpGt(x, U, square(x - U), Type(0.0));
    return pen;
  };
  nll += penalty_weight * (
    penalize_range(r_fast,       Type(0.0),  Type(1.5)) +
    penalize_range(r_slow,       Type(0.0),  Type(0.8)) +
    penalize_range(K_coral,      Type(30.0), Type(100.0)) +
    penalize_range(g_max,        Type(0.0),  Type(5.0)) +
    penalize_range(K_prey,       Type(1.0),  Type(60.0)) +
    penalize_range(pref_fast,    Type(0.1),  Type(10.0)) +
    penalize_range(pref_slow,    Type(0.1),  Type(10.0)) +
    penalize_range(s0_cots,      Type(0.1),  Type(0.99)) +
    penalize_range(theta_surv,   Type(0.0),  Type(4.0)) +
    penalize_range(r0_recruit,   Type(0.0),  Type(5.0)) +
    penalize_range(alpha_imm,    Type(0.0),  Type(5.0)) +
    penalize_range(kc_carry,     Type(0.0),  Type(1.0)) +
    penalize_range(Topt_cots,    Type(20.0), Type(33.0)) +
    penalize_range(sigmaT_cots,  Type(0.5),  Type(6.0)) +
    penalize_range(Topt_coral,   Type(28.0), Type(31.0)) + // tightened to reflect parameters.json updates
    penalize_range(sigmaT_coral, Type(0.5),  Type(6.0)) +
    penalize_range(T_bleach,     Type(27.0), Type(34.0)) +
    penalize_range(k_bleach,     Type(0.1),  Type(5.0)) +
    penalize_range(m_bleach_fast,Type(0.0),  Type(2.0)) +
    penalize_range(m_bleach_slow,Type(0.0),  Type(2.0)) +
    penalize_range(k_allee,      Type(0.0),  Type(20.0)) +
    penalize_range(c50_allee,    Type(0.0),  Type(3.0)) +
    penalize_range(m_heat_cots,  Type(0.0),  Type(2.0))
  );

  // ----------------------------
  // STATE VECTORS FOR PREDICTIONS
  // ----------------------------
  vector<Type> cots_pred(n);  // Predicted adult COTS (indiv m^-2)
  vector<Type> fast_pred(n);  // Predicted fast coral cover (%)
  vector<Type> slow_pred(n);  // Predicted slow coral cover (%)

  // INITIAL CONDITIONS: Set predictions to first observed values (no data leakage in transitions)
  cots_pred(0) = cots_dat(0); // Initialize from data at t=0
  fast_pred(0) = fast_dat(0); // Initialize from data at t=0
  slow_pred(0) = slow_dat(0); // Initialize from data at t=0

  // ----------------------------
  // DOCUMENTATION OF DYNAMICS (all updates use lagged predictions at t-1)
  // 1) Prey index P_t-1 = pref_fast * fast_t-1 + pref_slow * slow_t-1 (weighted coral availability)
  // 2) Environmental modifiers:
  //    a) temp_repro = exp(-0.5 * ((sst_t-1 - Topt_cots)/sigmaT_cots)^2) for COTS reproduction
  //    b) temp_growth = exp(-0.5 * ((sst_t-1 - Topt_coral)/sigmaT_coral)^2) for coral growth
  //    c) bleach_level = logistic(sst_t-1; T_bleach, k_bleach) for bleaching stress
  // 3) COTS processes:
  //    a) Survival_t = s0_cots * (P/(K_prey + P))^theta_surv * exp(-m_heat_cots * bleach_level)  [0..1]
  //    b) Local recruitment = r0_recruit * COTS_{t-1} * temp_repro * P/(K_prey + P)
  //    c) Allee recruitment multiplier = inv_logit(k_allee*(COTS_{t-1} - c50_allee))
  //    d) Immigration recruits = alpha_imm * cotsimm_{t-1}
  //    e) Density regulation on recruits = exp(- COTS_{t-1} / (kc_carry * P + eps))
  //    f) COTS_{t} = Survival_t * COTS_{t-1} + (Local + Immigration) * Allee * DensityReg
  // 4) Coral processes:
  //    a) Space-limited growth: r_* * coral_{t-1} * (1 - (fast_{t-1}+slow_{t-1})/K_coral) * temp_growth
  //    b) Multi-prey Type-II grazing:
  //       - Total grazing potential per COTS: G = g_max * P/(K_prey + P)
  //       - Preference-weighted availability share_f = pref_fast*fast / (pref_fast*fast + pref_slow*slow + eps)
  //       - Fraction grazed = 1 - exp(- G * COTS / coral_{t-1}), applied separately to fast and slow
  //    c) Bleaching fractional mortality: 1 - exp(- m_bleach_* * bleach_level)
  //    d) Coral_{t} = Coral_{t-1} + Growth - GrazingLoss - BleachLoss
  // ----------------------------

  // Time loop
  for (int t = 1; t < n; t++) {
    // Lagged state variables (avoid any use of current observations; no data leakage)
    Type c_prev = cots_pred(t-1);     // Previous COTS density
    Type f_prev = fast_pred(t-1);     // Previous fast coral cover
    Type s_prev = slow_pred(t-1);     // Previous slow coral cover

    // Forcing at t-1
    Type sst_prev = sst_dat(t-1);     // Temperature at previous year
    Type imm_prev = cotsimm_dat(t-1); // Immigration at previous year

    // 1) Prey index (weighted coral availability) with stability guard
    Type P = pref_fast * f_prev + pref_slow * s_prev + eps; // Weighted sum; eps avoids zero

    // 2) Environmental modifiers
    Type temp_repro = exp( Type(-0.5) * square((sst_prev - Topt_cots) / (sigmaT_cots + eps)) );   // COTS reproduction modifier
    Type temp_growth = exp( Type(-0.5) * square((sst_prev - Topt_coral) / (sigmaT_coral + eps)) );// Coral growth modifier
    Type bleach_level = inv_logit( k_bleach * (sst_prev - T_bleach) );                            // Bleaching stress [0..1]

    // 3) COTS dynamics
    // a) Survival (bounded in [0,1]) with prey effect and heat-stress penalty
    Type surv_preydrv = s0_cots * pow(P / (K_prey + P), theta_surv); // Higher prey => higher survival
    Type surv_env = exp( - m_heat_cots * bleach_level );             // Heat stress reduces survival smoothly
    Type surv = surv_preydrv * surv_env;
    surv = min_ceiling(max_floor(surv, Type(0.0)), Type(1.0));       // Numerical guard: keep within [0,1] softly

    // b) Local recruitment (fecundity * environment * prey limitation)
    Type local_recr = r0_recruit * c_prev * temp_repro * (P / (K_prey + P)); // indiv m^-2 yr^-1

    // c) Smooth Allee effect multiplier (0..1)
    Type allee = inv_logit( k_allee * (c_prev - c50_allee) );

    // d) Immigration recruits
    Type imm_recr = alpha_imm * imm_prev; // indiv m^-2 yr^-1

    // e) Density regulation acting on the sum of recruits, via prey-dependent carrying capacity
    Type K_cots_eff = kc_carry * P + eps;            // Effective carrying capacity scaling with prey (indiv m^-2)
    Type dens_reg = exp( - c_prev / (K_cots_eff) );  // Beverton-Ricker type saturation on recruitment

    // f) Update COTS
    Type c_next = surv * c_prev + (local_recr + imm_recr) * allee * dens_reg + eps; // Keep strictly positive

    // 4) Coral dynamics
    // a) Space-limited growth (logistic) with temperature modulation
    Type total_cover_prev = f_prev + s_prev + eps;                // Total coral cover (%)
    Type space_term = (Type(1.0) - total_cover_prev / (K_coral + eps)); // Space availability (can be negative if over 100%)
    Type growth_fast = r_fast * f_prev * space_term * temp_growth;      // Growth of fast coral (% per yr)
    Type growth_slow = r_slow * s_prev * space_term * temp_growth;      // Growth of slow coral (% per yr)

    // b) Multi-prey Holling type II grazing with preference-weighted allocation
    Type G = g_max * (P / (K_prey + P)); // Total grazing potential per COTS (fractional per-year of available coral)
    // Preference-weighted availability shares
    Type denom_pref = pref_fast * f_prev + pref_slow * s_prev + eps;
    Type share_fast = (pref_fast * f_prev + eps) / denom_pref; // Fraction of grazing directed to fast coral
    Type share_slow = (pref_slow * s_prev + eps) / denom_pref; // Fraction to slow coral
    // Fractional losses (bounded < 1) using saturation to ensure losses do not exceed available cover
    Type frac_grazed_fast = Type(1.0) - exp( - (G * c_prev * share_fast) / (f_prev + eps) );
    Type frac_grazed_slow = Type(1.0) - exp( - (G * c_prev * share_slow)  / (s_prev + eps) );
    Type loss_graz_fast = f_prev * frac_grazed_fast; // % cover lost to COTS grazing (fast)
    Type loss_graz_slow = s_prev * frac_grazed_slow; // % cover lost to COTS grazing (slow)

    // c) Bleaching fractional mortalities (smooth logistic vs temperature, capped via exp form)
    Type frac_bleach_fast = Type(1.0) - exp( - m_bleach_fast * bleach_level );
    Type frac_bleach_slow = Type(1.0) - exp( - m_bleach_slow * bleach_level );
    Type loss_bleach_fast = f_prev * frac_bleach_fast; // % cover lost to bleaching (fast)
    Type loss_bleach_slow = s_prev * frac_bleach_slow; // % cover lost to bleaching (slow)

    // d) Update corals (ensure they remain non-negative through likelihood handling and eps guards)
    Type f_next = f_prev + growth_fast - loss_graz_fast - loss_bleach_fast;
    Type s_next = s_prev + growth_slow - loss_graz_slow - loss_bleach_slow;

    // Assign predictions
    cots_pred(t) = c_next; // Predicted COTS at time t
    fast_pred(t) = f_next; // Predicted fast coral at time t
    slow_pred(t) = s_next; // Predicted slow coral at time t
  }

  // ----------------------------
  // LIKELIHOOD: Lognormal observation model for strictly positive quantities
  // Apply to all time points including t=0 (no skipping)
  // ----------------------------
  for (int t = 0; t < n; t++) {
    // Stability guards for logs
    Type c_pred_pos = cots_pred(t) + eps;
    Type f_pred_pos = fast_pred(t) + eps;
    Type s_pred_pos = slow_pred(t) + eps;

    Type c_obs_pos = cots_dat(t) + eps;
    Type f_obs_pos = fast_dat(t) + eps;
    Type s_obs_pos = slow_dat(t) + eps;

    nll -= dnorm(log(c_obs_pos), log(c_pred_pos), sd_cots, true); // COTS likelihood
    nll -= dnorm(log(f_obs_pos), log(f_pred_pos), sd_fast, true); // Fast coral likelihood
    nll -= dnorm(log(s_obs_pos), log(s_pred_pos), sd_slow, true); // Slow coral likelihood
  }

  // ----------------------------
  // REPORTING
  // ----------------------------
  REPORT(cots_pred); // Prediction vector for COTS
  REPORT(fast_pred); // Prediction vector for fast coral
  REPORT(slow_pred); // Prediction vector for slow coral

  return nll;
}
