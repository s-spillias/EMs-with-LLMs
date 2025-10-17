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
Type max_floor(Type x, Type m) { return CppAD::CondExpGt(x, m, x, m); } // max(x, m)

template<class Type>
Type min_ceiling(Type x, Type M) { return CppAD::CondExpLt(x, M, x, M); } // min(x, M)

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
  PARAMETER(r_fast);         // Intrinsic growth rate of fast coral (yr^-1)
  PARAMETER(r_slow);         // Intrinsic growth rate of slow coral (yr^-1)
  PARAMETER(K_coral);        // Coral community carrying capacity (% cover)
  PARAMETER(g_max);          // Max per-capita COTS grazing capacity (% cover indiv^-1 yr^-1)
  PARAMETER(K_prey);         // Half-saturation constant for prey index (% cover)
  PARAMETER(pref_fast);      // Preference weight for fast coral (dimensionless)
  PARAMETER(pref_slow);      // Preference weight for slow coral (dimensionless)
  PARAMETER(s0_cots);        // Baseline annual survival probability of adult COTS
  PARAMETER(theta_surv);     // Shape parameter for prey effect on COTS survival
  PARAMETER(theta_repr);     // Shape parameter for prey effect on local recruitment
  PARAMETER(r0_recruit);     // Local per-adult annual recruitment to adult class
  PARAMETER(alpha_imm);      // Conversion of larval immigration to adult recruits
  PARAMETER(kc_carry);       // Scaling linking prey to COTS carrying capacity (indiv m^-2 per % prey)
  PARAMETER(Topt_cots);      // COTS reproduction optimal temperature (C)
  PARAMETER(sigmaT_cots);    // Width of thermal performance curve for COTS reproduction (C)
  PARAMETER(Topt_coral);     // Coral growth optimal temperature (C)
  PARAMETER(sigmaT_coral);   // Width of thermal performance curve for coral growth (C)
  PARAMETER(T_bleach);       // Bleaching onset temperature (C)
  PARAMETER(k_bleach);       // Steepness of bleaching logistic vs temperature (C^-1)
  PARAMETER(m_bleach_fast);  // Bleaching mortality coefficient for fast coral
  PARAMETER(m_bleach_slow);  // Bleaching mortality coefficient for slow coral
  PARAMETER(k_allee);        // Steepness of smooth Allee threshold on COTS recruitment
  PARAMETER(c50_allee);      // Adult COTS density at 50% of Allee effect
  PARAMETER(sd_log_cots);    // Observation SD on log-scale for COTS
  PARAMETER(sd_log_fast);    // Observation SD on log-scale for fast coral
  PARAMETER(sd_log_slow);    // Observation SD on log-scale for slow coral
  PARAMETER(m_heat_cots);    // Heat-stress survival penalty for adult COTS

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
  // pref_fast [0.1, 10], pref_slow [0.1, 10], s0_cots [0.1, 0.99], theta_surv [0.0, 4.0], theta_repr [0.0, 4.0]
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
    penalize_range(theta_repr,   Type(0.0),  Type(4.0)) +
    penalize_range(r0_recruit,   Type(0.0),  Type(5.0)) +
    penalize_range(alpha_imm,    Type(0.0),  Type(5.0)) +
    penalize_range(kc_carry,     Type(0.0),  Type(1.0)) +
    penalize_range(Topt_cots,    Type(20.0), Type(33.0)) +
    penalize_range(sigmaT_cots,  Type(0.5),  Type(6.0)) +
    penalize_range(Topt_coral,   Type(28.0), Type(31.0)) +
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
  // STATE VECTORS
  // ----------------------------
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Initialize states using first observation (lagged for subsequent prediction steps)
  cots_pred(0) = max_floor(cots_dat(0), eps);
  fast_pred(0) = max_floor(fast_dat(0), Type(0.0));
  slow_pred(0) = max_floor(slow_dat(0), Type(0.0));

  // ----------------------------
  // PROCESS MODEL
  // ----------------------------
  for (int t = 1; t < n; ++t) {
    // Use lagged covariates (t-1) to avoid any data leakage per instructions
    Type Tlag = sst_dat(t - 1);
    Type imm_lag = cotsimm_dat(t - 1);

    // Thermal performance modifiers (Gaussian)
    Type temp_coral = exp(-Type(0.5) * square((Tlag - Topt_coral) / (sigmaT_coral + eps)));
    Type temp_cots  = exp(-Type(0.5) * square((Tlag - Topt_cots)  / (sigmaT_cots  + eps)));

    // Bleaching index (logistic increasing with temperature)
    Type bleach = inv_logit(k_bleach * (Tlag - T_bleach));

    // Previous states
    Type Cf = max_floor(fast_pred(t - 1), Type(0.0));
    Type Cs = max_floor(slow_pred(t - 1), Type(0.0));
    Type Ct = max_floor(cots_pred(t - 1), Type(0.0));
    Type Ctot = Cf + Cs;

    // Prey index and saturating prey term
    Type prey_index = pref_fast * Cf + pref_slow * Cs; // weighted coral cover
    Type prey_term = prey_index / (K_prey + prey_index + eps); // in [0,1)

    // COTS survival (prey-limited, heat-stress penalty)
    Type s_adult = s0_cots * pow(max_floor(prey_term, Type(0.0)), theta_surv) * exp(-m_heat_cots * bleach);
    s_adult = min_ceiling(max_floor(s_adult, Type(0.0)), Type(1.0));

    // Local recruitment with prey Hill exponent and Allee effect
    Type allee = inv_logit(k_allee * (Ct - c50_allee)); // in (0,1)
    Type R_local_raw = r0_recruit * Ct * temp_cots * pow(max_floor(prey_term, Type(0.0)), theta_repr) * allee;

    // Prey-dependent Beverton-Holt saturation via effective carrying capacity
    Type Kcots_eff = kc_carry * prey_index; // indiv m^-2
    Type R_local = R_local_raw / (Type(1.0) + Ct / (Kcots_eff + eps));

    // Immigration (lagged)
    Type R_imm = alpha_imm * max_floor(imm_lag, Type(0.0));

    // Update COTS
    Type Ct_next = s_adult * Ct + R_local + R_imm;
    Ct_next = max_floor(Ct_next, Type(0.0));
    cots_pred(t) = Ct_next;

    // Grazing allocation based on preferences and availability
    Type denom_diet = pref_fast * Cf + pref_slow * Cs + eps;
    Type w_fast = (pref_fast * Cf) / denom_diet;
    Type w_slow = (pref_slow * Cs) / denom_diet;

    // Per-capita total grazing saturates with prey availability
    Type g_percap = g_max * prey_term; // % cover per indiv per yr
    Type G_total = g_percap * Ct;      // total grazing pressure (lagged Ct)

    // Allocate and cap grazing by available cover
    Type G_fast = min_ceiling(G_total * w_fast, Cf);
    Type G_slow = min_ceiling(G_total * w_slow, Cs);

    // Coral growth (temperature-modified logistic with total-coral crowding)
    Type grow_fast = r_fast * Cf * (Type(1.0) - Ctot / (K_coral + eps)) * temp_coral;
    Type grow_slow = r_slow * Cs * (Type(1.0) - Ctot / (K_coral + eps)) * temp_coral;

    // Bleaching mortality on corals (proportional to bleaching level and current cover)
    Type B_fast = m_bleach_fast * bleach * Cf;
    Type B_slow = m_bleach_slow * bleach * Cs;

    // Update corals
    Type Cf_next = Cf + grow_fast - G_fast - B_fast;
    Type Cs_next = Cs + grow_slow - G_slow - B_slow;

    // Non-negativity floors and simple upper caps (0 to 100% cover)
    Cf_next = min_ceiling(max_floor(Cf_next, Type(0.0)), Type(100.0));
    Cs_next = min_ceiling(max_floor(Cs_next, Type(0.0)), Type(100.0));

    fast_pred(t) = Cf_next;
    slow_pred(t) = Cs_next;
  }

  // ----------------------------
  // OBSERVATION MODEL (lognormal likelihood)
  // ----------------------------
  for (int t = 0; t < n; ++t) {
    // Guard logs with floors
    Type cots_obs = max_floor(cots_dat(t), eps);
    Type fast_obs = max_floor(fast_dat(t), eps);
    Type slow_obs = max_floor(slow_dat(t), eps);

    Type cots_mu = max_floor(cots_pred(t), eps);
    Type fast_mu = max_floor(fast_pred(t), eps);
    Type slow_mu = max_floor(slow_pred(t), eps);

    nll -= dnorm(log(cots_obs), log(cots_mu), sd_cots, true);
    nll -= dnorm(log(fast_obs), log(fast_mu), sd_fast, true);
    nll -= dnorm(log(slow_obs), log(slow_mu), sd_slow, true);
  }

  // ----------------------------
  // REPORTS
  // ----------------------------
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);

  return nll;
}
