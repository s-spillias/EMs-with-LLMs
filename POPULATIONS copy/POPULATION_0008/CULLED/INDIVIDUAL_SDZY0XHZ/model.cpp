#include <TMB.hpp>

// Helper: square
template<class Type>
Type sq(Type x) { return x * x; }

// Helper: stable inverse-logit
template<class Type>
Type invlogit_stable(Type x) {
  if (x > Type(35)) return Type(1);
  if (x < Type(-35)) return Type(0);
  return Type(1) / (Type(1) + exp(-x));
}

// Helper: softplus for smooth positivity (AD-safe, no log1p)
template<class Type>
Type softplus(Type x, Type k = Type(10)) {
  // Numerically stable implementation using AD-safe log/exp.
  // sp(k*x) = log(1 + exp(k*x)) / k, computed stably without log1p.
  Type y = k * x;
  Type thresh = Type(30); // switch to linear regime to avoid overflow
  Type pos_branch = y + log(Type(1) + exp(-y)); // for moderate positive y
  Type neg_branch = log(Type(1) + exp(y));      // for y <= 0
  Type sp = CppAD::CondExpGt(y, thresh, y, CppAD::CondExpGt(y, Type(0), pos_branch, neg_branch));
  return sp / k;
}

// Helper: logit with epsilon safety
template<class Type>
Type safe_logit(Type p, Type eps = Type(1e-8)) {
  Type pe = CppAD::CondExpLt(p, eps, eps, p);                       // lower clip (smooth in AD sense)
  pe = CppAD::CondExpGt(pe, Type(1) - eps, Type(1) - eps, pe);      // upper clip
  return log((pe + eps) / (Type(1) - pe + eps));
}

// Smooth penalty to keep parameter within [L, U]
template<class Type>
Type bound_penalty(Type p, Type L, Type U, Type w, Type k = Type(5)) {
  // Penalize below L and above U using smooth softplus barriers
  Type pen_low  = sq( softplus(L - p, k) );
  Type pen_high = sq( softplus(p - U, k) );
  return w * (pen_low + pen_high);
}

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --------------------------
  // DATA (time series inputs)
  // --------------------------
  DATA_VECTOR(Year);         // Calendar year (integer year)
  DATA_VECTOR(cots_dat);     // Observed COTS density (individuals per m^2), strictly positive
  DATA_VECTOR(fast_dat);     // Observed fast-growing coral cover (%) (Acropora spp.)
  DATA_VECTOR(slow_dat);     // Observed slow-growing coral cover (%) (Faviidae + Porites)
  DATA_VECTOR(sst_dat);      // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);  // External COTS larval immigration (individuals per m^2 per year)

  int N = Year.size();       // Number of time steps (years)
  Type eps = Type(1e-8);     // Small constant for numerical stability, division/log protection

  // --------------------------
  // PARAMETERS (ecological and statistical)
  // --------------------------

  // Coral intrinsic growth (Beverton–Holt style density regulation)
  PARAMETER(rF);          // year^-1; intrinsic growth rate fast coral (Acropora); to estimate from dynamics
  PARAMETER(rS);          // year^-1; intrinsic growth rate slow coral (Faviidae/Porites); typically lower than rF
  PARAMETER(beta_space);  // (proportion^-1); strength of space competition (F+S) in Beverton–Holt denominator
  PARAMETER(K_space);     // proportion; effective maximum occupiable coral cover (0-1), i.e., free space cap

  // Background coral mortality and temperature sensitivities
  PARAMETER(dF_base);         // year^-1; background mortality fast coral (non-bleaching)
  PARAMETER(dS_base);         // year^-1; background mortality slow coral (non-bleaching)
  PARAMETER(heat_sens_F);     // dimensionless (0-1); proportional growth suppression of fast coral under heat stress
  PARAMETER(heat_sens_S);     // dimensionless (0-1); proportional growth suppression of slow coral under heat stress
  PARAMETER(T_bleach);        // deg C; SST center where bleaching risk accelerates
  PARAMETER(bleach_slope);    // (deg C)^-1; slope of bleaching logistic
  PARAMETER(m_bleach_max_F);  // year^-1; additional mortality rate at extreme heat for fast coral
  PARAMETER(m_bleach_max_S);  // year^-1; additional mortality rate at extreme heat for slow coral

  // COTS foraging (multi-prey functional response)
  PARAMETER(aF);          // year^-1; attack/encounter rate on fast coral (preference included)
  PARAMETER(aS);          // year^-1; attack/encounter rate on slow coral
  PARAMETER(hF);          // year; handling time for fast coral prey (saturates consumption)
  PARAMETER(hS);          // year; handling time for slow coral prey
  PARAMETER(q_func);      // dimensionless >=1; exponent for Type II (1) to Type III (>1) response

  // COTS demography (boom-bust processes)
  PARAMETER(rC_max);      // year^-1; maximum per-capita growth rate (fecundity potential) for COTS
  PARAMETER(mC_base);     // year^-1; baseline mortality of COTS (predation/natural)
  PARAMETER(epsilon_food);// dimensionless (0-1); efficiency converting per-capita food intake to reproduction
  PARAMETER(K_food);      // proportion; half-saturation of per-capita intake for reproduction
  PARAMETER(Kc0);         // ind m^-2; baseline carrying capacity for COTS when no coral is present
  PARAMETER(kCF);         // ind m^-2 per proportion; added carrying capacity per unit fast coral
  PARAMETER(kCS);         // ind m^-2 per proportion; added carrying capacity per unit slow coral
  PARAMETER(A50);         // ind m^-2; Allee-effect half-saturation density (mate limitation)
  PARAMETER(Topt);        // deg C; optimum SST for COTS reproductive performance
  PARAMETER(sigma_T);     // deg C; breadth (SD) of thermal performance curve
  PARAMETER(gamma_imm);   // dimensionless; scaling on external larval immigration (cotsimm_dat)
  PARAMETER(lag_recruit); // dimensionless (0-1); weight on lagged recruitment driver (total across t-1 and t-2)
  PARAMETER(lag2_share);  // dimensionless (0-1); share of lag_recruit assigned to t-2 (rest goes to t-1)

  // Observation model standard deviations (stability enforced with minimum SD)
  PARAMETER(sd_lncots);      // SD of log-observation errors for COTS (lognormal)
  PARAMETER(sd_logit_fast);  // SD of logit-observation errors for fast coral (logit-normal on proportion)
  PARAMETER(sd_logit_slow);  // SD of logit-observation errors for slow coral (logit-normal on proportion)

  // Penalty weight for keeping parameters in biologically meaningful ranges
  PARAMETER(w_pen);       // dimensionless; overall weight on smooth bound penalties

  // --------------------------
  // NUMERICAL STABILITY SETTINGS
  // --------------------------
  Type min_sd = Type(0.05); // Minimum SD added in quadrature to avoid tiny variances
  Type sd_cots_eff  = sqrt(sq(sd_lncots)     + sq(min_sd)); // smooth floor on SDs
  Type sd_fast_eff  = sqrt(sq(sd_logit_fast) + sq(min_sd));
  Type sd_slow_eff  = sqrt(sq(sd_logit_slow) + sq(min_sd));

  // --------------------------
  // STATE VECTORS (predictions)
  // --------------------------
  vector<Type> cots_pred(N); // individuals per m^2 (to match cots_dat units)
  vector<Type> fast_pred(N); // percent cover (to match fast_dat units)
  vector<Type> slow_pred(N); // percent cover (to match slow_dat units)

  // Internal state in proportions for coral cover (0-1 scale)
  vector<Type> F_state(N);   // fast coral proportion (0-1)
  vector<Type> S_state(N);   // slow coral proportion (0-1)

  // Initialize states directly from data (avoid data leakage by using only at t=0)
  cots_pred(0) = cots_dat(0);          // Initial COTS density from data
  fast_pred(0) = fast_dat(0);          // Initial fast coral (%)
  slow_pred(0) = slow_dat(0);          // Initial slow coral (%)
  F_state(0)   = fast_dat(0) / Type(100); // Convert percent to proportion for internal dynamics
  S_state(0)   = slow_dat(0) / Type(100);

  // --------------------------
  // NEGATIVE LOG-LIKELIHOOD
  // --------------------------
  Type nll = Type(0);

  // --------------------------
  // PARAMETER BOUND PENALTIES (smooth)
  // Suggested biologically plausible ranges are enforced softly (not hard constraints).
  // These numbers correspond to the recommended bounds in parameters.json.
  // --------------------------
  nll += bound_penalty(rF,              Type(0.0),   Type(2.0),   w_pen);   // fast coral growth
  nll += bound_penalty(rS,              Type(0.0),   Type(1.0),   w_pen);   // slow coral growth
  nll += bound_penalty(beta_space,      Type(0.0),   Type(20.0),  w_pen);   // space competition intensity
  nll += bound_penalty(K_space,         Type(0.2),   Type(0.95),  w_pen);   // max occupiable coral cover

  nll += bound_penalty(dF_base,         Type(0.0),   Type(0.8),   w_pen);   // fast coral natural mortality
  nll += bound_penalty(dS_base,         Type(0.0),   Type(0.6),   w_pen);   // slow coral natural mortality
  nll += bound_penalty(heat_sens_F,     Type(0.0),   Type(1.0),   w_pen);   // heat sensitivity fast coral
  nll += bound_penalty(heat_sens_S,     Type(0.0),   Type(1.0),   w_pen);   // heat sensitivity slow coral
  nll += bound_penalty(T_bleach,        Type(29.0),  Type(34.5),  w_pen);   // bleaching threshold temp (aligned to params.json)
  nll += bound_penalty(bleach_slope,    Type(0.1),   Type(5.0),   w_pen);   // bleaching slope
  nll += bound_penalty(m_bleach_max_F,  Type(0.0),   Type(1.0),   w_pen);   // max bleaching mortality (fast coral)
  nll += bound_penalty(m_bleach_max_S,  Type(0.0),   Type(1.0),   w_pen);   // max bleaching mortality (slow coral)

  nll += bound_penalty(aF,              Type(0.0),   Type(20.0),  w_pen);   // attack rate fast coral
  nll += bound_penalty(aS,              Type(0.0),   Type(20.0),  w_pen);   // attack rate slow coral
  nll += bound_penalty(hF,              Type(0.01),  Type(5.0),   w_pen);   // handling time fast coral
  nll += bound_penalty(hS,              Type(0.01),  Type(5.0),   w_pen);   // handling time slow coral
  nll += bound_penalty(q_func,          Type(1.0),   Type(3.0),   w_pen);   // functional response exponent

  nll += bound_penalty(rC_max,          Type(0.0),   Type(10.0),  w_pen);   // max COTS growth rate
  nll += bound_penalty(mC_base,         Type(0.0013),Type(2.56),  w_pen);   // COTS baseline mortality (aligned to params.json)
  nll += bound_penalty(epsilon_food,    Type(0.0),   Type(1.0),   w_pen);   // food->recruit efficiency
  nll += bound_penalty(K_food,          Type(0.01),  Type(0.8),   w_pen);   // food half-saturation
  nll += bound_penalty(Kc0,             Type(0.0),   Type(2.0),   w_pen);   // base carrying capacity
  nll += bound_penalty(kCF,             Type(0.0),   Type(50.0),  w_pen);   // carrying capacity per fast coral
  nll += bound_penalty(kCS,             Type(0.0),   Type(50.0),  w_pen);   // carrying capacity per slow coral
  nll += bound_penalty(A50,             Type(0.01),  Type(5.0),   w_pen);   // Allee half density
  nll += bound_penalty(Topt,            Type(28.0),  Type(30.0),  w_pen);   // optimal SST for COTS (aligned to params.json)
  nll += bound_penalty(sigma_T,         Type(0.5),   Type(5.0),   w_pen);   // thermal breadth
  nll += bound_penalty(gamma_imm,       Type(0.0),   Type(10.0),  w_pen);   // immigration scaler
  nll += bound_penalty(lag_recruit,     Type(0.0),   Type(1.0),   w_pen);   // total lag weight
  nll += bound_penalty(lag2_share,      Type(0.0),   Type(1.0),   w_pen);   // share to two-year lag

  nll += bound_penalty(sd_lncots,       Type(0.01),  Type(2.0),   w_pen);   // obs SD (log COTS)
  nll += bound_penalty(sd_logit_fast,   Type(0.01),  Type(2.0),   w_pen);   // obs SD (logit fast)
  nll += bound_penalty(sd_logit_slow,   Type(0.01),  Type(2.0),   w_pen);   // obs SD (logit slow)
  nll += bound_penalty(w_pen,           Type(0.001), Type(100.0), w_pen);   // penalty weight itself

  // --------------------------
  // MODEL EQUATIONS (discrete annual steps)
  // Numbered description:
  // (1) Bleaching index: B_t = logistic(bleach_slope * (SST_t - T_bleach))
  // (2) Coral predation: multi-prey Holling disk equation with exponent q:
  //     per_pred_i = a_i * P_i^q / (1 + sum_j a_j h_j P_j^q)
  // (3) Hazard-based removal: R_i = P_i * (1 - exp(-C_{t-1} * per_pred_i / (P_i + eps)))
  // (4) Coral non-predation mortality: M_i = P_i_survive * (1 - exp(-(d_i + m_bleach_max_i * B_t)))
  // (5) Coral growth (space-limited BH-like): G_i = (r_i * P_i_survive_afterMort) / (1 + beta_space * (P_F_survive + P_S_survive))
  // (6) Heat suppression of growth: G_i_eff = G_i * max(0, 1 - heat_sens_i * B_t)
  // (7) Coral update: P_i(t) = P_i_survive_afterMort + G_i_eff, then apply soft caps [eps, K_space]
  // (8) COTS intake per predator: I = per_pred_F + per_pred_S
  // (9) COTS per-capita growth modifier with distributed lag:
  //     Let X_t = I_sat_t * T_perf_t.
  //     w0 = 1 - lag_recruit; w1 = lag_recruit * (1 - lag2_share); w2 = lag_recruit * lag2_share.
  //     rC = rC_max * epsilon_food * (w0*X_t + w1*X_{t-1} + w2*X_{t-2}) * (C/(A50 + C))
  // (10) COTS density regulation: deltaC = rC * C * (1 - C / Kc), Kc = Kc0 + kCF*P_F + kCS*P_S
  // (11) COTS update: C_t = C_{t-1} + deltaC - mC_base * C_{t-1} + gamma_imm * immigration_t
  // --------------------------

  // Seed lagged food/temperature terms from t=0 (uses only initial states and SST_0)
  Type F0 = F_state(0);
  Type S0 = S_state(0);
  Type Fq0 = pow(CppAD::CondExpLt(F0, eps, eps, F0), q_func);
  Type Sq0 = pow(CppAD::CondExpLt(S0, eps, eps, S0), q_func);
  Type denom0 = Type(1.0) + aF * hF * Fq0 + aS * hS * Sq0;
  Type per_pred_F0 = aF * Fq0 / (denom0 + eps);
  Type per_pred_S0 = aS * Sq0 / (denom0 + eps);
  Type I_per_pred0 = per_pred_F0 + per_pred_S0;
  Type I_sat_prev1 = I_per_pred0 / (K_food + I_per_pred0 + eps);
  Type T_perf_prev1 = exp(-Type(0.5) * sq((sst_dat(0) - Topt) / (sigma_T + eps)));
  // Initialize second-lag buffers equal to first (no extra info at t=0)
  Type I_sat_prev2 = I_sat_prev1;
  Type T_perf_prev2 = T_perf_prev1;

  for (int t = 1; t < N; t++) {
    // Previous states (predictions only; no data leakage)
    Type C_prev = cots_pred(t-1); // COTS density at t-1 (ind m^-2)
    Type F_prev = F_state(t-1);   // Fast coral proportion at t-1
    Type S_prev = S_state(t-1);   // Slow coral proportion at t-1

    // Forcing at time t
    Type SST_t = sst_dat(t);                // Sea surface temperature at t (deg C)
    Type IMM_t = cotsimm_dat(t);            // Immigration at t (ind m^-2 yr^-1)

    // (1) Bleaching index (0-1) increasing with SST
    Type B_t = invlogit_stable(bleach_slope * (SST_t - T_bleach)); // Bleaching risk index

    // (2) Multi-prey functional response (Holling disc with exponent q)
    Type Fq = pow(CppAD::CondExpLt(F_prev, eps, eps, F_prev), q_func); // F^q, safe at 0
    Type Sq = pow(CppAD::CondExpLt(S_prev, eps, eps, S_prev), q_func); // S^q, safe at 0
    Type denom = Type(1.0) + aF * hF * Fq + aS * hS * Sq;              // Handling-limited denominator
    Type per_pred_F = aF * Fq / (denom + eps);                          // per-predator annual attack on fast coral
    Type per_pred_S = aS * Sq / (denom + eps);                          // per-predator annual attack on slow coral

    // (3) Hazard-based removal (prevents overconsumption beyond available coral)
    Type cons_F_total = C_prev * per_pred_F;                            // total fast coral consumption (proportion units per area)
    Type cons_S_total = C_prev * per_pred_S;                            // total slow coral consumption
    Type haz_F = cons_F_total / (F_prev + eps);                         // hazard of removal for fast coral
    Type haz_S = cons_S_total / (S_prev + eps);                         // hazard of removal for slow coral
    Type remF_frac = Type(1) - exp(-haz_F);                             // fraction removed from fast coral
    Type remS_frac = Type(1) - exp(-haz_S);                             // fraction removed from slow coral
    Type R_F = remF_frac * F_prev;                                      // amount of fast coral removed by predation
    Type R_S = remS_frac * S_prev;                                      // amount of slow coral removed by predation

    // Ensure non-negative after predation
    Type F_after_pred = CppAD::CondExpLt(F_prev - R_F, eps, eps, F_prev - R_F);
    Type S_after_pred = CppAD::CondExpLt(S_prev - R_S, eps, eps, S_prev - R_S);

    // (4) Coral non-predation mortality (applied to remaining coral after predation)
    Type mort_rate_F = dF_base + m_bleach_max_F * B_t;
    Type mort_rate_S = dS_base + m_bleach_max_S * B_t;
    Type M_F = F_after_pred * (Type(1) - exp(-mort_rate_F));
    Type M_S = S_after_pred * (Type(1) - exp(-mort_rate_S));
    Type F_survive = CppAD::CondExpLt(F_after_pred - M_F, eps, eps, F_after_pred - M_F);
    Type S_survive = CppAD::CondExpLt(S_after_pred - M_S, eps, eps, S_after_pred - M_S);

    // (5) Coral growth (space-limited BH-like) on survivors
    Type denom_space = Type(1) + beta_space * (F_survive + S_survive);
    Type G_F = (rF * F_survive) / (denom_space + eps);
    Type G_S = (rS * S_survive) / (denom_space + eps);

    // (6) Heat suppression of growth (0..1 multiplier)
    Type heat_fac_F = CppAD::CondExpLt(Type(1) - heat_sens_F * B_t, Type(0), Type(0), Type(1) - heat_sens_F * B_t);
    Type heat_fac_S = CppAD::CondExpLt(Type(1) - heat_sens_S * B_t, Type(0), Type(0), Type(1) - heat_sens_S * B_t);
    Type G_F_eff = G_F * heat_fac_F;
    Type G_S_eff = G_S * heat_fac_S;

    // (7) Coral update with soft caps [eps, K_space]
    Type F_t = F_survive + G_F_eff;
    Type S_t = S_survive + G_S_eff;
    // Cap at K_space and floor at eps
    F_t = CppAD::CondExpGt(F_t, K_space, K_space, F_t);
    S_t = CppAD::CondExpGt(S_t, K_space, K_space, S_t);
    F_t = CppAD::CondExpLt(F_t, eps, eps, F_t);
    S_t = CppAD::CondExpLt(S_t, eps, eps, S_t);

    // Store internal state and percent predictions
    F_state(t) = F_t;
    S_state(t) = S_t;
    fast_pred(t) = F_t * Type(100);
    slow_pred(t) = S_t * Type(100);

    // (8) COTS intake per predator and saturation for reproduction driver
    Type I_per_pred = per_pred_F + per_pred_S;
    Type I_sat_t = I_per_pred / (K_food + I_per_pred + eps);

    // (9) Temperature performance at time t and distributed-lag mix
    Type T_perf_t = exp(-Type(0.5) * sq((SST_t - Topt) / (sigma_T + eps)));
    Type w0 = Type(1) - lag_recruit;
    Type w1 = lag_recruit * (Type(1) - lag2_share);
    Type w2 = lag_recruit * lag2_share;
    Type X_t = I_sat_t * T_perf_t;
    Type X_mix = w0 * X_t + w1 * (I_sat_prev1 * T_perf_prev1) + w2 * (I_sat_prev2 * T_perf_prev2);

    // Allee effect factor (bounded and safe)
    Type allee = C_prev / (A50 + C_prev + eps);

    // Effective per-capita growth rate for COTS (non-negative)
    Type rC_eff = rC_max * epsilon_food * X_mix * allee;

    // (10) Density regulation via carrying capacity linked to coral
    Type Kc = Kc0 + kCF * F_prev + kCS * S_prev;
    Kc = CppAD::CondExpLt(Kc, eps, eps, Kc); // ensure positive

    // (11) COTS update (growth - mortality + immigration), bounded below by eps
    Type deltaC_growth = rC_eff * C_prev * (Type(1) - C_prev / (Kc + eps));
    Type C_t = C_prev + deltaC_growth - mC_base * C_prev + gamma_imm * IMM_t;
    C_t = CppAD::CondExpLt(C_t, eps, eps, C_t);

    cots_pred(t) = C_t;

    // Update lag buffers for next step (shift: prev2 <- prev1, prev1 <- current)
    I_sat_prev2 = I_sat_prev1;
    T_perf_prev2 = T_perf_prev1;
    I_sat_prev1 = I_sat_t;
    T_perf_prev1 = T_perf_t;
  }

  // --------------------------
  // OBSERVATION MODEL (likelihood)
  // --------------------------
  for (int t = 0; t < N; t++) {
    // COTS: lognormal around predicted cots_pred
    Type logy = log(cots_dat(t) + eps);
    Type logmu = log(cots_pred(t) + eps);
    nll -= dnorm(logy, logmu, sd_cots_eff, true);

    // Fast coral: logit-normal around F_state (observed given as percent)
    Type yF_prop = (fast_dat(t) / Type(100));
    Type yS_prop = (slow_dat(t) / Type(100));
    Type logit_yF = safe_logit(yF_prop);
    Type logit_yS = safe_logit(yS_prop);
    Type logit_muF = safe_logit(F_state(t));
    Type logit_muS = safe_logit(S_state(t));
    nll -= dnorm(logit_yF, logit_muF, sd_fast_eff, true);
    nll -= dnorm(logit_yS, logit_muS, sd_slow_eff, true);
  }

  // --------------------------
  // REPORTS
  // --------------------------
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(F_state);
  REPORT(S_state);

  return nll;
}
