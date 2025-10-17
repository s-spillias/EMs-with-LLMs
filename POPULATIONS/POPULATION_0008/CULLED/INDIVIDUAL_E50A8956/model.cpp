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
  PARAMETER(dF_base);     // year^-1; background mortality fast coral (non-bleaching)
  PARAMETER(dS_base);     // year^-1; background mortality slow coral (non-bleaching)
  PARAMETER(heat_sens_F); // dimensionless (0-1); proportional growth suppression of fast coral under heat stress
  PARAMETER(heat_sens_S); // dimensionless (0-1); proportional growth suppression of slow coral under heat stress
  PARAMETER(T_bleach);    // deg C; SST center where bleaching risk accelerates
  PARAMETER(bleach_slope);// (deg C)^-1; slope of bleaching logistic
  PARAMETER(m_bleach_max);// year^-1; additional mortality rate at extreme heat (asymptote)

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
  PARAMETER(alpha_larv);  // dimensionless; pulse-sharpness exponent for lagged larval thermal survival
  PARAMETER(theta_C);     // dimensionless; theta-Ricker shape for density regulation (1=Ricker, >1 overcompensation)

  // New: COTS density-dependent mortality (predation/disease/cannibalism)
  PARAMETER(mC_dd_max);   // year^-1; maximum additional mortality at high COTS density
  PARAMETER(Kc_dd);       // ind m^-2; half-saturation density for density-dependent mortality

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
  nll += bound_penalty(rF,           Type(0.0),   Type(2.0),   w_pen);   // fast coral growth
  nll += bound_penalty(rS,           Type(0.0),   Type(1.0),   w_pen);   // slow coral growth
  nll += bound_penalty(beta_space,   Type(0.0),   Type(20.0),  w_pen);   // space competition intensity
  nll += bound_penalty(K_space,      Type(0.2),   Type(0.95),  w_pen);   // max occupiable coral cover

  nll += bound_penalty(dF_base,      Type(0.0),   Type(0.8),   w_pen);   // fast coral natural mortality
  nll += bound_penalty(dS_base,      Type(0.0),   Type(0.6),   w_pen);   // slow coral natural mortality
  nll += bound_penalty(heat_sens_F,  Type(0.0),   Type(1.0),   w_pen);   // heat sensitivity fast coral
  nll += bound_penalty(heat_sens_S,  Type(0.0),   Type(1.0),   w_pen);   // heat sensitivity slow coral
  nll += bound_penalty(T_bleach,     Type(29.0),  Type(34.5),  w_pen);   // bleaching threshold temp (aligned to params.json)
  nll += bound_penalty(bleach_slope, Type(0.1),   Type(5.0),   w_pen);   // bleaching slope
  nll += bound_penalty(m_bleach_max, Type(0.0),   Type(1.0),   w_pen);   // max bleaching mortality

  nll += bound_penalty(aF,           Type(0.0),   Type(20.0),  w_pen);   // attack rate fast coral
  nll += bound_penalty(aS,           Type(0.0),   Type(20.0),  w_pen);   // attack rate slow coral
  nll += bound_penalty(hF,           Type(0.01),  Type(5.0),   w_pen);   // handling time fast coral
  nll += bound_penalty(hS,           Type(0.01),  Type(5.0),   w_pen);   // handling time slow coral
  nll += bound_penalty(q_func,       Type(1.0),   Type(3.0),   w_pen);   // functional response exponent

  nll += bound_penalty(rC_max,       Type(0.0),   Type(10.0),  w_pen);   // max COTS growth rate
  nll += bound_penalty(mC_base,      Type(0.0013),Type(2.56),  w_pen);   // COTS baseline mortality (aligned to params.json)
  nll += bound_penalty(epsilon_food, Type(0.0),   Type(1.0),   w_pen);   // food->recruit efficiency
  nll += bound_penalty(K_food,       Type(0.01),  Type(0.8),   w_pen);   // food half-saturation
  nll += bound_penalty(Kc0,          Type(0.0),   Type(2.0),   w_pen);   // base carrying capacity
  nll += bound_penalty(kCF,          Type(0.0),   Type(50.0),  w_pen);   // carrying capacity per fast coral
  nll += bound_penalty(kCS,          Type(0.0),   Type(50.0),  w_pen);   // carrying capacity per slow coral
  nll += bound_penalty(A50,          Type(0.01),  Type(5.0),   w_pen);   // Allee half density
  nll += bound_penalty(Topt,         Type(24.0),  Type(31.0),  w_pen);   // optimal SST for COTS
  nll += bound_penalty(sigma_T,      Type(0.5),   Type(5.0),   w_pen);   // thermal breadth
  nll += bound_penalty(gamma_imm,    Type(0.0),   Type(10.0),  w_pen);   // immigration scaler
  nll += bound_penalty(alpha_larv,   Type(0.5),   Type(10.0),  w_pen);   // pulse sharpness exponent
  nll += bound_penalty(theta_C,      Type(0.5),   Type(5.0),   w_pen);   // theta-Ricker shape

  // New parameter penalties: density-dependent mortality
  nll += bound_penalty(mC_dd_max,    Type(0.0),   Type(5.0),   w_pen);   // max extra mortality
  nll += bound_penalty(Kc_dd,        Type(0.01),  Type(10.0),  w_pen);   // half-saturation density

  nll += bound_penalty(sd_lncots,    Type(0.01),  Type(2.0),   w_pen);   // obs SD (log COTS)
  nll += bound_penalty(sd_logit_fast,Type(0.01),  Type(2.0),   w_pen);   // obs SD (logit fast)
  nll += bound_penalty(sd_logit_slow,Type(0.01),  Type(2.0),   w_pen);   // obs SD (logit slow)
  nll += bound_penalty(w_pen,        Type(0.001), Type(100.0), w_pen);   // penalty weight itself

  // --------------------------
  // MODEL EQUATIONS (discrete annual steps)
  // Numbered description:
  // (1) Bleaching index: B_t = logistic(bleach_slope * (SST_t - T_bleach))
  // (2) Coral predation: multi-prey Holling disk equation with exponent q:
  //     per_pred_i = a_i * P_i^q / (1 + sum_j a_j h_j P_j^q)
  // (3) Hazard-based removal: R_i = P_i * (1 - exp(-C_{t-1} * per_pred_i / (P_i + eps)))
  // (4) Coral non-predation mortality: M_i = P_i * (1 - exp(-(d_i + m_bleach_max * B_t)))
  // (5) Coral growth (space-limited BH-like): G_i = (r_i * P_i_survive) / (1 + beta_space * (P_F_survive + P_S_survive))
  // (6) Heat suppression of growth: G_i_eff = G_i * (1 - heat_sens_i * B_t)
  // (7) Coral update: P_i(t) = P_i_survive + G_i_eff, then exponential mortality applied within step
  // (8) COTS intake per predator: I = per_pred_F + per_pred_S
  // (9) COTS per-capita growth modifier (lagged environment): rC = rC_max * epsilon_food * I/(K_food + I) * [T_perf(t-1)]^alpha_larv * (C/(A50 + C))
  // (10) COTS carrying capacity: Kc = Kc0 + kCF*P_F + kCS*P_S
  // (11) COTS update (theta-Ricker + density-dependent mortality): C_t = C_{t-1} * exp(rC * (1 - (C_{t-1}/Kc)^theta_C)) - (mC_base + mC_dd(C_{t-1})) * C_{t-1} + gamma_imm * immigration_{t-1}
  // --------------------------

  for (int t = 1; t < N; t++) {
    // Previous states (predictions only; no data leakage)
    Type C_prev = cots_pred(t-1); // COTS density at t-1 (ind m^-2)
    Type F_prev = F_state(t-1);   // Fast coral proportion at t-1
    Type S_prev = S_state(t-1);   // Slow coral proportion at t-1

    // Forcing: current-year for coral stress; previous-year for larval processes
    Type SST_t   = sst_dat(t);                 // Sea surface temperature at t (deg C) - affects coral this year
    Type SST_tm1 = sst_dat(t-1);               // Sea surface temperature at t-1 (deg C) - affects larval survival
    Type IMM_tm1 = cotsimm_dat(t-1);           // Immigration at t-1 (ind m^-2 yr^-1), realized as recruits at t

    // (1) Bleaching index (0-1) increasing with SST (current year -> coral)
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
    Type R_F = remF_frac * F_prev;                                      // amount of fast coral removed
    Type R_S = remS_frac * S_prev;                                      // amount of slow coral removed

    // Survivors after predation (non-negative)
    Type F_survive = CppAD::CondExpLt(F_prev - R_F, eps, eps, F_prev - R_F);
    Type S_survive = CppAD::CondExpLt(S_prev - R_S, eps, eps, S_prev - R_S);

    // (4) Non-predation mortality (background + bleaching) via exponential decay
    Type mBleach = m_bleach_max * B_t;                                  // extra mortality due to heat
    Type F_after_mort = F_survive * exp(-(dF_base + mBleach));          // fast coral after mortality
    Type S_after_mort = S_survive * exp(-(dS_base + mBleach));          // slow coral after mortality

    // (5) Space-limited growth (Beverton–Holt-like)
    Type denom_space = Type(1) + beta_space * (F_survive + S_survive);  // crowding term
    Type G_F = rF * F_after_mort / (denom_space + eps);                 // raw growth fast coral
    Type G_S = rS * S_after_mort / (denom_space + eps);                 // raw growth slow coral

    // (6) Heat suppression of coral growth (0..1 multiplier)
    Type g_mult_F = Type(1) - heat_sens_F * B_t;                        // growth multiplier fast coral
    Type g_mult_S = Type(1) - heat_sens_S * B_t;                        // growth multiplier slow coral
    g_mult_F = CppAD::CondExpLt(g_mult_F, eps, eps, g_mult_F);          // keep non-negative smoothly
    g_mult_S = CppAD::CondExpLt(g_mult_S, eps, eps, g_mult_S);

    // (7) Coral updates and soft cap at K_space
    Type F_next = F_after_mort + g_mult_F * G_F;                        // next fast coral proportion
    Type S_next = S_after_mort + g_mult_S * G_S;                        // next slow coral proportion
    // keep within [eps, K_space] softly (via conditional clamps)
    F_next = CppAD::CondExpLt(F_next, eps, eps, F_next);
    S_next = CppAD::CondExpLt(S_next, eps, eps, S_next);
    F_next = CppAD::CondExpGt(F_next, K_space, K_space, F_next);
    S_next = CppAD::CondExpGt(S_next, K_space, K_space, S_next);

    // (8) Per-predator intake (sum across prey)
    Type I_per_pred = per_pred_F + per_pred_S;                          // total intake per predator (proportion/yr)

    // (9) Per-capita COTS growth modulated by food, lagged temperature, and Allee effect
    Type I_sat = I_per_pred / (K_food + I_per_pred + eps);              // saturating food index (0..1)
    Type T_perf_lag = exp(-Type(0.5) * sq((SST_tm1 - Topt) / (sigma_T + eps))); // Gaussian thermal performance at t-1 (0..1)
    Type T_pulse = pow(T_perf_lag + eps, alpha_larv);                   // sharpened survival pulse
    Type Allee_m = C_prev / (A50 + C_prev + eps);                       // Allee factor (0..1)
    Type rC = rC_max * epsilon_food * I_sat * T_pulse * Allee_m;        // effective per-capita growth rate

    // (10) Carrying capacity linked to coral
    Type Kc = Kc0 + kCF * F_prev + kCS * S_prev;                        // carrying capacity (ind m^-2)
    Kc = CppAD::CondExpLt(Kc, eps, eps, Kc);                            // ensure positive

    // (11) Theta-Ricker density regulation with baseline + density-dependent mortality and lagged immigration
    Type dens_term = pow(C_prev / (Kc + eps), theta_C);
    Type G_Ricker = exp(rC * (Type(1) - dens_term));

    // New: density-dependent mortality term (saturating with C_prev)
    Type mC_dd = mC_dd_max * (C_prev / (Kc_dd + C_prev + eps));

    Type C_next_raw = C_prev * G_Ricker - (mC_base + mC_dd) * C_prev + gamma_imm * IMM_tm1;
    Type C_next = softplus(C_next_raw, Type(5));                        // smooth positivity

    // Assign states and predictions (convert coral to %)
    cots_pred(t) = C_next;                     // COTS prediction at t
    F_state(t)   = F_next;                     // fast coral proportion at t
    S_state(t)   = S_next;                     // slow coral proportion at t
    fast_pred(t) = Type(100) * F_state(t);     // fast coral % cover prediction
    slow_pred(t) = Type(100) * S_state(t);     // slow coral % cover prediction
  }

  // --------------------------
  // LIKELIHOOD (use all observations with stability safeguards)
  // --------------------------
  for (int t = 0; t < N; t++) {
    // COTS: lognormal likelihood (strictly positive)
    Type y_cots = cots_dat(t);                                    // observed COTS density
    Type mu_cots = cots_pred(t);                                  // predicted COTS density
    nll -= dnorm(log(y_cots + eps), log(mu_cots + eps), sd_cots_eff, true); // lognormal via log-scale normal

    // Coral: logit-normal on proportions (0-1), convert from percent
    Type y_fast_prop = CppAD::CondExpLt(fast_dat(t)/Type(100), eps, eps, fast_dat(t)/Type(100)); // observed fast (prop)
    Type y_slow_prop = CppAD::CondExpLt(slow_dat(t)/Type(100), eps, eps, slow_dat(t)/Type(100)); // observed slow (prop)
    Type mu_fast_prop = CppAD::CondExpLt(fast_pred(t)/Type(100), eps, eps, fast_pred(t)/Type(100)); // predicted fast (prop)
    Type mu_slow_prop = CppAD::CondExpLt(slow_pred(t)/Type(100), eps, eps, slow_pred(t)/Type(100)); // predicted slow (prop)

    // Apply logit transform safely
    Type y_fast_logit = safe_logit(y_fast_prop, eps);
    Type y_slow_logit = safe_logit(y_slow_prop, eps);
    Type mu_fast_logit = safe_logit(mu_fast_prop, eps);
    Type mu_slow_logit = safe_logit(mu_slow_prop, eps);

    // Add Gaussian log-likelihood on logit scale
    nll -= dnorm(y_fast_logit, mu_fast_logit, sd_fast_eff, true);
    nll -= dnorm(y_slow_logit, mu_slow_logit, sd_slow_eff, true);
  }

  // --------------------------
  // REPORTING
  // --------------------------
  REPORT(cots_pred); // Predicted COTS densities (ind m^-2)
  REPORT(fast_pred); // Predicted fast coral % cover
  REPORT(slow_pred); // Predicted slow coral % cover

  // Optionally expose internal states for diagnostics
  REPORT(F_state);
  REPORT(S_state);

  return nll;
}
