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
  DATA_VECTOR(cots_dat);     // Observed COTS adult density (individuals per m^2), strictly positive
  DATA_VECTOR(fast_dat);     // Observed fast-growing coral cover (%) (Acropora spp.)
  DATA_VECTOR(slow_dat);     // Observed slow-growing coral cover (%) (Faviidae + Porites)
  DATA_VECTOR(sst_dat);      // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);  // External larval/settler immigration (individuals per m^2 per year) -> juvenile pool

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
  PARAMETER(rC_max);      // year^-1; maximum per-capita recruitment rate to juveniles
  PARAMETER(mC_base);     // year^-1; baseline adult mortality of COTS (predation/natural)
  PARAMETER(epsilon_food);// dimensionless (0-1); efficiency converting per-capita intake to recruitment
  PARAMETER(K_food);      // proportion; half-saturation of per-capita intake for recruitment
  PARAMETER(Kc0);         // ind m^-2; baseline adult carrying capacity when no coral is present
  PARAMETER(kCF);         // ind m^-2 per proportion; added adult carrying capacity per unit fast coral
  PARAMETER(kCS);         // ind m^-2 per proportion; added adult carrying capacity per unit slow coral
  PARAMETER(A50);         // ind m^-2; Allee-effect half-saturation density (mate limitation) on recruitment
  PARAMETER(Topt);        // deg C; optimum SST for COTS reproductive performance
  PARAMETER(sigma_T);     // deg C; breadth (SD) of thermal performance curve
  PARAMETER(gamma_imm);   // dimensionless; scaling on external larval immigration (adds to juveniles)

  // NEW: Juvenile stage parameters
  PARAMETER(mJ_base);     // year^-1; baseline juvenile mortality (cryptic stage)
  PARAMETER(p_mat);       // 0-1; annual probability that a surviving juvenile matures to adult

  // Observation model standard deviations (stability enforced with minimum SD)
  PARAMETER(sd_lncots);      // SD of log-observation errors for COTS adults (lognormal)
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
  vector<Type> cots_pred(N); // Adults (ind m^-2) to match cots_dat units
  vector<Type> fast_pred(N); // percent cover
  vector<Type> slow_pred(N); // percent cover

  // Internal states for coral (proportions 0-1)
  vector<Type> F_state(N);   // fast coral proportion
  vector<Type> S_state(N);   // slow coral proportion

  // NEW: Juvenile state (ind m^-2), unobserved
  vector<Type> J_state(N);   // juvenile COTS density

  // Initialize states directly from data at t=0 (no data leakage beyond initialization)
  cots_pred(0) = cots_dat(0);          // Adults
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  F_state(0)   = fast_dat(0) / Type(100);
  S_state(0)   = slow_dat(0) / Type(100);
  // Juveniles not observed; initialize proportional to adults as a neutral prior
  J_state(0)   = cots_pred(0);

  // --------------------------
  // NEGATIVE LOG-LIKELIHOOD
  // --------------------------
  Type nll = Type(0);

  // --------------------------
  // PARAMETER BOUND PENALTIES (smooth)
  // --------------------------
  nll += bound_penalty(rF,           Type(0.0),   Type(2.0),   w_pen);   // fast coral growth
  nll += bound_penalty(rS,           Type(0.0),   Type(1.0),   w_pen);   // slow coral growth
  nll += bound_penalty(beta_space,   Type(0.0),   Type(20.0),  w_pen);   // space competition intensity
  nll += bound_penalty(K_space,      Type(0.2),   Type(0.95),  w_pen);   // max occupiable coral cover

  nll += bound_penalty(dF_base,      Type(0.0),   Type(0.8),   w_pen);   // fast coral natural mortality
  nll += bound_penalty(dS_base,      Type(0.0),   Type(0.6),   w_pen);   // slow coral natural mortality
  nll += bound_penalty(heat_sens_F,  Type(0.0),   Type(1.0),   w_pen);   // heat sensitivity fast coral
  nll += bound_penalty(heat_sens_S,  Type(0.0),   Type(1.0),   w_pen);   // heat sensitivity slow coral
  nll += bound_penalty(T_bleach,     Type(29.0),  Type(34.5),  w_pen);   // bleaching threshold temp
  nll += bound_penalty(bleach_slope, Type(0.1),   Type(5.0),   w_pen);   // bleaching slope
  nll += bound_penalty(m_bleach_max, Type(0.0),   Type(1.0),   w_pen);   // max bleaching mortality

  nll += bound_penalty(aF,           Type(0.0),   Type(20.0),  w_pen);   // attack rate fast coral
  nll += bound_penalty(aS,           Type(0.0),   Type(20.0),  w_pen);   // attack rate slow coral
  nll += bound_penalty(hF,           Type(0.01),  Type(5.0),   w_pen);   // handling time fast coral
  nll += bound_penalty(hS,           Type(0.01),  Type(5.0),   w_pen);   // handling time slow coral
  nll += bound_penalty(q_func,       Type(1.0),   Type(3.0),   w_pen);   // functional response exponent

  nll += bound_penalty(rC_max,       Type(0.0),   Type(10.0),  w_pen);   // max per-capita recruitment
  nll += bound_penalty(mC_base,      Type(0.0013),Type(2.56),  w_pen);   // adult baseline mortality
  nll += bound_penalty(epsilon_food, Type(0.0),   Type(1.0),   w_pen);   // food->recruit efficiency
  nll += bound_penalty(K_food,       Type(0.01),  Type(0.8),   w_pen);   // food half-saturation
  nll += bound_penalty(Kc0,          Type(0.0),   Type(2.0),   w_pen);   // base carrying capacity
  nll += bound_penalty(kCF,          Type(0.0),   Type(50.0),  w_pen);   // carrying capacity per fast coral
  nll += bound_penalty(kCS,          Type(0.0),   Type(50.0),  w_pen);   // carrying capacity per slow coral
  nll += bound_penalty(A50,          Type(0.01),  Type(5.0),   w_pen);   // Allee half density
  nll += bound_penalty(Topt,         Type(24.0),  Type(31.0),  w_pen);   // optimal SST for COTS
  nll += bound_penalty(sigma_T,      Type(0.5),   Type(5.0),   w_pen);   // thermal breadth
  nll += bound_penalty(gamma_imm,    Type(0.0),   Type(10.0),  w_pen);   // immigration scaler

  nll += bound_penalty(mJ_base,      Type(0.2),   Type(6.0),   w_pen);   // juvenile mortality
  nll += bound_penalty(p_mat,        Type(0.05),  Type(0.9),   w_pen);   // juvenile maturation probability

  nll += bound_penalty(sd_lncots,    Type(0.01),  Type(2.0),   w_pen);   // obs SD (log COTS)
  nll += bound_penalty(sd_logit_fast,Type(0.01),  Type(2.0),   w_pen);   // obs SD (logit fast)
  nll += bound_penalty(sd_logit_slow,Type(0.01),  Type(2.0),   w_pen);   // obs SD (logit slow)
  nll += bound_penalty(w_pen,        Type(0.001), Type(100.0), w_pen);   // penalty weight itself

  // --------------------------
  // MODEL EQUATIONS (discrete annual steps)
  // --------------------------
  for (int t = 1; t < N; t++) {
    // Previous states
    Type A_prev = cots_pred(t-1); // Adults at t-1 (ind m^-2)
    Type J_prev = J_state(t-1);   // Juveniles at t-1 (ind m^-2)
    Type F_prev = F_state(t-1);   // Fast coral (proportion)
    Type S_prev = S_state(t-1);   // Slow coral (proportion)

    // Forcing at time t
    Type SST_t = sst_dat(t);                // Sea surface temperature (deg C)
    Type IMM_t = cotsimm_dat(t);            // Larval/settler immigration (ind m^-2 yr^-1)

    // (1) Bleaching index (0-1) increasing with SST
    Type B_t = invlogit_stable(bleach_slope * (SST_t - T_bleach)); // Bleaching risk index

    // (2) Multi-prey functional response (Holling disc with exponent q) for adult feeding
    Type Fq = pow(CppAD::CondExpLt(F_prev, eps, eps, F_prev), q_func); // F^q, safe at 0
    Type Sq = pow(CppAD::CondExpLt(S_prev, eps, eps, S_prev), q_func); // S^q, safe at 0
    Type denom = Type(1.0) + aF * hF * Fq + aS * hS * Sq;              // Handling-limited denominator
    Type per_pred_F = aF * Fq / (denom + eps);                          // per-predator annual attack on fast coral
    Type per_pred_S = aS * Sq / (denom + eps);                          // per-predator annual attack on slow coral

    // (3) Hazard-based removal (prevents overconsumption beyond available coral)
    Type cons_F_total = A_prev * per_pred_F;                            // total fast coral consumption
    Type cons_S_total = A_prev * per_pred_S;                            // total slow coral consumption
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

    // (8) Per-predator intake (sum across prey) by adults
    Type I_per_pred = per_pred_F + per_pred_S;                          // total intake per predator (proportion/yr)

    // (9) Per-capita recruitment to juveniles modulated by food, temperature, and Allee effect
    Type I_sat = I_per_pred / (K_food + I_per_pred + eps);              // saturating food index (0..1)
    Type T_perf = exp(-Type(0.5) * sq((SST_t - Topt) / (sigma_T + eps)));// Gaussian thermal performance (0..1)
    Type Allee_m = A_prev / (A50 + A_prev + eps);                       // Allee factor (0..1)
    Type rC = rC_max * epsilon_food * I_sat * T_perf * Allee_m;         // recruits per adult per year
    Type Recruit = rC * A_prev;                                         // absolute recruits to juvenile pool

    // (10) Adult carrying capacity linked to coral (limits maturation into adult stage)
    Type Kc = Kc0 + kCF * F_prev + kCS * S_prev;                        // adult carrying capacity (ind m^-2)
    Kc = CppAD::CondExpLt(Kc, eps, eps, Kc);                            // ensure positive

    // Juvenile dynamics: survival, maturation, immigration, recruitment
    Type J_survive = J_prev * exp(-mJ_base);                             // juvenile survivors
    Type Mature = p_mat * J_survive;                                     // candidates to mature this year
    // Habitat-limited maturation into adults (0 when A_prev >= Kc)
    Type mat_mult = Type(1) - A_prev / (Kc + eps);
    mat_mult = CppAD::CondExpLt(mat_mult, Type(0), Type(0), mat_mult);
    Type Mature_eff = mat_mult * Mature;

    Type J_next_raw = J_survive - Mature + Recruit + gamma_imm * IMM_t;  // next juveniles
    Type J_next = softplus(J_next_raw, Type(5));                         // smooth positivity

    // Adult dynamics: survival and addition from maturation
    Type A_survive = A_prev * exp(-mC_base);                              // adult survivors
    Type A_next_raw = A_survive + Mature_eff;                             // next adults (pre-positivity)
    Type A_next = softplus(A_next_raw, Type(5));                          // smooth positivity

    // Assign states and predictions (convert coral to %)
    cots_pred(t) = A_next;                     // Adults prediction at t
    J_state(t)   = J_next;                     // Juveniles prediction at t
    F_state(t)   = F_next;                     // fast coral proportion at t
    S_state(t)   = S_next;                     // slow coral proportion at t
    fast_pred(t) = Type(100) * F_state(t);     // fast coral % cover prediction
    slow_pred(t) = Type(100) * S_state(t);     // slow coral % cover prediction
  }

  // --------------------------
  // LIKELIHOOD (use all observations with stability safeguards)
  // --------------------------
  for (int t = 0; t < N; t++) {
    // COTS adults: lognormal likelihood (strictly positive)
    Type y_cots = cots_dat(t);
    // Use unbiased mean on natural scale by subtracting 0.5*sd^2 on log scale
    Type mu_log_cots = log(cots_pred(t) + eps) - Type(0.5) * sq(sd_cots_eff);
    nll -= dnorm(log(y_cots + eps), mu_log_cots, sd_cots_eff, true);

    // Fast coral: logit-normal on proportion (convert observed % to proportion)
    Type y_fast_prop = (fast_dat(t) / Type(100));
    // clip to (eps, 1-eps) implicitly in safe_logit
    Type mu_logit_fast = safe_logit(F_state(t), eps);
    nll -= dnorm(safe_logit(y_fast_prop, eps), mu_logit_fast, sd_fast_eff, true);

    // Slow coral: logit-normal on proportion
    Type y_slow_prop = (slow_dat(t) / Type(100));
    Type mu_logit_slow = safe_logit(S_state(t), eps);
    nll -= dnorm(safe_logit(y_slow_prop, eps), mu_logit_slow, sd_slow_eff, true);
  }

  // Optionally report states (not required, but useful for diagnostics)
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(F_state);
  REPORT(S_state);

  return nll;
}
