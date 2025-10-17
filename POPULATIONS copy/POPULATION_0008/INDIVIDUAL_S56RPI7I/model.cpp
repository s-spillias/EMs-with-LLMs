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

// Manual Normal negative log-likelihood (stable)
template<class Type>
Type nll_norm(Type x, Type mu, Type sd) {
  Type z = (x - mu) / (sd + Type(1e-12));
  return Type(0.5) * log(Type(6.283185307179586)) + log(sd + Type(1e-12)) + Type(0.5) * z * z;
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

  // New: COTS starvation mortality parameters (coral-dependent)
  PARAMETER(mC_starv_max); // year^-1; maximum extra mortality when coral is scarce
  PARAMETER(K_starv);      // proportion; half-saturation coral cover for starvation mortality

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
  nll += bound_penalty(rF,           Type(0.0),    Type(2.0),   w_pen);   // fast coral growth
  nll += bound_penalty(rS,           Type(0.0),    Type(1.0),   w_pen);   // slow coral growth
  nll += bound_penalty(beta_space,   Type(0.0),    Type(20.0),  w_pen);   // space competition intensity
  nll += bound_penalty(K_space,      Type(0.2),    Type(0.95),  w_pen);   // max occupiable coral cover

  nll += bound_penalty(dF_base,      Type(0.0),    Type(0.8),   w_pen);   // fast coral natural mortality
  nll += bound_penalty(dS_base,      Type(0.0),    Type(0.6),   w_pen);   // slow coral natural mortality
  nll += bound_penalty(heat_sens_F,  Type(0.0),    Type(1.0),   w_pen);   // heat sensitivity fast coral
  nll += bound_penalty(heat_sens_S,  Type(0.0),    Type(1.0),   w_pen);   // heat sensitivity slow coral

  // Updated to match parameters.json (literature-informed)
  nll += bound_penalty(T_bleach,     Type(29.0),   Type(34.5),  w_pen);   // bleaching threshold temp
  nll += bound_penalty(bleach_slope, Type(0.1),    Type(5.0),   w_pen);   // bleaching slope
  nll += bound_penalty(m_bleach_max, Type(0.0),    Type(1.0),   w_pen);   // max bleaching mortality

  nll += bound_penalty(aF,           Type(0.0),    Type(20.0),  w_pen);   // attack rate fast coral
  nll += bound_penalty(aS,           Type(0.0),    Type(20.0),  w_pen);   // attack rate slow coral
  nll += bound_penalty(hF,           Type(0.01),   Type(5.0),   w_pen);   // handling time fast coral
  nll += bound_penalty(hS,           Type(0.01),   Type(5.0),   w_pen);   // handling time slow coral
  nll += bound_penalty(q_func,       Type(1.0),    Type(3.0),   w_pen);   // functional response exponent

  nll += bound_penalty(rC_max,       Type(0.0),    Type(10.0),  w_pen);   // max COTS growth rate

  // Updated to match parameters.json (literature-informed)
  nll += bound_penalty(mC_base,      Type(0.0013), Type(2.56),  w_pen);   // COTS baseline mortality

  nll += bound_penalty(epsilon_food, Type(0.0),    Type(1.0),   w_pen);   // food->recruit efficiency
  nll += bound_penalty(K_food,       Type(0.01),   Type(0.8),   w_pen);   // food half-saturation
  nll += bound_penalty(Kc0,          Type(0.0),    Type(2.0),   w_pen);   // base carrying capacity
  nll += bound_penalty(kCF,          Type(0.0),    Type(50.0),  w_pen);   // carrying capacity per fast coral
  nll += bound_penalty(kCS,          Type(0.0),    Type(50.0),  w_pen);   // carrying capacity per slow coral
  nll += bound_penalty(A50,          Type(0.01),   Type(5.0),   w_pen);   // Allee half density
  nll += bound_penalty(Topt,         Type(24.0),   Type(31.0),  w_pen);   // optimal SST for COTS
  nll += bound_penalty(sigma_T,      Type(0.5),    Type(5.0),   w_pen);   // thermal breadth
  nll += bound_penalty(gamma_imm,    Type(0.0),    Type(10.0),  w_pen);   // immigration scaler

  // New starvation mortality parameter bounds
  nll += bound_penalty(mC_starv_max, Type(0.0),    Type(3.0),   w_pen);   // max starvation mortality
  nll += bound_penalty(K_starv,      Type(0.01),   Type(0.8),   w_pen);   // half-saturation of coral cover

  nll += bound_penalty(sd_lncots,    Type(0.01),   Type(2.0),   w_pen);   // obs SD (log COTS)
  nll += bound_penalty(sd_logit_fast,Type(0.01),   Type(2.0),   w_pen);   // obs SD (logit fast)
  nll += bound_penalty(sd_logit_slow,Type(0.01),   Type(2.0),   w_pen);   // obs SD (logit slow)
  nll += bound_penalty(w_pen,        Type(0.001),  Type(100.0), w_pen);   // penalty weight itself

  // --------------------------
  // MODEL EQUATIONS (discrete annual steps)
  // Numbered description:
  // (1) Bleaching index: B_t = logistic(bleach_slope * (SST_t - T_bleach))
  // (2) Coral predation: multi-prey Holling disc eq with exponent q:
  //     per_pred_i = a_i * P_i^q / (1 + sum_j a_j h_j P_j^q)
  // (3) Hazard-based removal: R_i = P_i * (1 - exp(-C_{t-1} * per_pred_i / (P_i + eps)))
  // (4) Coral non-predation mortality: survivors *= exp(-(d_i + m_bleach_max * B_t))
  // (5) Coral growth (space-limited BH-like): G_i = (r_i * P_i_survive) / (1 + beta_space * (P_F_survive + P_S_survive))
  // (6) Heat suppression of growth: G_i_eff = G_i * (1 - heat_sens_i * B_t)
  // (7) Coral update: P_i(t) = P_i_survive + G_i_eff
  // (8) COTS intake per predator: I = per_pred_F + per_pred_S
  // (9) COTS per-capita growth modifier: rC = rC_max * epsilon_food * I/(K_food + I) * exp(-0.5 * ((SST - Topt)/sigma_T)^2) * (C/(A50 + C))
  // (10) COTS density regulation: deltaC = rC * C * (1 - C / Kc), Kc = Kc0 + kCF*P_F + kCS*P_S
  // (11) COTS update: C_t = C_{t-1} + deltaC - (mC_base + mC_starv) * C_{t-1} + gamma_imm * immigration_t
  // --------------------------

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

    // (3) Hazard-based predation removal on coral
    Type lambdaF = C_prev * per_pred_F / (F_prev + eps);
    Type lambdaS = C_prev * per_pred_S / (S_prev + eps);
    Type R_F = F_prev * (Type(1) - exp(-lambdaF));
    Type R_S = S_prev * (Type(1) - exp(-lambdaS));
    Type F_after_pred = F_prev - R_F;
    Type S_after_pred = S_prev - R_S;

    // (4) Non-predation mortality (background + bleaching)
    Type muF = dF_base + m_bleach_max * B_t;
    Type muS = dS_base + m_bleach_max * B_t;
    Type F_survive = F_after_pred * exp(-muF);
    Type S_survive = S_after_pred * exp(-muS);

    // (5) Space-limited growth (Beverton–Holt-like)
    Type total_survive = F_survive + S_survive;
    Type GF = (rF * F_survive) / (Type(1) + beta_space * total_survive);
    Type GS = (rS * S_survive) / (Type(1) + beta_space * total_survive);

    // (6) Heat suppression of growth
    Type GF_eff = GF * (Type(1) - heat_sens_F * B_t);
    Type GS_eff = GS * (Type(1) - heat_sens_S * B_t);

    // (7) Coral update and clamping to [0, K_space]
    Type F_new = F_survive + GF_eff;
    Type S_new = S_survive + GS_eff;

    // Clamp to [0, K_space] smoothly with CondExp
    F_new = CppAD::CondExpLt(F_new, Type(0), Type(0), F_new);
    S_new = CppAD::CondExpLt(S_new, Type(0), Type(0), S_new);
    F_new = CppAD::CondExpGt(F_new, K_space, K_space, F_new);
    S_new = CppAD::CondExpGt(S_new, K_space, K_space, S_new);

    // (8) COTS intake per predator (from previous coral states)
    Type I = per_pred_F + per_pred_S;

    // (9) COTS per-capita growth modifier
    Type perf_food = epsilon_food * (I / (K_food + I + eps));
    Type perf_temp = exp(-Type(0.5) * sq((SST_t - Topt) / (sigma_T + eps)));
    Type perf_allee = C_prev / (A50 + C_prev + eps);
    Type rC = rC_max * perf_food * perf_temp * perf_allee;

    // (10) Density regulation via carrying capacity dependent on previous coral
    Type Kc = Kc0 + kCF * F_prev + kCS * S_prev;
    Kc = CppAD::CondExpLt(Kc, eps, eps, Kc);
    Type deltaC = rC * C_prev * (Type(1) - C_prev / (Kc + eps));

    // (11) Starvation mortality increases when coral cover is low
    Type P_total = F_prev + S_prev;
    Type mC_starv = mC_starv_max * (K_starv / (K_starv + P_total + eps));
    Type mC_total = mC_base + mC_starv;

    // COTS update with immigration (discrete-time approximation)
    Type C_next = C_prev + deltaC - mC_total * C_prev + gamma_imm * IMM_t;
    C_next = CppAD::CondExpLt(C_next, eps, eps, C_next); // enforce positivity

    // Save updated states
    cots_pred(t) = C_next;
    F_state(t) = F_new;
    S_state(t) = S_new;
    fast_pred(t) = F_new * Type(100);
    slow_pred(t) = S_new * Type(100);
  }

  // --------------------------
  // OBSERVATION MODEL (likelihood)
  // --------------------------
  for (int t = 0; t < N; t++) {
    // COTS: lognormal errors
    Type log_obs_c = log(cots_dat(t) + eps);
    Type log_pred_c = log(cots_pred(t) + eps);
    nll += nll_norm(log_obs_c, log_pred_c, sd_cots_eff);

    // Corals: logit-normal errors on proportions
    Type yF = fast_dat(t) / Type(100);
    Type yS = slow_dat(t) / Type(100);
    nll += nll_norm(safe_logit(yF), safe_logit(F_state(t)), sd_fast_eff);
    nll += nll_norm(safe_logit(yS), safe_logit(S_state(t)), sd_slow_eff);
  }

  // --------------------------
  // REPORTING
  // --------------------------
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(F_state);
  REPORT(S_state);

  return nll;
}
