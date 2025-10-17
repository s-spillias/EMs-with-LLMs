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

// Clip to minimum value (AD-safe)
template<class Type>
Type clip_min(Type x, Type m) {
  return CppAD::CondExpLt(x, m, m, x);
}

// Clamp into [0, 1] (AD-safe)
template<class Type>
Type clamp01(Type x) {
  Type y = CppAD::CondExpLt(x, Type(0), Type(0), x);
  return CppAD::CondExpGt(y, Type(1), Type(1), y);
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
  PARAMETER(lag_recruit); // dimensionless (0-1); weight on 1-year lag in recruitment driver
  PARAMETER(nu_Allee);    // dimensionless >=1; Hill exponent for Allee effect steepness

  // Observation model standard deviations (stability enforced with minimum SD)
  PARAMETER(sd_lncots);      // SD of log-observation errors for COTS (lognormal)
  PARAMETER(sd_logit_fast);  // SD of logit-observation errors for fast coral (logit-normal on proportion)
  PARAMETER(sd_logit_slow);  // SD of logit-observation errors for slow coral (logit-normal on proportion)

  // Penalty weight for keeping parameters in biologically meaningful ranges
  PARAMETER(w_pen);       // dimensionless; overall weight on smooth bound penalties

  // --------------------------
  // NUMERICAL STABILITY SETTINGS AND EFFECTIVE PARAMETERS
  // --------------------------
  Type min_sd = Type(0.05); // Minimum SD added in quadrature to avoid tiny variances
  Type sd_cots_eff  = sqrt(sq(sd_lncots)     + sq(min_sd)); // smooth floor on SDs
  Type sd_fast_eff  = sqrt(sq(sd_logit_fast) + sq(min_sd));
  Type sd_slow_eff  = sqrt(sq(sd_logit_slow) + sq(min_sd));

  // Ensure penalty weight cannot approach zero (decoupled from its own penalty)
  Type w_pen_eff = clip_min(w_pen + Type(0.1), Type(0.1));

  // Clamp key parameters to biologically/ numerically safe ranges for use in equations
  Type rF_effpar = clip_min(rF, Type(0.0));
  Type rS_effpar = clip_min(rS, Type(0.0));
  Type beta_space_eff = clip_min(beta_space, Type(0.0));
  Type K_space_eff = clamp01(K_space);

  Type dF_base_eff = clip_min(dF_base, Type(0.0));
  Type dS_base_eff = clip_min(dS_base, Type(0.0));
  Type heat_sens_F_eff = clamp01(heat_sens_F);
  Type heat_sens_S_eff = clamp01(heat_sens_S);
  Type bleach_slope_eff = clip_min(bleach_slope, Type(0.0));
  Type m_bleach_max_eff = clip_min(m_bleach_max, Type(0.0));

  Type aF_eff = clip_min(aF, Type(0.0));
  Type aS_eff = clip_min(aS, Type(0.0));
  Type hF_eff = clip_min(hF, Type(1e-6));
  Type hS_eff = clip_min(hS, Type(1e-6));
  Type q_eff  = clip_min(q_func, Type(1.0)); // enforce Type II minimum

  Type rC_max_eff = clip_min(rC_max, Type(0.0));
  Type mC_base_eff = clip_min(mC_base, Type(0.0));
  Type epsilon_food_eff = clamp01(epsilon_food);
  Type K_food_eff = clip_min(K_food, Type(1e-6));
  Type Kc0_eff = clip_min(Kc0, Type(0.0));
  Type kCF_eff = clip_min(kCF, Type(0.0));
  Type kCS_eff = clip_min(kCS, Type(0.0));
  Type A50_eff = clip_min(A50, Type(1e-6));
  Type sigma_T_eff = clip_min(sigma_T, Type(1e-4));
  Type gamma_imm_eff = clip_min(gamma_imm, Type(0.0));
  Type lag_recruit_eff = clamp01(lag_recruit);
  Type nu_eff = CppAD::CondExpLt(nu_Allee, Type(1.0), Type(1.0),
                  CppAD::CondExpGt(nu_Allee, Type(6.0), Type(6.0), nu_Allee));

  // --------------------------
  // STATE VECTORS (predictions)
  // --------------------------
  vector<Type> cots_pred(N); // individuals per m^2 (to match cots_dat units)
  vector<Type> fast_pred(N); // percent cover (to match fast_dat units)
  vector<Type> slow_pred(N); // percent cover (to match slow_dat units)

  // Internal state in proportions for coral cover (0-1 scale)
  vector<Type> F_state(N);   // fast coral proportion (0-1)
  vector<Type> S_state(N);   // slow coral proportion (0-1)

  // --------------------------
  // INITIAL CONDITIONS (t = 0): set from data (allowed; no leakage for t > 0)
  // --------------------------
  cots_pred(0) = clip_min(cots_dat(0), Type(1e-6));     // Initial COTS density from data (ensure > 0)
  fast_pred(0) = fast_dat(0);                            // Initial fast coral (%)
  slow_pred(0) = slow_dat(0);                            // Initial slow coral (%)
  F_state(0)   = clamp01(fast_dat(0) / Type(100));       // Convert percent to proportion for internal dynamics
  S_state(0)   = clamp01(slow_dat(0) / Type(100));

  // --------------------------
  // NEGATIVE LOG-LIKELIHOOD
  // --------------------------
  Type nll = Type(0);

  // --------------------------
  // PARAMETER BOUND PENALTIES (smooth)
  // Suggested biologically plausible ranges are enforced softly (not hard constraints).
  // --------------------------
  nll += bound_penalty(rF,           Type(0.0),   Type(2.0),   w_pen_eff);   // fast coral growth
  nll += bound_penalty(rS,           Type(0.0),   Type(1.0),   w_pen_eff);   // slow coral growth
  nll += bound_penalty(beta_space,   Type(0.0),   Type(20.0),  w_pen_eff);   // space competition intensity
  nll += bound_penalty(K_space,      Type(0.2),   Type(0.95),  w_pen_eff);   // max occupiable coral cover

  nll += bound_penalty(dF_base,      Type(0.0),   Type(0.8),   w_pen_eff);   // fast coral natural mortality
  nll += bound_penalty(dS_base,      Type(0.0),   Type(0.6),   w_pen_eff);   // slow coral natural mortality
  nll += bound_penalty(heat_sens_F,  Type(0.0),   Type(1.0),   w_pen_eff);   // heat sensitivity fast coral
  nll += bound_penalty(heat_sens_S,  Type(0.0),   Type(1.0),   w_pen_eff);   // heat sensitivity slow coral
  nll += bound_penalty(T_bleach,     Type(29.0),  Type(34.5),  w_pen_eff);   // bleaching threshold temp
  nll += bound_penalty(bleach_slope, Type(0.1),   Type(5.0),   w_pen_eff);   // bleaching slope
  nll += bound_penalty(m_bleach_max, Type(0.0),   Type(1.0),   w_pen_eff);   // max bleaching mortality

  nll += bound_penalty(aF,           Type(0.0),   Type(20.0),  w_pen_eff);   // attack rate fast coral
  nll += bound_penalty(aS,           Type(0.0),   Type(20.0),  w_pen_eff);   // attack rate slow coral
  nll += bound_penalty(hF,           Type(0.01),  Type(5.0),   w_pen_eff);   // handling time fast coral
  nll += bound_penalty(hS,           Type(0.01),  Type(5.0),   w_pen_eff);   // handling time slow coral
  nll += bound_penalty(q_func,       Type(1.0),   Type(3.0),   w_pen_eff);   // functional response exponent

  nll += bound_penalty(rC_max,       Type(0.0),   Type(10.0),  w_pen_eff);   // max COTS growth rate
  nll += bound_penalty(mC_base,      Type(0.0013),Type(2.56),  w_pen_eff);   // COTS baseline mortality
  nll += bound_penalty(epsilon_food, Type(0.0),   Type(1.0),   w_pen_eff);   // food->recruit efficiency
  nll += bound_penalty(K_food,       Type(0.01),  Type(0.8),   w_pen_eff);   // food half-saturation
  nll += bound_penalty(Kc0,          Type(0.0),   Type(2.0),   w_pen_eff);   // base carrying capacity
  nll += bound_penalty(kCF,          Type(0.0),   Type(50.0),  w_pen_eff);   // carrying capacity per fast coral
  nll += bound_penalty(kCS,          Type(0.0),   Type(50.0),  w_pen_eff);   // carrying capacity per slow coral
  nll += bound_penalty(A50,          Type(0.01),  Type(5.0),   w_pen_eff);   // Allee half density
  nll += bound_penalty(Topt,         Type(24.0),  Type(31.0),  w_pen_eff);   // optimal SST for COTS
  nll += bound_penalty(sigma_T,      Type(0.5),   Type(5.0),   w_pen_eff);   // thermal breadth
  nll += bound_penalty(gamma_imm,    Type(0.0),   Type(10.0),  w_pen_eff);   // immigration scaler
  nll += bound_penalty(lag_recruit,  Type(0.0),   Type(1.0),   w_pen_eff);   // lag weight
  nll += bound_penalty(nu_Allee,     Type(1.0),   Type(6.0),   w_pen_eff);   // Allee steepness (Hill exponent)

  nll += bound_penalty(sd_lncots,    Type(0.01),  Type(2.0),   w_pen_eff);   // obs SD (log COTS)
  nll += bound_penalty(sd_logit_fast,Type(0.01),  Type(2.0),   w_pen_eff);   // obs SD (logit fast)
  nll += bound_penalty(sd_logit_slow,Type(0.01),  Type(2.0),   w_pen_eff);   // obs SD (logit slow)
  // Penalize w_pen itself with a fixed weight to avoid the "turn off penalties" loophole
  nll += bound_penalty(w_pen,        Type(0.001), Type(100.0), Type(1.0));

  // --------------------------
  // MODEL EQUATIONS (discrete annual steps)
  // Notes:
  // (1) Bleaching index: B_t = logistic(bleach_slope * (SST_t - T_bleach))
  // (2) Coral predation: multi-prey Holling disk with exponent q:
  //     per_pred_i = a_i * P_i^q / (1 + sum_j a_j h_j P_j^q)
  // (3) Hazard-based removal: R_i = P_i * (1 - exp(-C_{t-1} * per_pred_i / (P_i + eps)))
  // (4) Coral non-predation mortality: sequential hazard on survivors, with rate d_i + m_bleach_max * B_t
  // (5) Coral growth (space-limited BH-like): G_i = r_i_eff * P_i_survive / (1 + beta_space * (P_F_survive + P_S_survive))
  //     then cap total coral to K_space by proportional rescaling if exceeded.
  // (6) COTS reproduction driver uses food saturation, temperature performance, Hill-type Allee effect, and a 1-year lag.
  // (7) COTS update: C_t = C_{t-1} * exp(rC * (1 - C_{t-1}/Kc) - mC_base) + gamma_imm * cotsimm_dat_t
  // --------------------------

  // Initialize lagged recruitment driver with t=0 instantaneous value
  Type Fq_init = pow(F_state(0) + eps, q_eff);
  Type Sq_init = pow(S_state(0) + eps, q_eff);
  Type denom_init = Type(1.0) + aF_eff * hF_eff * Fq_init + aS_eff * hS_eff * Sq_init;
  Type intake_init = (aF_eff * Fq_init + aS_eff * Sq_init) / (denom_init + eps);
  Type food_term_init = intake_init / (K_food_eff + intake_init + eps);
  Type food_driver_init = epsilon_food_eff * food_term_init;
  Type temp_perf_init = exp(-Type(0.5) * sq((sst_dat(0) - Topt) / (sigma_T_eff + eps)));
  Type rec_driver_prev = food_driver_init * temp_perf_init; // at t=0

  // --------------------------
  // TIME LOOP
  // --------------------------
  for (int t = 1; t < N; t++) {
    // Previous states (no data leakage)
    Type F_prev = F_state(t - 1);
    Type S_prev = S_state(t - 1);
    Type C_prev = cots_pred(t - 1);

    // (1) Bleaching index for year t
    Type B_t = invlogit_stable(bleach_slope_eff * (sst_dat(t) - T_bleach));

    // (2) Multi-prey functional response (per-predator intake rates on each coral)
    Type Fq = pow(F_prev + eps, q_eff);
    Type Sq = pow(S_prev + eps, q_eff);
    Type denom = Type(1.0) + aF_eff * hF_eff * Fq + aS_eff * hS_eff * Sq;

    Type per_pred_F = aF_eff * Fq / (denom + eps);
    Type per_pred_S = aS_eff * Sq / (denom + eps);

    // (3) Hazard-based coral removal by predation
    Type lambda_F = C_prev * per_pred_F / (F_prev + eps);
    Type lambda_S = C_prev * per_pred_S / (S_prev + eps);
    Type R_F = F_prev * (Type(1.0) - exp(-lambda_F));
    Type R_S = S_prev * (Type(1.0) - exp(-lambda_S));
    Type F_after_pred = clamp01(F_prev - R_F);
    Type S_after_pred = clamp01(S_prev - R_S);

    // (4) Non-predation mortality (background + bleaching)
    Type mu_F = dF_base_eff + m_bleach_max_eff * B_t;
    Type mu_S = dS_base_eff + m_bleach_max_eff * B_t;
    Type M_F = F_after_pred * (Type(1.0) - exp(-mu_F));
    Type M_S = S_after_pred * (Type(1.0) - exp(-mu_S));
    Type F_survive = clamp01(F_after_pred - M_F);
    Type S_survive = clamp01(S_after_pred - M_S);

    // (5) Coral growth with heat suppression and space limitation
    Type gsup_F = clamp01(Type(1.0) - heat_sens_F_eff * B_t);
    Type gsup_S = clamp01(Type(1.0) - heat_sens_S_eff * B_t);
    Type rF_eff_growth = rF_effpar * gsup_F;
    Type rS_eff_growth = rS_effpar * gsup_S;

    Type crowd = F_survive + S_survive;
    Type growth_denom = Type(1.0) + beta_space_eff * crowd;

    Type G_F = rF_eff_growth * F_survive / (growth_denom + eps);
    Type G_S = rS_eff_growth * S_survive / (growth_denom + eps);

    Type F_next = F_survive + G_F;
    Type S_next = S_survive + G_S;

    // Cap total coral cover to K_space by proportional rescaling if exceeded
    Type total_after_growth = F_next + S_next;
    Type over_cap = CppAD::CondExpGt(total_after_growth, K_space_eff, Type(1.0), Type(0.0));
    Type scale = CppAD::CondExpGt(total_after_growth, eps, K_space_eff / (total_after_growth + eps), Type(1.0));
    F_next = CppAD::CondExpEq(over_cap, Type(1.0), F_next * scale, F_next);
    S_next = CppAD::CondExpEq(over_cap, Type(1.0), S_next * scale, S_next);

    // Bound to [0,1]
    F_next = clamp01(F_next);
    S_next = clamp01(S_next);

    // Store coral predictions (as proportions internally, percent as outputs)
    F_state(t)  = F_next;
    S_state(t)  = S_next;
    fast_pred(t) = F_next * Type(100.0);
    slow_pred(t) = S_next * Type(100.0);

    // --------------------------
    // COTS dynamics
    // --------------------------

    // Food intake (per predator) for reproduction driver, using previous coral state
    Type intake = (aF_eff * Fq + aS_eff * Sq) / (denom + eps);
    Type food_term = intake / (K_food_eff + intake + eps);
    Type food_driver = epsilon_food_eff * food_term;

    // Temperature performance (Gaussian) for reproduction
    Type temp_perf = exp(-Type(0.5) * sq((sst_dat(t) - Topt) / (sigma_T_eff + eps)));

    // Lagged recruitment driver blend
    Type rec_inst = food_driver * temp_perf;
    Type rec_eff = (Type(1.0) - lag_recruit_eff) * rec_inst + lag_recruit_eff * rec_driver_prev;

    // Hill-type Allee effect (mate limitation; sharpness via nu_eff)
    Type Cn = pow(clip_min(C_prev, Type(1e-12)), nu_eff);
    Type A50n = pow(A50_eff, nu_eff);
    Type Allee_m = Cn / (A50n + Cn + eps);

    // Per-capita growth rate for COTS
    Type rC = rC_max_eff * rec_eff * Allee_m;

    // Carrying capacity as function of previous coral state
    Type Kc = Kc0_eff + kCF_eff * F_prev + kCS_eff * S_prev;

    // COTS update: logistic-type with mortality and immigration (all using previous states)
    Type C_next = C_prev * exp(rC * (Type(1.0) - C_prev / (Kc + eps)) - mC_base_eff) + gamma_imm_eff * cotsimm_dat(t);
    C_next = clip_min(C_next, Type(1e-12));
    cots_pred(t) = C_next;

    // Update lag memory for next step using instantaneous (previous) conditions
    rec_driver_prev = rec_inst;
  }

  // --------------------------
  // OBSERVATION MODEL
  // --------------------------
  for (int t = 0; t < N; t++) {
    // COTS: lognormal observation model
    Type c_pred = clip_min(cots_pred(t), Type(1e-12));
    Type c_obs = clip_min(cots_dat(t), Type(1e-12));
    nll -= dnorm(log(c_obs), log(c_pred), sd_cots_eff, true);

    // Coral: logit-normal on proportions
    Type pf_obs = clamp01(fast_dat(t) / Type(100.0));
    Type ps_obs = clamp01(slow_dat(t) / Type(100.0));
    Type pf_pred = clamp01(F_state(t));
    Type ps_pred = clamp01(S_state(t));

    Type lf_obs = safe_logit(pf_obs);
    Type ls_obs = safe_logit(ps_obs);
    Type lf_pred = safe_logit(pf_pred);
    Type ls_pred = safe_logit(ps_pred);

    nll -= dnorm(lf_obs, lf_pred, sd_fast_eff, true);
    nll -= dnorm(ls_obs, ls_pred, sd_slow_eff, true);
  }

  // --------------------------
  // REPORTS
  // --------------------------
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(F_state);
  REPORT(S_state);

  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);
  ADREPORT(F_state);
  ADREPORT(S_state);

  return nll;
}
