#include <TMB.hpp>

// Helper: square
template<class Type>
Type sq(Type x) { return x * x; }

// Helper: clamp to [0,1]
template<class Type>
Type clamp01(Type x) {
  return CppAD::CondExpLt(x, Type(0), Type(0),
         CppAD::CondExpGt(x, Type(1), Type(1), x));
}

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
  using namespace density;

  // --------------------------
  // DATA (time series inputs)
  // --------------------------
  DATA_VECTOR(Year);         // Calendar year (integer year)
  DATA_VECTOR(cots_dat);     // Observed COTS density (ind/m^2), strictly positive
  DATA_VECTOR(fast_dat);     // Observed fast-growing coral cover (percent 0-100)
  DATA_VECTOR(slow_dat);     // Observed slow-growing coral cover (percent 0-100)
  DATA_VECTOR(sst_dat);      // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);  // Exogenous larval immigration (ind/m^2 per year)

  int T = Year.size();

  // --------------------------
  // PARAMETERS
  // --------------------------
  // Coral growth and space competition
  PARAMETER(rF);
  PARAMETER(rS);
  PARAMETER(beta_space);
  PARAMETER(K_space);
  PARAMETER(dF_base);
  PARAMETER(dS_base);

  // Heat-stress effects on corals
  PARAMETER(heat_sens_F);
  PARAMETER(heat_sens_S);
  PARAMETER(T_bleach);
  PARAMETER(bleach_slope);
  PARAMETER(m_bleach_max);

  // COTS foraging functional response
  PARAMETER(aF);
  PARAMETER(aS);
  PARAMETER(hF);
  PARAMETER(hS);
  PARAMETER(q_func);

  // COTS demographic parameters
  PARAMETER(rC_max);
  PARAMETER(mC_base);
  PARAMETER(mC_starv_max); // environmental modifier (food limitation)
  PARAMETER(epsilon_food);
  PARAMETER(K_food);
  PARAMETER(Kc0);
  PARAMETER(kCF);
  PARAMETER(kCS);
  PARAMETER(A50);
  PARAMETER(Topt);
  PARAMETER(sigma_T);
  PARAMETER(gamma_imm); // scaling for exogenous larval immigration

  // Observation error SDs
  PARAMETER(sd_lncots);
  PARAMETER(sd_logit_fast);
  PARAMETER(sd_logit_slow);

  // Weight for smooth bound penalties
  PARAMETER(w_pen);

  // Initial conditions (estimated)
  PARAMETER(logit_F0);
  PARAMETER(logit_S0);
  PARAMETER(log_C0);

  // --------------------------
  // STATE VECTORS
  // --------------------------
  vector<Type> F(T); // fast coral (0-1)
  vector<Type> S(T); // slow coral (0-1)
  vector<Type> C(T); // COTS density (ind/m^2)

  // Explicit prediction vectors for outputs (used in likelihood and reporting)
  // For reporting, match units of the data: coral in percent (0-100), COTS in ind/m^2
  vector<Type> fast_pred(T);
  vector<Type> slow_pred(T);
  vector<Type> cots_pred(T);

  // Initialize states from parameters
  F(0) = invlogit_stable(logit_F0);
  S(0) = invlogit_stable(logit_S0);
  // Ensure initial total coral does not exceed K_space by proportional scaling
  {
    Type total0 = F(0) + S(0);
    Type over = CppAD::CondExpGt(total0, K_space, Type(1), Type(0));
    Type scale = CppAD::CondExpEq(over, Type(1), K_space / (total0 + Type(1e-12)), Type(1));
    F(0) *= scale;
    S(0) *= scale;
  }
  C(0) = exp(log_C0);

  // Initialize prediction vectors at t=0 (scale coral to percent for reporting)
  fast_pred(0) = F(0) * Type(100.0);
  slow_pred(0) = S(0) * Type(100.0);
  cots_pred(0) = C(0);

  // --------------------------
  // PROCESS MODEL (no data leakage)
  // Uses only previous time-step predicted states and exogenous drivers.
  // --------------------------
  for (int t = 0; t < T - 1; ++t) {
    // Space limitation using logistic crowding form: (1 - total/K)^beta (>=0)
    Type total_coral = F(t) + S(t);
    Type crowd_frac = Type(1) - total_coral / (K_space + Type(1e-12));
    crowd_frac = CppAD::CondExpLt(crowd_frac, Type(0), Type(0), crowd_frac);
    Type space_lim = pow(crowd_frac, beta_space);

    // Heat stress (bleaching probability-like index)
    Type pBleach = invlogit_stable(bleach_slope * (sst_dat(t) - T_bleach));
    // Growth suppression factors (clamped >= 0)
    Type gF = CppAD::CondExpLt((Type(1) - heat_sens_F * pBleach), Type(0), Type(0), (Type(1) - heat_sens_F * pBleach));
    Type gS = CppAD::CondExpLt((Type(1) - heat_sens_S * pBleach), Type(0), Type(0), (Type(1) - heat_sens_S * pBleach));
    Type m_bleach = m_bleach_max * pBleach;

    // Multi-prey Holling functional response with exponent q_func (Type II if q=1; Type III if q>1)
    Type Fq = pow(CppAD::CondExpLt(F(t), Type(0), Type(0), F(t)), q_func);
    Type Sq = pow(CppAD::CondExpLt(S(t), Type(0), Type(0), S(t)), q_func);
    Type denom = Type(1) + aF * hF * Fq + aS * hS * Sq;
    Type fiF = aF * Fq / denom; // per-capita intake on fast coral
    Type fiS = aS * Sq / denom; // per-capita intake on slow coral
    Type I_tot = fiF + fiS;     // total per-capita intake

    // Predation losses (cannot exceed available coral)
    Type lossF = C(t) * fiF;
    Type lossS = C(t) * fiS;
    lossF = CppAD::CondExpGt(lossF, F(t), F(t), lossF);
    lossS = CppAD::CondExpGt(lossS, S(t), S(t), lossS);

    // Coral updates (Euler step over 1-year intervals)
    Type F_next = F(t)
      + rF * F(t) * space_lim * gF
      - dF_base * F(t)
      - m_bleach * F(t)
      - lossF;

    Type S_next = S(t)
      + rS * S(t) * space_lim * gS
      - dS_base * S(t)
      - m_bleach * S(t)
      - lossS;

    // Enforce non-negativity
    F_next = CppAD::CondExpLt(F_next, Type(0), Type(0), F_next);
    S_next = CppAD::CondExpLt(S_next, Type(0), Type(0), S_next);

    // Enforce total coral <= K_space via proportional scaling if needed
    Type total_next = F_next + S_next;
    Type over_next = CppAD::CondExpGt(total_next, K_space, Type(1), Type(0));
    Type scale_next = CppAD::CondExpEq(over_next, Type(1), K_space / (total_next + Type(1e-12)), Type(1));
    F(t + 1) = F_next * scale_next;
    S(t + 1) = S_next * scale_next;

    // COTS reproduction modifiers
    Type perf_T = exp(-Type(0.5) * sq((sst_dat(t) - Topt) / (sigma_T + Type(1e-12)))); // Gaussian thermal performance
    Type allee = C(t) / (A50 + C(t) + Type(1e-12));
    Type food_eff = epsilon_food * I_tot / (K_food + I_tot + Type(1e-12));
    Type rC_eff = rC_max * perf_T * allee * food_eff;

    // Carrying capacity depends on coral
    Type Kc = Kc0 + kCF * F(t) + kCS * S(t);
    Kc = CppAD::CondExpLt(Kc, Type(1e-8), Type(1e-8), Kc);

    // Starvation/food-limitation mortality scales with free space fraction
    Type free_space_frac = Type(1) - total_coral / (K_space + Type(1e-12));
    free_space_frac = CppAD::CondExpLt(free_space_frac, Type(0), Type(0), free_space_frac);
    Type mC_starv = mC_starv_max * free_space_frac;

    // Population update (logistic-like births minus mortality) + exogenous immigration
    Type births = rC_eff * C(t) * (Type(1) - C(t) / Kc);
    Type deaths = (mC_base + mC_starv) * C(t);
    Type immigration = gamma_imm * cotsimm_dat(t);
    Type C_next = C(t) + births - deaths + immigration;

    // Enforce non-negativity
    C(t + 1) = CppAD::CondExpLt(C_next, Type(0), Type(0), C_next);

    // Update prediction vectors at t+1 (scale coral to percent for reporting)
    fast_pred(t + 1) = F(t + 1) * Type(100.0);
    slow_pred(t + 1) = S(t + 1) * Type(100.0);
    cots_pred(t + 1) = C(t + 1);
  }

  // --------------------------
  // LIKELIHOOD (observation model)
  // Only uses predicted states; no data are fed back into the process.
  // --------------------------
  Type nll = Type(0);

  for (int t = 0; t < T; ++t) {
    // COTS: lognormal errors
    Type mu_logC = log(cots_pred(t) + Type(1e-8));
    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), mu_logC, sd_lncots + Type(1e-12), true);

    // Coral: logit-normal errors; convert observed percent to proportions
    Type mu_fast = safe_logit(clamp01(F(t)));
    Type mu_slow = safe_logit(clamp01(S(t)));
    nll -= dnorm(safe_logit(clamp01(fast_dat(t) / Type(100.0))), mu_fast, sd_logit_fast + Type(1e-12), true);
    nll -= dnorm(safe_logit(clamp01(slow_dat(t) / Type(100.0))), mu_slow, sd_logit_slow + Type(1e-12), true);
  }

  // --------------------------
  // Smooth parameter bound penalties (keep within ecological ranges)
  // --------------------------
  nll += bound_penalty(rF,           Type(0.0),   Type(2.0),   w_pen);
  nll += bound_penalty(rS,           Type(0.0),   Type(1.0),   w_pen);
  nll += bound_penalty(beta_space,   Type(0.0),   Type(20.0),  w_pen);
  nll += bound_penalty(K_space,      Type(0.2),   Type(0.95),  w_pen);
  nll += bound_penalty(dF_base,      Type(0.0),   Type(0.8),   w_pen);
  nll += bound_penalty(dS_base,      Type(0.0),   Type(0.6),   w_pen);

  nll += bound_penalty(heat_sens_F,  Type(0.0),   Type(1.0),   w_pen);
  nll += bound_penalty(heat_sens_S,  Type(0.0),   Type(1.0),   w_pen);
  nll += bound_penalty(T_bleach,     Type(29.0),  Type(34.5),  w_pen);
  nll += bound_penalty(bleach_slope, Type(0.1),   Type(5.0),   w_pen);
  nll += bound_penalty(m_bleach_max, Type(0.0),   Type(1.0),   w_pen);

  nll += bound_penalty(aF,           Type(0.0),   Type(20.0),  w_pen);
  nll += bound_penalty(aS,           Type(0.0),   Type(20.0),  w_pen);
  nll += bound_penalty(hF,           Type(0.01),  Type(5.0),   w_pen);
  nll += bound_penalty(hS,           Type(0.01),  Type(5.0),   w_pen);
  nll += bound_penalty(q_func,       Type(1.0),   Type(3.0),   w_pen);

  nll += bound_penalty(rC_max,       Type(0.0),   Type(10.0),  w_pen);
  nll += bound_penalty(mC_base,      Type(0.0013),Type(2.56),  w_pen);
  nll += bound_penalty(mC_starv_max, Type(0.0),   Type(2.0),   w_pen);
  nll += bound_penalty(epsilon_food, Type(0.0),   Type(1.0),   w_pen);
  nll += bound_penalty(K_food,       Type(0.01),  Type(0.8),   w_pen);
  nll += bound_penalty(Kc0,          Type(0.0),   Type(2.0),   w_pen);
  nll += bound_penalty(kCF,          Type(0.0),   Type(50.0),  w_pen);
  nll += bound_penalty(kCS,          Type(0.0),   Type(50.0),  w_pen);
  nll += bound_penalty(A50,          Type(0.01),  Type(5.0),   w_pen);
  nll += bound_penalty(Topt,         Type(24.0),  Type(31.0),  w_pen);
  nll += bound_penalty(sigma_T,      Type(0.5),   Type(5.0),   w_pen);
  nll += bound_penalty(gamma_imm,    Type(0.0),   Type(10.0),  w_pen);

  nll += bound_penalty(sd_lncots,    Type(0.01),  Type(2.0),   w_pen);
  nll += bound_penalty(sd_logit_fast,Type(0.01),  Type(2.0),   w_pen);
  nll += bound_penalty(sd_logit_slow,Type(0.01),  Type(2.0),   w_pen);
  nll += bound_penalty(w_pen,        Type(0.001), Type(100.0), w_pen);

  // --------------------------
  // REPORTS
  // --------------------------
  REPORT(F);
  REPORT(S);
  REPORT(C);
  REPORT(fast_pred); // percent
  REPORT(slow_pred); // percent
  REPORT(cots_pred);

  // Include objective function in report for downstream processing
  Type objective = nll;
  REPORT(objective);

  return nll;
}
