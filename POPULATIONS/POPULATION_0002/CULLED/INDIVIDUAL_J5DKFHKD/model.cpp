#include <TMB.hpp>

// Smooth, numerically stable helper functions
template<class Type>
Type log1p_ad(Type x) {
  // AD-safe approximation of log1p(x) using log(1+x)
  return log(Type(1) + x);
}

template<class Type>
Type softplus(Type x) {
  // Smooth approximation to max(0, x) with AD-safe, numerically stable branches
  // softplus(x) = log(1 + exp(x))
  // Use stable forms to avoid overflow when x >> 0 and precision loss when x << 0
  Type zero = Type(0);
  return CppAD::CondExpGt(x, zero,
                          x + log1p_ad(exp(-x)), // for x > 0
                          log1p_ad(exp(x)));     // for x <= 0
}

template<class Type>
Type logit01(Type p, Type eps) {
  // p should be in [0,1]; internally clamped using eps to avoid infinities
  Type p_clamp_low  = CppAD::CondExpLt(p, eps, eps, p);
  Type p_clamp_high = CppAD::CondExpGt(p_clamp_low, Type(1) - eps, Type(1) - eps, p_clamp_low);
  return log(p_clamp_high + eps) - log(Type(1) - p_clamp_high + eps);
}

// Bound-penalty helper: smooth penalty if x is outside [lo, hi]
template<class Type>
Type square(Type x) { return x * x; }

template<class Type>
Type positive_part(Type x) {
  // Smooth positive part ~ max(0, x)
  return softplus(x) - softplus(-x);
}

template<class Type>
Type bound_penalty(Type x, Type lo, Type hi, Type w) {
  // Penalize only when outside [lo, hi] using smooth positive parts
  Type pen_lo = square(positive_part(lo - x)); // >0 if x < lo
  Type pen_hi = square(positive_part(x - hi)); // >0 if x > hi
  return w * (pen_lo + pen_hi);
}

template<class Type>
Type smooth_cap_flux(Type stock, Type flux, Type eps) {
  // Ensures removal flux <= stock via a smooth saturating mapping:
  // flux_eff = stock * (1 - exp(-flux / (stock + eps)))
  Type denom = stock + eps;
  return stock * (Type(1) - exp(-flux / denom));
}

/*
Numbered model equations (annual timestep, t = 1..T-1; data indexed t = 0..T-1):
1) Free space-limited coral growth (shared space, percent cover units):
   F_t+1 = F_t + rF * F_t * (1 - (F_t + S_t)/100) - Pred_F(A_t, F_t) - muF * F_t - Bleach_F(SST_t) * F_t
   S_t+1 = S_t + rS * S_t * (1 - (F_t + S_t)/100) - Pred_S(A_t, S_t) - muS * S_t - Bleach_S(SST_t) * S_t

2) COTS functional response on corals (Type III in coral resource, linear in predator density):
   Pred_F = smooth_cap_flux(F_t, aF * A_t * F_t^hF / (KF^hF + F_t^hF), eps)
   Pred_S = smooth_cap_flux(S_t, aS * A_t * S_t^hS / (KS^hS + S_t^hS), eps)

3) Temperature-driven bleaching (smooth ramp via softplus):
   Bleach_F(SST) = bF_bleach * softplus((SST - T_bleach)/delta_bleach)
   Bleach_S(SST) = bS_bleach * softplus((SST - T_bleach)/delta_bleach)

4) Food saturation index (0..1) from coral prey availability (Acropora preference prefF):
   FoodSat = prefF * F_t / (KF_food + F_t) + (1 - prefF) * S_t / (KS_food + S_t)

5) Temperature-modified adult survival with starvation penalty:
   sA_base(SST) = invlogit(sA_logit + sA_T * (SST - Topt_S))
   sA_eff = sA_base * exp(-m_starv * (1 - FoodSat))

6) Resource- and temperature-modified Ricker recruitment with density dependence:
   R_temp(SST) = 1 + a_temp * exp(-0.5 * ((SST - Topt_R)/sigmaT_R)^2)
   R_food      = 1 + a_food * FoodSat
   Recruit     = rA * A_t * exp(-betaA * A_t) * R_temp * R_food

7) Immigration contribution (external larvae -> recruits):
   Immigr = gamma_imm * cotsimm_dat_t

8) COTS state update:
   A_t+1 = sA_eff * A_t + Recruit + Immigr

Observation models:
- COTS (strictly positive): lognormal with floor on sigma
- Coral cover (bounded [0,100]): logit-normal on proportion scale with floors on sigma
Initial conditions:
- A_pred(0) = cots_dat(0); F_pred(0) = fast_dat(0); S_pred(0) = slow_dat(0)
No data leakage: Only previous-step predictions are used in state updates.
*/

template<class Type>
Type objective_function<Type>::operator() () {
  // --------------------
  // DATA (time-aligned)
  // --------------------
  DATA_VECTOR(Year);          // Year (calendar year), used for alignment and reporting
  DATA_VECTOR(cots_dat);      // Adult COTS density (individuals/m^2)
  DATA_VECTOR(fast_dat);      // Fast coral cover (Acropora spp.) in percent
  DATA_VECTOR(slow_dat);      // Slow coral cover (Faviidae/Porites) in percent
  DATA_VECTOR(sst_dat);       // Sea-surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat);   // External COTS larval immigration (individuals/m^2/year)

  // ---------------
  // PARAMETERS
  // ---------------
  // Coral intrinsic growth (year^-1)
  PARAMETER(rF);              // rF: intrinsic growth rate of fast coral (year^-1). Initial ~0.3; higher for Acropora.
  PARAMETER(rS);              // rS: intrinsic growth rate of slow coral (year^-1). Initial ~0.1; lower than Acropora.

  // Coral background mortality (fraction year^-1)
  PARAMETER(muF);             // muF: background mortality rate fast coral (fraction/year).
  PARAMETER(muS);             // muS: background mortality rate slow coral (fraction/year).

  // COTS predation on corals (percent removed per year per predator density)
  PARAMETER(aF);              // aF: attack rate on fast coral (% cover per (year * ind/m^2)).
  PARAMETER(aS);              // aS: attack rate on slow coral (% cover per (year * ind/m^2)).
  PARAMETER(KF);              // KF: half-saturation for fast coral predation (percent cover).
  PARAMETER(KS);              // KS: half-saturation for slow coral predation (percent cover).
  PARAMETER(hF);              // hF: shape exponent for fast coral functional response (dimensionless, >=1).
  PARAMETER(hS);              // hS: shape exponent for slow coral functional response (dimensionless, >=1).

  // Thermal bleaching parameters
  PARAMETER(bF_bleach);       // bF_bleach: bleaching mortality scale for fast coral (% per ramp unit).
  PARAMETER(bS_bleach);       // bS_bleach: bleaching mortality scale for slow coral (% per ramp unit).
  PARAMETER(T_bleach);        // T_bleach: temperature where bleaching ramp begins (deg C).
  PARAMETER(delta_bleach);    // delta_bleach: smoothing width for bleaching ramp (deg C).

  // Food saturation index parameters for COTS (dimensionless)
  PARAMETER(prefF);           // prefF: preference weight for fast coral in FoodSat (0..1); slow weight = 1 - prefF.
  PARAMETER(KF_food);         // KF_food: half-saturation for fast coral in FoodSat (percent cover).
  PARAMETER(KS_food);         // KS_food: half-saturation for slow coral in FoodSat (percent cover).

  // COTS survival and starvation
  PARAMETER(sA_logit);        // sA_logit: baseline adult annual survival on logit scale (dimensionless).
  PARAMETER(sA_T);            // sA_T: temperature slope for survival per deg C (logit units per deg C).
  PARAMETER(Topt_S);          // Topt_S: reference temperature for survival effect (deg C).
  PARAMETER(m_starv);         // m_starv: starvation mortality intensity (per year) when FoodSat -> 0.

  // COTS recruitment (Ricker) and modifiers
  PARAMETER(rA);              // rA: baseline recruitment scaling (year^-1) in Ricker term.
  PARAMETER(betaA);           // betaA: Ricker density-dependence (m^2/ind), controls bust after boom.
  PARAMETER(a_food);          // a_food: recruitment multiplier amplitude from FoodSat (dimensionless >=0).
  PARAMETER(Topt_R);          // Topt_R: temperature of peak recruitment (deg C).
  PARAMETER(sigmaT_R);        // sigmaT_R: width of temp window for recruitment (deg C).
  PARAMETER(a_temp);          // a_temp: amplitude of temp recruitment multiplier (dimensionless >=0).

  // Immigration scaling
  PARAMETER(gamma_imm);       // gamma_imm: efficiency converting immigration (ind/m^2/yr) to adult additions (dimensionless).

  // Observation model standard deviations
  PARAMETER(sigma_logCOTS);   // sigma_logCOTS: SD for log COTS observations.
  PARAMETER(sigma_logitF);    // sigma_logitF: SD for logit fast coral proportion.
  PARAMETER(sigma_logitS);    // sigma_logitS: SD for logit slow coral proportion.

  // Optional penalty weight (can be fixed via data; optimizing is acceptable but we will keep small)
  PARAMETER(penalty_weight);  // penalty_weight: smooth bound penalty weight (dimensionless).

  // ----------------
  // INITIALIZATION
  // ----------------
  int n = cots_dat.size();
  Type eps = Type(1e-8);           // small numerical constant
  Type sd_floor_log = Type(0.05);  // minimum SD in log space
  Type sd_floor_lgt = Type(0.1);   // minimum SD in logit space

  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Initial conditions from observations to avoid being parameters (no leakage beyond t=0)
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // ----------
  // DYNAMICS
  // ----------
  Type nll = Type(0.0); // negative log-likelihood accumulator
  Type pen = Type(0.0); // smooth penalties accumulator

  for (int t = 1; t < n; t++) {
    // Previous states (no data leakage)
    Type A_prev = cots_pred(t - 1); // COTS density (ind/m^2)
    Type F_prev = fast_pred(t - 1); // Fast coral cover (%)
    Type S_prev = slow_pred(t - 1); // Slow coral cover (%)

    // Temperature at previous step (affects transitions to t)
    Type Tprev = sst_dat(t - 1);

    // Food saturation index (0..1), smooth and saturating in coral availability
    Type Food_fast = F_prev / (KF_food + F_prev + eps);
    Type Food_slow = S_prev / (KS_food + S_prev + eps);
    // Clamp prefF to [0,1] via invlogit transformation inside computation (keeps parameter free but effect bounded)
    Type prefF_eff = invlogit(prefF);
    Type FoodSat = prefF_eff * Food_fast + (Type(1) - prefF_eff) * Food_slow;

    // COTS survival with temperature effect and starvation penalty
    Type sA_base = invlogit(sA_logit + sA_T * (Tprev - Topt_S));
    Type sA_eff  = sA_base * exp(-m_starv * (Type(1) - FoodSat)); // extra mortality when FoodSat is low

    // Recruitment modifiers
    Type R_temp = Type(1) + a_temp * exp(-Type(0.5) * square((Tprev - Topt_R) / (sigmaT_R + eps)));
    Type R_food = Type(1) + a_food * FoodSat;

    // Ricker recruitment (boom potential) with density dependence (bust control)
    Type Recruit = rA * A_prev * exp(-betaA * A_prev) * R_temp * R_food;

    // Immigration pulse (episodic trigger) uses previous year's forcing
    Type Immigr = gamma_imm * cotsimm_dat(t - 1);

    // Update COTS state
    Type A_new = sA_eff * A_prev + Recruit + Immigr;
    A_new = CppAD::CondExpLt(A_new, eps, eps, A_new); // keep strictly positive

    // Thermal bleaching ramps (smooth) applied to coral
    Type ramp_bleach = softplus((Tprev - T_bleach) / (delta_bleach + eps)); // >= 0
    Type BleachF = bF_bleach * ramp_bleach;
    Type BleachS = bS_bleach * ramp_bleach;

    // Coral growth (space-limited, shared carrying capacity at 100% cover)
    Type crowd = (F_prev + S_prev) / Type(100.0);
    Type growF = rF * F_prev * (Type(1) - crowd);
    Type growS = rS * S_prev * (Type(1) - crowd);

    // COTS predation (functional responses), smoothed cap so removal <= current stock
    Type numF = pow(F_prev + eps, hF);
    Type denF = pow(KF + eps, hF) + numF;
    Type rawPredF = aF * A_prev * numF / (denF + eps);
    Type PredF = smooth_cap_flux(F_prev, rawPredF, eps);

    Type numS = pow(S_prev + eps, hS);
    Type denS = pow(KS + eps, hS) + numS;
    Type rawPredS = aS * A_prev * numS / (denS + eps);
    Type PredS = smooth_cap_flux(S_prev, rawPredS, eps);

    // Update coral states
    Type F_new = F_prev + growF - PredF - muF * F_prev - BleachF * F_prev;
    Type S_new = S_prev + growS - PredS - muS * S_prev - BleachS * S_prev;

    // Keep within feasible domain (>= eps)
    F_new = CppAD::CondExpLt(F_new, eps, eps, F_new);
    S_new = CppAD::CondExpLt(S_new, eps, eps, S_new);

    // Assign predictions
    cots_pred(t) = A_new;
    fast_pred(t) = F_new;
    slow_pred(t) = S_new;
  }

  // --------------------------
  // OBSERVATION LIKELIHOODS
  // --------------------------
  // Use all observations; apply floors to standard deviations
  Type sd_log_cots  = sqrt(square(sigma_logCOTS) + square(sd_floor_log));
  Type sd_lgt_fast  = sqrt(square(sigma_logitF)  + square(sd_floor_lgt));
  Type sd_lgt_slow  = sqrt(square(sigma_logitS)  + square(sd_floor_lgt));

  for (int t = 0; t < n; t++) {
    // COTS (lognormal)
    Type y_c = log(cots_dat(t) + eps);
    Type mu_c = log(cots_pred(t) + eps);
    nll -= dnorm(y_c, mu_c, sd_log_cots, true);

    // Fast coral (logit-normal on proportion)
    Type yF_prop  = (fast_dat(t)) / Type(100.0);
    Type muF_prop = (fast_pred(t)) / Type(100.0);
    Type yF_lgt   = logit01(yF_prop, eps);
    Type muF_lgt  = logit01(muF_prop, eps);
    nll -= dnorm(yF_lgt, muF_lgt, sd_lgt_fast, true);

    // Slow coral (logit-normal on proportion)
    Type yS_prop  = (slow_dat(t)) / Type(100.0);
    Type muS_prop = (slow_pred(t)) / Type(100.0);
    Type yS_lgt   = logit01(yS_prop, eps);
    Type muS_lgt  = logit01(muS_prop, eps);
    nll -= dnorm(yS_lgt, muS_lgt, sd_lgt_slow, true);
  }

  // --------------------------
  // SMOOTH PARAMETER PENALTIES
  // --------------------------
  // Suggested biological ranges (encoded as penalties, not hard constraints)
  pen += bound_penalty(rF,        Type(0.0),  Type(2.0),    penalty_weight);
  pen += bound_penalty(rS,        Type(0.0),  Type(1.0),    penalty_weight);
  pen += bound_penalty(muF,       Type(0.0),  Type(1.0),    penalty_weight);
  pen += bound_penalty(muS,       Type(0.0),  Type(1.0),    penalty_weight);
  pen += bound_penalty(aF,        Type(0.0),  Type(50.0),   penalty_weight);
  pen += bound_penalty(aS,        Type(0.0),  Type(50.0),   penalty_weight);
  pen += bound_penalty(KF,        Type(0.1),  Type(80.0),   penalty_weight);
  pen += bound_penalty(KS,        Type(0.1),  Type(80.0),   penalty_weight);
  pen += bound_penalty(hF,        Type(1.0),  Type(4.0),    penalty_weight);
  pen += bound_penalty(hS,        Type(1.0),  Type(4.0),    penalty_weight);
  pen += bound_penalty(bF_bleach, Type(0.0),  Type(50.0),   penalty_weight);
  pen += bound_penalty(bS_bleach, Type(0.0),  Type(50.0),   penalty_weight);
  pen += bound_penalty(T_bleach,  Type(25.0), Type(33.0),   penalty_weight);
  pen += bound_penalty(delta_bleach, Type(0.1), Type(3.0),  penalty_weight);
  // prefF is passed through invlogit inside model; keep raw prefF ~ (-5..5) for numerical stability
  pen += bound_penalty(prefF,     Type(-10.0), Type(10.0),  penalty_weight);
  pen += bound_penalty(KF_food,   Type(0.1),  Type(80.0),   penalty_weight);
  pen += bound_penalty(KS_food,   Type(0.1),  Type(80.0),   penalty_weight);
  pen += bound_penalty(sA_logit,  Type(-5.0), Type(5.0),    penalty_weight);
  pen += bound_penalty(sA_T,      Type(-2.0), Type(2.0),    penalty_weight);
  pen += bound_penalty(Topt_S,    Type(24.0), Type(32.0),   penalty_weight);
  pen += bound_penalty(m_starv,   Type(0.0),  Type(10.0),   penalty_weight);
  pen += bound_penalty(rA,        Type(0.0),  Type(10.0),   penalty_weight);
  pen += bound_penalty(betaA,     Type(0.0),  Type(20.0),   penalty_weight);
  pen += bound_penalty(a_food,    Type(0.0),  Type(10.0),   penalty_weight);
  pen += bound_penalty(Topt_R,    Type(24.0), Type(32.0),   penalty_weight);
  pen += bound_penalty(sigmaT_R,  Type(0.2),  Type(5.0),    penalty_weight);
  pen += bound_penalty(a_temp,    Type(0.0),  Type(10.0),   penalty_weight);
  pen += bound_penalty(gamma_imm, Type(0.0),  Type(10.0),   penalty_weight);
  pen += bound_penalty(sigma_logCOTS, Type(0.01), Type(2.0), penalty_weight);
  pen += bound_penalty(sigma_logitF,  Type(0.01), Type(2.0), penalty_weight);
  pen += bound_penalty(sigma_logitS,  Type(0.01), Type(2.0), penalty_weight);

  nll += pen;

  // -------------
  // REPORTING
  // -------------
  REPORT(Year);       // For alignment in outputs
  REPORT(cots_pred);  // Predicted COTS density (ind/m^2)
  REPORT(fast_pred);  // Predicted fast coral cover (%)
  REPORT(slow_pred);  // Predicted slow coral cover (%)

  // Optionally report intermediate components for diagnostics
  REPORT(sd_log_cots);
  REPORT(sd_lgt_fast);
  REPORT(sd_lgt_slow);
  REPORT(pen);

  return nll;
}
