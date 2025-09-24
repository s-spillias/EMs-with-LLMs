#include <TMB.hpp>

// Smooth helper: inverse logit
template<class Type>
Type invlogit(Type x) {
  return Type(1.0) / (Type(1.0) + exp(-x));
}

// Smooth helper: softplus for non-negativity
template<class Type>
Type softplus(Type x) {
  return log(Type(1.0) + exp(x));
}

// Smooth saturation to [0, high): maps real x smoothly into [0, high)
template<class Type>
Type smooth_sat_upper(Type x, Type high) {
  Type xp = softplus(x);                                 // >= 0 smoothly
  return high * xp / (xp + high + Type(1e-8));           // in [0, high)
}

// Smooth penalty if x outside [low, high]; weight w controls strength
template<class Type>
Type smooth_bounds_penalty(Type x, Type low, Type high, Type w) {
  // Penalize lower violations
  Type pl = softplus((low - x) * w);
  // Penalize upper violations
  Type pu = softplus((x - high) * w);
  return pl + pu;
}

/*
EQUATION SUMMARY (annual time step; t indexes years):
1) Temperature response (Gaussian):
   g_T_X(t) = exp(-0.5 * ((SST(t) - Topt_X) / Tsig_X)^2)  for X in {fast, slow, cots}
2) Bleaching risk (smooth threshold):
   B(t) = 1 / (1 + exp(-k_bleach * (SST(t) - T_bleach)))
3) Food limitation for COTS (saturating by prey availability; F and S are proportions):
   Food(t) = (wF * F(t) + wS * S(t)) / (k_food + wF * F(t) + wS * S(t) + eps)
4) Allee effect on COTS reproduction (smooth):
   A(C) = A_min + (1 - A_min) * invlogit(allee_k * (C - C_crit))
5) Preference-weighted availability for predation (q >= 1):
   Avail(t) = pF * F(t)^q + pS * S(t)^q + eps
6) Per-capita COTS intake (Holling II with handling time h):
   I*(t) = a * Avail(t) / (1 + a * h * Avail(t))
7) Total COTS grazing pressure (proportion of coral cover per year):
   G_total(t) = C(t) * I*(t)
   Allocation to fast/slow by preference and availability:
   G_fast(t) = e_predF * G_total(t) * (pF * F(t)^q) / Avail(t)
   G_slow(t) = e_predS * G_total(t) * (pS * S(t)^q) / Avail(t)
8) Coral dynamics (logistic growth, minus predation and bleaching mortality), in % cover:
   F_next_raw = F + rF * g_T_fast(t) * F * (1 - (F + S) / Kc) - Kc * G_fast(t) - m_bleachF * B(t) * F
   S_next_raw = S + rS * g_T_slow(t) * S * (1 - (F + S) / Kc) - Kc * G_slow(t) - m_bleachS * B(t) * S
   F_next = smooth_sat_upper(F_next_raw, Kc); S_next = smooth_sat_upper(S_next_raw, Kc)
9) COTS dynamics (Ricker-like with resource and SST modulation, Allee, and crowding; plus immigration), in individuals/m^2:
   g_pc(t) = r_cots * Food(t) * g_T_cots(t) * A(C(t)) - beta * C(t)
   C_next = C * exp(g_pc(t)) + immigration(t) + eps

OBSERVATION MODEL (lognormal, all data strictly positive after adding small constant):
   log(y_dat(t) + eps) ~ Normal(log(y_pred(t) + eps), sd_eff)
   with sd_eff^2 = exp(2 * log_sd) + min_sd^2
*/

template<class Type>
Type objective_function<Type>::operator() () {
  Type nll = 0.0;                                        // total negative log-likelihood
  const Type eps = Type(1e-8);                           // numerical stabilizer

  // -----------------------------
  // DATA INPUTS
  // -----------------------------
  DATA_VECTOR(Year);                                     // observation years (calendar years)
  DATA_VECTOR(cots_dat);                                 // observed adult COTS (individuals/m^2)
  DATA_VECTOR(fast_dat);                                 // observed fast coral cover (%)
  DATA_VECTOR(slow_dat);                                 // observed slow coral cover (%)
  DATA_VECTOR(sst_dat);                                  // observed SST (Celsius), external forcing
  DATA_VECTOR(cotsimm_dat);                              // observed larval immigration (indiv/m^2/yr), external forcing

  int nT = cots_dat.size();                              // number of time steps, must match other series
  // Soft penalty if input lengths do not match; we still use min length to compute predictions
  int nT_force = sst_dat.size();
  int nT_imm = cotsimm_dat.size();
  int nT_fast = fast_dat.size();
  int nT_slow = slow_dat.size();
  int nT_common = CppAD::Integer(CppAD::CondExpLt(nT, nT_force, nT, nT_force));
  nT_common = CppAD::Integer(CppAD::CondExpLt(nT_common, nT_imm, nT_common, nT_imm));
  nT_common = CppAD::Integer(CppAD::CondExpLt(nT_common, nT_fast, nT_common, nT_fast));
  nT_common = CppAD::Integer(CppAD::CondExpLt(nT_common, nT_slow, nT_common, nT_slow));
  if (nT_common != nT || nT_common != nT_force || nT_common != nT_imm || nT_common != nT_fast || nT_common != nT_slow) {
    // Smoothly increase nll if sizes mismatch (encourages consistent inputs without crashing)
    nll += softplus(Type(0.5) * fabs(Type(nT - nT_common))) +
           softplus(Type(0.5) * fabs(Type(nT_force - nT_common))) +
           softplus(Type(0.5) * fabs(Type(nT_imm - nT_common))) +
           softplus(Type(0.5) * fabs(Type(nT_fast - nT_common))) +
           softplus(Type(0.5) * fabs(Type(nT_slow - nT_common)));
  }
  int T = nT_common;                                     // use the common length safely

  // -----------------------------
  // PARAMETERS (ecological rates, preferences, temperature effects)
  // -----------------------------
  PARAMETER(log_rF);                                     // log of fast coral intrinsic growth rate rF (yr^-1); init from literature on Acropora recovery rates
  PARAMETER(log_rS);                                     // log of slow coral intrinsic growth rate rS (yr^-1); init from literature on massive coral recovery rates
  PARAMETER(K_coral);                                    // total coral carrying capacity Kc (% cover, 0-100); init from site expectations

  PARAMETER(log_a);                                      // log attack/search rate a (yr^-1, in proportion units); init from COTS feeding studies
  PARAMETER(log_h);                                      // log handling time h (yr); controls max intake 1/h
  PARAMETER(logit_pF);                                   // logit preference weight for fast coral (0-1); higher => stronger selection for Acropora
  PARAMETER(logit_pS);                                   // logit preference weight for slow coral (0-1); complements pF but free to learn selectivity
  PARAMETER(log_q);                                      // log of functional response exponent q (>= 1); q=1 Type II, q>1 Type III-like
  PARAMETER(logit_e_predF);                              // logit of consumption-to-cover loss efficiency for fast coral (0-1); scale uncertainty in grazing effect
  PARAMETER(logit_e_predS);                              // logit of consumption-to-cover loss efficiency for slow coral (0-1)

  PARAMETER(log_r_cots);                                 // log maximum per-capita COTS growth rate r_cots (yr^-1); integrates fecundity and survival
  PARAMETER(log_beta);                                   // log crowding coefficient beta (m^2/indiv/yr); strength of self-limitation
  PARAMETER(logit_k_food);                               // logit of food half-saturation k_food (proportion, 0-1); resource threshold for COTS recruitment
  PARAMETER(log_wF);                                     // log weight for fast coral in food function wF (dimensionless, >=0)
  PARAMETER(log_wS);                                     // log weight for slow coral in food function wS (dimensionless, >=0)
  PARAMETER(logit_A_min);                                // logit of minimum Allee multiplier A_min (0-1); lower bound on recruitment at very low density
  PARAMETER(C_crit);                                     // critical adult density for Allee inflection C_crit (indiv/m^2); where recruitment accelerates
  PARAMETER(log_allee_k);                                // log steepness of Allee curve allee_k (m^2/indiv); larger => sharper transition

  PARAMETER(Topt_fast);                                  // optimal SST for fast coral growth (C); center of Gaussian response
  PARAMETER(Tsig_fast);                                  // width (sd) of SST response for fast coral (C); tolerance to deviations
  PARAMETER(Topt_slow);                                  // optimal SST for slow coral growth (C)
  PARAMETER(Tsig_slow);                                  // width (sd) for slow coral SST response (C)
  PARAMETER(Topt_cots);                                  // optimal SST for COTS recruitment (C)
  PARAMETER(Tsig_cots);                                  // width (sd) for COTS SST response (C)
  PARAMETER(T_bleach);                                   // bleaching threshold temperature (C); logistic midpoint for bleaching risk
  PARAMETER(log_k_bleach);                               // log slope of bleaching logistic function (1/C); controls smoothness of onset
  PARAMETER(logit_m_bleachF);                            // logit of additional annual mortality fraction on fast coral due to bleaching (0-1)
  PARAMETER(logit_m_bleachS);                            // logit of additional annual mortality fraction on slow coral due to bleaching (0-1)

  // Observation model standard deviations (lognormal)
  PARAMETER(log_sd_cots);                                // log sd for COTS observation error (log scale); absorbs sampling/measurement variance
  PARAMETER(log_sd_fast);                                // log sd for fast coral observation error (log scale)
  PARAMETER(log_sd_slow);                                // log sd for slow coral observation error (log scale)

  // Initial conditions (t = Year[0])
  PARAMETER(log_C0);                                     // log initial adult COTS abundance (indiv/m^2)
  PARAMETER(logit_coral_total0);                         // logit of initial total coral fraction of Kc (0-1)
  PARAMETER(logit_fast_share0);                          // logit fraction of total coral that is fast (0-1)

  // -----------------------------
  // TRANSFORMED PARAMETERS
  // -----------------------------
  // Ecological rates and preferences
  Type rF = exp(log_rF);                                 // rF in yr^-1, positive
  Type rS = exp(log_rS);                                 // rS in yr^-1, positive
  Type Kc = K_coral;                                     // total coral carrying capacity (%)
  Type a = exp(log_a);                                   // attack/search rate (yr^-1)
  Type h = exp(log_h);                                   // handling time (yr)
  Type pF = invlogit(logit_pF);                          // preference for fast coral (0-1)
  Type pS = invlogit(logit_pS);                          // preference for slow coral (0-1)
  Type q = exp(log_q) + Type(1.0);                       // exponent >= 1 (shifted to guarantee >=1)
  Type e_predF = invlogit(logit_e_predF);                // efficiency on fast (0-1)
  Type e_predS = invlogit(logit_e_predS);                // efficiency on slow (0-1)

  // COTS demography and food limitation
  Type r_cots = exp(log_r_cots);                         // per-capita max growth (yr^-1)
  Type beta = exp(log_beta);                             // crowding coefficient
  Type k_food = invlogit(logit_k_food);                  // half-saturation (proportion 0-1)
  Type wF = exp(log_wF);                                 // weight >= 0
  Type wS = exp(log_wS);                                 // weight >= 0
  Type A_min = invlogit(logit_A_min);                    // minimum Allee multiplier (0-1)
  Type allee_k = exp(log_allee_k);                       // Allee steepness

  // Temperature effects
  Type k_bleach = exp(log_k_bleach);                     // slope for bleaching logistic (1/C)
  Type m_bleachF = invlogit(logit_m_bleachF);            // annual bleaching mortality fraction for fast coral (0-1)
  Type m_bleachS = invlogit(logit_m_bleachS);            // annual bleaching mortality fraction for slow coral (0-1)

  // Observation model
  const Type min_sd = Type(0.05);                        // minimum SD on log scale to avoid degeneracy
  Type sd_cots = sqrt(exp(2.0 * log_sd_cots) + min_sd * min_sd);  // effective lognormal sd
  Type sd_fast = sqrt(exp(2.0 * log_sd_fast) + min_sd * min_sd);  // effective lognormal sd
  Type sd_slow = sqrt(exp(2.0 * log_sd_slow) + min_sd * min_sd);  // effective lognormal sd

  // Initial states
  Type C = exp(log_C0);                                  // initial COTS (indiv/m^2)
  Type coral_frac0 = invlogit(logit_coral_total0);       // fraction of Kc (0-1)
  Type fast_share0 = invlogit(logit_fast_share0);        // fraction (0-1)
  Type F = Kc * coral_frac0 * fast_share0;               // initial fast coral (%)
  Type S = Kc * coral_frac0 * (Type(1.0) - fast_share0); // initial slow coral (%)

  // -----------------------------
  // SOFT BIOLOGICAL BOUNDS (penalties; no hard constraints)
  // -----------------------------
  Type pen = 0.0;                                        // accumulate smooth penalties
  pen += smooth_bounds_penalty(rF, Type(0.01), Type(2.0), Type(5.0));
  pen += smooth_bounds_penalty(rS, Type(0.005), Type(1.0), Type(5.0));
  pen += smooth_bounds_penalty(Kc, Type(10.0), Type(100.0), Type(0.5));
  pen += smooth_bounds_penalty(a, Type(0.001), Type(50.0), Type(1.0));
  pen += smooth_bounds_penalty(h, Type(0.01), Type(10.0), Type(1.0));
  pen += smooth_bounds_penalty(pF, Type(0.0), Type(1.0), Type(10.0));
  pen += smooth_bounds_penalty(pS, Type(0.0), Type(1.0), Type(10.0));
  pen += smooth_bounds_penalty(q, Type(1.0), Type(3.0), Type(2.0));
  pen += smooth_bounds_penalty(e_predF, Type(0.0), Type(1.0), Type(10.0));
  pen += smooth_bounds_penalty(e_predS, Type(0.0), Type(1.0), Type(10.0));
  pen += smooth_bounds_penalty(r_cots, Type(0.05), Type(5.0), Type(1.0));
  pen += smooth_bounds_penalty(beta, Type(0.0), Type(5.0), Type(1.0));
  pen += smooth_bounds_penalty(k_food, Type(0.001), Type(0.9), Type(5.0));
  pen += smooth_bounds_penalty(wF, Type(0.1), Type(5.0), Type(1.0));
  pen += smooth_bounds_penalty(wS, Type(0.01), Type(5.0), Type(1.0));
  pen += smooth_bounds_penalty(A_min, Type(0.0), Type(0.5), Type(10.0));
  pen += smooth_bounds_penalty(C_crit, Type(0.0), Type(2.0), Type(2.0));
  pen += smooth_bounds_penalty(allee_k, Type(0.1), Type(10.0), Type(1.0));
  pen += smooth_bounds_penalty(Topt_fast, Type(20.0), Type(33.0), Type(0.5));
  pen += smooth_bounds_penalty(Topt_slow, Type(20.0), Type(33.0), Type(0.5));
  pen += smooth_bounds_penalty(Topt_cots, Type(20.0), Type(33.0), Type(0.5));
  pen += smooth_bounds_penalty(Tsig_fast, Type(0.1), Type(5.0), Type(1.0));
  pen += smooth_bounds_penalty(Tsig_slow, Type(0.1), Type(5.0), Type(1.0));
  pen += smooth_bounds_penalty(Tsig_cots, Type(0.1), Type(5.0), Type(1.0));
  pen += smooth_bounds_penalty(T_bleach, Type(26.0), Type(33.0), Type(1.0));
  pen += smooth_bounds_penalty(k_bleach, Type(0.1), Type(10.0), Type(1.0));
  pen += smooth_bounds_penalty(m_bleachF, Type(0.0), Type(1.0), Type(5.0));
  pen += smooth_bounds_penalty(m_bleachS, Type(0.0), Type(1.0), Type(5.0));

  // -----------------------------
  // STATE VECTORS FOR REPORTING
  // -----------------------------
  vector<Type> cots_pred(T);                             // predicted COTS (indiv/m^2)
  vector<Type> fast_pred(T);                             // predicted fast coral (%)
  vector<Type> slow_pred(T);                             // predicted slow coral (%)

  // Set predictions at t = 0 (initial state)
  if (T > 0) {
    cots_pred(0) = C;
    fast_pred(0) = F;
    slow_pred(0) = S;
  }

  // -----------------------------
  // TIME LOOP: STATE UPDATES
  // -----------------------------
  for (int t = 0; t < T - 1; ++t) {
    // Forcing at time t
    Type sst = sst_dat(t);                               // SST at year t (C)
    Type imm = cotsimm_dat(t);                           // immigration at t (indiv/m^2/yr), >= 0 assumed

    // Proportional coral cover (0-1) based on Kc
    Type Fp = F / (Kc + eps);                            // fast coral proportion
    Type Sp = S / (Kc + eps);                            // slow coral proportion

    // Temperature responses (Gaussian)
    Type gT_fast = exp(-Type(0.5) * pow((sst - Topt_fast) / (Tsig_fast + eps), 2.0)); // 0-1
    Type gT_slow = exp(-Type(0.5) * pow((sst - Topt_slow) / (Tsig_slow + eps), 2.0)); // 0-1
    Type gT_cots = exp(-Type(0.5) * pow((sst - Topt_cots) / (Tsig_cots + eps), 2.0)); // 0-1

    // Bleaching probability (smooth logistic)
    Type B = invlogit(k_bleach * (sst - T_bleach));      // 0-1

    // Preference-weighted availability for predation
    Type Avail = pF * pow(Fp + eps, q) + pS * pow(Sp + eps, q) + eps;

    // Per-capita COTS intake (Holling II)
    Type Istar = a * Avail / (Type(1.0) + a * h * Avail + eps); // proportion per starfish per year

    // Total grazing pressure and allocation to coral groups
    Type G_total = C * Istar;                            // proportion of coral removed per year
    Type share_fast = (pF * pow(Fp + eps, q)) / (Avail); // fraction of grazing on fast
    Type share_slow = (pS * pow(Sp + eps, q)) / (Avail); // fraction of grazing on slow
    Type G_fast = e_predF * G_total * share_fast;        // proportion of Kc
    Type G_slow = e_predS * G_total * share_slow;        // proportion of Kc

    // Coral dynamics (raw updates, then smooth saturation to [0, Kc))
    Type growthF = rF * gT_fast * F * (Type(1.0) - (F + S) / (Kc + eps)); // logistic growth
    Type growthS = rS * gT_slow * S * (Type(1.0) - (F + S) / (Kc + eps)); // logistic growth
    Type bleachMortF = m_bleachF * B * F;               // bleaching mortality on fast
    Type bleachMortS = m_bleachS * B * S;               // bleaching mortality on slow

    Type F_next_raw = F + growthF - Kc * G_fast - bleachMortF; // % cover
    Type S_next_raw = S + growthS - Kc * G_slow - bleachMortS; // % cover

    Type F_next = smooth_sat_upper(F_next_raw, Kc);      // keep in [0, Kc) smoothly
    Type S_next = smooth_sat_upper(S_next_raw, Kc);      // keep in [0, Kc) smoothly

    // COTS dynamics (Ricker-like)
    Type Food = (wF * Fp + wS * Sp) / (k_food + wF * Fp + wS * Sp + eps); // 0-1
    Type A_allee = A_min + (Type(1.0) - A_min) * invlogit(allee_k * (C - C_crit)); // 0-1+
    Type g_pc = r_cots * Food * gT_cots * A_allee - beta * C; // per-capita log growth
    Type C_next = C * exp(g_pc) + imm + eps;             // indiv/m^2, strictly positive

    // Advance states (predictions never use current observations; only previous predicted states and forcing)
    F = F_next;
    S = S_next;
    C = C_next;

    // Save predictions for next time index
    cots_pred(t + 1) = C;
    fast_pred(t + 1) = F;
    slow_pred(t + 1) = S;
  }

  // -----------------------------
  // LIKELIHOOD: all observations included (lognormal)
  // -----------------------------
  for (int t = 0; t < T; ++t) {
    // COTS
    Type yC = cots_dat(t) + eps;
    Type muC = cots_pred(t) + eps;
    nll -= dnorm(log(yC), log(muC), sd_cots, true);

    // Fast coral
    Type yF = fast_dat(t) + eps;
    Type muF = fast_pred(t) + eps;
    nll -= dnorm(log(yF), log(muF), sd_fast, true);

    // Slow coral
    Type yS = slow_dat(t) + eps;
    Type muS = slow_pred(t) + eps;
    nll -= dnorm(log(yS), log(muS), sd_slow, true);
  }

  // Add penalties
  nll += pen;

  // -----------------------------
  // REPORTS
  // -----------------------------
  REPORT(cots_pred);                                     // predicted adults (indiv/m^2)
  REPORT(fast_pred);                                     // predicted fast coral cover (%)
  REPORT(slow_pred);                                     // predicted slow coral cover (%)
  REPORT(pen);                                           // total smooth penalty
  REPORT(Kc);                                            // report realized Kc for interpretability
  REPORT(rF);
  REPORT(rS);
  REPORT(a);
  REPORT(h);
  REPORT(pF);
  REPORT(pS);
  REPORT(q);
  REPORT(e_predF);
  REPORT(e_predS);
  REPORT(r_cots);
  REPORT(beta);
  REPORT(k_food);
  REPORT(wF);
  REPORT(wS);
  REPORT(A_min);
  REPORT(C_crit);
  REPORT(allee_k);
  REPORT(Topt_fast);
  REPORT(Tsig_fast);
  REPORT(Topt_slow);
  REPORT(Tsig_slow);
  REPORT(Topt_cots);
  REPORT(Tsig_cots);
  REPORT(T_bleach);
  REPORT(k_bleach);
  REPORT(m_bleachF);
  REPORT(m_bleachS);
  REPORT(sd_cots);
  REPORT(sd_fast);
  REPORT(sd_slow);

  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);

  return nll;
}
