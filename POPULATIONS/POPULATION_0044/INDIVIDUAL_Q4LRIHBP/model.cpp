#include <TMB.hpp>

// Helper functions with smoothness and numerical stability

template<class Type>
Type invlogit(Type x) { // Smooth map from R to (0,1)
  return Type(1) / (Type(1) + exp(-x));
}

template<class Type>
Type logit(Type p) { // Stable logit with epsilon protection
  Type eps = Type(1e-8);
  p = CppAD::CondExpLt(p, eps, eps, p);
  p = CppAD::CondExpGt(p, Type(1) - eps, Type(1) - eps, p);
  return log(p / (Type(1) - p));
}

template<class Type>
Type sqr(Type x) { return x * x; } // Square

template<class Type>
Type soft_penalty(Type x, Type L, Type U, Type k, Type w) {
  // Smoothly penalize deviations outside [L,U] using softplus-like tails
  // k: steepness, w: weight
  Type pL = log1p(exp(k * (L - x))) / k;
  Type pU = log1p(exp(k * (x - U))) / k;
  return w * (sqr(pL) + sqr(pU));
}

/*
Numbered equations (discrete yearly steps, t = 1..T-1; all state updates use t-1 values):
1) Temperature modifier (0..1):
   temp_mod_t = exp(-0.5 * ((sst_{t-1} - T_opt) / T_sd)^2)

2) Food index from coral cover (0..1 saturating):
   fF = F_{t-1} / (KF_half + F_{t-1}); fS = S_{t-1} / (KS_half + S_{t-1})
   food_raw = wF_food * fF + wS_food * fS
   food_mod_t = 1 - exp(-e_food * food_raw)

3) Allee effect (0..1):
   allee_mod_t = invlogit(k_allee * (C_{t-1} - C_crit))

4) Per-capita COTS growth (Ricker-like):
   g_t = rC_max * temp_mod_t * food_mod_t * allee_mod_t - mC - betaC * C_{t-1}

5) Effective immigration (individuals/m2/year):
   I_eff_t = eta_imm * cotsimm_{t-1} * temp_mod_t * (0.5 + 0.5 * food_mod_t)

6) COTS density update (individuals/m2):
   C_t = C_{t-1} * exp(g_t) + I_eff_t + eps

7) Coral predation (Beddington–DeAngelis; cover fractions per year):
   MortF_t = qF * C_{t-1} * F_{t-1} / (HF + F_{t-1} + phiF * S_{t-1} + gamma_pred * C_{t-1} + eps)
   MortS_t = qS * C_{t-1} * S_{t-1} / (HS + S_{t-1} + phiS * F_{t-1} + gamma_pred * C_{t-1} + eps)

8) Bleaching mortality ramp with SST (0..1 risk):
   bleach_risk_t = invlogit(b_slope * (sst_{t-1} - T_bleach))
   BleachF_t = bF_max * bleach_risk_t * F_{t-1}
   BleachS_t = bS_max * bleach_risk_t * S_{t-1}

9) Coral logistic growth with competition (fractions per year):
   growthF_t = rF * F_{t-1} * (1 - (F_{t-1} + comp_FS * S_{t-1}) / K_tot_prop)
   growthS_t = rS * S_{t-1} * (1 - (S_{t-1} + comp_SF * F_{t-1}) / K_tot_prop)

10) Coral updates with smooth bounding to (0,1):
   F_raw = F_{t-1} + growthF_t - MortF_t - BleachF_t
   S_raw = S_{t-1} + growthS_t - MortS_t - BleachS_t
   F_t = smooth_clip01(F_raw); S_t = smooth_clip01(S_raw)

Observation models:
- COTS (strictly positive): lognormal, log(y_t) ~ N(log(C_t), sigma_cots_obs)
- Coral covers (0..1): Normal on logit scale, logit(y_t) ~ N(logit(F_t), sigma_fast_obs) and similarly for S_t
*/

template<class Type>
Type objective_function<Type>::operator() ()
{
  // -----------------------------
  // DATA INPUTS (read-only)
  // -----------------------------
  DATA_VECTOR(Year);         // Year (calendar year), used as the time index
  DATA_VECTOR(sst_dat);      // Sea-Surface Temperature (deg C), exogenous forcing
  DATA_VECTOR(cotsimm_dat);  // Larval immigration rate (individuals/m2/year), exogenous forcing
  DATA_VECTOR(cots_dat);     // Observed adult COTS density (individuals/m2)
  DATA_VECTOR(fast_dat);     // Observed fast coral cover (% of substrate; Acropora)
  DATA_VECTOR(slow_dat);     // Observed slow coral cover (% of substrate; Faviidae/Porites)

  int T = Year.size();       // Number of time steps (years)
  Type eps = Type(1e-8);     // Small constant to avoid division by zero
  Type eps_prop = Type(1e-6);// Small constant for proportions to keep within (0,1)

  // -----------------------------
  // PARAMETERS (all lines annotated)
  // -----------------------------
  PARAMETER(log_C0);    // log initial adult COTS density at Year[0] (log(individuals/m2)); estimated from initial conditions
  PARAMETER(zF0);       // initial fast coral proportion (logit scale; unitless); transforms via invlogit to 0-1
  PARAMETER(zS0);       // initial slow coral proportion (logit scale; unitless); transforms via invlogit to 0-1

  PARAMETER(rC_max);    // maximum COTS per-capita growth rate (year^-1); literature/expert ranges 0–5
  PARAMETER(mC);        // baseline COTS natural mortality (year^-1); typical 0–3
  PARAMETER(betaC);     // COTS density dependence strength in Ricker term (m^2 individual^-1 year^-1); larger -> stronger self-limitation
  PARAMETER(e_food);    // scaling of food index to reproduction (dimensionless); controls saturation strength
  PARAMETER(wF_food);   // weight of fast coral in food index (dimensionless); reflects dietary preference for Acropora
  PARAMETER(wS_food);   // weight of slow coral in food index (dimensionless); lower than wF_food
  PARAMETER(KF_half);   // half-saturation for fast coral contribution (proportion 0–1); where contribution is half-max
  PARAMETER(KS_half);   // half-saturation for slow coral contribution (proportion 0–1)
  PARAMETER(T_opt);     // optimal SST for COTS performance (deg C); peak of thermal performance curve
  PARAMETER(T_sd);      // width (sd) of thermal performance curve (deg C); breadth of thermal tolerance
  PARAMETER(k_allee);   // steepness of Allee effect (1 / (individuals/m2)); higher values sharpen low-density limitation
  PARAMETER(C_crit);    // COTS density at which growth transitions (individuals/m2); Allee inflection
  PARAMETER(eta_imm);   // efficiency converting larval immigration to local recruits (dimensionless, 0–1); survival to adult class

  PARAMETER(rF);        // intrinsic growth rate of fast coral (year^-1); typical 0–1
  PARAMETER(rS);        // intrinsic growth rate of slow coral (year^-1); typical 0–0.5
  PARAMETER(K_tot_prop);// total coral carrying capacity as a proportion of substrate (0–1); e.g., 0.5–0.9
  PARAMETER(comp_FS);   // competition coefficient: effect of slow coral on fast coral carrying term (dimensionless)
  PARAMETER(comp_SF);   // competition coefficient: effect of fast coral on slow coral carrying term (dimensionless)

  PARAMETER(qF);        // predation coefficient on fast coral (cover loss per COTS per year); preference strength
  PARAMETER(qS);        // predation coefficient on slow coral (cover loss per COTS per year); lower than qF
  PARAMETER(HF);        // half-saturation term for fast coral in Beddington–DeAngelis denominator (proportion 0–1)
  PARAMETER(HS);        // half-saturation term for slow coral in Beddington–DeAngelis denominator (proportion 0–1)
  PARAMETER(phiF);      // cross-saturation of slow coral on fast coral predation denominator (dimensionless)
  PARAMETER(phiS);      // cross-saturation of fast coral on slow coral predation denominator (dimensionless)
  PARAMETER(gamma_pred);// predator interference coefficient (per (individuals/m2)); reduces per-capita consumption at high COTS density

  PARAMETER(bF_max);    // max additional fast coral mortality from bleaching at extreme SST (year^-1)
  PARAMETER(bS_max);    // max additional slow coral mortality from bleaching at extreme SST (year^-1)
  PARAMETER(T_bleach);  // SST (deg C) at which bleaching risk = 50% (logistic midpoint)
  PARAMETER(b_slope);   // slope of bleaching logistic vs SST (1/deg C); smooth threshold behavior

  PARAMETER(log_sigma_cots_obs); // log observation SD for COTS (log scale); ensures positivity and adds stability
  PARAMETER(log_sigma_fast_obs); // log observation SD for fast coral (logit scale)
  PARAMETER(log_sigma_slow_obs); // log observation SD for slow coral (logit scale)

  // -----------------------------
  // DERIVED TRANSFORMS AND CONSTANTS
  // -----------------------------
  Type C0 = exp(log_C0);          // initial COTS density (individuals/m2), positive
  Type F0 = invlogit(zF0);        // initial fast coral proportion (0–1)
  Type S0 = invlogit(zS0);        // initial slow coral proportion (0–1)

  // Fixed minimum observation SDs to avoid degeneracy at small values
  Type sd_min_log   = Type(0.05); // minimum SD on log scale
  Type sd_min_logit = Type(0.05); // minimum SD on logit scale
  Type sigma_cots_obs = sd_min_log   + exp(log_sigma_cots_obs);  // effective SD for COTS
  Type sigma_fast_obs = sd_min_logit + exp(log_sigma_fast_obs);  // effective SD for fast coral
  Type sigma_slow_obs = sd_min_logit + exp(log_sigma_slow_obs);  // effective SD for slow coral

  // -----------------------------
  // STATE AND PREDICTION VECTORS
  // -----------------------------
  vector<Type> C(T);               // COTS density (individuals/m2)
  vector<Type> F(T);               // fast coral proportion (0–1)
  vector<Type> S(T);               // slow coral proportion (0–1)

  vector<Type> cots_dat_pred(T);   // prediction for cots_dat (individuals/m2)
  vector<Type> fast_dat_pred(T);   // prediction for fast_dat (% cover)
  vector<Type> slow_dat_pred(T);   // prediction for slow_dat (% cover)
  vector<Type> sst_dat_pred(T);    // exogenous, set equal to sst_dat
  vector<Type> cotsimm_dat_pred(T);// exogenous, set equal to cotsimm_dat

  // Initialize states at first time
  C(0) = C0;
  F(0) = F0;
  S(0) = S0;

  // Forcing "predictions" equal to data to satisfy reporting requirements for _dat/_pred
  for (int t = 0; t < T; t++) {
    sst_dat_pred(t)     = sst_dat(t);
    cotsimm_dat_pred(t) = cotsimm_dat(t);
  }

  // Smooth clip function mapping any real value to (eps_prop, 1 - eps_prop)
  auto clip01 = [&](Type x)->Type{
    Type mid = Type(0.5);
    Type z = (x - mid) / (Type(0.5) + eps);    // scaled around mid
    Type y = invlogit(z * Type(5.0));          // slope 5 provides gentle saturation
    return eps_prop + (Type(1) - Type(2) * eps_prop) * y;
  };

  // -----------------------------
  // DYNAMICS (time-forward; no data leakage)
  // -----------------------------
  for (int t = 1; t < T; t++) {
    // Previous states
    Type Cprev = C(t - 1);
    Type Fprev = F(t - 1);
    Type Sprev = S(t - 1);

    // Previous forcings
    Type sst_prev = sst_dat(t - 1);
    Type imm_prev = cotsimm_dat(t - 1);

    // (1) Temperature performance (0..1)
    Type temp_mod = exp(Type(-0.5) * sqr((sst_prev - T_opt) / (T_sd + eps)));

    // (2) Food index (0..1) via saturating contributions of F and S
    Type fF = Fprev / (KF_half + Fprev + eps);
    Type fS = Sprev / (KS_half + Sprev + eps);
    Type food_raw = wF_food * fF + wS_food * fS;
    Type food_mod = Type(1) - exp(-e_food * food_raw);

    // (3) Allee effect (0..1) smooth sigmoid
    Type allee_mod = invlogit(k_allee * (Cprev - C_crit));

    // (4) Per-capita growth (Ricker-like)
    Type g = rC_max * temp_mod * food_mod * allee_mod - mC - betaC * Cprev;

    // (5) Effective immigration modified by temperature and food
    Type I_eff = eta_imm * imm_prev * temp_mod * (Type(0.5) + Type(0.5) * food_mod);

    // (6) COTS update (ensure positivity with +eps)
    Type C_mu = Cprev * exp(g) + I_eff + eps;
    C(t) = C_mu;

    // (7) Predation on corals (Beddington–DeAngelis, selective on fast coral)
    Type denomF = HF + Fprev + phiF * Sprev + gamma_pred * Cprev + eps;
    Type denomS = HS + Sprev + phiS * Fprev + gamma_pred * Cprev + eps;
    Type MortF = qF * Cprev * Fprev / denomF; // loss of fast coral fraction per year
    Type MortS = qS * Cprev * Sprev / denomS; // loss of slow coral fraction per year

    // (8) Bleaching mortality ramp with SST
    Type bleach_risk = invlogit(b_slope * (sst_prev - T_bleach)); // 0..1
    Type BleachF = bF_max * bleach_risk * Fprev;
    Type BleachS = bS_max * bleach_risk * Sprev;

    // (9) Coral logistic growth with competition
    Type growthF = rF * Fprev * (Type(1) - (Fprev + comp_FS * Sprev) / (K_tot_prop + eps));
    Type growthS = rS * Sprev * (Type(1) - (Sprev + comp_SF * Fprev) / (K_tot_prop + eps));

    // (10) Update corals with smooth bounding to (0,1)
    Type F_raw = Fprev + growthF - MortF - BleachF;
    Type S_raw = Sprev + growthS - MortS - BleachS;
    F(t) = clip01(F_raw);
    S(t) = clip01(S_raw);
  }

  // -----------------------------
  // BUILD PREDICTIONS IN OBSERVATION UNITS
  // -----------------------------
  for (int t = 0; t < T; t++) {
    cots_dat_pred(t) = C(t);                // individuals/m2
    fast_dat_pred(t) = F(t) * Type(100.0);  // percent
    slow_dat_pred(t) = S(t) * Type(100.0);  // percent
  }

  // -----------------------------
  // LIKELIHOOD
  // -----------------------------
  Type nll = 0.0;

  // COTS observations: lognormal errors
  for (int t = 0; t < T; t++) {
    Type mu = log(cots_dat_pred(t) + eps);
    Type y  = log(cots_dat(t)      + eps);
    nll -= dnorm(y, mu, sigma_cots_obs, true);
  }

  // Coral observations: normal on logit of proportions
  for (int t = 0; t < T; t++) {
    Type pF_obs = fast_dat(t) / Type(100.0);
    Type pS_obs = slow_dat(t) / Type(100.0);

    // Smoothly bound observation proportions away from 0/1
    Type pF = eps_prop + (Type(1) - Type(2) * eps_prop) * pF_obs;
    Type pS = eps_prop + (Type(1) - Type(2) * eps_prop) * pS_obs;

    Type muF = logit(F(t));
    Type muS = logit(S(t));
    Type yF  = logit(pF);
    Type yS  = logit(pS);

    nll -= dnorm(yF, muF, sigma_fast_obs, true);
    nll -= dnorm(yS, muS, sigma_slow_obs, true);
  }

  // -----------------------------
  // SOFT BOUNDS (smooth penalties within biological ranges)
  // -----------------------------
  Type k = Type(10.0); // soft bound steepness
  Type w = Type(1.0);  // penalty weight

  nll += soft_penalty(rC_max,     Type(0.0),  Type(5.0),  k, w);
  nll += soft_penalty(mC,         Type(0.0),  Type(3.0),  k, w);
  nll += soft_penalty(betaC,      Type(0.0),  Type(10.0), k, w);
  nll += soft_penalty(e_food,     Type(0.0),  Type(10.0), k, w);
  nll += soft_penalty(wF_food,    Type(0.0),  Type(10.0), k, w);
  nll += soft_penalty(wS_food,    Type(0.0),  Type(10.0), k, w);
  nll += soft_penalty(KF_half,    Type(0.001),Type(1.0),  k, w);
  nll += soft_penalty(KS_half,    Type(0.001),Type(1.0),  k, w);
  nll += soft_penalty(T_opt,      Type(24.0), Type(32.0), k, w);
  nll += soft_penalty(T_sd,       Type(0.1),  Type(5.0),  k, w);
  nll += soft_penalty(k_allee,    Type(0.0),  Type(10.0), k, w);
  nll += soft_penalty(C_crit,     Type(0.0),  Type(2.0),  k, w);
  nll += soft_penalty(eta_imm,    Type(0.0),  Type(1.0),  k, w);

  nll += soft_penalty(rF,         Type(0.0),  Type(2.0),  k, w);
  nll += soft_penalty(rS,         Type(0.0),  Type(1.5),  k, w);
  nll += soft_penalty(K_tot_prop, Type(0.2),  Type(0.95), k, w);
  nll += soft_penalty(comp_FS,    Type(0.0),  Type(2.0),  k, w);
  nll += soft_penalty(comp_SF,    Type(0.0),  Type(2.0),  k, w);

  nll += soft_penalty(qF,         Type(0.0),  Type(10.0), k, w);
  nll += soft_penalty(qS,         Type(0.0),  Type(10.0), k, w);
  nll += soft_penalty(HF,         Type(0.001),Type(1.0),  k, w);
  nll += soft_penalty(HS,         Type(0.001),Type(1.0),  k, w);
  nll += soft_penalty(phiF,       Type(0.0),  Type(2.0),  k, w);
  nll += soft_penalty(phiS,       Type(0.0),  Type(2.0),  k, w);
  nll += soft_penalty(gamma_pred, Type(0.0),  Type(10.0), k, w);

  nll += soft_penalty(bF_max,     Type(0.0),  Type(1.0),  k, w);
  nll += soft_penalty(bS_max,     Type(0.0),  Type(1.0),  k, w);
  nll += soft_penalty(T_bleach,   Type(26.0), Type(33.0), k, w);
  nll += soft_penalty(b_slope,    Type(0.1),  Type(10.0), k, w);

  // -----------------------------
  // REPORTING
  // -----------------------------
  REPORT(Year);               // time index
  REPORT(cots_dat_pred);      // predicted COTS density (individuals/m2)
  REPORT(fast_dat_pred);      // predicted fast coral cover (%)
  REPORT(slow_dat_pred);      // predicted slow coral cover (%)
  REPORT(sst_dat_pred);       // exogenous SST (deg C)
  REPORT(cotsimm_dat_pred);   // exogenous immigration (ind/m2/yr)

  REPORT(C);                  // internal states in native units (density for COTS)
  REPORT(F);                  // internal fast coral proportion (0–1)
  REPORT(S);                  // internal slow coral proportion (0–1)

  REPORT(sigma_cots_obs);     // effective observation SDs
  REPORT(sigma_fast_obs);
  REPORT(sigma_slow_obs);

  // Return negative log-likelihood
  return nll;
}
