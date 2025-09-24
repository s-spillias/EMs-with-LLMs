#include <TMB.hpp>

// Utility: softplus with adjustable steepness (beta); smooth approximation to max(0, x)
template<class Type>
Type softplus(Type x, Type beta) {
  // Numerically stable softplus that works with AD types without relying on std::log1p
  // softplus(z) = log(1 + exp(z)); for large z, use z + exp(-z) to avoid overflow
  Type z = beta * x;
  Type one = Type(1);
  Type res = CppAD::CondExpGt(z, Type(20), z + exp(-z), log(one + exp(z)));
  return res / beta;
}

// Utility: smooth positive part (beta controls sharpness)
template<class Type>
Type smooth_pos(Type x, Type beta) {
  return softplus(x, beta);
}

// Utility: smooth cap to [0, K] by applying smooth_pos twice
template<class Type>
Type smooth_cap0K(Type x, Type K, Type beta) {
  // First ensure positive, then ensure <= K using smooth operations
  Type xp = smooth_pos(x, beta);
  Type cap = K - smooth_pos(K - xp, beta);
  return cap;
}

// Utility: smooth minimum of two positive values
template<class Type>
Type smooth_min(Type a, Type b, Type tiny) {
  // 0.5*(a + b - sqrt((a - b)^2 + tiny))
  Type diff = a - b;
  return Type(0.5) * (a + b - sqrt(diff * diff + tiny));
}

// Utility: Gaussian thermal performance curve (0..1)
template<class Type>
Type thermal_perf(Type T, Type Topt, Type sigma) {
  Type eps = Type(1e-8);
  return exp(-((T - Topt) * (T - Topt)) / (Type(2) * sigma * sigma + eps));
}

// Utility: smooth sigmoid
template<class Type>
Type sigmoid(Type x, Type k, Type x0) {
  return Type(1) / (Type(1) + exp(-k * (x - x0)));
}

// Smooth penalty for keeping a parameter in [lo, hi] (no hard bounds)
template<class Type>
Type range_penalty(Type x, Type lo, Type hi, Type scale) {
  // Penalize distance outside [lo, hi] using softplus; scale controls strength
  Type below = softplus(lo - x, Type(5)); // >0 if x<lo
  Type above = softplus(x - hi, Type(5)); // >0 if x>hi
  return scale * (below + above);
}

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ---------------------------
  // DATA (all observations used in likelihood)
  // ---------------------------
  DATA_VECTOR(Year);          // Year (calendar year)
  DATA_VECTOR(sst_dat);       // Sea-surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);   // Larval immigration rate (ind m^-2 yr^-1)
  DATA_VECTOR(cots_dat);      // Adult COTS density (ind m^-2)
  DATA_VECTOR(fast_dat);      // Fast coral cover (%)
  DATA_VECTOR(slow_dat);      // Slow coral cover (%)

  int n = Year.size();        // number of time steps

  // ---------------------------
  // PARAMETERS (comments include units and guidance)
  // ---------------------------

  // Coral colonization rates (year^-1)
  PARAMETER(rF);           // rF: Intrinsic colonization/growth rate of fast coral (year^-1); literature/expert ranges 0.2–1.5
  PARAMETER(rS);           // rS: Intrinsic colonization/growth rate of slow coral (year^-1); literature/expert ranges 0.05–0.8

  // Coral background mortality (year^-1) and thermal stress multipliers (year^-1)
  PARAMETER(mF);           // mF: Baseline mortality of fast coral (year^-1); higher during stress; 0.05–0.5 typical
  PARAMETER(mS);           // mS: Baseline mortality of slow coral (year^-1); 0.02–0.3 typical
  PARAMETER(mF_bleach);    // mF_bleach: Additional mortality at high thermal stress for fast coral (year^-1); 0–1
  PARAMETER(mS_bleach);    // mS_bleach: Additional mortality at high thermal stress for slow coral (year^-1); 0–1

  // Coral carrying capacity (proportion of substrate)
  PARAMETER(K_total);      // K_total: Total coral cover capacity (0–1); fraction of benthos that can be coral (dimensionless)

  // Coral thermal performance parameters
  PARAMETER(ToptF);        // ToptF: Thermal optimum for fast coral growth (°C); typically ~26–29 °C
  PARAMETER(sigmaTF);      // sigmaTF: Thermal breadth for fast coral (°C); typically 1–4 °C
  PARAMETER(ToptS);        // ToptS: Thermal optimum for slow coral (°C); ~26–29 °C
  PARAMETER(sigmaTS);      // sigmaTS: Thermal breadth for slow coral (°C); 1–4 °C

  // COTS functional response and diet preference
  PARAMETER(prefF);        // prefF: Preference for fast coral in diet (0–1); larger => stronger Acropora selection
  PARAMETER(aC);           // aC: Encounter/attack rate for COTS multi-prey feeding (m^2 ind^-1 yr^-1, scaled); controls speed of predation
  PARAMETER(hC);           // hC: Handling time parameter (yr); increases saturation at high coral cover
  PARAMETER(cInterf);      // cInterf: Intraspecific interference coefficient in feeding (m^2 ind^-1); reduces per-capita intake at high C

  // Conversion efficiencies from consumption to coral cover loss (dimensionless, 0–1)
  PARAMETER(eF);           // eF: Efficiency for fast coral loss per unit COTS feeding (0–1)
  PARAMETER(eS);           // eS: Efficiency for slow coral loss per unit COTS feeding (0–1)

  // COTS population dynamics
  PARAMETER(rC0);          // rC0: Maximum per-capita reproductive rate of COTS (year^-1)
  PARAMETER(dC);           // dC: Adult COTS mortality rate (year^-1)
  PARAMETER(KC);           // KC: COTS carrying capacity (ind m^-2)
  PARAMETER(A);            // A: Allee density scale for reproduction (ind m^-2), sets density where mate limitation eases

  // COTS thermal performance
  PARAMETER(ToptC);        // ToptC: Thermal optimum for COTS performance (°C)
  PARAMETER(sigmaTC);      // sigmaTC: Thermal breadth for COTS (°C)

  // Food limitation of COTS reproduction
  PARAMETER(k_food);       // k_food: Half-saturation of food index for COTS reproduction (proportion), 0–1

  // Immigration trigger and gain
  PARAMETER(gammaImm);     // gammaImm: Gain on larval immigration (ind m^-2 yr^-1 per unit of cotsimm_dat)
  PARAMETER(tauImm);       // tauImm: Smooth threshold (individuals m^-2 yr^-1) for immigration effectiveness
  PARAMETER(kImm);         // kImm: Steepness of immigration sigmoid (unit^-1)
  PARAMETER(i0);           // i0: Baseline immigration (ind m^-2 yr^-1) independent of cotsimm_dat

  // Observation model parameters (raw, transformed inside to ensure positivity)
  PARAMETER(log_sd_cots_raw);  // log_sd_cots_raw: Raw parameter mapped to lognormal SD for COTS observations
  PARAMETER(phi_fast_raw);     // phi_fast_raw: Raw parameter mapped to Beta precision for fast coral observations
  PARAMETER(phi_slow_raw);     // phi_slow_raw: Raw parameter mapped to Beta precision for slow coral observations

  // Initial states at first time step (Year[0])
  PARAMETER(F0);           // F0: Initial fast coral cover proportion (0–1)
  PARAMETER(S0);           // S0: Initial slow coral cover proportion (0–1)
  PARAMETER(C0);           // C0: Initial adult COTS density (ind m^-2)

  // Optional upper cap for COTS for stability (soft cap)
  PARAMETER(Cmax);         // Cmax: Soft upper cap for COTS density (ind m^-2), typically > KC

  // ---------------------------
  // Settings and small constants
  // ---------------------------
  Type eps = Type(1e-8);           // Numerical stability small constant
  Type beta_sp = Type(20.0);       // Smoothness for softplus/capping (higher = sharper)
  Type tiny = Type(1e-12);         // Tiny constant for smooth_min
  Type min_sd = Type(0.05);        // Minimum SD for lognormal obs to avoid zero-variance
  Type min_phi = Type(2.0);        // Minimum Beta precision (variance finite)

  // Transformed observation parameters
  Type sigma_log_cots = min_sd + softplus(log_sd_cots_raw, Type(1.0));             // Ensure > min_sd
  Type phi_fast = min_phi + softplus(phi_fast_raw, Type(1.0));                      // Ensure > min_phi
  Type phi_slow = min_phi + softplus(phi_slow_raw, Type(1.0));                      // Ensure > min_phi

  // ---------------------------
  // Containers for predictions
  // ---------------------------
  vector<Type> cots_pred(n);   // COTS density prediction (ind m^-2)
  vector<Type> fast_pred(n);   // Fast coral cover prediction (%)
  vector<Type> slow_pred(n);   // Slow coral cover prediction (%)

  // ---------------------------
  // Negative log-likelihood accumulator
  // ---------------------------
  Type nll = Type(0.0);

  // ---------------------------
  // Parameter range penalties (smooth, weakly-informative)
  // ---------------------------
  nll += range_penalty(rF, Type(0.0), Type(2.0), Type(0.01));
  nll += range_penalty(rS, Type(0.0), Type(1.0), Type(0.01));
  nll += range_penalty(mF, Type(0.0), Type(1.0), Type(0.01));
  nll += range_penalty(mS, Type(0.0), Type(1.0), Type(0.01));
  nll += range_penalty(mF_bleach, Type(0.0), Type(1.0), Type(0.01));
  nll += range_penalty(mS_bleach, Type(0.0), Type(1.0), Type(0.01));
  nll += range_penalty(K_total, Type(0.3), Type(0.98), Type(0.02));
  nll += range_penalty(prefF, Type(0.0), Type(1.0), Type(0.02));
  nll += range_penalty(aC, Type(0.0), Type(10.0), Type(0.005));
  nll += range_penalty(hC, Type(0.0), Type(5.0), Type(0.005));
  nll += range_penalty(cInterf, Type(0.0), Type(2.0), Type(0.005));
  nll += range_penalty(eF, Type(0.0), Type(1.0), Type(0.02));
  nll += range_penalty(eS, Type(0.0), Type(1.0), Type(0.02));
  nll += range_penalty(rC0, Type(0.0), Type(10.0), Type(0.005));
  nll += range_penalty(dC, Type(0.0), Type(2.0), Type(0.005));
  nll += range_penalty(KC, Type(0.1), Type(20.0), Type(0.002));
  nll += range_penalty(Cmax, KC, Type(50.0), Type(0.002)); // Cmax >= KC
  nll += range_penalty(A, Type(0.0), Type(5.0), Type(0.005));
  nll += range_penalty(ToptF, Type(23.0), Type(31.0), Type(0.001));
  nll += range_penalty(ToptS, Type(23.0), Type(31.0), Type(0.001));
  nll += range_penalty(ToptC, Type(23.0), Type(31.0), Type(0.001));
  nll += range_penalty(sigmaTF, Type(0.5), Type(5.0), Type(0.001));
  nll += range_penalty(sigmaTS, Type(0.5), Type(5.0), Type(0.001));
  nll += range_penalty(sigmaTC, Type(0.5), Type(5.0), Type(0.001));
  nll += range_penalty(k_food, Type(0.001), Type(1.0), Type(0.01));
  nll += range_penalty(gammaImm, Type(0.0), Type(50.0), Type(0.001));
  nll += range_penalty(tauImm, Type(0.0), Type(5.0), Type(0.001));
  nll += range_penalty(kImm, Type(0.1), Type(20.0), Type(0.001));
  nll += range_penalty(i0, Type(0.0), Type(2.0), Type(0.001));
  nll += range_penalty(F0, Type(0.0), Type(0.9), Type(0.02));
  nll += range_penalty(S0, Type(0.0), Type(0.9), Type(0.02));
  nll += range_penalty(C0, Type(0.0), Type(20.0), Type(0.002));

  // ---------------------------
  // INITIAL CONDITIONS
  // ---------------------------
  Type F = smooth_cap0K(F0, K_total - eps, beta_sp);    // Smoothly bound F in [0, K_total)
  Type S = smooth_cap0K(S0, K_total - eps, beta_sp);    // Smoothly bound S in [0, K_total)
  Type C = smooth_cap0K(C0, Cmax, beta_sp);             // Smoothly bound C in [0, Cmax]

  // ---------------------------
  // MODEL EQUATIONS (discrete-time, annual)
  // Numbered description:
  // (1) Free space: U_t = max(0, K_total − (F_t + S_t)) [smooth]
  // (2) Thermal performance (coral i ∈ {F,S}): g_i(T_t) = exp(−(T − Topt_i)^2 / (2 sigmaT_i^2))
  // (3) Coral colonization: G_i = r_i * g_i(T) * F_i * (U_t / (K_total + eps))
  // (4) Thermal stress mortality: M_i = (m_i + m_i_bleach * (1 − g_i(T))) * F_i
  // (5) Food index for COTS: Q_t = prefF * F_t + (1 − prefF) * S_t
  // (6) COTS per-capita feeding: H_t = aC * Q_t / (1 + aC * hC * Q_t + cInterf * C_t)
  // (7) Coral loss to COTS: L_F = eF * C_t * H_t * prefF ; L_S = eS * C_t * H_t * (1 − prefF)
  // (8) Coral updates: F_{t+1} = cap0K(F_t + G_F − M_F − L_F), similarly for S with smooth capping and space sharing
  // (9) COTS Allee effect: A_t = C_t / (A + C_t)
  // (10) Food limitation of reproduction: Rf_t = Q_t / (k_food + Q_t)
  // (11) Immigration trigger: I_t = (i0 + gammaImm * imm_t * sigmoid_k(imm_t, tauImm)) * thermal_perf_COTS(T)
  // (12) COTS update: C_{t+1} = cap0K(C_t + rC0 * A_t * Rf_t * thermal_perf_COTS(T) * C_t * (1 − C_t / KC) − dC * C_t + I_t)
  // ---------------------------

  for (int t = 0; t < n; t++) {

    // Environmental drivers at t
    Type Tt = sst_dat(t);                           // SST at year t (°C)
    Type imm_t = cotsimm_dat(t);                    // Larval immigration rate at year t (ind m^-2 yr^-1)

    // (1) Free space (smooth, >=0)
    Type U_raw = K_total - (F + S);
    Type U = smooth_pos(U_raw, beta_sp);            // Free space proportion (dimensionless)

    // (2) Thermal performance for corals
    Type gF = thermal_perf(Tt, ToptF, sigmaTF);     // 0..1 modifier for fast corals
    Type gS = thermal_perf(Tt, ToptS, sigmaTS);     // 0..1 modifier for slow corals

    // (3) Coral colonization into free space
    Type G_F = rF * gF * F * (U / (K_total + eps)); // Fast coral colonization (proportion yr^-1)
    Type G_S = rS * gS * S * (U / (K_total + eps)); // Slow coral colonization (proportion yr^-1)

    // (4) Thermal stress mortality (background + extra under stress)
    Type M_F = (mF + mF_bleach * (Type(1.0) - gF)) * F;  // Fast coral mortality (proportion yr^-1)
    Type M_S = (mS + mS_bleach * (Type(1.0) - gS)) * S;  // Slow coral mortality (proportion yr^-1)

    // (5) Food index for COTS (weighted coral availability)
    Type Q = prefF * F + (Type(1.0) - prefF) * S;        // Dimensionless, 0..K_total

    // (6) COTS per-capita feeding with saturation and interference
    Type H = aC * Q / (Type(1.0) + aC * hC * Q + cInterf * C + eps); // yr^-1

    // (7) Coral loss due to COTS (partitioned by preference)
    Type L_F = eF * C * H * prefF;                       // Fast coral loss (proportion yr^-1)
    Type L_S = eS * C * H * (Type(1.0) - prefF);         // Slow coral loss (proportion yr^-1)

    // Raw coral updates
    Type F_raw_next = F + G_F - M_F - L_F;               // Unbounded update for F
    Type S_raw_next = S + G_S - M_S - L_S;               // Unbounded update for S

    // (8) Smoothly bound each coral to [0, K_total], then smoothly enforce total coral <= K_total
    Type F_next_cap = smooth_cap0K(F_raw_next, K_total - eps, beta_sp);
    Type S_next_cap = smooth_cap0K(S_raw_next, K_total - eps, beta_sp);
    // Smooth proportional sharing if sum exceeds K_total
    Type sumFS = F_next_cap + S_next_cap + eps;
    Type sumFS_cap = smooth_min(sumFS, K_total, tiny);    // Cap total to K_total smoothly
    Type pF_share = F_next_cap / sumFS;                   // Proportional share (dimensionless)
    Type F_next = pF_share * sumFS_cap;                   // Final F_{t+1}
    Type S_next = (Type(1.0) - pF_share) * sumFS_cap;     // Final S_{t+1}

    // (9) COTS Allee effect
    Type Aeff = C / (A + C + eps);                       // 0..1

    // (10) Food limitation of COTS reproduction
    Type Rf = Q / (k_food + Q + eps);                    // 0..1

    // COTS thermal performance
    Type gC = thermal_perf(Tt, ToptC, sigmaTC);          // 0..1

    // (11) Immigration trigger (smooth threshold) and temperature modulation
    Type imm_gate = sigmoid(imm_t, kImm, tauImm);        // 0..1
    Type I_t = (i0 + gammaImm * imm_t * imm_gate) * gC;  // ind m^-2 yr^-1

    // (12) COTS update with logistic density regulation and soft upper cap
    Type C_growth = rC0 * Aeff * Rf * gC * C * (Type(1.0) - C / (KC + eps)); // ind m^-2 yr^-1
    Type C_loss = dC * C;                                 // ind m^-2 yr^-1
    Type C_raw_next = C + C_growth - C_loss + I_t;        // Unbounded update
    Type C_next = smooth_cap0K(C_raw_next, Cmax, beta_sp);// Softly bound to [0, Cmax]

    // Save predictions (reporting scale)
    cots_pred(t) = C_next;                                // COTS density prediction (ind m^-2)
    fast_pred(t) = (F_next * Type(100.0));                // Fast coral cover prediction (%)
    slow_pred(t) = (S_next * Type(100.0));                // Slow coral cover prediction (%)

    // Observation model:
    // COTS: lognormal on positive data
    Type yC = cots_dat(t) + eps;
    Type muC = cots_pred(t) + eps;
    nll -= dnorm(log(yC), log(muC), sigma_log_cots, true);

    // Corals: Beta on proportions in (0,1)
    Type yF_prop = CppAD::CondExpLt(fast_dat(t)/Type(100.0), eps, eps, CppAD::CondExpGt(fast_dat(t)/Type(100.0), Type(1.0)-eps, Type(1.0)-eps, fast_dat(t)/Type(100.0)));
    Type yS_prop = CppAD::CondExpLt(slow_dat(t)/Type(100.0), eps, eps, CppAD::CondExpGt(slow_dat(t)/Type(100.0), Type(1.0)-eps, Type(1.0)-eps, slow_dat(t)/Type(100.0)));
    Type muF_prop = CppAD::CondExpLt(fast_pred(t)/Type(100.0), eps, eps, CppAD::CondExpGt(fast_pred(t)/Type(100.0), Type(1.0)-eps, Type(1.0)-eps, fast_pred(t)/Type(100.0)));
    Type muS_prop = CppAD::CondExpLt(slow_pred(t)/Type(100.0), eps, eps, CppAD::CondExpGt(slow_pred(t)/Type(100.0), Type(1.0)-eps, Type(1.0)-eps, slow_pred(t)/Type(100.0)));

    Type alphaF = muF_prop * phi_fast + eps;
    Type betaF  = (Type(1.0) - muF_prop) * phi_fast + eps;
    Type alphaS = muS_prop * phi_slow + eps;
    Type betaS  = (Type(1.0) - muS_prop) * phi_slow + eps;

    nll -= dbeta(yF_prop, alphaF, betaF, true);
    nll -= dbeta(yS_prop, alphaS, betaS, true);

    // Advance states for next step using model-predicted values only (no data leakage)
    F = F_next;
    S = S_next;
    C = C_next;
  }

  // ---------------------------
  // REPORTING
  // ---------------------------
  REPORT(cots_pred);       // COTS density predictions (ind m^-2)
  REPORT(fast_pred);       // Fast coral cover predictions (%)
  REPORT(slow_pred);       // Slow coral cover predictions (%)

  // Also report transformed observation parameters and some derived effects for diagnostics
  REPORT(sigma_log_cots);
  REPORT(phi_fast);
  REPORT(phi_slow);

  // ADREPORT for uncertainty on key derived parameters
  ADREPORT(sigma_log_cots);
  ADREPORT(phi_fast);
  ADREPORT(phi_slow);

  return nll;
}
