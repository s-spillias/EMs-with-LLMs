#include <TMB.hpp>

// Helper: numerically stable inverse-logit (sigmoid)
template<class Type>
Type invlogit_stable(Type x) {
  // Avoid overflow in exp by limiting input range
  x = CppAD::CondExpLt(x, Type(-35), Type(-35), x);
  x = CppAD::CondExpGt(x, Type(35),  Type(35),  x);
  return Type(1) / (Type(1) + exp(-x));
}

// AD-safe maximum using conditional expression (avoids std::fmax with AD types)
template<class Type>
Type tmb_max(Type a, Type b) {
  return CppAD::CondExpGt(a, b, a, b);
}

// AD-safe absolute value
template<class Type>
Type tmb_abs(Type x) {
  return CppAD::CondExpGe(x, Type(0), x, -x);
}

// Helper: softplus for smooth non-negativity (AD-safe and numerically stable)
// softplus(x) = max(0,x) + log(1 + exp(-|x|))
template<class Type>
Type softplus(Type x) {
  Type mx = tmb_max(x, Type(0));
  Type ax = tmb_abs(x);
  return mx + log(Type(1) + exp(-ax));
}

// Smooth ReLU used for soft bound penalties
template<class Type>
Type smooth_relu(Type x) {
  // Approximates max(0, x) smoothly using softplus
  return softplus(x);
}

// Small helper for squaring (avoids pow with AD types)
template<class Type>
Type square(Type x) { return x * x; }

template<class Type>
Type objective_function<Type>::operator() () {
  using namespace density;

  // ---------------------------------------------------------------------------
  // DATA: time and observations
  // ---------------------------------------------------------------------------
  DATA_VECTOR(Year);           // Year (calendar year), used for alignment/reporting
  DATA_VECTOR(cots_dat);       // Adult COTS density (individuals/m^2)
  DATA_VECTOR(fast_dat);       // Fast-growing coral cover, Acropora (%) of benthos
  DATA_VECTOR(slow_dat);       // Slow-growing coral cover, Faviidae/Porites (%)
  DATA_VECTOR(sst_dat);        // Sea-surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat);    // Larval immigration rate (individuals/m^2/year)

  int n = cots_dat.size();     // Number of time steps (years)

  // ---------------------------------------------------------------------------
  // PARAMETERS: ecological rates and sensitivities
  // Each parameter line includes units and guidance for initialization.
  // ---------------------------------------------------------------------------
  PARAMETER(r_max);            // year^-1; Max per-capita recruitment rate from local reproduction; start from 1-3 (initial estimate)
  PARAMETER(alpha_C);          // (m^2/ind); Density-dependence in recruitment (saturating fecundity); start ~0.3-0.7 (initial estimate)
  PARAMETER(m0);               // year^-1; Baseline instantaneous adult mortality of COTS; start ~0.3-0.7 (literature)
  PARAMETER(m_food);           // year^-1; Extra mortality when food is scarce; start ~0.2-0.8 (initial estimate)
  PARAMETER(R_half);           // % cover; Half-saturation of resource effect on reproduction/survival; start 5-20% (initial estimate)
  PARAMETER(q_A);              // dimensionless; Food value weight of Acropora per % cover; start 0.8-1.0 (literature)
  PARAMETER(q_S);              // dimensionless; Food value weight of slow corals per % cover; start 0.3-0.7 (literature)

  PARAMETER(T_thr);            // deg C; Temperature threshold for enhanced larval survival (outbreak trigger); start 27-29 (literature)
  PARAMETER(k_T);              // 1/deg C; Steepness of temperature effect on larval survival; start 0.5-2 (initial estimate)
  PARAMETER(eta_I);            // dimensionless; Scaling on exogenous larval immigration; start ~1 (initial estimate)

  PARAMETER(g_A);              // year^-1; Intrinsic growth rate of Acropora; start 0.4-0.8 (literature)
  PARAMETER(g_S);              // year^-1; Intrinsic growth rate of slow corals; start 0.1-0.4 (literature)
  PARAMETER(K_space);          // % cover; Effective total coral carrying capacity (shared space); start 70-90% (literature)
  PARAMETER(k_space);          // 1/%; Steepness of free-space limitation; start 0.1-0.5 (initial estimate)

  PARAMETER(Topt_A);           // deg C; Temperature optimum for Acropora growth; start ~27.5 (literature)
  PARAMETER(Topt_S);           // deg C; Temperature optimum for slow coral growth; start ~27.0 (literature)
  PARAMETER(sigma_T_coral);    // deg C; Thermal breadth of coral growth Gaussian; start 0.8-2 (literature)

  PARAMETER(a_A);              // % cover per (ind/m^2)/year; Max per-starfish kill rate on Acropora (Holling II numerator); start 0.5-2 (literature)
  PARAMETER(a_S);              // % cover per (ind/m^2)/year; Max per-starfish kill rate on slow corals; start 0.2-1 (literature)
  PARAMETER(h_A);              // % cover; Half-saturation coral cover for attack on Acropora (Holling II denominator); start 5-20% (literature)
  PARAMETER(h_S);              // % cover; Half-saturation for slow corals; start 10-30% (literature)

  PARAMETER(Topt_feed);        // deg C; Temperature optimum for COTS feeding activity; start ~27.5 (literature)
  PARAMETER(sigma_T_feed);     // deg C; Thermal breadth for feeding activity; start 1-3 (literature)

  PARAMETER(m_dd_max);         // year^-1; Extra density-dependent mortality at high COTS density (bust mechanism); start 0.5-2 (initial estimate)
  PARAMETER(C_dd_mid);         // ind/m^2; Midpoint COTS density for added mortality; start 0.5-2 (initial estimate)
  PARAMETER(k_dd);             // (m^2/ind); Steepness of density-dependent mortality; start 1-4 (initial estimate)

  // Observation error (lognormal SDs on log scale)
  PARAMETER(sd_cots);          // dimensionless; Log-scale SD for COTS observations; start 0.1-0.5 (initial estimate)
  PARAMETER(sd_fast);          // dimensionless; Log-scale SD for Acropora observations; start 0.05-0.3 (initial estimate)
  PARAMETER(sd_slow);          // dimensionless; Log-scale SD for slow coral observations; start 0.05-0.3 (initial estimate)

  // ---------------------------------------------------------------------------
  // Small constants and setup
  // ---------------------------------------------------------------------------
  Type eps = Type(1e-8);       // Small constant for numerical stability
  Type nll = Type(0.0);        // Negative log-likelihood accumulator

  // Minimum standard deviations for stability (AD-safe max)
  Type sdmin = Type(0.05);
  Type sd_cots_eff = tmb_max(sd_cots, sdmin);
  Type sd_fast_eff = tmb_max(sd_fast, sdmin);
  Type sd_slow_eff = tmb_max(sd_slow, sdmin);

  // ---------------------------------------------------------------------------
  // Soft bound penalties (encourage biologically plausible ranges; smooth)
  // penalty = w * [relu(lo - p)^2 + relu(p - hi)^2], with smooth ReLU
  // ---------------------------------------------------------------------------
  Type wpen = Type(1e-3); // Small weight to avoid overpowering likelihood

  auto pen_range = [&](Type p, Type lo, Type hi) -> Type {
    Type lo_excess = smooth_relu(lo - p);
    Type hi_excess = smooth_relu(p - hi);
    return wpen * (lo_excess * lo_excess + hi_excess * hi_excess);
  };

  // Apply penalties (chosen plausible ranges)
  nll += pen_range(r_max,       Type(0.0),  Type(10.0));
  nll += pen_range(alpha_C,     Type(0.0),  Type(10.0));
  nll += pen_range(m0,          Type(0.0),  Type(3.0));
  nll += pen_range(m_food,      Type(0.0),  Type(3.0));
  nll += pen_range(R_half,      Type(0.1),  Type(100.0));
  nll += pen_range(q_A,         Type(0.0),  Type(2.0));
  nll += pen_range(q_S,         Type(0.0),  Type(2.0));
  nll += pen_range(T_thr,       Type(20.0), Type(35.0));
  nll += pen_range(k_T,         Type(0.0),  Type(5.0));
  nll += pen_range(eta_I,       Type(0.0),  Type(5.0));
  nll += pen_range(g_A,         Type(0.0),  Type(2.0));
  nll += pen_range(g_S,         Type(0.0),  Type(2.0));
  nll += pen_range(K_space,     Type(30.0), Type(99.0));
  nll += pen_range(k_space,     Type(0.0),  Type(2.0));
  nll += pen_range(Topt_A,      Type(20.0), Type(35.0));
  nll += pen_range(Topt_S,      Type(20.0), Type(35.0));
  nll += pen_range(sigma_T_coral,Type(0.3), Type(5.0));
  nll += pen_range(a_A,         Type(0.0),  Type(5.0));
  nll += pen_range(a_S,         Type(0.0),  Type(5.0));
  nll += pen_range(h_A,         Type(0.1),  Type(60.0));
  nll += pen_range(h_S,         Type(0.1),  Type(60.0));
  nll += pen_range(Topt_feed,   Type(20.0), Type(35.0));
  nll += pen_range(sigma_T_feed,Type(0.3),  Type(6.0));
  nll += pen_range(m_dd_max,    Type(0.0),  Type(5.0));
  nll += pen_range(C_dd_mid,    Type(0.0),  Type(5.0));
  nll += pen_range(k_dd,        Type(0.0),  Type(10.0));
  nll += pen_range(sd_cots,     Type(0.01), Type(2.0));
  nll += pen_range(sd_fast,     Type(0.01), Type(2.0));
  nll += pen_range(sd_slow,     Type(0.01), Type(2.0));

  // ---------------------------------------------------------------------------
  // STATE VECTORS: predictions (initialized from first observation)
  // ---------------------------------------------------------------------------
  vector<Type> cots_pred(n);   // Predicted COTS density (ind/m^2)
  vector<Type> fast_pred(n);   // Predicted Acropora cover (%)
  vector<Type> slow_pred(n);   // Predicted slow coral cover (%)

  // Auxiliary predictions for diagnostics
  vector<Type> R_pred(n);          // Resource index
  vector<Type> C_recruit_pred(n);  // COTS recruitment contribution
  vector<Type> C_survive_pred(n);  // COTS survivors after mortality
  vector<Type> lossA_pred(n);      // Predation loss on Acropora
  vector<Type> lossS_pred(n);      // Predation loss on slow corals
  vector<Type> free_space_pred(n); // Free-space factor (0-1)

  // Initialize with observed initial conditions (no data leakage in transitions)
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Initialize diagnostics at t=0 for completeness (based on initial state, not used to predict t=0)
  {
    Type A0 = fast_pred(0);
    Type S0 = slow_pred(0);
    Type C0 = cots_pred(0);
    Type T0 = sst_dat(0);

    Type R0 = q_A * A0 + q_S * S0;                             // (1) Resource index
    R_pred(0) = R0;
    Type sR0 = R0 / (R_half + R0 + eps);                       // (2) Resource saturation (0-1)

    Type fTrec0 = invlogit_stable(k_T * (T0 - T_thr));         // (3) Temp effect on recruitment (0-1)
    Type Z0 = m0 + m_food * (Type(1.0) - sR0) +                // (4) Baseline + food-limited mortality
              m_dd_max * invlogit_stable(k_dd * (C0 - C_dd_mid)); // (5) Density-dependent extra mortality
    C_survive_pred(0) = C0 * exp(-Z0);                          // (6) Survivors
    C_recruit_pred(0) = r_max * C0 * sR0 * fTrec0 / (Type(1.0) + alpha_C * C0) // (7) Local recruitment
                        + eta_I * cotsimm_dat(0);               // (8) Immigration

    Type dfeed0 = (T0 - Topt_feed) / (sigma_T_feed + eps);
    Type fTfeed0 = exp(-Type(0.5) * square(dfeed0));           // (9) Temp effect on feeding
    lossA_pred(0) = a_A * C0 * (A0 / (h_A + A0 + eps)) * fTfeed0; // (10) Predation on Acropora
    lossS_pred(0) = a_S * C0 * (S0 / (h_S + S0 + eps)) * fTfeed0; // (11) Predation on slow corals

    Type free0 = Type(1.0) / (Type(1.0) + exp(k_space * ((A0 + S0) - K_space))); // (12) Free-space factor
    free_space_pred(0) = free0;
  }

  // ---------------------------------------------------------------------------
  // DYNAMIC EQUATIONS (t >= 1) - use only previous-step states to avoid leakage
  // ---------------------------------------------------------------------------
  for (int t = 1; t < n; t++) {
    // Previous states
    Type C = cots_pred(t - 1);   // COTS density at t-1
    Type A = fast_pred(t - 1);   // Acropora at t-1
    Type S = slow_pred(t - 1);   // Slow corals at t-1
    Type T = sst_dat(t - 1);     // Use prior year's SST as driver for transitions
    Type I = cotsimm_dat(t - 1); // Use prior year's immigration as driver

    // (1) Resource index: weighted edible coral availability (units: % cover)
    Type R = q_A * A + q_S * S;
    R_pred(t) = R;

    // (2) Resource saturation (0-1), smooth Michaelis-Menten form
    Type sR = R / (R_half + R + eps);

    // (3) Temperature effect on larval survival/recruitment (0-1), smooth threshold
    Type fTrec = invlogit_stable(k_T * (T - T_thr));

    // (4) Instantaneous mortality components and survivors
    Type Z = m0 + m_food * (Type(1.0) - sR) +
             m_dd_max * invlogit_stable(k_dd * (C - C_dd_mid)); // extra mortality when C exceeds C_dd_mid
    Type C_survive = C * exp(-Z);
    C_survive_pred(t) = C_survive;

    // (5) Local reproduction with resource and temperature limitation and density compensation
    Type C_recruit = r_max * C * sR * fTrec / (Type(1.0) + alpha_C * C);

    // (6) Additive exogenous larval immigration scaled by eta_I
    C_recruit += eta_I * I;
    C_recruit_pred(t) = C_recruit;

    // (7) Next-step COTS density (ensure non-negativity smoothly)
    Type C_next = softplus(C_survive + C_recruit + eps);
    cots_pred(t) = C_next;

    // (8) Temperature effects on feeding/activity (0-1) using Gaussian thermal performance
    Type dfeed = (T - Topt_feed) / (sigma_T_feed + eps);
    Type fTfeed = exp(-Type(0.5) * square(dfeed));

    // (9) Holling II predation losses on corals (units: % cover/year)
    Type lossA = a_A * C * (A / (h_A + A + eps)) * fTfeed;
    Type lossS = a_S * C * (S / (h_S + S + eps)) * fTfeed;
    lossA_pred(t) = lossA;
    lossS_pred(t) = lossS;

    // (10) Free-space limitation shared by corals (0-1), smooth sigmoidal
    Type free_space = Type(1.0) / (Type(1.0) + exp(k_space * ((A + S) - K_space)));
    free_space_pred(t) = free_space;

    // (11) Temperature effects on coral growth (0-1) via Gaussian performance around T_opt
    Type dTA = (T - Topt_A) / (sigma_T_coral + eps);
    Type dTS = (T - Topt_S) / (sigma_T_coral + eps);
    Type gTA = exp(-Type(0.5) * square(dTA));
    Type gTS = exp(-Type(0.5) * square(dTS));

    // (12) Coral growth increments (logistic-like with shared space and temperature)
    Type dA_grow = g_A * A * free_space * gTA;
    Type dS_grow = g_S * S * free_space * gTS;

    // (13) Next-step coral covers with smooth non-negativity
    Type A_next = softplus(A + dA_grow - lossA);
    Type S_next = softplus(S + dS_grow - lossS);

    fast_pred(t) = A_next;
    slow_pred(t) = S_next;
  }

  // ---------------------------------------------------------------------------
  // LIKELIHOOD: lognormal for strictly positive observations
  // Include all observations; use eps to avoid log(0).
  // ---------------------------------------------------------------------------
  for (int t = 0; t < n; t++) {
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sd_cots_eff, true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sd_fast_eff, true);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sd_slow_eff, true);
  }

  // ---------------------------------------------------------------------------
  // EQUATION SUMMARY (for reference):
  // 1) R_t = q_A * A_{t-1} + q_S * S_{t-1}
  // 2) sR_t = R_t / (R_half + R_t)
  // 3) fTrec_t = invlogit(k_T * (SST_{t-1} - T_thr))
  // 4) Z_t = m0 + m_food * (1 - sR_t) + m_dd_max * invlogit(k_dd * (C_{t-1} - C_dd_mid))
  // 5) C_survive_t = C_{t-1} * exp(-Z_t)
  // 6) C_recruit_t = r_max * C_{t-1} * sR_t * fTrec_t / (1 + alpha_C * C_{t-1}) + eta_I * I_{t-1}
  // 7) C_t = softplus(C_survive_t + C_recruit_t)
  // 8) fTfeed_t = exp(-0.5) * (((SST_{t-1} - Topt_feed)/sigma_T_feed)^2)
  // 9) lossA_t = a_A * C_{t-1} * (A_{t-1} / (h_A + A_{t-1})) * fTfeed_t; lossS_t analogous
  // 10) free_space_t = 1 / (1 + exp(k_space * ((A_{t-1}+S_{t-1}) - K_space)))
  // 11) gT_{A,S,t} = exp(-0.5) * (((SST_{t-1} - Topt_{A,S})/sigma_T_coral)^2)
  // 12) dA_grow_t = g_A * A_{t-1} * free_space_t * gT_{A,t}; dS_grow_t analogous
  // 13) A_t = softplus(A_{t-1} + dA_grow_t - lossA_t); S_t analogous
  // ---------------------------------------------------------------------------

  // REPORT all predictions and key diagnostics
  REPORT(Year);
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(R_pred);
  REPORT(C_recruit_pred);
  REPORT(C_survive_pred);
  REPORT(lossA_pred);
  REPORT(lossS_pred);
  REPORT(free_space_pred);

  return nll;
}
