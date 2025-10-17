#include <TMB.hpp>

// Helper: stable inverse-logit to keep values in (0,1)
template<class Type>
Type invlogit(Type x) { return Type(1) / (Type(1) + exp(-x)); }

// Helper: numerically stable softplus; behaves like max(0, x) but smooth
template<class Type>
Type softplus(Type x) {
  Type thresh = Type(20);
  return CppAD::CondExpGt(x, thresh, x, log1p(exp(x)));
}

// Helper: clamp to (eps, 1-eps) smoothly by two softplus hinges
template<class Type>
Type clamp01(Type x, Type eps) {
  Type lo = eps;
  Type hi = Type(1) - eps;
  // Push up if below eps
  Type below = softplus(lo - x);
  // Push down if above 1-eps
  Type above = softplus(x - hi);
  return x + below - above;
}

// Helper: smooth bound penalty (avoid lambda to ease linters/parsers)
template<class Type>
Type bound_penalty(Type x, Type lo, Type hi, Type w) {
  return w * pow(softplus(lo - x), 2) + w * pow(softplus(x - hi), 2);
}

/*
Equations (discrete annual time, using t-1 -> t; never using current observations in prediction):
1) Environmental anomaly:
   sst_anom(t) = sst_dat(t) - mean(sst_dat)

2) COTS environmental multiplier (bounded, smooth):
   f_sst(t-1) = f_sst_lo + (f_sst_hi - f_sst_lo) * invlogit(beta_sst_cots * sst_anom(t-1))

3) Prey availability for COTS reproduction (saturating Michaelis-Menten):
   prey_avail(t-1) = pref_fast * F(t-1) + pref_slow * S(t-1)
   f_prey(t-1) = prey_avail(t-1) / (K_prey + prey_avail(t-1) + eps)

4) Allee effect (suppresses per-capita growth when A is very low):
   f_allee(t-1) = A(t-1) / (A(t-1) + A_crit + eps)

5) COTS per-capita net rate with self-limitation (logistic-like, smooth):
   r_eff(t-1) = r_cots_max * f_prey(t-1) * f_sst(t-1) * f_allee(t-1) - m_cots - c_cots_density * A(t-1)

6) Adult COTS update with multiplicative growth and additive immigration:
   A(t) = A(t-1) * exp(r_eff(t-1)) + e_cots_imm * cotsimm_dat(t-1)

7) Multi-prey Holling II/III consumption by COTS (selectivity and saturation):
   V = pref_fast * F(t-1)^q + pref_slow * S(t-1)^q + eps
   Cons_per_pred = attack * V / (1 + attack * handling * V + eps)
   Share_F = (pref_fast * F(t-1)^q) / V ; Share_S analogous
   Pred_F = A(t-1) * Cons_per_pred * Share_F
   Pred_S = A(t-1) * Cons_per_pred * Share_S

8) Coral free-space limitation (smooth; K_tot in proportion of bottom):
   free(t-1) = softplus(K_tot - F(t-1) - S(t-1))
   Growth_F = r_fast * F(t-1) * (free(t-1) / (K_tot + eps))
   Growth_S = r_slow * S(t-1) * (free(t-1) / (K_tot + eps))

9) Thermal stress penalty on coral growth (smooth ramp above threshold):
   stress = softplus(sst_anom(t-1) - tau_bleach)
   gF = exp(-beta_bleach_fast * stress)
   gS = exp(-beta_bleach_slow * stress)

10) Coral updates with smooth bounding to [0, K_tot]:
    F_raw = F(t-1) + gF * Growth_F - Pred_F - m_fast * F(t-1)
    S_raw = S(t-1) + gS * Growth_S - Pred_S - m_slow * S(t-1)
    // Smoothly enforce 0 <= X <= K_tot
    X_pos = softplus(X_raw)                   // non-negative
    X(t) = K_tot - softplus(K_tot - X_pos)    // upper-bounded

Likelihoods:
11) COTS observation (strictly positive): lognormal
    log(cots_dat(t)) ~ Normal(log(cots_pred(t)), sigma_cots_eff)

12) Coral observations (percent, mapped to proportion): logit-normal
    logit(fast_dat(t)/100) ~ Normal(logit(fast_pred(t)/100), sigma_fast_eff)
    logit(slow_dat(t)/100) ~ Normal(logit(slow_pred(t)/100), sigma_slow_eff)

Minimum standard deviations are enforced via sigma_eff = sqrt(sigma^2 + sigma_min^2)
Parameter bounds are softly penalized using softplus-based quadratic penalties (no hard constraints).
*/

// Data inputs (follow TMB conventions)
template<class Type>
Type objective_function<Type>::operator() () {
  // Time and forcings
  DATA_VECTOR(Year);          // Calendar year (year), used only for indexing and reporting
  DATA_VECTOR(sst_dat);       // Sea-surface temperature (Celsius), annual
  DATA_VECTOR(cotsimm_dat);   // Larval immigration (individuals m^-2 year^-1), annual

  // Observations (responses)
  DATA_VECTOR(cots_dat);      // Adult COTS abundance (individuals m^-2), positive
  DATA_VECTOR(fast_dat);      // Fast coral cover (Acropora) in percent [0,100]
  DATA_VECTOR(slow_dat);      // Slow coral cover (Faviidae/Porites) in percent [0,100]

  int n = Year.size();        // Number of time steps (years)
  Type eps = Type(1e-8);      // Small constant for numerical stability in divisions and logs

  // PARAMETERS (with inline documentation and suggested units)
  PARAMETER(r_cots_max);      // year^-1; Max per-capita growth rate of adult-equivalent COTS; estimated from data (initial estimate)
  PARAMETER(m_cots);          // year^-1; Background adult mortality of COTS; initial estimate
  PARAMETER(c_cots_density);  // (m^2 ind^-1) year^-1; Self-limitation coefficient producing logistic-like damping; initial estimate
  PARAMETER(e_cots_imm);      // (dimensionless) m^2; Conversion from larval immigration to adult-equivalent density per year; initial estimate
  PARAMETER(A_crit);          // individuals m^-2; Allee threshold scale; initial estimate
  PARAMETER(K_prey);          // proportion (0-1); Half-saturation of prey availability in COTS recruitment; initial estimate

  PARAMETER(beta_sst_cots);   // (Celsius^-1); Slope of SST effect on COTS recruitment (logistic scale); literature/initial estimate
  PARAMETER(f_sst_lo);        // dimensionless; Lower bound multiplier for SST effect on COTS recruitment; literature/initial estimate
  PARAMETER(f_sst_hi);        // dimensionless; Upper bound multiplier for SST effect on COTS recruitment; literature/initial estimate

  PARAMETER(attack);          // (year^-1 ind^-1 m^2); Attack rate of COTS on corals; initial estimate
  PARAMETER(handling);        // year; Handling time; initial estimate
  PARAMETER(pref_fast);       // dimensionless in [0,1]; Preference weight toward fast coral; literature/initial estimate
  PARAMETER(pref_slow);       // dimensionless in [0,1]; Preference weight toward slow coral; literature/initial estimate
  PARAMETER(holling_q);       // dimensionless; Shape exponent (q=1 Type II, q>1 Type III); literature/initial estimate

  PARAMETER(r_fast);          // year^-1; Intrinsic growth rate of fast corals; literature/initial estimate
  PARAMETER(r_slow);          // year^-1; Intrinsic growth rate of slow corals; literature/initial estimate
  PARAMETER(m_fast);          // year^-1; Background mortality of fast corals; initial estimate
  PARAMETER(m_slow);          // year^-1; Background mortality of slow corals; initial estimate
  PARAMETER(K_tot);           // proportion (0-1); Total available space for coral (fast+slow) as fraction of substrate; literature/initial estimate

  PARAMETER(beta_bleach_fast);// (dimensionless per Celsius anomaly); Bleaching penalty slope for fast coral; literature/initial estimate
  PARAMETER(beta_bleach_slow);// (dimensionless per Celsius anomaly); Bleaching penalty slope for slow coral; literature/initial estimate
  PARAMETER(tau_bleach);      // Celsius; SST anomaly threshold where bleaching penalties increase; literature/initial estimate

  // Observation error parameters
  PARAMETER(sigma_cots_log);     // SD on log scale for COTS observations; initial estimate
  PARAMETER(sigma_fast_logit);   // SD on logit scale for fast coral proportions; initial estimate
  PARAMETER(sigma_slow_logit);   // SD on logit scale for slow coral proportions; initial estimate

  // Suggested smooth bound penalties (biological ranges)
  Type pen = Type(0.0);       // Accumulated penalty for soft bounds

  // Bounds (must align with parameters.json suggestions)
  pen += bound_penalty(r_cots_max,      Type(0.0),  Type(10.0), Type(1.0));
  pen += bound_penalty(m_cots,          Type(0.0),  Type(2.0),  Type(1.0));
  pen += bound_penalty(c_cots_density,  Type(0.0),  Type(10.0), Type(1.0));
  pen += bound_penalty(e_cots_imm,      Type(0.0),  Type(2.0),  Type(1.0));
  pen += bound_penalty(A_crit,          Type(0.0),  Type(5.0),  Type(1.0));
  pen += bound_penalty(K_prey,          Type(1e-6), Type(1.0),  Type(1.0));

  pen += bound_penalty(beta_sst_cots,   Type(-5.0), Type(5.0),  Type(0.5));
  pen += bound_penalty(f_sst_lo,        Type(0.2),  Type(1.0),  Type(0.5));
  pen += bound_penalty(f_sst_hi,        Type(1.0),  Type(2.5),  Type(0.5));

  pen += bound_penalty(attack,          Type(0.0),  Type(50.0), Type(1.0));
  pen += bound_penalty(handling,        Type(0.0),  Type(10.0), Type(1.0));
  pen += bound_penalty(pref_fast,       Type(0.0),  Type(1.0),  Type(1.0));
  pen += bound_penalty(pref_slow,       Type(0.0),  Type(1.0),  Type(1.0));
  pen += bound_penalty(holling_q,       Type(1.0),  Type(3.0),  Type(1.0));

  pen += bound_penalty(r_fast,          Type(0.0),  Type(2.0),  Type(1.0));
  pen += bound_penalty(r_slow,          Type(0.0),  Type(1.0),  Type(1.0));
  pen += bound_penalty(m_fast,          Type(0.0),  Type(1.0),  Type(1.0));
  pen += bound_penalty(m_slow,          Type(0.0),  Type(1.0),  Type(1.0));
  pen += bound_penalty(K_tot,           Type(0.1),  Type(0.95), Type(1.0));

  pen += bound_penalty(beta_bleach_fast,Type(0.0),  Type(5.0),  Type(0.5));
  pen += bound_penalty(beta_bleach_slow,Type(0.0),  Type(5.0),  Type(0.5));
  pen += bound_penalty(tau_bleach,      Type(0.0),  Type(5.0),  Type(0.5));

  pen += bound_penalty(sigma_cots_log,    Type(0.01), Type(2.0), Type(1.0));
  pen += bound_penalty(sigma_fast_logit,  Type(0.01), Type(2.0), Type(1.0));
  pen += bound_penalty(sigma_slow_logit,  Type(0.01), Type(2.0), Type(1.0));

  // Precompute SST anomaly
  vector<Type> sst_anom(n);        // SST anomalies (Celsius)
  Type sst_mean = Type(0.0);
  for (int t = 0; t < n; t++) {    // sum for mean
    sst_mean += sst_dat(t);
  }
  sst_mean /= Type(n);
  for (int t = 0; t < n; t++) {
    sst_anom(t) = sst_dat(t) - sst_mean;
  }

  // Prediction vectors (match observed variable names with _pred suffix)
  vector<Type> cots_pred(n);       // COTS adults (ind m^-2), predicted
  vector<Type> fast_pred(n);       // Fast coral (%), predicted
  vector<Type> slow_pred(n);       // Slow coral (%), predicted

  // Auxiliary diagnostic vectors
  vector<Type> f_sst_vec(n);       // SST multiplier for COTS recruitment
  vector<Type> f_prey_vec(n);      // Prey multiplier for COTS recruitment
  vector<Type> r_eff_vec(n);       // Per-capita net rate for COTS
  vector<Type> predF_vec(n);       // Predation losses on fast coral (proportion per year)
  vector<Type> predS_vec(n);       // Predation losses on slow coral (proportion per year)
  vector<Type> free_space_vec(n);  // Free space (proportion)

  // Initialize predictions from observed initial conditions (no optimized initial states)
  cots_pred(0) = cots_dat(0);                                 // ind m^-2
  fast_pred(0) = fast_dat(0);                                 // percent
  slow_pred(0) = slow_dat(0);                                 // percent

  // Initialize diagnostics at t=0 for reporting only
  {
    Type A0 = cots_pred(0);                                                     // ind m^-2
    Type pF0 = clamp01(fast_pred(0) / Type(100.0), Type(1e-6));                 // proportion
    Type pS0 = clamp01(slow_pred(0) / Type(100.0), Type(1e-6));                 // proportion
    f_sst_vec(0) = f_sst_lo + (f_sst_hi - f_sst_lo) * invlogit(beta_sst_cots * sst_anom(0));
    Type prey_avail0 = pref_fast * pF0 + pref_slow * pS0;
    f_prey_vec(0) = prey_avail0 / (K_prey + prey_avail0 + eps);
    r_eff_vec(0) = r_cots_max * f_prey_vec(0) * f_sst_vec(0) * (A0 / (A0 + A_crit + eps)) - m_cots - c_cots_density * A0;
    predF_vec(0) = Type(0.0);
    predS_vec(0) = Type(0.0);
    free_space_vec(0) = softplus(K_tot - pF0 - pS0);
  }

  // Forward simulation without leakage: use only (t-1) predictions and forcings
  for (int t = 1; t < n; t++) {
    // Previous predicted coral in proportion and COTS abundance
    Type pF_prev = clamp01(fast_pred(t-1) / Type(100.0), Type(1e-6));           // proportion
    Type pS_prev = clamp01(slow_pred(t-1) / Type(100.0), Type(1e-6));           // proportion

    // Environmental multiplier based on previous year
    Type f_sst = f_sst_lo + (f_sst_hi - f_sst_lo) * invlogit(beta_sst_cots * sst_anom(t-1)); // dimensionless
    f_sst_vec(t) = f_sst;

    // Prey availability for COTS recruitment (saturating)
    Type prey_avail = pref_fast * pF_prev + pref_slow * pS_prev;                 // proportion-weighted
    Type f_prey = prey_avail / (K_prey + prey_avail + eps);                      // dimensionless [0,1)
    f_prey_vec(t) = f_prey;

    // Allee effect based on previous predicted COTS
    Type f_allee = cots_pred(t-1) / (cots_pred(t-1) + A_crit + eps);             // dimensionless [0,1)

    // Per-capita net rate with self-limitation
    Type r_eff = r_cots_max * f_prey * f_sst * f_allee - m_cots - c_cots_density * cots_pred(t-1);
    r_eff_vec(t) = r_eff;

    // Prediction equation: COTS abundance
    cots_pred(t) = cots_pred(t-1) * exp(r_eff) + e_cots_imm * cotsimm_dat(t-1);  // ind m^-2

    // Multi-prey Holling functional response for predation on corals
    Type Fq = pow(pF_prev + eps, holling_q);
    Type Sq = pow(pS_prev + eps, holling_q);
    Type V = pref_fast * Fq + pref_slow * Sq + eps;
    Type cons_per_pred = attack * V / (Type(1.0) + attack * handling * V + eps);  // proportion per predator per year
    Type share_F = (pref_fast * Fq) / V;                                          // dimensionless
    Type share_S = (pref_slow * Sq) / V;                                          // dimensionless
    Type Pred_F = cots_pred(t-1) * cons_per_pred * share_F;                       // proportion per year
    Type Pred_S = cots_pred(t-1) * cons_per_pred * share_S;                       // proportion per year
    predF_vec(t) = Pred_F;
    predS_vec(t) = Pred_S;

    // Free space limitation (smooth)
    Type free_space = softplus(K_tot - pF_prev - pS_prev);                        // proportion >= 0
    free_space_vec(t) = free_space;

    // Coral growth with SST penalties above threshold (smooth ramp)
    Type stress = softplus(sst_anom(t-1) - tau_bleach);                           // Celsius, >=0
    Type gF = exp(-beta_bleach_fast * stress);                                    // dimensionless
    Type gS = exp(-beta_bleach_slow * stress);                                    // dimensionless
    Type Growth_F = r_fast * pF_prev * (free_space / (K_tot + eps));              // proportion per year
    Type Growth_S = r_slow * pS_prev * (free_space / (K_tot + eps));              // proportion per year

    // Raw coral updates (proportions)
    Type F_raw = pF_prev + gF * Growth_F - Pred_F - m_fast * pF_prev;
    Type S_raw = pS_prev + gS * Growth_S - Pred_S - m_slow * pS_prev;

    // Smoothly bound to [0, K_tot]
    Type F_pos = softplus(F_raw);                          // >=0
    Type S_pos = softplus(S_raw);                          // >=0
    Type F_new = K_tot - softplus(K_tot - F_pos);          // <=K_tot
    Type S_new = K_tot - softplus(K_tot - S_pos);          // <=K_tot

    // Prediction equation: fast_pred (percent)
    fast_pred(t) = Type(100.0) * F_new;                    // percent

    // Prediction equation: slow_pred (percent)
    slow_pred(t) = Type(100.0) * S_new;                    // percent
  }

  // Likelihood with minimum standard deviations for stability
  Type sigma_min = Type(0.05); // floor
  Type sigma_cots_eff = sqrt( pow(sigma_cots_log, 2) + pow(sigma_min, 2) );      // log scale
  Type sigma_fast_eff = sqrt( pow(sigma_fast_logit, 2) + pow(sigma_min, 2) );    // logit scale
  Type sigma_slow_eff = sqrt( pow(sigma_slow_logit, 2) + pow(sigma_min, 2) );    // logit scale

  Type nll = Type(0.0);

  for (int t = 0; t < n; t++) {
    // COTS: lognormal
    nll -= dnorm( log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots_eff, true );

    // Fast coral: logit-normal on proportions
    Type pF_obs = clamp01(fast_dat(t) / Type(100.0), Type(1e-6));
    Type pF_pred = clamp01(fast_pred(t) / Type(100.0), Type(1e-6));
    Type logitF_obs = log(pF_obs + eps) - log(Type(1.0) - pF_obs + eps);
    Type logitF_pred = log(pF_pred + eps) - log(Type(1.0) - pF_pred + eps);
    nll -= dnorm( logitF_obs, logitF_pred, sigma_fast_eff, true );

    // Slow coral: logit-normal on proportions
    Type pS_obs = clamp01(slow_dat(t) / Type(100.0), Type(1e-6));
    Type pS_pred = clamp01(slow_pred(t) / Type(100.0), Type(1e-6));
    Type logitS_obs = log(pS_obs + eps) - log(Type(1.0) - pS_obs + eps);
    Type logitS_pred = log(pS_pred + eps) - log(Type(1.0) - pS_pred + eps);
    nll -= dnorm( logitS_obs, logitS_pred, sigma_slow_eff, true );
  }

  // Add smooth bound penalties
  nll += pen;

  // Reporting: ensure all _pred variables are reported
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  // Additional diagnostics helpful for interpretation
  REPORT(sst_anom);
  REPORT(f_sst_vec);
  REPORT(f_prey_vec);
  REPORT(r_eff_vec);
  REPORT(predF_vec);
  REPORT(predS_vec);
  REPORT(free_space_vec);
  REPORT(pen);

  return nll;
}
