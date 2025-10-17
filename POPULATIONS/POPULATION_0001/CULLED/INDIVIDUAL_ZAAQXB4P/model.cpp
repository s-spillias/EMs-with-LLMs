#include <TMB.hpp>

using namespace density;

// Helper: inverse-logit with numerical safety
template<class Type>
Type invlogit_safe(Type x){
  // maps R to (0,1) smoothly
  return Type(1) / (Type(1) + exp(-x));
}

// Helper: softplus with slope parameter k (>0) for smooth non-negativity
template<class Type>
Type softplus_k(Type x, Type k){
  // log(1 + exp(k x)) / k; avoids negative values smoothly
  // Use AD-safe log and exp; avoid std::log1p which is not AD-overloaded
  return log(Type(1.0) + exp(k * x)) / k;
}

// Helper: smooth clipping to [lo, hi] using softplus; k controls sharpness
template<class Type>
Type softclip(Type x, Type lo, Type hi, Type k){
  // Returns a smooth approximation of min(max(x, lo), hi)
  // lo + softplus(x-lo) - softplus(x-hi) ensures smooth transitions
  return lo + softplus_k(x - lo, k) - softplus_k(x - hi, k);
}

// Helper: logit with soft clipping to avoid boundaries
template<class Type>
Type logit_safe(Type p, Type eps, Type k){
  Type p_clip = softclip(p, eps, Type(1.0) - eps, k);
  return log(p_clip / (Type(1.0) - p_clip));
}

// Numbered model equations (conceptual):
// (1) COTS carrying capacity: K_A(t-1) = kK0 + kK_perc * (wK_F * F_{t-1} + wK_S * S_{t-1})
// (2) COTS Allee multiplier: Phi_A(t-1) = invlogit(k_allee * (A_{t-1} - A_crit))
// (3) SST modifier for COTS (hump-shaped): f_SST_A(t-1) = 1 + beta_sst_A * exp(-0.5 * ((sst_{t-1} - sst_ref) / sst_scale_A)^2)
// (4) Multi-prey functional response (Type II/III):
//     RF = F_{t-1}/100, RS = S_{t-1}/100
//     den = 1 + hF*aF*RF^q + hS*aS*RS^q
//     C_F = A_{t-1} * (aF*pref_F*RF^q) / den
//     C_S = A_{t-1} * (aS*pref_S*RS^q) / den
// (5) Coral updates (fast/slow):
//     growth_i = r_i * X_{t-1} * (1 - (F_{t-1}+S_{t-1})/100) * (1 - alpha_bleach_growth_i * invlogit((sst - sst_bleach)/sst_scale_bleach))
//     mort_i   = (m_i_base + m_i_bleach * invlogit((sst - sst_bleach)/sst_scale_bleach)) * X_{t-1}
//     pred_i   = kappa_pred_i * C_i
//     X_t      = clip( X_{t-1} + growth_i - mort_i - pred_i, 0, 100 )
// (6) Immigration pipeline (new):
//     J1_t     = k_settle * cotsimm_{t-1}
//     J2_t     = phi_J * J1_{t-1}
//     A_influx = phi_J * J2_{t-1}          // adults from immigration after ~2 years
// (7) COTS update with fecundity boosted by per-capita feeding:
//     percap_cons = (C_F + C_S) / (A_{t-1} + tiny)
//     fecundity_boost = 1 + eta_fec * percap_cons
//     recruit   = rA * f_SST_A * Phi_A * fecundity_boost * A_{t-1} / (1 + A_{t-1}/(K_A + tiny))
//     mort      = (mA + mA_starve * invlogit(k_starve * (c_starve50 - percap_cons))) * A_{t-1}
//     A_t       = clip( A_{t-1} + recruit + A_influx - mort, tiny, +infty )
// All clips are smooth via softclip to avoid non-differentiabilities.

// Data inputs (time series)
template<class Type>
Type objective_function<Type>::operator() () {
  // Small constants for numerical stability
  const Type tiny = Type(1e-8);    // prevent division by zero
  const Type epsp = Type(1e-6);    // for proportions
  const Type kSmooth = Type(5.0);  // smoothness for softclip
  const Type bigA = Type(1e6);     // large upper bound for COTS/juveniles

  // Time vector
  DATA_VECTOR(Year);           // calendar year (year)
  // Forcing variables (exogenous)
  DATA_VECTOR(sst_dat);        // sea-surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat);    // larval immigration (ind m^-2 yr^-1 proxy)
  // Response variables (observations)
  DATA_VECTOR(cots_dat);       // adult COTS density (ind m^-2)
  DATA_VECTOR(fast_dat);       // fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);       // slow-growing coral cover (%)

  int T = Year.size();         // number of time steps
  // Safety: ensure vectors have equal length (soft penalty if mismatch)
  int Tchk1 = sst_dat.size();
  int Tchk2 = cotsimm_dat.size();
  int Tchk3 = cots_dat.size();
  int Tchk4 = fast_dat.size();
  int Tchk5 = slow_dat.size();

  // Parameters — ecological rates and scalings
  PARAMETER(rA);                // COTS intrinsic growth rate (year^-1)
  PARAMETER(mA);                // COTS natural mortality rate (year^-1)
  // Resource-limited (starvation) mortality controls
  PARAMETER(mA_starve);         // Maximum added mortality from starvation (year^-1)
  PARAMETER(c_starve50);        // Per-capita consumption at which starvation mortality is half-max (yr^-1)
  PARAMETER(k_starve);          // Steepness of starvation response (dimensionless)
  PARAMETER(kK_perc);           // COTS K scaling per % coral
  PARAMETER(kK0);               // baseline COTS K independent of coral (ind m^-2)
  PARAMETER(wK_F);              // weight of fast coral in K (0..1)
  PARAMETER(wK_S);              // weight of slow coral in K (0..1)
  PARAMETER(A_crit);            // Allee threshold for COTS (ind m^-2)
  PARAMETER(k_allee);           // Steepness of Allee effect (m^2 ind^-1)
  PARAMETER(beta_sst_A);        // SST effect amplitude on COTS growth (dimensionless)
  PARAMETER(sst_ref);           // Reference SST for COTS response (deg C)
  PARAMETER(sst_scale_A);       // SST scale for COTS response (deg C)

  // Immigration pipeline parameters (replaces adult-additive immigration)
  PARAMETER(k_settle);          // conversion from cotsimm_dat to juveniles J1
  PARAMETER(phi_J);             // juvenile annual survival through cryptic stages (0..1)

  PARAMETER(eta_fec);           // Fecundity boost from per-capita feeding (dimensionless)

  // Functional response parameters (COTS feeding on corals)
  PARAMETER(q_fr);              // Shape of functional response (q=1 Type II); dimensionless
  PARAMETER(aF);                // Attack rate on fast coral (yr^-1)
  PARAMETER(aS);                // Attack rate on slow coral (yr^-1)
  PARAMETER(hF);                // Handling time toward fast coral (yr)
  PARAMETER(hS);                // Handling time toward slow coral (yr)
  PARAMETER(pref_F);            // Preference multiplier for fast coral (dimensionless)
  PARAMETER(pref_S);            // Preference multiplier for slow coral (dimensionless)
  PARAMETER(kappa_predF);       // Conversion of per-capita feeding to % loss
  PARAMETER(kappa_predS);       // Same for slow coral

  // Coral growth and mortality
  PARAMETER(rF);                // Intrinsic growth fast coral (yr^-1)
  PARAMETER(rS);                // Intrinsic growth slow coral (yr^-1)
  PARAMETER(mF_base);           // Base mortality fast coral (yr^-1)
  PARAMETER(mS_base);           // Base mortality slow coral (yr^-1)
  PARAMETER(mF_bleach);         // SST-driven extra mortality fast coral (yr^-1 multiplier)
  PARAMETER(mS_bleach);         // SST-driven extra mortality slow coral (yr^-1 multiplier)
  PARAMETER(sst_bleach);        // Bleaching onset SST (deg C)
  PARAMETER(sst_scale_bleach);  // Scale of bleaching response (deg C)
  PARAMETER(alpha_bleach_growthF); // SST suppression amplitude on fast coral growth (0-1)
  PARAMETER(alpha_bleach_growthS); // SST suppression amplitude on slow coral growth (0-1)

  // Initial states (estimated to avoid using observations in state recursion)
  PARAMETER(A0);                // initial adult COTS density (ind m^-2)
  PARAMETER(J10);               // initial first-year juvenile density (ind m^-2)
  PARAMETER(J20);               // initial second-year juvenile density (ind m^-2)
  PARAMETER(F0);                // initial fast coral cover (%)
  PARAMETER(S0);                // initial slow coral cover (%)

  // Observation model (log/logit-normal)
  PARAMETER(log_sigma_cots);    // log SD for log COTS obs
  PARAMETER(log_sigma_fast);    // log SD for logit fast coral obs
  PARAMETER(log_sigma_slow);    // log SD for logit slow coral obs

  // Likelihood accumulator
  Type nll = 0.0;

  // Soft penalties for length mismatches (never skip data; just penalize)
  if(T != Tchk1) nll += pow(Type(T - Tchk1), 2);
  if(T != Tchk2) nll += pow(Type(T - Tchk2), 2);
  if(T != Tchk3) nll += pow(Type(T - Tchk3), 2);
  if(T != Tchk4) nll += pow(Type(T - Tchk4), 2);
  if(T != Tchk5) nll += pow(Type(T - Tchk5), 2);

  // Convert log SDs to SDs with a minimum floor for numerical stability
  const Type min_sd = Type(0.05); // minimum observation SD
  Type sigma_cots = exp(log_sigma_cots) + min_sd;
  Type sigma_fast = exp(log_sigma_fast) + min_sd;
  Type sigma_slow = exp(log_sigma_slow) + min_sd;

  // Soft parameter bounds via penalties (biologically meaningful ranges)
  auto pen_bounds = [&](Type x, Type lo, Type hi, Type w){
    // penalty increases smoothly outside [lo, hi]
    return w * ( softplus_k(lo - x, Type(2.0)) + softplus_k(x - hi, Type(2.0)) );
  };
  // Accumulate penalties
  nll += pen_bounds(rA,   Type(0.0), Type(3.0), Type(1.0));
  nll += pen_bounds(mA,   Type(0.0), Type(2.0), Type(1.0));
  nll += pen_bounds(mA_starve, Type(0.0), Type(3.0), Type(0.8));
  nll += pen_bounds(c_starve50, Type(0.0), Type(3.0), Type(0.5));
  nll += pen_bounds(k_starve, Type(0.0), Type(50.0), Type(0.2));
  nll += pen_bounds(kK_perc, Type(0.0), Type(0.2), Type(0.5));
  nll += pen_bounds(kK0,  Type(0.0), Type(5.0), Type(0.5));
  nll += pen_bounds(wK_F, Type(0.0), Type(1.0), Type(0.2));
  nll += pen_bounds(wK_S, Type(0.0), Type(1.0), Type(0.2));
  nll += pen_bounds(A_crit, Type(0.0), Type(2.0), Type(0.5));
  nll += pen_bounds(k_allee, Type(0.0), Type(20.0), Type(0.5));
  nll += pen_bounds(beta_sst_A, Type(-1.0), Type(1.0), Type(0.2));
  nll += pen_bounds(sst_ref,  Type(28.0), Type(29.0), Type(0.2));
  nll += pen_bounds(sst_scale_A, Type(0.1), Type(5.0), Type(0.2));

  // Immigration pipeline
  nll += pen_bounds(k_settle, Type(0.0), Type(3.0), Type(0.5));
  nll += pen_bounds(phi_J,    Type(0.0), Type(1.0), Type(0.5));

  nll += pen_bounds(eta_fec,   Type(0.0), Type(3.0), Type(0.5));
  // q_fr effectively Type II per literature
  nll += pen_bounds(q_fr,    Type(1.0), Type(1.000000001), Type(0.5));
  nll += pen_bounds(aF,      Type(0.0), Type(5.0), Type(0.3));
  nll += pen_bounds(aS,      Type(0.0), Type(5.0), Type(0.3));
  nll += pen_bounds(hF,      Type(0.0), Type(5.0), Type(0.3));
  nll += pen_bounds(hS,      Type(0.0), Type(5.0), Type(0.3));
  nll += pen_bounds(pref_F,  Type(0.0), Type(2.0), Type(0.2));
  nll += pen_bounds(pref_S,  Type(0.0), Type(2.0), Type(0.2));
  nll += pen_bounds(kappa_predF, Type(0.0), Type(10.0), Type(0.3));
  nll += pen_bounds(kappa_predS, Type(0.0), Type(10.0), Type(0.3));

  nll += pen_bounds(rF,      Type(0.1), Type(0.5), Type(0.2));
  nll += pen_bounds(rS,      Type(0.0), Type(1.5), Type(0.2));
  nll += pen_bounds(mF_base, Type(0.0), Type(2.0), Type(0.2));
  nll += pen_bounds(mS_base, Type(0.0), Type(2.0), Type(0.2));
  nll += pen_bounds(mF_bleach, Type(0.0), Type(2.0), Type(0.2));
  nll += pen_bounds(mS_bleach, Type(0.0), Type(2.0), Type(0.2));
  nll += pen_bounds(sst_bleach, Type(29.0), Type(34.5), Type(0.2));
  nll += pen_bounds(sst_scale_bleach, Type(0.1), Type(5.0), Type(0.2));
  nll += pen_bounds(alpha_bleach_growthF, Type(0.0), Type(1.0), Type(0.2));
  nll += pen_bounds(alpha_bleach_growthS, Type(0.0), Type(1.0), Type(0.2));

  // Initial conditions penalties
  nll += pen_bounds(A0,  Type(0.0), Type(50.0), Type(0.2));
  nll += pen_bounds(J10, Type(0.0), Type(50.0), Type(0.2));
  nll += pen_bounds(J20, Type(0.0), Type(50.0), Type(0.2));
  nll += pen_bounds(F0,  Type(0.0), Type(100.0), Type(0.2));
  nll += pen_bounds(S0,  Type(0.0), Type(100.0), Type(0.2));

  // State vectors (named prediction vectors)
  vector<Type> cots_pred(T);
  vector<Type> fast_pred(T);
  vector<Type> slow_pred(T);

  // Internal juvenile pipeline states
  vector<Type> J1_pred(T);
  vector<Type> J2_pred(T);

  // Initialize states (do not use observations)
  cots_pred(0) = softclip(A0, tiny, bigA, kSmooth);
  J1_pred(0)   = softclip(J10, tiny, bigA, kSmooth);
  J2_pred(0)   = softclip(J20, tiny, bigA, kSmooth);
  fast_pred(0) = softclip(F0, Type(0.0), Type(100.0), kSmooth);
  slow_pred(0) = softclip(S0, Type(0.0), Type(100.0), kSmooth);

  // Time recursion (use previous time step states and exogenous inputs only)
  for(int t = 1; t < T; ++t){
    // Previous states
    Type A_prev  = cots_pred(t-1);
    Type F_prev  = fast_pred(t-1);
    Type S_prev  = slow_pred(t-1);
    Type J1_prev = J1_pred(t-1);
    Type J2_prev = J2_pred(t-1);

    // Scaled coral proportions
    Type RF = F_prev / Type(100.0);
    Type RS = S_prev / Type(100.0);

    // Multi-prey functional response
    Type RFq = pow(RF + tiny, q_fr);
    Type RSq = pow(RS + tiny, q_fr);
    Type den = Type(1.0) + hF * aF * RFq + hS * aS * RSq;
    Type C_F = A_prev * (aF * pref_F * RFq) / den; // consumption on fast coral
    Type C_S = A_prev * (aS * pref_S * RSq) / den; // consumption on slow coral

    // Bleaching/stress index (uses previous time step exogenous forcing)
    Type B = invlogit_safe((sst_dat(t-1) - sst_bleach) / sst_scale_bleach);

    // Coral dynamics
    Type space_lim = Type(1.0) - (F_prev + S_prev) / Type(100.0);
    space_lim = softclip(space_lim, Type(0.0), Type(1.0), kSmooth);

    Type growthF = rF * F_prev * space_lim * (Type(1.0) - alpha_bleach_growthF * B);
    Type mortF   = (mF_base + mF_bleach * B) * F_prev;
    Type predF   = kappa_predF * C_F;
    Type F_next  = softclip(F_prev + growthF - mortF - predF, Type(0.0), Type(100.0), kSmooth);

    Type growthS = rS * S_prev * space_lim * (Type(1.0) - alpha_bleach_growthS * B);
    Type mortS   = (mS_base + mS_bleach * B) * S_prev;
    Type predS   = kappa_predS * C_S;
    Type S_next  = softclip(S_prev + growthS - mortS - predS, Type(0.0), Type(100.0), kSmooth);

    // Immigration pipeline (delayed recruitment to adults)
    Type J1_next = softclip(k_settle * cotsimm_dat(t-1), tiny, bigA, kSmooth);
    Type J2_next = softclip(phi_J * J1_prev, tiny, bigA, kSmooth);
    Type A_influx = softclip(phi_J * J2_prev, Type(0.0), bigA, kSmooth);

    // COTS carrying capacity based on coral availability
    Type K_A_raw = kK0 + kK_perc * (wK_F * F_prev + wK_S * S_prev);
    Type K_A = softclip(K_A_raw, tiny, bigA, kSmooth);

    // Allee effect and SST response
    Type Phi_A = invlogit_safe(k_allee * (A_prev - A_crit));
    Type sst_z = (sst_dat(t-1) - sst_ref) / sst_scale_A;
    Type f_SST_A = Type(1.0) + beta_sst_A * exp(-Type(0.5) * sst_z * sst_z);

    // Per-capita feeding and fecundity boost
    Type percap_cons = (C_F + C_S) / (A_prev + tiny);
    Type fecundity_boost = Type(1.0) + eta_fec * percap_cons;

    // COTS dynamics: recruitment with density dependence, plus delayed influx, minus mortality
    Type recruit = rA * f_SST_A * Phi_A * fecundity_boost * A_prev / (Type(1.0) + A_prev / (K_A + tiny));
    Type mort    = (mA + mA_starve * invlogit_safe(k_starve * (c_starve50 - percap_cons))) * A_prev;
    Type A_next  = softclip(A_prev + recruit + A_influx - mort, tiny, bigA, kSmooth);

    // Assign next-step states
    cots_pred(t) = A_next;
    fast_pred(t) = F_next;
    slow_pred(t) = S_next;
    J1_pred(t)   = J1_next;
    J2_pred(t)   = J2_next;
  }

  // Observation likelihoods
  for(int t = 0; t < T; ++t){
    // Adults (log-normal)
    Type logA_obs  = log(cots_dat(t) + tiny);
    Type logA_pred = log(cots_pred(t) + tiny);
    nll -= dnorm(logA_obs, logA_pred, sigma_cots, true);

    // Fast coral (% -> proportion -> logit-normal)
    Type fast_obs_p  = softclip(fast_dat(t) / Type(100.0), epsp, Type(1.0) - epsp, kSmooth);
    Type fast_pred_p = softclip(fast_pred(t) / Type(100.0), epsp, Type(1.0) - epsp, kSmooth);
    Type fast_obs_logit  = log(fast_obs_p / (Type(1.0) - fast_obs_p));
    Type fast_pred_logit = log(fast_pred_p / (Type(1.0) - fast_pred_p));
    nll -= dnorm(fast_obs_logit, fast_pred_logit, sigma_fast, true);

    // Slow coral (% -> proportion -> logit-normal)
    Type slow_obs_p  = softclip(slow_dat(t) / Type(100.0), epsp, Type(1.0) - epsp, kSmooth);
    Type slow_pred_p = softclip(slow_pred(t) / Type(100.0), epsp, Type(1.0) - epsp, kSmooth);
    Type slow_obs_logit  = log(slow_obs_p / (Type(1.0) - slow_obs_p));
    Type slow_pred_logit = log(slow_pred_p / (Type(1.0) - slow_pred_p));
    nll -= dnorm(slow_obs_logit, slow_pred_logit, sigma_slow, true);
  }

  // Reports for diagnostics
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(J1_pred);
  REPORT(J2_pred);

  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);

  return nll;
}
