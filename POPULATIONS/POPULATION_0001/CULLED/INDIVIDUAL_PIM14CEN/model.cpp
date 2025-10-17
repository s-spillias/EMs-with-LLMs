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
// (6) COTS update with fecundity boosted by per-capita feeding and lagged immigration:
//     percap_cons = (C_F + C_S) / (A_{t-1} + tiny)
//     fecundity_boost = 1 + eta_fec * percap_cons
//     recruit   = rA * f_SST_A * Phi_A * fecundity_boost * A_{t-1} / (1 + A_{t-1}/(K_A + tiny))
//     immig_eff = gamma_I * sum_{k=0..2} w_k(tau_lag_I) * cotsimm_{t-1-k} (weights normalized; robust at boundaries)
//     mort      = mA * A_{t-1}
//     A_t       = clip( A_{t-1} + recruit + immig_eff - mort, tiny, +infty )
// All clips are smooth via softclip to avoid non-differentiabilities.

// Data inputs (time series)
// IMPORTANT: Use the exact same names as in the CSV headers.
template<class Type>
Type objective_function<Type>::operator() () {
  // Small constants for numerical stability
  const Type tiny = Type(1e-8);    // prevent division by zero
  const Type epsp = Type(1e-6);    // for proportions
  const Type kSmooth = Type(5.0);  // smoothness for softclip
  const Type bigA = Type(1e6);     // large upper bound for COTS

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
  PARAMETER(rA);                // COTS intrinsic growth rate (year^-1); init from literature/initial estimate
  PARAMETER(mA);                // COTS natural mortality rate (year^-1); initial estimate/literature
  PARAMETER(mA_dd_max);         // Max extra density-dependent adult mortality rate (year^-1)
  PARAMETER(A_half_dd);         // Half-saturation density for extra mortality (ind m^-2)
  PARAMETER(kK_perc);           // COTS K scaling per % coral (%^-1 * ind m^-2); initial estimate
  PARAMETER(kK0);               // baseline COTS K independent of coral (ind m^-2); initial estimate
  PARAMETER(wK_F);              // weight of fast coral in K (dimensionless [0,1]); initial estimate
  PARAMETER(wK_S);              // weight of slow coral in K (dimensionless [0,1]); initial estimate
  PARAMETER(A_crit);            // Allee threshold for COTS (ind m^-2); initial estimate/literature
  PARAMETER(k_allee);           // Steepness of Allee effect (m^2 ind^-1); initial estimate
  PARAMETER(beta_sst_A);        // SST effect amplitude on COTS growth (dimensionless); initial estimate
  PARAMETER(sst_ref);           // Reference SST for COTS response (deg C); literature/initial estimate
  PARAMETER(sst_scale_A);       // SST scale for COTS response (deg C); initial estimate

  PARAMETER(gamma_I);           // Conversion from cotsimm_dat to adult addition (ind m^-2 per (ind m^-2 yr^-1 proxy)); initial estimate
  PARAMETER(eta_fec);           // Fecundity boost from per-capita feeding (dimensionless)
  PARAMETER(tau_lag_I);         // Characteristic lag scale (years) for immigration kernel

  // Functional response parameters (COTS feeding on corals)
  PARAMETER(q_fr);              // Shape of functional response (q=1 Type II); dimensionless
  PARAMETER(aF);                // Attack rate on fast coral (yr^-1); initial estimate
  PARAMETER(aS);                // Attack rate on slow coral (yr^-1); initial estimate
  PARAMETER(hF);                // Handling time toward fast coral (yr); initial estimate
  PARAMETER(hS);                // Handling time toward slow coral (yr); initial estimate
  PARAMETER(pref_F);            // Preference multiplier for fast coral (dimensionless); initial estimate
  PARAMETER(pref_S);            // Preference multiplier for slow coral (dimensionless); initial estimate
  PARAMETER(kappa_predF);       // Conversion of per-capita feeding to % loss ( % per (ind m^-2 yr^-1) ); initial estimate
  PARAMETER(kappa_predS);       // Same for slow coral (% per (ind m^-2 yr^-1)); initial estimate

  // Coral growth and mortality
  PARAMETER(rF);                // Intrinsic growth fast coral (yr^-1); literature/initial estimate
  PARAMETER(rS);                // Intrinsic growth slow coral (yr^-1); literature/initial estimate
  PARAMETER(mF_base);           // Base mortality fast coral (yr^-1); initial estimate
  PARAMETER(mS_base);           // Base mortality slow coral (yr^-1); initial estimate
  PARAMETER(mF_bleach);         // SST-driven extra mortality fast coral (yr^-1 multiplier); initial estimate
  PARAMETER(mS_bleach);         // SST-driven extra mortality slow coral (yr^-1 multiplier); initial estimate
  PARAMETER(sst_bleach);        // Bleaching onset SST (deg C); literature/initial estimate
  PARAMETER(sst_scale_bleach);  // Scale of bleaching response (deg C); initial estimate
  PARAMETER(alpha_bleach_growthF); // SST suppression amplitude on fast coral growth (0-1); initial estimate
  PARAMETER(alpha_bleach_growthS); // SST suppression amplitude on slow coral growth (0-1); initial estimate

  // Starvation mortality parameters (new)
  PARAMETER(mA_starv_max);      // Max starvation-induced adult mortality rate (yr^-1)
  PARAMETER(cons_half_starv);   // Half-saturation per-capita consumption for starvation mortality (yr^-1)

  // Initial states (estimated to avoid using observations in state recursion)
  PARAMETER(A0);                // initial adult COTS density (ind m^-2)
  PARAMETER(F0);                // initial fast coral cover (%)
  PARAMETER(S0);                // initial slow coral cover (%)

  // Observation model (log/logit-normal)
  PARAMETER(log_sigma_cots);    // log SD for log COTS obs; initial estimate
  PARAMETER(log_sigma_fast);    // log SD for logit fast coral obs; initial estimate
  PARAMETER(log_sigma_slow);    // log SD for logit slow coral obs; initial estimate

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
  nll += pen_bounds(mA_dd_max, Type(0.0), Type(3.0), Type(0.5));
  nll += pen_bounds(A_half_dd, Type(0.05), Type(2.0), Type(0.5));
  nll += pen_bounds(kK_perc, Type(0.0), Type(0.2), Type(0.5));
  nll += pen_bounds(kK0,  Type(0.0), Type(5.0), Type(0.5));
  nll += pen_bounds(wK_F, Type(0.0), Type(1.0), Type(0.2));
  nll += pen_bounds(wK_S, Type(0.0), Type(1.0), Type(0.2));
  nll += pen_bounds(A_crit, Type(0.0), Type(2.0), Type(0.5));
  nll += pen_bounds(k_allee, Type(0.0), Type(20.0), Type(0.5));
  nll += pen_bounds(beta_sst_A, Type(-1.0), Type(1.0), Type(0.2));
  // Align with updated literature ranges
  nll += pen_bounds(sst_ref,  Type(28.0), Type(29.0), Type(0.2));
  nll += pen_bounds(sst_scale_A, Type(0.1), Type(5.0), Type(0.2));
  nll += pen_bounds(gamma_I, Type(0.0), Type(3.0), Type(0.5));
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

  // Bounds for lag time constant
  nll += pen_bounds(tau_lag_I, Type(0.1), Type(5.0), Type(0.5));

  // Bounds for starvation mortality parameters
  nll += pen_bounds(mA_starv_max, Type(0.0), Type(3.0), Type(0.5));
  nll += pen_bounds(cons_half_starv, Type(0.01), Type(3.0), Type(0.3));

  // State vectors (named as required prediction vectors)
  vector<Type> cots_pred(T);
  vector<Type> fast_pred(T);
  vector<Type> slow_pred(T);

  // Initialize states (do not use observations)
  cots_pred(0) = softclip(A0, tiny, bigA, kSmooth);
  fast_pred(0) = softclip(F0, Type(0.0), Type(100.0), kSmooth);
  slow_pred(0) = softclip(S0, Type(0.0), Type(100.0), kSmooth);

  // Time recursion (use previous time step states and exogenous inputs only)
  for(int t = 1; t < T; ++t){
    // Previous states
    Type A_prev = cots_pred(t-1);
    Type F_prev = fast_pred(t-1);
    Type S_prev = slow_pred(t-1);

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

    // COTS carrying capacity from coral cover
    Type K_A = kK0 + kK_perc * (wK_F * F_prev + wK_S * S_prev);
    K_A = softclip(K_A, tiny, bigA, kSmooth);

    // Allee effect and SST modifier
    Type Phi_A   = invlogit_safe(k_allee * (A_prev - A_crit));
    Type f_SST_A = Type(1.0) + beta_sst_A * exp(-Type(0.5) * pow((sst_dat(t-1) - sst_ref) / sst_scale_A, 2));

    // Fecundity boost via per-capita feeding
    Type percap_cons = (C_F + C_S) / (A_prev + tiny);
    Type fecundity_boost = Type(1.0) + eta_fec * percap_cons;

    // Recruitment with density dependence (Beverton-Holt-like)
    Type recruit = rA * f_SST_A * Phi_A * fecundity_boost * A_prev / (Type(1.0) + A_prev / (K_A + tiny));

    // Immigration with exponential lag kernel over 0..2 years; robust at boundaries
    Type tau = tau_lag_I;
    // Unnormalized weights for k=0,1,2
    Type w0 = Type(1.0);
    Type w1 = exp(-Type(1.0) / tau);
    Type w2 = exp(-Type(2.0) / tau);

    // Normalize over available lags (avoid using negative indices)
    Type wsum = Type(0.0);
    if(t - 1 >= 0) wsum += w0;
    if(t - 2 >= 0) wsum += w1;
    if(t - 3 >= 0) wsum += w2;
    // Guard against very early steps (wsum==0 should not happen as t>=1)
    Type immig_eff = Type(0.0);
    if(wsum > Type(0.0)){
      if(t - 1 >= 0) immig_eff += (w0 / wsum) * cotsimm_dat(t-1);
      if(t - 2 >= 0) immig_eff += (w1 / wsum) * cotsimm_dat(t-2);
      if(t - 3 >= 0) immig_eff += (w2 / wsum) * cotsimm_dat(t-3);
      immig_eff *= gamma_I;
    }

    // Adult mortality: baseline plus high-density extra loss
    Type mortA = mA * A_prev;

    // Smooth high-density extra mortality (disease/predation/culling)
    Type A_half_loc = softclip(A_half_dd, tiny, bigA, kSmooth);
    Type mortA_dd_rate = mA_dd_max * pow(A_prev, Type(2.0)) / (pow(A_half_loc, Type(2.0)) + pow(A_prev, Type(2.0)));
    Type mortA_dd = mortA_dd_rate * A_prev;

    // Starvation-induced extra mortality (new; increases when per-capita intake is low)
    const Type n_starv = Type(2.0); // Hill exponent (fixed)
    Type cons_half_loc = softclip(cons_half_starv, tiny, Type(10.0), kSmooth);
    Type mA_starv_rate = mA_starv_max * pow(cons_half_loc, n_starv) /
                         (pow(cons_half_loc, n_starv) + pow(percap_cons + tiny, n_starv));
    Type mortA_starv = mA_starv_rate * A_prev;

    // Update COTS
    Type A_next = softclip(A_prev + recruit + immig_eff - mortA - mortA_dd - mortA_starv, tiny, bigA, kSmooth);

    // Assign next states
    cots_pred(t) = A_next;
    fast_pred(t) = F_next;
    slow_pred(t) = S_next;
  }

  // Observation likelihood
  for(int t = 0; t < T; ++t){
    // COTS: log-normal observation model
    Type log_obs_A  = log(cots_dat(t) + tiny);
    Type log_pred_A = log(cots_pred(t) + tiny);
    nll -= dnorm(log_obs_A, log_pred_A, sigma_cots, true);

    // Coral: logit-normal observation model on % scaled to [0,1]
    Type pF_obs  = (fast_dat(t) / Type(100.0)) * (Type(1.0) - 2*epsp) + epsp;
    Type pS_obs  = (slow_dat(t) / Type(100.0)) * (Type(1.0) - 2*epsp) + epsp;
    Type pF_pred = (fast_pred(t) / Type(100.0)) * (Type(1.0) - 2*epsp) + epsp;
    Type pS_pred = (slow_pred(t) / Type(100.0)) * (Type(1.0) - 2*epsp) + epsp;

    Type logit_obs_F  = logit_safe(pF_obs, epsp, Type(20.0));
    Type logit_pred_F = logit_safe(pF_pred, epsp, Type(20.0));
    nll -= dnorm(logit_obs_F, logit_pred_F, sigma_fast, true);

    Type logit_obs_S  = logit_safe(pS_obs, epsp, Type(20.0));
    Type logit_pred_S = logit_safe(pS_pred, epsp, Type(20.0));
    nll -= dnorm(logit_obs_S, logit_pred_S, sigma_slow, true);
  }

  // Reporting
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}
