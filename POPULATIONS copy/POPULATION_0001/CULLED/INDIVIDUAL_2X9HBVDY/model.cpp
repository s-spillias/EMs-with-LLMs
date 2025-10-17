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
// (6) COTS update with fecundity boosted by per-capita feeding, coral-dependent settlement, and lagged immigration:
//     percap_cons = (C_F + C_S) / (A_{t-1} + tiny)
//     fecundity_boost = 1 + eta_fec * percap_cons
//     f_settle = invlogit( k_settle * ( w_settle_F * F_{t-1} + w_settle_S * S_{t-1} - c_settle50 ) )
//     recruit   = rA * f_SST_A * Phi_A * fecundity_boost * f_settle * A_{t-1} / (1 + A_{t-1}/(K_A + tiny))
//     immig_eff = gamma_I * [ (1 - rho_lag_I) * cotsimm_{t-1} + rho_lag_I * cotsimm_{t-2} ]   (for t=1, use cotsimm_{t-1} for both terms)
//     mort      = (mA + mA_starve * invlogit(k_starve * (c_starve50 - percap_cons))) * A_{t-1}
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
  // Added: resource-limited (starvation) mortality controls
  PARAMETER(mA_starve);         // Maximum added mortality from starvation (year^-1)
  PARAMETER(c_starve50);        // Per-capita consumption at which starvation mortality is half-max (yr^-1)
  PARAMETER(k_starve);          // Steepness of starvation response (dimensionless)
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
  PARAMETER(rho_lag_I);         // Fraction of immigration realized with +1 year delay (dimensionless 0..1)

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

  // Initial states (estimated to avoid using observations in state recursion)
  PARAMETER(A0);                // initial adult COTS density (ind m^-2)
  PARAMETER(F0);                // initial fast coral cover (%)
  PARAMETER(S0);                // initial slow coral cover (%)

  // Observation model (log/logit-normal)
  PARAMETER(log_sigma_cots);    // log SD for log COTS obs; initial estimate
  PARAMETER(log_sigma_fast);    // log SD for logit fast coral obs; initial estimate
  PARAMETER(log_sigma_slow);    // log SD for logit slow coral obs; initial estimate

  // New: settlement/survival modifier parameters
  PARAMETER(w_settle_F);        // weight of fast coral in settlement modifier (0..1)
  PARAMETER(w_settle_S);        // weight of slow coral in settlement modifier (0..1)
  PARAMETER(c_settle50);        // percent cover where f_settle = 0.5
  PARAMETER(k_settle);          // steepness (per % cover) of settlement modifier

  // Likelihood accumulator
  Type nll = 0.0;

  // Soft penalties for length mismatches (never skip data; just penalize)
  if(T != Tchk1) nll += pow(Type(T - Tchk1), 2);
  if(T != Tchk2) nll += pow(Type(T - Tchk2), 2);
  if(T != Tchk3) nll += pow(Type(T - Tchk3), 2);
  if(T != Tchk4) nll += pow(Type(T - Tchk4), 2);
  if(T != Tchk5) nll += pow(Type(T - Tchk5), 2);

  // Convert log SDs to SDs with a minimum floor for numerical stability
  Type sigma_cots = exp(log_sigma_cots) + Type(1e-6);
  Type sigma_fast = exp(log_sigma_fast) + Type(1e-6);
  Type sigma_slow = exp(log_sigma_slow) + Type(1e-6);

  // State vectors
  vector<Type> A(T);
  vector<Type> F(T);
  vector<Type> S(T);

  // Explicit prediction vectors for pipeline validation/export
  vector<Type> cots_pred(T);
  vector<Type> fast_pred(T);
  vector<Type> slow_pred(T);

  // Initialize with parameters (not observations) to avoid data leakage
  Type A_prev = softclip(A0, tiny, bigA, kSmooth);
  Type F_prev = softclip(F0, Type(0.0), Type(100.0), kSmooth);
  Type S_prev = softclip(S0, Type(0.0), Type(100.0), kSmooth);

  // Time recursion
  for(int t = 0; t < T; t++){
    // Forcing at previous step (use t-1, but for t=0 fall back to 0)
    int t_forc = (t == 0 ? 0 : t - 1);
    Type sst_prev = sst_dat(t_forc);

    // 1) Carrying capacity from coral cover at previous step
    Type K_A = kK0 + kK_perc * (wK_F * F_prev + wK_S * S_prev);
    K_A = softclip(K_A, tiny, bigA, kSmooth);

    // 2) Allee effect on COTS
    Type Phi_A = invlogit_safe(k_allee * (A_prev - A_crit));

    // 3) SST modifier (hump-shaped Gaussian) on COTS growth
    Type z_sst = (sst_prev - sst_ref) / sst_scale_A;
    Type f_SST_A = Type(1.0) + beta_sst_A * exp(Type(-0.5) * z_sst * z_sst);
    // Ensure strictly positive modifier
    f_SST_A = softclip(f_SST_A, Type(0.01), Type(10.0), kSmooth);

    // 4) Multi-prey functional response (feeding)
    Type RF = F_prev / Type(100.0);
    Type RS = S_prev / Type(100.0);
    // Avoid negative/overflow at extremes
    RF = softclip(RF, Type(0.0), Type(1.0), kSmooth);
    RS = softclip(RS, Type(0.0), Type(1.0), kSmooth);

    Type RFq = pow(RF + tiny, q_fr);
    Type RSq = pow(RSq + tiny, q_fr);

    Type den = Type(1.0) + hF * aF * RFq + hS * aS * RSq;
    den = softclip(den, Type(1e-6), Type(1e6), kSmooth);

    // Consumption rates on each coral type (ind m^-2 yr^-1 equivalent)
    Type C_F = A_prev * (aF * pref_F * RFq) / den;
    Type C_S = A_prev * (aS * pref_S * RSq) / den;

    // Per-capita consumption
    Type percap_cons = (C_F + C_S) / (A_prev + tiny);

    // Fecundity boost from feeding
    Type fecundity_boost = Type(1.0) + eta_fec * percap_cons;
    fecundity_boost = softclip(fecundity_boost, Type(0.0), Type(100.0), kSmooth);

    // 5) Coral updates with growth, mortality, bleaching, and predation
    Type f_bleach = invlogit_safe((sst_prev - sst_bleach) / sst_scale_bleach);

    // Fast coral
    Type growthF = rF * F_prev * (Type(1.0) - (F_prev + S_prev) / Type(100.0)) *
                   (Type(1.0) - alpha_bleach_growthF * f_bleach);
    Type mortF = (mF_base + mF_bleach * f_bleach) * F_prev;
    Type predF = kappa_predF * C_F;
    Type F_now = F_prev + growthF - mortF - predF;
    F_now = softclip(F_now, Type(0.0), Type(100.0), kSmooth);

    // Slow coral
    Type growthS = rS * S_prev * (Type(1.0) - (F_prev + S_prev) / Type(100.0)) *
                   (Type(1.0) - alpha_bleach_growthS * f_bleach);
    Type mortS = (mS_base + mS_bleach * f_bleach) * S_prev;
    Type predS = kappa_predS * C_S;
    Type S_now = S_prev + growthS - mortS - predS;
    S_now = softclip(S_now, Type(0.0), Type(100.0), kSmooth);

    // 6) Recruitment modifier linking coral to settlement/survival (previous-step states)
    Type C_comp = w_settle_F * F_prev + w_settle_S * S_prev; // in %
    Type f_settle = invlogit_safe(k_settle * (C_comp - c_settle50));

    // COTS immigration with 1-year lag fraction
    Type immig_eff;
    if(t == 0){
      immig_eff = gamma_I * cotsimm_dat(0);
    } else if(t == 1){
      // Use t-1 for both contemporaneous and lagged terms
      immig_eff = gamma_I * ((Type(1.0) - rho_lag_I) * cotsimm_dat(t - 1) + rho_lag_I * cotsimm_dat(t - 1));
    } else {
      immig_eff = gamma_I * ((Type(1.0) - rho_lag_I) * cotsimm_dat(t - 1) + rho_lag_I * cotsimm_dat(t - 2));
    }

    // Starvation mortality multiplier when per-capita consumption is low
    Type f_starve = invlogit_safe(k_starve * (c_starve50 - percap_cons));

    // COTS recruitment and mortality
    Type recruit = rA * f_SST_A * Phi_A * fecundity_boost * f_settle *
                   A_prev / (Type(1.0) + A_prev / (K_A + tiny));
    Type mortA = (mA + mA_starve * f_starve) * A_prev;

    Type A_now = A_prev + recruit + immig_eff - mortA;
    A_now = softclip(A_now, tiny, bigA, kSmooth);

    // Save states
    A(t) = A_now;
    F(t) = F_now;
    S(t) = S_now;

    // Populate explicit prediction vectors
    cots_pred(t) = A_now;
    fast_pred(t) = F_now;
    slow_pred(t) = S_now;

    // Advance previous-step states
    A_prev = A_now;
    F_prev = F_now;
    S_prev = S_now;
  }

  // Observation model contributions to likelihood
  // COTS: lognormal on density
  for(int t = 0; t < T; t++){
    Type y_obs = cots_dat(t);
    Type y_pred = cots_pred(t);
    Type ln_obs = log(y_obs + tiny);
    Type ln_pred = log(y_pred + tiny);
    // Lognormal: ln(y) ~ Normal(ln(mu), sigma)
    nll -= dnorm(ln_obs, ln_pred, sigma_cots, true);
    // Jacobian of log-transform (for density of y): add log(y_obs)
    nll += log(y_obs + tiny);
  }

  // Coral covers: logit-normal on proportions
  for(int t = 0; t < T; t++){
    // Fast coral
    Type p_obsF = softclip(fast_dat(t) / Type(100.0), epsp, Type(1.0) - epsp, kSmooth);
    Type p_predF = softclip(fast_pred(t) / Type(100.0), epsp, Type(1.0) - epsp, kSmooth);
    Type z_obsF = log(p_obsF / (Type(1.0) - p_obsF));
    Type z_predF = log(p_predF / (Type(1.0) - p_predF));
    nll -= dnorm(z_obsF, z_predF, sigma_fast, true);
    // Jacobian of logit-transform
    nll += log(p_obsF) + log(Type(1.0) - p_obsF);

    // Slow coral
    Type p_obsS = softclip(slow_dat(t) / Type(100.0), epsp, Type(1.0) - epsp, kSmooth);
    Type p_predS = softclip(slow_pred(t) / Type(100.0), epsp, Type(1.0) - epsp, kSmooth);
    Type z_obsS = log(p_obsS / (Type(1.0) - p_obsS));
    Type z_predS = log(p_predS / (Type(1.0) - p_predS));
    nll -= dnorm(z_obsS, z_predS, sigma_slow, true);
    // Jacobian
    nll += log(p_obsS) + log(Type(1.0) - p_obsS);
  }

  // Soft identifiability/regularization penalties (do not force hard constraints)
  // Encourage weights to sum near 1 to avoid overparameterization
  nll += Type(1e-2) * pow(w_settle_F + w_settle_S - Type(1.0), 2);
  nll += Type(1e-2) * pow(wK_F + wK_S - Type(1.0), 2);

  // Report states and useful intermediates
  REPORT(A);
  REPORT(F);
  REPORT(S);
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  ADREPORT(A);
  ADREPORT(F);
  ADREPORT(S);
  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);

  return nll;
}
