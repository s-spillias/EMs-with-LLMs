#include <TMB.hpp>

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
// (6) COTS update with lagged immigration and lagged reproduction:
//     feed_gain = eta_A * (C_F + C_S)
//     A_eff_rep = (1 - rho_repro_lag) * A_{t-1} + rho_repro_lag * A_{t-2}   (for t=1, use A_{t-1} for both terms)
//     recruit   = rA * f_SST_A * Phi_A(t-1) * A_eff_rep / (1 + A_eff_rep/(K_A + tiny))
//     immig_eff = gamma_I * [ (1 - rho_lag_I) * cotsimm_{t-1} + rho_lag_I * cotsimm_{t-2} ]   (for t=1, use cotsimm_{t-1} for both terms)
//     mort      = mA * A_{t-1}
//     A_t       = clip( A_{t-1} + recruit + feed_gain + immig_eff - mort, tiny, +infty )
// All clips are smooth via softclip to avoid non-differentiabilities.

// Data inputs (time series)
// IMPORTANT: Use the exact same names as in the CSV headers.
template<class Type>
Type objective_function<Type>::operator() () {
  // Small constants for numerical stability
  const Type tiny = Type(1e-8);    // prevent division by zero
  const Type epsp = Type(1e-6);    // for proportions
  const Type kSmooth = Type(5.0);  // smoothness for softclip

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
  PARAMETER(eta_A);             // Efficiency of converting consumption into net adult gain (ind m^-2 per (ind m^-2 yr^-1)); initial estimate
  PARAMETER(rho_lag_I);         // Fraction of immigration realized with +1 year delay (dimensionless 0..1)
  PARAMETER(rho_repro_lag);     // Fraction of adult reproduction realized with +1 year extra delay (dimensionless 0..1)

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
  nll += pen_bounds(eta_A,   Type(0.0), Type(3.0), Type(0.5));
  // q_fr effectively Type II per literature
  nll += pen_bounds(q_fr,    Type(1.0), Type(1.000000001), Type(0.5));
  nll += pen_bounds(aF,      Type(0.0), Type(5.0), Type(0.3));
  nll += pen_bounds(aS,      Type(0.0), Type(5.0), Type(0.3));
  nll += pen_bounds(hF,      Type(0.0), Type(5.0), Type(0.3));
  nll += pen_bounds(hS,      Type(0.0), Type(5.0), Type(0.3));
  nll += pen_bounds(pref_F,  Type(0.0), Type(2.0), Type(0.2));
  nll += pen_bounds(pref_S,  Type(0.0), Type(2.0), Type(0.2));
  nll += pen_bounds(kappa_predF, Type(0.0), Type(10.0), Type(0.5));
  nll += pen_bounds(kappa_predS, Type(0.0), Type(10.0), Type(0.5));
  nll += pen_bounds(rF,      Type(0.0), Type(2.0), Type(0.5));
  nll += pen_bounds(rS,      Type(0.0), Type(1.5), Type(0.5));
  nll += pen_bounds(mF_base, Type(0.0), Type(2.0), Type(0.5));
  nll += pen_bounds(mS_base, Type(0.0), Type(2.0), Type(0.5));
  nll += pen_bounds(mF_bleach, Type(0.0), Type(2.0), Type(0.5));
  nll += pen_bounds(mS_bleach, Type(0.0), Type(2.0), Type(0.5));
  nll += pen_bounds(sst_bleach, Type(29.0), Type(34.5), Type(0.2));
  nll += pen_bounds(sst_scale_bleach, Type(0.1), Type(5.0), Type(0.2));
  nll += pen_bounds(alpha_bleach_growthF, Type(0.0), Type(1.0), Type(0.2));
  nll += pen_bounds(alpha_bleach_growthS, Type(0.0), Type(1.0), Type(0.2));
  nll += pen_bounds(log_sigma_cots, Type(-5.0), Type(3.0), Type(0.1));
  nll += pen_bounds(log_sigma_fast, Type(-5.0), Type(3.0), Type(0.1));
  nll += pen_bounds(log_sigma_slow, Type(-5.0), Type(3.0), Type(0.1));
  nll += pen_bounds(rho_lag_I, Type(0.0), Type(1.0), Type(0.2));
  nll += pen_bounds(rho_repro_lag, Type(0.0), Type(1.0), Type(0.2));

  // Prediction vectors (initialize with observed first data point to avoid leakage)
  vector<Type> cots_pred(T);  // predicted adult COTS density (ind m^-2)
  vector<Type> fast_pred(T);  // predicted fast-growing coral cover (%)
  vector<Type> slow_pred(T);  // predicted slow-growing coral cover (%)

  cots_pred(0) = cots_dat(0); // initial condition from data (no leakage beyond t=0)
  fast_pred(0) = fast_dat(0); // initial condition from data (no leakage beyond t=0)
  slow_pred(0) = slow_dat(0); // initial condition from data (no leakage beyond t=0)

  // Derived quantities for reporting
  vector<Type> K_A(T);        // coral-dependent COTS carrying capacity
  vector<Type> cons_fast(T);  // per-step feeding on fast coral (ind m^-2 yr^-1)
  vector<Type> cons_slow(T);  // per-step feeding on slow coral (ind m^-2 yr^-1)
  vector<Type> sst_mod_A(T);  // SST modifier on COTS growth
  vector<Type> allee_mult(T); // Allee multiplier

  // Time stepping (never use current-step responses for prediction)
  for(int t=1; t<T; t++){
    // Previous-step predictions (state)
    Type A_prev = cots_pred(t-1);
    Type F_prev = fast_pred(t-1);
    Type S_prev = slow_pred(t-1);

    // Forcing at previous step
    Type sst_prev = sst_dat(t-1);
    Type imm_prev = cotsimm_dat(t-1);

    // (1) COTS carrying capacity depends on coral cover (smooth, always >= tiny)
    K_A(t-1) = kK0 + kK_perc * (wK_F * F_prev + wK_S * S_prev); // ind m^-2
    Type K_A_prev = K_A(t-1) + tiny;

    // (3) SST modifier for COTS growth using a hump-shaped Gaussian around sst_ref
    Type zA = (sst_prev - sst_ref) / (sst_scale_A + tiny);
    sst_mod_A(t-1) = Type(1.0) + beta_sst_A * exp(Type(-0.5) * zA * zA);

    // (2) Allee effect multiplier (0..1)
    allee_mult(t-1) = invlogit_safe(k_allee * (A_prev - A_crit));

    // (4) Functional response calculations (resources as proportions)
    Type RF = (F_prev/Type(100.0));
    Type RS = (S_prev/Type(100.0));
    // Ensure strictly positive proportions
    RF = CppAD::CondExpGt(RF, epsp, RF, epsp);
    RS = CppAD::CondExpGt(RS, epsp, RS, epsp);

    Type aF_eff = aF * pref_F;
    Type aS_eff = aS * pref_S;

    Type RFq = pow(RF, q_fr);
    Type RSq = pow(RS, q_fr);

    Type den = Type(1.0) + hF * aF_eff * RFq + hS * aS_eff * RSq; // denominator (dimensionless)
    // Per-capita feeding rates on each prey (ind m^-2 yr^-1 per COTS multiplied by A)
    Type CF = A_prev * (aF_eff * RFq) / (den + tiny); // consumption on fast coral (ind m^-2 yr^-1)
    Type CS = A_prev * (aS_eff * RSq) / (den + tiny); // consumption on slow coral (ind m^-2 yr^-1)
    cons_fast(t) = CF;
    cons_slow(t) = CS;

    // (5) Coral updates with growth, mortality, and predation
    // Bleaching pressure (0..1 increasing with SST)
    Type bleach_press = invlogit_safe((sst_prev - sst_bleach) / (sst_scale_bleach + tiny));

    // Growth suppression factor from SST (1 - alpha * pressure)
    Type gF_env = Type(1.0) - alpha_bleach_growthF * bleach_press;
    Type gS_env = Type(1.0) - alpha_bleach_growthS * bleach_press;

    // Space limitation by total coral cover
    Type free_space = Type(1.0) - (F_prev + S_prev)/Type(100.0);
    free_space = CppAD::CondExpGt(free_space, Type(0.0), free_space, Type(0.0));

    // Logistic-like regrowth adjusted by environment
    Type grow_F = rF * F_prev * free_space * gF_env;
    Type grow_S = rS * S_prev * free_space * gS_env;

    // Mortality (base + SST-driven)
    Type mort_F = (mF_base + mF_bleach * bleach_press) * F_prev;
    Type mort_S = (mS_base + mS_bleach * bleach_press) * S_prev;

    // Predation loss (convert consumption to % loss)
    Type pred_F = kappa_predF * CF;
    Type pred_S = kappa_predS * CS;

    // Update with smooth clipping to [0, 100]
    Type F_next_unc = F_prev + grow_F - mort_F - pred_F;
    Type S_next_unc = S_prev + grow_S - mort_S - pred_S;

    Type F_next = softclip(F_next_unc, Type(0.0), Type(100.0), kSmooth);
    Type S_next = softclip(S_next_unc, Type(0.0), Type(100.0), kSmooth);

    // (6) COTS update
    // Effective spawner/producer abundance for reproduction with optional +1y delay
    Type A_prev2 = (t >= 2 ? cots_pred(t-2) : cots_pred(t-1));
    Type A_eff_rep = (Type(1.0) - rho_repro_lag) * A_prev + rho_repro_lag * A_prev2;

    Type recruit = rA * sst_mod_A(t-1) * allee_mult(t-1) * A_eff_rep / (Type(1.0) + A_eff_rep/(K_A_prev));
    Type feed_gain = eta_A * (CF + CS);

    // Lagged immigration mixture (uses only t-1 and t-2 forcing)
    Type imm_prev2 = (t >= 2 ? cotsimm_dat(t-2) : cotsimm_dat(t-1));
    Type imm_eff = gamma_I * ((Type(1.0) - rho_lag_I) * imm_prev + rho_lag_I * imm_prev2);

    Type mortA = mA * A_prev;
    Type A_next_unc = A_prev + recruit + feed_gain + imm_eff - mortA;

    // Smooth non-negativity for COTS (upper softness proportional to previous K)
    Type A_next = softclip(A_next_unc, tiny, A_prev + K_A_prev + Type(10.0), kSmooth);

    // Assign predictions at current time step
    fast_pred(t) = F_next;
    slow_pred(t) = S_next;
    cots_pred(t) = A_next;
  }

  // Fill last-step derived variables for completeness
  K_A(T-1) = kK0 + kK_perc * (wK_F * fast_pred(T-1) + wK_S * slow_pred(T-1));
  {
    Type zA_last = (sst_dat(T-1) - sst_ref) / (sst_scale_A + tiny);
    sst_mod_A(T-1) = Type(1.0) + beta_sst_A * exp(Type(-0.5) * zA_last * zA_last);
  }
  allee_mult(T-1) = invlogit_safe(k_allee * (cots_pred(T-1) - A_crit));
  cons_fast(0) = Type(0.0); // undefined at t=0 (no step); set to 0 for reporting
  cons_slow(0) = Type(0.0);

  // Observation likelihoods (include all observations at all times)
  for(int t=0; t<T; t++){
    // COTS (lognormal on positive densities)
    Type yA = log( CppAD::CondExpGt(cots_dat(t), tiny, cots_dat(t), tiny) );
    Type muA = log( CppAD::CondExpGt(cots_pred(t), tiny, cots_pred(t), tiny) );
    nll -= dnorm(yA, muA, sigma_cots, true);

    // Fast coral (logit-normal on proportions)
    Type pF_obs = (fast_dat(t) + Type(1e-3)) / (Type(100.0) + Type(2e-3));
    Type pF_prd = (fast_pred(t) + Type(1e-3)) / (Type(100.0) + Type(2e-3));
    pF_obs = CppAD::CondExpLt(pF_obs, epsp, epsp, pF_obs);
    pF_obs = CppAD::CondExpGt(pF_obs, Type(1.0)-epsp, Type(1.0)-epsp, pF_obs);
    pF_prd = CppAD::CondExpLt(pF_prd, epsp, epsp, pF_prd);
    pF_prd = CppAD::CondExpGt(pF_prd, Type(1.0)-epsp, Type(1.0)-epsp, pF_prd);
    Type zF_obs = log(pF_obs / (Type(1.0) - pF_obs));
    Type zF_prd = log(pF_prd / (Type(1.0) - pF_prd));
    nll -= dnorm(zF_obs, zF_prd, sigma_fast, true);

    // Slow coral (logit-normal)
    Type pS_obs = (slow_dat(t) + Type(1e-3)) / (Type(100.0) + Type(2e-3));
    Type pS_prd = (slow_pred(t) + Type(1e-3)) / (Type(100.0) + Type(2e-3));
    pS_obs = CppAD::CondExpLt(pS_obs, epsp, epsp, pS_obs);
    pS_obs = CppAD::CondExpGt(pS_obs, Type(1.0)-epsp, Type(1.0)-epsp, pS_obs);
    pS_prd = CppAD::CondExpLt(pS_prd, epsp, epsp, pS_prd);
    pS_prd = CppAD::CondExpGt(pS_prd, Type(1.0)-epsp, Type(1.0)-epsp, pS_prd);
    Type zS_obs = log(pS_obs / (Type(1.0) - pS_obs));
    Type zS_prd = log(pS_prd / (Type(1.0) - pS_prd));
    nll -= dnorm(zS_obs, zS_prd, sigma_slow, true);
  }

  // Report all predictions and selected derived quantities
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(K_A);
  REPORT(cons_fast);
  REPORT(cons_slow);
  REPORT(sst_mod_A);
  REPORT(allee_mult);
  REPORT(sigma_cots);
  REPORT(sigma_fast);
  REPORT(sigma_slow);

  return nll;
}
