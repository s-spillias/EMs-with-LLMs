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

// AD-safe max
template<class Type>
Type tmb_max(Type a, Type b){
  return CppAD::CondExpGt(a, b, a, b);
}

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ---------------------------
  // Data inputs (time series)
  // ---------------------------
  DATA_VECTOR(cots_dat);     // observed adult COTS density (ind m^-2)
  DATA_VECTOR(fast_dat);     // observed fast coral cover (%)
  DATA_VECTOR(slow_dat);     // observed slow coral cover (%)
  DATA_VECTOR(sst_dat);      // sea surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat);  // larval immigration proxy (units per yr)

  int T = cots_dat.size();

  // ---------------------------
  // Parameters (scalars)
  // ---------------------------
  PARAMETER(rA);
  PARAMETER(mA);
  PARAMETER(mA_dd_max);
  PARAMETER(A_half_dd);
  PARAMETER(kK_perc);
  PARAMETER(kK0);
  PARAMETER(wK_F);
  PARAMETER(wK_S);
  PARAMETER(A_crit);
  PARAMETER(k_allee);
  PARAMETER(beta_sst_A);
  PARAMETER(sst_ref);
  PARAMETER(sst_scale_A);
  PARAMETER(gamma_I);
  PARAMETER(eta_fec);
  PARAMETER(tau_lag_I);
  PARAMETER(q_fr);
  PARAMETER(aF);
  PARAMETER(aS);
  PARAMETER(hF);
  PARAMETER(hS);
  PARAMETER(pref_F);
  PARAMETER(pref_S);
  PARAMETER(kappa_predF);
  PARAMETER(kappa_predS);
  PARAMETER(rF);
  PARAMETER(rS);
  PARAMETER(mF_base);
  PARAMETER(mS_base);
  PARAMETER(mF_bleach);
  PARAMETER(mS_bleach);
  PARAMETER(sst_bleach);
  PARAMETER(sst_scale_bleach);
  PARAMETER(alpha_bleach_growthF);
  PARAMETER(alpha_bleach_growthS);
  PARAMETER(C_crit);
  PARAMETER(k_coral_allee);
  PARAMETER(mA_starv_max);
  PARAMETER(cons_half_starv);
  PARAMETER(A0);
  PARAMETER(F0);
  PARAMETER(S0);
  PARAMETER(log_sigma_cots);
  PARAMETER(log_sigma_fast);
  PARAMETER(log_sigma_slow);

  // ---------------------------
  // Setup and storage
  // ---------------------------
  Type nll = 0.0;

  // Predictions (latent trajectories; deterministic process here)
  vector<Type> A(T);
  vector<Type> F(T);
  vector<Type> S(T);

  // Explicit prediction vectors corresponding to *_dat responses
  vector<Type> cots_pred(T);
  vector<Type> fast_pred(T);
  vector<Type> slow_pred(T);

  // Diagnostics to REPORT
  vector<Type> K_A(T);
  vector<Type> Phi_A(T);
  vector<Type> B_idx(T); // bleaching index
  vector<Type> cons_pc(T); // per-capita consumption (yr^-1)
  vector<Type> predF_loss(T);
  vector<Type> predS_loss(T);

  // Numerical constants
  const Type eps = Type(1e-8);
  const Type kclip = Type(10.0); // sharpness for softclip

  // Observation model SDs (add a small floor inside model)
  const Type minSigma = Type(1e-3);
  Type sigma_cots = exp(log_sigma_cots) + minSigma;
  Type sigma_fast = exp(log_sigma_fast) + minSigma;
  Type sigma_slow = exp(log_sigma_slow) + minSigma;

  // Initialize states at t=0 from parameters (no data leakage)
  A(0) = softclip(A0, Type(0.0), Type(1e6), kclip);
  F(0) = softclip(F0, Type(0.0), Type(100.0), kclip);
  S(0) = softclip(S0, Type(0.0), Type(100.0), kclip);

  // Also initialize prediction vectors at t=0
  cots_pred(0) = A(0);
  fast_pred(0) = F(0);
  slow_pred(0) = S(0);

  // Record derived at t=0 (using previous states where required)
  {
    // Compute bleaching index using SST at t=0 (applies to transitions into t=1)
    Type B0 = invlogit_safe((sst_dat(0) - sst_bleach) / (sst_scale_bleach + eps));
    B_idx(0) = B0;

    // Functional response based on resources at t=0
    Type RF = F(0) / Type(100.0);
    Type RS = S(0) / Type(100.0);
    Type q = q_fr;
    Type RFq = pow(softclip(RF, Type(0.0), Type(1.0), kclip), q);
    Type RSq = pow(softclip(RS, Type(0.0), Type(1.0), kclip), q);
    Type den = Type(1.0) + hF * aF * RFq + hS * aS * RSq;
    Type cF_pc = (aF * pref_F * RFq) / (den + eps); // per-capita rate on F
    Type cS_pc = (aS * pref_S * RSq) / (den + eps); // per-capita rate on S
    cons_pc(0) = cF_pc + cS_pc;

    // Predation losses at t=0 (used to transition to t=1)
    Type C_F_total = A(0) * cF_pc;
    Type C_S_total = A(0) * cS_pc;
    predF_loss(0) = kappa_predF * C_F_total;
    predS_loss(0) = kappa_predS * C_S_total;

    // COTS derived at t=0
    K_A(0) = kK0 + kK_perc * (wK_F * F(0) + wK_S * S(0));
    K_A(0) = tmb_max(K_A(0), Type(1e-6)); // ensure positive
    Phi_A(0) = invlogit_safe(k_allee * (A(0) - A_crit));
  }

  // Time loop for states (deterministic propagation)
  for(int t = 1; t < T; t++){
    // Prior (previous) states
    Type A_prev = A(t-1);
    Type F_prev = F(t-1);
    Type S_prev = S(t-1);

    // SST-based modifiers using previous SST
    Type sst_prev = sst_dat(t-1);
    // Bleaching index (0-1)
    Type B = invlogit_safe((sst_prev - sst_bleach) / (sst_scale_bleach + eps));
    B_idx(t) = B;

    // Coral space limitation (free space fraction)
    Type space_prev = tmb_max(Type(0.0), Type(1.0) - (F_prev + S_prev) / Type(100.0));

    // Coral depensation (broodstock limitation) using total coral cover at t-1
    Type X_tot = F_prev + S_prev; // percent
    Type Phi_C = invlogit_safe(k_coral_allee * (X_tot - C_crit));

    // Functional response on coral at t-1
    Type RF = F_prev / Type(100.0);
    Type RS = S_prev / Type(100.0);
    Type q = q_fr;
    Type RFq = pow(softclip(RF, Type(0.0), Type(1.0), kclip), q);
    Type RSq = pow(softclip(RS, Type(0.0), Type(1.0), kclip), q);
    Type den = Type(1.0) + hF * aF * RFq + hS * aS * RSq;
    Type cF_pc = (aF * pref_F * RFq) / (den + eps); // per-capita rate on F
    Type cS_pc = (aS * pref_S * RSq) / (den + eps); // per-capita rate on S
    Type c_pc   = cF_pc + cS_pc;                    // total per-capita consumption
    cons_pc(t) = c_pc;

    // Total consumption by all COTS
    Type C_F_total = A_prev * cF_pc; // ind m^-2 * yr^-1 -> (ind m^-2 yr^-1)
    Type C_S_total = A_prev * cS_pc;

    // Predation-induced coral cover loss (%/yr)
    Type predF = kappa_predF * C_F_total;
    Type predS = kappa_predS * C_S_total;
    predF_loss(t) = predF;
    predS_loss(t) = predS;

    // Coral growth modifiers due to bleaching
    Type gsup_F = softclip(Type(1.0) - alpha_bleach_growthF * B, Type(0.0), Type(1.0), kclip);
    Type gsup_S = softclip(Type(1.0) - alpha_bleach_growthS * B, Type(0.0), Type(1.0), kclip);

    // Coral intrinsic growth (logistic into free space) with depensation and bleaching suppression
    Type growthF = rF * F_prev * space_prev * gsup_F * Phi_C;
    Type growthS = rS * S_prev * space_prev * gsup_S * Phi_C;

    // Coral mortality (baseline + bleaching-induced), proportional to cover
    Type mF = mF_base + mF_bleach * B;
    Type mS = mS_base + mS_bleach * B;

    // Update coral states; clip to [0,100]
    Type F_next = F_prev + growthF - mF * F_prev - predF;
    Type S_next = S_prev + growthS - mS * S_prev - predS;

    F_next = softclip(F_next, Type(0.0), Type(100.0), kclip);
    S_next = softclip(S_next, Type(0.0), Type(100.0), kclip);

    F(t) = F_next;
    S(t) = S_next;

    // COTS carrying capacity and modifiers at t-1
    Type K_prev = kK0 + kK_perc * (wK_F * F_prev + wK_S * S_prev);
    K_prev = tmb_max(K_prev, Type(1e-6)); // avoid zero/negative K
    K_A(t) = K_prev;

    // COTS Allee effect and SST growth modifier at t-1
    Type PhiA_prev = invlogit_safe(k_allee * (A_prev - A_crit));
    Phi_A(t) = PhiA_prev;

    Type sst_dev = (sst_prev - sst_ref) / (sst_scale_A + eps);
    Type f_SST_A = Type(1.0) + beta_sst_A * exp(-Type(0.5) * sst_dev * sst_dev);

    // Fecundity boost from feeding (dimensionless multiplier >= 0)
    Type f_fec = Type(1.0) + eta_fec * c_pc;
    f_fec = tmb_max(f_fec, Type(0.0));

    // Extra high-density mortality (saturating with A_prev)
    Type m_dd = mA_dd_max * (A_prev / (A_prev + A_half_dd + eps));

    // Starvation mortality declines with per-capita consumption
    Type m_starv = mA_starv_max * (cons_half_starv / (cons_half_starv + c_pc + eps));

    // Logistic-like net growth with modifiers
    Type growthA = rA * PhiA_prev * f_SST_A * f_fec * A_prev * (Type(1.0) - A_prev / (K_prev + eps));

    // Mortality terms
    Type mortA = (mA + m_dd + m_starv) * A_prev;

    // Immigration with 0–2 year exponential lag kernel (normalized)
    Type tau = tau_lag_I; // assume >0 per parameter bounds
    Type w0 = exp(-Type(0.0) / (tau + eps));
    Type w1 = exp(-Type(1.0) / (tau + eps));
    Type w2 = exp(-Type(2.0) / (tau + eps));
    Type wsum = w0 + w1 + w2;

    Type I_eff = Type(0.0);
    // Use available lags only; earlier than t=0 contributes 0
    I_eff += (w0 / wsum) * cotsimm_dat(t);
    if(t - 1 >= 0) I_eff += (w1 / wsum) * cotsimm_dat(t - 1);
    if(t - 2 >= 0) I_eff += (w2 / wsum) * cotsimm_dat(t - 2);
    I_eff *= gamma_I;

    // Update COTS state; enforce non-negativity
    Type A_next = A_prev + growthA - mortA + I_eff;
    A_next = tmb_max(A_next, Type(0.0));

    A(t) = A_next;

    // Update prediction vectors at time t
    cots_pred(t) = A(t);
    fast_pred(t) = F(t);
    slow_pred(t) = S(t);
  }

  // ---------------------------
  // Observation models
  // ---------------------------

  // COTS: lognormal observation error
  for(int t = 0; t < T; t++){
    Type mu_log = log(cots_pred(t) + eps);
    Type y_log = log(cots_dat(t) + eps);
    nll -= dnorm(y_log, mu_log, sigma_cots, true);
  }

  // Coral: logit-normal observation error on fraction scale
  const Type eps_p = Type(1e-6);
  for(int t = 0; t < T; t++){
    // Fast coral
    Type p_pred_F = softclip(fast_pred(t) / Type(100.0), eps_p, Type(1.0) - eps_p, kclip);
    Type p_obs_F  = softclip(fast_dat(t) / Type(100.0), eps_p, Type(1.0) - eps_p, kclip);
    Type eta_pred_F = logit_safe(p_pred_F, eps_p, kclip);
    Type eta_obs_F  = logit_safe(p_obs_F,  eps_p, kclip);
    nll -= dnorm(eta_obs_F, eta_pred_F, sigma_fast, true);

    // Slow coral
    Type p_pred_S = softclip(slow_pred(t) / Type(100.0), eps_p, Type(1.0) - eps_p, kclip);
    Type p_obs_S  = softclip(slow_dat(t) / Type(100.0), eps_p, Type(1.0) - eps_p, kclip);
    Type eta_pred_S = logit_safe(p_pred_S, eps_p, kclip);
    Type eta_obs_S  = logit_safe(p_obs_S,  eps_p, kclip);
    nll -= dnorm(eta_obs_S, eta_pred_S, sigma_slow, true);
  }

  // ---------------------------
  // Reports
  // ---------------------------
  REPORT(A);
  REPORT(F);
  REPORT(S);
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(K_A);
  REPORT(Phi_A);
  REPORT(B_idx);
  REPORT(cons_pc);
  REPORT(predF_loss);
  REPORT(predS_loss);
  ADREPORT(A);
  ADREPORT(F);
  ADREPORT(S);
  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);

  return nll;
}
