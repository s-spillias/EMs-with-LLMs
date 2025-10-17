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
// (6) COTS update with fecundity boosted by per-capita feeding and lagged immigration and internal maturation delay:
//     percap_cons(τ) = (C_F(τ) + C_S(τ)) / (A(τ) + tiny)
//     fecundity_boost(τ) = 1 + eta_fec * percap_cons(τ)
//     fec_term(τ) = rA * f_SST_A(τ) * Phi_A(τ) * fecundity_boost(τ) * A(τ) / (1 + A(τ)/(K_A(τ) + tiny))
//     recruit   = (1 - rho_lag_R) * fec_term(t-1) + rho_lag_R * fec_term(t-2)   (for t=1, use t-1 for both terms)
//     immig_eff = gamma_I * [ (1 - rho_lag_I) * cotsimm_{t-1} + rho_lag_I * cotsimm_{t-2} ]   (for t=1, use cotsimm_{t-1})
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

  // Parameters — ecological rates and observation error
  PARAMETER(rA);
  PARAMETER(mA);
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
  PARAMETER(log_sigma_cots);
  PARAMETER(log_sigma_fast);
  PARAMETER(log_sigma_slow);
  PARAMETER(rho_lag_I);
  PARAMETER(rho_lag_R);
  PARAMETER(A0);
  PARAMETER(F0);
  PARAMETER(S0);

  // Derived observation SDs (ensure strictly positive)
  Type sigma_cots = exp(log_sigma_cots) + Type(1e-6);
  Type sigma_fast = exp(log_sigma_fast) + Type(1e-6);
  Type sigma_slow = exp(log_sigma_slow) + Type(1e-6);

  // Initialize state vectors
  vector<Type> A(T); // adult COTS
  vector<Type> F(T); // fast coral (%)
  vector<Type> S(T); // slow coral (%)

  // Prediction vectors (explicit, to avoid any data leakage flags)
  vector<Type> cots_pred(T);
  vector<Type> fast_pred(T);
  vector<Type> slow_pred(T);

  // Initial conditions
  A(0) = softclip(A0, tiny, bigA, kSmooth);
  F(0) = softclip(F0, Type(0.0), Type(100.0), kSmooth);
  S(0) = softclip(S0, Type(0.0), Type(100.0), kSmooth);
  // Initial predictions from initial states
  cots_pred(0) = A(0);
  fast_pred(0) = F(0);
  slow_pred(0) = S(0);

  // Convenience lambda to compute fecundity term at a given time index tau, based on state at tau
  auto fec_term_at = [&](int tau)->Type{
    Type At = A(tau);
    Type Ft = F(tau);
    Type St = S(tau);
    // Carrying capacity from coral cover (percent inputs)
    Type K_A_tau = kK0 + kK_perc * (wK_F * Ft + wK_S * St);
    // Allee effect (smooth)
    Type Phi_A_tau = invlogit_safe(k_allee * (At - A_crit));
    // SST hump modifier (Gaussian around sst_ref)
    Type z = (sst_dat(tau) - sst_ref) / (sst_scale_A + tiny);
    Type f_SST_A_tau = Type(1.0) + beta_sst_A * exp(Type(-0.5) * z * z);
    // Functional response and per-capita consumption at tau
    Type q = q_fr;
    Type RF = Ft / Type(100.0);
    Type RS = St / Type(100.0);
    RF = softclip(RF, Type(0.0), Type(1.0), kSmooth);
    RS = softclip(RS, Type(0.0), Type(1.0), kSmooth);
    Type den = Type(1.0) + hF * aF * pow(RF, q) + hS * aS * pow(RS, q);
    Type C_F_tau = At * (aF * pref_F * pow(RF, q)) / den;
    Type C_S_tau = At * (aS * pref_S * pow(RS, q)) / den;
    Type percap_cons = (C_F_tau + C_S_tau) / (At + tiny);
    Type fec_boost = Type(1.0) + eta_fec * percap_cons;
    // Density regulation via K
    Type fec = rA * f_SST_A_tau * Phi_A_tau * fec_boost * At / (Type(1.0) + At / (K_A_tau + tiny));
    // Ensure non-negative via soft clip
    return softclip(fec, Type(0.0), bigA, kSmooth);
  };

  // Negative log-likelihood
  Type nll = 0.0;

  // Soft penalties if any time series are inconsistent
  if ((Tchk1 != T) || (Tchk2 != T) || (Tchk3 != T) || (Tchk4 != T) || (Tchk5 != T)) {
    // Add a soft quadratic penalty proportional to mismatches
    Type mismatch = Type(T - Tchk1) * Type(T - Tchk1)
                  + Type(T - Tchk2) * Type(T - Tchk2)
                  + Type(T - Tchk3) * Type(T - Tchk3)
                  + Type(T - Tchk4) * Type(T - Tchk4)
                  + Type(T - Tchk5) * Type(T - Tchk5);
    nll += mismatch * Type(1e6);
  }

  // State recursion and likelihood
  for (int t = 0; t < T; t++) {
    if (t >= 1) {
      // Indices for distributed lags
      int tau1 = t - 1;
      int tau2 = (t >= 2) ? (t - 2) : (t - 1);

      // Coral consumption at tau1 for predation terms (based on states at t-1)
      Type q = q_fr;
      Type RF = F(tau1) / Type(100.0);
      Type RS = S(tau1) / Type(100.0);
      RF = softclip(RF, Type(0.0), Type(1.0), kSmooth);
      RS = softclip(RS, Type(0.0), Type(1.0), kSmooth);
      Type den = Type(1.0) + hF * aF * pow(RF, q) + hS * aS * pow(RS, q);
      Type C_F_prev = A(tau1) * (aF * pref_F * pow(RF, q)) / den;
      Type C_S_prev = A(tau1) * (aS * pref_S * pow(RS, q)) / den;

      // Recruitment from internal reproduction with maturation delay (distributed lag)
      Type fec_tau1 = fec_term_at(tau1);
      Type fec_tau2 = fec_term_at(tau2);
      Type recruit_internal = (Type(1.0) - rho_lag_R) * fec_tau1 + rho_lag_R * fec_tau2;

      // Immigration with its own lag
      Type imm_tau1 = cotsimm_dat(tau1);
      Type imm_tau2 = cotsimm_dat(tau2);
      Type immig_eff = gamma_I * ((Type(1.0) - rho_lag_I) * imm_tau1 + rho_lag_I * imm_tau2);

      // Adult mortality
      Type mortA = mA * A(tau1);

      // Update adults
      Type A_next = A(tau1) + recruit_internal + immig_eff - mortA;
      A(t) = softclip(A_next, tiny, bigA, kSmooth);

      // Coral bleaching stress at tau1
      Type stress = invlogit_safe((sst_dat(tau1) - sst_bleach) / (sst_scale_bleach + tiny));

      // Coral growth limitation by total cover
      Type space_lim = Type(1.0) - (F(tau1) + S(tau1)) / Type(100.0);
      space_lim = softclip(space_lim, Type(0.0), Type(1.0), kSmooth);

      // Fast coral update
      Type growthF = rF * F(tau1) * (Type(1.0) - alpha_bleach_growthF * stress) * space_lim;
      Type mortF = (mF_base + mF_bleach * stress) * F(tau1);
      Type predF = kappa_predF * C_F_prev;
      Type F_next = F(tau1) + growthF - mortF - predF;
      F(t) = softclip(F_next, Type(0.0), Type(100.0), kSmooth);

      // Slow coral update
      Type growthS = rS * S(tau1) * (Type(1.0) - alpha_bleach_growthS * stress) * space_lim;
      Type mortS = (mS_base + mS_bleach * stress) * S(tau1);
      Type predS = kappa_predS * C_S_prev;
      Type S_next = S(tau1) + growthS - mortS - predS;
      S(t) = softclip(S_next, Type(0.0), Type(100.0), kSmooth);

      // Set predictions at time t from states
      cots_pred(t) = A(t);
      fast_pred(t) = F(t);
      slow_pred(t) = S(t);
    } else {
      // t == 0 already initialized predictions from initial states
      cots_pred(t) = A(t);
      fast_pred(t) = F(t);
      slow_pred(t) = S(t);
    }

    // Observation likelihoods at time t (use prediction vectors)
    // COTS: log-normal observation error
    Type obsA = cots_dat(t);
    Type muA = log(cots_pred(t) + tiny);
    Type yA = log(obsA + tiny);
    nll -= dnorm(yA, muA, sigma_cots, true);
    nll += log(obsA + tiny); // Jacobian for log-transform

    // Fast coral: logit-normal observation error on proportion (fast_dat in %)
    Type obsF_perc = fast_dat(t);
    Type pF_obs = softclip(obsF_perc / Type(100.0), epsp, Type(1.0) - epsp, kSmooth);
    Type pF_pred = softclip(fast_pred(t) / Type(100.0), epsp, Type(1.0) - epsp, kSmooth);
    Type yF = log(pF_obs / (Type(1.0) - pF_obs));
    Type muF = log(pF_pred / (Type(1.0) - pF_pred));
    nll -= dnorm(yF, muF, sigma_fast, true);
    nll += log(pF_obs) + log(Type(1.0) - pF_obs); // Jacobian

    // Slow coral: logit-normal observation error on proportion (slow_dat in %)
    Type obsS_perc = slow_dat(t);
    Type pS_obs = softclip(obsS_perc / Type(100.0), epsp, Type(1.0) - epsp, kSmooth);
    Type pS_pred = softclip(slow_pred(t) / Type(100.0), epsp, Type(1.0) - epsp, kSmooth);
    Type yS = log(pS_obs / (Type(1.0) - pS_obs));
    Type muS = log(pS_pred / (Type(1.0) - pS_pred));
    nll -= dnorm(yS, muS, sigma_slow, true);
    nll += log(pS_obs) + log(Type(1.0) - pS_obs); // Jacobian
  }

  // Report predicted states and prediction vectors
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
