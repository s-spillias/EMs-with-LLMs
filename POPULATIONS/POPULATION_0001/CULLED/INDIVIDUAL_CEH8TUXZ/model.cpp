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

// Helper: saturating Hill function for non-linear immigration efficacy
// H(x; I_half, nu) = x^nu / (x^nu + I_half^nu), x >= 0
template<class Type>
Type hill_saturating(Type x, Type I_half, Type nu, Type tiny){
  // Ensure non-negative input and positive half-saturation
  Type x_pos = CppAD::CondExpGt(x, Type(0.0), x, Type(0.0));
  Type Ih    = CppAD::CondExpGt(I_half, tiny, I_half, tiny);
  Type xnu   = pow(x_pos, nu);
  Type Ihnu  = pow(Ih,    nu);
  return xnu / (xnu + Ihnu + tiny);
}

// Helper: logit with safety bounds
template<class Type>
Type logit_safe(Type p, Type eps){
  Type pp = CppAD::CondExpLt(p, eps, eps, p);
  pp = CppAD::CondExpGt(pp, Type(1.0) - eps, Type(1.0) - eps, pp);
  return log(pp / (Type(1.0) - pp));
}

template<class Type>
Type objective_function<Type>::operator() () {
  // Small constants for numerical stability
  const Type tiny = Type(1e-8);    // prevent division by zero
  const Type epsp = Type(1e-6);    // for proportions/logit transforms
  const Type kSmooth = Type(5.0);  // smoothness for softclip

  // Data inputs (time series)
  DATA_VECTOR(Year);        // calendar year (year)
  // Forcing variables (exogenous)
  DATA_VECTOR(sst_dat);     // sea-surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat); // larval immigration (ind m^-2 yr^-1 proxy)
  // Response variables (observations)
  DATA_VECTOR(cots_dat);    // adult COTS density (ind m^-2)
  DATA_VECTOR(fast_dat);    // fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);    // slow-growing coral cover (%)

  int T = Year.size();      // number of time steps

  // Parameters — ecological rates and scalings
  PARAMETER(rA);                // COTS intrinsic growth rate (year^-1)
  PARAMETER(mA);                // COTS natural mortality rate (year^-1)
  PARAMETER(kK_perc);           // COTS K scaling per % coral
  PARAMETER(kK0);               // baseline COTS K independent of coral
  PARAMETER(wK_F);              // weight of fast coral in K
  PARAMETER(wK_S);              // weight of slow coral in K
  PARAMETER(A_crit);            // Allee threshold for COTS
  PARAMETER(k_allee);           // Steepness of Allee effect
  PARAMETER(beta_sst_A);        // SST effect amplitude on COTS growth
  PARAMETER(sst_ref);           // Reference SST for COTS response
  PARAMETER(sst_scale_A);       // SST scale for COTS response

  PARAMETER(gamma_I);           // Conversion from cotsimm_dat to adult addition
  PARAMETER(eta_A);             // Efficiency of converting consumption into net adult gain
  PARAMETER(rho_lag_I);         // Fraction of immigration realized with +1 year delay

  // Immigration nonlinearity parameters (Hill function)
  PARAMETER(I_half);            // Half-saturation of immigration efficacy
  PARAMETER(nu_I);              // Hill coefficient

  // Functional response parameters (COTS feeding on corals)
  PARAMETER(q_fr);              // Shape of functional response (q=1 Type II)
  PARAMETER(aF);                // Attack rate on fast coral (yr^-1)
  PARAMETER(aS);                // Attack rate on slow coral (yr^-1)
  PARAMETER(hF);                // Handling time toward fast coral (yr)
  PARAMETER(hS);                // Handling time toward slow coral (yr)
  PARAMETER(pref_F);            // Preference multiplier for fast coral
  PARAMETER(pref_S);            // Preference multiplier for slow coral
  PARAMETER(kappa_predF);       // Convert feeding to % loss (fast)
  PARAMETER(kappa_predS);       // Convert feeding to % loss (slow)

  // Coral growth and mortality
  PARAMETER(rF);                // Intrinsic growth fast coral (yr^-1)
  PARAMETER(rS);                // Intrinsic growth slow coral (yr^-1)
  PARAMETER(mF_base);           // Base mortality fast coral (yr^-1)
  PARAMETER(mS_base);           // Base mortality slow coral (yr^-1)
  PARAMETER(mF_bleach);         // SST-driven extra mortality fast coral (yr^-1)
  PARAMETER(mS_bleach);         // SST-driven extra mortality slow coral (yr^-1)
  PARAMETER(sst_bleach);        // Bleaching onset SST (deg C)
  PARAMETER(sst_scale_bleach);  // Scale of bleaching response (deg C)
  PARAMETER(alpha_bleach_growthF); // SST suppression amplitude on fast coral growth (0-1)
  PARAMETER(alpha_bleach_growthS); // SST suppression amplitude on slow coral growth (0-1)

  // Observation model (log/logit-normal)
  PARAMETER(log_sigma_cots);    // log SD for log COTS obs
  PARAMETER(log_sigma_fast);    // log SD for logit fast coral obs
  PARAMETER(log_sigma_slow);    // log SD for logit slow coral obs

  // Likelihood accumulator
  Type nll = 0.0;

  // Soft penalties for length mismatches (never skip data; just penalize)
  if(T != sst_dat.size())     nll += pow(Type(T - sst_dat.size()), 2);
  if(T != cotsimm_dat.size()) nll += pow(Type(T - cotsimm_dat.size()), 2);
  if(T != cots_dat.size())    nll += pow(Type(T - cots_dat.size()), 2);
  if(T != fast_dat.size())    nll += pow(Type(T - fast_dat.size()), 2);
  if(T != slow_dat.size())    nll += pow(Type(T - slow_dat.size()), 2);

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
  nll += pen_bounds(sst_ref,  Type(28.0), Type(29.0), Type(0.2));
  nll += pen_bounds(sst_scale_A, Type(0.1), Type(5.0), Type(0.2));
  nll += pen_bounds(gamma_I, Type(0.0), Type(3.0), Type(0.5));
  nll += pen_bounds(eta_A,   Type(0.0), Type(3.0), Type(0.5));
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
  nll += pen_bounds(I_half, Type(0.0), Type(1000.0), Type(0.2));
  nll += pen_bounds(nu_I,   Type(1.0), Type(8.0),    Type(0.2));

  // Prediction vectors
  vector<Type> cots_pred(T);  // predicted adult COTS density (ind m^-2)
  vector<Type> fast_pred(T);  // predicted fast-growing coral cover (%)
  vector<Type> slow_pred(T);  // predicted slow-growing coral cover (%)

  // Initialize from data at t=0 (no leakage beyond t=0)
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Derived quantities for reporting (initialize to zero to avoid uninitialized values)
  vector<Type> K_A(T);        K_A.setZero();       // coral-dependent COTS carrying capacity
  vector<Type> cons_fast(T);  cons_fast.setZero(); // per-step feeding on fast coral (ind m^-2 yr^-1)
  vector<Type> cons_slow(T);  cons_slow.setZero(); // per-step feeding on slow coral (ind m^-2 yr^-1)
  vector<Type> sst_mod_A(T);  sst_mod_A.setZero(); // SST modifier on COTS growth
  vector<Type> allee_mult(T); allee_mult.setZero(); // Allee multiplier

  // Time stepping (never use current-step responses for prediction)
  for(int t=1; t<T; t++){
    // Previous-step predictions (state)
    Type A_prev = cots_pred(t-1);
    Type F_prev = fast_pred(t-1);
    Type S_prev = slow_pred(t-1);

    // Forcing at previous step
    Type sst_prev = sst_dat(t-1);
    Type imm_prev1 = cotsimm_dat(t-1);
    // Safe integer index for t-2 (use 0 when t < 2)
    int idx2 = (t >= 2) ? (t - 2) : 0;
    Type imm_prev2 = cotsimm_dat(idx2);

    // (1) COTS carrying capacity depends on coral cover
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
    fast_pred(t) = softclip(F_next_unc, Type(0.0), Type(100.0), kSmooth);
    slow_pred(t) = softclip(S_next_unc, Type(0.0), Type(100.0), kSmooth);

    // (6) COTS update with lagged, non-linear immigration
    // Feeding feedback
    Type feed_gain = eta_A * (CF + CS);

    // Density-regulated recruitment (Beverton-Holt-like), modulated by SST and Allee
    Type recruit = rA * sst_mod_A(t-1) * allee_mult(t-1) * A_prev / (Type(1.0) + A_prev / K_A_prev);

    // Immigration efficacy via Hill transform applied to lagged immigration
    Type H1 = hill_saturating(imm_prev1, I_half, nu_I, tiny);
    Type H2 = hill_saturating(imm_prev2, I_half, nu_I, tiny);
    Type immig_eff = gamma_I * ((Type(1.0) - rho_lag_I) * H1 + rho_lag_I * H2);

    // Natural mortality
    Type mort_A = mA * A_prev;

    // State update with lower bound to avoid zero extinction by rounding
    Type A_next_unc = A_prev + recruit + feed_gain + immig_eff - mort_A;
    cots_pred(t) = softclip(A_next_unc, tiny, Type(1e9), kSmooth); // upper bound large to avoid binding
  }

  // Observation models
  // COTS: log-normal (on densities)
  for(int t=0; t<T; t++){
    Type log_obs = log(cots_dat(t) + tiny);
    Type log_mu  = log(cots_pred(t) + tiny);
    nll -= dnorm(log_obs, log_mu, sigma_cots, true);
  }
  // Corals: logit-normal on proportions
  for(int t=0; t<T; t++){
    // Fast coral
    Type p_obs_f = (fast_dat(t)/Type(100.0));
    Type p_mu_f  = (fast_pred(t)/Type(100.0));
    Type lo = epsp, hi = Type(1.0) - epsp;
    p_obs_f = CppAD::CondExpLt(p_obs_f, lo, lo, p_obs_f);
    p_obs_f = CppAD::CondExpGt(p_obs_f, hi, hi, p_obs_f);
    p_mu_f  = CppAD::CondExpLt(p_mu_f,  lo, lo, p_mu_f);
    p_mu_f  = CppAD::CondExpGt(p_mu_f,  hi, hi, p_mu_f);
    nll -= dnorm(logit_safe(p_obs_f, epsp), logit_safe(p_mu_f, epsp), sigma_fast, true);

    // Slow coral
    Type p_obs_s = (slow_dat(t)/Type(100.0));
    Type p_mu_s  = (slow_pred(t)/Type(100.0));
    p_obs_s = CppAD::CondExpLt(p_obs_s, lo, lo, p_obs_s);
    p_obs_s = CppAD::CondExpGt(p_obs_s, hi, hi, p_obs_s);
    p_mu_s  = CppAD::CondExpLt(p_mu_s,  lo, lo, p_mu_s);
    p_mu_s  = CppAD::CondExpGt(p_mu_s,  hi, hi, p_mu_s);
    nll -= dnorm(logit_safe(p_obs_s, epsp), logit_safe(p_mu_s, epsp), sigma_slow, true);
  }

  // Reports (for diagnostics)
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(K_A);
  REPORT(cons_fast);
  REPORT(cons_slow);
  REPORT(sst_mod_A);
  REPORT(allee_mult);

  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);

  return nll;
}
