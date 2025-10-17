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
  PARAMETER(kK_perc);           // COTS K scaling per % coral (%^-1 * ind m^-2); initial estimate
  PARAMETER(kK0);               // baseline COTS K independent of coral (ind m^-2); initial estimate
  PARAMETER(wK_F);              // weight of fast coral in K (dimensionless [0,1]); initial estimate
  PARAMETER(wK_S);              // weight of slow coral in K (dimensionless [0,1]); initial estimate
  PARAMETER(A_crit);            // Allee threshold for COTS (ind m^-2); initial estimate/literature
  PARAMETER(k_allee);           // Steepness of Allee effect (m^2 ind^-1); initial estimate
  PARAMETER(beta_sst_A);        // SST effect amplitude on COTS growth (dimensionless); initial estimate
  PARAMETER(sst_ref);           // Reference SST for COTS response (deg C); literature/initial estimate
  PARAMETER(sst_scale_A);       // SST scale for COTS response (deg C); initial estimate

  PARAMETER(gamma_I);           // Conversion from cotsimm_dat to juvenile addition (ind m^-2 per (ind m^-2 yr^-1 proxy)); initial estimate
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

  // Juvenile stage parameters (new)
  PARAMETER(sJ);                // Larval-to-juvenile survival/settlement fraction (dimensionless)
  PARAMETER(mJ);                // Juvenile mortality rate (yr^-1)
  PARAMETER(phiJ);              // Juvenile maturation rate to adults (yr^-1)

  // Initial states (estimated to avoid using observations in state recursion)
  PARAMETER(A0);                // initial adult COTS density (ind m^-2)
  PARAMETER(F0);                // initial fast coral cover (%)
  PARAMETER(S0);                // initial slow coral cover (%)
  PARAMETER(J0);                // initial juvenile COTS density (ind m^-2) (new)

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

  nll += pen_bounds(rho_lag_I, Type(0.0), Type(1.0), Type(0.5));

  // New juvenile stage penalties
  nll += pen_bounds(sJ,   Type(0.0), Type(0.5), Type(0.5));
  nll += pen_bounds(mJ,   Type(0.0), Type(3.0), Type(0.5));
  nll += pen_bounds(phiJ, Type(0.0), Type(1.0), Type(0.5));

  // State vectors (named as required prediction vectors)
  vector<Type> cots_pred(T);
  vector<Type> fast_pred(T);
  vector<Type> slow_pred(T);
  vector<Type> juv_pred(T); // new juvenile state

  // Initialize states (do not use observations)
  cots_pred(0) = softclip(A0, tiny, bigA, kSmooth);
  fast_pred(0) = softclip(F0, Type(0.0), Type(100.0), kSmooth);
  slow_pred(0) = softclip(S0, Type(0.0), Type(100.0), kSmooth);
  juv_pred(0)  = softclip(J0, tiny, bigA, kSmooth);

  // Time recursion (use previous time step states and exogenous inputs only)
  for(int t = 1; t < T; ++t){
    // Previous states
    Type A_prev = cots_pred(t-1);
    Type F_prev = fast_pred(t-1);
    Type S_prev = slow_pred(t-1);
    Type J_prev = juv_pred(t-1);

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

    // Allee effect and SST modifier (on larval production)
    Type Phi_A   = invlogit_safe(k_allee * (A_prev - A_crit));
    Type f_SST_A = Type(1.0) + beta_sst_A * exp(-Type(0.5) * pow((sst_dat(t-1) - sst_ref) / sst_scale_A, 2));

    // Fecundity boost via per-capita feeding
    Type total_cons = C_F + C_S;
    Type percap_cons = total_cons / (A_prev + tiny);
    Type fecundity_boost = Type(1.0) + eta_fec * percap_cons;

    // Larval production from adults (feeds juvenile pool)
    Type L_prod = rA * f_SST_A * Phi_A * fecundity_boost * A_prev;

    // Exogenous immigration with 1-year distributed lag
    Type I_t1 = cotsimm_dat(t-1);
    Type I_t2 = (t-2 >= 0) ? cotsimm_dat(t-2) : Type(0.0);
    Type I_lag = (Type(1.0) - rho_lag_I) * I_t1 + rho_lag_I * I_t2;

    // Juvenile update
    Type J_next = (Type(1.0) - mJ) * J_prev + sJ * L_prod + gamma_I * I_lag;
    J_next = softclip(J_next, tiny, bigA, kSmooth);

    // Maturation to adults with Beverton-Holt style density dependence on adults via K_A
    Type matured = phiJ * J_prev / (Type(1.0) + A_prev / (K_A + tiny));

    // Adult update
    Type A_next = A_prev + matured - mA * A_prev;
    A_next = softclip(A_next, tiny, bigA, kSmooth);

    // Save next states
    fast_pred(t) = F_next;
    slow_pred(t) = S_next;
    juv_pred(t)  = J_next;
    cots_pred(t) = A_next;
  }

  // Observation likelihoods
  // COTS: log-normal on adult density
  for(int t = 0; t < T; ++t){
    Type y = cots_dat(t);
    Type mu = cots_pred(t);
    // clip for log
    y = softclip(y, tiny, bigA, kSmooth);
    mu = softclip(mu, tiny, bigA, kSmooth);
    nll -= dnorm(log(y), log(mu), sigma_cots, true);
  }

  // Coral: logit-normal on % cover scaled to proportion
  for(int t = 0; t < T; ++t){
    // Fast coral
    Type yF = fast_dat(t) / Type(100.0);
    Type muF = fast_pred(t) / Type(100.0);
    Type z_yF = logit_safe(yF, epsp, kSmooth);
    Type z_muF = logit_safe(muF, epsp, kSmooth);
    nll -= dnorm(z_yF, z_muF, sigma_fast, true);

    // Slow coral
    Type yS = slow_dat(t) / Type(100.0);
    Type muS = slow_pred(t) / Type(100.0);
    Type z_yS = logit_safe(yS, epsp, kSmooth);
    Type z_muS = logit_safe(muS, epsp, kSmooth);
    nll -= dnorm(z_yS, z_muS, sigma_slow, true);
  }

  // Reports
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(juv_pred);

  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);
  ADREPORT(juv_pred);

  return nll;
}
