#include <TMB.hpp>

// Helper: inverse logit with numerical safety
template <class Type>
Type inv_logit_safe(Type x) {
  return Type(1.0) / (Type(1.0) + exp(-x));
}

// Helper: smooth positive-part to avoid negative states (numerically stable, differentiable)
template <class Type>
Type smooth_pos(Type x, Type delta) {
  // Returns ~ max(x, 0) but smooth; delta sets smoothness scale
  return (x + sqrt(x * x + delta * delta)) / Type(2.0);
}

// AD-safe min/max using CppAD conditional expressions
template <class Type>
Type tmb_fmax(Type a, Type b) {
  return CppAD::CondExpGe(a, b, a, b);
}
template <class Type>
Type tmb_fmin(Type a, Type b) {
  return CppAD::CondExpLe(a, b, a, b);
}

// Gaussian thermal performance curve (unitless multiplier in [0,1])
template <class Type>
Type gauss_perf(Type x, Type mu, Type sd, Type eps) {
  sd = tmb_fmax(sd, eps);
  Type z = (x - mu) / sd;
  return exp(Type(-0.5) * z * z);
}

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;

  Type nll = 0.0;                                         // Negative log-likelihood accumulator

  // Small constants for numerical stability
  const Type eps = Type(1e-8);                            // Generic epsilon for denominators and clamps
  const Type delta = Type(1e-6);                          // Smoothness scale for smooth_pos
  const Type min_sigma = Type(0.05);                      // Minimum observation SD (absolute floor)
  const Type min_phi = Type(5.0);                         // Minimum Beta precision to avoid extreme tails
  const Type max_prop = Type(0.999);                      // Upper bound for proportions used in likelihood

  // =========================
  // DATA (do not modify units)
  // =========================
  DATA_VECTOR(Year);             // Calendar year (year); used to align series and index time steps
  DATA_VECTOR(sst_dat);          // Sea-surface temperature (°C); external forcing
  DATA_VECTOR(cotsimm_dat);      // Larval immigration rate (individuals m^-2 year^-1); external forcing
  DATA_VECTOR(cots_dat);         // Adult COTS abundance (individuals m^-2); response variable
  DATA_VECTOR(fast_dat);         // Fast coral cover (Acropora) (fraction 0-1 of substrate); response variable
  DATA_VECTOR(slow_dat);         // Slow coral cover (Faviidae+Porites) (fraction 0-1 of substrate); response variable

  int T = Year.size();           // Number of time steps (years)

  // ==========================================
  // PARAMETERS (raw; transformed below as needed)
  // Each line includes: name, units, interpretation, and guidance.
  // ==========================================

  // COTS demography and outbreak trigger
  PARAMETER(log_r_cots_max);     // log(year^-1); max per-capita growth rate of adults given ample food and optimal temp; to estimate from outbreak ascent rates
  PARAMETER(log_m_cots);         // log(year^-1); background adult mortality rate; estimated from decline phases outside predation feedbacks
  PARAMETER(log_alpha_imm);      // log((adults m^-2) / (immigrants m^-2 yr^-1)); conversion of larval immigration to new adults within a year
  PARAMETER(imm_thr);            // individuals m^-2 yr^-1; immigration threshold center for smooth trigger; set by magnitude of pulses needed for outbreaks
  PARAMETER(imm_k);              // (yr m^2 individuals^-1); slope of immigration trigger (higher = sharper); controls onset sharpness
  PARAMETER(logit_Hh_food);      // logit(proportion); half-saturation of edible coral in food limitation (Michaelis–Menten); inferred from growth vs. coral cover
  PARAMETER(E_thr);              // proportion (0-1); smooth threshold of edible coral index for COTS growth; low cover suppresses growth
  PARAMETER(k_E);                // unitless; slope of edible coral threshold (higher = sharper transition)
  PARAMETER(sst_opt_cots);       // °C; thermal optimum for COTS demographic performance (recruitment/survival)
  PARAMETER(log_sst_sd_cots);    // log(°C); thermal breadth (SD) of COTS performance curve
  PARAMETER(log_Kcots0);         // log(individuals m^-2); baseline COTS carrying capacity independent of food
  PARAMETER(log_Kcots1);         // log(individuals m^-2); increment of COTS carrying capacity per unit edible coral (proportion)

  // Predation functional response and preference
  PARAMETER(log_attack_max);     // log(proportion coral yr^-1 per predator); max per-predator consumption rate
  PARAMETER(logit_h_type3);      // logit(proportion); half-saturation (proportion) for type III response (on edible coral index)
  PARAMETER(tau_pref_fast);      // dimensionless (logit preference); diet preference toward fast coral (Acropora)

  // Coral growth and mortality
  PARAMETER(log_r_fast);         // log(year^-1); intrinsic growth rate for fast coral
  PARAMETER(log_r_slow);         // log(year^-1); intrinsic growth rate for slow coral
  PARAMETER(logit_K_c);          // logit(proportion); total coral carrying capacity (F+S) as fraction of substrate
  PARAMETER(log_m_fast);         // log(year^-1); background mortality of fast coral (non-COTS)
  PARAMETER(log_m_slow);         // log(year^-1); background mortality of slow coral (non-COTS)
  PARAMETER(sst_opt_fast);       // °C; thermal optimum for fast coral growth
  PARAMETER(log_sst_sd_fast);    // log(°C); thermal breadth (SD) for fast coral growth
  PARAMETER(sst_opt_slow);       // °C; thermal optimum for slow coral growth
  PARAMETER(log_sst_sd_slow);    // log(°C); thermal breadth (SD) for slow coral growth

  // Observation model
  PARAMETER(log_sigma_cots);     // log(SD on log scale); observation SD for lognormal error on COTS abundance
  PARAMETER(log_phi_fast);       // log(precision); Beta precision for fast coral observation
  PARAMETER(log_phi_slow);       // log(precision); Beta precision for slow coral observation

  // Outbreak/density-dependent and starvation mortality
  PARAMETER(log_m_disease);      // log(year^-1); additional outbreak-associated mortality scale
  PARAMETER(N_burst_thr);        // individuals m^-2; adult density threshold for outbreak mortality onset
  PARAMETER(k_burst);            // (m^2 individuals^-1); steepness of outbreak mortality activation
  PARAMETER(log_m_starv);        // log(year^-1); starvation-related adult mortality scale linked to food limitation

  // ==========================
  // Transform parameters
  // ==========================
  Type r_cots_max = exp(log_r_cots_max);
  Type m_cots     = exp(log_m_cots);
  Type alpha_imm  = exp(log_alpha_imm);
  Type Hh_food    = inv_logit_safe(logit_Hh_food); // in (0,1)
  Type sst_sd_cots = exp(log_sst_sd_cots);
  Type Kcots0     = exp(log_Kcots0);
  Type Kcots1     = exp(log_Kcots1);
  Type attack_max = exp(log_attack_max);
  Type h_type3    = inv_logit_safe(logit_h_type3); // in (0,1)
  Type pref_fast  = inv_logit_safe(tau_pref_fast); // in (0,1)
  Type pref_slow  = Type(1.0) - pref_fast;

  Type r_fast     = exp(log_r_fast);
  Type r_slow     = exp(log_r_slow);
  Type K_c        = inv_logit_safe(logit_K_c); // total coral carrying capacity (fraction of substrate)
  Type m_fast     = exp(log_m_fast);
  Type m_slow     = exp(log_m_slow);
  Type sst_sd_fast = exp(log_sst_sd_fast);
  Type sst_sd_slow = exp(log_sst_sd_slow);

  Type sigma_cots = tmb_fmax(exp(log_sigma_cots), min_sigma);
  Type phi_fast   = tmb_fmax(exp(log_phi_fast), min_phi);
  Type phi_slow   = tmb_fmax(exp(log_phi_slow), min_phi);

  Type m_disease  = exp(log_m_disease);
  Type m_starv    = exp(log_m_starv);

  // ==========================
  // State predictions (one-step ahead; do not use current observations)
  // ==========================
  vector<Type> N_pred(T);
  vector<Type> F_pred(T);
  vector<Type> S_pred(T);
  // Required prediction vectors (naming for framework checks)
  vector<Type> cots_pred(T);
  vector<Type> fast_pred(T);
  vector<Type> slow_pred(T);

  // Initialize previous states without using observed response data to avoid leakage
  // Choose ecologically plausible small-to-moderate initial conditions
  Type N_prev = tmb_fmax(Type(0.1), Kcots0 * Type(0.5));   // adults m^-2
  Type F_prev = tmb_fmin(Type(0.5) * K_c, Type(0.45));     // fraction of substrate
  Type S_prev = tmb_fmin(Type(0.5) * K_c, Type(0.45));     // fraction of substrate

  for (int t = 0; t < T; t++) {
    // Environmental performance modifiers (use current external drivers)
    Type perf_cots = gauss_perf(sst_dat(t), sst_opt_cots, sst_sd_cots, eps);
    Type perf_fast = gauss_perf(sst_dat(t), sst_opt_fast, sst_sd_fast, eps);
    Type perf_slow = gauss_perf(sst_dat(t), sst_opt_slow, sst_sd_slow, eps);

    // Edible coral index based on previous coral states and preference weights
    Type E_prev = pref_fast * F_prev + pref_slow * S_prev;                 // edible fraction in [0,1]
    // Michaelis-Menten food saturation (fsat in [0,1])
    Type fsat = E_prev / (E_prev + Hh_food + eps);
    // Additional smooth threshold suppression at very low edible coral
    Type f_Ethr = inv_logit_safe(k_E * (E_prev - E_thr));

    // Type III functional response modifier on edible coral index
    Type E2 = E_prev * E_prev;
    Type h2 = h_type3 * h_type3 + eps;
    Type f_type3 = E2 / (h2 + E2);                                         // in [0,1]

    // Per-predator consumption rate (proportion coral per year)
    Type cons_rate_per_pred = attack_max * f_type3;                         // fraction/yr
    // Total potential consumption (fraction/yr)
    Type cons_total = cons_rate_per_pred * N_prev;

    // Split consumption across coral groups weighted by availability and preference
    Type avail_weight_fast = pref_fast * F_prev;
    Type avail_weight_slow = pref_slow * S_prev;
    Type denom_w = avail_weight_fast + avail_weight_slow + eps;
    Type share_fast = avail_weight_fast / denom_w;
    Type share_slow = avail_weight_slow / denom_w;

    // Cap group-specific consumption by available coral
    Type dF_cots_pot = cons_total * share_fast;
    Type dS_cots_pot = cons_total * share_slow;
    Type dF_cots = tmb_fmin(dF_cots_pot, F_prev);                           // cannot consume more than available
    Type dS_cots = tmb_fmin(dS_cots_pot, S_prev);

    // Coral logistic growth with total carrying capacity K_c (resource limitation via total cover)
    Type C_prev = F_prev + S_prev;
    Type crowd = Type(1.0) - C_prev / (K_c + eps);                          // in (-inf,1]; will be <=1
    crowd = tmb_fmax(crowd, Type(-1.0));                                    // avoid extreme negatives

    Type dF_growth = r_fast * perf_fast * F_prev * crowd;
    Type dS_growth = r_slow * perf_slow * S_prev * crowd;

    Type dF_bg_mort = m_fast * F_prev;
    Type dS_bg_mort = m_slow * S_prev;

    // Update coral states (ensure non-negative and <=1)
    Type F_next = smooth_pos(F_prev + dF_growth - dF_bg_mort - dF_cots, delta);
    Type S_next = smooth_pos(S_prev + dS_growth - dS_bg_mort - dS_cots, delta);
    // Cap at maximum proportion (<=1)
    F_next = tmb_fmin(F_next, Type(1.0));
    S_next = tmb_fmin(S_next, Type(1.0));

    // COTS carrying capacity depends on edible coral
    Type K_cots = Kcots0 + Kcots1 * E_prev;
    K_cots = tmb_fmax(K_cots, Type(0.0));                                   // capacity cannot be negative

    // Density-triggered outbreak mortality
    Type f_burst = inv_logit_safe(k_burst * (N_prev - N_burst_thr));
    // Starvation mortality increases as food saturation declines
    Type m_starv_eff = m_starv * (Type(1.0) - fsat);
    // Total mortality
    Type m_eff = m_cots + m_disease * f_burst + m_starv_eff;

    // Growth (resource, temperature, and threshold limited)
    Type r_eff = r_cots_max * fsat * f_Ethr * perf_cots;

    // Logistic growth and mortality
    Type dN_growth = r_eff * N_prev * (Type(1.0) - N_prev / (K_cots + eps));
    Type dN_mort = m_eff * N_prev;

    // Immigration trigger (smooth threshold) and recruitment
    Type imm_gate = inv_logit_safe(imm_k * (cotsimm_dat(t) - imm_thr));
    Type recruits = alpha_imm * cotsimm_dat(t) * imm_gate * perf_cots;

    // Update COTS state (ensure non-negative)
    Type N_next = smooth_pos(N_prev + dN_growth - dN_mort + recruits, delta);

    // Store predictions (one-step-ahead using previous states only)
    N_pred(t) = N_next;
    F_pred(t) = F_next;
    S_pred(t) = S_next;
    // Store in required-named prediction vectors
    cots_pred(t) = N_next;
    fast_pred(t) = F_next;
    slow_pred(t) = S_next;

    // ==========================
    // Observation likelihoods
    // ==========================
    // COTS: lognormal observation model on abundance
    Type y_cots = tmb_fmax(cots_dat(t), eps);
    Type mu_cots = tmb_fmax(cots_pred(t), eps);
    nll -= dnorm(log(y_cots), log(mu_cots), sigma_cots, true);

    // Coral: Beta observation models on proportions
    Type y_fast = tmb_fmin(tmb_fmax(fast_dat(t), eps), max_prop);
    Type y_slow = tmb_fmin(tmb_fmax(slow_dat(t), eps), max_prop);
    Type mu_fast = tmb_fmin(tmb_fmax(fast_pred(t), eps), max_prop);
    Type mu_slow = tmb_fmin(tmb_fmax(slow_pred(t), eps), max_prop);

    // Beta parameterization: mean = mu, precision = phi
    nll -= dbeta(y_fast, mu_fast * phi_fast, (Type(1.0) - mu_fast) * phi_fast, true);
    nll -= dbeta(y_slow, mu_slow * phi_slow, (Type(1.0) - mu_slow) * phi_slow, true);

    // Advance states
    N_prev = N_next;
    F_prev = F_next;
    S_prev = S_next;
  }

  // ==========================
  // Reports (optional)
  // ==========================
  REPORT(N_pred);
  REPORT(F_pred);
  REPORT(S_pred);
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}
