#include <TMB.hpp>

// Helper functions with comments
template<class Type>
Type invlogit(Type x) { // Smoothly maps real line to (0,1), unitless
  return Type(1) / (Type(1) + exp(-x));
}

template<class Type>
Type softplus(Type x) { // Smooth positive transform to avoid negatives; units follow input
  return log(Type(1) + exp(x));
}

template<class Type>
Type square(Type x) { return x * x; } // Convenience, units: x^2

// Model description of equations (numbers referenced in comments below):
/*
(1) Attack rate temperature effect: a_eff_t = a0 * exp(theta_a_sst * (sst_t - sst_ref))
(2) Multi-prey Holling II per-capita consumption:
    c_f = (a_eff_t * pref_fast * F)/(1 + a_eff_t * h * (pref_fast * F + (1 - pref_fast) * S) + eps)
    c_s = (a_eff_t * (1 - pref_fast) * S)/(1 + a_eff_t * h * (pref_fast * F + (1 - pref_fast) * S) + eps)
(3) Vulnerability (smooth threshold): gamma_i = invlogit(k_v_i * (X_i - thr_v_i))
(4) Coral predation loss: loss_i = pred_scale_i * C * c_i * gamma_i
(5) Coral growth (space-limited, thermal modulation):
    g_i = exp(-0.5 * ((sst - sst_opt_i)/sst_sd_i)^2)
    Growth_i = r_i * X_i * (1 - (F + S)/K_tot) * g_i
(6) Coral updates: X_i(t) = X_i(t-1) + Growth_i - loss_i   (i ∈ {fast, slow})
(7) Food index and saturation for COTS reproduction:
    food = wrep_fast * F + (1 - wrep_fast) * S
    s_food = (alpha_food * food) / (1 + alpha_food * food)
(8) Coral cover outbreak trigger (smooth): trig = invlogit(k_tr * ((F + S) - coral_thr))
(9) COTS reproduction and immigration (thermal modulation g_C):
    g_C = exp(-0.5 * ((sst - sst_opt_C)/sst_sd_C)^2)
    R = fec_eff * C * s_food * trig * g_C + imm_eff * cotsimm * g_C
(10) COTS mortality (temperature-modified, bounded): mort = 0.99 * invlogit(logit_m0 + m_temp * (sst - sst_opt_C))
(11) Density dependence (Beverton–Holt-like): BH(x) = x / (1 + betaC * x)
(12) COTS update:
     C_post = C * (1 - mort) + R
     C(t) = BH(C_post)
(13) Observation model (lognormal, with sigma floors):
     y ~ LogNormal(log(pred), sigma_y), applied to cots_dat, fast_dat, slow_dat
*/

// Data inputs
template<class Type>
Type objective_function<Type>::operator() () {
  using CppAD::log;
  Type nll = 0.0;                                  // Negative log-likelihood accumulator, unitless
  const Type eps = Type(1e-8);                     // Small constant to avoid division by zero
  const Type tiny = Type(1e-12);                   // Smaller constant for logs and floors

  // Time series data (must use exact names/columns)
  DATA_VECTOR(Year);                               // Calendar year, integer or real
  DATA_VECTOR(sst_dat);                            // Sea-surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);                        // COTS larval immigration rate (indiv m^-2 yr^-1)
  DATA_VECTOR(cots_dat);                           // Observed adult COTS abundance (indiv m^-2)
  DATA_VECTOR(fast_dat);                           // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);                           // Observed slow-growing coral cover (%)

  int T = cots_dat.size();                         // Number of time steps (years)
  // Ensure all series have equal length, otherwise add a large penalty
  if ((int)Year.size() != T || (int)sst_dat.size() != T || (int)cotsimm_dat.size() != T ||
      (int)fast_dat.size() != T || (int)slow_dat.size() != T) {
    nll += Type(1e6);                              // Penalty if input lengths mismatch
  }

  // PARAMETERS (unconstrained), with transforms to biologically meaningful ranges
  // Coral growth
  PARAMETER(log_r_fast);                           // log intrinsic growth rate fast coral (yr^-1), initial via literature
  PARAMETER(log_r_slow);                           // log intrinsic growth rate slow coral (yr^-1), initial via literature
  PARAMETER(kappa_tot);                            // unconstrained -> total coral carrying capacity K_tot (%) via invlogit

  // Predation functional response
  PARAMETER(log_a0);                               // log base attack rate per COTS per % coral (yr^-1 %-1)
  PARAMETER(log_h);                                // log handling time (yr %^-1) per COTS across prey
  PARAMETER(logit_pref_fast);                      // preference for fast coral in [0,1], unitless
  PARAMETER(log_pred_scale_fast);                  // log scaling from per-capita consumption to % cover loss for fast
  PARAMETER(log_pred_scale_slow);                  // log scaling from per-capita consumption to % cover loss for slow
  PARAMETER(log_k_v_fast);                         // log slope for vulnerability sigmoid (per %)
  PARAMETER(log_k_v_slow);                         // log slope for vulnerability sigmoid (per %)
  PARAMETER(logit_thr_v_fast);                     // logit-transformed threshold (% in [0,100]) for fast vulnerability
  PARAMETER(logit_thr_v_slow);                     // logit-transformed threshold (% in [0,100]) for slow vulnerability

  // COTS reproduction and immigration
  PARAMETER(log_fec_eff);                          // log fecundity/recruitment efficiency (yr^-1)
  PARAMETER(log_alpha_food);                       // log saturation coefficient for food effect (%-1)
  PARAMETER(logit_wrep_fast);                      // weight of fast coral in food index [0,1], unitless
  PARAMETER(log_imm_eff);                          // log scaling of immigration contribution (indiv m^-2 per unit cotsimm_dat)

  // Environmental effects (temperature)
  PARAMETER(theta_a_sst);                          // slope linking SST to attack rate (°C^-1)
  PARAMETER(sst_ref);                              // reference SST for attack rate effect (°C)
  PARAMETER(sst_opt_fast);                         // thermal optimum for fast coral growth (°C)
  PARAMETER(log_sst_sd_fast);                      // log SD of fast coral thermal response (°C)
  PARAMETER(sst_opt_slow);                         // thermal optimum for slow coral growth (°C)
  PARAMETER(log_sst_sd_slow);                      // log SD of slow coral thermal response (°C)
  PARAMETER(sst_opt_C);                            // thermal optimum for COTS reproduction (°C)
  PARAMETER(log_sst_sd_C);                         // log SD of COTS thermal response (°C)

  // COTS mortality and density dependence
  PARAMETER(logit_m0);                             // baseline annual mortality in (0,1), unitless
  PARAMETER(m_temp);                               // temperature slope on mortality (°C^-1), unitless
  PARAMETER(log_betaC);                            // log Beverton–Holt density parameter (m^2 indiv^-1)

  // Outbreak trigger on coral
  PARAMETER(logit_coral_thr);                      // logit-transformed threshold total coral (%) in [0,100]
  PARAMETER(log_k_tr);                             // log slope for outbreak trigger (per %)

  // Initial states (strictly positive)
  PARAMETER(log_C0);                               // log initial adult COTS (indiv m^-2)
  PARAMETER(log_F0);                               // log initial fast coral cover (%)
  PARAMETER(log_S0);                               // log initial slow coral cover (%)

  // Observation error (lognormal SDs)
  PARAMETER(log_sigma_cots);                       // log SD for log(COTS)
  PARAMETER(log_sigma_fast);                       // log SD for log(fast coral)
  PARAMETER(log_sigma_slow);                       // log SD for log(slow coral)

  // Transforms to constrained spaces
  Type r_fast = exp(log_r_fast);                   // yr^-1, positive
  Type r_slow = exp(log_r_slow);                   // yr^-1, positive

  // Total coral carrying capacity in [%], bounded between Kmin and Kmax
  const Type Kmin = Type(20.0);                    // lower plausible bound for total coral cover (%)
  const Type Kmax = Type(90.0);                    // upper plausible bound for total coral cover (%)
  Type K_tot = Kmin + (Kmax - Kmin) * invlogit(kappa_tot); // (%) bounded smoothly

  // Predation parameters
  Type a0 = exp(log_a0);                           // base attack rate (yr^-1 %-1)
  Type h = exp(log_h);                             // handling time (yr %^-1)
  Type pref_fast = invlogit(logit_pref_fast);      // preference for fast (0-1)
  Type pref_slow = Type(1.0) - pref_fast;          // preference for slow (0-1)
  Type pred_scale_fast = exp(log_pred_scale_fast); // % cover loss per (COTS * consumption)
  Type pred_scale_slow = exp(log_pred_scale_slow); // % cover loss per (COTS * consumption)
  Type k_v_fast = exp(log_k_v_fast);               // slope for vulnerability sigmoid (>0), per %
  Type k_v_slow = exp(log_k_v_slow);               // slope for vulnerability sigmoid (>0), per %
  // Map thresholds (0-100%) using invlogit on a [0,1] fraction then scale by 100
  Type thr_v_fast = Type(100.0) * invlogit(logit_thr_v_fast); // threshold % for fast vulnerability
  Type thr_v_slow = Type(100.0) * invlogit(logit_thr_v_slow); // threshold % for slow vulnerability

  // Reproduction and immigration
  Type fec_eff = exp(log_fec_eff);                 // yr^-1, positive
  Type alpha_food = exp(log_alpha_food);           // %-1, positive
  Type wrep_fast = invlogit(logit_wrep_fast);      // weight in [0,1]
  Type wrep_slow = Type(1.0) - wrep_fast;          // complement weight
  Type imm_eff = exp(log_imm_eff);                 // indiv m^-2 per unit cotsimm_dat

  // Temperature effects
  Type sst_sd_fast = exp(log_sst_sd_fast) + eps;   // °C, positive
  Type sst_sd_slow = exp(log_sst_sd_slow) + eps;   // °C, positive
  Type sst_sd_C = exp(log_sst_sd_C) + eps;         // °C, positive

  // Mortality and density dependence
  Type m0 = invlogit(logit_m0);                    // baseline annual mortality probability (0-1)
  Type betaC = exp(log_betaC);                     // m^2 indiv^-1, positive

  // Outbreak trigger parameters
  Type k_tr = exp(log_k_tr);                       // per %, positive
  Type coral_thr = Type(100.0) * invlogit(logit_coral_thr); // threshold coral %, [0,100]

  // Initial states (positive)
  Type C0 = exp(log_C0);                           // indiv m^-2
  Type F0 = exp(log_F0);                           // %
  Type S0 = exp(log_S0);                           // %

  // Observation SDs (with floors)
  const Type sigma_min = Type(0.05);               // minimum SD on log-scale to stabilize likelihood
  Type sigma_cots = exp(log_sigma_cots) + sigma_min; // log SD for COTS
  Type sigma_fast = exp(log_sigma_fast) + sigma_min; // log SD for fast coral
  Type sigma_slow = exp(log_sigma_slow) + sigma_min; // log SD for slow coral

  // State vectors for predictions (names avoid '_dat' to prevent leakage flags)
  vector<Type> cots_pred(T);                       // Predicted adult COTS (indiv m^-2) for cots_dat
  vector<Type> fast_pred(T);                       // Predicted fast-growing coral cover (%) for fast_dat
  vector<Type> slow_pred(T);                       // Predicted slow-growing coral cover (%) for slow_dat

  // Initialize states at t=0 with parameters (no data leakage)
  cots_pred(0) = C0;                               // Initial COTS abundance prediction
  fast_pred(0) = F0;                               // Initial fast coral cover prediction
  slow_pred(0) = S0;                               // Initial slow coral cover prediction

  // Process model loop
  for (int t = 1; t < T; t++) {
    // Previous states
    Type C_prev = cots_pred(t - 1);                // indiv m^-2 at t-1
    Type F_prev = fast_pred(t - 1);                // % at t-1
    Type S_prev = slow_pred(t - 1);                // % at t-1

    // Environmental drivers at previous step (avoid contemporaneous leakage)
    Type sst_prev = sst_dat(t - 1);                // °C
    Type imm_prev = cotsimm_dat(t - 1);            // indiv m^-2 yr^-1

    // (1) Temperature effect on attack rate
    Type a_eff = a0 * exp(theta_a_sst * (sst_prev - sst_ref)); // yr^-1 %-1

    // (2) Multi-prey Holling II per-capita consumption (per COTS)
    Type denom = Type(1.0) + a_eff * h * (pref_fast * F_prev + pref_slow * S_prev) + eps; // dimensionless
    Type c_fast = a_eff * pref_fast * F_prev / denom;        // % yr^-1 per COTS on fast
    Type c_slow = a_eff * pref_slow * S_prev / denom;        // % yr^-1 per COTS on slow

    // (3) Vulnerability sigmoid (low cover refuge)
    Type gamma_fast = invlogit(k_v_fast * (F_prev - thr_v_fast)); // 0-1
    Type gamma_slow = invlogit(k_v_slow * (S_prev - thr_v_slow)); // 0-1

    // (4) Predation losses (convert to % cover loss per year)
    Type loss_fast = pred_scale_fast * C_prev * c_fast * gamma_fast; // % yr^-1
    Type loss_slow = pred_scale_slow * C_prev * c_slow * gamma_slow; // % yr^-1

    // (5) Thermal modulation for coral growth
    Type g_fast = exp(-Type(0.5) * square((sst_prev - sst_opt_fast) / (sst_sd_fast + eps))); // 0-1
    Type g_slow = exp(-Type(0.5) * square((sst_prev - sst_opt_slow) / (sst_sd_slow + eps))); // 0-1
    Type space_lim = (Type(1.0) - (F_prev + S_prev) / (K_tot + eps)); // dimensionless
    // Smoothly limit by space: allow negative feedback if sum exceeds K_tot
    Type Growth_fast = r_fast * F_prev * space_lim * g_fast;  // % yr^-1
    Type Growth_slow = r_slow * S_prev * space_lim * g_slow;  // % yr^-1

    // (6) Coral updates
    Type F_next_raw = F_prev + Growth_fast - loss_fast;       // % next
    Type S_next_raw = S_prev + Growth_slow - loss_slow;       // % next

    // Soft positivity to avoid negative covers (smooth, no hard cutoffs)
    Type F_next = softplus(F_next_raw) + tiny;                 // % next (>=0)
    Type S_next = softplus(S_next_raw) + tiny;                 // % next (>=0)

    // (7) Food index and saturation for COTS reproduction
    Type food_index = wrep_fast * F_prev + wrep_slow * S_prev; // %, weighted
    Type s_food = (alpha_food * food_index) / (Type(1.0) + alpha_food * food_index + eps); // 0-1

    // (8) Outbreak trigger based on total coral cover (smooth threshold)
    Type trig = invlogit(k_tr * ((F_prev + S_prev) - coral_thr)); // 0-1

    // (9) Thermal modulation for COTS reproduction and immigration
    Type g_C = exp(-Type(0.5) * square((sst_prev - sst_opt_C) / (sst_sd_C + eps))); // 0-1
    Type R = fec_eff * C_prev * s_food * trig * g_C + imm_eff * imm_prev * g_C;      // indiv m^-2 yr^-1

    // (10) Temperature-modified mortality, bounded < 1
    Type mort = Type(0.99) * invlogit(logit_m0 + m_temp * (sst_prev - sst_opt_C));   // 0-0.99

    // (11) Beverton–Holt density dependence
    auto BH = [&](Type x) { return x / (Type(1.0) + betaC * x + eps); }; // indiv m^-2

    // (12) COTS update
    Type C_post = C_prev * (Type(1.0) - mort) + R;               // indiv m^-2
    Type C_next = BH(C_post) + tiny;                             // indiv m^-2 (>=0)

    // Assign next states
    cots_pred(t) = C_next;                                       // indiv m^-2
    fast_pred(t) = F_next;                                       // %
    slow_pred(t) = S_next;                                       // %
  }

  // (13) Observation likelihood: lognormal with sigma floors
  for (int t = 0; t < T; t++) {
    // COTS
    Type yC = log(cots_dat(t) + eps);                            // log observed COTS
    Type muC = log(cots_pred(t) + eps);                          // log predicted COTS
    nll -= dnorm(yC, muC, sigma_cots, true);                     // add log-likelihood

    // Fast coral
    Type yF = log(fast_dat(t) + eps);                            // log observed fast coral (%)
    Type muF = log(fast_pred(t) + eps);                          // log predicted fast coral (%)
    nll -= dnorm(yF, muF, sigma_fast, true);                     // add log-likelihood

    // Slow coral
    Type yS = log(slow_dat(t) + eps);                            // log observed slow coral (%)
    Type muS = log(slow_pred(t) + eps);                          // log predicted slow coral (%)
    nll -= dnorm(yS, muS, sigma_slow, true);                     // add log-likelihood
  }

  // Weakly-informative priors (smooth penalties) to stabilize estimation (units in log or logit space)
  nll -= dnorm(log_r_fast, log(Type(0.5)), Type(1.0), true);     // prior for fast growth
  nll -= dnorm(log_r_slow, log(Type(0.2)), Type(1.0), true);     // prior for slow growth
  nll -= dnorm(log_a0, log(Type(0.05)), Type(1.0), true);        // prior for attack rate
  nll -= dnorm(log_h, log(Type(0.2)), Type(1.0), true);          // prior for handling time
  nll -= dnorm(log_pred_scale_fast, log(Type(0.02)), Type(2.0), true); // prior pred scale fast
  nll -= dnorm(log_pred_scale_slow, log(Type(0.01)), Type(2.0), true); // prior pred scale slow
  nll -= dnorm(theta_a_sst, Type(0.0), Type(0.2), true);         // small slope around zero
  nll -= dnorm(log_betaC, log(Type(0.3)), Type(1.0), true);      // prior for density dependence
  nll -= dnorm(log_sigma_cots, log(Type(0.3)), Type(1.0), true); // prior obs SDs
  nll -= dnorm(log_sigma_fast, log(Type(0.2)), Type(1.0), true);
  nll -= dnorm(log_sigma_slow, log(Type(0.2)), Type(1.0), true);

  // REPORT predicted states and key derived quantities (for diagnostics)
  REPORT(cots_pred);                                             // indiv m^-2
  REPORT(fast_pred);                                             // %
  REPORT(slow_pred);                                             // %
  REPORT(K_tot);                                                 // %
  REPORT(pref_fast);                                             // unitless
  REPORT(wrep_fast);                                             // unitless
  REPORT(betaC);                                                 // m^2 indiv^-1
  REPORT(m0);                                                    // unitless
  REPORT(sigma_cots);                                            // log SD
  REPORT(sigma_fast);                                            // log SD
  REPORT(sigma_slow);                                            // log SD

  return nll;                                                    // Return total negative log-likelihood
}
