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
  DATA_VECTOR(fast_dat);         // Fast coral cover (Acropora) (%) of substrate; response variable
  DATA_VECTOR(slow_dat);         // Slow coral cover (Faviidae+Porites) (%) of substrate; response variable

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
  PARAMETER(log_A_cots);         // log(individuals m^-2); Allee threshold scale for mate limitation on adult per-capita growth

  // Predation functional response and preference
  PARAMETER(log_attack_max);     // log(proportion coral yr^-1 per predator); max per-predator consumption rate
  PARAMETER(logit_h_type3);      // logit(proportion); half-saturation (proportion) for type III response (on edible coral index)
  PARAMETER(tau_pref_fast);      // unitless; logit preference for fast coral in diet allocation; >0 favors fast coral
  // New: thermal dependence of feeding
  PARAMETER(sst_opt_feed);       // °C; thermal optimum for COTS feeding rate
  PARAMETER(log_sst_sd_feed);    // log(°C); thermal breadth (SD) for feeding rate

  // Coral vital rates and carrying capacity
  PARAMETER(log_r_fast);         // log(year^-1); intrinsic growth of fast coral
  PARAMETER(log_r_slow);         // log(year^-1); intrinsic growth of slow coral
  PARAMETER(logit_K_c);          // logit(proportion); total coral carrying capacity (F+S), as fraction of substrate
  PARAMETER(log_m_fast);         // log(year^-1); background mortality (non-COTS) of fast coral
  PARAMETER(log_m_slow);         // log(year^-1); background mortality (non-COTS) of slow coral
  PARAMETER(sst_opt_fast);       // °C; thermal optimum for fast coral growth
  PARAMETER(log_sst_sd_fast);    // log(°C); thermal breadth for fast coral
  PARAMETER(sst_opt_slow);       // °C; thermal optimum for slow coral growth
  PARAMETER(log_sst_sd_slow);    // log(°C); thermal breadth for slow coral

  // Observation model parameters
  PARAMETER(log_sigma_cots);     // log; observation SD on log(COTS), lognormal error
  PARAMETER(log_phi_fast);       // log; Beta precision (fast coral) for proportion data
  PARAMETER(log_phi_slow);       // log; Beta precision (slow coral) for proportion data

  // ========================
  // Transform parameters
  // ========================
  Type r_cots_max = exp(log_r_cots_max);                       // year^-1
  Type m_cots = exp(log_m_cots);                               // year^-1
  Type alpha_imm = exp(log_alpha_imm);                         // (adults m^-2)/(immigrants m^-2 yr^-1)
  Type Hh_food = inv_logit_safe(logit_Hh_food);                // proportion [0,1]
  Type E_threshold = CppAD::CondExpLt(E_thr, Type(0.0), Type(0.0), CppAD::CondExpGt(E_thr, Type(1.0), Type(1.0), E_thr)); // softly clamp to [0,1] in transform
  Type sst_sd_cots = exp(log_sst_sd_cots) + Type(1e-6);        // °C
  Type Kcots0 = exp(log_Kcots0);                               // individuals m^-2
  Type Kcots1 = exp(log_Kcots1);                               // individuals m^-2 per proportion
  Type A_cots = exp(log_A_cots);                               // individuals m^-2; Allee scale
  Type attack_max = exp(log_attack_max);                       // proportion coral yr^-1 per predator
  Type h_type3 = inv_logit_safe(logit_h_type3);                // proportion [0,1]
  Type pref_fast = inv_logit_safe(tau_pref_fast);              // [0,1]; diet preference weight on fast coral
  Type r_fast = exp(log_r_fast);                               // year^-1
  Type r_slow = exp(log_r_slow);                               // year^-1
  Type K_c = inv_logit_safe(logit_K_c) * Type(0.98);           // proportion, keep below 1 smoothly
  Type m_fast = exp(log_m_fast);                               // year^-1
  Type m_slow = exp(log_m_slow);                               // year^-1
  Type sst_sd_fast = exp(log_sst_sd_fast) + Type(1e-6);        // °C
  Type sst_sd_slow = exp(log_sst_sd_slow) + Type(1e-6);        // °C
  Type sigma_cots = exp(log_sigma_cots) + min_sigma;           // observation SD on log-scale
  Type phi_fast = exp(log_phi_fast) + min_phi;                 // Beta precision fast
  Type phi_slow = exp(log_phi_slow) + min_phi;                 // Beta precision slow
  // Feeding thermal breadth
  Type sst_sd_feed = exp(log_sst_sd_feed) + Type(1e-6);        // °C

  // ========================
  // STORAGE FOR PREDICTIONS
  // ========================
  vector<Type> cots_pred(T);              // individuals m^-2; predicted adult COTS
  vector<Type> fast_pred(T);              // %; predicted fast coral cover
  vector<Type> slow_pred(T);              // %; predicted slow coral cover

  // Diagnostics and intermediates (reported)
  vector<Type> edible_index_pred(T);      // proportion; edible coral index used for COTS processes
  vector<Type> temp_mod_cots_pred(T);     // unitless; thermal modifier for COTS
  vector<Type> temp_mod_fast_pred(T);     // unitless; thermal modifier for fast coral
  vector<Type> temp_mod_slow_pred(T);     // unitless; thermal modifier for slow coral
  vector<Type> temp_mod_feed_pred(T);     // unitless; thermal modifier for feeding
  vector<Type> cons_total_pred(T);        // proportion yr^-1; coral consumption per area (all predators)
  vector<Type> cons_fast_pred(T);         // proportion yr^-1; consumption allocated to fast coral
  vector<Type> cons_slow_pred(T);         // proportion yr^-1; consumption allocated to slow coral

  // ========================
  // INITIAL CONDITIONS
  // ========================
  cots_pred(0) = cots_dat(0);                   // Use observed initial adult density (ind m^-2)
  fast_pred(0) = fast_dat(0);                   // Use observed initial fast coral cover (%)
  slow_pred(0) = slow_dat(0);                   // Use observed initial slow coral cover (%)

  // Internal state in proportions for coral; adults in original units
  Type N_prev = cots_pred(0);                   // adults m^-2 at t=0 (state)
  Type F_prev = fast_pred(0) / Type(100.0);     // fast coral proportion at t=0
  Type S_prev = slow_pred(0) / Type(100.0);     // slow coral proportion at t=0

  // Initialize diagnostics at t=0 based on initial states
  {
    Type sst0 = sst_dat(0);
    Type E0 = pref_fast * F_prev + (Type(1.0) - pref_fast) * S_prev;              // edible coral index (weighted)
    Type fT_cots0 = exp(-Type(0.5) * pow((sst0 - sst_opt_cots) / sst_sd_cots, 2)); // Gaussian thermal mod
    Type fT_fast0 = exp(-Type(0.5) * pow((sst0 - sst_opt_fast) / sst_sd_fast, 2)); // fast coral thermal mod
    Type fT_slow0 = exp(-Type(0.5) * pow((sst0 - sst_opt_slow) / sst_sd_slow, 2)); // slow coral thermal mod
    Type fT_feed0 = exp(-Type(0.5) * pow((sst0 - sst_opt_feed) / sst_sd_feed, 2)); // feeding thermal mod
    Type cons_ppred0 = attack_max * fT_feed0 * (E0 * E0) / (E0 * E0 + h_type3 * h_type3 + eps); // type III per-predator consumption with temp
    Type cons_all0 = cons_ppred0 * N_prev;                                         // area-level consumption
    Type avail_fast0 = pref_fast * F_prev;
    Type avail_slow0 = (Type(1.0) - pref_fast) * S_prev;
    Type denom0 = avail_fast0 + avail_slow0 + eps;
    Type share_fast0 = avail_fast0 / denom0;
    Type share_slow0 = avail_slow0 / denom0;

    edible_index_pred(0) = E0;
    temp_mod_cots_pred(0) = fT_cots0;
    temp_mod_fast_pred(0) = fT_fast0;
    temp_mod_slow_pred(0) = fT_slow0;
    temp_mod_feed_pred(0) = fT_feed0;
    cons_total_pred(0) = cons_all0;
    cons_fast_pred(0) = cons_all0 * share_fast0;
    cons_slow_pred(0) = cons_all0 * share_slow0;
  }

  // =====================================
  // DYNAMICS (t = 1..T-1; use t-1 states)
  // =====================================
  for (int t = 1; t < T; t++) {
    // Forcing at previous step (avoid data leakage)
    Type sst = sst_dat(t - 1);                          // °C at t-1
    Type imm = cotsimm_dat(t - 1);                      // immigrants m^-2 yr^-1 at t-1

    // 1) Thermal modifiers (Gaussian performance curves)
    Type fT_cots = exp(-Type(0.5) * pow((sst - sst_opt_cots) / sst_sd_cots, 2)); // unitless [0,1]
    Type fT_fast = exp(-Type(0.5) * pow((sst - sst_opt_fast) / sst_sd_fast, 2)); // unitless [0,1]
    Type fT_slow = exp(-Type(0.5) * pow((sst - sst_opt_slow) / sst_sd_slow, 2)); // unitless [0,1]
    Type fT_feed = exp(-Type(0.5) * pow((sst - sst_opt_feed) / sst_sd_feed, 2)); // unitless [0,1]

    // 2) Edible coral index and food limitation
    Type E = pref_fast * F_prev + (Type(1.0) - pref_fast) * S_prev;               // proportion edible coral
    Type fsat = E / (E + Hh_food + eps);                                          // saturating food limitation
    Type fthr = inv_logit_safe(k_E * (E - E_threshold));                          // smooth threshold on food
    Type f_food = fsat * fthr;                                                    // combined food limitation modifier

    // 3) Immigration trigger and conversion
    Type f_imm = inv_logit_safe(imm_k * (imm - imm_thr));                         // smooth threshold on immigration
    Type R_imm = alpha_imm * imm * f_imm * fT_cots;                                // recruits to adults per area (yr^-1)

    // 4) COTS density dependence via resource-modified capacity and Allee effect
    Type K_cots = Kcots0 + Kcots1 * E;                                            // individuals m^-2; increases with edible coral
    Type f_allee = N_prev / (N_prev + A_cots + eps);                              // unitless [0,1]; mate limitation
    Type percap_growth = r_cots_max * f_food * fT_cots * f_allee;                 // year^-1; resource, temp, and Allee modified
    Type crowding = (Type(1.0) - N_prev / (K_cots + eps));                        // unitless; logistic crowding term

    // 5) Update adult COTS abundance (Euler step; smooth positivity)
    Type N_tmp = N_prev + N_prev * (percap_growth * crowding - m_cots) + R_imm;   // individuals m^-2
    Type N_new = smooth_pos(N_tmp, delta) + eps;                                   // enforce positivity smoothly

    // 6) Predation functional response (Type III) and allocation
    Type cons_per_pred = attack_max * fT_feed * (E * E) / (E * E + h_type3 * h_type3 + eps); // per-predator consumption (prop coral yr^-1)
    Type cons_all = cons_per_pred * N_prev;                                       // total consumption pressure (prop yr^-1)
    Type avail_fast = pref_fast * F_prev;
    Type avail_slow = (Type(1.0) - pref_fast) * S_prev;
    Type denom = avail_fast + avail_slow + eps;
    Type share_fast = avail_fast / denom;                                         // diet share to fast coral
    Type share_slow = avail_slow / denom;                                         // diet share to slow coral
    Type L_fast = cons_all * share_fast;                                          // loss (prop yr^-1) of fast coral
    Type L_slow = cons_all * share_slow;                                          // loss (prop yr^-1) of slow coral

    // 7) Coral growth with competition for space and background mortality
    Type total_coral = F_prev + S_prev;                                           // proportion
    Type growth_fast = r_fast * F_prev * (Type(1.0) - total_coral / (K_c + eps)) * fT_fast; // fast coral growth
    Type growth_slow = r_slow * S_prev * (Type(1.0) - total_coral / (K_c + eps)) * fT_slow; // slow coral growth

    Type F_tmp = F_prev + growth_fast - L_fast - m_fast * F_prev;                 // unbounded update
    Type S_tmp = S_prev + growth_slow - L_slow - m_slow * S_prev;                 // unbounded update

    // Smoothly enforce non-negativity (allow model to pay penalty only via dynamics)
    Type F_new = smooth_pos(F_tmp, delta);                                        // proportion >= 0
    Type S_new = smooth_pos(S_tmp, delta);                                        // proportion >= 0

    // 8) Update predictions (convert coral back to % for reporting/likelihood)
    cots_pred(t) = N_new;                                                         // individuals m^-2
    fast_pred(t) = CppAD::CondExpGt(F_new * Type(100.0), Type(100.0), Type(100.0), F_new * Type(100.0)); // %
    slow_pred(t) = CppAD::CondExpGt(S_new * Type(100.0), Type(100.0), Type(100.0), S_new * Type(100.0)); // %

    // Store diagnostics
    edible_index_pred(t) = E;
    temp_mod_cots_pred(t) = fT_cots;
    temp_mod_fast_pred(t) = fT_fast;
    temp_mod_slow_pred(t) = fT_slow;
    temp_mod_feed_pred(t) = fT_feed;
    cons_total_pred(t) = cons_all;
    cons_fast_pred(t) = L_fast;
    cons_slow_pred(t) = L_slow;

    // Advance state (use internal units)
    N_prev = N_new;
    F_prev = CppAD::CondExpGt(F_new, K_c, K_c, F_new);                            // softly cap by Kc in next step
    S_prev = CppAD::CondExpGt(S_new, K_c, K_c, S_new);
  }

  // =====================================
  // LIKELIHOOD
  // =====================================
  // COTS: lognormal observation model
  for (int t = 0; t < T; t++) {
    Type y = cots_dat(t) + eps;                       // observed adults m^-2
    Type mu = cots_pred(t) + eps;                     // predicted adults m^-2
    nll -= dnorm(log(y), log(mu), sigma_cots, true);  // lognormal kernel
    nll += log(y);                                     // Jacobian term for log-transform
  }

  // Coral: Beta observation model on proportions (0,1)
  for (int t = 0; t < T; t++) {
    // Fast coral
    Type yF = (fast_dat(t) / Type(100.0));                                      // observed proportion
    Type muF = (fast_pred(t) / Type(100.0));                                    // predicted proportion
    // Clamp to open interval for Beta
    yF = CppAD::CondExpLt(yF, eps, eps, CppAD::CondExpGt(yF, max_prop, max_prop - eps, yF));
    muF = CppAD::CondExpLt(muF, eps, eps, CppAD::CondExpGt(muF, max_prop, max_prop - eps, muF));
    Type aF = muF * phi_fast + eps;
    Type bF = (Type(1.0) - muF) * phi_fast + eps;
    nll -= dbeta(yF, aF, bF, true);

    // Slow coral
    Type yS = (slow_dat(t) / Type(100.0));                                      // observed proportion
    Type muS = (slow_pred(t) / Type(100.0));                                    // predicted proportion
    yS = CppAD::CondExpLt(yS, eps, eps, CppAD::CondExpGt(yS, max_prop, max_prop - eps, yS));
    muS = CppAD::CondExpLt(muS, eps, eps, CppAD::CondExpGt(muS, max_prop, max_prop - eps, muS));
    Type aS = muS * phi_slow + eps;
    Type bS = (Type(1.0) - muS) * phi_slow + eps;
    nll -= dbeta(yS, aS, bS, true);
  }

  // =====================================
  // Soft regularization (priors/penalties) to keep parameters biologically plausible
  // These act as smooth penalties, not hard bounds.
  // =====================================
  // Example weakly-informative priors:
  nll -= dnorm(log_r_cots_max, Type(log(1.0)), Type(1.0), true);     // r_cots_max ~ LogNormal(meanlog=0, sdlog=1)
  nll -= dnorm(log_m_cots, Type(log(0.5)), Type(1.0), true);         // m_cots ~ LogNormal(ln 0.5, 1)
  nll -= dnorm(log_r_fast, Type(log(0.3)), Type(1.0), true);         // r_fast ~ LogNormal(ln 0.3, 1)
  nll -= dnorm(log_r_slow, Type(log(0.15)), Type(1.0), true);        // r_slow ~ LogNormal(ln 0.15, 1)
  nll -= dnorm(logit_K_c, Type(0.0), Type(2.0), true);               // Kc centered ~0.5 on logit scale (broad)
  nll -= dnorm(log_A_cots, Type(log(0.2)), Type(1.0), true);         // Allee scale centered near 0.2 ind m^-2 (broad)
  // New weak priors for feeding temperature response
  nll -= dnorm(sst_opt_feed, Type(29.0), Type(1.5), true);           // feeding optimum around 29°C, broad
  nll -= dnorm(log_sst_sd_feed, Type(log(1.5)), Type(1.0), true);    // breadth centered near 1.5°C on log scale, broad

  // =====================================
  // REPORTING
  // =====================================
  REPORT(cots_pred);            // predicted adults (ind m^-2)
  REPORT(fast_pred);            // predicted fast coral (%)
  REPORT(slow_pred);            // predicted slow coral (%)

  // Diagnostics
  REPORT(edible_index_pred);    // edible coral index (proportion)
  REPORT(temp_mod_cots_pred);   // thermal modifier for COTS
  REPORT(temp_mod_fast_pred);   // thermal modifier for fast coral
  REPORT(temp_mod_slow_pred);   // thermal modifier for slow coral
  REPORT(temp_mod_feed_pred);   // thermal modifier for feeding
  REPORT(cons_total_pred);      // total coral consumption pressure
  REPORT(cons_fast_pred);       // fast coral consumption component
  REPORT(cons_slow_pred);       // slow coral consumption component

  return nll;
}

/*
Model equations (annual time step; all quantities at t depend only on states/forcings at t-1):

1) Thermal modifiers (Gaussian performance curves)
   f_T,x(t-1) = exp( -0.5 * ((SST(t-1) - T_opt,x) / sigma_T,x)^2 ), for x ∈ {COTS, fast, slow, feed}.

2) Edible coral index (proportion)
   E(t-1) = p_fast * F(t-1) + (1 - p_fast) * S(t-1).

3) Food limitation for COTS (two-part)
   f_food(t-1) = [ E / (E + Hh_food) ] * inv_logit( k_E * (E - E_thr) ).

4) Immigration trigger and conversion
   f_imm(t-1) = inv_logit( k_imm * (Imm(t-1) - Imm_thr) ),
   R_imm(t-1) = alpha_imm * Imm(t-1) * f_imm(t-1) * f_T,COTS(t-1).

5) Resource-modified carrying capacity for COTS and Allee effect
   K_COTS(t-1) = Kcots0 + Kcots1 * E(t-1).
   f_allee(t-1) = N(t-1) / ( N(t-1) + A_cots ).

6) COTS population update (Euler, logistic crowding, Allee, smooth positivity)
   N(t) = pos( N(t-1) + N(t-1) * [ r_max * f_food * f_T,COTS * f_allee * (1 - N(t-1)/K_COTS) - m_COTS ] + R_imm(t-1) ).

7) Predation functional response (Type III) with thermal feeding and allocation to coral groups
   c_ppred(t-1) = a_max * f_T,feed(t-1) * E^2 / (E^2 + h_type3^2),
   C_all(t-1) = N(t-1) * c_ppred(t-1),
   share_fast = (p_fast * F) / (p_fast * F + (1 - p_fast) * S),
   L_fast = C_all * share_fast,  L_slow = C_all * (1 - share_fast).

8) Coral dynamics with logistic competition and background mortality
   dF = r_fast * F * (1 - (F+S)/K_c) * f_T,fast - L_fast - m_fast * F,
   dS = r_slow * S * (1 - (F+S)/K_c) * f_T,slow - L_slow - m_slow * S,
   F(t) = pos(F(t-1) + dF), S(t) = pos(S(t-1) + dS).

Observation models:
- COTS: y_COTS ~ Lognormal(meanlog = log(N_pred), sd = sigma_cots).
- Corals: y_fast_prop ~ Beta(mu = F_pred, phi = phi_fast); y_slow_prop ~ Beta(mu = S_pred, phi = phi_slow).

Initial conditions:
- N_pred(0) = cots_dat(0); F_pred(0) = fast_dat(0)/100; S_pred(0) = slow_dat(0)/100.

All '_pred' variables are aligned with data names and reported via REPORT().
*/
