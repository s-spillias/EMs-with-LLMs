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

  // Predation functional response and preference
  PARAMETER(log_attack_max);     // log(proportion coral yr^-1 per predator); max per-predator consumption rate
  PARAMETER(logit_h_type3);      // logit(proportion); half-saturation (proportion) for type III response (on edible coral index)
  PARAMETER(tau_pref_fast);      // unitless; logit preference for fast coral in diet allocation; >0 favors fast coral

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

  // Outbreak-associated density-dependent mortality (extended)
  PARAMETER(log_m_disease);      // log(year^-1); scale of additional mortality that activates during outbreaks
  PARAMETER(N_burst_thr);        // individuals m^-2; adult density threshold for outbreak mortality onset
  PARAMETER(k_burst);            // (m^2 individuals^-1); steepness of outbreak mortality activation
  PARAMETER(E_burst_thr);        // proportion (0-1); edible coral threshold for starvation/disease activation
  PARAMETER(k_burst_E);          // unitless; steepness of starvation activation with respect to low edible coral

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
  Type m_disease = exp(log_m_disease);                         // year^-1; max additional outbreak mortality
  Type E_burst_threshold = CppAD::CondExpLt(E_burst_thr, Type(0.0), Type(0.0), CppAD::CondExpGt(E_burst_thr, Type(1.0), Type(1.0), E_burst_thr)); // clamp [0,1]
  Type sigma_cots = exp(log_sigma_cots) + min_sigma;           // observation SD on log-scale
  Type phi_fast = exp(log_phi_fast) + min_phi;                 // Beta precision fast
  Type phi_slow = exp(log_phi_slow) + min_phi;                 // Beta precision slow

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
  vector<Type> cons_total_pred(T);        // proportion yr^-1; coral consumption per area (realized; all COTS)
  vector<Type> cons_fast_pred(T);         // proportion yr^-1; consumption allocated to fast coral (realized)
  vector<Type> cons_slow_pred(T);         // proportion yr^-1; consumption allocated to slow coral (realized)
  vector<Type> burst_mort_mod_pred(T);    // unitless [0,1]; outbreak mortality activation modifier (density)
  vector<Type> starve_mort_mod_pred(T);   // unitless [0,1]; starvation mortality activation modifier (low edible coral)

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
    // Temperature modifiers (Gaussian performance curves)
    Type tm_cots0 = exp(-Type(0.5) * pow((sst0 - sst_opt_cots) / sst_sd_cots, 2.0));
    Type tm_fast0 = exp(-Type(0.5) * pow((sst0 - sst_opt_fast) / sst_sd_fast, 2.0));
    Type tm_slow0 = exp(-Type(0.5) * pow((sst0 - sst_opt_slow) / sst_sd_slow, 2.0));
    temp_mod_cots_pred(0) = tm_cots0;
    temp_mod_fast_pred(0) = tm_fast0;
    temp_mod_slow_pred(0) = tm_slow0;

    // Edible coral index
    Type E0 = pref_fast * F_prev + (Type(1.0) - pref_fast) * S_prev;
    edible_index_pred(0) = E0;

    // Functional response (Type III) and realized consumption at t=0 (diagnostic only)
    Type fr_type3_0 = pow(E0, 2.0) / (pow(h_type3, 2.0) + pow(E0, 2.0) + eps);
    Type raw_cons0 = attack_max * N_prev * fr_type3_0 * tm_cots0; // proportion yr^-1
    Type ef0 = pref_fast * F_prev;
    Type es0 = (Type(1.0) - pref_fast) * S_prev;
    Type den0 = ef0 + es0 + eps;
    Type share_fast0 = ef0 / den0;
    Type share_slow0 = es0 / den0;
    Type cons_fast0 = raw_cons0 * share_fast0;
    Type cons_slow0 = raw_cons0 * share_slow0;
    // Realized (limited by availability)
    Type cons_fast_real0 = CppAD::CondExpGt(cons_fast0, F_prev, F_prev, cons_fast0);
    Type cons_slow_real0 = CppAD::CondExpGt(cons_slow0, S_prev, S_prev, cons_slow0);
    cons_fast_pred(0) = cons_fast_real0;
    cons_slow_pred(0) = cons_slow_real0;
    cons_total_pred(0) = cons_fast_real0 + cons_slow_real0;

    // Outbreak/starvation mortality modifiers at t=0
    Type f_burst_N0 = inv_logit_safe(k_burst * (N_prev - N_burst_thr));
    Type f_starve_E0 = inv_logit_safe(k_burst_E * (E_burst_threshold - E0));
    burst_mort_mod_pred(0) = f_burst_N0;
    starve_mort_mod_pred(0) = f_starve_E0;
  }

  // ========================
  // STATE TRANSITIONS
  // ========================
  for (int t = 1; t < T; t++) {
    // Use previous-step states for process dynamics (no data leakage)
    Type C_prev = F_prev + S_prev;                         // total coral proportion
    Type E_prev = pref_fast * F_prev + (Type(1.0) - pref_fast) * S_prev; // edible coral index (proportion)

    // Temperature modifiers based on environmental forcing (use current year's SST)
    Type sst_t = sst_dat(t);
    Type tm_cots = exp(-Type(0.5) * pow((sst_t - sst_opt_cots) / sst_sd_cots, 2.0));
    Type tm_fast = exp(-Type(0.5) * pow((sst_t - sst_opt_fast) / sst_sd_fast, 2.0));
    Type tm_slow = exp(-Type(0.5) * pow((sst_t - sst_opt_slow) / sst_sd_slow, 2.0));
    temp_mod_cots_pred(t) = tm_cots;
    temp_mod_fast_pred(t) = tm_fast;
    temp_mod_slow_pred(t) = tm_slow;

    // Food limitation for COTS (Michaelis–Menten) with smooth low-food gate
    Type food_mm = E_prev / (Hh_food + E_prev + eps);
    Type food_gate = inv_logit_safe(k_E * (E_prev - E_threshold));
    Type food_mod = food_mm * food_gate;

    // COTS carrying capacity as function of edible coral
    Type Kcots = Kcots0 + Kcots1 * E_prev;

    // Outbreak-associated additional mortality, gated by high density AND low edible coral
    Type f_burst_N = inv_logit_safe(k_burst * (N_prev - N_burst_thr));
    Type f_starve_E = inv_logit_safe(k_burst_E * (E_burst_threshold - E_prev));
    burst_mort_mod_pred(t) = f_burst_N;
    starve_mort_mod_pred(t) = f_starve_E;
    Type m_eff = m_cots + m_disease * f_burst_N * f_starve_E;

    // Immigration-triggered recruitment (smooth threshold on immigration rate)
    Type imm_t = cotsimm_dat(t);
    Type imm_gate = inv_logit_safe(imm_k * (imm_t - imm_thr));
    Type R_imm = alpha_imm * imm_t * imm_gate; // adults m^-2 yr^-1

    // COTS population update (logistic growth - mortality + immigration), Euler step
    Type g_cots = r_cots_max * food_mod * tm_cots;
    Type growth_term = g_cots * N_prev * (Type(1.0) - N_prev / (Kcots + eps));
    Type dN = growth_term - m_eff * N_prev + R_imm;
    Type N_curr = smooth_pos(N_prev + dN, delta);

    // COTS-driven coral consumption (Type III functional response on edible coral)
    Type fr_type3 = pow(E_prev, 2.0) / (pow(h_type3, 2.0) + pow(E_prev, 2.0) + eps);
    Type raw_cons = attack_max * N_prev * fr_type3 * tm_cots; // proportion yr^-1

    // Diet allocation with availability weighting
    Type ef = pref_fast * F_prev;
    Type es = (Type(1.0) - pref_fast) * S_prev;
    Type den = ef + es + eps;
    Type share_fast = ef / den;
    Type share_slow = es / den;
    Type cons_fast = raw_cons * share_fast;
    Type cons_slow = raw_cons * share_slow;

    // Realized consumption limited by available cover
    Type cons_fast_real = CppAD::CondExpGt(cons_fast, F_prev, F_prev, cons_fast);
    Type cons_slow_real = CppAD::CondExpGt(cons_slow, S_prev, S_prev, cons_slow);
    cons_fast_pred(t) = cons_fast_real;
    cons_slow_pred(t) = cons_slow_real;
    cons_total_pred(t) = cons_fast_real + cons_slow_real;

    // Coral growth with competition on total cover and temperature modifiers
    Type dF_grow = r_fast * tm_fast * F_prev * (Type(1.0) - C_prev / (K_c + eps));
    Type dS_grow = r_slow * tm_slow * S_prev * (Type(1.0) - C_prev / (K_c + eps));

    // Background (non-COTS) mortality
    Type dF_bg_mort = m_fast * F_prev;
    Type dS_bg_mort = m_slow * S_prev;

    // Update coral states, enforce non-negativity
    Type F_temp = smooth_pos(F_prev + dF_grow - dF_bg_mort - cons_fast_real, delta);
    Type S_temp = smooth_pos(S_prev + dS_grow - dS_bg_mort - cons_slow_real, delta);

    // Optional proportional rescaling if total exceeds carrying capacity
    Type C_temp = F_temp + S_temp;
    Type over = C_temp - K_c;
    Type scale = CppAD::CondExpGt(over, Type(0.0), K_c / (C_temp + eps), Type(1.0));
    Type F_curr = F_temp * scale;
    Type S_curr = S_temp * scale;

    // Save edible index at t (post-update, diagnostic)
    edible_index_pred(t) = pref_fast * F_curr + (Type(1.0) - pref_fast) * S_curr;

    // Save predictions in original units
    cots_pred(t) = N_curr;
    fast_pred(t) = F_curr * Type(100.0);
    slow_pred(t) = S_curr * Type(100.0);

    // Advance state
    N_prev = N_curr;
    F_prev = F_curr;
    S_prev = S_curr;
  }

  // ========================
  // OBSERVATION MODEL
  // ========================
  for (int t = 0; t < T; t++) {
    // COTS: lognormal observation error
    Type yN = log(cots_dat(t) + eps);
    Type muN = log(cots_pred(t) + eps);
    nll -= dnorm(yN, muN, sigma_cots, true);

    // Coral: Beta observation model on proportions
    Type yF = fast_dat(t) / Type(100.0);
    Type yS = slow_dat(t) / Type(100.0);
    // Predicted proportions
    Type pF = fast_pred(t) / Type(100.0);
    Type pS = slow_pred(t) / Type(100.0);
    // Clamp to (eps, 1 - eps)
    yF = CppAD::CondExpLt(yF, eps, eps, CppAD::CondExpGt(yF, max_prop, max_prop, yF));
    yS = CppAD::CondExpLt(yS, eps, eps, CppAD::CondExpGt(yS, max_prop, max_prop, yS));
    pF = CppAD::CondExpLt(pF, eps, eps, CppAD::CondExpGt(pF, max_prop, max_prop, pF));
    pS = CppAD::CondExpLt(pS, eps, eps, CppAD::CondExpGt(pS, max_prop, max_prop, pS));

    Type aF = pF * phi_fast;
    Type bF = (Type(1.0) - pF) * phi_fast;
    Type aS = pS * phi_slow;
    Type bS = (Type(1.0) - pS) * phi_slow;

    nll -= dbeta(yF, aF, bF, true);
    nll -= dbeta(yS, aS, bS, true);
  }

  // ========================
  // REPORTS
  // ========================
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(edible_index_pred);
  REPORT(temp_mod_cots_pred);
  REPORT(temp_mod_fast_pred);
  REPORT(temp_mod_slow_pred);
  REPORT(cons_total_pred);
  REPORT(cons_fast_pred);
  REPORT(cons_slow_pred);
  REPORT(burst_mort_mod_pred);
  REPORT(starve_mort_mod_pred);

  return nll;
}
