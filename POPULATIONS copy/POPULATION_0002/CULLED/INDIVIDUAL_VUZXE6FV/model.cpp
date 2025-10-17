#include <TMB.hpp>

// Utility: smooth non-negative floor (approx max(0,x)) to avoid hard cutoff
template<class Type>
Type smooth_pos(Type x, Type eps) {
  // 0.5 * (x + sqrt(x^2 + eps)) is C1-smooth and >= 0
  return Type(0.5) * (x + sqrt(x * x + eps));
}

// Utility: saturating (Hill-type) function: x^q / (K^q + x^q)
template<class Type>
Type sat_hill(Type x, Type K, Type q, Type eps) {
  Type xq = pow(x + eps, q);
  Type Kq = pow(K + eps, q);
  return xq / (Kq + xq + eps);
}

// Utility: Gaussian-shaped performance curve around an optimum
template<class Type>
Type gauss_perf(Type x, Type mu, Type sigma, Type eps) {
  // Guard very small sd using AD-safe conditional
  sigma = CppAD::CondExpLt(sigma, Type(1e-6), Type(1e-6), sigma);
  Type z = (x - mu) / (sigma + eps);
  return exp(Type(-0.5) * z * z);
}

// Utility: numerically stable softplus for AD Types
template<class Type>
Type softplus(Type x) {
  // For large x, softplus(x) ~ x; for smaller x, use log(1) + exp(x))
  return CppAD::CondExpGt(x, Type(20.0), x, log(Type(1) + exp(x)));
}

// Utility: Soft box penalty to keep parameters within plausible [L,U] without hard constraints
template<class Type>
Type soft_box_penalty(Type x, Type L, Type U, Type k, Type w) {
  // Larger k makes steeper walls; w scales the penalty contribution to nll
  Type penL = softplus(-k * (x - L));                                      // penalize x << L
  Type penU = softplus(-k * (U - x));                                      // penalize x >> U
  return w * (penL + penU);
}

// Utility: AD-safe clamp to open unit interval (eps, 1 - eps)
template<class Type>
Type clamp_open01(Type p, Type eps) {
  p = CppAD::CondExpLt(p, eps, eps, p);
  p = CppAD::CondExpGt(p, Type(1) - eps, Type(1) - eps, p);
  return p;
}

/*
EQUATION OVERVIEW (all annual, t = Year index):
1) Coral growth (fast/slow):
   F_{t+1} = F_t + r_F(T_t) F_t [1 - (F_t + S_t)/K_tot] - m_F F_t - M_bleach(T_t) F_t - Rm_F(C_t, F_t)
   S_{t+1} = S_t + r_S(T_t) S_t [1 - (F_t + S_t)/K_tot] - m_S S_t - M_bleach(T_t) S_t - Rm_S(C_t, S_t)
   where r_g(T) is a Gaussian thermal performance modifier, M_bleach(T) is a smooth logistic bleaching mortality,
   and Rm_g is COTS predation with Type-II/III saturation and a smooth cap to not exceed available coral.

2) COTS reproduction and survival with 1-year maturation delay:
   Food_t = (pref_fast * F_t + (1 - pref_fast) * S_t) / (K_food + pref_fast * F_t + (1 - pref_fast) * S_t)
   EnvLarv_t = Gaussian(SST_t; Topt_larv, Tsd_larv)
   A_gate_t = invlogit((C_t - A_thresh) / tau_A)   [smooth Allee-like gate]
   Pulse_t = 1 + A_pulse * H_t, where H_t is a rectified Hill function of positive SST anomalies
   L_t = phi * C_t * Food_t * EnvLarv_t * exp(-beta * C_t) * A_gate_t * Pulse_t + k_imm * cotsimm_dat(t)
   sA_t = exp(- (mA + mA_food * (1 - Food_t)))    [food-dependent adult survival]
   C_{t+1} = sA_t * C_t + mu_adult * L_t

3) Predation (component of Eq. 1):
   Cons_fast_raw = alpha_fast * C_t * sat_hill(F_t, K_pred_fast, q_pred)
   Rm_F = F_t * (1 - exp(-Cons_fast_raw / (F_t + eps)))   [smooth cap ≤ F_t]
   Similarly for slow coral with alpha_slow, K_pred_slow.

4) Observation models (use all observations):
   - COTS abundance (individuals/m^2): lognormal with sd floor.
   - Coral cover (percent): transform to fractions, apply logit-normal with sd floors.

Initial conditions:
   cots_pred(0), fast_pred(0), slow_pred(0) are estimated via parameters (no data leakage).
   For t≥1: use only previous-step predictions and forcing inputs to compute current predictions (no data leakage).
*/

template<class Type>
Type objective_function<Type>::operator() () {
  Type eps = Type(1e-8);                                                  // small constant to avoid division by zero
  Type nll = 0;                                                           // negative log-likelihood accumulator
  Type pen = 0;                                                           // parameter soft-penalty accumulator

  // -----------------------
  // DATA (READ-ONLY INPUTS)
  // -----------------------
  DATA_VECTOR(Year);              // Year (year): time index, used for alignment and reporting
  DATA_VECTOR(cots_dat);          // Adult COTS abundance (individuals/m^2): response variable
  DATA_VECTOR(fast_dat);          // Fast-growing coral (Acropora) cover (%): response variable
  DATA_VECTOR(slow_dat);          // Slow-growing coral (Faviidae/Porites) cover (%): response variable
  DATA_VECTOR(sst_dat);           // Sea-surface temperature (°C): environmental forcing
  DATA_VECTOR(cotsimm_dat);       // COTS larval immigration (individuals/m^2/year): exogenous forcing

  // Time-series length inferred from observations
  int n = cots_dat.size();

  // ---------------
  // PARAMETERS
  // ---------------

  // Coral intrinsic growth rates (year^-1), positive; initial estimates from ecology of Acropora vs massive corals
  PARAMETER(log_r_fast);          // log of r_F (year^-1), ensures positivity
  PARAMETER(log_r_slow);          // log of r_S (year^-1), ensures positivity

  // Coral background mortality (year^-1), positive small
  PARAMETER(log_m_fast);          // log of background mortality for fast coral (year^-1)
  PARAMETER(log_m_slow);          // log of background mortality for slow coral (year^-1)

  // Bleaching mortality parameters (apply equally to both coral groups)
  PARAMETER(log_m_bleach);        // log of maximum additional bleaching mortality rate (year^-1)
  PARAMETER(T_bleach);            // SST (°C) at which bleaching mortality inflects upward
  PARAMETER(log_tau_bleach);      // log of temperature transition width (°C) controlling smoothness

  // Predation parameters (COTS consumption on coral cover)
  PARAMETER(log_alpha_fast);      // log of max area-clearing rate on fast coral per adult (%-cover per indiv per year)
  PARAMETER(log_alpha_slow);      // log of max area-clearing rate on slow coral per adult (%-cover per indiv per year)
  PARAMETER(log_K_pred_fast);     // log of half-saturation cover for predation on fast coral (% cover)
  PARAMETER(log_K_pred_slow);     // log of half-saturation cover for predation on slow coral (% cover)

  // Food preference and maturation
  PARAMETER(pref_fast_logit);     // logit of preference for fast coral in food index (unitless, maps to [0,1])
  PARAMETER(logit_mu_adult);      // logit of fraction of larvae maturing into adults in 1 year (unitless in [0,1])

  // Adult survival and food-stress mortality components
  PARAMETER(log_mA);              // log of baseline adult mortality rate (year^-1)
  PARAMETER(log_mA_food);         // log of additional adult mortality scaling when food is scarce (year^-1)

  // Reproduction and density dependence
  PARAMETER(log_phi);             // log of fecundity scaling (larval equivalents per adult per year)
  PARAMETER(log_beta);            // log of density-dependence strength in Ricker term (per (indiv/m^2))

  // Food limitation scale
  PARAMETER(log_K_food);          // log of half-saturation for food index in fecundity (% cover)

  // Environmental effects on larvae and coral growth (thermal performance)
  PARAMETER(Topt_larv);           // Optimal SST for larval survival (°C)
  PARAMETER(log_Tsd_larv);        // log of SD of larval thermal performance (°C)
  PARAMETER(Topt_coral);          // Optimal SST for coral growth (°C)
  PARAMETER(log_Tsd_coral);       // log of SD of coral growth thermal performance (°C)

  // Reproductive gate (Allee-like) on adults
  PARAMETER(A_thresh);            // Adult density threshold for strong reproduction (indiv/m^2)
  PARAMETER(log_tau_A);           // log of smoothness (indiv/m^2) of the Allee gate

  // Immigration scaling (converts larval immigration to adult-equivalent recruits)
  PARAMETER(log_k_imm);           // log of conversion from cotsimm_dat to recruit equivalents (unitless scaling)

  // Observation error parameters (on transformed scales)
  PARAMETER(log_sd_log_cots);     // log SD for lognormal observation on COTS (log scale)
  PARAMETER(log_sd_logit_fast);   // log SD for logit-normal observation on fast coral (logit scale)
  PARAMETER(log_sd_logit_slow);   // log SD for logit-normal observation on slow coral (logit scale)

  // Scalars previously provided as data are now parameters on log scale
  PARAMETER(log_K_tot);           // log total substrate carrying capacity for combined coral cover (% cover)
  PARAMETER(log_min_sd);          // log minimum SD floor used in observation likelihoods
  PARAMETER(log_q_pred);          // log predation saturation shape (q=1 type II, q=2 type III)

  // SST anomaly pulse on larval survival
  PARAMETER(log_A_pulse);         // log amplitude of anomaly-driven pulse (dimensionless)
  PARAMETER(T_thresh_pulse);      // SST anomaly threshold (°C) for pulse activation
  PARAMETER(log_tau_pulse);       // log smoothness (°C) around threshold
  PARAMETER(log_gamma_pulse);     // log Hill exponent controlling sharpness/episodicity of pulses

  // Initial state parameters (avoid data leakage from observations)
  PARAMETER(log_cots_init);       // log initial adult COTS density (indiv/m^2)
  PARAMETER(logit_fast_init);     // logit initial fraction of K_tot for fast coral (0..1)
  PARAMETER(logit_slow_init);     // logit initial fraction of K_tot for slow coral (0..1)

  // ---------------
  // TRANSFORMED PARAMETERS AND PENALTIES
  // ---------------
  Type r_fast = exp(log_r_fast);
  Type r_slow = exp(log_r_slow);
  Type m_fast = exp(log_m_fast);
  Type m_slow = exp(log_m_slow);

  Type m_bleach = exp(log_m_bleach);
  Type tau_bleach = exp(log_tau_bleach);

  Type alpha_fast = exp(log_alpha_fast);
  Type alpha_slow = exp(log_alpha_slow);
  Type K_pred_fast = exp(log_K_pred_fast);
  Type K_pred_slow = exp(log_K_pred_slow);

  Type pref_fast = invlogit(pref_fast_logit);
  Type mu_adult = invlogit(logit_mu_adult);

  Type mA = exp(log_mA);
  Type mA_food = exp(log_mA_food);

  Type phi = exp(log_phi);
  Type beta = exp(log_beta);

  Type K_food = exp(log_K_food);

  Type Tsd_larv = exp(log_Tsd_larv);
  Type Tsd_coral = exp(log_Tsd_coral);

  Type tau_A = exp(log_tau_A);

  Type k_imm = exp(log_k_imm);

  Type sd_log_cots  = exp(log_sd_log_cots);
  Type sd_logit_fast = exp(log_sd_logit_fast);
  Type sd_logit_slow = exp(log_sd_logit_slow);

  // New transformed scalars
  Type K_tot = exp(log_K_tot);                                            // % cover
  Type min_sd = exp(log_min_sd);                                          // transformed units
  Type q_pred = exp(log_q_pred);                                          // dimensionless (>0)

  // Pulse parameters
  Type A_pulse = exp(log_A_pulse);
  Type tau_pulse = exp(log_tau_pulse);
  Type gamma_pulse = exp(log_gamma_pulse);

  // New transformed initial states
  Type cots_init = exp(log_cots_init);
  Type p_fast_init = invlogit(logit_fast_init);
  Type p_slow_init = invlogit(logit_slow_init);

  // Soft biological bounds (do not impose hard constraints)
  pen += soft_box_penalty(r_fast,       Type(0.01),   Type(1.5),   Type(5),  Type(0.05));
  pen += soft_box_penalty(r_slow,       Type(0.005),  Type(0.8),   Type(5),  Type(0.05));
  pen += soft_box_penalty(alpha_fast,   Type(0.1),    Type(30.0),  Type(5),  Type(0.05));
  pen += soft_box_penalty(alpha_slow,   Type(0.01),   Type(15.0),  Type(5),  Type(0.05));
  pen += soft_box_penalty(K_pred_fast,  Type(1.0),    Type(60.0),  Type(5),  Type(0.05));
  pen += soft_box_penalty(K_pred_slow,  Type(1.0),    Type(60.0),  Type(5),  Type(0.05));
  pen += soft_box_penalty(mu_adult,     Type(0.05),   Type(0.8),   Type(10), Type(0.05));
  pen += soft_box_penalty(mA,           Type(0.05),   Type(2.0),   Type(5),  Type(0.05));
  pen += soft_box_penalty(mA_food,      Type(0.01),   Type(3.0),   Type(5),  Type(0.05));
  pen += soft_box_penalty(phi,          Type(0.1),    Type(20.0),  Type(5),  Type(0.05));
  pen += soft_box_penalty(beta,         Type(0.0),    Type(5.0),   Type(5),  Type(0.05));
  pen += soft_box_penalty(K_food,       Type(1.0),    Type(100.0), Type(5),  Type(0.05));
  pen += soft_box_penalty(Topt_larv,    Type(25.0),   Type(31.0),  Type(5),  Type(0.05));
  pen += soft_box_penalty(Tsd_larv,     Type(0.5),    Type(4.0),   Type(5),  Type(0.05));
  pen += soft_box_penalty(Topt_coral,   Type(25.0),   Type(30.5),  Type(5),  Type(0.05));
  pen += soft_box_penalty(Tsd_coral,    Type(0.5),    Type(4.0),   Type(5),  Type(0.05));
  pen += soft_box_penalty(T_bleach,     Type(28.0),   Type(32.0),  Type(5),  Type(0.05));
  pen += soft_box_penalty(tau_bleach,   Type(0.1),    Type(2.0),   Type(5),  Type(0.05));
  pen += soft_box_penalty(m_bleach,     Type(0.0),    Type(2.0),   Type(5),  Type(0.05));
  pen += soft_box_penalty(A_thresh,     Type(0.05),   Type(1.5),   Type(5),  Type(0.05));
  pen += soft_box_penalty(tau_A,        Type(0.05),   Type(2.0),   Type(5),  Type(0.05));
  pen += soft_box_penalty(K_tot,        Type(60.0),   Type(120.0), Type(5),  Type(0.05));
  pen += soft_box_penalty(q_pred,       Type(0.5),    Type(3.0),   Type(5),  Type(0.05));
  pen += soft_box_penalty(sd_log_cots,  Type(0.01),   Type(2.0),   Type(5),  Type(0.05));
  pen += soft_box_penalty(sd_logit_fast,Type(0.01),   Type(2.0),   Type(5),  Type(0.05));
  pen += soft_box_penalty(sd_logit_slow,Type(0.01),   Type(2.0),   Type(5),  Type(0.05));
  pen += soft_box_penalty(min_sd,       Type(1e-6),   Type(0.1),   Type(5),  Type(0.05));
  pen += soft_box_penalty(A_pulse,      Type(0.0),    Type(5.0),   Type(5),  Type(0.05));
  pen += soft_box_penalty(T_thresh_pulse,Type(0.0),   Type(1.5),   Type(5),  Type(0.05));
  pen += soft_box_penalty(tau_pulse,    Type(0.05),   Type(1.5),   Type(5),  Type(0.05));
  pen += soft_box_penalty(gamma_pulse,  Type(1.0),    Type(5.0),   Type(5),  Type(0.05));
  // Initial state plausibility
  pen += soft_box_penalty(p_fast_init,  Type(0.0),    Type(0.98),  Type(10), Type(0.05));
  pen += soft_box_penalty(p_slow_init,  Type(0.0),    Type(0.98),  Type(10), Type(0.05));

  // -----------------------
  // DERIVED QUANTITIES
  // -----------------------
  // Mean SST for anomaly computation (environmental forcing; not a response variable)
  Type mean_sst = 0;
  for (int i = 0; i < n; i++) mean_sst += sst_dat(i);
  mean_sst /= Type(n);

  // State vectors (predictions)
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Initial conditions: estimated parameters (no data leakage)
  cots_pred(0) = cots_init;
  fast_pred(0) = p_fast_init * K_tot;
  slow_pred(0) = p_slow_init * K_tot;

  // -----------------------
  // STATE DYNAMICS
  // -----------------------
  for (int t = 1; t < n; t++) {
    // Use previous-step predictions and previous year's forcing (t-1) for the transition
    Type T = sst_dat(t - 1);

    // Food availability index in [0,1)
    Type food_num = pref_fast * fast_pred(t - 1) + (Type(1) - pref_fast) * slow_pred(t - 1);
    Type Food = food_num / (K_food + food_num + eps);

    // Environmental effects on larvae
    Type EnvLarv = gauss_perf(T, Topt_larv, Tsd_larv, eps);

    // Smooth Allee-like reproduction gate
    Type A_gate = invlogit((cots_pred(t - 1) - A_thresh) / (tau_A + eps));

    // SST anomaly pulse multiplier (rectified Hill function; baseline 1 when anomaly below threshold)
    Type anom = T - mean_sst;
    Type aplus = smooth_pos(anom - T_thresh_pulse, eps);
    Type num = pow(aplus + eps, gamma_pulse);
    Type den = pow(tau_pulse + eps, gamma_pulse) + num + eps;
    Type H = num / den;
    Type Pulse = Type(1) + A_pulse * H;

    // Larval production with density dependence (Ricker), food limitation, Allee gate, and pulse
    Type L_det = phi * cots_pred(t - 1) * Food * EnvLarv * exp(-beta * cots_pred(t - 1)) * A_gate * Pulse
                 + k_imm * cotsimm_dat(t - 1);

    // Adult survival (food-dependent)
    Type sA = exp(-(mA + mA_food * (Type(1) - Food)));

    // Update adults
    Type c_next = sA * cots_pred(t - 1) + mu_adult * L_det;
    c_next = CppAD::CondExpLt(c_next, Type(0), Type(0), c_next);
    cots_pred(t) = c_next;

    // Coral thermal performance modifiers
    Type g_coral = gauss_perf(T, Topt_coral, Tsd_coral, eps);

    // Crowding (logistic) term based on total cover
    Type tot_prev = fast_pred(t - 1) + slow_pred(t - 1);
    Type crowd = Type(1) - tot_prev / (K_tot + eps);

    // Bleaching mortality rate (0 to m_bleach)
    Type bleach_frac = invlogit((T - T_bleach) / (tau_bleach + eps));
    Type M_bleach = m_bleach * bleach_frac;

    // Predation removals with smooth cap (cannot exceed available coral)
    Type Cons_fast_raw = alpha_fast * cots_pred(t - 1) * sat_hill(fast_pred(t - 1), K_pred_fast, q_pred, eps);
    Type Rm_fast = fast_pred(t - 1) * (Type(1) - exp(-Cons_fast_raw / (fast_pred(t - 1) + eps)));

    Type Cons_slow_raw = alpha_slow * cots_pred(t - 1) * sat_hill(slow_pred(t - 1), K_pred_slow, q_pred, eps);
    Type Rm_slow = slow_pred(t - 1) * (Type(1) - exp(-Cons_slow_raw / (slow_pred(t - 1) + eps)));

    // Coral updates
    Type f_next = fast_pred(t - 1)
                  + r_fast * g_coral * fast_pred(t - 1) * crowd
                  - m_fast * fast_pred(t - 1)
                  - M_bleach * fast_pred(t - 1)
                  - Rm_fast;

    Type s_next = slow_pred(t - 1)
                  + r_slow * g_coral * slow_pred(t - 1) * crowd
                  - m_slow * slow_pred(t - 1)
                  - M_bleach * slow_pred(t - 1)
                  - Rm_slow;

    // Non-negativity clamps
    f_next = CppAD::CondExpLt(f_next, Type(0), Type(0), f_next);
    s_next = CppAD::CondExpLt(s_next, Type(0), Type(0), s_next);

    fast_pred(t) = f_next;
    slow_pred(t) = s_next;
  }

  // -----------------------
  // OBSERVATION MODELS
  // -----------------------
  Type sdC = CppAD::CondExpLt(sd_log_cots,  min_sd, min_sd, sd_log_cots);
  Type sdF = CppAD::CondExpLt(sd_logit_fast,min_sd, min_sd, sd_logit_fast);
  Type sdS = CppAD::CondExpLt(sd_logit_slow,min_sd, min_sd, sd_logit_slow);

  for (int t = 0; t < n; t++) {
    // COTS: lognormal on abundance with sd floor and small additive floor to avoid log(0)
    Type y_obs_c = log(cots_dat(t) + min_sd);
    Type y_mu_c  = log(cots_pred(t) + min_sd);
    nll -= dnorm(y_obs_c, y_mu_c, sdC, true);

    // Coral: logit-normal on fraction of K_tot
    // Fast coral
    Type p_obs_f = clamp_open01(fast_dat(t) / (K_tot + eps), Type(1e-6));
    Type p_mu_f  = clamp_open01(fast_pred(t) / (K_tot + eps), Type(1e-6));
    Type z_obs_f = log(p_obs_f / (Type(1) - p_obs_f));
    Type z_mu_f  = log(p_mu_f  / (Type(1) - p_mu_f));
    nll -= dnorm(z_obs_f, z_mu_f, sdF, true);

    // Slow coral
    Type p_obs_s = clamp_open01(slow_dat(t) / (K_tot + eps), Type(1e-6));
    Type p_mu_s  = clamp_open01(slow_pred(t) / (K_tot + eps), Type(1e-6));
    Type z_obs_s = log(p_obs_s / (Type(1) - p_obs_s));
    Type z_mu_s  = log(p_mu_s  / (Type(1) - p_mu_s));
    nll -= dnorm(z_obs_s, z_mu_s, sdS, true);
  }

  // Add soft penalties
  nll += pen;

  // -----------------------
  // REPORTING
  // -----------------------
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(K_tot);
  REPORT(q_pred);
  REPORT(mean_sst);
  REPORT(min_sd);
  REPORT(A_pulse);
  REPORT(tau_pulse);
  REPORT(gamma_pulse);

  ADREPORT(K_tot);
  ADREPORT(q_pred);
  ADREPORT(A_pulse);
  ADREPORT(tau_pulse);
  ADREPORT(gamma_pulse);

  return nll;
}
