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
  Type penL = softplus(-k * (x - L)); // penalize x << L
  Type penU = softplus(-k * (U - x)); // penalize x >> U
  return w * (penL + penU);
}

// Utility: AD-safe clamp to open unit interval (eps, 1 - eps)
template<class Type>
Type clamp_open01(Type p, Type eps) {
  p = CppAD::CondExpLt(p, eps, eps, p);
  p = CppAD::CondExpGt(p, Type(1) - eps, Type(1) - eps, p);
  return p;
}

template<class Type>
Type inv_logit(Type x) {
  return Type(1) / (Type(1) + exp(-x));
}

template<class Type>
Type logit_x(Type p, Type eps) {
  p = clamp_open01(p, eps);
  return log(p / (Type(1) - p));
}

/*
EQUATION OVERVIEW (all annual, t = Year index):
1) Coral growth (fast/slow):
   F_{t+1} = F_t + r_F(T_t) F_t [1 - (F_t + S_t + R_t)/K_tot] - m_F F_t - M_bleach(T_t) F_t - Rm_F(C_t, F_t)
   S_{t+1} = S_t + r_S(T_t) S_t [1 - (F_t + S_t + R_t)/K_tot] - m_S S_t - M_bleach(T_t) S_t - Rm_S(C_t, S_t)
   where r_g(T) is a Gaussian thermal performance modifier, M_bleach(T) is a smooth logistic bleaching mortality,
   and Rm_g is COTS predation with Type-II/III saturation and a smooth cap to not exceed available coral.
   New: R_t (rubble) occupies substrate after coral loss and decays over time, delaying recovery.

2) COTS reproduction, juvenile survival, and adult survival with maturation:
   L_t = phi * C_t * Food_t * EnvLarv_t * exp(-beta * C_t) * A_gate_t * Pulse_t + k_imm * cotsimm_dat(t)
   J_{t+1} = sJ_eff_t * J_t + L_t,  where sJ_eff_t = sJ * exp(-k_can * (C_t + J_t))
   C_{t+1} = sA_t * C_t + mJ * sJ_eff_t * J_t
   where mJ = invlogit(logit_mu_adult) is the annual maturation fraction from juvenile to adult,
   sJ = invlogit(logit_sJ) is annual baseline juvenile survival,
   and sA_t is food- and temperature-dependent adult survival.

3) Predation (component of Eq. 1):
   Cons_fast_raw = alpha_fast * C_t * sat_hill(F_t, K_pred_fast, q_pred)
   Cons_slow_raw = alpha_slow * C_t * sat_hill(S_t, K_pred_slow, q_pred)
   Cons_* are smoothly capped to not exceed available coral using a differentiable min.

4) Feedbacks:
   a) Density-dependent juvenile mortality from conspecific crowding/cannibalism via k_can, applied to total density (C_t + J_t).
   b) New rubble/stabilization state R_t: R_{t+1} = e^{-k_rub} R_t + rho_rub * (loss_F_t + loss_S_t)
      where loss_g_t = m_g * G_t + M_bleach_t * G_t + cons_g_t.
*/

template<class Type>
Type objective_function<Type>::operator() () {
  Type nll = 0.0;

  // Small constants
  const Type eps = Type(1e-8);
  const Type eps_obs = Type(1e-8);
  const Type clamp_eps = Type(1e-8);

  // Data
  DATA_VECTOR(Year);         // length T
  DATA_VECTOR(cots_dat);     // observed adults (response)
  DATA_VECTOR(fast_dat);     // observed fast coral % cover (response)
  DATA_VECTOR(slow_dat);     // observed slow coral % cover (response)
  DATA_VECTOR(sst_dat);      // forcing
  DATA_VECTOR(cotsimm_dat);  // forcing

  int T = Year.size();

  // Parameters
  PARAMETER(log_r_fast);
  PARAMETER(log_r_slow);
  PARAMETER(log_m_fast);
  PARAMETER(log_m_slow);
  PARAMETER(log_m_bleach);
  PARAMETER(T_bleach);
  PARAMETER(log_tau_bleach);

  PARAMETER(log_alpha_fast);
  PARAMETER(log_alpha_slow);
  PARAMETER(log_K_pred_fast);
  PARAMETER(log_K_pred_slow);
  PARAMETER(log_q_pred);

  PARAMETER(pref_fast_logit);

  // Reinterpreted: annual maturation fraction from juvenile to adult
  PARAMETER(logit_mu_adult);
  PARAMETER(log_mA);
  PARAMETER(log_mA_food);
  // New temperature-driven adult mortality parameters
  PARAMETER(log_mA_temp);
  PARAMETER(T_mA);
  PARAMETER(log_tau_mA);

  PARAMETER(log_phi);
  PARAMETER(log_beta);
  PARAMETER(log_K_food);

  PARAMETER(Topt_larv);
  PARAMETER(log_Tsd_larv);

  PARAMETER(log_A_pulse);
  PARAMETER(T_thresh_pulse);
  PARAMETER(log_tau_pulse);
  PARAMETER(log_gamma_pulse);

  PARAMETER(Topt_coral);
  PARAMETER(log_Tsd_coral);

  PARAMETER(A_thresh);
  PARAMETER(log_tau_A);

  PARAMETER(log_k_imm);

  PARAMETER(log_sd_log_cots);
  PARAMETER(log_sd_logit_fast);
  PARAMETER(log_sd_logit_slow);

  PARAMETER(log_K_tot);
  PARAMETER(log_min_sd);

  PARAMETER(log_cots_init);
  PARAMETER(logit_fast_init);
  PARAMETER(logit_slow_init);

  // New juvenile-stage parameters
  PARAMETER(logit_sJ);       // annual juvenile survival (logit)
  PARAMETER(log_juv_init);   // initial juvenile density

  // New parameter: density-dependent juvenile mortality strength (cannibalism/competition)
  PARAMETER(log_k_can);

  // New rubble/stabilization parameters and initial state
  PARAMETER(log_k_rub);        // rubble exponential decay rate (year^-1)
  PARAMETER(logit_rho_rub);    // fraction of coral loss becoming rubble (0..1 on logit)
  PARAMETER(log_rubble_init);  // initial rubble cover

  // Transforms
  Type r_fast0 = exp(log_r_fast);
  Type r_slow0 = exp(log_r_slow);
  Type m_fast = exp(log_m_fast);
  Type m_slow = exp(log_m_slow);
  Type m_bleach = exp(log_m_bleach);
  Type tau_bleach = exp(log_tau_bleach);

  Type alpha_fast = exp(log_alpha_fast);
  Type alpha_slow = exp(log_alpha_slow);
  Type K_pred_fast = exp(log_K_pred_fast);
  Type K_pred_slow = exp(log_K_pred_slow);
  Type q_pred = exp(log_q_pred);

  Type pref_fast = inv_logit(pref_fast_logit);

  // Reinterpreted: maturation fraction from juvenile to adult per year
  Type mJ = inv_logit(logit_mu_adult);
  Type mA = exp(log_mA);
  Type mA_food = exp(log_mA_food);
  Type mA_temp = exp(log_mA_temp);
  Type tau_mA = exp(log_tau_mA);

  Type phi = exp(log_phi);
  Type beta = exp(log_beta);
  Type K_food = exp(log_K_food);

  Type Tsd_larv = exp(log_Tsd_larv);

  Type A_pulse = exp(log_A_pulse);
  Type tau_pulse = exp(log_tau_pulse);
  Type gamma_pulse = exp(log_gamma_pulse);

  Type Tsd_coral = exp(log_Tsd_coral);

  Type tau_A = exp(log_tau_A);

  Type k_imm = exp(log_k_imm);

  Type sd_log_cots = exp(log_sd_log_cots);
  Type sd_logit_fast = exp(log_sd_logit_fast);
  Type sd_logit_slow = exp(log_sd_logit_slow);

  Type K_tot = exp(log_K_tot);
  Type min_sd = exp(log_min_sd);

  Type C0 = exp(log_cots_init);
  Type F0 = inv_logit(logit_fast_init) * K_tot;
  Type S0 = inv_logit(logit_slow_init) * K_tot;

  // Juvenile transforms
  Type sJ = inv_logit(logit_sJ);
  Type J0 = exp(log_juv_init);

  // New density-dependent juvenile mortality parameter
  Type k_can = exp(log_k_can);

  // Rubble transforms and initial state
  Type k_rub = exp(log_k_rub);            // decay rate
  Type rho_rub = inv_logit(logit_rho_rub);
  Type R0 = exp(log_rubble_init);

  // Enforce sd floors (AD-safe)
  sd_log_cots = CppAD::CondExpLt(sd_log_cots, min_sd, min_sd, sd_log_cots);
  sd_logit_fast = CppAD::CondExpLt(sd_logit_fast, min_sd, min_sd, sd_logit_fast);
  sd_logit_slow = CppAD::CondExpLt(sd_logit_slow, min_sd, min_sd, sd_logit_slow);

  // Precompute mean SST for anomaly-based pulse
  Type sst_mean = 0.0;
  for (int t = 0; t < T; t++) sst_mean += sst_dat(t);
  sst_mean /= Type(T);

  // Prediction vectors (also used as state vectors; explicit equations; no data leakage)
  vector<Type> cots_pred(T);
  vector<Type> fast_pred(T);
  vector<Type> slow_pred(T);
  vector<Type> juv_pred(T);    // juvenile state
  vector<Type> rubble_pred(T); // rubble state

  // Initialize predictions at t = 0
  cots_pred(0) = C0;
  fast_pred(0) = F0;
  slow_pred(0) = S0;
  juv_pred(0) = J0;
  rubble_pred(0) = R0;

  // Process model: forward simulate using only previous-step predictions and exogenous forcings
  for (int t = 0; t < T - 1; t++) {
    Type sst = sst_dat(t);

    // Coral growth modifiers
    Type g_fast_T = gauss_perf(sst, Topt_coral, Tsd_coral, eps);
    Type g_slow_T = gauss_perf(sst, Topt_coral, Tsd_coral, eps);

    Type rF = r_fast0 * g_fast_T;
    Type rS = r_slow0 * g_slow_T;

    // Bleaching mortality (smooth logistic gate)
    Type bleach_gate = inv_logit((sst - T_bleach) / (tau_bleach + eps));
    Type M_bleach = m_bleach * bleach_gate;

    // Current states (predictions at time t)
    Type Ct = cots_pred(t);
    Type Ft = fast_pred(t);
    Type St = slow_pred(t);
    Type Jt = juv_pred(t);
    Type Rt = rubble_pred(t);

    // Predation (Type-II/III saturation), smoothly capped to not exceed available coral
    Type cons_fast_raw = alpha_fast * Ct * sat_hill(Ft, K_pred_fast, q_pred, eps);
    Type cons_slow_raw = alpha_slow * Ct * sat_hill(St, K_pred_slow, q_pred, eps);

    Type cons_fast = Ft - smooth_pos(Ft - cons_fast_raw, eps); // ~min(cons_fast_raw, F)
    Type cons_slow = St - smooth_pos(St - cons_slow_raw, eps); // ~min(cons_slow_raw, S)

    // Coral losses (that contribute to rubble)
    Type loss_F = m_fast * Ft + M_bleach * Ft + cons_fast;
    Type loss_S = m_slow * St + M_bleach * St + cons_slow;

    // Logistic coral growth with rubble occupying substrate
    Type crowd = (Ft + St + Rt) / (K_tot + eps);

    Type F_next = Ft + rF * Ft * (Type(1) - crowd) - m_fast * Ft - M_bleach * Ft - cons_fast;
    Type S_next = St + rS * St * (Type(1) - crowd) - m_slow * St - M_bleach * St - cons_slow;

    // Floor at zero smoothly
    F_next = smooth_pos(F_next, eps);
    S_next = smooth_pos(S_next, eps);

    // Food index (0..1) based on live coral only
    Type food_avail = pref_fast * Ft + (Type(1) - pref_fast) * St;
    Type Food = food_avail / (K_food + food_avail + eps);

    // Larval environment performance
    Type EnvLarv = gauss_perf(sst, Topt_larv, Tsd_larv, eps);

    // Reproduction Allee gate
    Type A_gate = inv_logit((Ct - A_thresh) / (tau_A + eps));

    // SST anomaly pulse (rectified Hill of positive anomalies beyond threshold)
    Type anom = sst - sst_mean;
    Type pos = smooth_pos(anom - T_thresh_pulse, eps);
    Type H = pow(pos, gamma_pulse) / (pow(tau_pulse, gamma_pulse) + pow(pos, gamma_pulse) + eps);
    Type Pulse = Type(1) + A_pulse * H;

    // Larval production + immigration (juvenile-equivalent recruits after settlement)
    Type L = phi * Ct * Food * EnvLarv * exp(-beta * Ct) * A_gate * Pulse + k_imm * cotsimm_dat(t);

    // Adult survival with food- and temperature-modulated mortality
    Type M_temp = mA_temp * inv_logit((sst - T_mA) / (tau_mA + eps));
    Type sA = exp(-(mA + mA_food * (Type(1) - Food) + M_temp));

    // Stage-structured transitions with density-dependent juvenile survival (cannibalism/competition)
    Type D_crowd = Ct + Jt; // total conspecific density
    Type sJ_eff = sJ * exp(-k_can * D_crowd);
    // Surviving juveniles split into maturing and remaining juveniles
    Type J_survive = sJ_eff * Jt;
    Type C_next = sA * Ct + mJ * J_survive;
    Type J_next = (Type(1) - mJ) * J_survive + L;

    // Rubble dynamics: exponential decay plus new rubble from coral losses
    Type R_next = exp(-k_rub) * Rt + rho_rub * (loss_F + loss_S);

    // Floor at zero smoothly for state variables and softly cap rubble to K_tot
    C_next = smooth_pos(C_next, eps);
    J_next = smooth_pos(J_next, eps);
    R_next = smooth_pos(R_next, eps);
    // Soft cap: R_next <= K_tot
    R_next = K_tot - smooth_pos(K_tot - R_next, eps);

    // Assign predictions for t+1
    cots_pred(t + 1) = C_next;
    fast_pred(t + 1) = F_next;
    slow_pred(t + 1) = S_next;
    juv_pred(t + 1) = J_next;
    rubble_pred(t + 1) = R_next;
  }

  // Observation likelihoods (do not feed observations back into the process)
  for (int t = 0; t < T; t++) {
    // COTS: lognormal on abundance
    Type y_cots = CppAD::CondExpLt(cots_dat(t), eps_obs, eps_obs, cots_dat(t)); // avoid log(0)
    Type mu_cots = log(cots_pred(t) + eps_obs);
    nll -= dnorm(log(y_cots), mu_cots, sd_log_cots, true);

    // Coral fast: logit-normal on fraction of K_tot
    Type p_fast_obs = clamp_open01(fast_dat(t) / (K_tot + eps), clamp_eps);
    Type p_fast_pred = clamp_open01(fast_pred(t) / (K_tot + eps), clamp_eps);
    Type z_fast_obs = logit_x(p_fast_obs, clamp_eps);
    Type z_fast_pred = logit_x(p_fast_pred, clamp_eps);
    nll -= dnorm(z_fast_obs, z_fast_pred, sd_logit_fast, true);

    // Coral slow: logit-normal on fraction of K_tot
    Type p_slow_obs = clamp_open01(slow_dat(t) / (K_tot + eps), clamp_eps);
    Type p_slow_pred = clamp_open01(slow_pred(t) / (K_tot + eps), clamp_eps);
    Type z_slow_obs = logit_x(p_slow_obs, clamp_eps);
    Type z_slow_pred = logit_x(p_slow_pred, clamp_eps);
    nll -= dnorm(z_slow_obs, z_slow_pred, sd_logit_slow, true);
  }

  // Optional soft penalties (light-touch, can be tuned or removed)
  // Example: keep preference away from extremes to avoid identifiability issues
  nll += soft_box_penalty(pref_fast_logit, Type(-5.0), Type(5.0), Type(1.0), Type(1e-4));

  // Reports
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(juv_pred);
  REPORT(rubble_pred);
  REPORT(K_tot);
  REPORT(sd_log_cots);
  REPORT(sd_logit_fast);
  REPORT(sd_logit_slow);
  REPORT(mJ);   // maturation fraction
  REPORT(sJ);   // baseline juvenile survival
  REPORT(k_can); // density-dependent juvenile mortality strength
  REPORT(k_rub); // rubble decay rate
  REPORT(rho_rub); // fraction of coral loss becoming rubble

  return nll;
}
