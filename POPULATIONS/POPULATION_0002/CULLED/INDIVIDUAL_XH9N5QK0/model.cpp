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
  // For large x, softplus(x) ~ x; for smaller x, use log(1) + exp(x)
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

// Utility: logit with AD-safe clamping
template<class Type>
Type logit_f(Type p, Type eps) {
  p = clamp_open01(p, eps);
  return log(p) - log(Type(1) - p);
}

/*
EQUATION OVERVIEW (all annual, t = Year index):
1) Coral growth (fast/slow):
   F_{t+1} = F_t + r_F(T_t) F_t [1 - (F_t + S_t)/K_tot] - m_F F_t - M_bleach(T_t) F_t - Rm_F(C_t, F_t)
   S_{t+1} = S_t + r_S(T_t) S_t [1 - (F_t + S_t)/K_tot] - m_S S_t - M_bleach(T_t) S_t - Rm_S(C_t, S_t)
   where r_g(T) is a Gaussian thermal performance modifier, M_bleach(T) is a smooth logistic bleaching mortality,
   and Rm_g is COTS predation with Type-II/III saturation and a smooth cap to not exceed available coral.

2) COTS reproduction and survival with 1-year maturation delay:
   Food_t = Hill(pref_fast * F_t + (1 - pref_fast) * S_t; K_food, q_food)
   EnvLarv_t = Gaussian(SST_t; Topt_larv, Tsd_larv)
   A_gate_t = invlogit((C_t - A_thresh) / tau_A)   [smooth Allee-like gate]
   L_t = phi * C_t * Food_t * EnvLarv_t * exp(-beta * C_t) * A_gate_t + k_imm * cotsimm_dat(t)
   sA_t = exp(- (mA + mA_food * (1 - Food_t)))    [food-dependent adult survival]
   C_{t+1} = sA_t * C_t + mu_adult * L_t

3) Predation (component of Eq. 1):
   Cons_fast_raw = alpha_fast * C_t * sat_hill(F_t, K_pred_fast, q_pred)
   Rm_F = F_t * (1 - exp(-Cons_fast_raw / (F_t + eps)))   [smooth cap ≤ F_t]
   Similarly for slow coral with alpha_slow, K_pred_slow.

4) Observation models (use all observations):
   - COTS abundance (individuals/m^2): lognormal with sd floor.
   - Coral cover (percent): transform to fractions relative to K_tot, apply logit-normal with sd floors.

Initial conditions:
   cots_pred(0) = cots_dat(0); fast_pred(0) = fast_dat(0); slow_pred(0) = slow_dat(0).
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
  PARAMETER(log_q_food);          // log of Hill exponent for food saturation (dimensionless, q_food >= 1)

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
  Type q_food = exp(log_q_food);

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

  // Soft biological bounds (do not impose hard constraints)
  pen += soft_box_penalty(r_fast,  Type(0.01), Type(1.5), Type(5), Type(0.05));
  pen += soft_box_penalty(r_slow,  Type(0.005), Type(0.8), Type(5), Type(0.05));
  pen += soft_box_penalty(alpha_fast, Type(0.1), Type(30.0), Type(5), Type(0.05));
  pen += soft_box_penalty(alpha_slow, Type(0.01), Type(15.0), Type(5), Type(0.05));
  pen += soft_box_penalty(K_pred_fast, Type(1.0), Type(60.0), Type(5), Type(0.05));
  pen += soft_box_penalty(K_pred_slow, Type(1.0), Type(60.0), Type(5), Type(0.05));
  pen += soft_box_penalty(mu_adult, Type(0.05), Type(0.8), Type(10), Type(0.05));
  pen += soft_box_penalty(mA, Type(0.05), Type(2.0), Type(5), Type(0.05));
  pen += soft_box_penalty(mA_food, Type(0.01), Type(3.0), Type(5), Type(0.05));
  pen += soft_box_penalty(phi, Type(0.1), Type(20.0), Type(5), Type(0.05));
  pen += soft_box_penalty(beta, Type(0.0), Type(5.0), Type(5), Type(0.05));
  pen += soft_box_penalty(K_food, Type(1.0), Type(100.0), Type(5), Type(0.05));
  pen += soft_box_penalty(q_food, Type(1.0), Type(5.0), Type(10), Type(0.1));
  pen += soft_box_penalty(Topt_larv, Type(25.0), Type(31.0), Type(5), Type(0.05));
  pen += soft_box_penalty(Tsd_larv, Type(0.5), Type(4.0), Type(5), Type(0.05));
  pen += soft_box_penalty(Topt_coral, Type(25.0), Type(30.5), Type(5), Type(0.05));
  pen += soft_box_penalty(Tsd_coral, Type(0.5), Type(4.0), Type(5), Type(0.05));
  pen += soft_box_penalty(T_bleach, Type(28.0), Type(32.0), Type(5), Type(0.05));
  pen += soft_box_penalty(tau_bleach, Type(0.1), Type(2.0), Type(5), Type(0.05));
  pen += soft_box_penalty(K_tot, Type(60.0), Type(120.0), Type(5), Type(0.02)); // soft box around plausible substrate %

  // Apply sd floors
  sd_log_cots   = CppAD::CondExpLt(sd_log_cots,   min_sd, min_sd, sd_log_cots);
  sd_logit_fast = CppAD::CondExpLt(sd_logit_fast, min_sd, min_sd, sd_logit_fast);
  sd_logit_slow = CppAD::CondExpLt(sd_logit_slow, min_sd, min_sd, sd_logit_slow);

  // -----------------------
  // STATE VECTORS
  // -----------------------
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Initialize states from data at t=0 (allowed by spec)
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // For reporting auxiliary drivers
  vector<Type> Food(n);
  vector<Type> EnvLarv(n);
  vector<Type> A_gate(n);
  vector<Type> L_recruits(n);
  vector<Type> sA_surv(n);
  vector<Type> M_bleach_rt(n);

  for (int t = 0; t < n; t++) {
    // Current states (predicted)
    Type C_t = cots_pred(t);
    Type F_t = fast_pred(t);
    Type S_t = slow_pred(t);

    // Forcing at time t
    Type T_t = sst_dat(t);

    // Bleaching mortality (smooth logistic)
    Type z_bleach = (T_t - T_bleach) / (tau_bleach + eps);
    Type M_b = m_bleach * invlogit(z_bleach);
    M_bleach_rt(t) = M_b;

    // Coral thermal performance scaling of growth
    Type perf_coral = gauss_perf(T_t, Topt_coral, Tsd_coral, eps);

    // Predation terms with Hill saturation (q_pred) and smooth cap
    Type cons_fast_raw = alpha_fast * C_t * sat_hill(F_t, K_pred_fast, q_pred, eps);
    Type cons_slow_raw = alpha_slow * C_t * sat_hill(S_t, K_pred_slow, q_pred, eps);

    Type Rm_F = F_t * (Type(1) - exp(-cons_fast_raw / (F_t + eps)));
    Type Rm_S = S_t * (Type(1) - exp(-cons_slow_raw / (S_t + eps)));

    // Logistic growth with total cover limitation (F + S <= K_tot)
    Type growth_F = r_fast * perf_coral * F_t * (Type(1) - (F_t + S_t) / (K_tot + eps));
    Type growth_S = r_slow * perf_coral * S_t * (Type(1) - (F_t + S_t) / (K_tot + eps));

    Type mort_F = m_fast * F_t + M_b * F_t + Rm_F;
    Type mort_S = m_slow * S_t + M_b * S_t + Rm_S;

    // Food index from preferred coral composition with Hill saturation
    Type Food_num = pref_fast * F_t + (Type(1) - pref_fast) * S_t; // in % cover
    Type Food_t = sat_hill(Food_num, K_food, q_food, eps);
    Food(t) = Food_t;

    // Environmental larval survival modifier
    Type EnvLarv_t = gauss_perf(T_t, Topt_larv, Tsd_larv, eps);
    EnvLarv(t) = EnvLarv_t;

    // Adult reproduction gate (smooth Allee-like)
    Type A_gate_t = invlogit((C_t - A_thresh) / (tau_A + eps));
    A_gate(t) = A_gate_t;

    // Larval production and effective recruits
    Type L_t = phi * C_t * Food_t * EnvLarv_t * exp(-beta * C_t) * A_gate_t + k_imm * cotsimm_dat(t);
    L_recruits(t) = L_t;

    // Adult survival with food-dependent stress
    Type sA_t = exp(-(mA + mA_food * (Type(1) - Food_t)));
    sA_surv(t) = sA_t;

    // Observation likelihoods at time t (use predictions only; data used only in likelihood)
    // COTS: lognormal on abundance
    Type y_cots = cots_dat(t) + eps;                   // small floor to handle zeros
    Type mu_cots = log(C_t + eps);
    nll -= dnorm(log(y_cots), mu_cots, sd_log_cots, true);

    // Coral: logit-normal on fraction of K_tot
    Type y_fast_frac = clamp_open01(fast_dat(t) / (K_tot + eps), Type(1e-8));
    Type y_slow_frac = clamp_open01(slow_dat(t) / (K_tot + eps), Type(1e-8));
    Type x_fast_frac = clamp_open01(F_t / (K_tot + eps), Type(1e-8));
    Type x_slow_frac = clamp_open01(S_t / (K_tot + eps), Type(1e-8));

    Type mu_fast_logit = logit_f(x_fast_frac, Type(1e-8));
    Type mu_slow_logit = logit_f(x_slow_frac, Type(1e-8));
    Type y_fast_logit = logit_f(y_fast_frac, Type(1e-8));
    Type y_slow_logit = logit_f(y_slow_frac, Type(1e-8));

    nll -= dnorm(y_fast_logit, mu_fast_logit, sd_logit_fast, true);
    nll -= dnorm(y_slow_logit, mu_slow_logit, sd_logit_slow, true);

    // State update to t+1 (no data leakage: only previous predicted states and forcings)
    if (t < n - 1) {
      // Update coral
      Type F_next = F_t + growth_F - mort_F;
      Type S_next = S_t + growth_S - mort_S;

      // Non-negativity and cap at K_tot with AD-safe clamps
      F_next = smooth_pos(F_next, eps);
      S_next = smooth_pos(S_next, eps);
      // Enforce combined cap softly by proportionally scaling if exceeding K_tot
      Type tot_next = F_next + S_next + eps;
      Type over = CppAD::CondExpGt(tot_next, K_tot, tot_next / (K_tot + eps), Type(1));
      F_next = F_next / over;
      S_next = S_next / over;

      // Update COTS
      Type C_next = sA_t * C_t + mu_adult * L_t;
      C_next = smooth_pos(C_next, eps);

      fast_pred(t + 1) = F_next;
      slow_pred(t + 1) = S_next;
      cots_pred(t + 1) = C_next;
    }
  }

  // Add penalties
  nll += pen;

  // REPORTS
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(Food);
  REPORT(EnvLarv);
  REPORT(A_gate);
  REPORT(L_recruits);
  REPORT(sA_surv);
  REPORT(M_bleach_rt);

  // Derived parameters for reporting
  ADREPORT(r_fast);
  ADREPORT(r_slow);
  ADREPORT(alpha_fast);
  ADREPORT(alpha_slow);
  ADREPORT(K_pred_fast);
  ADREPORT(K_pred_slow);
  ADREPORT(pref_fast);
  ADREPORT(mu_adult);
  ADREPORT(mA);
  ADREPORT(mA_food);
  ADREPORT(phi);
  ADREPORT(beta);
  ADREPORT(K_food);
  ADREPORT(q_food);
  ADREPORT(Topt_larv);
  ADREPORT(Tsd_larv);
  ADREPORT(Topt_coral);
  ADREPORT(Tsd_coral);
  ADREPORT(A_thresh);
  ADREPORT(tau_A);
  ADREPORT(k_imm);
  ADREPORT(sd_log_cots);
  ADREPORT(sd_logit_fast);
  ADREPORT(sd_logit_slow);
  ADREPORT(K_tot);
  ADREPORT(q_pred);

  return nll;
}
