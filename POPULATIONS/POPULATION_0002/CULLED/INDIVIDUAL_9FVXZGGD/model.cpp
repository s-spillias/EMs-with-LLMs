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

/*
EQUATION OVERVIEW (all annual, t = Year index):
1) Coral growth (fast/slow):
   F_{t+1} = F_t + r_F(T_t) F_t [1 - (F_t + S_t)/K_tot] - m_F F_t - M_bleach(T_t) F_t - Rm_F(C_t, F_t)
   S_{t+1} = S_t + r_S(T_t) S_t [1 - (F_t + S_t)/K_tot] - m_S S_t - M_bleach(T_t) S_t - Rm_S(C_t, S_t)
   where r_g(T) is a Gaussian thermal performance modifier, M_bleach(T) is a smooth logistic bleaching mortality,
   and Rm_g is COTS predation with Type-II/III saturation and a smooth cap to not exceed available coral.

2) COTS life cycle with juvenile reservoir and food-dependent maturation:
   L_t = phi * C_t * Food_t * EnvLarv_t * exp(-beta * C_t) * A_gate_t + k_imm * cotsimm_dat(t)      [larval settlers]
   J_{t+1} = exp(-mJ) * J_t + L_t                                                                    [juveniles persist]
   mu_J(Food_t) = mu_adult * Food_t^{q_mature}                                                       [max maturation * food sensitivity]
   C_{t+1} = sA_t * C_t + mu_J(Food_t) * J_t                                                         [adults from juveniles]
   where sA_t = exp(- (mA + mA_food * (1 - Food_t))) is adult survival.

3) Predation (component of Eq. 1):
   Cons_fast_raw = alpha_fast * C_t * sat_hill(F_t, K_pred_fast, q_pred)
   Rm_F = F_t * (1 - exp(-Cons_fast_raw / (F_t + eps)))   [smooth cap ≤ F_t]
   Similarly for slow coral with alpha_slow, K_pred_slow.

4) Observation models (use all observations):
   - COTS abundance (individuals/m^2): lognormal with sd floor.
   - Coral cover (percent): transform to fractions, apply logit-normal with sd floors.

Initial conditions:
   cots_pred(0) = cots_dat(0); fast_pred(0) = fast_dat(0); slow_pred(0) = slow_dat(0);
   J_pred(0) = J0 (estimated).
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
  PARAMETER(logit_mu_adult);      // logit of maximum juvenile-to-adult maturation fraction per year (unitless in [0,1])

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

  // New parameters for juvenile reservoir dynamics
  PARAMETER(log_mJ);              // log of baseline juvenile mortality rate (year^-1)
  PARAMETER(log_q_mature);        // log of food sensitivity exponent for juvenile->adult maturation (dimensionless)
  PARAMETER(log_J0);              // log of initial juvenile density at t=0 (indiv/m^2)

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
  Type mu_adult = invlogit(logit_mu_adult); // now interpreted as max juvenile->adult maturation fraction per year

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

  // New transformed juvenile parameters
  Type mJ = exp(log_mJ);
  Type q_mature = exp(log_q_mature);
  Type J0 = exp(log_J0);

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
  pen += soft_box_penalty(Topt_larv, Type(25.0), Type(31.0), Type(5), Type(0.05));
  pen += soft_box_penalty(Tsd_larv, Type(0.5), Type(4.0), Type(5), Type(0.05));
  pen += soft_box_penalty(Topt_coral, Type(25.0), Type(30.5), Type(5), Type(0.05));
  pen += soft_box_penalty(Tsd_coral, Type(0.5), Type(4.0), Type(5), Type(0.05));
  pen += soft_box_penalty(T_bleach, Type(28.0), Type(32.0), Type(5), Type(0.05));
  pen += soft_box_penalty(tau_bleach, Type(0.1), Type(2.0), Type(5), Type(0.05));
  pen += soft_box_penalty(m_bleach, Type(0.0), Type(2.0), Type(5), Type(0.05));
  pen += soft_box_penalty(A_thresh, Type(0.05), Type(1.5), Type(5), Type(0.05));
  pen += soft_box_penalty(tau_A, Type(0.05), Type(1.5), Type(5), Type(0.05));
  pen += soft_box_penalty(k_imm, Type(0.05), Type(5.0), Type(5), Type(0.05));

  // New penalties for transformed scalars
  pen += soft_box_penalty(K_tot, Type(60.0), Type(120.0), Type(5), Type(0.1));    // % cover
  pen += soft_box_penalty(min_sd, Type(0.0), Type(0.5), Type(10), Type(0.1));     // sd floor
  pen += soft_box_penalty(q_pred, Type(1.0), Type(3.0), Type(10), Type(0.1));     // shape exponent

  // New penalties for juvenile dynamics
  pen += soft_box_penalty(mJ, Type(0.05), Type(3.0), Type(5), Type(0.05));        // juvenile mortality rate
  pen += soft_box_penalty(q_mature, Type(1.0), Type(6.0), Type(10), Type(0.05));  // food sensitivity exponent
  pen += soft_box_penalty(J0, Type(0.0), Type(2.0), Type(5), Type(0.05));         // plausible initial juveniles

  // ---------------
  // STATE PREDICTIONS
  // ---------------
  vector<Type> cots_pred(n);                                              // predicted adult COTS abundance (indiv/m^2)
  vector<Type> fast_pred(n);                                              // predicted fast coral cover (%)
  vector<Type> slow_pred(n);                                              // predicted slow coral cover (%)
  vector<Type> J_pred(n);                                                 // predicted juvenile COTS density (indiv/m^2)

  // Initial conditions from data or parameters (no optimization of coral/adult starting states)
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  J_pred(0) = J0;

  for (int t = 0; t < n - 1; t++) {
    // Retrieve previous-step state (no data leakage)
    Type C_t = cots_pred(t);
    Type F_t = fast_pred(t);
    Type S_t = slow_pred(t);
    Type J_t = J_pred(t);

    // Forcing at time t
    Type T_t = sst_dat(t);
    Type Imm_t = cotsimm_dat(t);

    // Thermal modifiers
    Type g_coral = gauss_perf(T_t, Topt_coral, Tsd_coral, eps);            // 0..1 multiplier on coral growth
    Type rF_eff = r_fast * g_coral;                                       // effective growth rate (fast coral)
    Type rS_eff = r_slow * g_coral;                                       // effective growth rate (slow coral)

    // Bleaching mortality (smooth logistic above T_bleach)
    Type bleach_gate = invlogit((T_t - T_bleach) / (tau_bleach + eps));   // 0..1
    Type M_bleach = m_bleach * bleach_gate;                               // additional mortality rate (year^-1)

    // COTS predation on corals with saturation and smooth cap
    Type satF = sat_hill(F_t, K_pred_fast, q_pred, eps);
    Type satS = sat_hill(S_t, K_pred_slow, q_pred, eps);
    Type ConsF_raw = alpha_fast * C_t * satF;                              // % cover per year pressure
    Type ConsS_raw = alpha_slow * C_t * satS;                              // % cover per year pressure

    // Smooth cap so removal cannot exceed available coral (Rm <= current cover)
    Type Rm_F = F_t * (Type(1) - exp(-ConsF_raw / (F_t + eps)));          // % cover removed from fast coral
    Type Rm_S = S_t * (Type(1) - exp(-ConsS_raw / (S_t + eps)));          // % cover removed from slow coral

    // Coral updates (logistic growth toward shared carrying capacity K_tot)
    Type total_cover = F_t + S_t;
    Type comp_term = (Type(1) - total_cover / (K_tot + eps));             // shared substrate limitation
    Type F_next = F_t
                + rF_eff * F_t * comp_term
                - m_fast * F_t
                - M_bleach * F_t
                - Rm_F;

    Type S_next = S_t
                + rS_eff * S_t * comp_term
                - m_slow * S_t
                - M_bleach * S_t
                - Rm_S;

    // Enforce non-negativity smoothly
    F_next = smooth_pos(F_next, eps);
    S_next = smooth_pos(S_next, eps);

    // Food index for COTS reproduction (saturating with preference)
    Type wF = pref_fast;
    Type wS = Type(1) - pref_fast;
    Type Food_num = wF * F_t + wS * S_t;                                  // % cover weighted by preference
    Type Food = Food_num / (K_food + Food_num + eps);                     // 0..1 food saturation for fecundity and maturation

    // Environmental modifier for larval survival
    Type EnvLarv = gauss_perf(T_t, Topt_larv, Tsd_larv, eps);             // 0..1

    // Smooth Allee-like gate on adult repro
    Type A_gate = invlogit((C_t - A_thresh) / (tau_A + eps));             // 0..1

    // Larval production at time t (recruits into juveniles)
    Type L_t = phi * C_t * Food * EnvLarv * exp(-beta * C_t) * A_gate
             + k_imm * Imm_t;                                            // include exogenous immigration

    // Adult survival (food-dependent)
    Type mA_eff = mA + mA_food * (Type(1) - Food);                        // higher when Food is low
    Type sA = exp(-mA_eff);                                               // survival fraction in [0,1]

    // Juvenile maturation fraction (depends on coral-derived Food)
    Type muJ = mu_adult * pow(clamp_open01(Food, Type(1e-8)), q_mature);  // in [0,1], more sensitive with larger q_mature

    // COTS state updates
    Type J_next = exp(-mJ) * J_t + L_t;                                   // juveniles persist and accumulate
    Type C_next = sA * C_t + muJ * J_t;                                   // adults from surviving adults + matured juveniles

    // Enforce non-negativity smoothly
    J_next = smooth_pos(J_next, eps);
    C_next = smooth_pos(C_next, eps);

    // Assign
    fast_pred(t + 1) = F_next;
    slow_pred(t + 1) = S_next;
    J_pred(t + 1) = J_next;
    cots_pred(t + 1) = C_next;
  }

  // ---------------
  // LIKELIHOOD: USE ALL OBSERVATIONS
  // ---------------
  // SD floors on transformed scales
  Type sd_cots_eff = sqrt(sd_log_cots * sd_log_cots + min_sd * min_sd);
  Type sd_fast_eff = sqrt(sd_logit_fast * sd_logit_fast + min_sd * min_sd);
  Type sd_slow_eff = sqrt(sd_logit_slow * sd_logit_slow + min_sd * min_sd);

  for (int t = 0; t < n; t++) {
    // COTS: lognormal on positive scale
    Type y_c = cots_dat(t);
    Type mu_c = log(cots_pred(t) + eps);
    Type obs_c = log(y_c + eps);
    nll -= dnorm(obs_c, mu_c, sd_cots_eff, true);

    // Coral: logit-normal on fraction scale
    Type yF_frac = (fast_dat(t) / Type(100.0));
    Type yS_frac = (slow_dat(t) / Type(100.0));
    // Clamp fractions to (0,1) open interval using AD-safe clamp
    yF_frac = clamp_open01(yF_frac, Type(1e-8));
    yS_frac = clamp_open01(yS_frac, Type(1e-8));

    Type pF_pred = (fast_pred(t) / Type(100.0));
    Type pS_pred = (slow_pred(t) / Type(100.0));
    pF_pred = clamp_open01(pF_pred, Type(1e-8));
    pS_pred = clamp_open01(pS_pred, Type(1e-8));

    Type muF = logit(pF_pred);
    Type muS = logit(pS_pred);
    Type obsF = logit(yF_frac);
    Type obsS = logit(yS_frac);

    nll -= dnorm(obsF, muF, sd_fast_eff, true);
    nll -= dnorm(obsS, muS, sd_slow_eff, true);
  }

  // Add parameter penalties
  nll += pen;

  // ---------------
  // REPORTING
  // ---------------
  REPORT(cots_pred);   // predicted adult COTS abundance (indiv/m^2)
  REPORT(fast_pred);   // predicted fast coral cover (%)
  REPORT(slow_pred);   // predicted slow coral cover (%)
  REPORT(J_pred);      // predicted juvenile COTS density (indiv/m^2)

  // Optional diagnostics that help interpretation (not required but useful)
  REPORT(r_fast);
  REPORT(r_slow);
  REPORT(alpha_fast);
  REPORT(alpha_slow);
  REPORT(K_pred_fast);
  REPORT(K_pred_slow);
  REPORT(pref_fast);
  REPORT(mu_adult);
  REPORT(mA);
  REPORT(mA_food);
  REPORT(phi);
  REPORT(beta);
  REPORT(K_food);
  REPORT(Topt_larv);
  REPORT(Tsd_larv);
  REPORT(Topt_coral);
  REPORT(Tsd_coral);
  REPORT(T_bleach);
  REPORT(tau_bleach);
  REPORT(m_bleach);
  REPORT(A_thresh);
  REPORT(tau_A);
  REPORT(k_imm);
  REPORT(sd_log_cots);
  REPORT(sd_logit_fast);
  REPORT(sd_logit_slow);

  // New reports for converted scalars
  REPORT(K_tot);
  REPORT(min_sd);
  REPORT(q_pred);

  // New reports for juvenile parameters
  REPORT(mJ);
  REPORT(q_mature);
  REPORT(J0);

  return nll;
}
