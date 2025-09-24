#include <TMB.hpp>

// Helper: inverse-logit with numerical stability
template<class Type>
Type invlogit_stable(Type x) {
  // Avoid overflow in exp for large |x|
  if (x >= Type(0)) {
    Type z = exp(-x);
    return Type(1) / (Type(1) + z);
  } else {
    Type z = exp(x);
    return z / (Type(1) + z);
  }
}

// Helper: logit transform with small epsilon
template<class Type>
Type logit_stable(Type p) {
  Type eps = Type(1e-8);
  p = CppAD::CondExpLt(p, eps, eps, p);                 // clamp low
  p = CppAD::CondExpGt(p, Type(1)-eps, Type(1)-eps, p); // clamp high
  return log(p / (Type(1) - p));
}

// Smooth positive-part via softplus: ~ max(0,x) but smooth and AD-safe
template<class Type>
Type softplus(Type x, Type k) {
  // Stable softplus: (1/k)*log(1 + exp(k*x)) with branch to avoid overflow
  Type y = k * x;
  if (y > Type(0)) {
    // y positive: exp(-y) is small; x + log(1 + exp(-y))/k is stable
    return x + log(Type(1) + exp(-y)) / k;
  } else {
    // y negative: exp(y) is small; log(1 + exp(y))/k is stable
    return log(Type(1) + exp(y)) / k;
  }
}

// Smooth box-penalty to softly keep parameter within [lower, upper]
template<class Type>
Type penalty_box(Type x, Type lower, Type upper, Type k, Type w) {
  // k controls softness, w controls penalty weight
  Type below = softplus(lower - x, k); // >0 if x<lower
  Type above = softplus(x - upper, k); // >0 if x>upper
  return w * (below*below + above*above);
}

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;

  // -------------------------
  // DATA (READ ONLY)
  // -------------------------
  DATA_VECTOR(Year);         // Calendar year (integer), used for alignment/reporting
  DATA_VECTOR(cots_dat);     // Adult COTS density (ind m^-2), strictly positive
  DATA_VECTOR(fast_dat);     // Fast coral cover (%), in [0,100]
  DATA_VECTOR(slow_dat);     // Slow coral cover (%), in [0,100]
  DATA_VECTOR(sst_dat);      // Sea-surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);  // Larval immigration (ind m^-2 yr^-1)

  int T = Year.size(); // Number of time steps (years)

  // -------------------------
  // PARAMETERS (ESTIMATED)
  // -------------------------
  // Coral intrinsic growth (year^-1), fast and slow groups
  PARAMETER(log_r_f);     // log of intrinsic growth rate of fast coral (yr^-1); literature 0.2-1.0
  PARAMETER(log_r_s);     // log of intrinsic growth rate of slow coral (yr^-1); literature 0.05-0.5

  // Interspecific competition coefficients (dimensionless)
  PARAMETER(alpha_fs);    // Effect of slow coral on fast coral's carrying capacity; 0..1 typical
  PARAMETER(alpha_sf);    // Effect of fast coral on slow coral's carrying capacity; 0..1 typical

  // COTS predation functional response parameters
  PARAMETER(log_a_f);     // log attack/clearance rate on fast coral per predator per year
  PARAMETER(log_a_s);     // log attack/clearance rate on slow coral per predator per year
  PARAMETER(log_h);       // log handling-time-like saturation parameter (year)

  // COTS reproduction/recruitment
  PARAMETER(log_b0);      // log baseline fecundity-to-adult recruitment per adult (yr^-1)
  PARAMETER(log_eta_rec); // log recruitment efficiency scaling (dimensionless)
  PARAMETER(log_kA);      // log slope of Allee function (per (ind m^-2))
  PARAMETER(C_Allee);     // Allee threshold for mate-finding (ind m^-2)

  // COTS mortality components
  PARAMETER(log_m0);      // log baseline adult mortality rate (yr^-1)
  PARAMETER(log_mD);      // log density-dependent (disease/crowding) mortality coefficient
  PARAMETER(log_kd);      // log half-saturation for density mortality (ind m^-2)
  PARAMETER(log_mT);      // log temperature-linked excess mortality coefficient

  // Temperature effects
  PARAMETER(T_opt);         // COTS thermal optimum (°C)
  PARAMETER(log_sigma_T);   // log width of thermal performance curve (°C)
  PARAMETER(T_bleach);      // Coral bleaching soft-threshold temperature (°C)
  PARAMETER(log_deltaT);    // log softness (°C) for bleaching threshold transition

  // Coral bleaching sensitivities
  PARAMETER(log_bf);      // log bleaching sensitivity of fast coral (yr^-1 per unit soft-excess)
  PARAMETER(log_bs);      // log bleaching sensitivity of slow coral (yr^-1 per unit soft-excess)

  // Food dependence of COTS reproduction
  PARAMETER(logit_K_food);   // logit half-saturation of food index (proportion coral)
  PARAMETER(log_gamma_food); // log exponent shaping food effect (dimensionless)
  PARAMETER(logit_w_f);      // logit preference weight for fast coral in food index
  // Immigration efficiency
  PARAMETER(logit_phi_imm);  // logit fraction of larval immigration that becomes adults next year

  // Initial states
  PARAMETER(log_C0);       // log initial adult COTS density (ind m^-2)
  PARAMETER(logit_f0);     // logit initial fast coral proportion (0-1)
  PARAMETER(logit_s0);     // logit initial slow coral proportion (0-1)

  // Observation error (log-scales for SDs)
  PARAMETER(log_sigma_cots); // log SD for lognormal COTS observation error
  PARAMETER(log_sigma_fast); // log SD for logit-normal fast coral observation error
  PARAMETER(log_sigma_slow); // log SD for logit-normal slow coral observation error

  // -------------------------
  // TRANSFORMED PARAMETERS / CONSTANTS
  // -------------------------
  Type eps = Type(1e-8); // numerical floor to avoid division by zero

  // Coral growth rates
  Type r_f = exp(log_r_f); // yr^-1
  Type r_s = exp(log_r_s); // yr^-1

  // Predation FR params
  Type a_f = exp(log_a_f); // per predator per year
  Type a_s = exp(log_a_s); // per predator per year
  Type h   = exp(log_h);   // year

  // Reproduction and Allee
  Type b0      = exp(log_b0);       // yr^-1
  Type eta_rec = exp(log_eta_rec);  // dimensionless
  Type kA      = exp(log_kA);       // (ind m^-2)^-1

  // Mortality
  Type m0  = exp(log_m0);  // yr^-1
  Type mD  = exp(log_mD);  // yr^-1
  Type kd  = exp(log_kd);  // ind m^-2
  Type mT  = exp(log_mT);  // yr^-1

  // Temperature
  Type sigma_T  = exp(log_sigma_T); // °C
  Type deltaT   = exp(log_deltaT);  // °C softness

  // Bleaching sensitivity
  Type bf = exp(log_bf); // yr^-1
  Type bs = exp(log_bs); // yr^-1

  // Food effect
  Type K_food     = invlogit_stable(logit_K_food);  // proportion (0-1)
  Type gamma_food = exp(log_gamma_food);            // dimensionless
  Type w_f        = invlogit_stable(logit_w_f);     // preference weight on fast coral
  Type w_s        = Type(1.0) - w_f;                // preference weight on slow coral

  // Immigration efficiency
  Type phi_imm = invlogit_stable(logit_phi_imm);    // fraction (0-1)

  // Observation SD floors (ensure strictly positive)
  Type sd_floor_log = Type(0.05);   // floor on SD on log/logit scales
  Type sd_cots  = sqrt(exp(2.0*log_sigma_cots) + sd_floor_log*sd_floor_log);
  Type sd_fast  = sqrt(exp(2.0*log_sigma_fast) + sd_floor_log*sd_floor_log);
  Type sd_slow  = sqrt(exp(2.0*log_sigma_slow) + sd_floor_log*sd_floor_log);

  // -------------------------
  // STATE TRAJECTORIES
  // -------------------------
  vector<Type> cots_pred(T); // predicted COTS density (ind m^-2)
  vector<Type> fast_pred(T); // predicted fast coral cover (%)
  vector<Type> slow_pred(T); // predicted slow coral cover (%)

  // Auxiliary reports
  vector<Type> food_index(T);     // weighted coral proportion
  vector<Type> temp_perf(T);      // COTS thermal performance (0-1)
  vector<Type> bleach_excess(T);  // unitless soft exceedance of bleaching threshold

  // Initialize states (year 0) from parameters only (no data leakage)
  Type C = exp(log_C0);                              // ind m^-2
  Type f = invlogit_stable(logit_f0);                // proportion (0-1)
  Type s = invlogit_stable(logit_s0);                // proportion (0-1)
  cots_pred(0) = C;
  fast_pred(0) = (f * Type(100.0));
  slow_pred(0) = (s * Type(100.0));

  // Initialize auxiliaries for t=0 using the first-year SST/forcings (covariates are allowed)
  food_index(0) = w_f * f + w_s * s;
  temp_perf(0)  = exp(-Type(0.5) * pow((sst_dat(0) - T_opt) / (sigma_T + eps), 2.0)); // 0..1
  bleach_excess(0) = softplus((sst_dat(0) - T_bleach) / (deltaT + eps), Type(5.0));   // unitless

  // -------------------------
  // PROCESS MODEL (Deterministic)
  // Equations (year t -> t+1) using only lagged states and covariates:
  // 1) Food index: W_t = w_f * f_t + w_s * s_t
  // 2) COTS temperature performance: g_T,t = exp(-0.5 * ((SST_t - T_opt)/sigma_T)^2)
  // 3) COTS reproduction modifier (food): g_F,t = (W_t / (K_food + W_t))^{gamma_food}
  // 4) COTS Allee function: A(C_t) = 1 / (1 + exp(-kA * (C_t - C_Allee)))
  // 5) COTS survival: S_t = exp( -m0 - mT * (1 - g_T,t) - mD * C_t / (kd + C_t) )
  // 6) COTS recruits: R_t = eta_rec * b0 * C_t * A(C_t) * g_F,t * g_T,t
  // 7) COTS update: C_{t+1} = C_t * S_t + R_t + phi_imm * immigration_t
  // 8) Coral predation (Holling II multi-prey):
  //    Cons_f,t = C_t * (a_f * f_t) / (1 + h * (a_f * f_t + a_s * s_t))
  //    Cons_s,t = C_t * (a_s * s_t) / (1 + h * (a_f * f_t + a_s * s_t))
  // 9) Coral bleaching loss (soft-threshold on SST):
  //    B_excess_t = softplus((SST_t - T_bleach)/deltaT, k=5)
  //    Bleach_f,t = bf * B_excess_t * f_t; Bleach_s,t = bs * B_excess_t * s_t
  // 10) Coral logistic growth with competition:
  //    Growth_f,t = r_f * f_t * max_s(0, 1 - f_t - alpha_fs * s_t)
  //    Growth_s,t = r_s * s_t * max_s(0, 1 - s_t - alpha_sf * f_t)
  // 11) Coral updates:
  //    f_{t+1} = f_t + Growth_f,t - Cons_f,t - Bleach_f,t
  //    s_{t+1} = s_t + Growth_s,t - Cons_s,t - Bleach_s,t
  // -------------------------
  for (int t = 1; t < T; t++) {
    // Drivers from previous time step (t-1) to avoid data leakage from current observations
    Type sst = sst_dat(t-1);              // °C
    Type imm = cotsimm_dat(t-1);          // ind m^-2 yr^-1

    // (1) Food index
    Type W = w_f * f + w_s * s;                                     // proportion edible coral
    // (2) Temperature performance for COTS (0..1)
    Type gT = exp(-Type(0.5) * pow((sst - T_opt) / (sigma_T + eps), 2.0));
    // (3) Food effect on reproduction (0..1 with shape)
    Type gF = pow(W / (K_food + W + eps), gamma_food);
    // (4) Allee effect (0..1)
    Type A = Type(1.0) / (Type(1.0) + exp(-kA * (C - C_Allee)));
    // (5) Survival (fraction [0,1] if parameters are reasonable)
    Type densMort = mD * (C / (kd + C + eps));                       // density-dependent mortality
    Type S_cots   = exp(-(m0 + mT * (Type(1.0) - gT) + densMort));   // survival fraction
    // (6) Recruits (ind m^-2 yr^-1)
    Type R = eta_rec * b0 * C * A * gF * gT;
    // (7) Update COTS
    Type C_next = C * S_cots + R + phi_imm * imm;
    C_next = CppAD::CondExpLt(C_next, eps, eps, C_next);             // positivity floor
    // (8) Predation on corals (Holling II multi-prey)
    Type denom = Type(1.0) + h * (a_f * f + a_s * s) + eps;
    Type Cons_f = C * (a_f * f) / denom;                             // proportion per year
    Type Cons_s = C * (a_s * s) / denom;                             // proportion per year
    // (9) Bleaching (soft threshold)
    Type B_excess = softplus((sst - T_bleach) / (deltaT + eps), Type(5.0)); // unitless
    Type Bleach_f = bf * B_excess * f;                               // proportion per year
    Type Bleach_s = bs * B_excess * s;                               // proportion per year
    // (10) Coral growth with smooth positive-part for free space
    Type Growth_f = r_f * f * softplus(Type(1.0) - f - alpha_fs * s, Type(10.0));
    Type Growth_s = r_s * s * softplus(Type(1.0) - s - alpha_sf * f, Type(10.0));
    // (11) Update corals
    Type f_next = f + Growth_f - Cons_f - Bleach_f;
    Type s_next = s + Growth_s - Cons_s - Bleach_s;
    // Apply soft floors/ceilings to keep proportions in (0,1)
    f_next = CppAD::CondExpLt(f_next, eps, eps, f_next);
    s_next = CppAD::CondExpLt(s_next, eps, eps, s_next);
    f_next = CppAD::CondExpGt(f_next, Type(1.0)-eps, Type(1.0)-eps, f_next);
    s_next = CppAD::CondExpGt(s_next, Type(1.0)-eps, Type(1.0)-eps, s_next);

    // Commit updates for next iteration
    C = C_next;
    f = f_next;
    s = s_next;

    // Store predictions (match time axis; do not use observations)
    cots_pred(t) = C;
    fast_pred(t) = f * Type(100.0);
    slow_pred(t) = s * Type(100.0);

    // Store auxiliaries (for reporting/diagnostics)
    food_index(t) = W;
    temp_perf(t)  = gT;
    bleach_excess(t) = B_excess;
  }

  // -------------------------
  // LIKELIHOOD
  // -------------------------
  Type nll = Type(0.0);

  for (int t = 0; t < T; t++) {
    // COTS (lognormal)
    Type y_c = cots_dat(t);
    Type mu_c = cots_pred(t);
    // Ensure positivity in transforms
    nll -= dnorm(log(y_c + eps), log(mu_c + eps), sd_cots, true);

    // Fast coral (logit-normal on proportions)
    Type y_f = (fast_dat(t) / Type(100.0));
    Type mu_f = (fast_pred(t) / Type(100.0));
    nll -= dnorm(logit_stable(y_f), logit_stable(mu_f), sd_fast, true);

    // Slow coral (logit-normal)
    Type y_s = (slow_dat(t) / Type(100.0));
    Type mu_s = (slow_pred(t) / Type(100.0));
    nll -= dnorm(logit_stable(y_s), logit_stable(mu_s), sd_slow, true);
  }

  // -------------------------
  // SOFT BIOLOGICAL BOUNDS (penalties)
  // -------------------------
  Type pen = Type(0.0);
  // Temperature bounds (°C)
  pen += penalty_box(T_opt,    Type(24.0), Type(31.5), Type(5.0), Type(1.0));
  pen += penalty_box(T_bleach, Type(28.0), Type(32.5), Type(5.0), Type(1.0));
  // Thermal width (°C)
  pen += penalty_box(sigma_T,  Type(0.5),  Type(5.0),  Type(5.0), Type(1.0));
  // Coral growth rates (yr^-1)
  pen += penalty_box(r_f, Type(0.05), Type(2.0), Type(5.0), Type(1.0));
  pen += penalty_box(r_s, Type(0.01), Type(1.0), Type(5.0), Type(1.0));
  // Predation rates and handling
  pen += penalty_box(a_f, Type(1e-4), Type(10.0), Type(5.0), Type(1.0));
  pen += penalty_box(a_s, Type(1e-4), Type(10.0), Type(5.0), Type(1.0));
  pen += penalty_box(h,   Type(1e-3), Type(10.0), Type(5.0), Type(1.0));
  // Mortality rates (yr^-1)
  pen += penalty_box(m0, Type(0.01), Type(2.0), Type(5.0), Type(1.0));
  pen += penalty_box(mD, Type(0.0),  Type(5.0), Type(5.0), Type(1.0));
  pen += penalty_box(kd, Type(0.01), Type(10.0),Type(5.0), Type(1.0));
  pen += penalty_box(mT, Type(0.0),  Type(2.0), Type(5.0), Type(1.0));
  // Bleaching sensitivity (yr^-1)
  pen += penalty_box(bf, Type(0.0),  Type(2.0), Type(5.0), Type(1.0));
  pen += penalty_box(bs, Type(0.0),  Type(2.0), Type(5.0), Type(1.0));
  // Food half-saturation (proportion)
  pen += penalty_box(K_food, Type(0.01), Type(0.7), Type(5.0), Type(1.0));
  // Preference and immigration efficiency already constrained via logit transforms

  nll += pen;

  // -------------------------
  // REPORTING
  // -------------------------
  REPORT(Year);             // time axis
  REPORT(cots_pred);        // predicted COTS density (ind m^-2)
  REPORT(fast_pred);        // predicted fast coral cover (%)
  REPORT(slow_pred);        // predicted slow coral cover (%)
  REPORT(food_index);       // weighted coral proportion (0-1)
  REPORT(temp_perf);        // COTS temperature performance (0-1)
  REPORT(bleach_excess);    // unitless soft exceedance

  // Also report key parameters on natural scales for interpretability
  REPORT(r_f);
  REPORT(r_s);
  REPORT(a_f);
  REPORT(a_s);
  REPORT(h);
  REPORT(b0);
  REPORT(eta_rec);
  REPORT(kA);
  REPORT(C_Allee);
  REPORT(m0);
  REPORT(mD);
  REPORT(kd);
  REPORT(mT);
  REPORT(T_opt);
  REPORT(sigma_T);
  REPORT(T_bleach);
  REPORT(deltaT);
  REPORT(bf);
  REPORT(bs);
  REPORT(K_food);
  REPORT(gamma_food);
  REPORT(w_f);
  REPORT(w_s);
  REPORT(phi_imm);

  return nll;
}
