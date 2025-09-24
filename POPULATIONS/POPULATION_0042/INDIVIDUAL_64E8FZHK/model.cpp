#include <TMB.hpp>

// Helper functions with numerical safety
template<class Type> Type invlogit_safe(Type x) {
  return Type(1) / (Type(1) + exp(-x));
}
template<class Type> Type logit_safe(Type p, Type eps) {
  // Clamp to (eps, 1-eps) to avoid infinities
  p = CppAD::CondExpLt(p, eps, eps, p);
  p = CppAD::CondExpGt(p, Type(1) - eps, Type(1) - eps, p);
  return log(p / (Type(1) - p));
}
template<class Type> Type smooth_relu(Type x, Type k) {
  // Smooth approximation to max(0, x) using softplus; k controls sharpness
  return log1p(exp(k * x)) / k;
}

// Numbered equation reference (see equations inline in loop):
// (1) Coral effective growth: r_g_eff = r_g * exp(-phi_g * (sst - sst_opt_g)^2)
// (2) Coral bleaching mortality (smooth threshold): M_bleach_g = m_bleach_max_g * logistic(k_bleach*(sst - bleach_thr_g))
// (3) Multi-prey Holling II feeding weights: denom = K_half + w_f*C_f + w_s*C_s
//     Per capita consumption on guild g: cons_g_pc = f_max * w_g * C_g / (denom + eps)
// (4) Predation hazard on coral: M_pred_g = alpha_pred * A * cons_g_pc
// (5) Coral update (multiplicative growth-survival): 
//     C_g(t+1) = C_g(t) * exp( r_g_eff*(1 - (C_f + C_s)/K_coral) - m_bg_g - M_bleach_g - M_pred_g )
// (6) Assimilated energy for reproduction: Energy = epsilon_assim * (kappa_f*Cons_f + kappa_s*Cons_s), Cons_g = A * cons_g_pc
// (7) Local recruitment (energy-limited, SST-modulated): 
//     R_loc = alpha_R * A * (Energy / (E_half + Energy)) * exp( -phi_larva * (sst - sst_opt_larva)^2 )
// (8) Exogenous recruitment from data: R_imm = imm_scale * cotsimm_dat
// (9) Beverton–Holt density dependence: R_tot = (R_loc + R_imm) / (1 + beta_R * A)
// (10) Adult survival: S_A = A * exp( - mA0 * exp(phi_cots*(sst - sst_opt_cots)^2) - mA_dd * A )
// (11) Adult update: A(t+1) = S_A + recruit_eff * R_tot
// (12) Observation models: 
//     log(cots_dat + eps) ~ Normal( log(cots_pred + eps), sd_cots )
//     logit(fast_dat/100) ~ Normal( logit(fast_pred/100), sd_fast )
//     logit(slow_dat/100) ~ Normal( logit(slow_pred/100), sd_slow )

template<class Type>
Type objective_function<Type>::operator() ()
{
  using CppAD::CondExpLt;
  using CppAD::CondExpGt;

  Type nll = Type(0);
  const Type eps = Type(1e-8);       // small constant to avoid division by zero
  const Type min_sd = Type(0.05);    // minimum observation SD on log/logit scales for stability

  // DATA INPUTS (vectors aligned by Year)
  DATA_VECTOR(Year);          // Year (calendar years, numeric)
  DATA_VECTOR(sst_dat);       // Sea-surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);   // Exogenous larval immigration (individuals/m^2/year)
  DATA_VECTOR(cots_dat);      // Observed adult COTS abundance (individuals/m^2)
  DATA_VECTOR(fast_dat);      // Observed fast coral cover (%)
  DATA_VECTOR(slow_dat);      // Observed slow coral cover (%)
  int n = Year.size();        // Number of time points

  // PARAMETERS (each line explains units and role)

  // Initial states
  PARAMETER(cots_init);            // Initial adult COTS density at Year[0] (individuals/m^2); start near observed
  PARAMETER(fast0_percent);        // Initial fast coral cover at Year[0] (%) to seed the state
  PARAMETER(slow0_percent);        // Initial slow coral cover at Year[0] (%) to seed the state

  // Coral growth and mortality
  PARAMETER(r_fast);               // (year^-1) intrinsic growth rate of fast coral; SST modifies via exp-quadratic
  PARAMETER(r_slow);               // (year^-1) intrinsic growth rate of slow coral; SST modifies via exp-quadratic
  PARAMETER(m_bg_fast);            // (year^-1) background mortality of fast coral
  PARAMETER(m_bg_slow);            // (year^-1) background mortality of slow coral
  PARAMETER(K_coral);              // (fraction 0–1) carrying capacity for total coral cover (fast + slow)

  // Feeding and predation on corals
  PARAMETER(w_fast);               // (unitless) preference weight for fast coral in feeding functional response
  PARAMETER(w_slow);               // (unitless) preference weight for slow coral in feeding functional response
  PARAMETER(f_max);                // (fraction/year) maximum per-predator consumption rate on coral cover
  PARAMETER(K_half);               // (fraction) half-saturation constant for multi-prey Holling II
  PARAMETER(alpha_pred);           // (year^-1 per unit consumption) scales consumption to coral predation hazard

  // Energy assimilation to reproduction
  PARAMETER(epsilon_assim);        // (unitless 0–1) assimilation efficiency from consumption to reproductive energy
  PARAMETER(kappa_fast);           // (unitless) energy quality multiplier for fast coral
  PARAMETER(kappa_slow);           // (unitless) energy quality multiplier for slow coral

  // Recruitment and maturation
  PARAMETER(alpha_R);              // (recruits per adult per year) local recruitment coefficient at ideal conditions
  PARAMETER(beta_R);               // (m^2/individual) Beverton–Holt density-dependence coefficient
  PARAMETER(E_half);               // (energy units) half-saturation for energy-limited reproduction
  PARAMETER(imm_scale);            // (adults-equivalent per (individual/m^2/year)) scaling of exogenous immigration
  PARAMETER(recruit_eff);          // (unitless 0–1) fraction of recruits maturing to adults within a year

  // Adult mortality (SST and density dependent)
  PARAMETER(mA0);                  // (year^-1) baseline adult mortality rate
  PARAMETER(mA_dd);                // (year^-1 per (individual/m^2)) density-dependent adult mortality

  // Temperature modifiers
  PARAMETER(sst_opt_fast);         // (°C) SST optimum for fast coral growth
  PARAMETER(sst_opt_slow);         // (°C) SST optimum for slow coral growth
  PARAMETER(sst_opt_cots);         // (°C) SST optimum for adult survival
  PARAMETER(sst_opt_larva);        // (°C) SST optimum for larval/early survival
  PARAMETER(phi_fast);             // (per (°C)^2) curvature for fast coral growth SST modifier
  PARAMETER(phi_slow);             // (per (°C)^2) curvature for slow coral growth SST modifier
  PARAMETER(phi_cots);             // (per (°C)^2) curvature for adult mortality SST modifier
  PARAMETER(phi_larva);            // (per (°C)^2) curvature for larval survival SST modifier

  // Bleaching parameters (smooth threshold)
  PARAMETER(bleach_thr_fast);      // (°C) SST threshold for fast coral bleaching (logistic ramp center)
  PARAMETER(bleach_thr_slow);      // (°C) SST threshold for slow coral bleaching (logistic ramp center)
  PARAMETER(m_bleach_max_fast);    // (year^-1) maximum additional mortality for fast coral at high SST
  PARAMETER(m_bleach_max_slow);    // (year^-1) maximum additional mortality for slow coral at high SST
  PARAMETER(k_bleach);             // (unitless) steepness of bleaching logistic ramp

  // Observation error parameters (log/logit scales)
  PARAMETER(log_sd_cots);          // log(SD) for log COTS abundance observation model
  PARAMETER(log_sd_fast);          // log(SD) for logit fast coral proportion observation model
  PARAMETER(log_sd_slow);          // log(SD) for logit slow coral proportion observation model

  // Smooth bound penalty controls
  PARAMETER(penalty_weight);       // (unitless) weight applied to soft bound penalties
  PARAMETER(k_bounds);             // (unitless) softness of the penalty (larger = sharper transition)

  // Soft penalties to keep parameters within biologically reasonable ranges (no hard constraints)
  auto add_bound_penalty = [&](Type x, Type lb, Type ub) {
    Type pen_low  = smooth_relu(lb - x, k_bounds);
    Type pen_high = smooth_relu(x - ub, k_bounds);
    nll += penalty_weight * (pen_low * pen_low + pen_high * pen_high);
  };

  // Apply penalties (bounds aligned with parameters.json suggestions)
  add_bound_penalty(r_fast,  Type(0.0),  Type(2.0));
  add_bound_penalty(r_slow,  Type(0.0),  Type(2.0));
  add_bound_penalty(m_bg_fast, Type(0.0), Type(1.0));
  add_bound_penalty(m_bg_slow, Type(0.0), Type(1.0));
  add_bound_penalty(K_coral, Type(0.2),  Type(1.0));
  add_bound_penalty(w_fast,  Type(0.0),  Type(10.0));
  add_bound_penalty(w_slow,  Type(0.0),  Type(10.0));
  add_bound_penalty(f_max,   Type(0.0),  Type(10.0));
  add_bound_penalty(K_half,  Type(0.001),Type(1.0));
  add_bound_penalty(alpha_pred, Type(0.0), Type(10.0));
  add_bound_penalty(epsilon_assim, Type(0.0), Type(1.0));
  add_bound_penalty(kappa_fast, Type(0.0), Type(2.0));
  add_bound_penalty(kappa_slow, Type(0.0), Type(2.0));
  add_bound_penalty(alpha_R, Type(0.0),   Type(100.0));
  add_bound_penalty(beta_R,  Type(0.0),   Type(10.0));
  add_bound_penalty(E_half,  Type(0.0001),Type(10.0));
  add_bound_penalty(imm_scale, Type(0.0), Type(10.0));
  add_bound_penalty(recruit_eff, Type(0.0), Type(1.0));
  add_bound_penalty(mA0,     Type(0.0),   Type(2.0));
  add_bound_penalty(mA_dd,   Type(0.0),   Type(5.0));
  add_bound_penalty(sst_opt_fast, Type(24.0), Type(32.0));
  add_bound_penalty(sst_opt_slow, Type(24.0), Type(32.0));
  add_bound_penalty(sst_opt_cots, Type(24.0), Type(32.0));
  add_bound_penalty(sst_opt_larva,Type(24.0), Type(32.0));
  add_bound_penalty(phi_fast,  Type(0.0), Type(2.0));
  add_bound_penalty(phi_slow,  Type(0.0), Type(2.0));
  add_bound_penalty(phi_cots,  Type(0.0), Type(2.0));
  add_bound_penalty(phi_larva, Type(0.0), Type(2.0));
  add_bound_penalty(bleach_thr_fast, Type(27.0), Type(33.0));
  add_bound_penalty(bleach_thr_slow, Type(27.0), Type(33.0));
  add_bound_penalty(m_bleach_max_fast, Type(0.0), Type(2.0));
  add_bound_penalty(m_bleach_max_slow, Type(0.0), Type(2.0));
  add_bound_penalty(k_bleach, Type(0.1), Type(20.0));
  add_bound_penalty(cots_init, Type(0.0), Type(10.0));
  add_bound_penalty(fast0_percent, Type(0.0), Type(80.0));
  add_bound_penalty(slow0_percent, Type(0.0), Type(80.0));
  // log SDs are left relatively unbounded; small soft bounds to avoid extreme values
  add_bound_penalty(log_sd_cots, Type(-5.0), Type(2.0));
  add_bound_penalty(log_sd_fast, Type(-5.0), Type(2.0));
  add_bound_penalty(log_sd_slow, Type(-5.0), Type(2.0));

  // Derived observation SDs with minimum floor
  Type sd_cots = min_sd + exp(log_sd_cots);
  Type sd_fast = min_sd + exp(log_sd_fast);
  Type sd_slow = min_sd + exp(log_sd_slow);

  // STATE TRAJECTORIES AND PREDICTIONS (always based on previous state; no data leakage)
  vector<Type> cots_pred(n);  // predicted adult COTS (individuals/m^2)
  vector<Type> fast_pred(n);  // predicted fast coral cover (%)
  vector<Type> slow_pred(n);  // predicted slow coral cover (%)

  // Initialize state at t=0 from parameters
  Type A = cots_init;                          // Adults (ind/m^2)
  Type Cf = fast0_percent / Type(100.0);       // Fast coral (fraction 0–1)
  Type Cs = slow0_percent / Type(100.0);       // Slow coral (fraction 0–1)

  for (int t = 0; t < n; ++t) {
    // 1) Output predictions for current time (aligned to observations at Year[t])
    cots_pred(t) = A;                  // indiv/m^2
    fast_pred(t) = Cf * Type(100.0);   // %
    slow_pred(t) = Cs * Type(100.0);   // %

    // 2) Environmental modifiers for transitions to next time step
    Type sst = sst_dat(t);   // SST forcing (°C)

    // Coral: growth rates modified by SST via symmetric exp-quadratic around optimum
    Type r_f_eff = r_fast * exp(-phi_fast * (sst - sst_opt_fast) * (sst - sst_opt_fast));  // (1)
    Type r_s_eff = r_slow * exp(-phi_slow * (sst - sst_opt_slow) * (sst - sst_opt_slow));  // (1)

    // Coral bleaching mortality as smooth logistic ramp around threshold (2)
    Type ramp_fast = invlogit_safe(k_bleach * (sst - bleach_thr_fast));
    Type ramp_slow = invlogit_safe(k_bleach * (sst - bleach_thr_slow));
    Type M_bleach_f = m_bleach_max_fast * ramp_fast;
    Type M_bleach_s = m_bleach_max_slow * ramp_slow;

    // Multi-prey Holling type-II functional response (3)
    Type denom = K_half + w_fast * Cf + w_slow * Cs + eps;
    Type cons_f_pc = f_max * w_fast * Cf / denom;  // per-capita consumption on fast coral
    Type cons_s_pc = f_max * w_slow * Cs / denom;  // per-capita consumption on slow coral

    // Predation hazard to corals (4)
    Type M_pred_f = alpha_pred * A * cons_f_pc;
    Type M_pred_s = alpha_pred * A * cons_s_pc;

    // Coral state updates (5) using multiplicative growth-survival (keeps positivity, smooth)
    Type crowding = (Cf + Cs) / (K_coral + eps);  // crowding factor relative to capacity
    Type Cf_next = Cf * exp(r_f_eff * (Type(1) - crowding) - m_bg_fast - M_bleach_f - M_pred_f);
    Type Cs_next = Cs * exp(r_s_eff * (Type(1) - crowding) - m_bg_slow - M_bleach_s - M_pred_s);

    // Assimilated energy for COTS reproduction (6)
    Type Cons_f = A * cons_f_pc;
    Type Cons_s = A * cons_s_pc;
    Type Energy = epsilon_assim * (kappa_fast * Cons_f + kappa_slow * Cons_s);

    // Local recruitment (energy-limited; SST-modified larval survival) (7)
    Type sst_larv_mod = exp(-phi_larva * (sst - sst_opt_larva) * (sst - sst_opt_larva));
    Type R_loc = alpha_R * A * (Energy / (E_half + Energy + eps)) * sst_larv_mod;

    // Exogenous immigration (for outbreak triggers) (8)
    Type R_imm = imm_scale * cotsimm_dat(t);

    // Density dependence in early stages (Beverton–Holt) (9)
    Type R_tot = (R_loc + R_imm) / (Type(1) + beta_R * A);

    // Adult survival (SST- and density-dependent) (10)
    Type M_A = mA0 * exp(phi_cots * (sst - sst_opt_cots) * (sst - sst_opt_cots)) + mA_dd * A;
    Type S_A = A * exp(-M_A);

    // Adult update with maturation of recruits (11)
    Type A_next = S_A + recruit_eff * R_tot;

    // Prepare for next iteration
    A = CppAD::CondExpLt(A_next, eps, eps, A_next);  // keep positive smoothly
    Cf = CppAD::CondExpLt(Cf_next, eps, eps, Cf_next);
    Cs = CppAD::CondExpLt(Cs_next, eps, eps, Cs_next);
  }

  // LIKELIHOOD: include all observations; use appropriate error families; protect with eps and min SDs (12)
  for (int t = 0; t < n; ++t) {
    // COTS (lognormal on strictly positive)
    Type y_c = cots_dat(t) + eps;
    Type mu_c = cots_pred(t) + eps;
    nll -= dnorm(log(y_c), log(mu_c), sd_cots, true);

    // Fast coral cover (% -> proportion in (0,1), logit-normal)
    Type y_f = fast_dat(t) / Type(100.0);
    Type mu_f = fast_pred(t) / Type(100.0);
    // Clamp to avoid logit problems
    y_f = CppAD::CondExpLt(y_f, eps, eps, y_f);
    y_f = CppAD::CondExpGt(y_f, Type(1) - eps, Type(1) - eps, y_f);
    mu_f = CppAD::CondExpLt(mu_f, eps, eps, mu_f);
    mu_f = CppAD::CondExpGt(mu_f, Type(1) - eps, Type(1) - eps, mu_f);
    nll -= dnorm(log(y_f / (Type(1) - y_f)), log(mu_f / (Type(1) - mu_f)), sd_fast, true);

    // Slow coral cover
    Type y_s = slow_dat(t) / Type(100.0);
    Type mu_s = slow_pred(t) / Type(100.0);
    y_s = CppAD::CondExpLt(y_s, eps, eps, y_s);
    y_s = CppAD::CondExpGt(y_s, Type(1) - eps, Type(1) - eps, y_s);
    mu_s = CppAD::CondExpLt(mu_s, eps, eps, mu_s);
    mu_s = CppAD::CondExpGt(mu_s, Type(1) - eps, Type(1) - eps, mu_s);
    nll -= dnorm(log(y_s / (Type(1) - y_s)), log(mu_s / (Type(1) - mu_s)), sd_slow, true);
  }

  // REPORT predictions to R
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}
