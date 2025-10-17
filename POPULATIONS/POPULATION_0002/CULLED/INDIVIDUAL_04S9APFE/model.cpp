#include <TMB.hpp>

// Helper functions
template<class Type>
Type softplus(Type x) { // Smooth positive-part; prevents hard cutoffs; CppAD-safe
  // Stable implementation: softplus(x) = log(1 + exp(-|x|)) + max(x, 0)
  Type zero = Type(0);
  Type pos = CppAD::CondExpGt(x, zero, x, zero);   // max(x, 0)
  Type absx = CppAD::CondExpGe(x, zero, x, -x);    // |x|
  return log(exp(-absx) + Type(1.0)) + pos;
}

template<class Type>
Type inv_logit(Type x) { // Logistic inverse
  return Type(1) / (Type(1) + exp(-x));
}

template<class Type>
Type logit01(Type p, Type eps) { // Stable logit in (eps,1-eps)
  p = CppAD::CondExpGt(p, Type(1)-eps, Type(1)-eps, p);
  p = CppAD::CondExpLt(p, eps, eps, p);
  return log(p/(Type(1)-p));
}

// Smooth penalty for parameter bounds (soft, not hard constraints)
template<class Type>
void add_bound_penalty(Type &nll, Type x, bool use_lower, Type lower, bool use_upper, Type upper, Type weight) {
  if (use_lower) {
    // Penalize x < lower with softplus(lower - x)
    nll += weight * softplus(lower - x);
  }
  if (use_upper) {
    // Penalize x > upper with softplus(x - upper)
    nll += weight * softplus(x - upper);
  }
}

template<class Type>
Type square(Type x) { return x*x; }

template<class Type>
Type max_eps(Type x, Type eps) { // smooth-ish lower bound via softplus shift
  // Ensures strictly positive result without hard cutoff
  return eps + softplus(x - eps);
}

template<class Type>
Type min_sd_floor(Type sd, Type floor_val) { // Enforce minimum SD smoothly
  return floor_val + softplus(sd - floor_val);
}

template<class Type>
Type tpc_gaussian(Type T, Type Topt, Type sigmaT) { // Thermal performance curve (0..1)
  Type z = (T - Topt) / (sigmaT + Type(1e-8));
  return exp(-Type(0.5) * z * z);
}

template<class Type>
Type positive_part_soft(Type x) { // Smooth positive part
  return Type(0.5) * (x + sqrt(x*x + Type(1e-8)));
}

template<class Type>
Type typeIII_FR(Type C, Type H) { // Type III functional response in [0,1]
  Type C2 = C*C;
  Type H2 = H*H + Type(1e-8);
  return C2 / (H2 + C2 + Type(1e-8));
}

template<class Type>
Type saturating01(Type x, Type K) { // Saturates to [0,1] as x increases
  return x / (K + x + Type(1e-8));
}

template<class Type>
Type soft_bleach(Type T, Type T_thresh, Type k) { // Smooth threshold for bleaching
  // Returns a non-negative stress factor increasing with T - T_thresh
  return softplus(k * (T - T_thresh)) / (k + Type(1e-8));
}

template<class Type>
Type objective_function<Type>::operator() ()
{
  Type nll = 0.0;                            // Negative log-likelihood accumulator
  const Type eps = Type(1e-8);               // Small constant for numerical stability
  const Type penalty_w = Type(1.0);          // Weight for soft parameter bound penalties
  const Type minSD = Type(1e-3);             // Minimum SD to avoid degeneracy

  // DATA INPUTS (match column names exactly)
  DATA_VECTOR(Year);                         // Year (calendar year)
  DATA_VECTOR(cots_dat);                     // Adult COTS density (indiv m^-2)
  DATA_VECTOR(fast_dat);                     // Fast coral cover (Acropora) (% of area)
  DATA_VECTOR(slow_dat);                     // Slow coral cover (Faviidae/Porites) (% of area)
  DATA_VECTOR(sst_dat);                      // Sea-surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);                  // COTS larval immigration (indiv m^-2 year^-1)

  int n = cots_dat.size();                   // Number of time steps (years)

  // PARAMETERS (all unconstrained; soft penalties enforce biological ranges)
  PARAMETER(r_A);            // year^-1 | Acropora intrinsic regrowth rate
  PARAMETER(r_S);            // year^-1 | Massive coral intrinsic regrowth rate
  PARAMETER(m_A);            // year^-1 | Acropora background mortality (non-predation, non-bleaching)
  PARAMETER(m_S);            // year^-1 | Massive coral background mortality
  PARAMETER(b_A);            // year^-1 | Bleaching sensitivity multiplier for Acropora mortality under heat stress
  PARAMETER(b_S);            // year^-1 | Bleaching sensitivity multiplier for massive coral mortality under heat stress
  PARAMETER(T_bleach);       // °C | Onset temperature for thermal stress (soft threshold)
  PARAMETER(k_bleach);       // 1/°C | Softness of bleaching threshold (larger -> sharper)
  PARAMETER(c_attack_A);     // dimensionless | Preference weight for allocating feeding to Acropora
  PARAMETER(c_attack_S);     // dimensionless | Preference weight for allocating feeding to massive corals
  PARAMETER(f_max);          // proportion indiv^-1 year^-1 | Max per-capita coral consumption (per COTS per year)
  PARAMETER(eta_switch);     // dimensionless (>=1) | Prey-switching exponent
  PARAMETER(H_half);         // indiv m^-2 | Half-saturation density for Type-III predation
  PARAMETER(r_C);            // year^-1 | COTS intrinsic growth (fecundity to adult recruitment potential)
  PARAMETER(m_C);            // year^-1 | COTS baseline mortality
  PARAMETER(starvation_scale); // dimensionless | Mortality multiplier when resources scarce
  PARAMETER(K_C);            // indiv m^-2 | Baseline COTS carrying capacity scaling factor
  PARAMETER(phi_A);          // dimensionless | Weight of Acropora in COTS food/carrying capacity
  PARAMETER(phi_S);          // dimensionless | Weight of massive corals in COTS food/carrying capacity
  PARAMETER(K_food);         // proportion | Half-saturation for resource-driven recruitment/capacity
  PARAMETER(Topt_C);         // °C | Optimal temperature for COTS reproduction
  PARAMETER(sigma_T_C);      // °C | Breadth of thermal performance curve for COTS reproduction
  PARAMETER(Alee_C);         // indiv m^-2 | Allee parameter for COTS (smooth low-density limitation)
  PARAMETER(gamma_A);        // indiv m^-2 year^-1 | Conversion from Acropora consumption to COTS recruits
  PARAMETER(gamma_S);        // indiv m^-2 year^-1 | Conversion from massive coral consumption to COTS recruits
  PARAMETER(imm_surv);       // dimensionless | Fraction of larval immigration surviving to adult class per year
  PARAMETER(K_tot);          // proportion (0..1) | Maximum combined cover of modeled coral groups
  // Observation model SDs
  PARAMETER(obs_sd_cots_ln);     // log-scale SD for COTS observations
  PARAMETER(obs_sd_fast);        // SD for Acropora cover (%)
  PARAMETER(obs_sd_slow);        // SD for massive coral cover (%)

  // SOFT PARAMETER BOUNDS (smooth penalties; proposed biological ranges)
  add_bound_penalty(nll, r_A, true, Type(0.0), true, Type(2.0), penalty_w);
  add_bound_penalty(nll, r_S, true, Type(0.0), true, Type(2.0), penalty_w);
  add_bound_penalty(nll, m_A, true, Type(0.0), true, Type(1.0), penalty_w);
  add_bound_penalty(nll, m_S, true, Type(0.0), true, Type(1.0), penalty_w);
  add_bound_penalty(nll, b_A, true, Type(0.0), true, Type(2.0), penalty_w);
  add_bound_penalty(nll, b_S, true, Type(0.0), true, Type(2.0), penalty_w);
  add_bound_penalty(nll, T_bleach, true, Type(26.0), true, Type(32.5), penalty_w);
  add_bound_penalty(nll, k_bleach, true, Type(0.05), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, c_attack_A, true, Type(0.0), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, c_attack_S, true, Type(0.0), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, f_max, true, Type(0.01), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, eta_switch, true, Type(1.0), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, H_half, true, Type(0.05), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, r_C, true, Type(0.0), true, Type(5.0), penalty_w);
  // Widened upper bound for m_C to avoid penalizing updated literature value (~2.56)
  add_bound_penalty(nll, m_C, true, Type(0.0), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, starvation_scale, true, Type(0.0), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, K_C, true, Type(0.05), true, Type(10.0), penalty_w);
  add_bound_penalty(nll, phi_A, true, Type(0.0), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, phi_S, true, Type(0.0), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, K_food, true, Type(0.01), true, Type(2.0), penalty_w);
  add_bound_penalty(nll, Topt_C, true, Type(24.0), true, Type(32.0), penalty_w);
  add_bound_penalty(nll, sigma_T_C, true, Type(0.1), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, Alee_C, true, Type(0.0), true, Type(2.0), penalty_w);
  add_bound_penalty(nll, gamma_A, true, Type(0.0), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, gamma_S, true, Type(0.0), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, imm_surv, true, Type(0.0), true, Type(1.0), penalty_w);
  add_bound_penalty(nll, K_tot, true, Type(0.2), true, Type(1.0), penalty_w);
  add_bound_penalty(nll, obs_sd_cots_ln, true, Type(0.01), true, Type(2.0), penalty_w);
  add_bound_penalty(nll, obs_sd_fast, true, Type(0.01), true, Type(10.0), penalty_w);
  add_bound_penalty(nll, obs_sd_slow, true, Type(0.01), true, Type(10.0), penalty_w);

  // PREDICTION VECTORS (initialize with observed first values to set initial conditions)
  vector<Type> cots_pred(n);                  // indiv m^-2
  vector<Type> fast_pred(n);                  // %
  vector<Type> slow_pred(n);                  // %

  cots_pred(0) = cots_dat(0);                 // Initialize from data (no data leakage forward)
  fast_pred(0) = fast_dat(0);                 // Initialize from data
  slow_pred(0) = slow_dat(0);                 // Initialize from data

  // Loop over time for process model (use t-1 values only)
  for (int t = 1; t < n; t++) {
    // Previous state (predicted, not observed)
    Type C_prev = cots_pred(t-1);             // indiv m^-2
    Type A_prev = fast_pred(t-1);             // %
    Type S_prev = slow_pred(t-1);             // %

    // Convert coral cover to proportions for internal rates
    Type A_prop = A_prev / Type(100.0);       // proportion
    Type S_prop = S_prev / Type(100.0);       // proportion

    // External drivers at previous time
    Type T_prev = sst_dat(t-1);               // °C
    Type IMM_prev = cotsimm_dat(t-1);         // indiv m^-2 year^-1

    // Functional responses and modifiers
    Type H = typeIII_FR(C_prev, H_half);      // Type-III response in [0,1] (predation intensity)
    Type bleach_stress = soft_bleach(T_prev, T_bleach, k_bleach); // thermal stress factor >= 0
    Type food_avail = saturating01(phi_A * A_prop + phi_S * S_prop, K_food); // 0..1 food saturation
    Type K_eff = K_C * saturating01(phi_A * A_prop + phi_S * S_prop + eps, K_food); // indiv m^-2 carrying cap
    Type rC_allee = r_C * (C_prev / (C_prev + Alee_C + eps)); // smooth Allee effect 0..r_C
    Type tpc = tpc_gaussian(T_prev, Topt_C, sigma_T_C); // 0..1 thermal performance

    // Multi-prey, predator-density driven predation with prey-switching (bounded by availability)
    Type A_w = c_attack_A * pow(A_prop + eps, eta_switch);
    Type S_w = c_attack_S * pow(S_prop + eps, eta_switch);
    Type W_sum = A_w + S_w + eps;
    Type alloc_A = A_w / W_sum;
    Type alloc_S = S_w / W_sum;

    // Total per-area consumption rate (proportion/year), scales with predator density and Type-III response
    Type cons_rate = f_max * H * C_prev; // proportion per year

    // Saturating removal from each coral group; ensures 0 <= pred_i_prop <= i_prop
    Type pred_A_prop = A_prop * (Type(1.0) - exp(-(cons_rate * alloc_A) / (A_prop + eps)));
    Type pred_S_prop = S_prop * (Type(1.0) - exp(-(cons_rate * alloc_S) / (S_prop + eps)));

    // Coral dynamics (percentage space)
    // dA/dt in proportion units, then convert to percentage increment
    Type grow_A_prop   = r_A * A_prop * (Type(1.0) - (A_prop + S_prop) / (K_tot + eps)); // logistic regrowth
    Type mort_A_prop   = m_A * A_prop;                     // background mortality
    Type bleach_A_prop = b_A * bleach_stress * A_prop;     // heat-stress mortality

    Type dA_prop = grow_A_prop - mort_A_prop - pred_A_prop - bleach_A_prop; // net change in proportion
    Type A_next_pct_raw = A_prev + Type(100.0) * dA_prop; // convert to % change and add to current %
    Type A_next_pct = softplus(A_next_pct_raw);           // smoothly ensure non-negativity

    // Slow-growing corals
    Type grow_S_prop   = r_S * S_prop * (Type(1.0) - (A_prop + S_prop) / (K_tot + eps)); // logistic regrowth
    Type mort_S_prop   = m_S * S_prop;                     // background mortality
    Type bleach_S_prop = b_S * bleach_stress * S_prop;     // heat-stress mortality

    Type dS_prop = grow_S_prop - mort_S_prop - pred_S_prop - bleach_S_prop;
    Type S_next_pct_raw = S_prev + Type(100.0) * dS_prop;
    Type S_next_pct = softplus(S_next_pct_raw);

    // COTS dynamics (density space)
    // Per-capita growth with Allee and thermal performance, logistic limitation by K_eff
    Type percap_growth = rC_allee * tpc; // year^-1
    Type density_reg   = (K_eff > eps) ? (Type(1.0) - C_prev / (K_eff + eps)) : Type(0.0);
    Type growth_C = percap_growth * C_prev * density_reg;

    // Recruitment from consumption (food-mediated), plus immigration
    Type rec_food = (gamma_A * pred_A_prop + gamma_S * pred_S_prop) * tpc; // indiv m^-2 year^-1
    Type rec_imm  = imm_surv * IMM_prev;                                    // indiv m^-2 year^-1

    // Mortality: baseline + starvation component scaled by lack of food
    Type m_starv = starvation_scale * (Type(1.0) - food_avail);
    Type deaths_C = (m_C + m_starv) * C_prev;

    Type C_next_raw = C_prev + growth_C - deaths_C + rec_food + rec_imm;
    Type C_next = positive_part_soft(C_next_raw); // avoid negative densities smoothly

    // Update state
    fast_pred(t) = A_next_pct;
    slow_pred(t) = S_next_pct;
    cots_pred(t) = C_next;
  }

  // Observation model
  Type sd_cots_ln = min_sd_floor(obs_sd_cots_ln, minSD);
  Type sd_fast    = min_sd_floor(obs_sd_fast,    minSD);
  Type sd_slow    = min_sd_floor(obs_sd_slow,    minSD);

  // Likelihood contributions (avoid using current data in state transition; okay to compare predictions to data)
  for (int t = 1; t < n; t++) { // start at 1 to avoid double-using initialization
    // COTS: lognormal
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sd_cots_ln, true);
    // Corals: Gaussian on percentage scale
    nll -= dnorm(fast_dat(t), fast_pred(t), sd_fast, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sd_slow, true);
  }

  // REPORT predicted states for downstream evaluation (some wrappers read REPORT, not ADREPORT)
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  // ADREPORT for gradients / uncertainty if needed
  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);

  return nll;
}
