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

// Hill-type saturating function for sharper outbreak thresholds
template<class Type>
Type hill_saturating(Type x, Type K, Type nu) {
  // Returns x^nu / (K^nu + x^nu) with stable evaluation for AD types
  const Type eps = Type(1e-8);
  Type xe = x + eps;
  Type Ke = K + eps;
  Type num = exp(nu * log(xe));
  Type den = exp(nu * log(Ke)) + num;
  return num / (den + eps);
}

template<class Type>
Type soft_bleach(Type T, Type T_thresh, Type k) { // Smooth threshold for bleaching
  // Returns a non-negative stress factor increasing with T - T_thresh
  return softplus(k * (T - T_thresh)) / (k + Type(1e-8));
}

// Clamp to [0, upper] using CppAD-compatible conditionals
template<class Type>
Type clamp0U(Type x, Type upper) {
  Type zero = Type(0);
  Type y = CppAD::CondExpLt(x, zero, zero, x);
  y = CppAD::CondExpGt(y, upper, upper, y);
  return y;
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
  DATA_VECTOR(fast_dat);                     // Fast coral cover (Acropora) (proportion 0-1)
  DATA_VECTOR(slow_dat);                     // Slow coral cover (Faviidae/Porites) (proportion 0-1)
  DATA_VECTOR(sst_dat);                      // Sea-surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);                  // COTS larval immigration (indiv m^-2 year^-1)

  int n = cots_dat.size();                   // Number of time steps (years)

  // PARAMETERS (all unconstrained; soft penalties enforce biological ranges)
  PARAMETER(r_A);            // year^-1 | Acropora intrinsic regrowth rate
  PARAMETER(r_S);            // year^-1 | Massive coral intrinsic regrowth rate
  PARAMETER(m_A);            // year^-1 | Acropora background mortality
  PARAMETER(m_S);            // year^-1 | Massive coral background mortality
  PARAMETER(b_A);            // year^-1 | Bleaching sensitivity multiplier for Acropora mortality
  PARAMETER(b_S);            // year^-1 | Bleaching sensitivity multiplier for massive coral mortality
  PARAMETER(T_bleach);       // °C | Onset temperature for thermal stress
  PARAMETER(k_bleach);       // 1/°C | Softness of bleaching threshold
  PARAMETER(c_attack_A);     // year^-1 | Attack/consumption rate on Acropora
  PARAMETER(c_attack_S);     // year^-1 | Attack/consumption rate on massive corals
  PARAMETER(H_half);         // indiv m^-2 | Half-saturation density for Type-III predation
  PARAMETER(r_C);            // year^-1 | COTS intrinsic growth (fecundity to adult recruitment potential)
  PARAMETER(m_C);            // year^-1 | COTS baseline mortality
  PARAMETER(starvation_scale); // dimensionless | Mortality multiplier when resources scarce
  PARAMETER(K_C);            // indiv m^-2 | Baseline COTS carrying capacity scaling factor
  PARAMETER(phi_A);          // dimensionless | Weight of Acropora in COTS food/carrying capacity
  PARAMETER(phi_S);          // dimensionless | Weight of massive corals in COTS food/carrying capacity
  PARAMETER(K_food);         // proportion | Half-saturation for resource-driven recruitment/capacity
  PARAMETER(food_hill_nu);   // dimensionless | Hill exponent controlling sharpness of resource-response
  PARAMETER(Topt_C);         // °C | Optimal temperature for COTS reproduction
  PARAMETER(sigma_T_C);      // °C | Breadth of thermal performance curve for COTS reproduction
  PARAMETER(Alee_C);         // indiv m^-2 | Allee parameter for COTS (smooth low-density limitation)
  PARAMETER(gamma_A);        // indiv m^-2 year^-1 | Conversion from Acropora consumption to COTS recruits
  PARAMETER(gamma_S);        // indiv m^-2 year^-1 | Conversion from massive coral consumption to COTS recruits
  PARAMETER(imm_surv);       // dimensionless | Fraction of larval immigration surviving to adult class per year
  PARAMETER(K_tot);          // proportion (0..1) | Maximum combined cover of modeled coral groups
  // New parameter: delayed larval survival to adults (internal production)
  PARAMETER(larv_surv);      // dimensionless | Fraction of internally produced larvae that survive to recruit as adults after one year
  // Observation model SDs
  PARAMETER(obs_sd_cots_ln);     // log-scale SD for COTS observations
  PARAMETER(obs_sd_fast);        // SD for Acropora cover (proportion)
  PARAMETER(obs_sd_slow);        // SD for massive coral cover (proportion)

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
  add_bound_penalty(nll, H_half, true, Type(0.05), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, r_C, true, Type(0.0), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, m_C, true, Type(0.0), true, Type(3.0), penalty_w);
  add_bound_penalty(nll, starvation_scale, true, Type(0.0), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, K_C, true, Type(0.05), true, Type(10.0), penalty_w);
  add_bound_penalty(nll, phi_A, true, Type(0.0), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, phi_S, true, Type(0.0), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, K_food, true, Type(0.01), true, Type(2.0), penalty_w);
  add_bound_penalty(nll, food_hill_nu, true, Type(1.0), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, Topt_C, true, Type(24.0), true, Type(32.0), penalty_w);
  add_bound_penalty(nll, sigma_T_C, true, Type(0.1), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, Alee_C, true, Type(0.0), true, Type(2.0), penalty_w);
  add_bound_penalty(nll, gamma_A, true, Type(0.0), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, gamma_S, true, Type(0.0), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, imm_surv, true, Type(0.0), true, Type(1.0), penalty_w);
  add_bound_penalty(nll, K_tot, true, Type(0.2), true, Type(1.0), penalty_w);
  add_bound_penalty(nll, larv_surv, true, Type(0.0), true, Type(1.0), penalty_w);
  add_bound_penalty(nll, obs_sd_cots_ln, true, Type(0.01), true, Type(2.0), penalty_w);
  add_bound_penalty(nll, obs_sd_fast, true, Type(0.01), true, Type(10.0), penalty_w);
  add_bound_penalty(nll, obs_sd_slow, true, Type(0.01), true, Type(10.0), penalty_w);

  // Effective SDs for observation models (ensure positive)
  Type sd_cots_ln = min_sd_floor(obs_sd_cots_ln, minSD);
  Type sd_fast     = min_sd_floor(obs_sd_fast,     minSD);
  Type sd_slow     = min_sd_floor(obs_sd_slow,     minSD);

  // PREDICTION VECTORS (initialize with observed first values to set initial conditions)
  vector<Type> C_pred(n);  // COTS adults (indiv m^-2)
  vector<Type> A_pred(n);  // Acropora cover (proportion 0..1)
  vector<Type> S_pred(n);  // Massive coral cover (proportion 0..1)

  // Required named prediction outputs matching response variables
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  vector<Type> rec_potential(n); rec_potential.setZero(); // Internal larval production at time t for t+1 recruitment
  vector<Type> rec_delay(n);     rec_delay.setZero();     // Delayed recruits entering adult class at time t

  // Condition on initial state using first observation (previous-time-step relative to t=1)
  C_pred(0) = max_eps(cots_dat(0), eps);
  A_pred(0) = clamp0U(fast_dat(0), K_tot);
  S_pred(0) = clamp0U(slow_dat(0), K_tot);

  // Mirror to required prediction vectors
  cots_pred(0) = C_pred(0);
  fast_pred(0) = A_pred(0);
  slow_pred(0) = S_pred(0);

  // Likelihood contribution for t=0 (observation model only)
  nll -= dnorm(log(cots_dat(0) + eps), log(C_pred(0) + eps), sd_cots_ln, true);
  nll -= dnorm(fast_dat(0), A_pred(0), sd_fast, true);
  nll -= dnorm(slow_dat(0), S_pred(0), sd_slow, true);

  // Time loop: predict states at t from states at t-1 (no usage of current *_dat in process equations)
  for (int t = 1; t < n; t++) {
    // Previous states
    Type C_prev = C_pred(t-1);
    Type A_prev = A_pred(t-1);
    Type S_prev = S_pred(t-1);

    // Environmental and exogenous drivers from previous year (avoid leakage)
    Type T_prev = sst_dat(t-1);
    Type imm_prev = cotsimm_dat(t-1);

    // Thermal stress for bleaching (soft threshold)
    Type stress = soft_bleach(T_prev, T_bleach, k_bleach); // non-negative

    // Functional response of predation based on COTS density (Type III)
    Type FR = typeIII_FR(C_prev, H_half);

    // Predation/consumption on corals
    Type pred_A = c_attack_A * FR * A_prev; // proportion removed from A
    Type pred_S = c_attack_S * FR * S_prev; // proportion removed from S

    // Coral logistic regrowth (limited by combined cover <= K_tot)
    Type cover_prev = A_prev + S_prev;
    Type reg_A = r_A * A_prev * (Type(1.0) - cover_prev / (K_tot + eps));
    Type reg_S = r_S * S_prev * (Type(1.0) - cover_prev / (K_tot + eps));

    // Background and bleaching mortalities
    Type mort_A = m_A * A_prev + b_A * stress * A_prev;
    Type mort_S = m_S * S_prev + b_S * stress * S_prev;

    // Intermediate coral updates
    Type A_tmp = A_prev + reg_A - pred_A - mort_A;
    Type S_tmp = S_prev + reg_S - pred_S - mort_S;

    // Enforce non-negativity
    A_tmp = CppAD::CondExpLt(A_tmp, Type(0), Type(0), A_tmp);
    S_tmp = CppAD::CondExpLt(S_tmp, Type(0), Type(0), S_tmp);

    // Enforce combined cover <= K_tot via proportional scaling if needed
    Type sum_pos = A_tmp + S_tmp + eps;
    Type scale = CppAD::CondExpGt(sum_pos, K_tot, K_tot / sum_pos, Type(1.0));
    A_pred(t) = A_tmp * scale;
    S_pred(t) = S_tmp * scale;

    // Resource availability index for COTS from previous corals
    Type food_prev = hill_saturating(phi_A * A_prev + phi_S * S_prev, K_food, food_hill_nu);

    // Thermal performance and Allee effect for reproduction (prev year)
    Type therm_prev = tpc_gaussian(T_prev, Topt_C, sigma_T_C);
    Type allee_prev = C_prev / (Alee_C + C_prev + eps);

    // Internal larval production at t-1 that recruits at t (delayed)
    Type rec_prod_prev = r_C * (gamma_A * pred_A + gamma_S * pred_S) * therm_prev * food_prev * allee_prev;
    rec_delay(t) = larv_surv * rec_prod_prev;

    // Immigration recruits based on previous exogenous larval supply
    Type imm_in = imm_surv * imm_prev;

    // Starvation-enhanced mortality (higher when food is scarce)
    Type m_eff = m_C + starvation_scale * (Type(1.0) - food_prev);
    // Survivors from previous year using exponential survival
    Type C_surv = C_prev * exp(-m_eff);

    // Add recruits
    Type C_tmp_add = rec_delay(t) + imm_in;
    Type C_tmp = C_surv + C_tmp_add;

    // Resource-modulated carrying capacity and smooth ceiling
    Type K_eff = K_C * food_prev + eps;
    // Smooth saturation: maps C_tmp to (0, K_eff)
    C_pred(t) = K_eff * (Type(1.0) - exp(- C_tmp / (K_eff)));

    // Mirror to required prediction vectors
    cots_pred(t) = C_pred(t);
    fast_pred(t) = A_pred(t);
    slow_pred(t) = S_pred(t);

    // Observation likelihood at time t
    nll -= dnorm(log(cots_dat(t) + eps), log(C_pred(t) + eps), sd_cots_ln, true);
    nll -= dnorm(fast_dat(t), A_pred(t), sd_fast, true);
    nll -= dnorm(slow_dat(t), S_pred(t), sd_slow, true);
  }

  // Report predicted state trajectories
  REPORT(A_pred);
  REPORT(S_pred);
  REPORT(C_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(cots_pred);

  // Return total negative log-likelihood
  return nll;
}
