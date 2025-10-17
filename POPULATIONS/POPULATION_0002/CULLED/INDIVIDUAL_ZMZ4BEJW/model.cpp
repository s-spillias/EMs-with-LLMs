#include <TMB.hpp>
using namespace density;

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
  PARAMETER(r_A);            // year^-1 | Acropora intrinsic regrowth rate; initial from literature/meta-analyses of coral recovery
  PARAMETER(r_S);            // year^-1 | Massive coral intrinsic regrowth rate; typically lower than Acropora
  PARAMETER(m_A);            // year^-1 | Acropora background mortality (non-predation, non-bleaching)
  PARAMETER(m_S);            // year^-1 | Massive coral background mortality
  PARAMETER(b_A);            // year^-1 | Bleaching sensitivity multiplier for Acropora mortality under heat stress
  PARAMETER(b_S);            // year^-1 | Bleaching sensitivity multiplier for massive coral mortality under heat stress
  PARAMETER(T_bleach);       // °C | Onset temperature for thermal stress (soft threshold)
  PARAMETER(k_bleach);       // 1/°C | Softness of bleaching threshold (larger -> sharper)
  PARAMETER(c_attack_A);     // year^-1 | Attack/consumption rate on Acropora (preference-weighted)
  PARAMETER(c_attack_S);     // year^-1 | Attack/consumption rate on massive corals
  PARAMETER(H_half);         // indiv m^-2 | Half-saturation density for Type-III predation
  PARAMETER(r_C);            // year^-1 | COTS intrinsic growth (fecundity to adult recruitment potential)
  PARAMETER(m_C);            // year^-1 | COTS baseline mortality (interpreted as maximum in food-poor conditions)
  PARAMETER(mC_min_frac);    // 0..1 | Minimum fraction of m_C when food availability is high
  PARAMETER(starvation_scale); // dimensionless | Shape exponent for starvation mortality vs (1 - food)
  PARAMETER(K_C);            // indiv m^-2 | Baseline COTS carrying capacity scaling factor
  PARAMETER(phi_A);          // dimensionless | Weight of Acropora in COTS food/carrying capacity
  PARAMETER(phi_S);          // dimensionless | Weight of massive corals in COTS food/carrying capacity
  PARAMETER(K_food);         // proportion | Half-saturation for resource-driven recruitment/capacity
  PARAMETER(Topt_C);         // °C | Optimal temperature for COTS reproduction
  PARAMETER(sigma_T_C);      // °C | Breadth of thermal performance curve for COTS reproduction
  PARAMETER(Alee_C);         // indiv m^-2 | Allee parameter for COTS (smooth low-density limitation)
  PARAMETER(gamma_A);        // indiv m^-2 year^-1 | Conversion from Acropora consumption (proportion*year^-1) to COTS recruits
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
  add_bound_penalty(nll, H_half, true, Type(0.05), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, r_C, true, Type(0.0), true, Type(5.0), penalty_w);
  // Align with parameters.json: m_C tightly bounded around 2.56
  add_bound_penalty(nll, m_C, true, Type(2.56), true, Type(2.5600000025600003), penalty_w);
  add_bound_penalty(nll, mC_min_frac, true, Type(0.0), true, Type(1.0), penalty_w);
  add_bound_penalty(nll, starvation_scale, true, Type(0.0), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, K_C, true, Type(0.05), true, Type(10.0), penalty_w);
  add_bound_penalty(nll, phi_A, true, Type(0.0), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, phi_S, true, Type(0.0), true, Type(5.0), penalty_w);
  add_bound_penalty(nll, K_food, true, Type(0.01), true, Type(2.0), penalty_w);
  // Align with parameters.json: Topt_C within [28.0, 29.0]
  add_bound_penalty(nll, Topt_C, true, Type(28.0), true, Type(29.0), penalty_w);
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

  // Observation SD floors
  Type sd_cots_ln = min_sd_floor(obs_sd_cots_ln, minSD);
  Type sd_fast = min_sd_floor(obs_sd_fast, minSD);
  Type sd_slow = min_sd_floor(obs_sd_slow, minSD);

  // Loop over time for process model (use t-1 values only)
  for (int t = 1; t < n; t++) {
    // Previous state (predicted, not observed)
    Type C_prev = cots_pred(t-1);             // indiv m^-2
    Type A_prev = fast_pred(t-1);             // %
    Type S_prev = slow_pred(t-1);             // %

    // Convert coral cover to proportions for internal rates
    Type A_prop = A_prev / Type(100.0);       // proportion
    Type S_prop = S_prev / Type(100.0);       // proportion

    // Predation pressure from COTS (Type-III response to predator density)
    Type pred_level = typeIII_FR(C_prev, H_half); // 0..1

    // Coral dynamics (logistic growth with losses: background, bleaching, COTS predation)
    Type total_cover_prev = A_prev + S_prev;
    Type growth_space = Type(1.0) - (total_cover_prev / (Type(100.0) * K_tot + eps));
    growth_space = CppAD::CondExpLt(growth_space, Type(0.0), Type(0.0), growth_space); // no negative growth space

    // Growth terms (logistic on available space)
    Type dA_growth = r_A * A_prev * growth_space;
    Type dS_growth = r_S * S_prev * growth_space;

    // Losses as hazard rates combined to a bounded fraction (0..1)
    Type bleach_factor = soft_bleach(sst_dat(t), T_bleach, k_bleach);
    Type loss_rate_A = m_A + b_A * bleach_factor + c_attack_A * pred_level;
    Type loss_rate_S = m_S + b_S * bleach_factor + c_attack_S * pred_level;
    // Convert to fractions lost within the year (bounded in [0,1))
    Type loss_frac_A = Type(1.0) - exp(-loss_rate_A);
    Type loss_frac_S = Type(1.0) - exp(-loss_rate_S);

    // Update coral covers with bounded losses
    Type A_new = A_prev + dA_growth - loss_frac_A * A_prev;
    Type S_new = S_prev + dS_growth - loss_frac_S * S_prev;

    // Enforce non-negativity
    A_new = CppAD::CondExpLt(A_new, Type(0.0), Type(0.0), A_new);
    S_new = CppAD::CondExpLt(S_new, Type(0.0), Type(0.0), S_new);

    // Enforce combined cap at K_tot (proportional rescaling if exceeded), AD-safe
    Type total_new = A_new + S_new;
    Type cap = K_tot * Type(100.0);
    Type scale = CppAD::CondExpGt(total_new, cap, cap / (total_new + eps), Type(1.0));
    A_new *= scale;
    S_new *= scale;

    // Recompute proportions for COTS food availability
    Type A_new_prop = A_new / Type(100.0);
    Type S_new_prop = S_new / Type(100.0);

    // Food availability and carrying capacity for COTS
    Type food_raw = phi_A * A_new_prop + phi_S * S_new_prop;     // weighted coral food
    Type food_sat = saturating01(food_raw, K_food);              // 0..1
    Type K_eff = K_C * (food_sat + eps);                         // indiv m^-2, avoid zero

    // Starvation-modulated mortality rate for COTS (from min fraction to full m_C)
    Type starv_term = CppAD::pow(Type(1.0) - food_sat, starvation_scale);
    Type mort_rate = m_C * (mC_min_frac + (Type(1.0) - mC_min_frac) * starv_term);

    // Thermal performance and Allee effect for reproduction/growth
    // Use previous year's SST to avoid using any current response-state info for growth drivers
    Type tperf = tpc_gaussian(sst_dat(t-1), Topt_C, sigma_T_C);
    Type allee = C_prev / (C_prev + Alee_C + eps);

    // Recruitment from consumption (proportional to predation pressure and coral availability)
    Type rec_food = gamma_A * (c_attack_A * pred_level) * A_prop
                  + gamma_S * (c_attack_S * pred_level) * S_prop;

    // External immigration to adults (exogenous driver)
    Type immigrants = imm_surv * cotsimm_dat(t);

    // COTS population update:
    // 1) Intrinsic density-regulated production via Ricker form (bounded decrement)
    Type r_eff = r_C * tperf * allee;
    Type C_growth = C_prev * (exp(r_eff * (Type(1.0) - C_prev / (K_eff + eps))) - Type(1.0));
    Type C_temp = max_eps(C_prev + C_growth, eps);

    // 2) Apply mortality as survival fraction (hazard-form)
    Type survivors = C_temp * exp(-mort_rate);

    // 3) Add food-mediated recruitment and immigration
    Type C_new = survivors + rec_food + immigrants;

    // Enforce non-negativity
    C_new = max_eps(C_new, eps);

    // Save predictions
    fast_pred(t) = A_new;
    slow_pred(t) = S_new;
    cots_pred(t) = C_new;

    // Observation likelihood at time t
    // COTS: lognormal on density
    Type log_obs_c = log(cots_dat(t) + eps);
    Type log_pre_c = log(cots_pred(t) + eps);
    nll -= dnorm(log_obs_c, log_pre_c, sd_cots_ln, true);

    // Corals: normal on % cover
    nll -= dnorm(fast_dat(t), fast_pred(t), sd_fast, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sd_slow, true);
  }

  // Optionally include t=0 observation likelihood (pred equals obs so contribution is finite)
  // Helps with consistent treatment of all timesteps, but adds no penalty if equal.
  Type log_obs_c0 = log(cots_dat(0) + eps);
  Type log_pre_c0 = log(cots_pred(0) + eps);
  nll -= dnorm(log_obs_c0, log_pre_c0, sd_cots_ln, true);
  nll -= dnorm(fast_dat(0), fast_pred(0), sd_fast, true);
  nll -= dnorm(slow_dat(0), slow_pred(0), sd_slow, true);

  // Reports for downstream use
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);

  return nll;
}
