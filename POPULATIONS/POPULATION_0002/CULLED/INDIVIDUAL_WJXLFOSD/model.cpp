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
  PARAMETER(m_C);            // year^-1 | COTS baseline mortality
  PARAMETER(starvation_scale); // dimensionless | Mortality multiplier when resources scarce
  PARAMETER(K_C);            // indiv m^-2 | Baseline COTS carrying capacity scaling factor
  PARAMETER(phi_A);          // dimensionless | Weight of Acropora in COTS food/carrying capacity
  PARAMETER(phi_S);          // dimensionless | Weight of massive corals in COTS food/carrying capacity
  PARAMETER(K_food);         // proportion | Half-saturation for resource-driven recruitment/capacity
  PARAMETER(Topt_C);         // °C | Optimal temperature for COTS reproduction
  PARAMETER(sigma_T_C);      // °C | Breadth of thermal performance curve for COTS reproduction
  PARAMETER(Alee_C);         // indiv m^-2 | Allee parameter for COTS (smooth low-density limitation)
  PARAMETER(gamma_A);        // indiv m^-2 year^-1 | Conversion from Acropora consumption (proportion*year^-1) to COTS recruits
  PARAMETER(gamma_S);        // indiv m^-2 year^-1 | Conversion from massive coral consumption to COTS recruits
  PARAMETER(imm_surv);       // dimensionless | Fraction of larval immigration surviving to settled juvenile cohort per year
  PARAMETER(juv_surv);       // dimensionless | One-year survival-to-adult fraction for the juvenile cohort (lagged recruitment)
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
  add_bound_penalty(nll, m_C, true, Type(0.0), true, Type(2.0), penalty_w);
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
  add_bound_penalty(nll, juv_surv, true, Type(0.0), true, Type(1.0), penalty_w);
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

  // One-year juvenile cohort lag (not observed)
  Type juv_prev = Type(0.0);                  // indiv m^-2 | last year's settled juvenile cohort

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

    // Coral predation (proportion per year)
    Type pred_A_prop = c_attack_A * H * A_prop; // proportion year^-1 removed from Acropora
    Type pred_S_prop = c_attack_S * H * S_prop; // proportion year^-1 removed from massive corals

    // Coral dynamics (percentage space; smooth positivity via softplus where needed)
    // dA/dt in proportion units, then convert to percentage increment
    Type grow_A_prop = r_A * A_prop * (Type(1.0) - (A_prop + S_prop) / (K_tot + eps)); // logistic regrowth
    Type mort_A_prop = m_A * A_prop; // background mortality
    Type bleach_A_prop = b_A * bleach_stress * A_prop; // heat-stress mortality

    Type dA_prop = grow_A_prop - mort_A_prop - pred_A_prop - bleach_A_prop; // net change in proportion
    Type A_next_pct_raw = A_prev + Type(100.0) * dA_prop; // convert to % change and add to current %

    // Smoothly ensure non-negativity (no hard clamp); allow >100 but penalize via likelihood penalties
    Type A_next_pct = softplus(A_next_pct_raw);

    // Slow-growing corals
    Type grow_S_prop = r_S * S_prop * (Type(1.0) - (A_prop + S_prop) / (K_tot + eps));
    Type mort_S_prop = m_S * S_prop;
    Type bleach_S_prop = b_S * bleach_stress * S_prop;

    Type dS_prop = grow_S_prop - mort_S_prop - pred_S_prop - bleach_S_prop;
    Type S_next_pct_raw = S_prev + Type(100.0) * dS_prop;
    Type S_next_pct = softplus(S_next_pct_raw);

    // COTS juvenile production (settlement) and delayed maturation
    Type rec_from_food = gamma_A * pred_A_prop + gamma_S * pred_S_prop; // indiv m^-2 year^-1 (settled juveniles)
    Type imm_settle = imm_surv * IMM_prev; // indiv m^-2 year^-1 (settled juveniles from immigration)
    Type juv_mature = juv_surv * juv_prev; // indiv m^-2 year^-1 (recruits to adult from last year's juvenile cohort)
    Type juv_new = rec_from_food + imm_settle; // indiv m^-2 (juvenile cohort to carry to next year)

    // COTS population dynamics (Ricker-like with resource and thermal modulation)
    Type percap_growth = rC_allee * tpc * food_avail * (Type(1.0) - C_prev / (K_eff + eps)); // year^-1
    Type percap_mort = m_C * (Type(1.0) + starvation_scale * (Type(1.0) - food_avail)); // year^-1

    Type C_next_core = C_prev * exp(percap_growth - percap_mort) + juv_mature; // indiv m^-2
    Type C_next = softplus(C_next_core); // ensure positivity smoothly

    // Assign predictions
    fast_pred(t) = A_next_pct;
    slow_pred(t) = S_next_pct;
    cots_pred(t) = C_next;

    // Update juvenile cohort lag for next step
    juv_prev = juv_new;

    // Soft penalties to discourage biologically implausible coral cover (>100%) without hard truncation
    nll += Type(0.1) * softplus((A_next_pct - Type(100.0)) / Type(5.0));
    nll += Type(0.1) * softplus((S_next_pct - Type(100.0)) / Type(5.0));
    // Encourage non-negative and moderate combined cover (<=100%) softly
    nll += Type(0.1) * softplus((-A_next_pct) / Type(2.0));
    nll += Type(0.1) * softplus((-S_next_pct) / Type(2.0));
    nll += Type(0.1) * softplus(((A_next_pct + S_next_pct) - Type(100.0)) / Type(5.0));
  }

  // Observation likelihoods
  Type sd_cots = min_sd_floor(obs_sd_cots_ln, minSD); // log-scale SD
  Type sd_fast = min_sd_floor(obs_sd_fast, minSD);    // % scale SD
  Type sd_slow = min_sd_floor(obs_sd_slow, minSD);    // % scale SD

  for (int t = 0; t < n; t++) {
    // COTS: lognormal
    Type y_c = log(cots_dat(t) + eps);
    Type mu_c = log(cots_pred(t) + eps);
    nll -= dnorm(y_c, mu_c, sd_cots, true);

    // Corals: Gaussian on % scale (boundedness handled via soft penalties, not in likelihood)
    nll -= dnorm(fast_dat(t), fast_pred(t), sd_fast, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sd_slow, true);
  }

  // REPORTING
  REPORT(cots_pred); // indiv m^-2 predictions aligned with Year
  REPORT(fast_pred); // % cover predictions aligned with Year
  REPORT(slow_pred); // % cover predictions aligned with Year

  // Also report key intermediate drivers for diagnostics (could add ADREPORTs if needed)
  return nll;
}

/*
Assessment & process equations (discrete annual steps; all rates per year unless stated)

Addition: One-year juvenile lag to better capture episodic adult outbreaks.
- New parameter juv_surv (0..1): fraction of last year's settled juvenile cohort that survives and matures to adults.
- imm_surv now maps larval immigration to the settled juvenile cohort (not directly to adults).

1) Predation intensity (Type-III):
   H_t = C_{t-1}^2 / (H_half^2 + C_{t-1}^2)

2) Thermal performance for COTS reproduction (Gaussian):
   TPC_t = exp(-0.5 * ((SST_{t-1} - Topt_C) / sigma_T_C)^2)

3) Food availability saturation (0..1):
   Food_t = (phi_A * A_{t-1} + phi_S * S_{t-1}) / (K_food + phi_A * A_{t-1} + phi_S * S_{t-1})

4) Effective carrying capacity for COTS:
   K_eff,t = K_C * Food_t

5) COTS Allee-modified intrinsic rate:
   rC_allee,t = r_C * (C_{t-1} / (C_{t-1} + Alee_C))

6) Coral predation losses (proportion per year):
   pred_A,t = c_attack_A * H_t * A_{t-1}
   pred_S,t = c_attack_S * H_t * S_{t-1}

7) Bleaching stress factor (smooth threshold):
   B_t = softplus(k_bleach * (SST_{t-1} - T_bleach)) / k_bleach

8) Coral dynamics (proportion units; then converted to %):
   dA_t = r_A*A_{t-1}*(1 - (A_{t-1}+S_{t-1})/K_tot) - m_A*A_{t-1} - pred_A,t - b_A*B_t*A_{t-1}
   dS_t = r_S*S_{t-1}*(1 - (A_{t-1}+S_{t-1})/K_tot) - m_S*S_{t-1} - pred_S,t - b_S*B_t*S_{t-1}
   A_t(%) = softplus(A_{t-1}(%) + 100 * dA_t)
   S_t(%) = softplus(S_{t-1}(%) + 100 * dS_t)

9) COTS juvenile production (settlement) and one-year lag:
   J_new,t = gamma_A * pred_A,t + gamma_S * pred_S,t + imm_surv * cotsimm_dat(t-1)
   Adult recruits from juveniles: R_juv->adult,t = juv_surv * J_{t-1}

10) COTS dynamics (indiv m^-2):
    C_t = softplus( C_{t-1} * exp( rC_allee,t * TPC_t * (1 - C_{t-1}/K_eff,t) - m_C*(1 + starvation_scale*(1 - Food_t)) )
                    + R_juv->adult,t )

Observation models:
11) log(cots_dat) ~ Normal(log(cots_pred), obs_sd_cots_ln)
12) fast_dat(%)  ~ Normal(fast_pred(%),  obs_sd_fast)
13) slow_dat(%)  ~ Normal(slow_pred(%),  obs_sd_slow)

Initial conditions:
14) cots_pred(0) = cots_dat(0); fast_pred(0) = fast_dat(0); slow_pred(0) = slow_dat(0).
15) J_{0} = 0 (unobserved juvenile cohort starts at zero to avoid information leak).
*/
