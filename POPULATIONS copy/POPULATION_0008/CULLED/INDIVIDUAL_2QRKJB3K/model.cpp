#include <TMB.hpp>  // Template Model Builder header for AD and likelihood optimization

// Utility: softplus to avoid hard max(.,0)
template <class Type>
Type softplus(Type x, Type k = Type(1)) {  // Smooth approximation to max(0,x); k increases closeness to max
  return (Type(1) / k) * log(Type(1) + exp(k * x));  // Always positive and differentiable
}

// Smooth penalty for soft bounds (no hard constraints)
template <class Type>
Type soft_barrier(Type x, Type lower, Type upper, Type strength) {
  Type pen = Type(0);                                    // Initialize penalty
  pen += strength * log(Type(1) + exp(lower - x));       // Penalize smoothly if x < lower (AD-safe)
  pen += strength * log(Type(1) + exp(x - upper));       // Penalize smoothly if x > upper (AD-safe)
  return pen;                                            // Return sum of smooth penalties
}

template<class Type>
Type objective_function<Type>::operator() ()
{
  // -----------------------------
  // DATA: Forcing and response
  // -----------------------------
  DATA_VECTOR(Year);            // Year (calendar year), used for alignment/reporting; not used in dynamics directly
  DATA_VECTOR(sst_dat);         // Sea-surface temperature (°C), exogenous environmental driver
  DATA_VECTOR(cotsimm_dat);     // Larval immigration rate (individuals m^-2 yr^-1), exogenous driver
  DATA_VECTOR(cots_dat);        // Observed adult COTS density (individuals m^-2), strictly positive
  DATA_VECTOR(fast_dat);        // Observed fast-growing coral cover (%), strictly positive
  DATA_VECTOR(slow_dat);        // Observed slow-growing coral cover (%), strictly positive

  int n = cots_dat.size();      // Number of time steps in response series; assumes all series aligned/same length

  // -----------------------------
  // PARAMETERS: Ecological rates
  // -----------------------------
  PARAMETER(r_fast);            // year^-1 | Intrinsic growth rate of fast-growing corals (Acropora spp.)
  PARAMETER(r_slow);            // year^-1 | Intrinsic growth rate of slow-growing corals (Faviidae/Porites spp.)
  PARAMETER(K_total);           // % cover | Total coral carrying capacity (fast + slow), space-limited
  PARAMETER(a_base);            // (percent^-1)*(year^-1)*(ind^-1) | Baseline attack-rate scaling in Holling II
  PARAMETER(pref_fast_logit);   // logit | Diet preference for fast corals; invlogit-> fraction of feeding on fast
  PARAMETER(handling_time);     // year | Handling time in Holling II, controls saturation at high prey
  PARAMETER(g_scale);           // % per unit response | Efficiency scaling of consumption to % coral loss

  PARAMETER(m0);                // year^-1 | Baseline adult COTS mortality in adequate food conditions
  PARAMETER(m_starv);           // year^-1 | Additional mortality due to food limitation (low total coral cover)

  PARAMETER(lambda_rec);        // year^-1 | Recruitment efficiency (Beverton–Holt numerator scaling)
  PARAMETER(b_density);         // per larval units | Beverton–Holt density dependence (denominator slope)
  PARAMETER(fec_prod);          // larvae per adult (scaled) per year | Local fecundity scaling
  PARAMETER(food_half_sat);     // % cover | Half-saturation for food-limited fecundity
  PARAMETER(immigration_scale); // dimensionless | Scales external larval immigration to local supply units

  PARAMETER(S_th);              // larval units | Center of smooth outbreak threshold in recruitment
  PARAMETER(k_th);              // per larval units | Steepness of outbreak logistic (higher = more abrupt)

  PARAMETER(T_opt);             // °C | Thermal optimum for larval survival/fecundity
  PARAMETER(T_sd);              // °C | Thermal breadth (std dev) of Gaussian thermal performance
  PARAMETER(bmax_bleach);       // fraction (0-1) | Maximum reduction in coral growth due to bleaching stress
  PARAMETER(T_bleach);          // °C | SST logistic bleaching inflection (onset of strong impact)
  PARAMETER(gamma_bleach);      // per °C | Steepness of bleaching logistic

  // Observation error parameters (log-scale to enforce positivity)
  PARAMETER(log_sd_cots);       // log SD | Observation error (lognormal) for COTS density
  PARAMETER(log_sd_fast);       // log SD | Observation error (lognormal) for fast coral cover
  PARAMETER(log_sd_slow);       // log SD | Observation error (lognormal) for slow coral cover

  // -----------------------------
  // CONSTANTS AND HELPERS
  // -----------------------------
  Type eps = Type(1e-8);        // Small constant for numerical stability to avoid division by zero/log(0)
  Type minsd = Type(0.05);      // Minimum SD on log scale to stabilize likelihood (prevents zero variance)
  Type cap_k = Type(10.0);      // Softness for softplus; larger => closer to hard max(.,0) but smooth
  Type pen_strength = Type(1e-3); // Soft-bound penalty strength (small, encourages but doesn't force bounds)

  // Derived observation SDs (ensure > minsd using combination)
  Type sd_cots = sqrt(exp(Type(2.0) * log_sd_cots) + minsd * minsd);  // Positive SD for COTS observations
  Type sd_fast = sqrt(exp(Type(2.0) * log_sd_fast) + minsd * minsd);  // Positive SD for fast coral observations
  Type sd_slow = sqrt(exp(Type(2.0) * log_sd_slow) + minsd * minsd);  // Positive SD for slow coral observations

  // Preference weights (bounded in [0,1]); use TMB's invlogit
  Type w_fast = invlogit(pref_fast_logit);                // Fraction of feeding directed to fast corals
  Type w_slow = Type(1.0) - w_fast;                      // Remaining fraction directed to slow corals

  // -----------------------------
  // STATE VECTORS (predictions)
  // -----------------------------
  vector<Type> cots_pred(n);     // Predicted adult COTS density (individuals m^-2)
  vector<Type> fast_pred(n);     // Predicted fast coral cover (%)
  vector<Type> slow_pred(n);     // Predicted slow coral cover (%)

  // INITIAL CONDITIONS: use observed t=0 to seed model state (no optimization/leakage beyond initialization)
  cots_pred(0) = cots_dat(0);    // Set initial COTS state from observation at first time step
  fast_pred(0) = fast_dat(0);    // Set initial fast coral cover from observation at first time step
  slow_pred(0) = slow_dat(0);    // Set initial slow coral cover from observation at first time step

  // -----------------------------
  // NEGATIVE LOG-LIKELIHOOD
  // -----------------------------
  Type nll = Type(0.0);          // Accumulator for negative log-likelihood

  // -----------------------------
  // SOFT BOUNDS PENALTIES (encourage biological plausibility without hard constraints)
  // -----------------------------
  nll += soft_barrier(r_fast, Type(0.05), Type(1.5), pen_strength);            // Encourage plausible range for r_fast
  nll += soft_barrier(r_slow, Type(0.01), Type(0.8), pen_strength);            // Encourage plausible range for r_slow
  nll += soft_barrier(K_total, Type(30.0), Type(95.0), pen_strength);          // Encourage realistic total cover
  nll += soft_barrier(a_base, Type(0.001), Type(0.5), pen_strength);           // Attack rate plausible range
  nll += soft_barrier(handling_time, Type(0.01), Type(3.0), pen_strength);     // Handling time plausible range
  nll += soft_barrier(g_scale, Type(0.2), Type(5.0), pen_strength);            // Consumption-to-cover scaling
  nll += soft_barrier(m0, Type(0.05), Type(1.5), pen_strength);                // Baseline mortality range
  nll += soft_barrier(m_starv, Type(0.0), Type(3.0), pen_strength);            // Starvation mortality range
  nll += soft_barrier(lambda_rec, Type(0.01), Type(2.0), pen_strength);        // Recruitment scaling plausible range
  nll += soft_barrier(b_density, Type(0.001), Type(5.0), pen_strength);        // BH density dependence range
  nll += soft_barrier(fec_prod, Type(0.1), Type(20.0), pen_strength);          // Fecundity scaling range
  nll += soft_barrier(food_half_sat, Type(5.0), Type(60.0), pen_strength);     // Half-saturation for fecundity
  nll += soft_barrier(immigration_scale, Type(0.01), Type(50.0), pen_strength);// Immigration scaling range
  nll += soft_barrier(S_th, Type(0.0), Type(10.0), pen_strength);              // Outbreak threshold center
  nll += soft_barrier(k_th, Type(0.1), Type(20.0), pen_strength);              // Outbreak steepness
  nll += soft_barrier(T_opt, Type(23.0), Type(31.0), pen_strength);            // Larval survival thermal optimum
  nll += soft_barrier(T_sd, Type(0.5), Type(5.0), pen_strength);               // Thermal breadth
  nll += soft_barrier(bmax_bleach, Type(0.0), Type(1.0), pen_strength);        // Max bleaching impact fraction
  nll += soft_barrier(T_bleach, Type(26.0), Type(32.0), pen_strength);         // Bleaching inflection temperature
  nll += soft_barrier(gamma_bleach, Type(0.1), Type(5.0), pen_strength);       // Bleaching steepness

  // -----------------------------
  // EQUATION DEFINITIONS (discrete-time, yearly step)
  // 1) Coral bleaching modifier: B(t) = 1 / (1 + exp(-γ (SST(t-1) - T_bleach)))
  // 2) Coral growth modifiers: G_fast = r_fast * fast * (1 - (fast+slow)/K_total) * (1 - bmax_bleach * B)
  //                             G_slow = r_slow * slow * (1 - (fast+slow)/K_total) * (1 - bmax_bleach * B)
  // 3) COTS consumption (Holling II): denom = 1 + handling_time * (a_f*fast + a_s*slow)
  //                                   C_fast = g_scale * cots * a_f * fast / denom
  //                                   C_slow = g_scale * cots * a_s * slow / denom
  //    where a_f = a_base * w_fast, a_s = a_base * w_slow
  // 4) Food effect on fecundity (saturating): F_food = (w_f*fast + w_s*slow) / (food_half_sat + w_f*fast + w_s*slow)
  // 5) Thermal performance for larvae: T_perf = exp(-0.5 * ((SST(t-1) - T_opt)/T_sd)^2)
  // 6) Larval supply: S = immigration_scale * cotsimm(t-1) + fec_prod * cots * F_food * T_perf
  // 7) Outbreak logistic multiplier: L = 1 / (1 + exp(-k_th * (S - S_th)))
  // 8) Recruitment (Beverton–Holt): R = lambda_rec * L * T_perf * S / (1 + b_density * S)
  // 9) COTS mortality: M = (m0 + m_starv * (1 - (fast+slow)/K_total)) * cots
  // 10) State updates:
  //     cots(t) = softplus(cots + R - M)           // positive and smooth
  //     fast(t) = softplus(fast + G_fast - C_fast) // positive and smooth
  //     slow(t) = softplus(slow + G_slow - C_slow) // positive and smooth
  // -----------------------------

  for (int t = 1; t < n; t++) {                        // Iterate over time, predicting based on previous states only
    // Previous states (t-1)
    Type cots_prev = cots_pred(t-1);                  // Previous adult COTS density (ind m^-2)
    Type fast_prev = fast_pred(t-1);                  // Previous fast coral cover (%)
    Type slow_prev = slow_pred(t-1);                  // Previous slow coral cover (%)
    Type sst_prev  = sst_dat(t-1);                    // Previous SST (°C)
    Type imm_prev  = cotsimm_dat(t-1);                // Previous larval immigration (ind m^-2 yr^-1)

    // 1) Bleaching stress (0..1)
    Type bleach = Type(1.0) / (Type(1.0) + exp(-gamma_bleach * (sst_prev - T_bleach))); // Logistic bleaching index

    // 2) Coral growth with competition and bleaching reduction
    Type total_prev = fast_prev + slow_prev + eps;    // Total coral cover (%), add eps for stability
    Type comp_term = (Type(1.0) - (total_prev / (K_total + eps)));                // Space-limited growth factor
    Type growth_fast = r_fast * fast_prev * comp_term * (Type(1.0) - bmax_bleach * bleach); // Fast coral growth
    Type growth_slow = r_slow * slow_prev * comp_term * (Type(1.0) - bmax_bleach * bleach); // Slow coral growth

    // 3) COTS consumption via Holling II with preference
    Type a_fast = a_base * w_fast;                    // Attack rate on fast coral
    Type a_slow = a_base * w_slow;                    // Attack rate on slow coral
    Type denom = Type(1.0) + handling_time * (a_fast * fast_prev + a_slow * slow_prev); // Saturation denominator
    Type cons_fast = g_scale * cots_prev * a_fast * fast_prev / (denom + eps);   // Consumption of fast (%)
    Type cons_slow = g_scale * cots_prev * a_slow * slow_prev / (denom + eps);   // Consumption of slow (%)

    // 4) Food limitation on fecundity (saturating with combined preferred coral)
    Type food_avail = w_fast * fast_prev + w_slow * slow_prev;                    // Weighted food availability (%)
    Type F_food = food_avail / (food_half_sat + food_avail + eps);                // Fractional fecundity (0..1)

    // 5) Thermal performance for larvae (Gaussian bell-shaped)
    Type T_perf = exp( - Type(0.5) * pow((sst_prev - T_opt) / (T_sd + eps), 2) ); // 0..1 thermal survival multiplier

    // 6) Larval supply combining immigration and local production
    Type S = immigration_scale * imm_prev + fec_prod * cots_prev * F_food * T_perf; // Total larval supply (scaled units)

    // 7) Outbreak logistic multiplier (smooth threshold on S)
    Type L = Type(1.0) / (Type(1.0) + exp(-k_th * (S - S_th)));                 // 0..1 outbreak-enabling multiplier

    // 8) Recruitment to adults via Beverton–Holt with environmental multipliers
    Type R = lambda_rec * L * T_perf * S / (Type(1.0) + b_density * S);         // New adults recruited (ind m^-2 yr^-1)

    // 9) Mortality including starvation when corals are depleted
    Type food_frac = (fast_prev + slow_prev) / (K_total + eps);                  // Fraction of carrying capacity remaining (0..1)
    Type M_rate = m0 + m_starv * (Type(1.0) - food_frac);                        // Total mortality rate (year^-1)
    Type M = M_rate * cots_prev;                                                 // Adults lost to mortality (ind m^-2 yr^-1)

    // 10) State updates with smooth positivity via softplus
    cots_pred(t) = softplus(cots_prev + R - M, cap_k);                           // Update COTS state (positive)
    fast_pred(t) = softplus(fast_prev + growth_fast - cons_fast, cap_k);         // Update fast coral cover (positive)
    slow_pred(t) = softplus(slow_prev + growth_slow - cons_slow, cap_k);         // Update slow coral cover (positive)
  }

  // -----------------------------
  // LIKELIHOOD: Lognormal observation model for strictly positive data
  // Use log-transform with small epsilon to ensure finite values
  // -----------------------------
  for (int t = 0; t < n; t++) {                                                   // Loop over all observations, include all
    // COTS likelihood (lognormal)
    Type lc_obs = log(cots_dat(t) + eps);                                         // Observed log COTS
    Type lc_prd = log(cots_pred(t) + eps);                                         // Predicted log COTS
    nll -= dnorm(lc_obs, lc_prd, sd_cots, true);                                   // Add log-density (negative for NLL)

    // Fast coral likelihood (lognormal)
    Type lf_obs = log(fast_dat(t) + eps);                                         // Observed log fast coral cover
    Type lf_prd = log(fast_pred(t) + eps);                                         // Predicted log fast coral cover
    nll -= dnorm(lf_obs, lf_prd, sd_fast, true);                                   // Add to NLL

    // Slow coral likelihood (lognormal)
    Type ls_obs = log(slow_dat(t) + eps);                                         // Observed log slow coral cover
    Type ls_prd = log(slow_pred(t) + eps);                                         // Predicted log slow coral cover
    nll -= dnorm(ls_obs, ls_prd, sd_slow, true);                                   // Add to NLL
  }

  // -----------------------------
  // REPORTING
  // -----------------------------
  REPORT(Year);                // Report time index for alignment
  REPORT(sst_dat);             // Report SST driver
  REPORT(cotsimm_dat);         // Report larval immigration driver

  REPORT(cots_pred);           // Report predicted COTS densities
  REPORT(fast_pred);           // Report predicted fast coral cover
  REPORT(slow_pred);           // Report predicted slow coral cover

  REPORT(sd_cots);             // Report derived observation SDs
  REPORT(sd_fast);             // Report derived observation SDs
  REPORT(sd_slow);             // Report derived observation SDs

  REPORT(r_fast);              // Report parameters for transparency
  REPORT(r_slow);
  REPORT(K_total);
  REPORT(a_base);
  REPORT(pref_fast_logit);
  REPORT(handling_time);
  REPORT(g_scale);
  REPORT(m0);
  REPORT(m_starv);
  REPORT(lambda_rec);
  REPORT(b_density);
  REPORT(fec_prod);
  REPORT(food_half_sat);
  REPORT(immigration_scale);
  REPORT(S_th);
  REPORT(k_th);
  REPORT(T_opt);
  REPORT(T_sd);
  REPORT(bmax_bleach);
  REPORT(T_bleach);
  REPORT(gamma_bleach);
  REPORT(log_sd_cots);
  REPORT(log_sd_fast);
  REPORT(log_sd_slow);

  return nll;                 // Return total negative log-likelihood for optimization
}
