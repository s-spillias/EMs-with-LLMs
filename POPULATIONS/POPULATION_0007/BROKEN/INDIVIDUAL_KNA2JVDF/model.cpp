#include <TMB.hpp>

// Smooth positive part to avoid hard cutoffs and preserve differentiability
template<class Type>
inline Type pospart(const Type& x) {
  return (x + CppAD::sqrt(x * x + Type(1e-8))) / Type(2.0); // smooth ReLU, epsilon prevents NaN
}

// Smooth upper cap using pospart: softmin(x, cap) ~ min(x, cap) but smooth at the hinge
template<class Type>
inline Type softmin(const Type& x, const Type& cap) {
  return cap - pospart(cap - x);
}

// Smooth quadratic penalty for parameters outside [lo, hi]
template<class Type>
inline Type range_penalty(const Type& x, const Type& lo, const Type& hi, const Type& w) {
  Type below = pospart(lo - x);    // >0 when x < lo
  Type above = pospart(x - hi);    // >0 when x > hi
  return w * (below * below + above * above); // quadratic penalty outside range
}

// Logit transform for % cover (0-100), kept strictly inside bounds
template<class Type>
inline Type logit_pct(const Type& x) {
  Type a = Type(1e-6); // small constant to avoid 0/100
  Type p = (x + a) / (Type(100.0) + Type(2.0) * a); // map [0,100] -> (0,1)
  return log(p / (Type(1.0) - p));
}

template<class Type>
Type objective_function<Type>::operator() () {
  // ------------------------
  // DATA
  // ------------------------
  DATA_VECTOR(Year);        // calendar year (integer-valued, but numeric vector)
  DATA_VECTOR(cots_dat);    // Adult COTS abundance (ind/m^2), strictly positive
  DATA_VECTOR(fast_dat);    // Fast coral cover (Acropora spp.) in %, bounded [0,100]
  DATA_VECTOR(slow_dat);    // Slow coral cover (Faviidae/Porites) in %, bounded [0,100]
  DATA_VECTOR(sst_dat);     // Sea Surface Temperature (°C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (ind/m^2/year)

  int T = Year.size(); // number of time steps (years)

  // ------------------------
  // PARAMETERS
  // ------------------------
  // COTS recruitment scaling (ind m^-2 yr^-1 at unit modifiers)
  PARAMETER(alpha_rec);   // Recruitment productivity scaling (units: ind m^-2 yr^-1); sets outbreak potential; initial estimate
  // Density-dependent fecundity exponent (dimensionless), >=1 increases superlinear recruitment
  PARAMETER(phi);         // Fecundity density exponent (unitless); shapes recruitment curvature; literature/initial estimate
  // Smooth Allee effect parameters
  PARAMETER(k_allee);     // Allee logistic steepness (m^2 ind^-1); higher values -> sharper threshold; initial estimate
  PARAMETER(C_allee);     // Allee threshold density (ind m^-2); density at which mating success rises; literature/initial estimate
  // Food/Resource saturation for adult condition and maturation (interpreted as adult condition)
  PARAMETER(K_R);         // Half-saturation coral cover for resource (%, 0-100+); initial estimate
  PARAMETER(wF);          // Weight of fast coral in resource index (unitless); initial estimate
  PARAMETER(wS);          // Weight of slow coral in resource index (unitless); initial estimate
  // Mortality terms
  PARAMETER(muC);         // Baseline adult mortality (yr^-1); initial estimate
  PARAMETER(gammaC);      // Density-dependent mortality (m^2 ind^-1 yr^-1); drives busts at high density; initial estimate
  // Temperature effect on recruitment (Gaussian peak)
  PARAMETER(T_opt_rec);   // Optimal SST for recruitment (°C); literature
  PARAMETER(beta_rec);    // Curvature of Gaussian temperature effect (°C^-2); larger -> narrower peak; initial estimate
  // Temperature effect on coral (bleaching loss above threshold)
  PARAMETER(T_opt_bleach); // Onset SST for bleaching loss (°C); literature
  PARAMETER(beta_bleach);  // Multiplier on growth under heat stress (unitless >=0); initial estimate
  PARAMETER(m_bleachF);    // Additional fast coral proportional loss per °C above threshold (yr^-1 °C^-1); initial estimate
  PARAMETER(m_bleachS);    // Additional slow coral proportional loss per °C above threshold (yr^-1 °C^-1); initial estimate
  // Coral intrinsic regrowth and space competition
  PARAMETER(rF);          // Fast coral intrinsic regrowth (yr^-1 on % scale); literature/initial
  PARAMETER(rS);          // Slow coral intrinsic regrowth (yr^-1 on % scale); literature/initial
  PARAMETER(K_tot);       // Total coral carrying capacity (% cover for fast+slow), <=100; literature/initial
  // COTS functional response on corals (multi-prey Holling with Type II/III blend)
  PARAMETER(aF);          // Attack/encounter parameter on fast coral (yr^-1 %^-etaF m^2 ind^-1 scaled); initial estimate
  PARAMETER(aS);          // Attack/encounter parameter on slow coral (yr^-1 %^-etaS m^2 ind^-1 scaled); initial estimate
  PARAMETER(etaF);        // Shape exponent for fast coral (>=1: Type-III-like at low cover); unitless; initial estimate
  PARAMETER(etaS);        // Shape exponent for slow coral (>=1: Type-III-like at low cover); unitless; initial estimate
  PARAMETER(h);           // Handling/satiation time scaler (yr %^-1); increases saturation with coral cover; initial estimate
  PARAMETER(qF);          // Efficiency converting feeding to % cover loss for fast (unitless, 0-1); literature/initial
  PARAMETER(qS);          // Efficiency converting feeding to % cover loss for slow (unitless, 0-1); literature/initial
  // Observation error parameters
  PARAMETER(sigma_cots);  // Lognormal sd for COTS (log-space); initial estimate
  PARAMETER(sigma_fast);  // Normal sd for logit(% fast); initial estimate
  PARAMETER(sigma_slow);  // Normal sd for logit(% slow); initial estimate
  // Juvenile stage parameters (resource-modulated maturation)
  PARAMETER(mJ_min);      // Minimum juvenile -> adult maturation rate (yr^-1), 0-1
  PARAMETER(mJ_max);      // Maximum juvenile -> adult maturation rate (yr^-1), 0-1
  PARAMETER(muJ);         // Juvenile mortality (yr^-1), >=0

  // New: Unobserved environmental recruitment modifier (lognormal AR(1))
  PARAMETER_VECTOR(u_rec);      // length T; environmental effect on recruitment (log scale)
  PARAMETER(sigma_rec_env);     // SD of u_rec innovations
  PARAMETER(ar1_rec_phi);       // AR(1) autocorrelation parameter in (-1,1)

  // New: Initial states (avoid data leakage from *_dat at t=0)
  PARAMETER(C0);  // initial adult COTS density (ind/m^2)
  PARAMETER(F0);  // initial fast coral cover (%)
  PARAMETER(S0);  // initial slow coral cover (%)

  // ------------------------
  // EQUATION DEFINITIONS (discrete-time, yearly)
  //
  // Stage-structured COTS (juveniles J, adults C):
  // 1) Adult condition index (from coral): R = wF*F + wS*S; f_food = R / (K_R + R)
  // 2) Smooth Allee function: f_Allee = 1 / (1 + exp(-k_allee*(C - C_allee)))
  // 3) Temperature effect on recruitment: f_Trec = exp( -beta_rec * (SST - T_opt_rec)^2 )
  // 4) Unobserved environmental recruitment modifier (episodic): env_mod = exp(u_rec(t)), AR(1) on u_rec
  // 5) Juvenile input: Rec_in = alpha_rec * C^phi * f_Allee * f_food * f_Trec * env_mod + immigration
  // 6) Maturation: mJ_eff = mJ_min + (mJ_max - mJ_min) * f_food
  // 7) Juveniles: J_{t+1} = J_t + Rec_in - mJ_eff*J_t - muJ*J_t
  // 8) Adults: C_{t+1} = C_t + mJ_eff*J_t - (muC + gammaC * C_t) * C_t
  // 9) Corals (shared space K_tot, bleaching modifier, multi-prey consumption)
  //     F_{t+1} = F_t + rF*F_t*(1 - (F_t+S_t)/K_tot)*exp(-beta_bleach*max(0,SST - T_opt_bleach)) - Cons_fast - B_fast
  //     S_{t+1} = S_t + rS*S_t*(1 - (F_t+S_t)/K_tot)*exp(-beta_bleach*max(0,SST - T_opt_bleach)) - Cons_slow - B_slow
  // Notes:
  // - Predictions at t use only states from t and covariates up to t (no use of *_dat response variables).
  // ------------------------

  // Negative log-likelihood accumulator
  Type nll = 0.0;
  const Type eps = Type(1e-8);      // small epsilon to stabilize divisions/logs
  const Type sd_floor = Type(0.05); // minimum sd used in likelihood for stability

  // Suggested biological ranges for smooth penalties (very broad, weakly enforced)
  const Type w_pen = Type(1e-3);

  // Apply smooth range penalties to keep parameters within plausible bounds (do not hard-constrain)
  nll += range_penalty(alpha_rec, Type(0.0),   Type(10.0),  w_pen);
  nll += range_penalty(phi,       Type(1.0),   Type(3.0),   w_pen);
  nll += range_penalty(k_allee,   Type(0.01),  Type(20.0),  w_pen);
  nll += range_penalty(C_allee,   Type(0.0),   Type(5.0),   w_pen);
  nll += range_penalty(K_R,       Type(1.0),   Type(100.0), w_pen);
  nll += range_penalty(wF,        Type(0.0),   Type(2.0),   w_pen);
  nll += range_penalty(wS,        Type(0.0),   Type(2.0),   w_pen);
  nll += range_penalty(muC,       Type(0.0),   Type(3.0),   w_pen);
  nll += range_penalty(gammaC,    Type(0.0),   Type(10.0),  w_pen);
  nll += range_penalty(T_opt_rec, Type(20.0),  Type(34.0),  w_pen);
  nll += range_penalty(beta_rec,  Type(0.0),   Type(2.0),   w_pen);
  nll += range_penalty(T_opt_bleach, Type(20.0), Type(34.0), w_pen);
  nll += range_penalty(beta_bleach,  Type(0.0),  Type(5.0),  w_pen);
  nll += range_penalty(m_bleachF,    Type(0.0),  Type(2.0),  w_pen);
  nll += range_penalty(m_bleachS,    Type(0.0),  Type(2.0),  w_pen);
  nll += range_penalty(rF,           Type(0.0),  Type(2.0),  w_pen);
  nll += range_penalty(rS,           Type(0.0),  Type(2.0),  w_pen);
  nll += range_penalty(K_tot,        Type(10.0), Type(100.0),w_pen);
  nll += range_penalty(aF,           Type(0.0),  Type(1.0),  w_pen);
  nll += range_penalty(aS,           Type(0.0),  Type(1.0),  w_pen);
  nll += range_penalty(etaF,         Type(1.0),  Type(3.0),  w_pen);
  nll += range_penalty(etaS,         Type(1.0),  Type(3.0),  w_pen);
  nll += range_penalty(h,            Type(0.0),  Type(1.0),  w_pen);
  nll += range_penalty(qF,           Type(0.0),  Type(1.0),  w_pen);
  nll += range_penalty(qS,           Type(0.0),  Type(1.0),  w_pen);
  nll += range_penalty(sigma_cots,   Type(0.01), Type(2.0),  w_pen);
  nll += range_penalty(sigma_fast,   Type(0.01), Type(2.0),  w_pen);
  nll += range_penalty(sigma_slow,   Type(0.01), Type(2.0),  w_pen);
  // Juvenile stage penalties (resource-modulated maturation)
  nll += range_penalty(mJ_min,       Type(0.0),  Type(1.0),  w_pen);
  nll += range_penalty(mJ_max,       Type(0.0),  Type(1.0),  w_pen);
  // Encourage mJ_max >= mJ_min smoothly
  nll += w_pen * pow(pospart(mJ_min - mJ_max), 2);
  nll += range_penalty(muJ,          Type(0.0),  Type(3.0),  w_pen);

  // New: penalties for recruitment environmental random effect hyperparameters
  nll += range_penalty(sigma_rec_env, Type(0.001), Type(2.0),  w_pen);
  nll += range_penalty(ar1_rec_phi,   Type(-0.99), Type(0.99), w_pen);

  // New: penalties for initial states
  nll += range_penalty(C0, Type(0.0),  Type(10.0),  w_pen);
  nll += range_penalty(F0, Type(0.0),  Type(100.0), w_pen);
  nll += range_penalty(S0, Type(0.0),  Type(100.0), w_pen);

  // Effective observation SDs (floor-added in quadrature for smoothness)
  Type s_cots = CppAD::sqrt(sigma_cots * sigma_cots + sd_floor * sd_floor); // log-space SD for COTS
  Type s_fast = CppAD::sqrt(sigma_fast * sigma_fast + sd_floor * sd_floor); // logit-space SD for fast coral
  Type s_slow = CppAD::sqrt(sigma_slow * sigma_slow + sd_floor * sd_floor); // logit-space SD for slow coral

  // STATE PREDICTIONS
  vector<Type> cots_pred(T); // predicted adult COTS abundance (ind/m^2)
  vector<Type> fast_pred(T); // predicted fast coral cover (%)
  vector<Type> slow_pred(T); // predicted slow coral cover (%)
  vector<Type> juv_pred(T);  // predicted juvenile COTS abundance (ind/m^2)
  vector<Type> mJ_eff_vec(T); // effective maturation for diagnostics

  // Random effects likelihood: AR(1) on u_rec
  {
    Type sigma = CppAD::sqrt(sigma_rec_env * sigma_rec_env + Type(1e-12));
    // Initial state
    nll -= dnorm(u_rec(0), Type(0.0), sigma, true);
    // Transitions
    for (int t = 1; t < T; t++) {
      Type sd = sigma * CppAD::sqrt(Type(1.0) - ar1_rec_phi * ar1_rec_phi + eps);
      nll -= dnorm(u_rec(t), ar1_rec_phi * u_rec(t - 1), sd, true);
    }
  }

  // Initialize predictions at t=0 from initial-state parameters (no use of *_dat)
  cots_pred(0) = pospart(C0);
  fast_pred(0) = softmin(pospart(F0), Type(100.0));
  slow_pred(0) = softmin(pospart(S0), Type(100.0));

  // Initialize juvenile pool at t=0 from deterministic larval input using t=0 covariates (no data leakage)
  {
    Type C0p = cots_pred(0) + eps;
    Type F0p = pospart(fast_pred(0));
    Type S0p = pospart(slow_pred(0));
    Type R0 = wF * F0p + wS * S0p;
    Type f_food0 = R0 / (K_R + R0 + eps);
    Type f_Allee0 = Type(1.0) / (Type(1.0) + exp(-k_allee * (C0p - C_allee)));
    Type dT0 = sst_dat(0) - T_opt_rec;
    Type f_Trec0 = exp(-beta_rec * dT0 * dT0);
    Type env_mod0 = exp(u_rec(0));
    Type Rec_in0 = alpha_rec * pow(C0p, phi) * f_Allee0 * f_food0 * f_Trec0 * env_mod0 + cotsimm_dat(0);
    Type mJ_eff0 = mJ_min + (mJ_max - mJ_min) * f_food0;
    mJ_eff_vec(0) = mJ_eff0;
    juv_pred(0) = pospart(Rec_in0 / (mJ_eff0 + muJ + eps)); // quasi steady-state for J at t=0
  }

  // TIME LOOP: explicit prediction equations using t-1 states for t >= 1
  for (int t = 1; t < T; t++) {
    // Previous states (predictions)
    Type C_prev = pospart(cots_pred(t - 1));
    Type F_prev = pospart(fast_pred(t - 1));
    Type S_prev = pospart(slow_pred(t - 1));
    Type J_prev = pospart(juv_pred(t - 1));

    // Covariates at t-1 and environmental modifier
    Type SST = sst_dat(t - 1);
    Type imm = cotsimm_dat(t - 1);
    Type env_mod = exp(u_rec(t - 1)); // unobserved episodic recruitment driver

    // Resource and modifiers (adult condition and maturation)
    Type R = wF * F_prev + wS * S_prev;
    Type f_food = R / (K_R + R + eps);
    Type f_Allee = Type(1.0) / (Type(1.0) + exp(-k_allee * (C_prev - C_allee)));
    Type dT = SST - T_opt_rec;
    Type f_Trec = exp(-beta_rec * dT * dT);

    // Recruitment to juveniles and maturation
    Type Rec_in = alpha_rec * pow(C_prev + eps, phi) * f_Allee * f_food * f_Trec * env_mod + imm;
    Type mJ_eff = mJ_min + (mJ_max - mJ_min) * f_food;
    mJ_eff_vec(t) = mJ_eff;

    // Juveniles update (non-negative)
    Type J_next = (Type(1.0) - mJ_eff - muJ) * J_prev + Rec_in;
    J_next = pospart(J_next);

    // Adult mortality and update (non-negative)
    Type Mort = (muC + gammaC * C_prev) * C_prev;
    Type C_next = C_prev + mJ_eff * J_prev - Mort;
    C_next = pospart(C_next);

    // Coral growth modifiers (shared space K_tot) and bleaching
    Type total_coral = F_prev + S_prev;
    Type space_factor = Type(1.0) - total_coral / (K_tot + eps);
    Type heat_excess = pospart(SST - T_opt_bleach);
    Type growth_mult = exp(-beta_bleach * heat_excess);

    Type G_fast = rF * F_prev * space_factor * growth_mult;
    Type G_slow = rS * S_prev * space_factor * growth_mult;

    // Multi-prey functional response consumption
    Type F_eta = CppAD::pow(F_prev + eps, etaF);
    Type S_eta = CppAD::pow(S_prev + eps, etaS);
    Type denom = Type(1.0) + h * (aF * F_eta + aS * S_eta);
    Type Cons_fast = qF * (aF * F_eta * C_prev) / (denom + eps);
    Type Cons_slow = qS * (aS * S_eta * C_prev) / (denom + eps);

    // Additional bleaching losses (proportional per °C above threshold)
    Type B_fast = m_bleachF * heat_excess * F_prev;
    Type B_slow = m_bleachS * heat_excess * S_prev;

    // Coral updates, bounded to [0,100]
    Type F_next = F_prev + G_fast - Cons_fast - B_fast;
    Type S_next = S_prev + G_slow - Cons_slow - B_slow;
    F_next = softmin(pospart(F_next), Type(100.0));
    S_next = softmin(pospart(S_next), Type(100.0));

    // Assign predictions for year t
    cots_pred(t) = C_next;
    fast_pred(t) = F_next;
    slow_pred(t) = S_next;
    juv_pred(t)  = J_next;
  }

  // ------------------------
  // LIKELIHOOD
  // ------------------------
  // Observation model:
  // - COTS adults: lognormal on abundance
  // - Corals: normal on logit(% cover)
  for (int t = 0; t < T; t++) {
    // COTS
    Type log_obs_c = log(cots_dat(t) + eps);
    Type log_pred_c = log(cots_pred(t) + eps);
    nll -= dnorm(log_obs_c, log_pred_c, s_cots, true);

    // Fast coral
    Type fast_bounded = softmin(pospart(fast_pred(t)), Type(100.0));
    Type logit_obs_f = logit_pct(fast_dat(t));
    Type logit_pred_f = logit_pct(fast_bounded);
    nll -= dnorm(logit_obs_f, logit_pred_f, s_fast, true);

    // Slow coral
    Type slow_bounded = softmin(pospart(slow_pred(t)), Type(100.0));
    Type logit_obs_s = logit_pct(slow_dat(t));
    Type logit_pred_s = logit_pct(slow_bounded);
    nll -= dnorm(logit_obs_s, logit_pred_s, s_slow, true);
  }

  // ------------------------
  // REPORTS
  // ------------------------
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(juv_pred);
  REPORT(mJ_eff_vec);
  REPORT(u_rec);

  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);
  ADREPORT(juv_pred);
  ADREPORT(mJ_eff_vec);
  ADREPORT(u_rec);

  return nll;
}
