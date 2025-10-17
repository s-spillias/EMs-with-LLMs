#include <TMB.hpp>

// Helper functions
template<class Type>
Type square(Type x) { return x * x; }

template<class Type>
Type inv_logit(Type x) { return Type(1) / (Type(1) + exp(-x)); }

template<class Type>
Type softplus(Type x) {
  // Numerically stable softplus using AD-compatible ops (avoid log1p with AD Type)
  // softplus(x) = log(1 + exp(x)) = { x + log(1 + exp(-x)) if x > 0; log(1 + exp(x)) otherwise }
  Type zero = Type(0);
  Type pos = x + log(Type(1) + exp(-x));
  Type neg = log(Type(1) + exp(x));
  return CppAD::CondExpGt(x, zero, pos, neg);
}

// Model
template<class Type>
Type objective_function<Type>::operator() () {
  // -----------------------------
  // DATA INPUTS (TMB conventions)
  // -----------------------------
  DATA_VECTOR(Year);         // Year (calendar year; used for reporting and alignment)
  DATA_VECTOR(cots_dat);     // Crown-of-thorns starfish density (individuals m^-2), strictly positive
  DATA_VECTOR(fast_dat);     // Fast-growing coral cover (Acropora spp.) in percent (% cover), positive
  DATA_VECTOR(slow_dat);     // Slow-growing coral cover (Faviidae/Porites) in percent (% cover), positive
  DATA_VECTOR(sst_dat);      // Sea-surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);  // COTS larval immigration rate (individuals m^-2 yr^-1)

  // -----------------------------------
  // PARAMETERS (process + observation)
  // -----------------------------------
  // COTS population dynamics
  PARAMETER(r_C);              // year^-1 | Intrinsic per-capita growth rate (baseline fecundity/survival) for COTS; initial estimate calibrated to observed outbreak rise rates
  PARAMETER(a_C);              // (m^2 ind^-1) | Ricker density-dependence strength for COTS (self-limitation); larger values accelerate bust after peaks
  PARAMETER(m0_C);             // year^-1 | Baseline COTS mortality unrelated to food
  PARAMETER(m_food_C);         // year^-1 | Additional COTS mortality when food is scarce (scaled by 1 - food_saturation)
  PARAMETER(K_food);           // % cover | Half-saturation constant for food limitation (weighted coral index where H_food = 0.5)

  // Temperature effects on COTS reproduction/outbreaks
  PARAMETER(T_opt_C);          // °C | Optimal SST for COTS reproduction/survival (peak of Gaussian response)
  PARAMETER(log_sigma_T_C);    // log(°C) | Log of SD of temperature response for COTS (ensures positivity)
  PARAMETER(amp_outbreak);     // dimensionless | Amplitude of additional outbreak multiplier at warm temps
  PARAMETER(k_outbreak);       // (°C^-1) | Steepness of logistic outbreak amplifier vs temperature
  PARAMETER(T_thr);            // °C | Temperature midpoint (threshold) of outbreak amplifier

  // Feeding/functional response and prey preference
  PARAMETER(p_fast_logit);     // logit(p) | Logit of COTS preference for fast coral (0-1 after inverse-logit); bias feeding toward Acropora
  PARAMETER(a_feed);           // (yr^-1 %^{-theta}) | Attack/encounter rate scaling in multi-prey Holling response
  PARAMETER(h_feed);           // (yr %^{theta}) | Handling-time-like parameter in functional response denominator
  PARAMETER(theta_FR);         // dimensionless (>=1) | Shape exponent for Type III-like response (>=1 smooth thresholding)

  // Efficiency of translating feeding to coral loss (process-specific efficiencies)
  PARAMETER(eff_f);            // (% cover per (ind m^-2 yr)) | Efficiency mapping feeding on fast coral to % cover loss
  PARAMETER(eff_s);            // (% cover per (ind m^-2 yr)) | Efficiency mapping feeding on slow coral to % cover loss

  // Coral growth and carrying capacity
  PARAMETER(r_F);              // year^-1 | Intrinsic growth rate of fast coral (Acropora)
  PARAMETER(r_S);              // year^-1 | Intrinsic growth rate of slow coral (Faviidae/Porites)
  PARAMETER(K_tot);            // % cover | Shared space-limited carrying capacity for total live coral (fast + slow)

  // Temperature responses for corals
  PARAMETER(T_opt_F);          // °C | Optimal SST for fast coral growth
  PARAMETER(log_sigma_T_F);    // log(°C) | Log of SD of temperature response for fast coral
  PARAMETER(T_opt_S);          // °C | Optimal SST for slow coral growth
  PARAMETER(log_sigma_T_S);    // log(°C) | Log of SD of temperature response for slow coral

  // Observation model (lognormal SDs on log scale)
  PARAMETER(log_sd_cots);      // log | Log of observation/process SD (log-scale) for cots_dat
  PARAMETER(log_sd_fast);      // log | Log of observation/process SD (log-scale) for fast_dat
  PARAMETER(log_sd_slow);      // log | Log of observation/process SD (log-scale) for slow_dat

  // Soft penalty weight (used to softly discourage biologically implausible states)
  PARAMETER(log_penalty_w);    // log | Log of penalty weight for bounds on state variables

  // -----------------------------
  // Derived/Transformed parameters
  // -----------------------------
  Type eps = Type(1e-8); // small constant for numerical stability
  int nT = cots_dat.size();

  // Time-varying predictions (initialized from data to avoid parameterized initial conditions)
  vector<Type> cots_pred(nT);  // ind m^-2
  vector<Type> fast_pred(nT);  // % cover
  vector<Type> slow_pred(nT);  // % cover

  // Additional reporting vectors for diagnostic rates
  vector<Type> cons_fast_vec(nT); // % cover loss per year attributed to fast coral consumption
  vector<Type> cons_slow_vec(nT); // % cover loss per year attributed to slow coral consumption
  vector<Type> H_food_vec(nT);    // Food limitation (0-1)
  vector<Type> outbreak_mult(nT); // Outbreak amplifier (>=1)
  vector<Type> temp_eff_C(nT);    // Temperature effect on COTS (0-1)
  vector<Type> temp_eff_F(nT);    // Temperature effect on fast coral (0-1)
  vector<Type> temp_eff_S(nT);    // Temperature effect on slow coral (0-1)

  // Transformations to enforce positivity and probabilities where needed
  Type sigma_T_C = exp(log_sigma_T_C); // °C
  Type sigma_T_F = exp(log_sigma_T_F); // °C
  Type sigma_T_S = exp(log_sigma_T_S); // °C

  Type sdlog_cots = exp(log_sd_cots);  // log-scale SD
  Type sdlog_fast = exp(log_sd_fast);  // log-scale SD
  Type sdlog_slow = exp(log_sd_slow);  // log-scale SD
  Type min_sdlog = Type(0.05);         // minimum log-scale SD (prevents overconfidence)

  Type penalty_w = exp(log_penalty_w); // penalty scaling > 0
  Type p_fast = inv_logit(p_fast_logit); // preference in [0,1]

  // -----------------------------
  // INITIAL CONDITIONS (from data)
  // -----------------------------
  // Use first observed values directly to initialize predictions (no data leakage in subsequent steps)
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Initialize diagnostics at t=0 for reporting (set using previous-step definitions)
  temp_eff_C(0) = exp(-Type(0.5) * square((sst_dat(0) - T_opt_C) / (sigma_T_C + eps)));
  outbreak_mult(0) = Type(1) + amp_outbreak * inv_logit(k_outbreak * (sst_dat(0) - T_thr));
  H_food_vec(0) = (p_fast * fast_dat(0) + (Type(1) - p_fast) * slow_dat(0)) /
                  (K_food + p_fast * fast_dat(0) + (Type(1) - p_fast) * slow_dat(0) + eps);
  temp_eff_F(0) = exp(-Type(0.5) * square((sst_dat(0) - T_opt_F) / (sigma_T_F + eps)));
  temp_eff_S(0) = exp(-Type(0.5) * square((sst_dat(0) - T_opt_S) / (sigma_T_S + eps)));
  cons_fast_vec(0) = Type(0);
  cons_slow_vec(0) = Type(0);

  // -----------------------------
  // PROCESS MODEL (discrete time)
  // -----------------------------
  // Numbered equations (all use previous-step state values to avoid data leakage):
  // (1) Food limitation index (H_food_t): H_food = (w_f * F + w_s * S) / (K_food + w_f * F + w_s * S)
  // (2) Temperature effects: E_C(T) = exp(-0.5 * ((T - T_opt_C)/sigma_T_C)^2); E_F/S similarly with their optima
  // (3) Outbreak amplifier: O(T) = 1 + amp_outbreak * inv_logit(k_outbreak * (T - T_thr))
  // (4) COTS Ricker with food and temperature: C_{t+1} = C_t * exp( r_C * H_food * E_C * O - m0_C - m_food_C*(1 - H_food) - a_C * C_t ) + I_t
  // (5) Multi-prey Holling response (Type II/III mix):
  //     Avail = p_fast*F^theta + (1-p_fast)*S^theta
  //     cons_per_C = a_feed * Avail / (1 + a_feed*h_feed*Avail)
  //     Allocation: cons_fast = cons_per_C * (p_fast*F^theta / (Avail)); cons_slow similar
  // (6) Coral dynamics (logistic growth with shared capacity and temperature effects minus predation losses):
  //     F_{t+1} = F_t + r_F*F_t*(1 - (F_t + S_t)/K_tot) * E_F - eff_f * cons_fast * C_t
  //     S_{t+1} = S_t + r_S*S_t*(1 - (F_t + S_t)/K_tot) * E_S - eff_s * cons_slow * C_t

  for (int t = 1; t < nT; t++) {
    // Previous-step states (predicted, no data leakage)
    Type C_prev = cots_pred(t - 1);
    Type F_prev = fast_pred(t - 1);
    Type S_prev = slow_pred(t - 1);

    // Forcing at previous step (smooth causality)
    Type T_prev = sst_dat(t - 1);
    Type I_prev = cotsimm_dat(t - 1);

    // (1) Food limitation (weighted coral index)
    Type H_food = (p_fast * F_prev + (Type(1) - p_fast) * S_prev) /
                  (K_food + p_fast * F_prev + (Type(1) - p_fast) * S_prev + eps);
    H_food_vec(t) = H_food;

    // (2) Temperature effects (Gaussian around optima)
    Type E_C = exp(-Type(0.5) * square((T_prev - T_opt_C) / (sigma_T_C + eps)));
    Type E_F = exp(-Type(0.5) * square((T_prev - T_opt_F) / (sigma_T_F + eps)));
    Type E_S = exp(-Type(0.5) * square((T_prev - T_opt_S) / (sigma_T_S + eps)));
    temp_eff_C(t) = E_C;
    temp_eff_F(t) = E_F;
    temp_eff_S(t) = E_S;

    // (3) Outbreak amplifier
    Type O_prev = Type(1) + amp_outbreak * inv_logit(k_outbreak * (T_prev - T_thr));
    outbreak_mult(t) = O_prev;

    // (4) COTS Ricker with food- and temperature-modified r, extra mortality under low food, plus immigration
    Type mC = m0_C + m_food_C * (Type(1) - H_food);
    Type r_eff = r_C * H_food * E_C * O_prev - mC; // net per-capita rate (without self-limitation)
    Type C_next = C_prev * exp(r_eff - a_C * C_prev) + I_prev;

    // (5) Multi-prey Holling consumption with Type III-like smooth thresholding and preference
    Type Fterm = pow(F_prev + eps, theta_FR);
    Type Sterm = pow(S_prev + eps, theta_FR);
    Type Avail = p_fast * Fterm + (Type(1) - p_fast) * Sterm;                // preference-weighted availability
    Type cons_per_C = a_feed * Avail / (Type(1) + a_feed * h_feed * Avail + eps); // per-capita feeding rate (yr^-1)
    // Allocation across prey (share by weighted availability)
    Type alloc_fast = (p_fast * Fterm) / (Avail + eps);
    Type alloc_slow = (Type(1) - p_fast) * Sterm / (Avail + eps);
    Type cons_fast = cons_per_C * alloc_fast; // per COTS on fast coral
    Type cons_slow = cons_per_C * alloc_slow; // per COTS on slow coral
    cons_fast_vec(t) = eff_f * cons_fast * C_prev; // realized % cover loss per year on fast coral
    cons_slow_vec(t) = eff_s * cons_slow * C_prev; // realized % cover loss per year on slow coral

    // (6) Coral population dynamics with shared space limitation and temperature modulation
    Type crowd = (F_prev + S_prev) / (K_tot + eps);
    Type F_next = F_prev + r_F * F_prev * (Type(1) - crowd) * E_F - cons_fast_vec(t);
    Type S_next = S_prev + r_S * S_prev * (Type(1) - crowd) * E_S - cons_slow_vec(t);

    // Assign to prediction vectors
    cots_pred(t) = C_next;
    fast_pred(t) = F_next;
    slow_pred(t) = S_next;
  }

  // -----------------------------
  // LIKELIHOOD (lognormal errors)
  // -----------------------------
  Type nll = Type(0);

  // Observation model applies to all time points including t=0
  for (int t = 0; t < nT; t++) {
    // ensure strictly positive arguments to log
    Type yC = cots_dat(t);
    Type muC = cots_pred(t);
    Type yF = fast_dat(t);
    Type muF = fast_pred(t);
    Type yS = slow_dat(t);
    Type muS = slow_pred(t);

    // Minimum log-scale SDs to avoid overconfidence
    Type sC = sdlog_cots + min_sdlog;
    Type sF = sdlog_fast + min_sdlog;
    Type sS = sdlog_slow + min_sdlog;

    // Lognormal likelihoods
    nll -= dnorm(log(yC + eps), log(muC + eps), sC, true);
    nll -= dnorm(log(yF + eps), log(muF + eps), sF, true);
    nll -= dnorm(log(yS + eps), log(muS + eps), sS, true);

    // Soft state bounds penalties (smooth, no hard truncation)
    // Penalize negative or overly large coral cover and negative COTS
    nll += penalty_w * square(softplus(-muC));               // COTS should be >= 0
    nll += penalty_w * (square(softplus(-muF)) + square(softplus(muF - K_tot))); // fast coral within [0, K_tot]
    nll += penalty_w * (square(softplus(-muS)) + square(softplus(muS - K_tot))); // slow coral within [0, K_tot]
    // Penalize total coral exceeding capacity (softly)
    nll += penalty_w * square(softplus((muF + muS) - K_tot));
  }

  // -----------------------------
  // REPORTING
  // -----------------------------
  REPORT(Year);
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(H_food_vec);
  REPORT(outbreak_mult);
  REPORT(temp_eff_C);
  REPORT(temp_eff_F);
  REPORT(temp_eff_S);
  REPORT(cons_fast_vec);
  REPORT(cons_slow_vec);

  return nll;
}
