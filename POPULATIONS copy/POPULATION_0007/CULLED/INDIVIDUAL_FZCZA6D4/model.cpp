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

  // Allee effect parameters (new)
  PARAMETER(A50_C);            // ind m^-2 | Half-saturation density for Allee multiplier on COTS positive growth
  PARAMETER(eta_Allee);        // dimensionless | Steepness of Allee (Hill) function

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
  Type eps = Type(1e-8);

  int N = Year.size();

  // Transformed parameters
  Type sigma_T_C = exp(log_sigma_T_C);
  Type sigma_T_F = exp(log_sigma_T_F);
  Type sigma_T_S = exp(log_sigma_T_S);

  Type sd_cots = exp(log_sd_cots);
  Type sd_fast = exp(log_sd_fast);
  Type sd_slow = exp(log_sd_slow);

  Type penalty_w = exp(log_penalty_w);

  Type p_fast = inv_logit(p_fast_logit);

  // State vectors (predictions)
  vector<Type> cots_pred(N);  // COTS density
  vector<Type> fast_pred(N);  // Fast coral (% cover)
  vector<Type> slow_pred(N);  // Slow coral (% cover)

  // Diagnostics
  vector<Type> H_food_series(N);
  vector<Type> A_alle_series(N);
  vector<Type> O_series(N);
  vector<Type> E_temp_C_series(N);
  vector<Type> E_temp_F_series(N);
  vector<Type> E_temp_S_series(N);
  vector<Type> g_tot_series(N);
  vector<Type> frac_fast_series(N);

  // Initialize states using first observations (allowed as initial conditions without leakage)
  cots_pred(0) = CppAD::CondExpGt(cots_dat(0), eps, cots_dat(0), eps);
  fast_pred(0) = CppAD::CondExpGt(fast_dat(0), eps, fast_dat(0), eps);
  slow_pred(0) = CppAD::CondExpGt(slow_dat(0), eps, slow_dat(0), eps);

  // Compute diagnostics for t = 0 (based on initial states and forcings at t=0)
  {
    Type F0 = fast_pred(0);
    Type S0 = slow_pred(0);
    Type C0 = cots_pred(0);
    Type T0 = sst_dat(0);

    // Food saturation (weighted by prey preference)
    Type foodW = p_fast * F0 + (Type(1) - p_fast) * S0;
    Type H_food0 = foodW / (foodW + K_food);
    H_food_series(0) = H_food0;

    // Temperature effects
    Type E_C0 = exp(-Type(0.5) * square((T0 - T_opt_C) / sigma_T_C));
    Type E_F0 = exp(-Type(0.5) * square((T0 - T_opt_F) / sigma_T_F));
    Type E_S0 = exp(-Type(0.5) * square((T0 - T_opt_S) / sigma_T_S));
    E_temp_C_series(0) = E_C0;
    E_temp_F_series(0) = E_F0;
    E_temp_S_series(0) = E_S0;

    // Outbreak amplifier (logistic) adds to 1
    Type logistic0 = Type(1) / (Type(1) + exp(-k_outbreak * (T0 - T_thr)));
    Type O0 = Type(1) + amp_outbreak * logistic0;
    O_series(0) = O0;

    // Allee multiplier
    Type Cpow = pow(C0 + eps, eta_Allee);
    Type A0 = Cpow / (Cpow + pow(A50_C + eps, eta_Allee));
    A_alle_series(0) = A0;

    // Functional response diagnostics
    Type Z = p_fast * pow(F0, theta_FR) + (Type(1) - p_fast) * pow(S0, theta_FR);
    Type g_tot0 = a_feed * Z / (Type(1) + h_feed * Z);
    g_tot_series(0) = g_tot0;
    Type frac_fast0 = CppAD::CondExpGt(Z, eps, (p_fast * pow(F0, theta_FR)) / Z, Type(0));
    frac_fast_series(0) = frac_fast0;
  }

  // State transition loop (use only previous-step predicted states)
  for (int t = 0; t < N - 1; t++) {
    Type C_prev = cots_pred(t);
    Type F_prev = fast_pred(t);
    Type S_prev = slow_pred(t);
    Type T_prev = sst_dat(t); // Forcings at time t

    // Food saturation (weighted by prey preference)
    Type foodW = p_fast * F_prev + (Type(1) - p_fast) * S_prev;
    Type H_food = foodW / (foodW + K_food);
    H_food_series(t) = H_food;

    // Temperature effects (Gaussian)
    Type E_C = exp(-Type(0.5) * square((T_prev - T_opt_C) / sigma_T_C));
    Type E_F = exp(-Type(0.5) * square((T_prev - T_opt_F) / sigma_T_F));
    Type E_S = exp(-Type(0.5) * square((T_prev - T_opt_S) / sigma_T_S));
    E_temp_C_series(t) = E_C;
    E_temp_F_series(t) = E_F;
    E_temp_S_series(t) = E_S;

    // Outbreak amplifier (logistic) adds to 1
    Type logistic_prev = Type(1) / (Type(1) + exp(-k_outbreak * (T_prev - T_thr)));
    Type O_prev = Type(1) + amp_outbreak * logistic_prev;
    O_series(t) = O_prev;

    // Allee multiplier
    Type Cpow = pow(C_prev + eps, eta_Allee);
    Type A_alle = Cpow / (Cpow + pow(A50_C + eps, eta_Allee));
    A_alle_series(t) = A_alle;

    // Mortality including food-scarcity component
    Type mC = m0_C + m_food_C * (Type(1) - H_food);

    // COTS net per-capita rate (Ricker with Allee and modifiers)
    Type r_eff = r_C * H_food * E_C * O_prev * A_alle - mC;

    // Update COTS (Ricker with immigration), ensure non-negative
    Type C_next = C_prev * exp(r_eff - a_C * C_prev) + cotsimm_dat(t);
    C_next = CppAD::CondExpGt(C_next, eps, C_next, eps);

    // Multi-prey functional response (Type III-like)
    Type Z = p_fast * pow(F_prev, theta_FR) + (Type(1) - p_fast) * pow(S_prev, theta_FR);
    Type g_tot = a_feed * Z / (Type(1) + h_feed * Z); // per-capita feeding rate (yr^-1)
    g_tot_series(t) = g_tot;
    Type frac_fast = CppAD::CondExpGt(Z, eps, (p_fast * pow(F_prev, theta_FR)) / Z, Type(0));
    frac_fast_series(t) = frac_fast;

    // Coral loss due to predation (convert to % cover loss with efficiencies)
    Type loss_fast = eff_f * C_prev * g_tot * frac_fast;
    Type loss_slow = eff_s * C_prev * g_tot * (Type(1) - frac_fast);

    // Coral logistic growth with shared carrying capacity
    Type total_prev = F_prev + S_prev;
    Type space_lim = (Type(1) - total_prev / (K_tot + eps));
    space_lim = CppAD::CondExpLt(space_lim, Type(-5.0), Type(-5.0), space_lim); // guard extreme negatives

    Type F_next = F_prev + r_F * E_F * F_prev * space_lim - loss_fast;
    Type S_next = S_prev + r_S * E_S * S_prev * space_lim - loss_slow;

    // Enforce non-negativity
    F_next = CppAD::CondExpGt(F_next, eps, F_next, eps);
    S_next = CppAD::CondExpGt(S_next, eps, S_next, eps);

    // Assign next states
    cots_pred(t + 1) = C_next;
    fast_pred(t + 1) = F_next;
    slow_pred(t + 1) = S_next;

    // For completeness, precompute diagnostics for t+1 using available states/forcings at t+1
    // (not used in transitions, only for reporting)
    if (t + 1 < N) {
      Type F1 = fast_pred(t + 1);
      Type S1 = slow_pred(t + 1);
      Type C1 = cots_pred(t + 1);
      Type T1 = sst_dat(t + 1);

      Type foodW1 = p_fast * F1 + (Type(1) - p_fast) * S1;
      H_food_series(t + 1) = foodW1 / (foodW1 + K_food);

      E_temp_C_series(t + 1) = exp(-Type(0.5) * square((T1 - T_opt_C) / sigma_T_C));
      E_temp_F_series(t + 1) = exp(-Type(0.5) * square((T1 - T_opt_F) / sigma_T_F));
      E_temp_S_series(t + 1) = exp(-Type(0.5) * square((T1 - T_opt_S) / sigma_T_S));

      Type logistic1 = Type(1) / (Type(1) + exp(-k_outbreak * (T1 - T_thr)));
      O_series(t + 1) = Type(1) + amp_outbreak * logistic1;

      Type Cpow1 = pow(C1 + eps, eta_Allee);
      A_alle_series(t + 1) = Cpow1 / (Cpow1 + pow(A50_C + eps, eta_Allee));

      Type Z1 = p_fast * pow(F1, theta_FR) + (Type(1) - p_fast) * pow(S1, theta_FR);
      g_tot_series(t + 1) = a_feed * Z1 / (Type(1) + h_feed * Z1);
      frac_fast_series(t + 1) = CppAD::CondExpGt(Z1, eps, (p_fast * pow(F1, theta_FR)) / Z1, Type(0));
    }
  }

  // -----------------------------
  // Likelihood
  // -----------------------------
  Type nll = Type(0.0);

  for (int t = 0; t < N; t++) {
    // Observation likelihoods (lognormal on log scale)
    Type yc = log(CppAD::CondExpGt(cots_dat(t), eps, cots_dat(t), eps));
    Type ycf = log(CppAD::CondExpGt(fast_dat(t), eps, fast_dat(t), eps));
    Type ycs = log(CppAD::CondExpGt(slow_dat(t), eps, slow_dat(t), eps));

    Type muc = log(CppAD::CondExpGt(cots_pred(t), eps, cots_pred(t), eps));
    Type muf = log(CppAD::CondExpGt(fast_pred(t), eps, fast_pred(t), eps));
    Type mus = log(CppAD::CondExpGt(slow_pred(t), eps, slow_pred(t), eps));

    nll -= dnorm(yc, muc, sd_cots, true);
    nll -= dnorm(ycf, muf, sd_fast, true);
    nll -= dnorm(ycs, mus, sd_slow, true);

    // Soft penalties for biological plausibility
    Type total_coral = fast_pred(t) + slow_pred(t);
    Type over = CppAD::CondExpGt(total_coral - K_tot, Type(0), total_coral - K_tot, Type(0));
    nll += penalty_w * square(over / (K_tot + eps));
  }

  // -----------------------------
  // Reporting
  // -----------------------------
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(H_food_series);
  REPORT(A_alle_series);
  REPORT(O_series);
  REPORT(E_temp_C_series);
  REPORT(E_temp_F_series);
  REPORT(E_temp_S_series);
  REPORT(g_tot_series);
  REPORT(frac_fast_series);

  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);

  return nll;
}
