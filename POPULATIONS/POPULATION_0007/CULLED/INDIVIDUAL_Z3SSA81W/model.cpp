#include <TMB.hpp>

// Numerically stable softplus: log(1 + exp(x)) implemented without overflow
// Uses conditional expression compatible with CppAD automatic differentiation.
template<class Type>
Type softplus(Type x) {
  Type zero = Type(0.0);
  // If x > 0: x + log(1 + exp(-x)) (safe since exp(-x) <= 1)
  // Else    : log(1 + exp(x))       (safe since exp(x) <= 1)
  return CppAD::CondExpGt(
    x, zero,
    x + log(Type(1.0) + exp(-x)),
    log(Type(1.0) + exp(x))
  );
}

// Smooth lower bound to avoid negative states (numerically stable "soft max")
// Returns a value >= lb with a smooth transition around lb.
template<class Type>
Type soft_lower_bound(Type x, Type lb) {
  Type k = Type(10.0);                                  // Smoothness parameter (higher = sharper)
  return lb + softplus(k * (x - lb)) / k;               // Smoothly approximates max(x, lb)
}

// Smooth penalty for keeping a parameter within [lower, upper] without hard constraints.
template<class Type>
Type smooth_bound_penalty(Type x, Type lower, Type upper) {
  Type k = Type(10.0);                                  // Smoothness parameter
  Type pen_low  = softplus(k * (lower - x)) / k;        // Penalty increases smoothly if x < lower
  Type pen_high = softplus(k * (x - upper)) / k;        // Penalty increases smoothly if x > upper
  return pen_low + pen_high;
}

template<class Type>
Type objective_function<Type>::operator() () {
  // -----------------------------
  // DATA (must be provided by user; time-aligned across vectors)
  // -----------------------------
  DATA_VECTOR(Year);        // Year (calendar year, e.g., 1980..2005)
  DATA_VECTOR(cots_dat);    // COTS adult abundance (individuals per m^2), strictly positive
  DATA_VECTOR(fast_dat);    // Fast-growing coral (Acropora) live cover (%), [0, 100]
  DATA_VECTOR(slow_dat);    // Slow-growing coral (Faviidae + Porites) live cover (%), [0, 100]
  DATA_VECTOR(sst_dat);     // Sea Surface Temperature (°C)
  DATA_VECTOR(cotsimm_dat); // Larval immigration rate (individuals per m^2 per year)

  int N = Year.size();      // Number of time steps (years)

  // -----------------------------
  // PARAMETERS (ecological and statistical)
  // -----------------------------

  // Coral intrinsic growth rates (year^-1)
  PARAMETER(r_fast);     // Intrinsic growth rate of Acropora (% cover per % cover per year)
  PARAMETER(r_slow);     // Intrinsic growth rate of Faviidae/Porites (% cover per % cover per year)

  // Coral carrying capacities (% cover)
  PARAMETER(K_fast);     // Carrying capacity for Acropora (% cover, <= 100)
  PARAMETER(K_slow);     // Carrying capacity for slow corals (% cover, <= 100)

  // Intergroup competition coefficients (dimensionless)
  PARAMETER(alpha_fs);   // Effect of slow coral on fast (Acropora) logistic saturation (0..1 typical)
  PARAMETER(alpha_sf);   // Effect of fast coral on slow logistic saturation (0..1 typical)

  // Background (non-bleaching, non-predation) mortalities (year^-1)
  PARAMETER(m_fast);     // Background mortality rate of fast coral (year^-1)
  PARAMETER(m_slow);     // Background mortality rate of slow coral (year^-1)

  // Temperature-driven bleaching mortality parameters
  PARAMETER(T_bleach);        // SST threshold for bleaching onset (°C)
  PARAMETER(b_bleach);        // Steepness of bleaching logistic response (1/°C)
  PARAMETER(mu_bleach_fast);  // Max additional bleaching mortality for fast coral (year^-1)
  PARAMETER(mu_bleach_slow);  // Max additional bleaching mortality for slow coral (year^-1)

  // COTS feeding selectivity and functional response (Holling type III)
  PARAMETER(p_fast_raw);      // Logit-scale preference for Acropora (dimensionless, transforms to 0..1)
  Type p_fast = invlogit(p_fast_raw);           // Preference weight for fast coral (0..1)
  Type p_slow = Type(1.0) - p_fast;             // Complementary preference for slow coral

  PARAMETER(a_attack);        // Attack rate scaling (per COTS per year per [% cover]^q_FR)
  PARAMETER(h_handling);      // Handling time scaling (year per [% cover]^q_FR)
  PARAMETER(q_FR_raw);        // Log-scale for Type-III exponent: q_FR = 1 + exp(q_FR_raw) >= 1
  Type q_FR = Type(1.0) + exp(q_FR_raw);        // Functional response exponent (>= 1)

  // Conversion from consumption to coral loss (units align % cover loss per COTS consumption rate)
  PARAMETER(conv_pred_to_mort_fast); // Efficiency converting consumption to Acropora loss (dimensionless)
  PARAMETER(conv_pred_to_mort_slow); // Efficiency converting consumption to slow coral loss (dimensionless)

  // COTS population dynamics
  PARAMETER(r_cots);       // Maximum per-capita growth rate of COTS (year^-1)
  PARAMETER(m_cots);       // Natural mortality rate of COTS (year^-1)
  PARAMETER(beta_cots);    // Density-dependence parameter in Ricker term (m^2 per individual)
  PARAMETER(gamma_food_raw); // Logit-scale weight determining how strongly food limits growth (0..1)
  Type gamma_food = invlogit(gamma_food_raw); // 0=no food effect, 1=fully food-limited

  // Temperature performance curve for COTS (bell-shaped)
  PARAMETER(T_opt);          // Optimal SST for COTS performance (°C)
  PARAMETER(sigma_T_raw);    // Log-scale width of temperature performance (σ > 0)
  Type sigma_T = exp(sigma_T_raw) + Type(1e-8);  // Ensure strictly positive

  // Food limitation (Acropora-driven) on COTS growth and settlement
  PARAMETER(half_sat_food_raw); // Log-scale half-saturation coral cover for food limitation (K50, %)
  Type half_sat_food = exp(half_sat_food_raw);    // K50 for food effects (%, >0)
  PARAMETER(q_food_raw);       // Log-scale for food nonlinearity exponent: q_food = 1 + exp(q_food_raw)
  Type q_food = Type(1.0) + exp(q_food_raw);      // >= 1 for threshold-like response

  PARAMETER(settle_eff_raw);        // Logit-scale settlement efficiency multiplier (0..1)
  Type settle_eff = invlogit(settle_eff_raw);     // Settlement efficiency fraction (0..1)
  PARAMETER(settle_food_weight_raw);// Logit-scale weight of fast coral in settlement habitat (0..1)
  Type settle_food_weight = invlogit(settle_food_weight_raw); // 0..1

  // Allee (mate limitation) parameters for adult COTS reproduction
  PARAMETER(A50_allee_raw);   // Log-scale half-saturation density for mate limitation (m^-2)
  Type A50_allee = exp(A50_allee_raw);
  PARAMETER(k_allee_raw);     // Log-scale shaping exponent: k_allee = 1 + exp(k_allee_raw) >= 1
  Type k_allee = Type(1.0) + exp(k_allee_raw);

  // Observation error (lognormal on strictly positive data), floor added for stability
  PARAMETER(log_sigma_cots);  // log SD for COTS log-observation error
  PARAMETER(log_sigma_fast);  // log SD for fast coral log-observation error
  PARAMETER(log_sigma_slow);  // log SD for slow coral log-observation error
  Type sigma_floor = Type(0.05);                 // Minimum SD to avoid zero-variance issues
  Type sigma_cots = sigma_floor + exp(log_sigma_cots);
  Type sigma_fast = sigma_floor + exp(log_sigma_fast);
  Type sigma_slow = sigma_floor + exp(log_sigma_slow);

  // Small constants for numerical stability
  Type eps   = Type(1e-8);  // Prevent division by zero
  Type delta = Type(1e-6);  // Positive shift for lognormal on near-zero values

  // -----------------------------
  // STATE PREDICTIONS
  // -----------------------------
  vector<Type> cots_pred(N);         // Predicted COTS (indiv m^-2)
  vector<Type> fast_pred(N);         // Predicted Acropora cover (%)
  vector<Type> slow_pred(N);         // Predicted slow coral cover (%)

  // Auxiliary process trackers for diagnostics and reporting
  vector<Type> w_bleach(N);          // Bleaching intensity [0..1] as function of SST
  vector<Type> f_temp(N);            // COTS temperature performance [0..1]
  vector<Type> f_food(N);            // COTS food limitation [0..1] from Acropora
  vector<Type> pred_fast_flux(N);    // Predation loss from fast coral (% cover per year)
  vector<Type> pred_slow_flux(N);    // Predation loss from slow coral (% cover per year)
  vector<Type> cons_rate_per_cots(N);// Consumption rate per COTS (scaled, % per COTS per year)
  vector<Type> immig_add(N);         // Immigration contribution to COTS (indiv m^-2 per year)
  vector<Type> coral_avail_q(N);     // Availability index in FR (^[q_FR])
  vector<Type> f_allee(N);           // Mate limitation factor for adult COTS [0..1]

  // INITIAL CONDITIONS: set from observed data at t=0 (no data leakage in transition equations)
  cots_pred(0) = cots_dat(0);        // COTS initial state from data (indiv m^-2)
  fast_pred(0) = fast_dat(0);        // Acropora initial state from data (% cover)
  slow_pred(0) = slow_dat(0);        // Slow coral initial state from data (% cover)

  // Initialize auxiliaries at t=0 for complete reporting
  w_bleach(0) = Type(1.0) / (Type(1.0) + exp(-b_bleach * (sst_dat(0) - T_bleach)));                 // Logistic( b*(SST - T_bleach) )
  f_temp(0)   = exp(-Type(0.5) * pow((sst_dat(0) - T_opt) / sigma_T, 2.0));                         // Gaussian performance
  f_food(0)   = pow(fast_pred(0) + eps, q_food) / (pow(half_sat_food, q_food) + pow(fast_pred(0) + eps, q_food)); // Saturating food
  pred_fast_flux(0) = Type(0.0);
  pred_slow_flux(0) = Type(0.0);
  cons_rate_per_cots(0) = Type(0.0);
  immig_add(0) = settle_eff * cotsimm_dat(0) *
                 ( settle_food_weight * (fast_pred(0) / (half_sat_food + fast_pred(0) + eps)) +
                   (Type(1.0) - settle_food_weight) * (slow_pred(0) / (half_sat_food + slow_pred(0) + eps)) );
  coral_avail_q(0) = Type(0.0);
  f_allee(0) = pow(cots_pred(0) / (A50_allee + cots_pred(0) + eps), k_allee);

  // -----------------------------
  // PROCESS EQUATIONS (t = 1..N-1; only previous states used to predict current)
  // Numbered equations for clarity:
  //
  // (1) Bleaching intensity (0..1): w_bleach_t = logistic( b_bleach * (SST_t - T_bleach) )
  // (2) COTS temp performance (0..1): f_temp_t = exp( -0.5 * ((SST_t - T_opt)/sigma_T)^2 )
  // (3) Food limitation (0..1): f_food_t = fast^{q_food} / (K50^{q_food} + fast^{q_food})
  // (4) Functional response (Type III): C_perCOTS = (a * A^{q_FR}) / (1 + a * h * A^{q_FR})
  //     where A = p_fast * fast + (1 - p_fast) * slow
  // (5) Coral predation allocation by availability: share_fast = (p_fast*fast) / (p_fast*fast + (1-p_fast)*slow)
  // (6) Coral dynamics (fast):
  //     fast_t = fast_{t-1}
  //              + r_fast*fast_{t-1} * (1 - (fast_{t-1} + alpha_fs*slow_{t-1})/K_fast)
  //              - m_fast*fast_{t-1}
  //              - mu_bleach_fast*w_bleach_t*fast_{t-1}
  //              - conv_fast * C_perCOTS * COTS_{t-1} * share_fast
  // (7) Coral dynamics (slow): analogous with r_slow, alpha_sf, m_slow, mu_bleach_slow, and (1-share_fast)
  // (8) COTS Ricker dynamics with food, temperature, and Allee:
  //     f_allee = [ N_{t-1} / (A50_allee + N_{t-1}) ]^{k_allee}
  //     growth_term = r_cots * [ (1 - gamma_food) + gamma_food * f_food_t ] * f_temp_t * f_allee
  //     cots_tmp = COTS_{t-1} * exp( growth_term - m_cots - beta_cots * COTS_{t-1} )
  // (9) Immigration filtered by coral-dependent settlement:
  //     f_settle = settle_eff * [ w * fast/(K50+fast) + (1-w) * slow/(K50+slow) ]
  //     COTS_t = cots_tmp + f_settle * cotsimm_dat_t
  // (10) Nonnegativity enforced smoothly: x_t = soft_lower_bound(x_t, 1e-8)
  // -----------------------------
  for (int t = 1; t < N; t++) {
    // Availability for feeding (weighted by preference)
    Type fast_prev = fast_pred(t - 1);
    Type slow_prev = slow_pred(t - 1);
    Type cots_prev = cots_pred(t - 1);

    // (1) Bleaching intensity from SST
    w_bleach(t) = Type(1.0) / (Type(1.0) + exp(-b_bleach * (sst_dat(t) - T_bleach)));

    // (2) Temperature performance for COTS (bell-shaped)
    f_temp(t) = exp(-Type(0.5) * pow((sst_dat(t) - T_opt) / sigma_T, 2.0));

    // (3) Food limitation primarily via Acropora
    f_food(t) = pow(fast_prev + eps, q_food) / (pow(half_sat_food, q_food) + pow(fast_prev + eps, q_food));

    // (4) Holling-type III consumption rate per COTS (scaled in % cover per COTS per year)
    Type A = p_fast * fast_prev + p_slow * slow_prev;       // Weighted prey availability in % cover
    Type A_q = pow(A + eps, q_FR);                          // Nonlinear refuge at low A
    coral_avail_q(t) = A_q;
    cons_rate_per_cots(t) = (a_attack * A_q) / (Type(1.0) + a_attack * h_handling * A_q + eps);

    // (5) Allocate predation to coral groups by weighted availability
    Type denom_share = p_fast * fast_prev + p_slow * slow_prev + eps;
    Type share_fast = (p_fast * fast_prev) / denom_share;   // Fraction of consumption from fast
    Type share_slow = Type(1.0) - share_fast;

    // Predation fluxes (converted to % cover loss per year)
    pred_fast_flux(t) = conv_pred_to_mort_fast * cons_rate_per_cots(t) * cots_prev * share_fast;
    pred_slow_flux(t) = conv_pred_to_mort_slow * cons_rate_per_cots(t) * cots_prev * share_slow;

    // (6) Fast coral dynamics (logistic growth + competition - background - bleaching - predation)
    Type fast_growth = r_fast * fast_prev * (Type(1.0) - (fast_prev + alpha_fs * slow_prev) / (K_fast + eps));
    Type fast_bleach = mu_bleach_fast * w_bleach(t) * fast_prev;
    Type fast_next = fast_prev + fast_growth - m_fast * fast_prev - fast_bleach - pred_fast_flux(t);
    fast_pred(t) = soft_lower_bound(fast_next, Type(1e-8)); // Smooth nonnegativity

    // (7) Slow coral dynamics (analogous)
    Type slow_growth = r_slow * slow_prev * (Type(1.0) - (slow_prev + alpha_sf * fast_prev) / (K_slow + eps));
    Type slow_bleach = mu_bleach_slow * w_bleach(t) * slow_prev;
    Type slow_next = slow_prev + slow_growth - m_slow * slow_prev - slow_bleach - pred_slow_flux(t);
    slow_pred(t) = soft_lower_bound(slow_next, Type(1e-8)); // Smooth nonnegativity

    // (8) COTS population growth (Ricker with food, temperature, Allee, and density dependence)
    f_allee(t) = pow(cots_prev / (A50_allee + cots_prev + eps), k_allee);
    Type growth_term = r_cots * ( (Type(1.0) - gamma_food) + gamma_food * f_food(t) ) * f_temp(t) * f_allee(t);
    Type cots_tmp = cots_prev * exp(growth_term - m_cots - beta_cots * cots_prev);

    // (9) Immigration with coral-dependent settlement efficiency (habitat filtering)
    Type f_settle = settle_eff * ( settle_food_weight * (fast_prev / (half_sat_food + fast_prev + eps)) +
                                  (Type(1.0) - settle_food_weight) * (slow_prev / (half_sat_food + slow_prev + eps)) );
    immig_add(t) = f_settle * cotsimm_dat(t);

    // Combine to get next COTS state
    Type cots_next = cots_tmp + immig_add(t);
    cots_pred(t) = soft_lower_bound(cots_next, Type(1e-8)); // Smooth nonnegativity
  }

  // -----------------------------
  // LIKELIHOOD: lognormal on all strictly positive observed variables
  // No data are skipped; minima added for numerical stability.
  // -----------------------------
  Type nll = Type(0.0);
  for (int t = 0; t < N; t++) {
    // COTS (indiv/m^2), lognormal error
    Type res_cots = log(cots_dat(t) + delta) - log(cots_pred(t) + delta);
    nll -= dnorm(res_cots, Type(0.0), sigma_cots, true);

    // Fast coral (% cover), lognormal error
    Type res_fast = log(fast_dat(t) + delta) - log(fast_pred(t) + delta);
    nll -= dnorm(res_fast, Type(0.0), sigma_fast, true);

    // Slow coral (% cover), lognormal error
    Type res_slow = log(slow_dat(t) + delta) - log(slow_pred(t) + delta);
    nll -= dnorm(res_slow, Type(0.0), sigma_slow, true);
  }

  // -----------------------------
  // SOFT PARAMETER BOUNDS (biologically motivated; smooth penalties, no hard constraints)
  // -----------------------------
  Type pen = Type(0.0);
  pen += smooth_bound_penalty(r_fast,  Type(0.0), Type(2.0));
  pen += smooth_bound_penalty(r_slow,  Type(0.0), Type(1.0));
  pen += smooth_bound_penalty(K_fast,  Type(10.0), Type(100.0));
  pen += smooth_bound_penalty(K_slow,  Type(10.0), Type(100.0));
  pen += smooth_bound_penalty(alpha_fs,Type(0.0), Type(1.0));
  pen += smooth_bound_penalty(alpha_sf,Type(0.0), Type(1.0));
  pen += smooth_bound_penalty(m_fast,  Type(0.0), Type(1.0));
  pen += smooth_bound_penalty(m_slow,  Type(0.0), Type(1.0));

  pen += smooth_bound_penalty(T_bleach,       Type(27.0), Type(32.0));
  pen += smooth_bound_penalty(b_bleach,       Type(0.1),  Type(10.0));
  pen += smooth_bound_penalty(mu_bleach_fast, Type(0.0),  Type(1.0));
  pen += smooth_bound_penalty(mu_bleach_slow, Type(0.0),  Type(1.0));

  // q_FR, q_food are transformed; apply penalties on the transformed (natural) scales
  pen += smooth_bound_penalty(q_FR,   Type(1.0),  Type(5.0));
  pen += smooth_bound_penalty(a_attack,      Type(0.0),  Type(1.0));
  pen += smooth_bound_penalty(h_handling,    Type(0.0),  Type(10.0));
  pen += smooth_bound_penalty(conv_pred_to_mort_fast, Type(0.0), Type(5.0));
  pen += smooth_bound_penalty(conv_pred_to_mort_slow, Type(0.0), Type(5.0));

  pen += smooth_bound_penalty(r_cots,  Type(0.0),  Type(5.0));
  pen += smooth_bound_penalty(m_cots,  Type(0.0),  Type(5.0));
  pen += smooth_bound_penalty(beta_cots, Type(0.0), Type(10.0));
  pen += smooth_bound_penalty(T_opt,   Type(24.0), Type(31.0));

  // sigma_T and half_sat_food are transformed, so penalize their natural values
  pen += smooth_bound_penalty(sigma_T,       Type(0.1),  Type(5.0));
  pen += smooth_bound_penalty(half_sat_food, Type(1.0),  Type(60.0));
  pen += smooth_bound_penalty(q_food,        Type(1.0),  Type(5.0));

  // New Allee parameters on natural scales
  pen += smooth_bound_penalty(A50_allee,     Type(0.01),  Type(1.0));
  pen += smooth_bound_penalty(k_allee,       Type(1.0),   Type(5.0));

  // settle_eff and settle_food_weight are in [0,1] by construction; no penalties needed
  // Observation SDs are handled via floors; no additional bounds

  // Weight of penalties relative to likelihood
  Type lambda = Type(10.0); // Moderate weight to guide parameters into biologically plausible ranges
  nll += lambda * pen;

  // -----------------------------
  // REPORTING
  // -----------------------------
  REPORT(Year);                 // Time index
  REPORT(cots_pred);            // Predicted COTS (indiv m^-2)
  REPORT(fast_pred);            // Predicted Acropora cover (%)
  REPORT(slow_pred);            // Predicted slow coral cover (%)
  REPORT(w_bleach);             // Bleaching intensity (0..1)
  REPORT(f_temp);               // COTS temperature performance (0..1)
  REPORT(f_food);               // Food limitation from Acropora (0..1)
  REPORT(pred_fast_flux);       // Predation loss on fast coral (% per year)
  REPORT(pred_slow_flux);       // Predation loss on slow coral (% per year)
  REPORT(cons_rate_per_cots);   // Consumption rate per COTS (scaled)
  REPORT(immig_add);            // Immigration additions to COTS
  REPORT(coral_avail_q);        // Availability index (A^q) driving functional response
  REPORT(f_allee);              // Mate limitation factor for adult COTS (0..1)

  // Also export key transformed parameters for interpretability
  ADREPORT(p_fast);
  ADREPORT(q_FR);
  ADREPORT(sigma_T);
  ADREPORT(half_sat_food);
  ADREPORT(q_food);
  ADREPORT(settle_eff);
  ADREPORT(settle_food_weight);
  ADREPORT(gamma_food);
  ADREPORT(A50_allee);
  ADREPORT(k_allee);

  return nll;
}
