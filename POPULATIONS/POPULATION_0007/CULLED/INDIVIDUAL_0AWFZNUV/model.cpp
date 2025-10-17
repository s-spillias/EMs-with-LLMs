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
  PARAMETER(p_fast_raw);      // Logit preference for Acropora in COTS diet
  PARAMETER(a_attack);        // Attack rate scaling
  PARAMETER(h_handling);      // Handling time scaling
  PARAMETER(q_FR_raw);        // Log-scale for Type III exponent

  // Conversion efficiencies from consumption to coral % cover loss
  PARAMETER(conv_pred_to_mort_fast);
  PARAMETER(conv_pred_to_mort_slow);

  // COTS population dynamics (Ricker with modifiers)
  PARAMETER(r_cots);          // Max per-capita growth rate
  PARAMETER(m_cots);          // Natural mortality
  PARAMETER(beta_cots);       // Density dependence strength
  PARAMETER(gamma_food_raw);  // Weight for food limitation (logit scale)

  // Temperature performance for COTS
  PARAMETER(T_opt);           // Optimal SST for COTS
  PARAMETER(sigma_T_raw);     // Log-scale width of temperature performance (σ > 0)

  // Food limitation half-saturation and non-linearity
  PARAMETER(half_sat_food_raw); // log-scale K50 in % cover
  PARAMETER(q_food_raw);        // log-scale exponent (q_food >= 1)

  // Reproductive Allee effect parameters (on adult density)
  PARAMETER(half_sat_allee_raw); // log-scale A50 in indiv m^-2
  PARAMETER(q_allee_raw);        // log-scale steepness (q_allee >= 1)

  // Settlement/immigration filtering
  PARAMETER(settle_eff_raw);       // Logit-scale settlement efficiency
  PARAMETER(settle_food_weight_raw); // Logit-scale weight of fast coral in settlement habitat index

  // Observation error (lognormal)
  PARAMETER(log_sigma_cots);
  PARAMETER(log_sigma_fast);
  PARAMETER(log_sigma_slow);

  // -----------------------------
  // Transforms and derived parameters
  // -----------------------------
  Type eps = Type(1e-8);

  // Diet preference
  Type p_fast = invlogit(p_fast_raw);
  Type p_slow = Type(1.0) - p_fast;

  // Functional response exponent
  Type q_FR = Type(1.0) + exp(q_FR_raw);

  // COTS temperature performance width
  Type sigma_T = exp(sigma_T_raw);

  // Food limitation parameters (natural scale)
  Type half_sat_food = exp(half_sat_food_raw);
  Type q_food = Type(1.0) + exp(q_food_raw);

  // Allee effect parameters (natural scale)
  Type half_sat_allee = exp(half_sat_allee_raw);
  Type q_allee = Type(1.0) + exp(q_allee_raw);

  // Food weighting in growth
  Type gamma_food = invlogit(gamma_food_raw);

  // Settlement transforms
  Type settle_eff = invlogit(settle_eff_raw);
  Type w_settle = invlogit(settle_food_weight_raw);

  // Observation SDs
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_fast = exp(log_sigma_fast);
  Type sigma_slow = exp(log_sigma_slow);

  // -----------------------------
  // State vectors
  // -----------------------------
  vector<Type> cots_pred(N);
  vector<Type> fast_pred(N);
  vector<Type> slow_pred(N);

  // Initialize states at t = 0 using observed values (used only as previous-step seeds)
  cots_pred(0) = CppAD::CondExpGt(cots_dat(0), Type(1e-8), cots_dat(0), Type(1e-8));
  fast_pred(0) = CppAD::CondExpGt(fast_dat(0), Type(1e-8), fast_dat(0), Type(1e-8));
  slow_pred(0) = CppAD::CondExpGt(slow_dat(0), Type(1e-8), slow_dat(0), Type(1e-8));

  // -----------------------------
  // Negative log-likelihood
  // -----------------------------
  Type nll = Type(0.0);

  // Observation likelihood at t=0
  nll -= dnorm(log(cots_dat(0)), log(cots_pred(0)), sigma_cots, true);
  // Handle potential zeros for coral cover
  nll -= dnorm(log(CppAD::CondExpGt(fast_dat(0), Type(1e-8), fast_dat(0), Type(1e-8))),
               log(CppAD::CondExpGt(fast_pred(0), Type(1e-8), fast_pred(0), Type(1e-8))), sigma_fast, true);
  nll -= dnorm(log(CppAD::CondExpGt(slow_dat(0), Type(1e-8), slow_dat(0), Type(1e-8))),
               log(CppAD::CondExpGt(slow_pred(0), Type(1e-8), slow_pred(0), Type(1e-8))), sigma_slow, true);

  // -----------------------------
  // Time loop for dynamics and observation model
  // -----------------------------
  for (int t = 1; t < N; t++) {
    // Previous-step states (never use current-step observed data in prediction)
    Type COTS_prev = cots_pred(t - 1);
    Type FAST_prev = fast_pred(t - 1);
    Type SLOW_prev = slow_pred(t - 1);

    // ---------------------------
    // Coral predation: Holling type III multi-resource
    // ---------------------------
    Type fast_q = pow(FAST_prev, q_FR);
    Type slow_q = pow(SLOW_prev, q_FR);
    Type S_res = p_fast * fast_q + p_slow * slow_q; // resource aggregation
    Type cons_total_per_pred = (a_attack * S_res) / (Type(1.0) + h_handling * a_attack * S_res);

    // Allocation to each coral group
    Type denom_alloc = S_res + Type(1e-8);
    Type prop_fast = (p_fast * fast_q) / denom_alloc;
    Type prop_slow = (p_slow * slow_q) / denom_alloc;

    Type cons_fast_per_pred = cons_total_per_pred * prop_fast; // per predator per year
    Type cons_slow_per_pred = cons_total_per_pred * prop_slow; // per predator per year

    // Convert to % cover loss per year (cap at available cover)
    Type pred_loss_fast = conv_pred_to_mort_fast * COTS_prev * cons_fast_per_pred;
    Type pred_loss_slow = conv_pred_to_mort_slow * COTS_prev * cons_slow_per_pred;

    pred_loss_fast = CppAD::CondExpGt(pred_loss_fast, FAST_prev, FAST_prev, pred_loss_fast);
    pred_loss_slow = CppAD::CondExpGt(pred_loss_slow, SLOW_prev, SLOW_prev, pred_loss_slow);

    // ---------------------------
    // Bleaching mortality (logistic in SST)
    // ---------------------------
    Type f_bleach = Type(1.0) / (Type(1.0) + exp(-b_bleach * (sst_dat(t) - T_bleach)));
    Type bleach_mort_fast = mu_bleach_fast * f_bleach;
    Type bleach_mort_slow = mu_bleach_slow * f_bleach;

    // ---------------------------
    // Coral dynamics (logistic growth + background + bleaching + predation)
    // ---------------------------
    // Fast (Acropora)
    Type growth_fast = r_fast * FAST_prev * (Type(1.0) - (FAST_prev + alpha_fs * SLOW_prev) / K_fast);
    Type mort_fast_bg = m_fast * FAST_prev;
    Type FAST_next = FAST_prev + growth_fast - mort_fast_bg - bleach_mort_fast * FAST_prev - pred_loss_fast;
    FAST_next = soft_lower_bound(FAST_next, Type(0.0));
    FAST_next = CppAD::CondExpGt(FAST_next, Type(100.0), Type(100.0), FAST_next); // cap at 100% cover

    // Slow (Faviidae + Porites)
    Type growth_slow = r_slow * SLOW_prev * (Type(1.0) - (SLOW_prev + alpha_sf * FAST_prev) / K_slow);
    Type mort_slow_bg = m_slow * SLOW_prev;
    Type SLOW_next = SLOW_prev + growth_slow - mort_slow_bg - bleach_mort_slow * SLOW_prev - pred_loss_slow;
    SLOW_next = soft_lower_bound(SLOW_next, Type(0.0));
    SLOW_next = CppAD::CondExpGt(SLOW_next, Type(100.0), Type(100.0), SLOW_next); // cap at 100% cover

    // ---------------------------
    // COTS growth modifiers
    // ---------------------------
    // Food limitation index (weighted by diet preference)
    Type Food_idx = p_fast * FAST_prev + p_slow * SLOW_prev; // in % cover
    // Saturating response for food limitation
    Type Food_q = pow(Food_idx, q_food);
    Type f_food = Food_q / (pow(half_sat_food, q_food) + Food_q);

    // Temperature performance (Gaussian)
    Type f_temp = exp(-Type(0.5) * pow((sst_dat(t) - T_opt) / sigma_T, 2));

    // Reproductive Allee effect on adults
    Type COTS_q = pow(COTS_prev, q_allee);
    Type f_allee = COTS_q / (pow(half_sat_allee, q_allee) + COTS_q);

    // Immigration filter by settlement habitat (uses previous-step corals; exogenous driver at t)
    Type habitat_settle = (w_settle * FAST_prev + (Type(1.0) - w_settle) * SLOW_prev) / Type(100.0);
    habitat_settle = CppAD::CondExpLt(habitat_settle, Type(0.0), Type(0.0), habitat_settle);
    habitat_settle = CppAD::CondExpGt(habitat_settle, Type(1.0), Type(1.0), habitat_settle);
    Type immigration_t = settle_eff * habitat_settle * cotsimm_dat(t);

    // ---------------------------
    // COTS dynamics (Ricker with modifiers)
    // ---------------------------
    Type growth_term = r_cots * ((Type(1.0) - gamma_food) + gamma_food * f_food) * f_temp * f_allee;
    Type COTS_next = COTS_prev * exp(growth_term - m_cots - beta_cots * COTS_prev) + immigration_t;
    COTS_next = soft_lower_bound(COTS_next, Type(1e-8));

    // Store predictions
    fast_pred(t) = FAST_next;
    slow_pred(t) = SLOW_next;
    cots_pred(t) = COTS_next;

    // Observation likelihood at time t (lognormal)
    nll -= dnorm(log(cots_dat(t)), log(COTS_next), sigma_cots, true);
    nll -= dnorm(log(CppAD::CondExpGt(fast_dat(t), Type(1e-8), fast_dat(t), Type(1e-8))),
                 log(CppAD::CondExpGt(FAST_next, Type(1e-8), FAST_next, Type(1e-8))), sigma_fast, true);
    nll -= dnorm(log(CppAD::CondExpGt(slow_dat(t), Type(1e-8), slow_dat(t), Type(1e-8))),
                 log(CppAD::CondExpGt(SLOW_next, Type(1e-8), SLOW_next, Type(1e-8))), sigma_slow, true);
  }

  // -----------------------------
  // Soft penalties for biological plausibility
  // -----------------------------
  Type penalty = Type(0.0);
  // Keep carrying capacities within [0, 100]
  penalty += smooth_bound_penalty(K_fast, Type(0.0), Type(100.0));
  penalty += smooth_bound_penalty(K_slow, Type(0.0), Type(100.0));
  // Allee parameters (natural scale guidance)
  penalty += smooth_bound_penalty(half_sat_allee, Type(0.001), Type(1.0));
  penalty += smooth_bound_penalty(q_allee, Type(1.0), Type(5.0));

  nll += penalty;

  // -----------------------------
  // REPORT/ADREPORT
  // -----------------------------
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  REPORT(p_fast);
  REPORT(q_FR);
  REPORT(sigma_T);
  REPORT(half_sat_food);
  REPORT(q_food);
  REPORT(half_sat_allee);
  REPORT(q_allee);
  REPORT(gamma_food);
  REPORT(settle_eff);
  REPORT(w_settle);

  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);

  return nll;
}
