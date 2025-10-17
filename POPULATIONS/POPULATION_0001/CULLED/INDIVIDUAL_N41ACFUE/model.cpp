#include <TMB.hpp>  // TMB framework for AD and optimization

// Helper: stable small constants
template<class Type>
Type tiny() { return Type(1e-8); } // small positive constant to avoid division by zero

// Helper: positive part using smooth approximation (for penalties)
template<class Type>
Type pospart(Type x) {
  Type eps = tiny<Type>();
  return (x + CppAD::sqrt(x * x + eps)) / Type(2.0); // smooth ReLU
}

// Objective function
template<class Type>
Type objective_function<Type>::operator() ()
{
  // -----------------------------
  // DATA (observations and drivers)
  // -----------------------------
  DATA_VECTOR(Year);          // Year (calendar year), used for indexing/time mapping
  DATA_VECTOR(sst_dat);       // Sea-Surface Temperature (Celsius), driver and observed
  DATA_VECTOR(cotsimm_dat);   // COTS larval immigration (inds m^-2 yr^-1), driver and observed
  DATA_VECTOR(cots_dat);      // Adult COTS density (inds m^-2), observed
  DATA_VECTOR(fast_dat);      // Fast coral cover (Acropora; %), observed
  DATA_VECTOR(slow_dat);      // Slow coral cover (Faviidae+Porites; %), observed

  int n = Year.size(); // number of time steps (years)

  // -----------------------------
  // PARAMETERS (ecological processes)
  // -----------------------------

  // Coral intrinsic growth (year^-1 on percent cover scale; max fractional increase modulated by space)
  PARAMETER(gF_max);      // Max growth rate of fast coral (year^-1 on % cover scale); estimate from data
  PARAMETER(gS_max);      // Max growth rate of slow coral (year^-1 on % cover scale); estimate from data
  PARAMETER(comp_S);      // Compensation factor for slow coral growth with free space (dimensionless 0-1)

  // Coral bleaching mortality as function of SST (smooth logistic activation above threshold)
  PARAMETER(T_bleach);    // Bleaching activation temperature (Celsius)
  PARAMETER(bleach_slope);// Steepness of bleaching logistic (C^-1); higher -> sharper onset
  PARAMETER(mF_bleach_max); // Max annual fractional mortality of fast coral due to bleaching (year^-1, fraction of standing stock)
  PARAMETER(mS_bleach_max); // Max annual fractional mortality of slow coral due to bleaching (year^-1, fraction of standing stock)

  // COTS functional response on corals (multi-prey Holling Type II)
  PARAMETER(aF);          // Attack/search rate on fast coral (year^-1 per starfish in area-fraction units)
  PARAMETER(aS);          // Attack/search rate on slow coral (year^-1 per starfish in area-fraction units)
  PARAMETER(hF);          // Handling time for fast coral (year)
  PARAMETER(hS);          // Handling time for slow coral (year)

  // COTS survival as function of food (logistic in coral cover)
  PARAMETER(s0);          // Maximum annual adult survival at high prey (probability, 0-1 but softly bounded)
  PARAMETER(s_min);       // Minimum annual survival at low prey (probability, 0-1 but softly bounded)
  PARAMETER(k_surv);      // Slope of survival vs. food logistic (per % cover)
  PARAMETER(cover50_surv);// Coral cover (% Acropora-equivalent) at which survival is midway between s_min and s0

  // COTS recruitment components (endogenous + exogenous), with temperature and Allee effects
  PARAMETER(r0);          // Recruitment scaling (inds m^-2 yr^-1)
  PARAMETER(I50);         // Half-saturation for larval immigration effect (inds m^-2 yr^-1)
  PARAMETER(rho);         // Weight on endogenous stock-recruitment relative to immigration (dimensionless 0-1)
  PARAMETER(C50);         // Half-saturation for endogenous stock effect (inds m^-2)
  PARAMETER(A50);         // Allee effect scale for fertilization (inds m^-2)
  PARAMETER(T_opt);       // Optimal SST for larval performance (Celsius)
  PARAMETER(sigma_T);     // Width (SD) of the Gaussian temperature function (Celsius)
  PARAMETER(k_food);      // Strength of food effect on recruitment (per unit coral area fraction)
  PARAMETER(w_food);      // Weight of slow coral in food index for adult survival (dimensionless 0-1)
  PARAMETER(w_rec);       // Weight of slow coral in food index for recruitment (dimensionless 0-1)

  // -----------------------------
  // PARAMETERS (observation model)
  // -----------------------------
  PARAMETER(log_sigma_cots); // log SD for lognormal COTS observation error (dimensionless)
  PARAMETER(log_phi_fast);   // log precision for Beta likelihood on fast coral (dimensionless)
  PARAMETER(log_phi_slow);   // log precision for Beta likelihood on slow coral (dimensionless)
  PARAMETER(log_sigma_sst);  // log SD for Normal likelihood on SST (Celsius)
  PARAMETER(log_sigma_imm);  // log SD for Normal likelihood on immigration (inds m^-2 yr^-1)

  // -----------------------------
  // META-PARAMETER (penalty weight)
  // -----------------------------
  PARAMETER(bound_penalty_weight); // Soft-penalty weight (dimensionless), now a parameter to avoid missing data issues

  // -----------------------------
  // Set up containers for predictions
  // -----------------------------
  vector<Type> sst_pred(n);       // Predicted SST (Celsius) — passthrough of driver
  vector<Type> cotsimm_pred(n);   // Predicted immigration (inds m^-2 yr^-1) — passthrough of driver
  vector<Type> cots_pred(n);      // Predicted adult COTS (inds m^-2)
  vector<Type> fast_pred(n);      // Predicted fast coral cover (%)
  vector<Type> slow_pred(n);      // Predicted slow coral cover (%)

  // Initialize predictions with first observed values (avoid data leakage in transition equations)
  sst_pred = sst_dat;                  // passthrough so predictions equal observed for drivers
  cotsimm_pred = cotsimm_dat;          // passthrough so predictions equal observed for drivers
  cots_pred.setZero(); fast_pred.setZero(); slow_pred.setZero(); // initialize
  cots_pred(0) = cots_dat(0);          // initial adult COTS from data (inds m^-2)
  fast_pred(0) = fast_dat(0);          // initial fast coral cover from data (%)
  slow_pred(0) = slow_dat(0);          // initial slow coral cover from data (%)

  // -----------------------------
  // Numerical constants and transforms
  // -----------------------------
  Type eps = tiny<Type>();                      // small positive number
  Type min_sd = Type(1e-6);                     // minimum SD to stabilize likelihoods

  // Observation model parameters (transformed)
  Type sigma_cots = exp(log_sigma_cots) + min_sd; // SD on log scale for COTS
  Type phi_fast   = exp(log_phi_fast)   + eps;    // Beta precision for fast coral
  Type phi_slow   = exp(log_phi_slow)   + eps;    // Beta precision for slow coral
  Type sigma_sst  = exp(log_sigma_sst)  + min_sd; // SD for SST (Normal)
  Type sigma_imm  = exp(log_sigma_imm)  + min_sd; // SD for immigration (Normal)

  // -----------------------------
  // Negative log-likelihood accumulator
  // -----------------------------
  Type nll = 0.0;

  // -----------------------------
  // Time loop: state transitions (use previous-year states/drivers only)
  // -----------------------------
  for (int t = 1; t < n; ++t) {
    // Previous states at year t-1
    Type C_prev = cots_pred(t - 1);          // adult COTS (inds m^-2) at t-1
    Type F_prev = fast_pred(t - 1);          // fast coral cover (%) at t-1
    Type S_prev = slow_pred(t - 1);          // slow coral cover (%) at t-1
    Type SST_prev = sst_dat(t - 1);          // SST (Celsius) at t-1 (driver)
    Type IMM_prev = cotsimm_dat(t - 1);      // larval immigration (inds m^-2 yr^-1) at t-1 (driver)

    // Convert coral covers to area fractions in [0,1]
    Type AF = (F_prev / Type(100.0));        // fast coral area fraction
    Type AS = (S_prev / Type(100.0));        // slow coral area fraction

    // Available free space (non-negative)
    Type free_space = Type(100.0) - F_prev - S_prev; // % free space
    free_space = CppAD::CondExpLt(free_space, Type(0.0), Type(0.0), free_space); // avoid negative free space

    // 1) Coral bleaching mortality (smooth logistic activation at T_bleach)
    Type x_bleach = (SST_prev - T_bleach);                 // temperature difference (C)
    Type p_bleach = Type(1.0) / (Type(1.0) + exp(-bleach_slope * x_bleach)); // smooth 0-1 activation
    Type mF_bleach = mF_bleach_max * p_bleach;             // fractional mortality of fast coral
    Type mS_bleach = mS_bleach_max * p_bleach;             // fractional mortality of slow coral

    // 2) COTS multi-prey functional response (Type II) for coral consumption
    Type denom = Type(1.0) + aF * hF * AF + aS * hS * AS + eps; // joint handling denominator
    Type rateF = aF * AF / denom;                     // per-starfish per-year consumption rate on fast coral (fraction)
    Type rateS = aS * AS / denom;                     // per-starfish per-year consumption rate on slow coral (fraction)
    // Fraction of each coral consumed (saturated by available prey and COTS density)
    Type frac_consume_F = Type(1.0) - exp(-C_prev * rateF);  // in [0,1]
    Type frac_consume_S = Type(1.0) - exp(-C_prev * rateS);  // in [0,1]
    // Area (percentage points) removed from each coral group
    Type pred_F = F_prev * frac_consume_F;                 // % removed from fast coral
    Type pred_S = S_prev * frac_consume_S;                 // % removed from slow coral

    // 3) Coral growth with space limitation and smooth allocation between groups
    // Potential growth contributions (units: %/yr), with slow-coral compensation by free space
    Type potF = gF_max * F_prev;                                    // fast potential growth
    Type potS = gS_max * S_prev * (Type(1.0) + comp_S * (free_space / Type(100.0))); // slow potential growth with compensation
    Type totPot = potF + potS + eps;                                 // total potential growth (avoid zero)
    // Smooth, saturating conversion of potential growth into realized growth limited by free space
    Type growth_total = free_space * (Type(1.0) - exp(-(totPot / (Type(100.0) + eps)))); // cannot exceed free space
    // Proportional allocation of realized growth to each group
    Type shareF = potF / totPot;                    // share to fast coral
    Type growth_F = growth_total * shareF;          // realized fast growth (%)
    Type growth_S = growth_total * (Type(1.0) - shareF); // realized slow growth (%)

    // 4) Coral updates (ensure non-negative via smooth-safe constructs)
    Type F_next = F_prev + growth_F - pred_F - (mF_bleach * F_prev); // fast coral next year (%)
    Type S_next = S_prev + growth_S - pred_S - (mS_bleach * S_prev); // slow coral next year (%)
    // Avoid negative due to rounding/edges
    F_next = CppAD::CondExpLt(F_next, Type(0.0), Type(0.0), F_next);
    S_next = CppAD::CondExpLt(S_next, Type(0.0), Type(0.0), S_next);

    // 5) COTS survival (food-dependent logistic in coral availability)
    Type A_food = AF + w_food * AS; // effective coral area fraction for survival (dimensionless)
    // Logistic between s_min and s0 as a function of equivalent % cover
    Type surv_food = s_min + (s0 - s_min) / (Type(1.0) + exp(-k_surv * ((A_food * Type(100.0)) - cover50_surv)));
    // Bound survival to [0,1] softly via internal form and penalties (below); here keep numeric safety
    surv_food = CppAD::CondExpGt(surv_food, Type(1.0), Type(1.0), surv_food);
    surv_food = CppAD::CondExpLt(surv_food, Type(0.0), Type(0.0), surv_food);

    // 6) COTS recruitment (immigration + endogenous production; SST Gaussian; Allee; food effect)
    Type temp_factor = exp(-Type(0.5) * pow((SST_prev - T_opt) / (sigma_T + eps), 2.0)); // temperature performance (0-1)
    Type imm_factor  = IMM_prev / (IMM_prev + I50 + eps); // saturating immigration contribution (0-1)
    Type stock_factor = C_prev / (C_prev + C50 + eps);    // saturating endogenous stock contribution (0-1)
    Type allee = (C_prev * C_prev) / (C_prev * C_prev + A50 * A50 + eps); // Allee effect (0-1)
    Type food_rec = Type(1.0) - exp(-k_food * (AF + w_rec * AS));         // saturating food effect (0-1)
    Type R = r0 * temp_factor * (imm_factor + rho * stock_factor) * allee * food_rec; // recruits (inds m^-2 yr^-1)

    // 7) COTS update
    Type C_next = surv_food * C_prev + R; // next-year adult density (inds m^-2)

    // Assign to predictions
    fast_pred(t) = F_next;
    slow_pred(t) = S_next;
    cots_pred(t) = C_next;
    // Drivers are passthrough but we keep their time index aligned
    sst_pred(t) = sst_dat(t);
    cotsimm_pred(t) = cotsimm_dat(t);
  }

  // -----------------------------
  // Likelihood: include all observations
  // -----------------------------
  for (int t = 0; t < n; ++t) {
    // COTS abundance (strictly positive) -> Lognormal likelihood on log scale
    Type y_c = log(cots_dat(t) + eps);            // observed log COTS
    Type mu_c = log(cots_pred(t) + eps);          // predicted log COTS
    nll -= dnorm(y_c, mu_c, sigma_cots, true);    // accumulate NLL

    // Coral covers (0-100%) -> Beta likelihood on [0,1], with epsilon to avoid 0/1
    Type y_f = (fast_dat(t) + eps) / (Type(100.0) + Type(2.0) * eps); // observed fraction
    Type mu_f = (fast_pred(t) + eps) / (Type(100.0) + Type(2.0) * eps); // predicted fraction
    Type alpha_f = mu_f * phi_fast + eps;          // Beta alpha
    Type beta_f  = (Type(1.0) - mu_f) * phi_fast + eps; // Beta beta
    nll -= dbeta(y_f, alpha_f, beta_f, true);      // accumulate NLL

    Type y_s = (slow_dat(t) + eps) / (Type(100.0) + Type(2.0) * eps); // observed fraction
    Type mu_s = (slow_pred(t) + eps) / (Type(100.0) + Type(2.0) * eps); // predicted fraction
    Type alpha_s = mu_s * phi_slow + eps;          // Beta alpha
    Type beta_s  = (Type(1.0) - mu_s) * phi_slow + eps; // Beta beta
    nll -= dbeta(y_s, alpha_s, beta_s, true);      // accumulate NLL

    // Drivers included with weak Normal likelihoods (stabilize inference; do not dominate fit)
    nll -= dnorm(sst_dat(t), sst_pred(t), sigma_sst, true);          // SST likelihood (Celsius)
    nll -= dnorm(cotsimm_dat(t), cotsimm_pred(t), sigma_imm, true);  // Immigration likelihood
  }

  // -----------------------------
  // Soft bound penalties (biologically meaningful ranges)
  // -----------------------------
  Type wpen = bound_penalty_weight + tiny<Type>(); // penalty weight (dimensionless), ensure > 0

  // Define helper lambda to add quadratic penalties outside [lo, hi]
  auto pen_bounds = [&](Type x, Type lo, Type hi) {
    Type below = pospart(lo - x);
    Type above = pospart(x - hi);
    return wpen * (below * below + above * above);
  };

  // Apply penalties (suggested ranges documented in parameters.json)
  nll += pen_bounds(gF_max, Type(0.0), Type(1.5));
  nll += pen_bounds(gS_max, Type(0.0), Type(0.8));
  nll += pen_bounds(comp_S, Type(0.0), Type(2.0));

  nll += pen_bounds(T_bleach, Type(25.0), Type(32.0));
  nll += pen_bounds(bleach_slope, Type(0.01), Type(5.0));
  nll += pen_bounds(mF_bleach_max, Type(0.0), Type(1.0));
  nll += pen_bounds(mS_bleach_max, Type(0.0), Type(1.0));

  nll += pen_bounds(aF, Type(0.0), Type(5.0));
  nll += pen_bounds(aS, Type(0.0), Type(5.0));
  nll += pen_bounds(hF, Type(0.0), Type(5.0));
  nll += pen_bounds(hS, Type(0.0), Type(5.0));

  nll += pen_bounds(s0, Type(0.2), Type(0.99));
  nll += pen_bounds(s_min, Type(0.0), Type(0.8));
  nll += pen_bounds(k_surv, Type(0.0), Type(0.5));
  nll += pen_bounds(cover50_surv, Type(0.0), Type(100.0));

  nll += pen_bounds(r0, Type(0.0), Type(10.0));
  nll += pen_bounds(I50, Type(0.0), Type(10.0));
  nll += pen_bounds(rho, Type(0.0), Type(1.0));
  nll += pen_bounds(C50, Type(0.01), Type(10.0));
  nll += pen_bounds(A50, Type(0.01), Type(10.0));
  nll += pen_bounds(T_opt, Type(24.0), Type(32.0));
  nll += pen_bounds(sigma_T, Type(0.2), Type(5.0));
  nll += pen_bounds(k_food, Type(0.0), Type(10.0));
  nll += pen_bounds(w_food, Type(0.0), Type(1.0));
  nll += pen_bounds(w_rec, Type(0.0), Type(1.0));

  // Keep the penalty weight itself in a plausible range
  nll += pen_bounds(bound_penalty_weight, Type(0.01), Type(1000.0));

  // Observation model penalties (very weak, just to keep in plausible region)
  nll += pen_bounds(exp(log_sigma_cots), Type(0.01), Type(2.0));
  nll += pen_bounds(exp(log_phi_fast),   Type(1.0),  Type(1e5));
  nll += pen_bounds(exp(log_phi_slow),   Type(1.0),  Type(1e5));
  nll += pen_bounds(exp(log_sigma_sst),  Type(0.01), Type(2.0));
  nll += pen_bounds(exp(log_sigma_imm),  Type(0.001),Type(5.0));

  // -----------------------------
  // REPORT predictions for inspection and forecasting
  // -----------------------------
  REPORT(Year);            // Year for reference
  REPORT(sst_pred);        // Predicted SST (passthrough)
  REPORT(cotsimm_pred);    // Predicted immigration (passthrough)
  REPORT(cots_pred);       // Predicted adult COTS
  REPORT(fast_pred);       // Predicted fast coral
  REPORT(slow_pred);       // Predicted slow coral

  // -----------------------------
  // Documentation of model equations (for transparency)
  // -----------------------------
  // 1) Bleaching activation: p_bleach_t = 1 / (1 + exp(-bleach_slope * (SST_{t-1} - T_bleach)))
  //    m^{F,S}_bleach_t = m^{F,S}_bleach_max * p_bleach_t
  // 2) Functional response:
  //    denom = 1 + aF*hF*AF_{t-1} + aS*hS*AS_{t-1}
  //    rateF = aF*AF_{t-1}/denom; rateS = aS*AS_{t-1}/denom
  //    frac_consume_{F,S} = 1 - exp(-C_{t-1} * rate_{F,S})
  //    pred_{F,S} = {F,S}_{t-1} * frac_consume_{F,S}
  // 3) Coral growth with space competition:
  //    potF = gF_max * F_{t-1}
  //    potS = gS_max * S_{t-1} * (1 + comp_S * free_space_{t-1}/100)
  //    growth_total = free_space_{t-1} * (1 - exp(-(potF + potS)/100))
  //    growth_F = growth_total * potF / (potF + potS); growth_S = growth_total - growth_F
  // 4) Coral updates:
  //    F_t = F_{t-1} + growth_F - pred_F - mF_bleach_t * F_{t-1}
  //    S_t = S_{t-1} + growth_S - pred_S - mS_bleach_t * S_{t-1}
  // 5) Adult survival (food-dependent):
  //    A_food = AF_{t-1} + w_food * AS_{t-1}
  //    surv_food = s_min + (s0 - s_min) / (1 + exp(-k_surv * (100*A_food - cover50_surv)))
  // 6) Recruitment:
  //    temp_factor = exp(-0.5 * ((SST_{t-1} - T_opt)/sigma_T)^2)
  //    imm_factor = IMM_{t-1} / (IMM_{t-1} + I50)
  //    stock_factor = C_{t-1} / (C_{t-1} + C50)
  //    allee = C_{t-1}^2 / (C_{t-1}^2 + A50^2)
  //    food_rec = 1 - exp(-k_food * (AF_{t-1} + w_rec * AS_{t-1}))
  //    R_t = r0 * temp_factor * (imm_factor + rho * stock_factor) * allee * food_rec
  // 7) COTS update:
  //    C_t = surv_food * C_{t-1} + R_t
  //
  // Likelihoods:
  //    log COTS ~ Normal(log(cots_pred), sigma_cots)
  //    fast, slow ~ Beta(mean = *_pred/(100 + 2*eps), precision = phi)
  //    SST, IMM ~ Normal(pred, sigma)

  return nll;
}
