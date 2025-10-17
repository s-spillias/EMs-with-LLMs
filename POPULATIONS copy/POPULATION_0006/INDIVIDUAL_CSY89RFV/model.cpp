#include <TMB.hpp>

// Smooth softplus with slope control; used for smooth penalties and clipping
template<class Type>
Type softplus_k(Type z, Type k) {
  // Numerically safe softplus with tunable sharpness k for AD types
  return log(Type(1.0) + exp(k * z)) / k;
}

// Smooth penalty for violating [lb, ub]; zero when inside, quadratic growth outside
template<class Type>
Type smooth_bounds_penalty(Type x, Type lb, Type ub, Type scale, Type ksharp) {
  Type pen_low  = softplus_k(lb - x, ksharp); // >0 if x < lb
  Type pen_high = softplus_k(x - ub, ksharp); // >0 if x > ub
  return (pen_low * pen_low + pen_high * pen_high) / (scale * scale + Type(1e-12));
}

// ADMB-style positive function to enforce x >= eps smoothly and accumulate penalty
template<class Type>
Type posfun(Type x, Type eps, Type &pen) {
  // If x >= eps, return x; otherwise return a smooth function approaching eps and add quadratic penalty
  if (x >= eps) return x;
  Type a = eps / (Type(2.0) - x / eps); // smooth replacement value >= eps
  Type d = x - eps;
  pen += d * d; // quadratic penalty for correction
  return a;
}

template<class Type>
Type objective_function<Type>::operator() () {
  // ---------------------------
  // DATA
  // ---------------------------
  DATA_VECTOR(Year);         // calendar year (integer-like), used for indexing/plotting; not used in dynamics directly
  DATA_VECTOR(cots_dat);     // Adult COTS abundance (individuals m^-2), strictly positive
  DATA_VECTOR(fast_dat);     // Fast coral cover (Acropora spp.) as percent cover (0-100)
  DATA_VECTOR(slow_dat);     // Slow coral cover (Faviidae/Porites spp.) as percent cover (0-100)
  DATA_VECTOR(sst_dat);      // Sea-surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat);  // COTS larval immigration rate (individuals m^-2 year^-1)

  int n = Year.size(); // number of time steps
  // ---------------------------
  // PARAMETERS (with units and roles)
  // ---------------------------
  PARAMETER(rC);            // year^-1 | Intrinsic COTS per-capita growth rate (baseline reproduction)
  PARAMETER(mC);            // year^-1 | Background adult COTS mortality rate
  PARAMETER(eC);            // (individuals m^-2) per (% cover) | Conversion efficiency of consumed coral into recruitment buffer
  PARAMETER(tau_mature);    // dimensionless fraction per year | Fraction of last year's buffer maturing to adults
  PARAMETER(K0);            // individuals m^-2 | Baseline COTS carrying capacity (food-independent)
  PARAMETER(K_food);        // (individuals m^-2) per (% cover) | Increment to COTS carrying capacity per % coral food
  PARAMETER(aF);            // (year^-1) per (individual m^-2) per (% cover) | Attack rate on fast corals in multi-prey Type II response
  PARAMETER(aS);            // (year^-1) per (individual m^-2) per (% cover) | Attack rate on slow corals in multi-prey Type II response
  PARAMETER(hF);            // year per (% cover) | Handling time on fast corals
  PARAMETER(hS);            // year per (% cover) | Handling time on slow corals
  PARAMETER(rF);            // year^-1 | Fast coral intrinsic growth rate
  PARAMETER(rS);            // year^-1 | Slow coral intrinsic growth rate
  PARAMETER(Kb);            // % cover | Total benthic coral carrying capacity for each group’s logistic term (effective maximum live coral space)
  PARAMETER(alpha_FS);      // dimensionless | Competition coefficient: effect of slow corals on fast coral carrying capacity
  PARAMETER(alpha_SF);      // dimensionless | Competition coefficient: effect of fast corals on slow coral carrying capacity
  PARAMETER(mB_fast);       // year^-1 | Maximum additional bleaching mortality rate for fast corals as SST exceeds threshold
  PARAMETER(mB_slow);       // year^-1 | Maximum additional bleaching mortality rate for slow corals as SST exceeds threshold
  PARAMETER(T_bleach);      // deg C | SST threshold for bleaching ramp
  PARAMETER(sd_bleach);     // deg C | Width of bleaching ramp (larger = smoother)
  PARAMETER(Topt_COTS);     // deg C | Thermal optimum for COTS reproduction/survival
  PARAMETER(Tsd_COTS);      // deg C | Thermal niche width (std dev) for COTS response
  PARAMETER(beta_food);     // per (% cover) | Food saturation coefficient for COTS reproduction
  PARAMETER(A_thr);         // individuals m^-2 | Allee-like smooth threshold for COTS reproduction
  PARAMETER(k_Allee);       // (individuals m^-2)^-1 | Steepness of Allee logistic
  PARAMETER(s_imm);         // (individuals m^-2) per (individuals m^-2 year^-1) | Adult-equivalent addition per unit larval immigration
  PARAMETER(wF);            // dimensionless | Weight of fast coral in food index
  PARAMETER(wS);            // dimensionless | Weight of slow coral in food index
  PARAMETER(log_sigma_cots);// log(year^-1 units of abundance on log scale) | Log of COTS observation std dev for lognormal errors
  PARAMETER(log_phi_fast);  // log(precision) | Log precision for Beta likelihood of fast coral proportion
  PARAMETER(log_phi_slow);  // log(precision) | Log precision for Beta likelihood of slow coral proportion

  // ---------------------------
  // CONSTANTS AND HELPERS
  // ---------------------------
  Type eps      = Type(1e-8);   // small constant to avoid division by zero
  Type eps_prop = Type(1e-6);   // small constant to keep proportions strictly within (0,1)
  Type pen      = Type(0.0);    // aggregate penalty for bounds and positivity
  Type ksharp   = Type(10.0);   // sharpness of softplus-based penalties
  Type pen_scale= Type(1.0);    // scale of penalties

  // Effective observation error parameters with smooth lower bounds
  Type min_sigma = Type(0.05); // minimum lognormal std dev to avoid overfitting on small values
  Type sigma_cots = exp(log_sigma_cots);                           // raw sigma
  Type sigma_eff  = sqrt(sigma_cots * sigma_cots + min_sigma * min_sigma); // smooth floor

  Type phi_min = Type(5.0);  // minimum Beta precision to ensure reasonable dispersion modeling
  Type phi_fast_eff = exp(log_phi_fast) + phi_min; // smooth lower bound
  Type phi_slow_eff = exp(log_phi_slow) + phi_min; // smooth lower bound

  // ---------------------------
  // STATE VECTORS AND INITIAL CONDITIONS (from data to avoid data leakage)
  // ---------------------------
  vector<Type> cots_pred(n); cots_pred.setZero(); // individuals m^-2
  vector<Type> fast_pred(n); fast_pred.setZero(); // % cover
  vector<Type> slow_pred(n); slow_pred.setZero(); // % cover
  vector<Type> cons_buf(n); cons_buf.setZero();   // individuals m^-2 year^-1 equivalent recruitment buffer

  // Initialize with observed first data point as required
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Initialize recruitment buffer at t = 0 based on t=0 states (used at t=1)
  {
    Type C0 = cots_pred(0);
    Type F0 = fast_pred(0);
    Type S0 = slow_pred(0);
    Type denom0 = Type(1.0) + aF * hF * F0 + aS * hS * S0 + eps;
    Type gF0 = aF * C0 * F0 / denom0;
    Type gS0 = aS * C0 * S0 / denom0;
    cons_buf(0) = eC * (gF0 + gS0);
  }

  // ---------------------------
  // PROCESS MODEL
  // Numbered equations (all use t-1 states and t-1 forcing, never current observations)
  // 1) Food index: Food_{t-1} = wF * Fast_{t-1} + wS * Slow_{t-1}
  // 2) Food saturation: f_food = 1 - exp(-beta_food * Food_{t-1})
  // 3) COTS temperature modifier: f_temp = exp(-0.5 * ((SST_{t-1} - Topt_COTS)/Tsd_COTS)^2)
  // 4) Allee logistic: s_Allee = invlogit(k_Allee * (COTS_{t-1} - A_thr))
  // 5) Multi-prey functional response denominator:
  //    D = 1 + aF*hF*Fast_{t-1} + aS*hS*Slow_{t-1}
  // 6) Per-year coral consumption:
  //    gF = aF * COTS_{t-1} * Fast_{t-1} / D
  //    gS = aS * COTS_{t-1} * Slow_{t-1} / D
  // 7) Coral bleaching ramp (0..1): b = logistic((SST_{t-1} - T_bleach)/sd_bleach)
  //    m_bleach_fast = mB_fast * b;  m_bleach_slow = mB_slow * b
  // 8) Coral updates (logistic with competition and losses):
  //    Fast_t = Fast_{t-1} + rF*Fast_{t-1}*(1 - (Fast_{t-1} + alpha_FS*Slow_{t-1})/Kb) - gF - m_bleach_fast*Fast_{t-1}
  //    Slow_t = Slow_{t-1} + rS*Slow_{t-1}*(1 - (Slow_{t-1} + alpha_SF*Fast_{t-1})/Kb) - gS - m_bleach_slow*Slow_{t-1}
  // 9) COTS carrying capacity: K_eff = K0 + K_food * Food_{t-1}
  // 10) Immigration to adults: Imm_adults = s_imm * cotsimm_{t-1} * f_temp * f_food
  // 11) Delayed recruitment buffer:
  //     cons_buf_t = eC * (gF + gS)   // stored for next year
  // 12) COTS update with delayed maturation:
  //     COTS_t = COTS_{t-1} +
  //              rC*f_temp*s_Allee*f_food*COTS_{t-1}*(1 - COTS_{t-1}/K_eff) +
  //              tau_mature * cons_buf_{t-1} - mC*COTS_{t-1} + Imm_adults
  // ---------------------------

  for (int t = 1; t < n; t++) {
    // Previous states (do not use current observations)
    Type C_prev   = cots_pred(t-1);
    Type F_prev   = fast_pred(t-1);
    Type S_prev   = slow_pred(t-1);
    Type SST_prev = sst_dat(t-1);
    Type IMM_prev = cotsimm_dat(t-1);
    Type cons_prev = cons_buf(t-1); // last year's recruitment buffer

    // (1) Food index and (2) saturating food effect
    Type food_index = wF * F_prev + wS * S_prev;                                    // % cover (weighted)
    Type f_food     = Type(1.0) - exp(-beta_food * (food_index));                    // 0..1 smooth saturation

    // (3) Temperature effect on COTS vital rates (Gaussian around optimum)
    Type Tsd_eff = Tsd_COTS + eps;                                                   // ensure >0
    Type f_temp  = exp( Type(-0.5) * pow( (SST_prev - Topt_COTS) / Tsd_eff, 2 ) );   // 0..1

    // (4) Allee-like smooth low-density limitation
    Type s_Allee = Type(1.0) / (Type(1.0) + exp(-k_Allee * (C_prev - A_thr)));      // 0..1

    // (5) Denominator for multi-prey Holling Type II functional response
    Type denom = Type(1.0) + aF * hF * F_prev + aS * hS * S_prev + eps;

    // (6) Coral consumption rates (percent cover per year)
    Type gF = aF * C_prev * F_prev / denom;                                          // % cover year^-1
    Type gS = aS * C_prev * S_prev / denom;                                          // % cover year^-1

    // (7) Temperature-driven bleaching ramp (smooth logistic above threshold)
    Type b_ramp = Type(1.0) / (Type(1.0) + exp( -(SST_prev - T_bleach) / (sd_bleach + eps) )); // 0..1
    Type m_bleach_fast = mB_fast * b_ramp;                                           // year^-1
    Type m_bleach_slow = mB_slow * b_ramp;                                           // year^-1

    // (8) Coral updates with logistic growth, interspecific competition, predation loss, and bleaching loss
    Type F_growth = rF * F_prev * ( Type(1.0) - (F_prev + alpha_FS * S_prev) / (Kb + eps) );  // % cover year^-1
    Type S_growth = rS * S_prev * ( Type(1.0) - (S_prev + alpha_SF * F_prev) / (Kb + eps) );  // % cover year^-1
    Type F_next_raw = F_prev + F_growth - gF - m_bleach_fast * F_prev;                         // % cover
    Type S_next_raw = S_prev + S_growth - gS - m_bleach_slow * S_prev;                         // % cover

    // Enforce non-negativity smoothly with posfun (adds to penalty if corrected)
    F_next_raw = posfun(F_next_raw, eps, pen); // keep >= eps
    S_next_raw = posfun(S_next_raw, eps, pen); // keep >= eps

    // (9) Food-modified COTS carrying capacity
    Type K_eff = K0 + K_food * food_index;                                          // individuals m^-2

    // (10) Immigration modified by environment
    Type Imm_adults = s_imm * IMM_prev * f_temp * f_food;                            // individuals m^-2 year^-1

    // (11) Delayed recruitment: buffer update for next year
    Type cons_next = eC * (gF + gS);                                                // individuals m^-2 year^-1 equivalent
    cons_buf(t) = cons_next;

    // (12) COTS population update: logistic + delayed maturation of buffer - mortality + immigration
    Type deltaC = rC * f_temp * s_Allee * f_food * C_prev * ( Type(1.0) - C_prev / (K_eff + eps) )
                  + tau_mature * cons_prev
                  - mC * C_prev
                  + Imm_adults;

    Type C_next_raw = C_prev + deltaC;                                               // individuals m^-2
    C_next_raw = posfun(C_next_raw, eps, pen);                                       // keep >= eps

    // Assign predictions
    fast_pred(t) = F_next_raw;
    slow_pred(t) = S_next_raw;
    cots_pred(t) = C_next_raw;
  }

  // ---------------------------
  // LIKELIHOOD
  // ---------------------------
  Type nll = Type(0.0);

  // COTS: lognormal errors on strictly positive data
  for (int t = 0; t < n; t++) {
    Type y_log = log(cots_dat(t) + eps);     // observed on log scale
    Type mu_log = log(cots_pred(t) + eps);   // predicted on log scale
    nll -= dnorm(y_log, mu_log, sigma_eff, true);
  }

  // Coral covers: Beta likelihoods on proportions in (0,1) with smooth epsilon mapping
  for (int t = 0; t < n; t++) {
    // Fast coral
    Type yF = fast_dat(t) / Type(100.0);
    Type muF = fast_pred(t) / Type(100.0);
    // Map to open interval (0,1) smoothly
    Type yF_c  = yF  * (Type(1.0) - 2.0 * eps_prop) + eps_prop;
    Type muF_c = muF * (Type(1.0) - 2.0 * eps_prop) + eps_prop;
    Type aFbeta = muF_c * phi_fast_eff + eps;
    Type bFbeta = (Type(1.0) - muF_c) * phi_fast_eff + eps;
    nll -= dbeta(yF_c, aFbeta, bFbeta, true);

    // Slow coral
    Type yS = slow_dat(t) / Type(100.0);
    Type muS = slow_pred(t) / Type(100.0);
    Type yS_c  = yS  * (Type(1.0) - 2.0 * eps_prop) + eps_prop;
    Type muS_c = muS * (Type(1.0) - 2.0 * eps_prop) + eps_prop;
    Type aSbeta = muS_c * phi_slow_eff + eps;
    Type bSbeta = (Type(1.0) - muS_c) * phi_slow_eff + eps;
    nll -= dbeta(yS_c, aSbeta, bSbeta, true);
  }

  // ---------------------------
  // PARAMETER SOFT BOUNDS (smooth penalties; no hard constraints)
  // Suggested biological ranges applied as soft penalties:
  // rC [0,5], mC [0,3], eC [0,10], tau_mature [0,1], K0 [0,5], K_food [0,1],
  // aF [0,0.2], aS [0,0.2], hF [0,5], hS [0,5],
  // rF [0,2], rS [0,1], Kb [40,95],
  // alpha_FS [0,2], alpha_SF [0,2],
  // mB_fast [0,2], mB_slow [0,2], T_bleach [27,33], sd_bleach [0.2,3],
  // Topt_COTS [25,32], Tsd_COTS [0.5,5], beta_food [0.001,0.5],
  // A_thr [0,2], k_Allee [0,50], s_imm [0,5], wF [0,5], wS [0,5]
  pen += smooth_bounds_penalty(rC,         Type(0.0),  Type(5.0),  pen_scale, ksharp);
  pen += smooth_bounds_penalty(mC,         Type(0.0),  Type(3.0),  pen_scale, ksharp);
  pen += smooth_bounds_penalty(eC,         Type(0.0),  Type(10.0), pen_scale, ksharp);
  pen += smooth_bounds_penalty(tau_mature, Type(0.0),  Type(1.0),  pen_scale, ksharp);
  pen += smooth_bounds_penalty(K0,         Type(0.0),  Type(5.0),  pen_scale, ksharp);
  pen += smooth_bounds_penalty(K_food,     Type(0.0),  Type(1.0),  pen_scale, ksharp);
  pen += smooth_bounds_penalty(aF,         Type(0.0),  Type(0.2),  pen_scale, ksharp);
  pen += smooth_bounds_penalty(aS,         Type(0.0),  Type(0.2),  pen_scale, ksharp);
  pen += smooth_bounds_penalty(hF,         Type(0.0),  Type(5.0),  pen_scale, ksharp);
  pen += smooth_bounds_penalty(hS,         Type(0.0),  Type(5.0),  pen_scale, ksharp);
  pen += smooth_bounds_penalty(rF,         Type(0.0),  Type(2.0),  pen_scale, ksharp);
  pen += smooth_bounds_penalty(rS,         Type(0.0),  Type(1.0),  pen_scale, ksharp);
  pen += smooth_bounds_penalty(Kb,         Type(40.0), Type(95.0), pen_scale, ksharp);
  pen += smooth_bounds_penalty(alpha_FS,   Type(0.0),  Type(2.0),  pen_scale, ksharp);
  pen += smooth_bounds_penalty(alpha_SF,   Type(0.0),  Type(2.0),  pen_scale, ksharp);
  pen += smooth_bounds_penalty(mB_fast,    Type(0.0),  Type(2.0),  pen_scale, ksharp);
  pen += smooth_bounds_penalty(mB_slow,    Type(0.0),  Type(2.0),  pen_scale, ksharp);
  pen += smooth_bounds_penalty(T_bleach,   Type(27.0), Type(33.0), pen_scale, ksharp);
  pen += smooth_bounds_penalty(sd_bleach,  Type(0.2),  Type(3.0),  pen_scale, ksharp);
  pen += smooth_bounds_penalty(Topt_COTS,  Type(25.0), Type(32.0), pen_scale, ksharp);
  pen += smooth_bounds_penalty(Tsd_COTS,   Type(0.5),  Type(5.0),  pen_scale, ksharp);
  pen += smooth_bounds_penalty(beta_food,  Type(0.001),Type(0.5),  pen_scale, ksharp);
  pen += smooth_bounds_penalty(A_thr,      Type(0.0),  Type(2.0),  pen_scale, ksharp);
  pen += smooth_bounds_penalty(k_Allee,    Type(0.0),  Type(50.0), pen_scale, ksharp);
  pen += smooth_bounds_penalty(s_imm,      Type(0.0),  Type(5.0),  pen_scale, ksharp);
  pen += smooth_bounds_penalty(wF,         Type(0.0),  Type(5.0),  pen_scale, ksharp);
  pen += smooth_bounds_penalty(wS,         Type(0.0),  Type(5.0),  pen_scale, ksharp);

  nll += pen; // add penalties to objective

  // ---------------------------
  // REPORTING
  // ---------------------------
  REPORT(cots_pred); // predicted COTS abundance (individuals m^-2)
  REPORT(fast_pred); // predicted fast coral cover (%)
  REPORT(slow_pred); // predicted slow coral cover (%)
  REPORT(sigma_eff); // effective observation error (COTS)
  REPORT(phi_fast_eff); // effective Beta precision (fast coral)
  REPORT(phi_slow_eff); // effective Beta precision (slow coral)
  REPORT(pen); // total penalty
  REPORT(cons_buf); // recruitment buffer (diagnostic)

  return nll;
}
