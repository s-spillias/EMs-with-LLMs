#include <TMB.hpp>

// Smooth positive part to avoid hard cutoffs and preserve differentiability
template<class Type>
inline Type pospart(const Type& x) {
  return (x + CppAD::sqrt(x * x + Type(1e-8))) / Type(2.0); // smooth ReLU, epsilon prevents NaN
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
  // COTS recruitment scaling (juvenile inputs at unit modifiers)
  PARAMETER(alpha_rec);   // Recruitment productivity scaling to juveniles (units: ind m^-2 yr^-1); sets outbreak potential; initial estimate
  // Density-dependent fecundity exponent (dimensionless), >=1 increases superlinear recruitment
  PARAMETER(phi);         // Fecundity density exponent (unitless); shapes recruitment curvature; literature/initial estimate
  // Smooth Allee effect parameters
  PARAMETER(k_allee);     // Allee logistic steepness (m^2 ind^-1); higher values -> sharper threshold; initial estimate
  PARAMETER(C_allee);     // Allee threshold density (ind m^-2); density at which mating success rises; literature/initial estimate
  // Food/Resource saturation for larval success
  PARAMETER(K_R);         // Half-saturation coral cover for resource (%, 0-100+); initial estimate
  PARAMETER(wF);          // Weight of fast coral in resource index (unitless); initial estimate
  PARAMETER(wS);          // Weight of slow coral in resource index (unitless); initial estimate
  // Mortality terms (adult)
  PARAMETER(muC);         // Baseline adult mortality (yr^-1); initial estimate
  PARAMETER(gammaC);      // Density-dependent mortality (m^2 ind^-1 yr^-1); drives busts at high density; initial estimate
  // Juvenile stage dynamics
  PARAMETER(mJ);          // Annual maturation fraction from juvenile to adult (yr^-1, 0-1); initial estimate
  PARAMETER(muJ);         // Juvenile proportional mortality (yr^-1, 0-1); initial estimate
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

  // ------------------------
  // EQUATION DEFINITIONS (discrete-time, yearly)
  //
  // 1) Resource index (food) at t-1: R = wF*F + wS*S; saturation f_food = R / (K_R + R)
  // 2) Smooth Allee function f_Allee = 1 / (1 + exp(-k_allee*(C - C_allee)))
  // 3) Temperature effect on COTS recruitment: f_Trec = exp( -beta_rec * (SST - T_opt_rec)^2 )
  // 4) Juvenile recruitment (plus immigration forcing): Rec = alpha_rec * C^phi * f_Allee * f_food * f_Trec + cotsimm
  // 5) Adult mortality: Mort_adult = (muC + gammaC * C) * C
  // 6) Juvenile maturation flux: Mat = mJ * J; juvenile mortality: Mort_juv = muJ * J
  // 7) Adult update: C_t = C + Mat - Mort_adult
  // 8) Juvenile update: J_t = J + Rec - Mat - Mort_juv
  // 9) Coral growth (shared space K_tot): G_{fast,slow} = r * Coral * (1 - (F+S)/K_tot) * exp(-beta_bleach * pos(SST - T_opt_bleach))
  // 10) Bleaching loss (additional): B_{fast} = m_bleachF * pos(SST - T_opt_bleach) * Fast; similarly for slow
  // 11) Multi-prey functional response (Type II/III blend):
  //     denom = 1 + h*(aF*F^etaF + aS*S^etaS)
  //     Cons_fast = qF * (aF * F^etaF * C) / denom; Cons_slow = qS * (aS * S^etaS * C) / denom
  // 12) Coral state updates:
  //     F_t = F + G_fast - Cons_fast - B_fast
  //     S_t = S + G_slow - Cons_slow - B_slow
  // Notes:
  // - All state updates use t-1 values (no data leakage).
  // - Small constants avoid division-by-zero and ensure smoothness.
  // ------------------------

  // Negative log-likelihood accumulator
  Type nll = 0.0;
  const Type eps = Type(1e-8);      // small epsilon to stabilize divisions/logs
  const Type sd_floor = Type(0.05); // minimum sd used in likelihood for stability

  // Suggested biological ranges for smooth penalties (very broad, weakly enforced)
  // Weight w_pen controls strength; kept small to avoid dominating data likelihood
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
  nll += range_penalty(mJ,        Type(0.0),   Type(1.0),   w_pen);
  nll += range_penalty(muJ,       Type(0.0),   Type(1.0),   w_pen);
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

  // Effective observation SDs (floor-added in quadrature for smoothness)
  Type s_cots = CppAD::sqrt(sigma_cots * sigma_cots + sd_floor * sd_floor); // log-space SD for COTS
  Type s_fast = CppAD::sqrt(sigma_fast * sigma_fast + sd_floor * sd_floor); // logit-space SD for fast coral
  Type s_slow = CppAD::sqrt(sigma_slow * sigma_slow + sd_floor * sd_floor); // logit-space SD for slow coral

  // STATE PREDICTIONS
  vector<Type> cots_pred(T); // predicted adult COTS abundance (ind/m^2)
  vector<Type> juv_pred(T);  // predicted juvenile COTS abundance (ind/m^2)
  vector<Type> fast_pred(T); // predicted fast coral cover (%)
  vector<Type> slow_pred(T); // predicted slow coral cover (%)

  // Initialize with first observations to avoid parameterized initial states (no data leakage)
  cots_pred(0) = cots_dat(0); // adult ind/m^2 at Year(0)
  fast_pred(0) = fast_dat(0); // % cover at Year(0)
  slow_pred(0) = slow_dat(0); // % cover at Year(0)
  juv_pred(0)  = Type(0.0);   // no direct observation; neutral initialization

  // Optional diagnostics (process terms)
  vector<Type> rec_vec(T);       // recruitment to juveniles (ind/m^2/yr)
  vector<Type> mat_vec(T);       // maturation flux to adults (ind/m^2/yr)
  vector<Type> mort_vec(T);      // adult mortality (ind/m^2/yr)
  vector<Type> consF_vec(T);     // consumption loss fast (%/yr)
  vector<Type> consS_vec(T);     // consumption loss slow (%/yr)

  rec_vec.setZero();
  mat_vec.setZero();
  mort_vec.setZero();
  consF_vec.setZero();
  consS_vec.setZero();

  // Time stepping using only previous-step states (no use of current observations)
  for (int t = 1; t < T; t++) {
    // Previous states
    Type C_prev = cots_pred(t - 1) + eps; // previous adult COTS density (ind/m^2), eps for stability
    Type J_prev = pospart(juv_pred(t - 1)); // previous juvenile COTS density (ind/m^2), nonnegative
    Type F_prev = pospart(fast_pred(t - 1)); // previous fast coral cover (%), nonnegative
    Type S_prev = pospart(slow_pred(t - 1)); // previous slow coral cover (%), nonnegative

    // 1) Resource index and saturation
    Type R = wF * F_prev + wS * S_prev; // weighted coral resource (%-weighted)
    Type f_food = R / (K_R + R + eps);  // saturating resource effect (0-1)

    // 2) Smooth Allee effect on adult density
    Type f_Allee = Type(1.0) / (Type(1.0) + exp(-k_allee * (C_prev - C_allee))); // logistic in C

    // 3) Temperature effect on recruitment (Gaussian peak around T_opt_rec)
    Type dT_rec = sst_dat(t - 1) - T_opt_rec; // SST deviation from optimal (°C)
    Type f_Trec = exp(-beta_rec * dT_rec * dT_rec); // 0-1 modifier for recruitment

    // 4) Recruitment to juveniles (plus immigration forcing)
    Type Rec = alpha_rec * pow(C_prev, phi) * f_Allee * f_food * f_Trec + cotsimm_dat(t - 1); // ind/m^2/yr
    rec_vec(t) = Rec;

    // 5) Adult mortality (baseline + density-dependent)
    Type Mort_adult = (muC + gammaC * C_prev) * C_prev; // ind/m^2/yr
    mort_vec(t) = Mort_adult;

    // 6) Juvenile maturation flux and juvenile mortality
    Type Mat = mJ * J_prev;     // ind/m^2/yr
    mat_vec(t) = Mat;
    Type Mort_juv = muJ * J_prev; // ind/m^2/yr

    // 7) Adult state update
    Type C_next = pospart(C_prev + Mat - Mort_adult); // ensure non-negative
    cots_pred(t) = C_next;

    // 8) Juvenile state update
    Type J_next = pospart(J_prev + Rec - Mat - Mort_juv); // ensure non-negative
    juv_pred(t) = J_next;

    // 9) Coral growth with shared space (logistic) and heat stress multiplier
    Type temp_excess = pospart(sst_dat(t - 1) - T_opt_bleach); // °C above threshold
    Type heat_mult = exp(-beta_bleach * temp_excess); // reduces growth smoothly when hot

    Type space_term = Type(1.0) - (F_prev + S_prev) / (K_tot + eps); // shared space competition
    Type G_fast = rF * F_prev * space_term * heat_mult;  // %/yr growth for fast coral
    Type G_slow = rS * S_prev * space_term * heat_mult;  // %/yr growth for slow coral

    // 10) Additional bleaching losses proportional to temp excess
    Type B_fast = m_bleachF * temp_excess * F_prev; // %/yr loss
    Type B_slow = m_bleachS * temp_excess * S_prev; // %/yr loss

    // 11) Multi-prey functional response (Type II/III blend)
    Type F_term = aF * pow(F_prev + eps, etaF); // encounter/attack term for fast
    Type S_term = aS * pow(S_prev + eps, etaS); // encounter/attack term for slow
    Type denom = Type(1.0) + h * (F_term + S_term); // saturation denominator (unitless)

    Type Cons_fast = qF * (F_term * C_prev) / (denom + eps); // %/yr consumed fast
    Type Cons_slow = qS * (S_term * C_prev) / (denom + eps); // %/yr consumed slow
    consF_vec(t) = Cons_fast;
    consS_vec(t) = Cons_slow;

    // 12) Coral state updates (ensure non-negativity; soft penalty if above 100)
    Type F_next = pospart(F_prev + G_fast - Cons_fast - B_fast); // next fast cover (%)
    Type S_next = pospart(S_prev + G_slow - Cons_slow - B_slow); // next slow cover (%)

    // Soft penalties for exceeding 100% cover (avoid hard truncation)
    nll += w_pen * pow(pospart(F_next - Type(100.0)), 2); // penalize F_next > 100
    nll += w_pen * pow(pospart(S_next - Type(100.0)), 2); // penalize S_next > 100
    nll += w_pen * pow(pospart(F_next + S_next - Type(100.0)), 2); // penalize total cover > 100

    fast_pred(t) = F_next;
    slow_pred(t) = S_next;
  }

  // ------------------------
  // LIKELIHOOD: include all observations (t = 0..T-1)
  // ------------------------
  for (int t = 0; t < T; t++) {
    // COTS: lognormal error
    Type y_c = log(cots_dat(t) + eps);        // observed log abundance
    Type mu_c = log(cots_pred(t) + eps);      // predicted log abundance
    nll -= dnorm(y_c, mu_c, s_cots, true);    // accumulate log-likelihood

    // Corals: logit-normal error on % cover in [0,100]
    Type y_f = logit_pct(fast_dat(t));        // observed logit(%)
    Type mu_f = logit_pct(fast_pred(t));      // predicted logit(%)
    nll -= dnorm(y_f, mu_f, s_fast, true);    // accumulate

    Type y_s = logit_pct(slow_dat(t));        // observed logit(%)
    Type mu_s = logit_pct(slow_pred(t));      // predicted logit(%)
    nll -= dnorm(y_s, mu_s, s_slow, true);    // accumulate
  }

  // ------------------------
  // REPORTING
  // ------------------------
  REPORT(Year);        // report time vector for alignment
  REPORT(cots_pred);   // predicted adult COTS abundance (ind/m^2)
  REPORT(juv_pred);    // predicted juvenile COTS abundance (ind/m^2)
  REPORT(fast_pred);   // predicted fast coral cover (%)
  REPORT(slow_pred);   // predicted slow coral cover (%)
  REPORT(rec_vec);     // process diagnostic: recruitment to juveniles
  REPORT(mat_vec);     // process diagnostic: maturation flux to adults
  REPORT(mort_vec);    // process diagnostic: adult mortality
  REPORT(consF_vec);   // process diagnostic: consumption fast
  REPORT(consS_vec);   // process diagnostic: consumption slow

  return nll; // return negative log-likelihood for minimization
}
