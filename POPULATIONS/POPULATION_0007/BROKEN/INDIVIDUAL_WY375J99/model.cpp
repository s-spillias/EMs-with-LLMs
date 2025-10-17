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
  PARAMETER(alpha_rec);   // Recruitment productivity scaling to juveniles (units: ind m^-2 yr^-1)
  // Density-dependent fecundity exponent (dimensionless), >=1 increases superlinear recruitment
  PARAMETER(phi);         // Fecundity density exponent (unitless)
  // Smooth Allee effect parameters
  PARAMETER(k_allee);     // Allee logistic steepness (m^2 ind^-1)
  PARAMETER(C_allee);     // Allee threshold density (ind m^-2)
  // Food/Resource saturation for larval success
  PARAMETER(K_R);         // Half-saturation coral cover for resource (%, 0-100+)
  PARAMETER(wF);          // Weight of fast coral in resource index (unitless)
  PARAMETER(wS);          // Weight of slow coral in resource index (unitless)
  // Mortality terms (adult)
  PARAMETER(muC);         // Baseline adult mortality (yr^-1)
  PARAMETER(gammaC);      // Density-dependent mortality (m^2 ind^-1 yr^-1)
  // Juvenile stage dynamics
  PARAMETER(mJ);          // Annual maturation fraction from juvenile to adult (yr^-1, 0-1)
  PARAMETER(muJ);         // Juvenile proportional mortality (yr^-1, 0-1)
  // Temperature effect on recruitment (Gaussian peak)
  PARAMETER(T_opt_rec);   // Optimal SST for recruitment (°C)
  PARAMETER(beta_rec);    // Curvature of Gaussian temperature effect (°C^-2)
  // Temperature effect on coral (bleaching loss above threshold)
  PARAMETER(T_opt_bleach); // Onset SST for bleaching loss (°C)
  PARAMETER(beta_bleach);  // Multiplier on growth under heat stress (unitless >=0)
  PARAMETER(m_bleachF);    // Additional fast coral proportional loss per °C above threshold (yr^-1 °C^-1)
  PARAMETER(m_bleachS);    // Additional slow coral proportional loss per °C above threshold (yr^-1 °C^-1)
  // Coral intrinsic regrowth and space competition
  PARAMETER(rF);          // Fast coral intrinsic regrowth (yr^-1 on % scale)
  PARAMETER(rS);          // Slow coral intrinsic regrowth (yr^-1 on % scale)
  PARAMETER(K_tot);       // Total coral carrying capacity (% cover for fast+slow), <=100
  // COTS functional response on corals (multi-prey Holling with Type II/III blend)
  PARAMETER(aF);          // Attack/encounter parameter on fast coral (yr^-1 %^-etaF m^2 ind^-1 scaled)
  PARAMETER(aS);          // Attack/encounter parameter on slow coral (yr^-1 %^-etaS m^2 ind^-1 scaled)
  PARAMETER(etaF);        // Shape exponent for fast coral (>=1)
  PARAMETER(etaS);        // Shape exponent for slow coral (>=1)
  PARAMETER(h);           // Handling/satiation time scaler (yr %^-1)
  PARAMETER(qF);          // Efficiency converting feeding to % cover loss for fast (0-1)
  PARAMETER(qS);          // Efficiency converting feeding to % cover loss for slow (0-1)
  // Observation error parameters
  PARAMETER(sigma_cots);  // Lognormal sd for COTS (log-space)
  PARAMETER(sigma_fast);  // Normal sd for logit(% fast)
  PARAMETER(sigma_slow);  // Normal sd for logit(% slow)
  // Environmental nutrient proxy parameters (new)
  PARAMETER(T_ref_nutr);  // SST (°C) at which nutrient availability is ~0.5 (transition midpoint)
  PARAMETER(kT_nutr);     // Steepness of nutrient decline with SST (°C^-1), >=0
  // Initial state parameters (to avoid using observations in predictions)
  PARAMETER(cots_init);   // Initial adult COTS abundance (ind/m^2), >=0
  PARAMETER(juv_init);    // Initial juvenile COTS abundance (ind/m^2), >=0
  PARAMETER(fast_init);   // Initial fast coral cover (%), [0,100]
  PARAMETER(slow_init);   // Initial slow coral cover (%), [0,100]

  // ------------------------
  // EQUATION DEFINITIONS (discrete-time, yearly)
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
  nll += range_penalty(mJ,        Type(0.0),   Type(1.0),   w_pen);
  nll += range_penalty(muJ,       Type(0.0),   Type(1.0),   w_pen);
  nll += range_penalty(T_opt_rec, Type(20.0),  Type(34.0),  w_pen);
  nll += range_penalty(beta_rec,  Type(0.0),   Type(2.0),   w_pen);
  // Updated to match parameters.json literature-bounded range
  nll += range_penalty(T_opt_bleach, Type(31.0), Type(34.3), w_pen);
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
  // New parameter penalties
  nll += range_penalty(T_ref_nutr,   Type(20.0), Type(32.0), w_pen);
  nll += range_penalty(kT_nutr,      Type(0.0),  Type(2.0),  w_pen);
  // Initial states penalties (plausibility)
  nll += range_penalty(cots_init,    Type(0.0),  Type(100.0), w_pen);
  nll += range_penalty(juv_init,     Type(0.0),  Type(100.0), w_pen);
  nll += range_penalty(fast_init,    Type(0.0),  Type(100.0), w_pen);
  nll += range_penalty(slow_init,    Type(0.0),  Type(100.0), w_pen);

  // Effective observation SDs (floor-added in quadrature for smoothness)
  Type s_cots = CppAD::sqrt(sigma_cots * sigma_cots + sd_floor * sd_floor); // log-space SD for COTS
  Type s_fast = CppAD::sqrt(sigma_fast * sigma_fast + sd_floor * sd_floor); // logit-space SD for fast coral
  Type s_slow = CppAD::sqrt(sigma_slow * sigma_slow + sd_floor * sd_floor); // logit-space SD for slow coral

  // STATE PREDICTIONS
  vector<Type> cots_pred(T); // predicted adult COTS abundance (ind/m^2)
  vector<Type> juv_pred(T);  // predicted juvenile COTS abundance (ind/m^2)
  vector<Type> fast_pred(T); // predicted fast coral cover (%)
  vector<Type> slow_pred(T); // predicted slow coral cover (%)

  // Optional diagnostics (process terms)
  vector<Type> rec_vec(T);       // recruitment to juveniles (ind/m^2/yr)
  vector<Type> mat_vec(T);       // maturation flux to adults (ind/m^2/yr)
  vector<Type> mort_vec(T);      // adult mortality (ind/m^2/yr)
  vector<Type> consF_vec(T);     // consumption loss fast (%/yr)
  vector<Type> consS_vec(T);     // consumption loss slow (%/yr)
  vector<Type> nutrient_idx(T);  // environmental nutrient availability index (0-1), diagnostic

  rec_vec.setZero();
  mat_vec.setZero();
  mort_vec.setZero();
  consF_vec.setZero();
  consS_vec.setZero();
  nutrient_idx.setZero();

  // Initialize with parameterized initial states (no use of observations; avoids data leakage)
  cots_pred(0) = pospart(cots_init);
  juv_pred(0)  = pospart(juv_init);
  fast_pred(0) = pospart(fast_init);
  slow_pred(0) = pospart(slow_init);

  // Soft penalties for exceeding 100% cover at initialization
  nll += w_pen * pow(pospart(fast_pred(0) - Type(100.0)), 2);
  nll += w_pen * pow(pospart(slow_pred(0) - Type(100.0)), 2);
  nll += w_pen * pow(pospart(fast_pred(0) + slow_pred(0) - Type(100.0)), 2);

  // Precompute nutrient availability index from SST for diagnostics (0..T-1)
  for (int t = 0; t < T; t++) {
    Type sst = sst_dat(t);
    nutrient_idx(t) = Type(1.0) / (Type(1.0) + exp(kT_nutr * (sst - T_ref_nutr)));
  }

  // Time stepping using only previous-step predicted states (no use of response *_dat in predictions)
  for (int t = 1; t < T; t++) {
    // Previous states
    Type C_prev = cots_pred(t - 1) + eps;      // adult COTS density (ind/m^2)
    Type J_prev = pospart(juv_pred(t - 1));    // juvenile COTS density (ind/m^2)
    Type F_prev = pospart(fast_pred(t - 1));   // fast coral cover (%)
    Type S_prev = pospart(slow_pred(t - 1));   // slow coral cover (%)

    // Environmental nutrient availability from previous year (0-1)
    Type N_prev = nutrient_idx(t - 1);

    // 1) Resource index and saturation
    Type R = wF * F_prev + wS * S_prev;                 // weighted coral resource (%)
    Type f_food = R / (K_R + R + eps);                  // saturating resource effect (0-1)

    // 2) Smooth Allee effect on adult density
    Type f_Allee = Type(1.0) / (Type(1.0) + exp(-k_allee * (C_prev - C_allee))); // logistic in C

    // 3) Temperature effect on recruitment (Gaussian peak around T_opt_rec) using t-1 SST
    Type dT_rec = sst_dat(t - 1) - T_opt_rec;           // SST deviation from optimal (°C)
    Type f_Trec = exp(-beta_rec * dT_rec * dT_rec);     // 0-1 modifier for recruitment

    // 4) Recruitment to juveniles (plus immigration forcing) using exogenous drivers at t-1
    Type Rec = alpha_rec * pow(C_prev, phi) * f_Allee * f_food * f_Trec + cotsimm_dat(t - 1); // ind/m^2/yr
    rec_vec(t) = Rec;

    // 5) Adult mortality (baseline + density-dependent)
    Type Mort_adult = (muC + gammaC * C_prev) * C_prev; // ind/m^2/yr
    mort_vec(t) = Mort_adult;

    // 6) Juvenile maturation flux and juvenile mortality
    Type Mat = mJ * J_prev;       // ind/m^2/yr
    mat_vec(t) = Mat;
    Type Mort_juv = muJ * J_prev; // ind/m^2/yr

    // 7) Coral modifiers using t-1 SST
    Type temp_excess = pospart(sst_dat(t - 1) - T_opt_bleach);        // °C above threshold
    Type heat_mult = exp(-beta_bleach * temp_excess);                 // growth reduction when hot
    Type space_term = Type(1.0) - (F_prev + S_prev) / (K_tot + eps);  // shared space competition

    Type G_fast = rF * F_prev * space_term * heat_mult * N_prev;      // %/yr growth fast coral
    Type G_slow = rS * S_prev * space_term * heat_mult * N_prev;      // %/yr growth slow coral

    // 8) Additional bleaching losses proportional to temp excess
    Type B_fast = m_bleachF * temp_excess * F_prev; // %/yr loss fast
    Type B_slow = m_bleachS * temp_excess * S_prev; // %/yr loss slow

    // 9) Multi-prey functional response (Type II/III blend)
    Type F_term = aF * pow(F_prev + eps, etaF); // encounter/attack term for fast
    Type S_term = aS * pow(S_prev + eps, etaS); // encounter/attack term for slow
    Type denom = Type(1.0) + h * (F_term + S_term); // saturation denominator

    Type Cons_fast = qF * (F_term * C_prev) / (denom + eps); // %/yr consumed fast
    Type Cons_slow = qS * (S_term * C_prev) / (denom + eps); // %/yr consumed slow
    consF_vec(t) = Cons_fast;
    consS_vec(t) = Cons_slow;

    // 10) Explicit prediction equations (inline; ensure non-negativity; add soft penalties if above 100)
    cots_pred(t) = pospart(C_prev + Mat - Mort_adult);                        // adults
    juv_pred(t)  = pospart(J_prev + Rec - Mat - Mort_juv);                    // juveniles
    fast_pred(t) = pospart(F_prev + G_fast - Cons_fast - B_fast);             // fast coral
    slow_pred(t) = pospart(S_prev + G_slow - Cons_slow - B_slow);             // slow coral

    // Soft penalties for exceeding 100% cover
    nll += w_pen * pow(pospart(fast_pred(t) - Type(100.0)), 2);               // penalize fast > 100
    nll += w_pen * pow(pospart(slow_pred(t) - Type(100.0)), 2);               // penalize slow > 100
    nll += w_pen * pow(pospart(fast_pred(t) + slow_pred(t) - Type(100.0)), 2);// penalize total cover > 100
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
  REPORT(nutrient_idx); // diagnostic: environmental nutrient availability index

  return nll; // return negative log-likelihood for minimization
}
