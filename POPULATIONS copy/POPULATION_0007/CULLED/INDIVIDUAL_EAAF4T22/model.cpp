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

// Clamp percentage to [0,100] smoothly using pospart
template<class Type>
inline Type clamp_pct(const Type& x) {
  return Type(100.0) - pospart(Type(100.0) - pospart(x));
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
  // Initial states
  PARAMETER(C0);  // initial adult COTS (ind/m^2)
  PARAMETER(J0);  // initial juvenile pool (ind/m^2)
  PARAMETER(F0);  // initial fast coral cover (%)
  PARAMETER(S0);  // initial slow coral cover (%)

  // COTS recruitment scaling (juvenile inputs at unit modifiers)
  PARAMETER(alpha_rec);   // Recruitment productivity scaling to juveniles (units: ind m^-2 yr^-1); sets outbreak potential; initial estimate
  // Density-dependent fecundity exponent (dimensionless), >=1 increases superlinear recruitment
  PARAMETER(phi);         // Fecundity density exponent (unitless); shapes recruitment curvature; literature/initial estimate
  // Smooth Allee effect parameters
  PARAMETER(k_allee);     // Allee logistic steepness (m^2 ind^-1); higher values -> sharper threshold; initial estimate
  PARAMETER(C_allee);     // Allee threshold density (ind m^-2); density at which mating success rises; literature/initial estimate
  // Stock–recruitment high-density taper (Beverton–Holt scale)
  PARAMETER(C_sat_rec);   // Adult density scale for stock–recruitment taper (ind m^-2); proposed improvement
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
  // 4) Juvenile recruitment (plus immigration forcing): Rec = alpha_rec * [C^phi / (1 + C/C_sat_rec)] * f_Allee * f_food * f_Trec + cotsimm
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
  // - All state updates use t-1 values (no data leakage of response variables).
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
  nll += range_penalty(C_sat_rec, Type(0.01),  Type(50.0),  w_pen);
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
  nll += range_penalty(K_tot,        Type(10.0), Type(100.0), w_pen);
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

  // ------------------------
  // STATE VECTORS
  // ------------------------
  vector<Type> cots_pred(T); // adult COTS
  vector<Type> J_pred(T);    // juveniles
  vector<Type> fast_pred(T); // fast coral %
  vector<Type> slow_pred(T); // slow coral %

  // Initialize states (clamp corals to [0,100], keep densities >=0)
  cots_pred(0) = pospart(C0);
  J_pred(0)    = pospart(J0);
  fast_pred(0) = clamp_pct(F0);
  slow_pred(0) = clamp_pct(S0);

  // ------------------------
  // FORWARD SIMULATION (use t-1 states to compute t)
  // ------------------------
  for (int t = 1; t < T; ++t) {
    // Previous states
    Type C = cots_pred(t - 1);
    Type J = J_pred(t - 1);
    Type F = fast_pred(t - 1);
    Type S = slow_pred(t - 1);

    // Exogenous drivers at t-1 to avoid leakage
    Type sst = sst_dat(t - 1);
    Type cotsimm = cotsimm_dat(t - 1);

    // 1) Resource index and saturation
    Type R = wF * F + wS * S;
    Type f_food = R / (K_R + R + eps);

    // 2) Allee effect (smooth logistic)
    Type f_Allee = Type(1.0) / (Type(1.0) + exp(-k_allee * (C - C_allee)));

    // 3) Temperature effect on recruitment
    Type dT = sst - T_opt_rec;
    Type f_Trec = exp(-beta_rec * dT * dT);

    // 4) Recruitment with Beverton–Holt taper
    Type stock = pow(C + eps, phi) / (Type(1.0) + C / (C_sat_rec + eps));
    Type Rec = alpha_rec * stock * f_Allee * f_food * f_Trec + cotsimm;

    // 5) Adult mortality (baseline + density-dependent)
    Type Mort_adult = (muC + gammaC * C) * C;

    // 6) Juvenile flows
    Type Mat = mJ * J;
    Type Mort_juv = muJ * J;

    // 7) Adult update
    Type C_next = C + Mat - Mort_adult;
    C_next = pospart(C_next);

    // 8) Juvenile update
    Type J_next = J + Rec - Mat - Mort_juv;
    J_next = pospart(J_next);

    // 9) Coral growth with shared space and bleaching growth reduction
    Type heat = pospart(sst - T_opt_bleach);
    Type growth_mod = exp(-beta_bleach * heat);
    Type space_term = Type(1.0) - (F + S) / (K_tot + eps);

    Type G_fast = rF * F * space_term * growth_mod;
    Type G_slow = rS * S * space_term * growth_mod;

    // 10) Bleaching additional losses
    Type B_fast = m_bleachF * heat * F;
    Type B_slow = m_bleachS * heat * S;

    // 11) Multi-prey functional response (Type II/III blend)
    Type Fp = pospart(F);
    Type Sp = pospart(S);
    Type denom = Type(1.0) + h * (aF * pow(Fp + eps, etaF) + aS * pow(Sp + eps, etaS));
    Type Cons_fast = qF * (aF * pow(Fp + eps, etaF) * C) / denom;
    Type Cons_slow = qS * (aS * pow(Sp + eps, etaS) * C) / denom;

    // 12) Coral updates and clamping to [0,100]
    Type F_next = F + G_fast - Cons_fast - B_fast;
    Type S_next = S + G_slow - Cons_slow - B_slow;
    F_next = clamp_pct(F_next);
    S_next = clamp_pct(S_next);

    // Store next-step predictions
    cots_pred(t) = C_next;
    J_pred(t)    = J_next;
    fast_pred(t) = F_next;
    slow_pred(t) = S_next;
  }

  // ------------------------
  // OBSERVATION MODEL
  // ------------------------
  // Smooth max with floor using pospart to keep AD-friendly
  Type sd_cots = sigma_cots + pospart(sd_floor - sigma_cots);
  Type sd_fast = sigma_fast + pospart(sd_floor - sigma_fast);
  Type sd_slow = sigma_slow + pospart(sd_floor - sigma_slow);

  for (int t = 0; t < T; ++t) {
    // COTS: lognormal error with Jacobian
    Type yC = cots_dat(t);
    Type muCpred = cots_pred(t);
    // Ensure positivity in log
    Type logy = log(yC + eps);
    Type logmu = log(muCpred + eps);
    nll -= dnorm(logy, logmu, sd_cots, true);
    nll += log(yC + eps); // Jacobian

    // Coral fast: normal on logit(%)
    Type yF = fast_dat(t);
    Type muF = clamp_pct(fast_pred(t));
    nll -= dnorm(logit_pct(yF), logit_pct(muF), sd_fast, true);

    // Coral slow: normal on logit(%)
    Type yS = slow_dat(t);
    Type muS = clamp_pct(slow_pred(t));
    nll -= dnorm(logit_pct(yS), logit_pct(muS), sd_slow, true);
  }

  // ------------------------
  // REPORTING
  // ------------------------
  REPORT(cots_pred);
  REPORT(J_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);

  return nll;
}
