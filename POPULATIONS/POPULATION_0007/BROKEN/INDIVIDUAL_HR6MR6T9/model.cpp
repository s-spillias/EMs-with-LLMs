#include <TMB.hpp>

// Helper: square
template <class Type>
inline Type sqr(Type x) { return x * x; }

// Helper: softplus for smooth positivity (use log(1 + exp(x)) to work with AD types)
template <class Type>
inline Type softplus(Type x) {
  // Note: log1p is not always overloaded for AD types; use log(1 + exp(x)) instead
  return log(Type(1.0) + exp(x));
}

// Helper: logistic (inverse-logit)
template <class Type>
inline Type logistic(Type x) {
  return Type(1.0) / (Type(1.0) + exp(-x));
}

// Helper: smooth floor to enforce x >= lo without hard cutoffs
template <class Type>
inline Type smooth_floor(Type x, Type lo) {
  // Returns approximately max(x, lo) but smooth: lo + softplus(x - lo)
  return lo + softplus(x - lo);
}

// Helper: smooth penalty to softly bound parameters in [lo, hi]
template <class Type>
inline Type bound_penalty(Type x, Type lo, Type hi) {
  // Quadratic penalty outside [lo, hi]; zero inside
  Type pen = Type(0);
  if (x < lo) pen += sqr(lo - x);
  if (x > hi) pen += sqr(x - hi);
  return pen;
}

template<class Type>
Type objective_function<Type>::operator() ()
{
  // -----------------------------
  // DATA (vectors must use DATA_VECTOR)
  // -----------------------------
  DATA_VECTOR(Year);         // Calendar year (year); used to compute time steps
  DATA_VECTOR(cots_dat);     // Adult COTS density observations (individuals / m^2)
  DATA_VECTOR(fast_dat);     // Fast-growing coral (Acropora) cover observations (% of area)
  DATA_VECTOR(slow_dat);     // Slow-growing coral (Faviidae + Porites) cover observations (% of area)
  DATA_VECTOR(sst_dat);      // Sea-surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);  // Larval immigration rate (individuals / m^2 / year)

  // -----------------------------
  // PARAMETERS (scalars; all with comments about units and roles)
  // -----------------------------
  PARAMETER(r_F);              // year^-1; intrinsic growth rate of fast coral (Acropora)
  PARAMETER(r_S);              // year^-1; intrinsic growth rate of slow coral (Faviidae/Porites)
  PARAMETER(K_F);              // %; carrying capacity for fast coral cover (percent of area)
  PARAMETER(K_S);              // %; carrying capacity for slow coral cover (percent of area)
  PARAMETER(alpha_FS);         // dimensionless; competition effect of slow on fast (per unit slow cover)
  PARAMETER(alpha_SF);         // dimensionless; competition effect of fast on slow

  PARAMETER(a_attack);         // (m^2 ind^-1 yr^-1) scaled; COTS attack rate coefficient in functional response
  PARAMETER(h_handling);       // yr; handling time in Holling type II
  PARAMETER(wF);               // dimensionless; prey weight/preference for fast coral in feeding
  PARAMETER(wS);               // dimensionless; prey weight/preference for slow coral in feeding
  PARAMETER(qF);               // % cover per feeding unit; conversion of feeding to fast-coral loss
  PARAMETER(qS);               // % cover per feeding unit; conversion of feeding to slow-coral loss

  PARAMETER(r_cots);           // year^-1; maximum intrinsic COTS per-capita growth (prey- and temp-modified)
  PARAMETER(m_cots);           // year^-1; baseline COTS mortality
  PARAMETER(k_density);        // (m^2 ind^-1 yr^-1); crowding density-dependence (reduces per-capita growth)
  PARAMETER(e_cots);           // ind m^-2 per feeding unit; efficiency converting feeding to recruits
  PARAMETER(food_half);        // dimensionless; half-saturation constant for food in COTS growth (weighted coral fraction)
  PARAMETER(s_dep);            // (ind^-1 m^2); steepness of depensation logistic
  PARAMETER(N_dep);            // ind m^-2; COTS depensation threshold (Allee inflection)
  PARAMETER(s_imm);            // dimensionless; survival to adult per unit immigration (adds adults)

  PARAMETER(T_opt_coral);      // °C; thermal optimum for coral growth modifier
  PARAMETER(sigma_T_coral);    // °C; thermal breadth for coral growth modifier
  PARAMETER(T_opt_cots);       // °C; thermal optimum for COTS reproduction modifier
  PARAMETER(sigma_T_cots);     // °C; thermal breadth for COTS reproduction modifier
  PARAMETER(mF_temp_max);      // year^-1; additional fast-coral mortality at strong thermal stress
  PARAMETER(mS_temp_max);      // year^-1; additional slow-coral mortality at strong thermal stress

  PARAMETER(log_sd_cots);      // log(sdlog); observation error (lognormal) for COTS
  PARAMETER(log_sd_fast);      // log(sd); observation error (normal) for fast coral cover (%)
  PARAMETER(log_sd_slow);      // log(sd); observation error (normal) for slow coral cover (%)

  // -----------------------------
  // Setup and constants
  // -----------------------------
  Type nll = Type(0);               // Negative log-likelihood accumulator
  const Type eps = Type(1e-8);      // Small constant for numerical stability

  // Constants for stability/penalties (not supplied as DATA_SCALAR to avoid runtime dependency)
  const Type sd_min = Type(0.1);        // Minimum observation SD (corals in %, COTS sdlog)
  const Type penalty_weight = Type(1.0);// Weight for smooth parameter-range penalties

  int n = cots_dat.size();          // Number of time steps from observations
  // Safety penalty if input lengths do not match (keeps likelihood defined)
  if ((Year.size() != n) || (fast_dat.size() != n) || (slow_dat.size() != n) ||
      (sst_dat.size() != n) || (cotsimm_dat.size() != n)) {
    Type mismatch = Type(fabs((double)Year.size() - (double)n)
                       + fabs((double)fast_dat.size() - (double)n)
                       + fabs((double)slow_dat.size() - (double)n)
                       + fabs((double)sst_dat.size()  - (double)n)
                       + fabs((double)cotsimm_dat.size() - (double)n));
    nll += Type(1e6) * (Type(1.0) + mismatch); // strong penalty to discourage misuse
  }

  // Prediction vectors (must be vector<Type> and reported)
  vector<Type> cots_pred(n);   // Predicted COTS density (ind m^-2)
  vector<Type> fast_pred(n);   // Predicted fast coral cover (%)
  vector<Type> slow_pred(n);   // Predicted slow coral cover (%)

  // Diagnostics/reporting of key process rates
  vector<Type> grazing_total(n);   // Total feeding rate per area (feeding units per year)
  vector<Type> temp_mod_coral(n);  // Coral temperature modifier (0-1)
  vector<Type> temp_mod_cots(n);   // COTS temperature modifier (0-1)
  vector<Type> food_index(n);      // Weighted coral availability index (dimensionless)

  // Initialize predictions with first observations to set initial conditions (no data leakage)
  cots_pred(0) = cots_dat(0);    // use observed initial COTS density
  fast_pred(0) = fast_dat(0);    // use observed initial fast coral cover (%)
  slow_pred(0) = slow_dat(0);    // use observed initial slow coral cover (%)

  // Initialize diagnostic vectors
  grazing_total(0) = Type(0);
  temp_mod_coral(0) = Type(1);
  temp_mod_cots(0)  = Type(1);
  food_index(0)     = (wF * (fast_pred(0) / Type(100.0))) + (wS * (slow_pred(0) / Type(100.0)));

  // Observation error SDs with smooth floors for stability
  Type sdlog_cots = smooth_floor(exp(log_sd_cots), sd_min);  // SD on log scale for lognormal
  Type sd_fast    = smooth_floor(exp(log_sd_fast),  sd_min); // SD (%) for fast coral observations
  Type sd_slow    = smooth_floor(exp(log_sd_slow),  sd_min); // SD (%) for slow coral observations

  // -----------------------------
  // DYNAMICS
  // Numbered equations (references in comments):
  // (1) G_t = a * N * R / (1 + a * h * R)   [Holling II total feeding rate]
  // (2) R = wF * F + wS * S                 [Weighted coral availability; F,S in fraction]
  // (3) Coral temp modifier: E_coral = exp(-0.5 * ((T - Topt_coral)/sigma_coral)^2)
  // (4) COTS temp modifier:  E_cots  = exp(-0.5 * ((T - Topt_cots)/sigma_cots)^2)
  // (5) Fast coral: dF = rF*E_coral*F*(1 - (F + alpha_FS*S)/K_F) - qF * G_t * pF - mF_temp_max*(1-E_coral)*F
  // (6) Slow coral: dS = rS*E_coral*S*(1 - (S + alpha_SF*F)/K_S) - qS * G_t * pS - mS_temp_max*(1-E_coral)*S
  // (7) COTS per-capita growth: g = r_cots * f_food * E_cots * Dep - m_cots - k_density * N
  // (8) f_food = R / (R + food_half), Dep = logistic(s_dep * (N - N_dep))
  // (9) N_next = N * exp(g * dt) + e_cots * G_t * dt + s_imm * Imm * dt, with softplus for positivity
  // Note: F,S are represented in fraction internally (0-1), predictions reported in %
  // -----------------------------

  for (int t = 1; t < n; ++t) {
    // Time step (years) with smooth floor
    Type delta_t = Year(t) - Year(t - 1);
    Type dt = smooth_floor(delta_t, Type(1e-6)); // smooth, positive dt

    // Previous-step state (use only predictions to avoid leakage)
    Type N_prev = cots_pred(t - 1);                    // ind m^-2
    Type F_prev_frac = (fast_pred(t - 1) / Type(100)); // fraction
    Type S_prev_frac = (slow_pred(t - 1) / Type(100)); // fraction

    // Environment modifiers
    Type T = sst_dat(t);
    Type Ec = exp( - Type(0.5) * sqr((T - T_opt_coral) / (sigma_T_coral + eps)) ); // (3) coral temp modifier
    Type Es = exp( - Type(0.5) * sqr((T - T_opt_cots)  / (sigma_T_cots  + eps)) ); // (4) COTS temp modifier

    // Weighted coral availability and preferences
    Type R = wF * F_prev_frac + wS * S_prev_frac + eps;  // (2) food index, epsilon to stabilize
    food_index(t) = R;

    // Holling II total feeding by COTS
    Type denom = Type(1.0) + a_attack * h_handling * R;  // dimensionless denominator
    Type G = (a_attack * N_prev * R) / (denom + eps);    // (1) feeding units per year
    grazing_total(t) = G;

    // Allocation of feeding to each coral group
    Type pF = (wF * F_prev_frac) / (R + eps);            // proportional allocation to fast coral
    Type pS = (wS * S_prev_frac) / (R + eps);            // proportional allocation to slow coral

    // Logistic growth with competition and temperature modifier
    Type K_F_frac = (K_F / Type(100.0));                 // carrying capacity as fraction
    Type K_S_frac = (K_S / Type(100.0));

    // Fast coral change (fraction per year)
    Type dF_growth = r_F * Ec * F_prev_frac * (Type(1.0) - (F_prev_frac + alpha_FS * S_prev_frac) / (K_F_frac + eps));
    Type dF_graz   = qF * G * pF;                        // cover removal in fraction units once scaled by qF
    Type dF_tempM  = mF_temp_max * (Type(1.0) - Ec) * F_prev_frac; // thermal stress mortality
    Type F_next_frac = F_prev_frac + dt * (dF_growth - dF_graz - dF_tempM);

    // Slow coral change (fraction per year)
    Type dS_growth = r_S * Ec * S_prev_frac * (Type(1.0) - (S_prev_frac + alpha_SF * F_prev_frac) / (K_S_frac + eps));
    Type dS_graz   = qS * G * pS;                        // cover removal due to grazing
    Type dS_tempM  = mS_temp_max * (Type(1.0) - Ec) * S_prev_frac; // thermal stress mortality
    Type S_next_frac = S_prev_frac + dt * (dS_growth - dS_graz - dS_tempM);

    // Ensure fractions remain non-negative via soft approach (no hard truncation)
    if (F_next_frac < Type(0)) F_next_frac = softplus(F_next_frac) - softplus(Type(0));
    if (S_next_frac < Type(0)) S_next_frac = softplus(S_next_frac) - softplus(Type(0));

    // COTS per-capita growth components
    Type f_food = R / (R + food_half + eps);             // (8) food limitation (0-1)
    Type Dep    = logistic(s_dep * (N_prev - N_dep));    // (8) depensation factor (0-1)
    Type g      = r_cots * f_food * Es * Dep - m_cots - k_density * N_prev; // (7) per-capita net rate (yr^-1)

    // Deterministic update plus recruits from feeding and immigration pulses
    Type N_det   = N_prev * exp(g * dt);                 // density after exponential per-capita growth
    Type recruit = e_cots * G * dt;                      // recruits from feeding
    Type immig   = s_imm * cotsimm_dat(t) * dt;          // exogenous larval immigration contribution
    Type N_next  = N_det + recruit + immig;

    // Soft positivity (avoid hard cutoffs)
    if (N_next < Type(0)) N_next = softplus(N_next);

    // Store predictions (convert coral fractions back to %)
    cots_pred(t) = N_next;
    fast_pred(t) = Type(100.0) * F_next_frac;
    slow_pred(t) = Type(100.0) * S_next_frac;

    // Save temperature modifiers for reporting
    temp_mod_coral(t) = Ec;
    temp_mod_cots(t)  = Es;
  }

  // -----------------------------
  // LIKELIHOOD: include all observations with appropriate distributions
  // -----------------------------
  for (int t = 0; t < n; ++t) {
    // COTS: lognormal via normal on log scale
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sdlog_cots, true);

    // Corals: use normal errors on % scale with minimum SD
    nll -= dnorm(fast_dat(t), fast_pred(t), sd_fast, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sd_slow, true);
  }

  // -----------------------------
  // Smooth penalties to keep parameters within biologically plausible ranges
  // -----------------------------
  nll += penalty_weight * (
      bound_penalty(r_F,        Type(0.0), Type(3.0)) +
      bound_penalty(r_S,        Type(0.0), Type(2.0)) +
      bound_penalty(K_F,        Type(0.0), Type(100.0)) +
      bound_penalty(K_S,        Type(0.0), Type(100.0)) +
      bound_penalty(alpha_FS,   Type(0.0), Type(2.0)) +
      bound_penalty(alpha_SF,   Type(0.0), Type(2.0)) +
      bound_penalty(a_attack,   Type(0.0), Type(10.0)) +
      bound_penalty(h_handling, Type(0.0), Type(5.0)) +
      bound_penalty(wF,         Type(0.0), Type(5.0)) +
      bound_penalty(wS,         Type(0.0), Type(5.0)) +
      bound_penalty(qF,         Type(0.0), Type(200.0)) +
      bound_penalty(qS,         Type(0.0), Type(200.0)) +
      bound_penalty(r_cots,     Type(0.0), Type(5.0)) +
      bound_penalty(m_cots,     Type(0.0), Type(3.0)) +
      bound_penalty(k_density,  Type(0.0), Type(5.0)) +
      bound_penalty(e_cots,     Type(0.0), Type(5.0)) +
      bound_penalty(food_half,  Type(0.001), Type(2.0)) +
      bound_penalty(s_dep,      Type(0.0), Type(10.0)) +
      bound_penalty(N_dep,      Type(0.0), Type(10.0)) +
      bound_penalty(s_imm,      Type(0.0), Type(2.0)) +
      bound_penalty(T_opt_coral,Type(20.0), Type(32.0)) +
      bound_penalty(sigma_T_coral,Type(0.1), Type(8.0)) +
      bound_penalty(T_opt_cots, Type(20.0), Type(32.0)) +
      bound_penalty(sigma_T_cots, Type(0.1), Type(8.0)) +
      bound_penalty(mF_temp_max, Type(0.0), Type(2.0)) +
      bound_penalty(mS_temp_max, Type(0.0), Type(2.0))
    );

  // -----------------------------
  // REPORTING
  // -----------------------------
  REPORT(Year);               // Time vector
  REPORT(cots_pred);          // Predicted COTS density (ind m^-2)
  REPORT(fast_pred);          // Predicted fast coral cover (%)
  REPORT(slow_pred);          // Predicted slow coral cover (%)

  // Diagnostics for interpretation
  REPORT(grazing_total);      // Total feeding rate
  REPORT(temp_mod_coral);     // Temperature modifier for corals
  REPORT(temp_mod_cots);      // Temperature modifier for COTS
  REPORT(food_index);         // Weighted prey availability index

  // For uncertainty propagation if desired
  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);

  return nll;
}
