#include <TMB.hpp>

/*
Template Model Builder (TMB) model
Topic: Crown-of-thorns starfish (COTS) outbreak dynamics with coral feedbacks

Equation set (discrete annual time step; t indexes Year):
1) Functional response for coral consumption by COTS (resource-limited, saturating, cannot exceed available coral):
   predF_t = F_t * (1 - exp(- aF * C_t / (hF + F_t + eps)))             // Fast coral (Acropora) consumption (percentage points/year)
   predS_t = S_t * (1 - exp(- aS * C_t / (hS + S_t + eps)))             // Slow coral (Faviidae, Porites) consumption (percentage points/year)

2) Temperature effects (smooth):
   thermal_t = minTherm + (1 - minTherm) * exp(-0.5 * ((SST_t - T_opt)/(T_width + eps))^2)   // Gaussian thermal performance in [minTherm, 1]
   bleach_t  = invlogit(k_bleach * (SST_t - T_bleach))                                        // Smooth bleaching intensity in [0,1]

3) Coral updates (logistic growth with total-cover crowding; selective predation; bleaching and background mortality):
   growthF_t = rF * F_t * (1 - (F_t + S_t) / (K_coral + eps))
   growthS_t = rS * S_t * (1 - (F_t + S_t) / (K_coral + eps))
   F_{t+1}   = softplus(F_t + growthF_t - predF_t - (mF0 * F_t) - (mBF * bleach_t * F_t))    // Softplus keeps cover non-negative
   S_{t+1}   = softplus(S_t + growthS_t - predS_t - (mS0 * S_t) - (mBS * bleach_t * S_t))

4) COTS recruitment, juvenile stage, and adult population update (food-fueled recruitment, Allee effect, immigration; Beverton–Holt for adults):
   R_food_t  = epsilon_repro * (predF_t + predS_t)                                            // Recruits fueled by consumed coral (ind/m2/year)
   A_t       = invlogit(k_allee * (C_t - C_A))                                                // Smooth Allee factor in [0,1]
   R_t       = thermal_t * R_food_t * A_t + alpha_imm * IMM_t                                 // Total larval production incl. immigration

   Juveniles:
     J_{t+1} = softplus(J_t + sJ * R_t - mJ * J_t - gJ * J_t)                                 // Early survival to juvenile pool, mortality, maturation

   Adults (Beverton–Holt crowding after adult mortality + maturation inflow):
     Num_t   = (1 - mC0) * C_t + gJ * J_t                                                     // Post-mortality adults + matured juveniles
     C_{t+1} = Num_t_pos / (1 + bC * Num_t_pos)                                               // Num_t_pos = softplus(Num_t), enforces positivity

5) Observation models (always include all data with floors on SDs):
   - COTS (strictly positive): Lognormal
       nll += -dnorm(log(cots_dat), log(cots_pred), sigma_cots, true) + log(cots_dat + eps)
   - Coral cover (0–100%): Logit-normal on proportions (x/100)
       nll += -dnorm(logit(fast_dat/100), logit(fast_pred/100), sigma_fast, true)
       nll += -dnorm(logit(slow_dat/100), logit(slow_pred/100), sigma_slow, true)

Initial conditions:
   cots_pred(0) = cots_dat(0)
   fast_pred(0) = fast_dat(0)
   slow_pred(0) = slow_dat(0)
   juven_pred(0) = J0
   Forcing predictions are identities: sst_pred = sst_dat; cotsimm_pred = cotsimm_dat.

Notes:
- All transitions use previous time step state variables only (no data leakage).
- Small constants (eps) used to prevent division by zero.
- Smooth penalties gently discourage parameters outside proposed biological ranges.
*/

template<class Type>
Type sqr(Type x) { return x * x; }

// Use TMB's built-in invlogit (defined in convenience.hpp); do not redefine

template<class Type>
Type logit_safe(Type p, Type eps) {
  // Numerically stable logit with small bounds
  Type pp = CppAD::CondExpLt(p, eps, eps, p);
  pp = CppAD::CondExpGt(pp, Type(1) - eps, Type(1) - eps, pp);
  return log(pp / (Type(1) - pp));
}

template<class Type>
Type softplus(Type x) {
  // Numerically stable softplus without using log1p (works with AD types)
  // For large x, softplus(x) ~ x; otherwise, use log(1) + exp(x)
  Type thresh = Type(20.0);
  return CppAD::CondExpGt(x, thresh, x, log(Type(1) + exp(x)));
}

template<class Type>
Type bound_penalty(Type x, Type lo, Type hi, Type w) {
  // Smooth penalty if x goes below lo or above hi (zero if inside range)
  Type pen_lo = softplus(lo - x);
  Type pen_hi = softplus(x - hi);
  return w * (pen_lo + pen_hi);
}

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ---------------------------
  // DATA (from CSVs; do not alter)
  // ---------------------------
  DATA_VECTOR(Year);            // Year (calendar year), consistent with the CSV first column
  DATA_VECTOR(sst_dat);         // Sea-Surface Temperature (deg C)
  DATA_VECTOR(cotsimm_dat);     // Larval immigration (ind/m2/year)
  DATA_VECTOR(cots_dat);        // Adult COTS density (ind/m2)
  DATA_VECTOR(fast_dat);        // Fast-growing coral cover (Acropora), percent
  DATA_VECTOR(slow_dat);        // Slow-growing coral cover (Faviidae/Porites), percent

  int n = Year.size();          // Number of time steps (years)

  // ---------------------------
  // PARAMETERS (with inline units and roles)
  // ---------------------------
  PARAMETER(rF);        // yr^-1; Intrinsic growth rate of fast coral (Acropora); governs recovery speed
  PARAMETER(rS);        // yr^-1; Intrinsic growth rate of slow coral (Faviidae/Porites)
  PARAMETER(K_coral);   // percent; Carrying capacity for total coral cover (F + S) in percent points

  PARAMETER(aF);        // dimensionless; Attack/encounter scaling on fast coral; controls how strongly COTS can remove fast coral
  PARAMETER(aS);        // dimensionless; Attack/encounter scaling on slow coral (lower than aF typically)
  PARAMETER(hF);        // percent; Saturation scale in predation on fast coral; larger => weaker per-capita removal at given cover
  PARAMETER(hS);        // percent; Saturation scale in predation on slow coral

  PARAMETER(mF0);       // yr^-1; Background mortality fraction of fast coral (non-COTS causes)
  PARAMETER(mS0);       // yr^-1; Background mortality fraction of slow coral
  PARAMETER(mBF);       // yr^-1; Bleaching-induced additional fractional mortality multiplier for fast coral
  PARAMETER(mBS);       // yr^-1; Bleaching-induced additional fractional mortality multiplier for slow coral

  PARAMETER(T_opt);     // deg C; Thermal optimum for COTS early life performance (fecundity/survival)
  PARAMETER(T_width);   // deg C; Width (sd) of Gaussian thermal performance curve
  PARAMETER(k_bleach);  // 1/deg C; Steepness of bleaching logistic with temperature
  PARAMETER(T_bleach);  // deg C; Midpoint temperature for bleaching response

  PARAMETER(epsilon_repro); // ind/(m2*percent); Efficiency converting consumed coral (percent points) into recruit production
  PARAMETER(sJ);            // dimensionless; Early survival/settlement efficiency into the cryptic juvenile class
  PARAMETER(alpha_imm);     // dimensionless; Scaling of immigration term (units-conversion/efficiency)

  PARAMETER(mC0);       // yr^-1; Baseline adult COTS mortality fraction
  PARAMETER(bC);        // 1/(ind/m2); Beverton–Holt crowding coefficient controlling density dependence
  PARAMETER(k_allee);   // 1/(ind/m2); Steepness of Allee effect on reproduction
  PARAMETER(C_A);       // ind/m2; Adult density at which Allee factor is 0.5 (mating/aggregation threshold)

  // New juvenile-stage parameters
  PARAMETER(mJ);        // yr^-1; Juvenile (cryptic) mortality fraction
  PARAMETER(gJ);        // yr^-1; Juvenile maturation fraction into adults
  PARAMETER(J0);        // ind/m2; Initial density in the juvenile compartment

  PARAMETER(sigma_cots_obs); // sd(log-scale); Observation sd for COTS (lognormal)
  PARAMETER(sigma_fast_obs); // sd; Observation sd for logit(fast/100)
  PARAMETER(sigma_slow_obs); // sd; Observation sd for logit(slow/100)

  // ---------------------------
  // PREPARE STORAGE FOR PREDICTIONS
  // ---------------------------
  vector<Type> cots_pred(n);      // Predicted adult COTS density (ind/m2)
  vector<Type> fast_pred(n);      // Predicted fast coral cover (%)
  vector<Type> slow_pred(n);      // Predicted slow coral cover (%)
  vector<Type> juven_pred(n);     // Predicted juvenile COTS density (ind/m2)
  vector<Type> sst_pred(n);       // Reported SST forcing (deg C, identity)
  vector<Type> cotsimm_pred(n);   // Reported immigration forcing (ind/m2/year, identity)

  // Initialize predictions using the first observed data point (no data leakage)
  cots_pred(0) = cots_dat(0);     // ind/m2; initial condition from data
  fast_pred(0) = fast_dat(0);     // percent; initial condition from data
  slow_pred(0) = slow_dat(0);     // percent; initial condition from data
  juven_pred(0) = J0;             // ind/m2; estimated initial cryptic juvenile density

  // Forcings treated as known inputs; predicted copies equal observed series
  sst_pred = sst_dat;
  cotsimm_pred = cotsimm_dat;

  // ---------------------------
  // CONSTANTS AND NUMERICAL SAFEGUARDS
  // ---------------------------
  Type eps = Type(1e-8);               // Small constant to prevent division by zero
  Type minTherm = Type(0.1);           // Min thermal performance to avoid zero recruitment
  Type sigma_floor_cots = Type(0.05);  // Minimum SD on log-scale for COTS
  Type sigma_floor_coral = Type(0.02); // Minimum SD for logit-normal coral

  // ---------------------------
  // STATE UPDATE LOOP
  // ---------------------------
  for (int t = 1; t < n; t++) {
    // Previous states (no use of current observations to avoid leakage)
    Type C_t = cots_pred(t - 1);     // ind/m2; adult COTS last year
    Type F_t = fast_pred(t - 1);     // percent; fast coral last year
    Type S_t = slow_pred(t - 1);     // percent; slow coral last year
    Type J_t = juven_pred(t - 1);    // ind/m2; juvenile COTS last year

    // Forcing in the interval [t-1, t)
    Type SST_t = sst_dat(t - 1);     // deg C; used to drive temperature responses
    Type IMM_t = cotsimm_dat(t - 1); // ind/m2/year; larval immigration pulse

    // 1) Functional responses: saturating, cannot exceed available coral (smoothly)
    Type qF = aF * C_t / (hF + F_t + eps);               // dimensionless; scaled predation pressure on fast coral
    Type qS = aS * C_t / (hS + S_t + eps);               // dimensionless; scaled predation pressure on slow coral
    Type predF = F_t * (Type(1) - exp(-qF));             // percent/year; removal of fast coral by COTS
    Type predS = S_t * (Type(1) - exp(-qS));             // percent/year; removal of slow coral by COTS

    // 2) Temperature effects (smooth thermal performance for recruits; smooth bleaching)
    Type thermal = minTherm + (Type(1) - minTherm) * exp(-Type(0.5) * sqr((SST_t - T_opt) / (T_width + eps))); // [minTherm,1]
    Type bleach = invlogit(k_bleach * (SST_t - T_bleach)); // [0,1]; bleaching intensity

    // 3) Coral dynamics (logistic growth with total-cover crowding, selective predation, bleaching + background mortality)
    Type total_cover = F_t + S_t;                         // percent
    Type growthF = rF * F_t * (Type(1) - total_cover / (K_coral + eps)); // percent/year; fast coral growth
    Type growthS = rS * S_t * (Type(1) - total_cover / (K_coral + eps)); // percent/year; slow coral growth

    Type mortF = mF0 * F_t + mBF * bleach * F_t;         // percent/year; non-predation losses of fast coral
    Type mortS = mS0 * S_t + mBS * bleach * S_t;         // percent/year; non-predation losses of slow coral

    Type F_next = softplus(F_t + growthF - predF - mortF);   // percent; next-year fast coral (>=0)
    Type S_next = softplus(S_t + growthS - predS - mortS);   // percent; next-year slow coral (>=0)

    // 4) COTS recruitment and population update (food-fueled + immigration; Allee + Beverton–Holt crowding)
    Type R_food = epsilon_repro * (predF + predS);        // ind/m2/year; recruits fueled by consumption
    Type A = invlogit(k_allee * (C_t - C_A));             // [0,1]; smooth Allee effect on reproduction
    Type R_t = thermal * R_food * A + alpha_imm * IMM_t;  // ind/m2/year; total larval production

    // Juvenile dynamics: early survival/settlement into cryptic pool, mortality and maturation to adults
    Type J_next = softplus(J_t + sJ * R_t - mJ * J_t - gJ * J_t); // ind/m2

    // Adults: post-mortality + maturation inflow, then Beverton–Holt regulation (stable, bounded)
    Type Num = (Type(1) - mC0) * C_t + gJ * J_t;          // ind/m2; adults after mortality plus matured juveniles
    Type Num_pos = softplus(Num);                         // ensure positivity (smooth)
    Type C_next = Num_pos / (Type(1) + bC * Num_pos);     // ind/m2; Beverton–Holt update

    // Assign predictions
    fast_pred(t) = F_next;                                // percent
    slow_pred(t) = S_next;                                // percent
    juven_pred(t) = J_next;                               // ind/m2
    cots_pred(t) = C_next;                                // ind/m2
  }

  // ---------------------------
  // LIKELIHOOD CALCULATION
  // ---------------------------
  Type nll = Type(0);

  // Floors on observation standard deviations
  Type sigma_cots = (sigma_cots_obs < sigma_floor_cots ? sigma_floor_cots : sigma_cots_obs);
  Type sigma_fast = (sigma_fast_obs < sigma_floor_coral ? sigma_floor_coral : sigma_fast_obs);
  Type sigma_slow = (sigma_slow_obs < sigma_floor_coral ? sigma_floor_coral : sigma_slow_obs);

  // COTS: lognormal likelihood (include Jacobian term -log(y))
  for (int t = 0; t < n; t++) {
    Type y = cots_dat(t) + eps;                // observed COTS
    Type mu = cots_pred(t) + eps;              // predicted COTS
    nll -= dnorm(log(y), log(mu), sigma_cots, true); // lognormal kernel
    nll += log(y);                              // Jacobian correction
  }

  // Coral covers: logit-normal on proportions (x scaled by 100)
  for (int t = 0; t < n; t++) {
    // Fast coral
    Type p_obs_f = (fast_dat(t) / Type(100));                     // proportion
    Type p_pre_f = (fast_pred(t) / Type(100));                    // proportion
    Type z_obs_f = logit_safe(p_obs_f, Type(1e-6));               // logit
    Type z_pre_f = logit_safe(p_pre_f, Type(1e-6));               // logit
    nll -= dnorm(z_obs_f, z_pre_f, sigma_fast, true);

    // Slow coral
    Type p_obs_s = (slow_dat(t) / Type(100));                     // proportion
    Type p_pre_s = (slow_pred(t) / Type(100));                    // proportion
    Type z_obs_s = logit_safe(p_obs_s, Type(1e-6));               // logit
    Type z_pre_s = logit_safe(p_pre_s, Type(1e-6));               // logit
    nll -= dnorm(z_obs_s, z_pre_s, sigma_slow, true);
  }

  // ---------------------------
  // SMOOTH PARAMETER BOUND PENALTIES (biologically plausible ranges)
  // ---------------------------
  Type w = Type(1.0); // penalty weight
  nll += bound_penalty(rF, Type(0.0), Type(2.0), w);
  nll += bound_penalty(rS, Type(0.0), Type(2.0), w);
  nll += bound_penalty(K_coral, Type(10.0), Type(100.0), w);

  nll += bound_penalty(aF, Type(0.0), Type(2.0), w);
  nll += bound_penalty(aS, Type(0.0), Type(2.0), w);
  nll += bound_penalty(hF, Type(1.0), Type(100.0), w);
  nll += bound_penalty(hS, Type(1.0), Type(100.0), w);

  nll += bound_penalty(mF0, Type(0.0), Type(1.0), w);
  nll += bound_penalty(mS0, Type(0.0), Type(1.0), w);
  nll += bound_penalty(mBF, Type(0.0), Type(1.0), w);
  nll += bound_penalty(mBS, Type(0.0), Type(1.0), w);

  nll += bound_penalty(T_opt, Type(24.0), Type(32.0), w);
  nll += bound_penalty(T_width, Type(0.1), Type(5.0), w);
  nll += bound_penalty(k_bleach, Type(0.1), Type(5.0), w);
  nll += bound_penalty(T_bleach, Type(27.0), Type(32.5), w);

  nll += bound_penalty(epsilon_repro, Type(0.0), Type(5.0), w);
  nll += bound_penalty(sJ, Type(0.0), Type(1.0), w);
  nll += bound_penalty(alpha_imm, Type(0.0), Type(2.0), w);

  nll += bound_penalty(mC0, Type(0.0), Type(0.9), w);
  nll += bound_penalty(bC, Type(0.0), Type(5.0), w);
  nll += bound_penalty(k_allee, Type(0.1), Type(10.0), w);
  nll += bound_penalty(C_A, Type(0.0), Type(5.0), w);

  // New juvenile parameter bounds and a soft constraint that mJ + gJ <= 1
  nll += bound_penalty(mJ, Type(0.0), Type(1.0), w);
  nll += bound_penalty(gJ, Type(0.0), Type(1.0), w);
  nll += bound_penalty(J0, Type(0.0), Type(5.0), w);
  // Soft penalty if (mJ + gJ) exceeds 1 (ensures the juvenile pool doesn't shrink faster than allowed)
  nll += softplus(mJ + gJ - Type(1.0));

  nll += bound_penalty(sigma_cots_obs, Type(0.01), Type(2.0), w);
  nll += bound_penalty(sigma_fast_obs, Type(0.01), Type(2.0), w);
  nll += bound_penalty(sigma_slow_obs, Type(0.01), Type(2.0), w);

  // ---------------------------
  // REPORTING
  // ---------------------------
  REPORT(Year);           // Report time vector for convenience
  REPORT(sst_pred);       // Reported forcing (identity)
  REPORT(cotsimm_pred);   // Reported forcing (identity)
  REPORT(cots_pred);      // Predicted adult COTS density (ind/m2)
  REPORT(juven_pred);     // Predicted juvenile COTS density (ind/m2)
  REPORT(fast_pred);      // Predicted fast coral cover (%)
  REPORT(slow_pred);      // Predicted slow coral cover (%)

  return nll;
}
