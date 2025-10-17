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

4) COTS recruitment and population update (food-fueled recruitment, Allee effect, immigration; Ricker crowding for overcompensation):
   R_food_t  = epsilon_repro * (predF_t + predS_t)                                            // Recruits fueled by consumed coral (ind/m2/year)
   A_t       = invlogit(k_allee * (C_t - C_A))                                                // Smooth Allee factor in [0,1]
   R_t       = thermal_t * R_food_t * A_t + alpha_imm * IMM_t                                 // Total recruitment including immigration
   Num_t     = (1 - mC0) * C_t + sJ * R_t                                                     // Post-mortality adults + surviving recruits
   C_{t+1}   = Num_t_pos * exp(- bC * Num_t_pos)                                              // Ricker crowding; Num_t_pos = softplus(Num_t)

5) Observation models (include all data with floors on SDs):
   - COTS (strictly positive): Lognormal
       nll -= dnorm(log(cots_dat), log(cots_pred), sigma_cots, true); nll += log(cots_dat + eps)
   - Coral cover (0–100%): Logit-normal on proportions (x/100)
       nll -= dnorm(logit(fast_dat/100), logit(fast_pred/100), sigma_fast, true)
       nll -= dnorm(logit(slow_dat/100), logit(slow_pred/100), sigma_slow, true)

Initial conditions:
   cots_pred(0) = cots_dat(0)
   fast_pred(0) = fast_dat(0)
   slow_pred(0) = slow_dat(0)
   Forcing predictions are identities: sst_pred = sst_dat; cotsimm_pred = cotsimm_dat.

Notes:
- All transitions use previous time step state variables only (no data leakage).
- Small constants (eps) used to prevent division by zero.
*/

template<class Type>
Type sqr(Type x) { return x * x; }

// Use TMB's built-in invlogit (defined in convenience.hpp)

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
  PARAMETER(sJ);            // dimensionless; Survival/settlement efficiency from recruits to 1-yr-old (entering adult class next step)
  PARAMETER(alpha_imm);     // dimensionless; Scaling of immigration term (units-conversion/efficiency)

  PARAMETER(mC0);       // yr^-1; Baseline adult COTS mortality fraction
  PARAMETER(bC);        // 1/(ind/m2); Ricker crowding coefficient controlling density dependence
  PARAMETER(k_allee);   // 1/(ind/m2); Steepness of Allee effect on reproduction
  PARAMETER(C_A);       // ind/m2; Adult density at which Allee factor is 0.5 (mating/aggregation threshold)

  PARAMETER(sigma_cots_obs); // sd(log-scale); Observation sd for COTS (lognormal)
  PARAMETER(sigma_fast_obs); // sd; Observation sd for logit(fast/100)
  PARAMETER(sigma_slow_obs); // sd; Observation sd for logit(slow/100)

  // ---------------------------
  // PREPARE STORAGE FOR PREDICTIONS
  // ---------------------------
  vector<Type> cots_pred(n);      // Predicted adult COTS density (ind/m2)
  vector<Type> fast_pred(n);      // Predicted fast coral cover (%)
  vector<Type> slow_pred(n);      // Predicted slow coral cover (%)
  vector<Type> sst_pred(n);       // Reported SST forcing (deg C, identity)
  vector<Type> cotsimm_pred(n);   // Reported immigration forcing (ind/m2/year, identity)

  // Initialize predictions using the first observed data point (no data leakage in transitions)
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

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

    // Forcing in the interval [t-1, t)
    Type SST_t = sst_pred(t - 1);
    Type IMM_t = cotsimm_pred(t - 1);

    // Coral consumption by COTS (saturating, resource-limited)
    Type predF_t = F_t * (Type(1) - exp(- aF * C_t / (hF + F_t + eps)));
    Type predS_t = S_t * (Type(1) - exp(- aS * C_t / (hS + S_t + eps)));

    // Temperature effects
    Type z = (SST_t - T_opt) / (T_width + eps);
    Type thermal_t = minTherm + (Type(1) - minTherm) * exp(-Type(0.5) * sqr(z));
    Type bleach_t  = invlogit(k_bleach * (SST_t - T_bleach));

    // Coral growth and updates (logistic with total cover crowding)
    Type growthF_t = rF * F_t * (Type(1) - (F_t + S_t) / (K_coral + eps));
    Type growthS_t = rS * S_t * (Type(1) - (F_t + S_t) / (K_coral + eps));

    Type F_next = softplus(F_t + growthF_t - predF_t - (mF0 * F_t) - (mBF * bleach_t * F_t));
    Type S_next = softplus(S_t + growthS_t - predS_t - (mS0 * S_t) - (mBS * bleach_t * S_t));

    // COTS recruitment and adult update with Ricker crowding
    Type R_food_t = epsilon_repro * (predF_t + predS_t);
    Type A_t      = invlogit(k_allee * (C_t - C_A));
    Type R_t      = thermal_t * R_food_t * A_t + alpha_imm * IMM_t;

    Type Num_t    = (Type(1) - mC0) * C_t + sJ * R_t;
    Type Num_pos  = softplus(Num_t);
    Type C_next   = Num_pos * exp(- bC * Num_pos);

    // Store next states
    fast_pred(t) = F_next;
    slow_pred(t) = S_next;
    cots_pred(t) = C_next;
  }

  // ---------------------------
  // OBSERVATION MODEL / LIKELIHOOD
  // ---------------------------
  Type sigma_cots = sigma_floor_cots + softplus(sigma_cots_obs);
  Type sigma_fast = sigma_floor_coral + softplus(sigma_fast_obs);
  Type sigma_slow = sigma_floor_coral + softplus(sigma_slow_obs);

  Type nll = Type(0.0);

  for (int t = 0; t < n; t++) {
    // COTS: lognormal on density
    Type y_c  = cots_dat(t);
    Type mu_c = cots_pred(t);
    nll -= dnorm(log(y_c + eps), log(mu_c + eps), sigma_cots, true);
    nll += log(y_c + eps); // Jacobian for log-transform

    // Coral covers: logit-normal on proportions
    Type yF_prop  = fast_dat(t) / Type(100.0);
    Type muF_prop = fast_pred(t) / Type(100.0);
    nll -= dnorm(logit_safe(yF_prop, eps), logit_safe(muF_prop, eps), sigma_fast, true);

    Type yS_prop  = slow_dat(t) / Type(100.0);
    Type muS_prop = slow_pred(t) / Type(100.0);
    nll -= dnorm(logit_safe(yS_prop, eps), logit_safe(muS_prop, eps), sigma_slow, true);
  }

  // ---------------------------
  // REPORTS
  // ---------------------------
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(sst_pred);
  REPORT(cotsimm_pred);

  return nll;
}
