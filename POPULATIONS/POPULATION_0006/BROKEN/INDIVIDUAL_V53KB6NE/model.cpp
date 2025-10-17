#include <TMB.hpp>

// Smooth utility functions for numerical stability and soft penalties
template<class Type>
Type softplus(Type x, Type k = Type(10.0)) { // smooth positive-part approximation; k sets steepness
  // Use logspace_add for numerical stability and AD-Type compatibility: log(1 + exp(k*x)) = logspace_add(0, k*x)
  return (Type(1.0)/k) * logspace_add(Type(0.0), k * x);
}

template<class Type>
Type smooth_cap_loss(Type available, Type raw_loss, Type eps = Type(1e-8)) {
  // Smoothly cap a loss so it cannot exceed 'available': loss = available * (1 - exp(-raw/available))
  return available * (Type(1.0) - exp(-raw_loss / (available + eps)));
}

template<class Type>
Type inv_logit(Type x) { // numerically stable logistic
  return Type(1.0) / (Type(1.0) + exp(-x));
}

template<class Type>
Type sqr(Type x) { return x * x; }

template<class Type>
Type soft_bound_penalty(Type x, Type lower, Type upper, Type w = Type(1.0)) {
  // Smooth quadratic penalty when outside [lower, upper]
  Type k = Type(10.0);
  Type pen_lo = w * sqr(softplus(lower - x, k)); // >0 when x<lower
  Type pen_hi = w * sqr(softplus(x - upper, k)); // >0 when x>upper
  return pen_lo + pen_hi;
}

/*
Equations (discrete-time with variable step dt = Year(i) - Year(i-1)):

State variables (predicted):
- A_t: cots_pred (individuals m^-2)
- F_t: fast_pred (percent cover, %)
- S_t: slow_pred (percent cover, %)
For computations, coral cover is converted to proportions: Cf = F_t/100, Cs = S_t/100.

Forcings (data):
- SST_t: sst_dat (Celsius)
- IMM_t: cotsimm_dat (individuals m^-2 year^-1)
- Year: calendar year

Parameters are described inline below.

1) Coral-dependent functional response and intake per adult (Type III prey dependence):
   m = 1 + exp(log_mexp)
   w_f = pref_fast (0..1), w_s = 1 - w_f
   F_index = w_f * Cf^m + w_s * Cs^m
   intake_per_adult = a * F_index / (1 + h * a * F_index + eps)          [units: proportion cover per adult per year]
   where a = exp(log_attack), h = exp(log_handling)

2) Partitioned predation on each coral group (per area):
   raw_pred_fast = A * a * w_f * Cf^m / (1 + h * a * F_index + eps)
   raw_pred_slow = A * a * w_s * Cs^m / (1 + h * a * F_index + eps)
   Predation losses applied with smooth capping so losses ≤ available coral:
   loss_fast_pred = smooth_cap_loss(Cf, dt * raw_pred_fast)
   loss_slow_pred = smooth_cap_loss(Cs, dt * raw_pred_slow)

3) Temperature modifiers (Gaussian performance):
   fT_cots = exp(-0.5 * ((SST - Topt_cots)/sigma_cots)^2)
   fT_fast = exp(-0.5 * ((SST - Topt_fast)/sigma_fast)^2)
   fT_slow = exp(-0.5 * ((SST - Topt_slow)/sigma_slow)^2)

4) Bleaching mortality (smooth ramp above threshold):
   ramp = 1 / (1 + exp(-gamma_bleach * (SST - T_bleach)))
   m_bleach_fast = bleach_amp * ramp     [year^-1]
   m_bleach_slow = bleach_amp * ramp     [year^-1]
   Applied as smooth-capped losses: loss_bleach = smooth_cap_loss(C, dt * m_bleach * C)

5) Coral growth and mortality:
   K is carrying capacity (proportion). Competition terms beta_fs, beta_sf in [0,2].
   dCf_growth = dt * g_fast_max * fT_fast * Cf * (1 - (Cf + beta_fs * Cs)/(K + eps))
   dCs_growth = dt * g_slow_max * fT_slow * Cs * (1 - (Cs + beta_sf * Cf)/(K + eps))
   Baseline mortalities: mu_fast_bg, mu_slow_bg (year^-1) with smooth capping on losses.

6) COTS recruitment with Allee effect and immigration:
   f_allee = 1 / (1 + exp(-k_allee * (A - Acrit)))
   recruits_rate_per_adult = f_allee * fT_cots * (fecund_bg + eff_fecund * intake_per_adult)
   recruits = dt * A * recruits_rate_per_adult
   immigration = dt * imm_eff * IMM_prev
   Adult survival over dt: surv = s_cots^dt
   A_next = A + recruits + immigration - (1 - surv) * A = surv * A + recruits + immigration

7) Observation model (lognormal with floors):
   For x in {cots, fast, slow}: log(x_dat + eps_obs) ~ Normal(log(x_pred + eps_obs), sd_x)
   sd_x = sd_floor + exp(log_sd_x)

Initialization:
   cots_pred(0) = cots_dat(0); fast_pred(0) = fast_dat(0); slow_pred(0) = slow_dat(0)
   Subsequent steps use only previous predictions and previous-step forcings.

All '_pred' variables are reported using REPORT().
*/

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------
  // Data
  // ------------------------------
  DATA_VECTOR(Year);        // Calendar year (numeric), used to compute dt between observations
  DATA_VECTOR(cots_dat);    // Adult COTS density (individuals m^-2)
  DATA_VECTOR(fast_dat);    // Fast-growing coral cover (%) - Acropora
  DATA_VECTOR(slow_dat);    // Slow-growing coral cover (%) - Faviidae/Porites
  DATA_VECTOR(sst_dat);     // Sea Surface Temperature (°C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (individuals m^-2 year^-1)

  int n = Year.size(); // Length of time series
  Type eps = Type(1e-8);     // Small constant to avoid division by zero
  Type eps_obs = Type(1e-6); // Small constant for observation model stability
  Type sd_floor = Type(0.05);// Minimum SD for likelihood to prevent degeneracy

  // ------------------------------
  // Parameters
  // ------------------------------
  PARAMETER(log_attack);          // log(a): Attack rate a (year^-1 per unit coral proportion); initial from literature or estimate
  PARAMETER(log_handling);        // log(h): Handling time scaling h (year); sets saturation; estimate
  PARAMETER(log_mexp);            // log(m-1): Exponent beyond 1 for Type III response (m = 1 + exp(log_mexp)); dimensionless
  PARAMETER(eff_fecund_log);      // log(e_eff): Efficiency converting intake to recruits per adult (year^-1 per coral proportion); estimate
  PARAMETER(fecund_bg_log);       // log(b0): Background recruits per adult per year independent of intake (year^-1); estimate
  PARAMETER(s_cots_logit);        // logit(s): Baseline adult survival probability per year (0..1); literature/estimate
  PARAMETER(pref_fast_logit);     // logit(p_f): Preference for fast coral (0..1); slow preference = 1 - p_f; dimensionless
  PARAMETER(log_g_fast);          // log(gF): Max fast-coral growth rate (year^-1); literature/estimate
  PARAMETER(log_g_slow);          // log(gS): Max slow-coral growth rate (year^-1); literature/estimate
  PARAMETER(K_logit);             // logit(K): Coral carrying capacity as proportion of area (0..1); literature/estimate
  PARAMETER(beta_fs_logit);       // logit(b_fs/2): Competition coefficient fast<=-slow scaled to (0,2): b_fs = 2*invlogit(param); dimensionless
  PARAMETER(beta_sf_logit);       // logit(b_sf/2): Competition coefficient slow<=-fast scaled to (0,2); dimensionless
  PARAMETER(Topt_fast);           // Optimal SST for fast-coral growth (°C); literature/estimate
  PARAMETER(Topt_slow);           // Optimal SST for slow-coral growth (°C); literature/estimate
  PARAMETER(Topt_cots);           // Optimal SST for COTS fecundity (°C); literature/estimate
  PARAMETER(log_sigmaT_fast);     // log(sigT_F): Thermal breadth for fast coral (°C); positive
  PARAMETER(log_sigmaT_slow);     // log(sigT_S): Thermal breadth for slow coral (°C); positive
  PARAMETER(log_sigmaT_cots);     // log(sigT_A): Thermal breadth for COTS fecundity (°C); positive
  PARAMETER(T_bleach);            // SST threshold for bleaching ramp (°C); literature/estimate
  PARAMETER(gamma_bleach_log);    // log(gamma): Steepness of bleaching ramp (°C^-1); positive
  PARAMETER(bleach_amp_log);      // log(amp): Max additional coral mortality rate under severe heat (year^-1); positive
  PARAMETER(mu_fast_bg_log);      // log(muF): Background fast-coral mortality (year^-1); positive
  PARAMETER(mu_slow_bg_log);      // log(muS): Background slow-coral mortality (year^-1); positive
  PARAMETER(Acrit_log);           // log(Ac): Allee threshold for COTS (individuals m^-2); positive
  PARAMETER(k_allee_log);         // log(kA): Steepness of Allee effect (m^2 individuals^-1); positive
  PARAMETER(imm_eff_logit);       // logit(q_imm): Efficiency converting larval immigration to local adults (0..1); dimensionless
  PARAMETER(log_sd_cots);         // log(sdA): Observation SD (log-scale) for COTS; positive
  PARAMETER(log_sd_fast);         // log(sdF): Observation SD (log-scale) for fast coral; positive
  PARAMETER(log_sd_slow);         // log(sdS): Observation SD (log-scale) for slow coral; positive

  // ------------------------------
  // Transformations and derived parameters
  // ------------------------------
  Type a = exp(log_attack);                         // Attack rate (year^-1 per coral proportion)
  Type h = exp(log_handling);                       // Handling scaling (year)
  Type mexp = Type(1.0) + exp(log_mexp);           // Type III exponent (>=1)
  Type eff_fecund = exp(eff_fecund_log);           // Intake-to-recruit efficiency (year^-1 per coral proportion)
  Type fecund_bg = exp(fecund_bg_log);             // Background reproduction (year^-1)
  Type s_cots = inv_logit(s_cots_logit);           // Adult survival probability per year (0..1)
  Type pref_fast = inv_logit(pref_fast_logit);     // Preference for fast coral (0..1)
  Type pref_slow = Type(1.0) - pref_fast;          // Preference for slow coral (0..1)
  Type g_fast = exp(log_g_fast);                   // Max growth fast coral (year^-1)
  Type g_slow = exp(log_g_slow);                   // Max growth slow coral (year^-1)
  Type K = inv_logit(K_logit);                     // Carrying capacity (proportion 0..1)
  Type beta_fs = Type(2.0) * inv_logit(beta_fs_logit); // Fast sens. to slow (0..2)
  Type beta_sf = Type(2.0) * inv_logit(beta_sf_logit); // Slow sens. to fast (0..2)
  Type sigma_fast = exp(log_sigmaT_fast);          // Thermal breadth fast coral (°C)
  Type sigma_slow = exp(log_sigmaT_slow);          // Thermal breadth slow coral (°C)
  Type sigma_cots = exp(log_sigmaT_cots);          // Thermal breadth COTS fecundity (°C)
  Type gamma_bleach = exp(gamma_bleach_log);       // Bleaching ramp steepness (°C^-1)
  Type bleach_amp = exp(bleach_amp_log);           // Max bleaching mortality rate (year^-1)
  Type mu_fast_bg = exp(mu_fast_bg_log);           // Background fast coral mortality (year^-1)
  Type mu_slow_bg = exp(mu_slow_bg_log);           // Background slow coral mortality (year^-1)
  Type Acrit = exp(Acrit_log);                     // Allee threshold (individuals m^-2)
  Type k_allee = exp(k_allee_log);                 // Allee steepness
  Type imm_eff = inv_logit(imm_eff_logit);         // Immigration efficiency (0..1)
  Type sd_cots = sd_floor + exp(log_sd_cots);      // Observation SD for COTS (log-scale), floored
  Type sd_fast = sd_floor + exp(log_sd_fast);      // Observation SD for fast coral (log-scale), floored
  Type sd_slow = sd_floor + exp(log_sd_slow);      // Observation SD for slow coral (log-scale), floored

  // ------------------------------
  // Soft biological bounds penalties (smooth)
  // ------------------------------
  Type nll = Type(0.0); // initialize negative log-likelihood
  nll += soft_bound_penalty(Topt_fast, Type(24.0), Type(32.0), Type(1.0));
  nll += soft_bound_penalty(Topt_slow, Type(23.0), Type(32.0), Type(1.0));
  nll += soft_bound_penalty(Topt_cots, Type(23.0), Type(32.0), Type(1.0));
  nll += soft_bound_penalty(sigma_fast, Type(0.3), Type(6.0), Type(1.0));
  nll += soft_bound_penalty(sigma_slow, Type(0.3), Type(6.0), Type(1.0));
  nll += soft_bound_penalty(sigma_cots, Type(0.3), Type(6.0), Type(1.0));
  nll += soft_bound_penalty(K, Type(0.2), Type(0.98), Type(1.0));
  nll += soft_bound_penalty(a, Type(0.01), Type(10.0), Type(1.0));
  nll += soft_bound_penalty(h, Type(0.01), Type(10.0), Type(1.0));
  nll += soft_bound_penalty(g_fast, Type(0.01), Type(2.0), Type(1.0));
  nll += soft_bound_penalty(g_slow, Type(0.01), Type(2.0), Type(1.0));
  nll += soft_bound_penalty(mu_fast_bg, Type(0.001), Type(1.0), Type(1.0));
  nll += soft_bound_penalty(mu_slow_bg, Type(0.001), Type(1.0), Type(1.0));
  nll += soft_bound_penalty(Acrit, Type(0.01), Type(5.0), Type(1.0));
  nll += soft_bound_penalty(k_allee, Type(0.1), Type(20.0), Type(1.0));
  nll += soft_bound_penalty(T_bleach, Type(25.0), Type(33.0), Type(1.0));
  nll += soft_bound_penalty(bleach_amp, Type(0.0), Type(1.5), Type(1.0));

  // ------------------------------
  // Predictions (initialize with observed initial conditions)
  // ------------------------------
  vector<Type> cots_pred(n); cots_pred.setZero();
  vector<Type> fast_pred(n); fast_pred.setZero();
  vector<Type> slow_pred(n); slow_pred.setZero();

  cots_pred(0) = cots_dat(0); // Initial adult COTS density (ind m^-2) from data
  fast_pred(0) = fast_dat(0); // Initial fast coral cover (%) from data
  slow_pred(0) = slow_dat(0); // Initial slow coral cover (%) from data

  // ------------------------------
  // Time stepping
  // ------------------------------
  for (int i = 1; i < n; i++) {
    Type dt = Year(i) - Year(i - 1);            // Time step in years
    dt = (dt <= Type(0.0) ? Type(1.0) : dt);    // Fallback to 1 year if non-positive (stability)

    // Previous step states (predicted), convert coral % to proportion [0,1]
    Type A_prev = cots_pred(i - 1);             // COTS density (ind m^-2)
    Type Cf_prev = fast_pred(i - 1) / Type(100.0); // Fast coral proportion
    Type Cs_prev = slow_pred(i - 1) / Type(100.0); // Slow coral proportion

    // Forcings at previous step
    Type SST_prev = sst_dat(i - 1);             // SST at previous step (°C)
    Type IMM_prev = cotsimm_dat(i - 1);         // Immigration at previous step (ind m^-2 year^-1)

    // Functional response (Type II with Type III prey dependence)
    Type F_index = pref_fast * pow(Cf_prev + eps, mexp) + pref_slow * pow(Cs_prev + eps, mexp); // prey index
    Type denom = (Type(1.0) + h * a * F_index + eps); // saturation denominator
    Type intake_per_adult = a * F_index / denom;      // proportion cover per adult per year

    // Partitioned predation pressure (per area per year)
    Type raw_pred_fast = A_prev * a * pref_fast * pow(Cf_prev + eps, mexp) / denom; // year^-1 * proportion
    Type raw_pred_slow = A_prev * a * pref_slow * pow(Cs_prev + eps, mexp) / denom; // year^-1 * proportion

    // Temperature performance modifiers
    Type fT_cots = exp(-Type(0.5) * sqr((SST_prev - Topt_cots) / (sigma_cots + eps))); // 0..1
    Type fT_fast = exp(-Type(0.5) * sqr((SST_prev - Topt_fast) / (sigma_fast + eps))); // 0..1
    Type fT_slow = exp(-Type(0.5) * sqr((SST_prev - Topt_slow) / (sigma_slow + eps))); // 0..1

    // Bleaching ramp (smooth)
    Type ramp = Type(1.0) / (Type(1.0) + exp(-gamma_bleach * (SST_prev - T_bleach)));
    Type m_bleach = bleach_amp * ramp; // additional mortality rate year^-1 (same for both coral groups here)

    // Coral growth (logistic with competition)
    Type dCf_growth = dt * g_fast * fT_fast * Cf_prev * (Type(1.0) - (Cf_prev + beta_fs * Cs_prev) / (K + eps));
    Type dCs_growth = dt * g_slow * fT_slow * Cs_prev * (Type(1.0) - (Cs_prev + beta_sf * Cf_prev) / (K + eps));

    // Coral losses: predation (smooth-capped), background mortality, bleaching (smooth-capped)
    Type loss_fast_pred = smooth_cap_loss(Cf_prev, dt * raw_pred_fast, eps);
    Type loss_slow_pred = smooth_cap_loss(Cs_prev, dt * raw_pred_slow, eps);

    Type loss_fast_bg = smooth_cap_loss(Cf_prev, dt * mu_fast_bg * Cf_prev, eps);
    Type loss_slow_bg = smooth_cap_loss(Cs_prev, dt * mu_slow_bg * Cs_prev, eps);

    Type loss_fast_bleach = smooth_cap_loss(Cf_prev, dt * m_bleach * Cf_prev, eps);
    Type loss_slow_bleach = smooth_cap_loss(Cs_prev, dt * m_bleach * Cs_prev, eps);

    // Update coral states (ensure proportions remain within [0, ~1] via dynamics; no hard clamps)
    Type Cf_next = Cf_prev + dCf_growth - (loss_fast_pred + loss_fast_bg + loss_fast_bleach);
    Type Cs_next = Cs_prev + dCs_growth - (loss_slow_pred + loss_slow_bg + loss_slow_bleach);

    // Prevent tiny negatives due to numeric roundoff using softplus shift (very small)
    Cf_next = Cf_next + softplus(-Cf_next, Type(20.0)) - softplus(Type(0.0), Type(20.0)); // nudges negatives toward ~0 smoothly
    Cs_next = Cs_next + softplus(-Cs_next, Type(20.0)) - softplus(Type(0.0), Type(20.0));

    // COTS recruitment: Allee + temperature + food intake, plus immigration
    Type f_allee = Type(1.0) / (Type(1.0) + exp(-k_allee * (A_prev - Acrit)));
    Type recruits_rate_per_adult = f_allee * fT_cots * (fecund_bg + eff_fecund * intake_per_adult);
    Type recruits = dt * A_prev * recruits_rate_per_adult;           // individuals m^-2
    Type immigration = dt * imm_eff * IMM_prev;                      // individuals m^-2

    // Adult survival over dt (converted from annual probability)
    Type surv_dt = pow(s_cots + eps, dt);                             // survival over dt
    Type A_next = surv_dt * A_prev + recruits + immigration;          // next-step adults

    // Prevent tiny negatives (should not occur, but keep stable)
    A_next = A_next + softplus(-A_next, Type(20.0)) - softplus(Type(0.0), Type(20.0));

    // Convert coral proportions back to percent for reporting/likelihood
    fast_pred(i) = Type(100.0) * Cf_next;
    slow_pred(i) = Type(100.0) * Cs_next;
    cots_pred(i) = A_next;
  }

  // ------------------------------
  // Likelihood: lognormal errors with floors, include all observations
  // ------------------------------
  for (int i = 0; i < n; i++) {
    // COTS
    nll -= dnorm(log(cots_dat(i) + eps_obs), log(cots_pred(i) + eps_obs), sd_cots, true);
    // Fast coral
    nll -= dnorm(log(fast_dat(i) + eps_obs), log(fast_pred(i) + eps_obs), sd_fast, true);
    // Slow coral
    nll -= dnorm(log(slow_dat(i) + eps_obs), log(slow_pred(i) + eps_obs), sd_slow, true);
  }

  // ------------------------------
  // Reporting
  // ------------------------------
  REPORT(Year);       // Report time axis
  REPORT(cots_pred);  // Predicted COTS (ind m^-2)
  REPORT(fast_pred);  // Predicted fast coral cover (%)
  REPORT(slow_pred);  // Predicted slow coral cover (%)

  // Also report key derived parameters for diagnostics
  REPORT(a);
  REPORT(h);
  REPORT(mexp);
  REPORT(eff_fecund);
  REPORT(fecund_bg);
  REPORT(s_cots);
  REPORT(pref_fast);
  REPORT(K);
  REPORT(beta_fs);
  REPORT(beta_sf);
  REPORT(sigma_fast);
  REPORT(sigma_slow);
  REPORT(sigma_cots);
  REPORT(T_bleach);
  REPORT(gamma_bleach);
  REPORT(bleach_amp);
  REPORT(mu_fast_bg);
  REPORT(mu_slow_bg);
  REPORT(Acrit);
  REPORT(k_allee);
  REPORT(imm_eff);
  REPORT(sd_cots);
  REPORT(sd_fast);
  REPORT(sd_slow);

  return nll;
}
