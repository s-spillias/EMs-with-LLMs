#include <TMB.hpp>

// Helper: inverse logit with numerical safety
template<class Type>
Type invlogit_safe(Type x) {
  return Type(1) / (Type(1) + exp(-x));
}

// Helper: smooth softplus for positivity: (1/k)*log(1+exp(k*x))
template<class Type>
Type softplus(Type x, Type k) {
  return (Type(1)/k) * log(Type(1) + exp(k * x));
}

// Helper: smooth barrier penalty to keep x within [L, U]
template<class Type>
Type smooth_barrier(Type x, Type L, Type U, Type scale, Type k) {
  // Positive part approximations via softplus to avoid hard cutoffs
  Type penL = softplus(L - x, k);                   // >0 if x < L, ~0 otherwise
  Type penU = softplus(x - U, k);                   // >0 if x > U, ~0 otherwise
  return scale * (penL * penL + penU * penU);
}

// Helper: soft clip to (0,1) for likelihood transforms (smooth, no hard clamp)
template<class Type>
Type softclip01(Type x, Type s) {
  return Type(0.5) * (tanh(s * (x - Type(0.5))) + Type(1));
}

template<class Type>
Type logit_safe01(Type p, Type eps) {
  // Map via softclip into (0,1), then guard with eps margins for numeric stability
  Type p_soft = softclip01(p, Type(5.0));           // smooth mapping to (0,1)
  Type p_adj  = p_soft * (Type(1) - Type(2) * eps) + eps; // interior to (eps, 1-eps)
  return log(p_adj / (Type(1) - p_adj));
}

template<class Type>
Type square(Type x) { return x * x; }

template<class Type>
Type dmax(Type a, Type b) { return CppAD::CondExpGt(a, b, a, b); }

template<class Type>
Type dmin(Type a, Type b) { return CppAD::CondExpLt(a, b, a, b); }

template<class Type>
Type pos(Type x) { return dmax(x, Type(0)); }

// TMB objective
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Small constants for numerical stability
  const Type eps = Type(1e-8);                      // prevents division by zero / log(0)
  const Type k_barrier = Type(5.0);                 // smoothness for barrier penalties
  const Type s_softclip = Type(5.0);                // smoothness in softclip01
  const Type sigma_min_logn = Type(0.05);           // minimum lognormal sd
  const Type sigma_min_norm = Type(0.05);           // minimum normal sd on transformed scales

  // DATA INPUTS (must match provided column names; Year must match datafile)
  DATA_VECTOR(Year);                                 // Year (calendar year)
  DATA_VECTOR(sst_dat);                              // Sea-Surface Temperature in Celsius
  DATA_VECTOR(cotsimm_dat);                          // COTS larval immigration (ind/m2/yr)
  DATA_VECTOR(cots_dat);                             // Adult COTS abundance (ind/m2)
  DATA_VECTOR(fast_dat);                             // Fast coral cover (%)
  DATA_VECTOR(slow_dat);                             // Slow coral cover (%)

  int n = Year.size();                               // Number of time steps
  // Sanity check: assume all vectors have same length (omitted hard checks to keep AD smooth)

  // PARAMETERS (scalars)
  PARAMETER(r_cots);          // year^-1; intrinsic per-capita growth of adult COTS excluding immigration; estimated from data
  PARAMETER(m_cots);          // year^-1; baseline adult mortality of COTS; estimated from data
  PARAMETER(gamma_cots);      // (ind/m2)^-1 yr^-1; density-dependence strength in COTS (self-limitation); estimated

  PARAMETER(a_fast);          // yr^-1; attack rate on fast coral (Acropora) per COTS per unit prey proportion; estimated
  PARAMETER(a_slow);          // yr^-1; attack rate on slow coral (Faviidae/Porites) per COTS per unit prey proportion; estimated
  PARAMETER(h_fast);          // yr; handling time for fast coral; estimated
  PARAMETER(h_slow);          // yr; handling time for slow coral; estimated

  PARAMETER(e_cons_to_rec);   // ind/m2 per unit coral proportion consumed per year; efficiency from consumption to recruits; estimated

  // Allee effect parameter for local recruitment (fertilization success)
  PARAMETER(N_allee50);       // ind/m2; adult density for 50% fertilization success (Hill exponent = 2)

  PARAMETER(g_fast);          // year^-1; intrinsic growth rate of fast coral; estimated
  PARAMETER(g_slow);          // year^-1; intrinsic growth rate of slow coral; estimated
  PARAMETER(K_total);         // proportion (0-1); shared coral carrying capacity for total cover; estimated

  PARAMETER(beta_imm);        // ind/m2/yr; scaling for larval immigration recruitment; estimated
  PARAMETER(K_imm);           // ind/m2/yr; half-saturation for immigration effect; estimated

  PARAMETER(T_thr_cots);      // deg C; SST logistic threshold for COTS larval survival; estimated
  PARAMETER(k_temp_cots);     // 1/deg C; slope of SST logistic for COTS; estimated

  PARAMETER(Topt_fast);       // deg C; thermal optimum for fast coral growth; estimated
  PARAMETER(Topt_slow);       // deg C; thermal optimum for slow coral growth; estimated
  PARAMETER(sigmaT_fast);     // deg C; thermal breadth (sd) for fast coral growth; estimated
  PARAMETER(sigmaT_slow);     // deg C; thermal breadth (sd) for slow coral growth; estimated

  PARAMETER(alpha_trigger);   // dimensionless; amplitude multiplier for outbreak trigger on COTS growth; estimated
  PARAMETER(k_trigger);       // dimensionless; steepness of outbreak trigger sigmoid; estimated
  PARAMETER(thresh_trigger);  // dimensionless; threshold (in units of prey+imm signal) for outbreak trigger; estimated

  // Observation error standard deviations
  PARAMETER(sigma_cots_obs);  // log scale sd for COTS (lognormal); estimated
  PARAMETER(sigma_fast_obs);  // sd on logit(proportion) for fast coral; estimated
  PARAMETER(sigma_slow_obs);  // sd on logit(proportion) for slow coral; estimated

  // STATE PREDICTIONS (match *_dat names with *_pred)
  vector<Type> cots_pred(n);  // predicted adult COTS abundance (ind/m2)
  vector<Type> fast_pred(n);  // predicted fast coral cover (%)
  vector<Type> slow_pred(n);  // predicted slow coral cover (%)

  // INITIAL CONDITIONS: set to first observation (no data leakage in subsequent steps)
  cots_pred(0) = cots_dat(0); // ind/m2
  fast_pred(0) = fast_dat(0); // %
  slow_pred(0) = slow_dat(0); // %

  // Negative log-likelihood accumulator
  Type nll = Type(0.0);

  // Parameter soft bounds via smooth penalties (biologically meaningful ranges)
  // COTS dynamics parameters
  nll += smooth_barrier(r_cots,       Type(0.0),  Type(5.0),  Type(1.0), k_barrier);
  nll += smooth_barrier(m_cots,       Type(0.0),  Type(5.0),  Type(1.0), k_barrier);
  nll += smooth_barrier(gamma_cots,   Type(0.0),  Type(10.0), Type(0.5), k_barrier);

  // Feeding parameters
  nll += smooth_barrier(a_fast,       Type(0.0),  Type(10.0), Type(0.5), k_barrier);
  nll += smooth_barrier(a_slow,       Type(0.0),  Type(10.0), Type(0.5), k_barrier);
  nll += smooth_barrier(h_fast,       Type(0.0),  Type(10.0), Type(0.2), k_barrier);
  nll += smooth_barrier(h_slow,       Type(0.0),  Type(10.0), Type(0.2), k_barrier);
  nll += smooth_barrier(e_cons_to_rec,Type(0.0),  Type(10.0), Type(0.2), k_barrier);

  // Allee parameter
  nll += smooth_barrier(N_allee50,    Type(0.01), Type(10.0), Type(0.2), k_barrier);

  // Coral growth parameters
  nll += smooth_barrier(g_fast,       Type(0.0),  Type(2.0),  Type(0.5), k_barrier);
  nll += smooth_barrier(g_slow,       Type(0.0),  Type(2.0),  Type(0.5), k_barrier);
  nll += smooth_barrier(K_total,      Type(0.1),  Type(1.0),  Type(1.0), k_barrier);

  // Immigration saturation
  nll += smooth_barrier(beta_imm,     Type(0.0),  Type(10.0), Type(0.2), k_barrier);
  nll += smooth_barrier(K_imm,        Type(0.01), Type(100.0),Type(0.1), k_barrier);

  // Temperature effects
  nll += smooth_barrier(T_thr_cots,   Type(20.0), Type(33.0), Type(0.1), k_barrier);
  nll += smooth_barrier(k_temp_cots,  Type(0.1),  Type(10.0), Type(0.1), k_barrier);

  nll += smooth_barrier(Topt_fast,    Type(24.0), Type(30.0), Type(0.1), k_barrier);
  nll += smooth_barrier(Topt_slow,    Type(24.0), Type(30.0), Type(0.1), k_barrier);
  nll += smooth_barrier(sigmaT_fast,  Type(0.5),  Type(5.0),  Type(0.1), k_barrier);
  nll += smooth_barrier(sigmaT_slow,  Type(0.5),  Type(5.0),  Type(0.1), k_barrier);

  // Outbreak trigger
  nll += smooth_barrier(alpha_trigger,Type(0.0),  Type(10.0), Type(0.2), k_barrier);
  nll += smooth_barrier(k_trigger,    Type(0.1),  Type(20.0), Type(0.1), k_barrier);
  nll += smooth_barrier(thresh_trigger,Type(0.0), Type(2.0),  Type(0.1), k_barrier);

  // Observation errors
  nll += smooth_barrier(sigma_cots_obs,Type(0.001),Type(2.0), Type(0.1), k_barrier);
  nll += smooth_barrier(sigma_fast_obs,Type(0.001),Type(2.0), Type(0.1), k_barrier);
  nll += smooth_barrier(sigma_slow_obs,Type(0.001),Type(2.0), Type(0.1), k_barrier);

  // Time loop for state predictions
  for (int t = 1; t < n; t++) {
    // Previous states (no use of current observations; avoids data leakage)
    Type N_prev = cots_pred(t-1);                   // COTS ind/m2
    Type F_prev = fast_pred(t-1) / Type(100.0);     // fast coral proportion (0-1)
    Type S_prev = slow_pred(t-1) / Type(100.0);     // slow coral proportion (0-1)
    Type TC_prev = F_prev + S_prev;                 // total coral proportion

    // Current environmental forcing influencing transition into time t
    Type T_t = sst_dat(t);                          // SST at year t (deg C)
    Type imm_t = cotsimm_dat(t);                    // larval immigration (ind/m2/yr)

    // Temperature modifiers
    // COTS larval survival/logistic temperature response around threshold
    Type f_T_cots = invlogit_safe(k_temp_cots * (T_t - T_thr_cots)); // [0,1]

    // Coral thermal performance (Gaussian unimodal response)
    Type f_T_fast = exp( - Type(0.5) * square( (T_t - Topt_fast) / (sigmaT_fast + eps) ) );
    Type f_T_slow = exp( - Type(0.5) * square( (T_t - Topt_slow) / (sigmaT_slow + eps) ) );

    // Coral logistic growth under shared space limitation
    Type growth_fast = g_fast * F_prev * (Type(1.0) - (TC_prev / (K_total + eps))) * f_T_fast;
    Type growth_slow = g_slow * S_prev * (Type(1.0) - (TC_prev / (K_total + eps))) * f_T_slow;

    // Multi-prey Holling Type II consumption by COTS
    // Denominator includes handling times across prey; add eps for stability
    Type denom = Type(1.0) + a_fast * h_fast * F_prev + a_slow * h_slow * S_prev + eps;
    Type cons_fast = (a_fast * F_prev * N_prev) / denom;  // proportion/yr consumed of fast
    Type cons_slow = (a_slow * S_prev * N_prev) / denom;  // proportion/yr consumed of slow
    Type cons_total = cons_fast + cons_slow;              // total coral proportion/yr consumed

    // Coral updates (proportions). Use continuous-time Euler step in annual increments.
    Type F_next = F_prev + growth_fast - cons_fast;       // next fast coral proportion
    Type S_next = S_prev + growth_slow - cons_slow;       // next slow coral proportion

    // Immigration saturation and outbreak trigger (smooth)
    Type imm_scaled = imm_t / (K_imm + imm_t + eps);      // [0,1) saturating immigration signal
    Type prey_signal = dmax(Type(0.0), (Type(1.0) * F_prev + Type(0.5) * S_prev)); // weighted prey
    Type trigger_sig = invlogit_safe(k_trigger * (imm_scaled + prey_signal - thresh_trigger)); // [0,1]

    // Allee effect on local recruitment (fertilization success; Hill exponent = 2)
    Type f_allee = square(N_prev) / (square(N_prev) + square(N_allee50) + eps); // [0,1]

    // Recruitment from feeding (efficiency-scaled), modulated by temperature and Allee
    Type rec_food = e_cons_to_rec * cons_total * f_T_cots * f_allee; // ind/m2/yr (local)
    // Recruitment from immigration (saturating), modulated by temperature (not by local Allee)
    Type rec_imm = beta_imm * f_T_cots * imm_scaled;       // ind/m2/yr (external)

    // COTS update (Euler discretization with density dependence and added recruits)
    Type percap = r_cots * f_T_cots * (Type(1.0) + alpha_trigger * trigger_sig) - m_cots - gamma_cots * N_prev;
    Type N_next = N_prev + N_prev * percap + rec_food + rec_imm; // ind/m2

    // Apply very small floor for stability in the state (avoid negative values)
    N_next = dmax(N_next, eps); // smooth via CondExp

    // Assign predictions on their natural scales
    cots_pred(t) = N_next;                     // ind/m2
    fast_pred(t) = (F_next * Type(100.0));     // %
    slow_pred(t) = (S_next * Type(100.0));     // %

    // Soft penalties to keep coral proportions within [0,1] without hard clipping
    nll += smooth_barrier(F_next, Type(0.0), Type(1.0), Type(5.0), k_barrier);
    nll += smooth_barrier(S_next, Type(0.0), Type(1.0), Type(5.0), k_barrier);
  }

  // LIKELIHOOD: include all observations with appropriate distributions and minimum SDs

  // 1) COTS abundance: lognormal error on positive scale
  Type sigma_cots = dmax(sigma_cots_obs, sigma_min_logn); // enforce minimum
  for (int t = 0; t < n; t++) {
    Type mu_log = log(cots_pred(t) + eps);                 // predicted log-mean
    Type y_log  = log(cots_dat(t) + eps);                  // observed log
    nll -= dnorm(y_log, mu_log, sigma_cots, true);         // lognormal via normal on logs
  }

  // 2) Coral covers: normal error on logit-transformed proportions
  Type sigma_fast = dmax(sigma_fast_obs, sigma_min_norm);
  Type sigma_slow = dmax(sigma_slow_obs, sigma_min_norm);
  for (int t = 0; t < n; t++) {
    // Predictions: convert % to proportion and softclip into (0,1)
    Type p_fast_pred = softclip01(fast_pred(t) / Type(100.0), s_softclip);
    Type p_slow_pred = softclip01(slow_pred(t) / Type(100.0), s_softclip);
    // Observations: scale to (0,1) and keep interior via eps-margin
    Type p_fast_obs = fast_dat(t) / Type(100.0);
    Type p_slow_obs = slow_dat(t) / Type(100.0);
    Type z_fast_obs = logit_safe01(p_fast_obs, Type(1e-6));
    Type z_slow_obs = logit_safe01(p_slow_obs, Type(1e-6));
    Type z_fast_pred = logit_safe01(p_fast_pred, Type(1e-6));
    Type z_slow_pred = logit_safe01(p_slow_pred, Type(1e-6));
    nll -= dnorm(z_fast_obs, z_fast_pred, sigma_fast, true);
    nll -= dnorm(z_slow_obs, z_slow_pred, sigma_slow, true);
  }

  // REPORT predictions for external use
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  // Documentation: equation summary
  // (1) f_T_cots = logistic(k_temp_cots * (SST_t - T_thr_cots))
  // (2) f_T_fast/slow = exp(-0.5 * ((SST_t - Topt_{i}) / sigmaT_{i})^2)
  // (3) Coral growth: dC_i = g_i * C_i * (1 - (C_f + C_s)/K_total) * f_T_i
  // (4) Consumption: cons_i = (a_i * C_i * N) / (1 + a_f*h_f*C_f + a_s*h_s*C_s)
  // (5) Immigration signal: imm_scaled = imm_t / (K_imm + imm_t)
  // (6) Outbreak trigger: trigger = logistic(k_trigger * (imm_scaled + prey_signal - thresh_trigger)), prey_signal = 1*F + 0.5*S
  // (7) Allee on local recruitment: f_allee = N^2 / (N^2 + N_allee50^2); rec_food = e_cons_to_rec * (cons_f + cons_s) * f_T_cots * f_allee
  // (8) External recruitment: rec_imm = beta_imm * f_T_cots * imm_scaled (not Allee-limited)
  // (9) COTS per-capita net rate: percap = r_cots * f_T_cots * (1 + alpha_trigger * trigger) - m_cots - gamma_cots * N
  // (10) COTS update: N_t = N_{t-1} + N_{t-1}*percap + rec_food + rec_imm; Coral updates: C_{i,t} = C_{i,t-1} + growth_i - cons_i

  return nll;
}
