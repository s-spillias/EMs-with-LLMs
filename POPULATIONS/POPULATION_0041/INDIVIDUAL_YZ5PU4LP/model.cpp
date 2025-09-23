#include <TMB.hpp>

// ------------------------------------------------------------
// Modeling episodic COTS outbreaks with coral feedbacks
// ------------------------------------------------------------
/*
Model summary (equations; all transitions use previous-step states only):

Let:
A_t  = adult COTS abundance (ind m^-2)
F_t  = fast coral (Acropora) cover fraction (0..1)
S_t  = slow coral (Faviidae/Porites) cover fraction (0..1)
K    = total coral carrying capacity (fraction of reef area; 0<K<1)
Food_t = alpha*F_t + (1-alpha)*S_t (weighted edible coral mix; 0..K)

Forcing:
SST_t = sea-surface temperature (°C)
L_t   = larval immigration (ind m^-2 yr^-1)

Auxiliary smooth functions:
invlogit(x) = 1 / (1 + exp(-x))
softplus(x) = log(1 + exp(x))

Temperature modifiers:
(1) Coral growth suitability (bell-shaped):
    G_coral,t(species) = exp( - ((SST_t - sst_opt_coral)^2) / (2 * tau_temp^2) * q_species )
    where q_fast, q_slow modulate sensitivity (>=0). 0..1.

(2) Bleaching pressure (smooth threshold):
    B_t = invlogit( beta_bleach * (SST_t - thr_bleach) ) in (0,1)
    Species-specific instantaneous bleaching loss rates m_bleach_fast, m_bleach_slow.

Predation (Holling type II; preference alpha for fast coral):
(3) Per-predator intake (fraction of cover per yr):
    I_t = (a * Food_t) / (1 + a * h * Food_t)
    where a = attack rate; h = handling time.
(4) Partitioned removal using a smooth cap (never exceeds available coral):
    R_fast,t = F_t * (1 - exp( -I_t * alpha * A_t / (F_t + eps) ))
    R_slow,t = S_t * (1 - exp( -I_t * (1-alpha) * A_t / (S_t + eps) ))

Coral dynamics with shared space (F+S ≤ K encouraged by growth form and penalties):
(5) F_{t+1} = F_t + r_fast * F_t * (1 - (F_t + S_t)/K) * G_coral,t(fast) - R_fast,t - [1 - exp(-m_bleach_fast * B_t)] * F_t
(6) S_{t+1} = S_t + r_slow * S_t * (1 - (F_t + S_t)/K) * G_coral,t(slow) - R_slow,t - [1 - exp(-m_bleach_slow * B_t)] * S_t

COTS dynamics (Ricker with covariates; always positive):
(7) Food index scaled to [0,1]:  X_t = Food_t / (K + eps)
    Food suitability (smooth threshold-like using logistic around sst_thr_A):
    T_A,t = invlogit( betaA * (SST_t - sst_thr_A) )
(8) Effective log growth: r_eff,t = rA0 + gamma_food * X_t + (T_A,t - 0.5)  // centered temp effect
(9) A_{t+1} = A_t * exp( r_eff,t - densA * A_t ) + lambda_imm * L_t + eta_repro * (I_t * A_t)

Observation models (all timesteps included):
(10) COTS: log(cots_dat_t + eps) ~ Normal( log(A_t + eps), sigma_cots )
(11) Fast coral (%):  fast_dat_t ~ Normal( 100*F_t, sigma_fast )
(12) Slow coral (%):  slow_dat_t ~ Normal( 100*S_t, sigma_slow )

Soft state constraints (smooth penalties, no hard clamps):
(13) Penalize F_t,S_t outside [0,K] and (F_t+S_t) outside [0,K] using softplus barriers scaled by pen_strength.
*/

template<class Type>
Type objective_function<Type>::operator() ()
{
  using CppAD::log1p;

  // -------------------- Data (from merged CSVs) --------------------
  DATA_VECTOR(Year);          // Year (time; first column; used for alignment only)
  DATA_VECTOR(cots_dat);      // Observed adult COTS (ind m^-2)
  DATA_VECTOR(fast_dat);      // Observed fast coral cover (%)
  DATA_VECTOR(slow_dat);      // Observed slow coral cover (%)
  DATA_VECTOR(sst_dat);       // Sea-surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);   // Larval immigration (ind m^-2 yr^-1)

  int T = Year.size();        // Number of timesteps (years)

  // -------------------- Parameters --------------------
  // Initial conditions
  PARAMETER(log_A0);          // ln of initial COTS abundance at t=0 (ind m^-2); choose near typical pre-outbreak level from data
  PARAMETER(logit_F0);        // logit of initial fast coral fraction as proportion of K (unitless), invlogit(logit_F0) in (0,1)
  PARAMETER(logit_S0);        // logit of initial slow coral fraction as proportion of remaining capacity (unitless), in (0,1)

  // COTS population dynamics
  PARAMETER(rA0);             // Baseline intrinsic log growth (per yr) in Ricker exponent for COTS; can be negative to represent decline
  PARAMETER(log_densA);       // ln density-feedback coefficient (m^2 ind^-1 yr^-1); larger means stronger self-limitation
  PARAMETER(gamma_food);      // Effect of food availability X_t (0..1) on COTS log growth (per yr)
  PARAMETER(betaA);           // Slope of logistic SST effect on COTS reproduction (per °C)
  PARAMETER(sst_thr_A);       // SST at which logistic enhancement is 0.5 (°C)
  PARAMETER(log_lambda_imm);  // ln( conversion from larval immigration to adults ) (m^2)
  PARAMETER(log_eta_repro);   // ln( efficiency converting intake to reproduction ) (ind predator^-1 yr^-1 in same units as A via intake)

  // Trophic interaction (predation)
  PARAMETER(alpha_pref_logit); // logit preference for fast coral (Acropora); invlogit gives alpha in (0,1)
  PARAMETER(log_a);            // ln attack rate a (yr^-1 per fraction coral)
  PARAMETER(log_h);            // ln handling time h (yr per fraction coral)

  // Coral growth and carrying capacity
  PARAMETER(log_rF);          // ln intrinsic growth for fast coral (yr^-1)
  PARAMETER(log_rS);          // ln intrinsic growth for slow coral (yr^-1)
  PARAMETER(K_tot_logit);     // logit of total coral carrying capacity K (fraction of reef; (0,1))

  // Temperature effects on coral growth
  PARAMETER(sst_opt_coral);   // SST optimum for coral growth (°C)
  PARAMETER(log_tau_temp);    // ln width parameter (°C) controlling breadth of growth-temperature curve
  PARAMETER(q_fast);          // Dimensionless sensitivity scaling for fast coral (>=0)
  PARAMETER(q_slow);          // Dimensionless sensitivity scaling for slow coral (>=0)

  // Bleaching (smooth threshold mortality)
  PARAMETER(thr_bleach);        // SST threshold for bleaching onset (°C)
  PARAMETER(beta_bleach);       // Slope of bleaching logistic vs temperature (per °C)
  PARAMETER(log_m_bleach_fast); // ln instantaneous bleaching loss rate for fast coral (yr^-1)
  PARAMETER(log_m_bleach_slow); // ln instantaneous bleaching loss rate for slow coral (yr^-1)

  // Observation errors (floors applied inside likelihood)
  PARAMETER(log_sigma_cots);   // ln SD for lognormal COTS observations
  PARAMETER(log_sigma_fast);   // ln SD for Normal fast coral (%)
  PARAMETER(log_sigma_slow);   // ln SD for Normal slow coral (%)

  // Soft penalty strength
  PARAMETER(log_pen_strength); // ln weight for soft state constraints

  // -------------------- Transforms and constants --------------------
  Type eps = Type(1e-8);                 // Small constant for numerical stability
  Type A0 = exp(log_A0) + eps;           // Initial COTS abundance (ind m^-2), strictly positive
  Type alpha = Type(1.0) / (Type(1.0) + exp(-alpha_pref_logit)); // Preference for fast coral (0..1)
  Type a = exp(log_a);                   // Attack rate (yr^-1 per fraction coral)
  Type h = exp(log_h);                   // Handling time (yr per fraction)
  Type rF = exp(log_rF);                 // Fast coral growth rate (yr^-1)
  Type rS = exp(log_rS);                 // Slow coral growth rate (yr^-1)
  Type K = Type(1.0) / (Type(1.0) + exp(-K_tot_logit));          // Total coral capacity (0..1)
  Type tau = exp(log_tau_temp);          // Temperature breadth (°C > 0)
  Type mBleachF = exp(log_m_bleach_fast);// Bleaching loss fast (yr^-1)
  Type mBleachS = exp(log_m_bleach_slow);// Bleaching loss slow (yr^-1)
  Type densA = exp(log_densA);           // Density feedback coefficient
  Type lambda_imm = exp(log_lambda_imm); // Immigration conversion
  Type eta_repro = exp(log_eta_repro);   // Intake-to-reproduction efficiency

  Type sigma_cots = exp(log_sigma_cots); // Obs SD for log COTS
  Type sigma_fast = exp(log_sigma_fast); // Obs SD for fast cover (%)
  Type sigma_slow = exp(log_sigma_slow); // Obs SD for slow cover (%)

  Type pen_w = exp(log_pen_strength);    // Penalty weight

  // Minimum SD floors (guards)
  sigma_cots = CppAD::CondExpLt(sigma_cots, Type(0.05), Type(0.05), sigma_cots);
  sigma_fast = CppAD::CondExpLt(sigma_fast, Type(0.5), Type(0.5), sigma_fast);
  sigma_slow = CppAD::CondExpLt(sigma_slow, Type(0.5), Type(0.5), sigma_slow);

  // Helper lambdas
  auto invlogit_fun = [&](Type x)->Type{ return Type(1.0) / (Type(1.0) + exp(-x)); };
  auto softplus_fun = [&](Type x)->Type{ return log1p(exp(x)); };
  auto barrier_pen = [&](Type x, Type lower, Type upper)->Type{
    // Smooth penalty encouraging x in [lower, upper]
    Type k = Type(10.0);
    Type p = softplus_fun(k * (lower - x)) + softplus_fun(k * (x - upper));
    return p / k;
  };

  // -------------------- State containers --------------------
  vector<Type> A_pred(T);   // COTS predictions (ind m^-2)
  vector<Type> F_frac(T);   // Fast coral fraction (0..1)
  vector<Type> S_frac(T);   // Slow coral fraction (0..1)

  // Initialize coral fractions from K with smooth decomposition that ensures F0+S0 <= K
  Type pF0 = invlogit_fun(logit_F0);       // (0..1)
  Type F0 = K * pF0;                       // share up to K
  Type pS0 = invlogit_fun(logit_S0);       // (0..1)
  Type S0 = (K - F0) * pS0;                // remaining capacity portion
  F_frac(0) = F0;
  S_frac(0) = S0;
  A_pred(0) = A0;

  // -------------------- Likelihood and penalties --------------------
  Type nll = 0.0;
  Type pen_states = 0.0;

  // ---- t = 0 observation likelihood ----
  // COTS lognormal
  nll -= dnorm(log(cots_dat(0) + eps), log(A_pred(0) + eps), sigma_cots, true);
  // Coral percent normals
  nll -= dnorm(fast_dat(0), F_frac(0) * Type(100.0), sigma_fast, true);
  nll -= dnorm(slow_dat(0), S_frac(0) * Type(100.0), sigma_slow, true);

  // State soft penalties at t=0
  pen_states += barrier_pen(F_frac(0), Type(0.0), K);
  pen_states += barrier_pen(S_frac(0), Type(0.0), K);
  pen_states += barrier_pen(F_frac(0) + S_frac(0), Type(0.0), K);

  // -------------------- State transitions and likelihood --------------------
  for (int t = 1; t < T; ++t) {
    Type F_prev = F_frac(t-1);
    Type S_prev = S_frac(t-1);
    Type A_prev = A_pred(t-1);
    Type SST_prev = sst_dat(t-1);
    Type L_prev = cotsimm_dat(t-1);

    // Temperature modifiers
    // Coral growth suitability (bell shape), bounded (0,1]
    Type diff_fast = (SST_prev - sst_opt_coral) / (tau + eps);
    Type diff_slow = diff_fast; // same optimum, different sensitivity
    Type G_fast = exp( - (diff_fast * diff_fast) * CppAD::CondExpLt(q_fast, Type(0.0), Type(0.0), q_fast + eps) );
    Type G_slow = exp( - (diff_slow * diff_slow) * CppAD::CondExpLt(q_slow, Type(0.0), Type(0.0), q_slow + eps) );

    // Bleaching pressure (0..1)
    Type B_prev = invlogit_fun(beta_bleach * (SST_prev - thr_bleach));

    // Food index and predation intake (Holling II)
    Type Food_prev = alpha * F_prev + (Type(1.0) - alpha) * S_prev; // (0..K)
    Type I_prev = (a * Food_prev) / (Type(1.0) + a * h * Food_prev + eps); // per predator per yr

    // Smooth, capacity-limited coral removal by predation
    Type R_fast = F_prev * (Type(1.0) - exp( - (I_prev * alpha * A_prev) / (F_prev + eps) ));
    Type R_slow = S_prev * (Type(1.0) - exp( - (I_prev * (Type(1.0) - alpha) * A_prev) / (S_prev + eps) ));

    // Bleaching instantaneous loss (smooth capped)
    Type lossF_bleach = F_prev * (Type(1.0) - exp(-mBleachF * B_prev));
    Type lossS_bleach = S_prev * (Type(1.0) - exp(-mBleachS * B_prev));

    // Coral logistic growth with shared space
    Type growthF = rF * F_prev * (Type(1.0) - (F_prev + S_prev) / (K + eps)) * G_fast;
    Type growthS = rS * S_prev * (Type(1.0) - (F_prev + S_prev) / (K + eps)) * G_slow;

    // Update coral states
    Type F_curr = F_prev + growthF - R_fast - lossF_bleach;
    Type S_curr = S_prev + growthS - R_slow - lossS_bleach;

    // Soft penalties for coral bounds and total occupancy
    pen_states += barrier_pen(F_curr, Type(0.0), K);
    pen_states += barrier_pen(S_curr, Type(0.0), K);
    pen_states += barrier_pen(F_curr + S_curr, Type(0.0), K);

    // COTS growth modifiers
    Type X_food = Food_prev / (K + eps);             // (0..1)
    Type T_A = invlogit_fun(betaA * (SST_prev - sst_thr_A)); // (0..1)
    Type r_eff = rA0 + gamma_food * X_food + (T_A - Type(0.5)); // centered temperature effect

    // Ricker with covariates + immigration + intake conversion
    Type A_curr = A_prev * exp(r_eff - densA * A_prev) + lambda_imm * L_prev + eta_repro * (I_prev * A_prev);

    // Store next states
    F_frac(t) = F_curr;
    S_frac(t) = S_curr;
    A_pred(t) = A_curr + eps; // strictly positive

    // Observation likelihood at time t
    nll -= dnorm(log(cots_dat(t) + eps), log(A_pred(t) + eps), sigma_cots, true);
    nll -= dnorm(fast_dat(t), F_frac(t) * Type(100.0), sigma_fast, true);
    nll -= dnorm(slow_dat(t), S_frac(t) * Type(100.0), sigma_slow, true);
  }

  // Add soft penalties
  nll += pen_w * pen_states;

  // -------------------- Reporting --------------------
  // Model predictions aligned to data (use required _pred suffix; corals reported in percent)
  vector<Type> fast_pred(T);
  vector<Type> slow_pred(T);
  for (int t = 0; t < T; ++t) {
    fast_pred(t) = F_frac(t) * Type(100.0); // % cover
    slow_pred(t) = S_frac(t) * Type(100.0); // % cover
  }
  REPORT(A_pred);        // internal COTS state (ind m^-2)
  REPORT(fast_pred);     // fast coral percent prediction
  REPORT(slow_pred);     // slow coral percent prediction
  REPORT(Year);          // report time for alignment
  REPORT(K);             // realized carrying capacity
  REPORT(alpha);         // realized preference
  REPORT(sigma_cots);    // realized obs SDs
  REPORT(sigma_fast);
  REPORT(sigma_slow);

  // Required names for ControlFile.R evaluation
  ADREPORT(A_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);

  return nll;
}
