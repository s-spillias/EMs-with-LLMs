#include <TMB.hpp>

// Smooth maximum approximation to avoid hard cutoffs (units: same as x)
template<class Type>
Type smooth_max(Type x, Type eps) {
  return Type(0.5) * (x + sqrt(x * x + eps)); // ~max(x,0) with smoothness controlled by eps
}

// Smooth minimum approximation (approximately min(x,y))
template<class Type>
Type smooth_min(Type x, Type y, Type eps) {
  Type diff = x - y;
  return Type(0.5) * (x + y - sqrt(diff * diff + eps));
}

// Logistic transform (dimensionless)
template<class Type>
Type inv_logit(Type x) {
  return Type(1) / (Type(1) + exp(-x));
}

// Logit transform with numerical safety
template<class Type>
Type safe_logit(Type p, Type eps) {
  p = CppAD::CondExpLt(p, eps, eps, p);
  p = CppAD::CondExpGt(p, Type(1) - eps, Type(1) - eps, p);
  return log(p / (Type(1) - p));
}

// Smooth bound penalty: zero inside [low, high], quadratic outside (units: penalty on NLL)
template<class Type>
Type penalty_bounds(Type x, Type low, Type high, Type lambda, Type eps) {
  Type below = smooth_max(low - x, eps);  // positive if x < low
  Type above = smooth_max(x - high, eps); // positive if x > high
  return lambda * (below * below + above * above);
}

/*
Numbered equation summary (annual time step, index t = 0..T-2):

1) Initial conditions (no data leakage):
   cots_pred(0) = cots_dat(0)
   fast_pred(0) = fast_dat(0)
   slow_pred(0) = slow_dat(0)

2) Temperature modifiers (Gaussian/bell-shaped performance):
   phi_T_COTS(t)  = exp(-0.5 * ((sst_dat(t) - Topt_cots)  / sigmaT_cots)^2)
   phi_T_CORAL(t) = exp(-0.5 * ((sst_dat(t) - Topt_coral) / sigmaT_coral)^2)

3) Fertilization success (pair-encounter, quadratic Allee-type effect):
   phi_spawn(t) = C_t^2 / (h_spawn^2 + C_t^2)

4) Food limitation for COTS survival (saturating on total coral cover):
   phi_food(t) = (A_t + S_t) / (foodK + A_t + S_t)

5) Selective predation per starfish (Type II/III with preference for Acropora):
   q = 1 + exp(log_q_FR)  // functional response exponent (>=1; q=1 Type II, q>1 Type III)
   wA = inv_logit(prefA_logit); wS = 1 - wA
   consA_per(t) = max_cons * wA * A_t^q / (hA + A_t^q)
   consS_per(t) = max_cons * wS * S_t^q / (hS + S_t^q)
   predA_eff(t) = A_t * [1 - exp(-C_t * consA_per(t) / (A_t + eps))] // smooth cap by availability
   predS_eff(t) = S_t * [1 - exp(-C_t * consS_per(t) / (S_t + eps))]

6) Coral growth (space-limited logistic with temperature modifier and background mortality):
   F_t = max(0, 100 - A_t - S_t) [implemented smoothly]
   growthA(t) = rA * A_t * (F_t / 100) * phi_T_CORAL(t)
   growthS(t) = rS * S_t * (F_t / 100) * phi_T_CORAL(t)

7) Adult COTS dynamics (survival + recruitment + immigration, with Beverton–Holt crowding):
   survC(t)    = C_t * exp( -[mC + mC_food * (1 - phi_food(t))] )
   recruits(t) = fec * C_t * phi_spawn(t) * phi_T_COTS(t)
   imm(t)      = alpha_imm * cotsimm_dat(t) / (k_imm + cotsimm_dat(t))
   C_tot(t)    = survC(t) + recruits(t) + imm(t)
   C_{t+1}     = C_tot(t) / (1 + beta_dd * C_tot(t))
*/

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(cots_dat);     // adult COTS density (ind m^-2)
  DATA_VECTOR(fast_dat);     // fast coral cover (%)
  DATA_VECTOR(slow_dat);     // slow coral cover (%)
  DATA_VECTOR(sst_dat);      // sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);  // larval immigration index (ind m^-2 yr^-1 or scaled)

  // Parameters
  PARAMETER(fec);
  PARAMETER(h_spawn);
  PARAMETER(mC);
  PARAMETER(mC_food);
  PARAMETER(alpha_imm);
  PARAMETER(k_imm);
  PARAMETER(Topt_cots);
  PARAMETER(sigmaT_cots);
  PARAMETER(rA);
  PARAMETER(rS);
  PARAMETER(hA);
  PARAMETER(hS);
  PARAMETER(max_cons);
  PARAMETER(Topt_coral);
  PARAMETER(sigmaT_coral);
  PARAMETER(mA0);
  PARAMETER(mS0);
  PARAMETER(foodK);
  PARAMETER(beta_dd);
  PARAMETER(prefA_logit);
  PARAMETER(log_q_FR);

  // Observation model parameters (on log-scale)
  PARAMETER(log_sigma_cots);
  PARAMETER(log_sigma_fast);
  PARAMETER(log_sigma_slow);

  Type nll = 0.0;

  int T = cots_dat.size();
  // Basic length checks (soft: rely on data consistency; hard checks could be added if desired)

  // Sigmas
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_fast = exp(log_sigma_fast);
  Type sigma_slow = exp(log_sigma_slow);

  // Functional response exponent and prey preference
  Type q = Type(1) + exp(log_q_FR);
  Type wA = inv_logit(prefA_logit);
  Type wS = Type(1) - wA;

  // Numerics
  const Type eps = Type(1e-10);
  const Type eps_pos = Type(1e-8);

  // State predictions
  vector<Type> cots_pred(T);
  vector<Type> fast_pred(T);
  vector<Type> slow_pred(T);

  // Helper to detect NA/NaN from data (NaN != NaN)
  auto notNA = [](Type x) -> bool { return x == x; };

  // Initial conditions from data at t=0 (no leakage beyond initialization)
  cots_pred(0) = notNA(cots_dat(0)) ? cots_dat(0) : Type(0.1);
  fast_pred(0) = notNA(fast_dat(0)) ? fast_dat(0) : Type(10.0);
  slow_pred(0) = notNA(slow_dat(0)) ? slow_dat(0) : Type(10.0);

  for (int t = 0; t < T - 1; ++t) {
    Type C = smooth_max(cots_pred(t), eps_pos);
    Type A = smooth_max(fast_pred(t), eps_pos);
    Type S = smooth_max(slow_pred(t), eps_pos);

    // 2) Temperature modifiers
    Type dT_cots  = (sst_dat(t) - Topt_cots) / (sigmaT_cots + eps_pos);
    Type dT_coral = (sst_dat(t) - Topt_coral) / (sigmaT_coral + eps_pos);
    Type phi_T_COTS  = exp(Type(-0.5) * dT_cots * dT_cots);
    Type phi_T_CORAL = exp(Type(-0.5) * dT_coral * dT_coral);

    // 3) Fertilization success (quadratic depensation)
    Type C2 = C * C;
    Type h2 = h_spawn * h_spawn;
    Type phi_spawn = C2 / (h2 + C2 + eps_pos);

    // 4) Food limitation for COTS survival
    Type totCoral = A + S;
    Type phi_food = totCoral / (foodK + totCoral + eps_pos);

    // 5) Selective predation per starfish
    Type Aq = pow(A + eps_pos, q);
    Type Sq = pow(S + eps_pos, q);
    Type consA_per = max_cons * wA * Aq / (hA + Aq + eps_pos); // % cover per starfish per yr allocated to A
    Type consS_per = max_cons * wS * Sq / (hS + Sq + eps_pos); // % cover per starfish per yr allocated to S

    // Smooth cap by availability (cannot exceed available cover)
    Type predA_eff = A * (Type(1) - exp(-C * consA_per / (A + eps_pos)));
    Type predS_eff = S * (Type(1) - exp(-C * consS_per / (S + eps_pos)));

    // 6) Coral growth and background mortality
    Type F = smooth_max(Type(100.0) - A - S, eps_pos); // free space in %
    Type growthA = rA * A * (F / Type(100.0)) * phi_T_CORAL;
    Type growthS = rS * S * (F / Type(100.0)) * phi_T_CORAL;
    Type mortA_bg = mA0 * A;
    Type mortS_bg = mS0 * S;

    // Update coral states; enforce [0,100] with smooth min/max
    Type A_next_lin = A + growthA - mortA_bg - predA_eff;
    Type S_next_lin = S + growthS - mortS_bg - predS_eff;

    Type A_next_nonneg = smooth_max(A_next_lin, eps_pos);
    Type S_next_nonneg = smooth_max(S_next_lin, eps_pos);

    Type A_next = smooth_min(A_next_nonneg, Type(100.0), eps_pos);
    Type S_next = smooth_min(S_next_nonneg, Type(100.0), eps_pos);

    // 7) Adult COTS dynamics
    Type mort_rate_C = mC + mC_food * (Type(1) - phi_food);
    Type survC = C * exp(-mort_rate_C);

    Type recruits = fec * C * phi_spawn * phi_T_COTS;

    Type imm_input = cotsimm_dat(t);
    Type immigration = alpha_imm * imm_input / (k_imm + imm_input + eps_pos);

    Type C_tot = survC + recruits + immigration;
    Type C_next = C_tot / (Type(1) + beta_dd * C_tot);

    // Enforce non-negativity
    C_next = smooth_max(C_next, eps_pos);

    // Assign next-step predictions
    fast_pred(t + 1) = A_next;
    slow_pred(t + 1) = S_next;
    cots_pred(t + 1) = C_next;
  }

  // Observation likelihood
  for (int t = 0; t < T; ++t) {
    // COTS: lognormal on density
    if (notNA(cots_dat(t))) {
      Type obs_log = log(cots_dat(t) + eps_pos);
      Type pred_log = log(cots_pred(t) + eps_pos);
      nll -= dnorm(obs_log, pred_log, sigma_cots, true);
    }
    // Coral: logit-normal on proportion of total area (0-1)
    if (notNA(fast_dat(t))) {
      Type p_obs = (fast_dat(t) / Type(100.0));
      Type p_pred = (fast_pred(t) / Type(100.0));
      Type z_obs = safe_logit(p_obs, Type(1e-6));
      Type z_pred = safe_logit(p_pred, Type(1e-6));
      nll -= dnorm(z_obs, z_pred, sigma_fast, true);
    }
    if (notNA(slow_dat(t))) {
      Type p_obs = (slow_dat(t) / Type(100.0));
      Type p_pred = (slow_pred(t) / Type(100.0));
      Type z_obs = safe_logit(p_obs, Type(1e-6));
      Type z_pred = safe_logit(p_pred, Type(1e-6));
      nll -= dnorm(z_obs, z_pred, sigma_slow, true);
    }
  }

  // Optional: mild penalties to keep percentages within [0,100] (should already be enforced)
  Type lambda_bounds = Type(0.0); // set >0 to activate
  for (int t = 0; t < T; ++t) {
    nll += lambda_bounds * penalty_bounds(fast_pred(t), Type(0.0), Type(100.0), Type(1.0), Type(1e-8));
    nll += lambda_bounds * penalty_bounds(slow_pred(t), Type(0.0), Type(100.0), Type(1.0), Type(1e-8));
  }

  // Reports
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(q);
  REPORT(wA);
  REPORT(wS);
  REPORT(sigma_cots);
  REPORT(sigma_fast);
  REPORT(sigma_slow);

  return nll;
}
