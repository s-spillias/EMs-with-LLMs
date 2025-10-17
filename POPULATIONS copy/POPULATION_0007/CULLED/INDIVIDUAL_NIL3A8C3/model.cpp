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

  // ------------------------
  // PARAMETERS
  // ------------------------
  // COTS demography and recruitment
  PARAMETER(alpha_rec);
  PARAMETER(phi);
  PARAMETER(k_allee);
  PARAMETER(C_allee);
  PARAMETER(K_R);
  PARAMETER(wF);
  PARAMETER(wS);
  PARAMETER(muC);
  PARAMETER(gammaC);
  PARAMETER(mu_starveC);
  PARAMETER(mJ);
  PARAMETER(muJ);

  // Temperature effects
  PARAMETER(T_opt_rec);
  PARAMETER(beta_rec);
  PARAMETER(T_opt_bleach);
  PARAMETER(beta_bleach);

  // Coral dynamics and predation
  PARAMETER(m_bleachF);
  PARAMETER(m_bleachS);
  PARAMETER(rF);
  PARAMETER(rS);
  PARAMETER(K_tot);
  PARAMETER(aF);
  PARAMETER(aS);
  PARAMETER(etaF);
  PARAMETER(etaS);
  PARAMETER(h);
  PARAMETER(qF);
  PARAMETER(qS);

  // Observation error SDs (directly parameterized; keep positive via bounds/penalty)
  PARAMETER(sigma_cots);
  PARAMETER(sigma_fast);
  PARAMETER(sigma_slow);

  // Initial states (avoid using observed t=0 values in process)
  PARAMETER(log_C0);
  PARAMETER(log_J0);
  PARAMETER(F0_raw); // unconstrained -> (0,100) via inverse-logit
  PARAMETER(S0_raw); // unconstrained -> (0,100) via inverse-logit

  // ------------------------
  // SETUP
  // ------------------------
  Type nll = Type(0.0);
  const int n = Year.size();
  const Type eps = Type(1e-8);

  // Length checks (soft penalty if mismatch)
  if ((cots_dat.size() != n) ||
      (fast_dat.size() != n) ||
      (slow_dat.size() != n) ||
      (sst_dat.size()  != n) ||
      (cotsimm_dat.size() != n)) {
    nll += Type(1e6); // large penalty to signal mismatch without throwing
  }

  // Transform initial state parameters
  Type C0 = exp(log_C0);
  Type J0 = exp(log_J0);
  Type F0 = Type(100.0) / (Type(1.0) + exp(-F0_raw));
  Type S0 = Type(100.0) / (Type(1.0) + exp(-S0_raw));

  // State vectors (predicted trajectories)
  vector<Type> cots_pred(n); // adult COTS (ind/m^2)
  vector<Type> Jpred(n);     // juvenile pool (arbitrary density)
  vector<Type> fast_pred(n); // fast coral cover (%)
  vector<Type> slow_pred(n); // slow coral cover (%)

  // Initialize states at t=0 from parameters (not from observations)
  if (n > 0) {
    cots_pred(0) = C0;
    Jpred(0)     = J0;
    fast_pred(0) = F0;
    slow_pred(0) = S0;
  }

  // ------------------------
  // PROCESS MODEL (no data leakage; uses only previous-step states)
  // ------------------------
  for (int t = 1; t < n; ++t) {
    // Previous states (ensure non-negative in calculations)
    Type C_prev = pospart(cots_pred(t - 1));
    Type J_prev = pospart(Jpred(t - 1));
    Type F_prev = pospart(fast_pred(t - 1));
    Type S_prev = pospart(slow_pred(t - 1));

    // Coral resource index and food availability
    Type R = wF * F_prev + wS * S_prev;                   // % cover-weighted resource
    Type f_food = R / (K_R + R + eps);                    // in [0,1]

    // Recruitment to juvenile pool
    Type Allee = Type(1.0) / (Type(1.0) + exp(-k_allee * (C_prev - C_allee)));
    Type temp_rec = exp(-beta_rec * pow(sst_dat(t - 1) - T_opt_rec, 2));
    Type recruits = alpha_rec * pow(C_prev + eps, phi) * Allee * f_food * temp_rec
                    + pospart(cotsimm_dat(t - 1));        // additive larval immigration

    // Juveniles: maturation and mortality
    Type matured = mJ * J_prev;
    Type mortJ = muJ * J_prev;
    Type J_next = pospart(J_prev + recruits - matured - mortJ);
    Jpred(t) = J_next;

    // Adults: baseline + density + starvation mortality
    Type starve = mu_starveC * (Type(1.0) - f_food);      // increases as food declines
    Type mortA = (muC + gammaC * C_prev + starve) * C_prev;
    Type C_next = pospart(C_prev + matured - mortA);
    cots_pred(t) = C_next;

    // Predation functional response on corals (multi-prey, saturating)
    Type AF = aF * pow(F_prev + eps, etaF);
    Type AS = aS * pow(S_prev + eps, etaS);
    Type A_sum = AF + AS;
    Type per_pred = A_sum / (Type(1.0) + h * A_sum);      // per-predator consumption rate
    Type shareF = AF / (A_sum + eps);
    Type shareS = AS / (A_sum + eps);
    Type total_consump = per_pred * C_prev;               // scaled by predator density

    Type pred_loss_F = qF * shareF * total_consump;       // % cover per year
    Type pred_loss_S = qS * shareS * total_consump;

    // Coral growth with shared space limit and bleaching losses
    Type temp_excess = pospart(sst_dat(t - 1) - T_opt_bleach);
    Type growth_mult = exp(-beta_bleach * temp_excess);   // reduces growth under heat stress

    Type space_frac = (F_prev + S_prev) / (K_tot + eps);
    Type one_minus_space = Type(1.0) - space_frac;

    Type gF = rF * F_prev * one_minus_space * growth_mult;
    Type gS = rS * S_prev * one_minus_space * growth_mult;

    Type bleachF = m_bleachF * temp_excess * F_prev;
    Type bleachS = m_bleachS * temp_excess * S_prev;

    Type F_next = pospart(F_prev + gF - pred_loss_F - bleachF);
    Type S_next = pospart(S_prev + gS - pred_loss_S - bleachS);

    fast_pred(t) = F_next;
    slow_pred(t) = S_next;
  }

  // ------------------------
  // PRIORS / PENALTIES (soft bounds following parameters.json guidance)
  // ------------------------
  Type pen_w = Type(1.0);
  nll += range_penalty(alpha_rec,    Type(0.0),   Type(10.0), pen_w);
  nll += range_penalty(phi,          Type(1.0),   Type(3.0),  pen_w);
  nll += range_penalty(k_allee,      Type(0.01),  Type(20.0), pen_w);
  nll += range_penalty(C_allee,      Type(0.0),   Type(5.0),  pen_w);
  nll += range_penalty(K_R,          Type(1.0),   Type(100.0),pen_w);
  nll += range_penalty(wF,           Type(0.0),   Type(2.0),  pen_w);
  nll += range_penalty(wS,           Type(0.0),   Type(2.0),  pen_w);
  nll += range_penalty(muC,          Type(0.0),   Type(3.0),  pen_w);
  nll += range_penalty(gammaC,       Type(0.0),   Type(10.0), pen_w);
  nll += range_penalty(mu_starveC,   Type(0.0),   Type(3.0),  pen_w);
  nll += range_penalty(mJ,           Type(0.0),   Type(1.0),  pen_w);
  nll += range_penalty(muJ,          Type(0.0),   Type(1.0),  pen_w);
  nll += range_penalty(T_opt_rec,    Type(20.0),  Type(34.0), pen_w);
  nll += range_penalty(beta_rec,     Type(0.0),   Type(2.0),  pen_w);
  nll += range_penalty(T_opt_bleach, Type(31.74), Type(34.3), pen_w);
  nll += range_penalty(beta_bleach,  Type(0.0),   Type(5.0),  pen_w);
  nll += range_penalty(m_bleachF,    Type(0.0),   Type(2.0),  pen_w);
  nll += range_penalty(m_bleachS,    Type(0.0),   Type(2.0),  pen_w);
  nll += range_penalty(rF,           Type(0.0),   Type(2.0),  pen_w);
  nll += range_penalty(rS,           Type(0.0),   Type(2.0),  pen_w);
  nll += range_penalty(K_tot,        Type(10.0),  Type(100.0),pen_w);
  nll += range_penalty(aF,           Type(0.0),   Type(1.0),  pen_w);
  nll += range_penalty(aS,           Type(0.0),   Type(1.0),  pen_w);
  nll += range_penalty(etaF,         Type(1.0),   Type(3.0),  pen_w);
  nll += range_penalty(etaS,         Type(1.0),   Type(3.0),  pen_w);
  nll += range_penalty(h,            Type(0.0),   Type(1.0),  pen_w);
  nll += range_penalty(qF,           Type(0.0),   Type(1.0),  pen_w);
  nll += range_penalty(qS,           Type(0.0),   Type(1.0),  pen_w);
  nll += range_penalty(sigma_cots,   Type(0.01),  Type(2.0),  pen_w);
  nll += range_penalty(sigma_fast,   Type(0.01),  Type(2.0),  pen_w);
  nll += range_penalty(sigma_slow,   Type(0.01),  Type(2.0),  pen_w);

  // Non-negativity for SDs in likelihood
  Type sd_cots = pospart(sigma_cots) + Type(1e-8);
  Type sd_fast = pospart(sigma_fast) + Type(1e-8);
  Type sd_slow = pospart(sigma_slow) + Type(1e-8);

  // ------------------------
  // OBSERVATION MODEL
  // ------------------------
  for (int t = 0; t < n; ++t) {
    // Adults: lognormal
    Type log_obs_c = log(cots_dat(t) + eps);
    Type log_pred_c = log(pospart(cots_pred(t)) + eps);
    nll -= dnorm(log_obs_c, log_pred_c, sd_cots, true);

    // Corals: logit-normal on % cover
    Type logit_obs_f = logit_pct(fast_dat(t));
    Type logit_pred_f = logit_pct(pospart(fast_pred(t)));
    nll -= dnorm(logit_obs_f, logit_pred_f, sd_fast, true);

    Type logit_obs_s = logit_pct(slow_dat(t));
    Type logit_pred_s = logit_pct(pospart(slow_pred(t)));
    nll -= dnorm(logit_obs_s, logit_pred_s, sd_slow, true);
  }

  // ------------------------
  // REPORTS
  // ------------------------
  REPORT(cots_pred);
  REPORT(Jpred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);

  return nll;
}
