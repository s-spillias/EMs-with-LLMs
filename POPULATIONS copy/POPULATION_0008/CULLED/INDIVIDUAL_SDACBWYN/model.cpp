#include <TMB.hpp>

// Template Model Builder model for COTS boom-bust dynamics and coral impacts
// Uses forcing: Year, sst_dat, cotsimm_dat
// Predicts: cots_pred (indiv m^-2), fast_pred (% cover), slow_pred (% cover)
// Observations: cots_dat, fast_dat, slow_dat (matched names, lognormal likelihood)

// Helper functions with small constants for stability
template<class Type>
Type inv_logit(Type x) { // Smooth logistic function
  return Type(1.0) / (Type(1.0) + exp(-x));
}

template<class Type>
Type square(Type x) { return x * x; }

template<class Type>
Type max_floor(Type x, Type m) { return CppAD::CondExpGt(x, m, x, m); } // Smooth enough floor via conditional

template<class Type>
Type min_ceiling(Type x, Type M) { return CppAD::CondExpLt(x, M, x, M); } // Smooth enough cap via conditional

// TMB objective function
template<class Type>
Type objective_function<Type>::operator() ()
{
  // ----------------------------
  // DATA
  // ----------------------------
  DATA_VECTOR(Year);        // Year vector (calendar year), used to align time steps
  DATA_VECTOR(sst_dat);     // Sea-surface temperature (C)
  DATA_VECTOR(cotsimm_dat); // Larval immigration (indiv m^-2 yr^-1)
  DATA_VECTOR(cots_dat);    // Observed adult COTS density (indiv m^-2)
  DATA_VECTOR(fast_dat);    // Observed fast coral cover (%)
  DATA_VECTOR(slow_dat);    // Observed slow coral cover (%)

  int n = Year.size();
  Type eps = Type(1e-9);

  // ----------------------------
  // PARAMETERS
  // ----------------------------
  PARAMETER(r_fast);
  PARAMETER(r_slow);
  PARAMETER(K_coral);
  PARAMETER(g_max);
  PARAMETER(K_prey);
  PARAMETER(K_graze);     // New: half-saturation for grazing functional response
  PARAMETER(pref_fast);
  PARAMETER(pref_slow);
  PARAMETER(theta_graze); // New: shape exponent for Type-III grazing
  PARAMETER(s0_cots);
  PARAMETER(theta_surv);
  PARAMETER(theta_recruit);
  PARAMETER(m_heat_cots);
  PARAMETER(r0_recruit);
  PARAMETER(alpha_imm);
  PARAMETER(kc_carry);
  PARAMETER(Topt_cots);
  PARAMETER(sigmaT_cots);
  PARAMETER(Topt_coral);
  PARAMETER(sigmaT_coral);
  PARAMETER(T_bleach);
  PARAMETER(k_bleach);
  PARAMETER(m_bleach_fast);
  PARAMETER(m_bleach_slow);
  PARAMETER(k_allee);
  PARAMETER(c50_allee);
  PARAMETER(sd_log_cots);
  PARAMETER(sd_log_fast);
  PARAMETER(sd_log_slow);

  // ----------------------------
  // STATE VECTORS (predictions)
  // ----------------------------
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Initialize states at t=0 from observations (no leakage beyond initialization)
  cots_pred(0) = max_floor(cots_dat(0), eps);
  fast_pred(0) = max_floor(fast_dat(0), eps);
  slow_pred(0) = max_floor(slow_dat(0), eps);

  // ----------------------------
  // PROCESS MODEL
  // ----------------------------
  for (int t = 1; t < n; t++) {
    // Previous states
    Type cots_prev = cots_pred(t - 1);
    Type fast_prev = fast_pred(t - 1);
    Type slow_prev = slow_pred(t - 1);

    // Environmental forcing at t-1 (avoid leakage of current observations)
    Type sst_prev = sst_dat(t - 1);

    // Temperature modulators
    Type temp_repro = exp(-Type(0.5) * square((sst_prev - Topt_cots) / max_floor(sigmaT_cots, Type(1e-6))));
    Type temp_coral = exp(-Type(0.5) * square((sst_prev - Topt_coral) / max_floor(sigmaT_coral, Type(1e-6))));

    // Bleaching stress (logistic in temperature)
    Type bleach_level = inv_logit(k_bleach * (sst_prev - T_bleach));

    // Prey availability index (preference-weighted coral cover)
    Type avail = pref_fast * fast_prev + pref_slow * slow_prev;
    avail = max_floor(avail, Type(0.0));

    // Saturating prey limitation fraction (used for COTS processes)
    Type prey_frac = avail / (K_prey + avail + eps);

    // ----------------------------
    // COTS dynamics
    // ----------------------------

    // Adult survival (prey- and heat-modulated)
    Type surv_mult = s0_cots * pow(prey_frac, max_floor(theta_surv, Type(0.0))) * exp(-m_heat_cots * bleach_level);
    // Clamp survival multiplier to [0, 1] for numerical stability
    surv_mult = min_ceiling(max_floor(surv_mult, Type(0.0)), Type(1.0));
    Type survivors = cots_prev * surv_mult;

    // Allee effect on recruitment (smooth, centered at c50_allee)
    Type allee = inv_logit(k_allee * (cots_prev - c50_allee));

    // Effective carrying capacity for recruitment saturation (linked to prey)
    Type K_eff = kc_carry * avail;
    K_eff = max_floor(K_eff, eps);

    // Type-III prey limitation for recruitment and settlement
    Type prey_lim_rec = pow(prey_frac, max_floor(theta_recruit, Type(0.0)));

    // Local recruitment to adult class (fecundity * juvenile survival to adult)
    Type recruit_local_raw = r0_recruit * cots_prev * temp_repro * prey_lim_rec;

    // Immigration realized settlement (conditioned on local prey)
    Type imm_settle_raw = alpha_imm * cotsimm_dat(t - 1) * prey_lim_rec;

    // Apply Allee to realized recruits (mating success etc.)
    Type recruits_raw = (recruit_local_raw + imm_settle_raw) * allee;

    // Beverton–Ricker style density regulation based on effective carrying capacity
    Type dens_mult = exp(-cots_prev / K_eff);
    Type recruits = recruits_raw * dens_mult;

    // Next-step COTS
    Type cots_next = survivors + recruits;
    cots_next = max_floor(cots_next, eps);
    cots_pred(t) = cots_next;

    // ----------------------------
    // Coral dynamics
    // ----------------------------

    // Space limitation factor shared by corals
    Type total_coral_prev = fast_prev + slow_prev;
    total_coral_prev = max_floor(total_coral_prev, Type(0.0));
    Type free_space = max_floor(K_coral - total_coral_prev, Type(0.0));
    Type space_frac = free_space / max_floor(K_coral, eps);

    // Per-capita grazing pressure: Type-III functional response on total weighted prey (decoupled from K_prey)
    Type theta_g = max_floor(theta_graze, Type(1.0));
    Type avail_pos = max_floor(avail, eps);
    Type avail_theta = pow(avail_pos, theta_g);
    Type Kg_theta = pow(max_floor(K_graze, eps), theta_g);
    Type percap_graz = g_max * (avail_theta / (Kg_theta + avail_theta));

    // Diet allocation across coral groups by weighted availability
    Type share_fast = (pref_fast * fast_prev) / (avail + eps);
    Type share_slow = (pref_slow * slow_prev) / (avail + eps);

    // Grazing losses
    Type graze_fast = cots_prev * percap_graz * share_fast;
    Type graze_slow = cots_prev * percap_graz * share_slow;

    // Bleaching mortality
    Type bleach_fast = m_bleach_fast * bleach_level * fast_prev;
    Type bleach_slow = m_bleach_slow * bleach_level * slow_prev;

    // Growth (temperature-modulated, space-limited)
    Type growth_fast = r_fast * fast_prev * space_frac * temp_coral;
    Type growth_slow = r_slow * slow_prev * space_frac * temp_coral;

    // Update corals
    Type fast_next = fast_prev + growth_fast - graze_fast - bleach_fast;
    Type slow_next = slow_prev + growth_slow - graze_slow - bleach_slow;

    // Enforce bounds [0, K_coral] and avoid NaN
    fast_next = min_ceiling(max_floor(fast_next, Type(0.0)), max_floor(K_coral, eps));
    slow_next = min_ceiling(max_floor(slow_next, Type(0.0)), max_floor(K_coral, eps));

    fast_pred(t) = fast_next;
    slow_pred(t) = slow_next;
  }

  // ----------------------------
  // LIKELIHOOD (lognormal errors)
  // ----------------------------
  Type nll = 0.0;
  Type sd_cots = max_floor(sd_log_cots, Type(1e-6));
  Type sd_fast = max_floor(sd_log_fast, Type(1e-6));
  Type sd_slow = max_floor(sd_log_slow, Type(1e-6));

  for (int t = 0; t < n; t++) {
    // COTS
    Type obs_cots = max_floor(cots_dat(t), eps);
    Type pred_cots = max_floor(cots_pred(t), eps);
    nll -= dnorm(log(obs_cots), log(pred_cots), sd_cots, true);

    // Fast coral
    Type obs_fast = max_floor(fast_dat(t), eps);
    Type pred_fast = max_floor(fast_pred(t), eps);
    nll -= dnorm(log(obs_fast), log(pred_fast), sd_fast, true);

    // Slow coral
    Type obs_slow = max_floor(slow_dat(t), eps);
    Type pred_slow = max_floor(slow_pred(t), eps);
    nll -= dnorm(log(obs_slow), log(pred_slow), sd_slow, true);
  }

  // ----------------------------
  // REPORTS
  // ----------------------------
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);

  return nll;
}
