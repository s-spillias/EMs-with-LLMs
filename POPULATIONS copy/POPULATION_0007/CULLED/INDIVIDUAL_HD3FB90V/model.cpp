#include <TMB.hpp>

// Helper: softplus for smooth non-negativity and smooth penalties
template<class Type>
Type softplus(const Type &x) {
  return log(Type(1.0) + exp(x)); // AD-safe smooth approx to max(0,x)
}

// Local smooth positivity function (AD-safe), similar to ADMB/TMB posfun
template<class Type>
Type posfun_smooth(const Type &x, const Type &eps, Type &pen) {
  // If x >= eps, return x; else return a smooth function ensuring positivity and accumulate penalty
  if (x >= eps) return x; // no penalty needed
  Type x_eps = x / (eps + Type(1e-12)); // scaled x to avoid division by zero
  Type res = eps / (Type(2.0) - x_eps); // smooth transition maintaining continuity
  // Penalty term increases as x falls below eps; small weight keeps it soft
  pen += Type(0.01) * (Type(2.0) - x_eps - (eps / (x + Type(1e-12))));
  return res;
}

/*
Equations (annual time step; t indexes Year):
Given states at t-1: N = cots_pred(t-1) [ind m^-2], F = fast_pred(t-1) [%], S = slow_pred(t-1) [%],
and exogenous drivers at t-1: T = sst_dat(t-1) [°C], I = cotsimm_dat(t-1) [ind m^-2 yr^-1].

1) Functional responses (Type III) for each coral group:
   f_F = F^q / (F^q + H_F^q + eps)
   f_S = S^q / (S^q + H_S^q + eps)
   feed_level = w * f_F + (1-w) * f_S
   where w in (0,1) is preference for fast coral, q>=1 shape, H_* half-saturation.

2) Per-predator consumption rates (in % cover per predator per year):
   cons_F = a_max * w * f_F
   cons_S = a_max * (1-w) * f_S

3) Coral predation losses (smooth cap so losses <= available cover):
   loss_F = F * (1 - exp( - (eps_pred_F * N * cons_F) / (F + eps) ))
   loss_S = S * (1 - exp( - (eps_pred_S * N * cons_S) / (S + eps) ))

4) Coral intrinsic growth with space limitation and interspecific competition:
   dF_grow = r_F * F * ( 1 - (F + alpha_FS * S) / (K_F + eps) )
   dS_grow = r_S * S * (  1 - (S + alpha_SF * F) / (K_S + eps) )

5) Temperature-driven bleaching mortality (smooth logistic around T_bleach):
   ble_mult = 1 / (1 + exp( -s_bleach * (T - T_bleach) ))
   dF_bleach = m_bleach_F_max * ble_mult * F
   dS_bleach = m_bleach_S_max * ble_mult * S

6) Coral updates (dt=1 year):
   F_t = posfun_smooth( F + dF_grow - loss_F - dF_bleach, eps, pen )
   S_t = posfun_smooth( S + dS_grow - loss_S - dS_bleach, eps, pen )

7) COTS reproduction modified by food, temperature, density, and mate-finding (Allee) effect:
   temp_mult = 1 + beta_temp_cots * tanh( (T - T_ref_cots) / T_scale_cots )
   dens_mod  = 1 / (1 + N / (K_cots + eps))         // smooth density limitation
   mate_success = N^h / (N^h + N_crit^h + eps)      // broadcast spawner Allee effect (h>=1)
   dN_birth  = epsN * b_max * temp_mult * feed_level * dens_mod * mate_success * N

8) COTS mortality with baseline, density, and starvation components:
   m_starv   = m_starv_max * (1 - feed_level)       // more starvation when little food
   dN_mort   = (m0 + mD * N + m_starv) * N

9) Larval immigration forcing:
   dN_imm    = k_imm * I

10) COTS update:
   N_t = posfun_smooth( N + dN_birth - dN_mort + dN_imm, eps, pen )

Observation model (for each t including t=0 to use all observations):
- cots_dat ~ LogNormal( log(cots_pred), sigma_cots )
- fast_dat ~ LogNormal( log(fast_pred), sigma_fast )
- slow_dat ~ LogNormal( log(slow_pred), sigma_slow )
with sigma_x = sqrt( exp(2*log_sigma_x) + min_sd^2 ) to avoid too-small SDs.
*/

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;

  // -----------------------------
  // DATA INPUTS
  // -----------------------------
  DATA_VECTOR(Year);          // Time index (years), used for sizing and reporting
  DATA_VECTOR(cots_dat);      // Observed adult COTS abundance (individuals/m^2)
  DATA_VECTOR(fast_dat);      // Observed fast coral cover (%)
  DATA_VECTOR(slow_dat);      // Observed slow coral cover (%)
  DATA_VECTOR(sst_dat);       // Observed sea-surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);   // Observed larval immigration (individuals/m^2/year)

  // -----------------------------
  // PARAMETERS (estimation scale)
  // -----------------------------
  PARAMETER(log_b_max);          // Log maximum per-capita COTS birth rate (year^-1), initial from literature compilations
  PARAMETER(epsN_raw);           // Raw logit for conversion efficiency (dimensionless), maps to (0,1)
  PARAMETER(log_a_max);          // Log maximum per-predator consumption rate (% cover predator^-1 year^-1)
  PARAMETER(pref_fast_raw);      // Raw logit for preference for fast coral (dimensionless), maps to (0,1)
  PARAMETER(log_H_f);            // Log half-saturation for fast coral (% cover)
  PARAMETER(log_H_s);            // Log half-saturation for slow coral (% cover)
  PARAMETER(log_m0);             // Log baseline COTS mortality rate (year^-1)
  PARAMETER(log_mD);             // Log density-dependent mortality coefficient ((ind/m^2)^-1 year^-1)
  PARAMETER(log_m_starv);        // Log maximum starvation mortality rate (year^-1)
  PARAMETER(log_K_cots);         // Log density scale for COTS density limitation (ind/m^2)
  PARAMETER(log_k_imm);          // Log scaling from immigration rate to adult additions (dimensionless)
  PARAMETER(T_ref_cots);         // Reference temperature for COTS reproduction center (°C)
  PARAMETER(beta_temp_cots_raw); // Raw logit amplitude of temperature effect on reproduction (dimensionless), maps to (0,1)
  PARAMETER(log_T_scale_cots);   // Log temperature scale (°C) setting smoothness of temp effect
  PARAMETER(T_bleach);           // Bleaching midpoint temperature (°C)
  PARAMETER(log_slope_bleach);   // Log slope of bleaching logistic (°C^-1)
  PARAMETER(log_m_bleach_f_max); // Log maximum bleaching mortality rate for fast coral (year^-1)
  PARAMETER(log_m_bleach_s_max); // Log maximum bleaching mortality rate for slow coral (year^-1)
  PARAMETER(log_r_f);            // Log intrinsic growth rate of fast coral (year^-1)
  PARAMETER(log_r_s);            // Log intrinsic growth rate of slow coral (year^-1)
  PARAMETER(K_f_raw);            // Raw logit for fast coral carrying capacity fraction of 100% cover (dimensionless)
  PARAMETER(K_s_raw);            // Raw logit for slow coral carrying capacity fraction of 100% cover (dimensionless)
  PARAMETER(alpha_fs_raw);       // Raw logit mapped to (0,2): competition of slow coral on fast coral (dimensionless)
  PARAMETER(alpha_sf_raw);       // Raw logit mapped to (0,2): competition of fast coral on slow coral (dimensionless)
  PARAMETER(eps_pred_f_raw);     // Raw logit efficiency of predation impact on fast coral (dimensionless), maps to (0,1)
  PARAMETER(eps_pred_s_raw);     // Raw logit efficiency of predation impact on slow coral (dimensionless), maps to (0,1)
  PARAMETER(log_q_plus);         // Log of (q-1) for Type III functional response shape (dimensionless)
  PARAMETER(log_N_crit);         // Log Allee threshold density for COTS mating success (ind/m^2)
  PARAMETER(log_h_allee_plus);   // Log of (h-1) for Allee mating success shape (dimensionless)
  PARAMETER(log_sigma_obs_cots); // Log observation SD for COTS (log-normal)
  PARAMETER(log_sigma_obs_fast); // Log observation SD for fast coral (log-normal)
  PARAMETER(log_sigma_obs_slow); // Log observation SD for slow coral (log-normal)

  // -----------------------------
  // TRANSFORMS AND CONSTANTS
  // -----------------------------
  int n = Year.size();                        // Number of time steps
  Type eps = Type(1e-8);                      // Small constant for numerical safety
  Type min_sd = Type(0.05);                   // Minimum SD to stabilize likelihood on log scale
  Type pen = Type(0.0);                       // Accumulator for smooth penalties from posfun_smooth
  Type lambda_pos = Type(10.0);               // Weight for positivity penalties (soft)

  // Positive parameters via exponentiation
  Type b_max = exp(log_b_max);                // Max per-capita birth rate (year^-1)
  Type a_max = exp(log_a_max);                // Max consumption per predator (% cover predator^-1 year^-1)
  Type H_f = exp(log_H_f);                    // Half-saturation cover (%) for fast coral
  Type H_s = exp(log_H_s);                    // Half-saturation cover (%) for slow coral
  Type m0 = exp(log_m0);                      // Baseline mortality (year^-1)
  Type mD = exp(log_mD);                      // Density-dependent mortality coefficient ((ind/m^2)^-1 year^-1)
  Type m_starv_max = exp(log_m_starv);        // Max starvation mortality (year^-1)
  Type K_cots = exp(log_K_cots);              // Density scale for COTS density limitation (ind/m^2)
  Type k_imm = exp(log_k_imm);                // Immigration scaling (dimensionless)
  Type T_scale_cots = exp(log_T_scale_cots);  // Temperature scale (°C)
  Type s_bleach = exp(log_slope_bleach);      // Bleaching slope (°C^-1)
  Type m_bleach_f_max = exp(log_m_bleach_f_max); // Max bleaching mortality fast coral (year^-1)
  Type m_bleach_s_max = exp(log_m_bleach_s_max); // Max bleaching mortality slow coral (year^-1)
  Type r_f = exp(log_r_f);                    // Fast coral intrinsic growth (year^-1)
  Type r_s = exp(log_r_s);                    // Slow coral intrinsic growth (year^-1)
  Type q = Type(1.0) + exp(log_q_plus);       // Type III shape parameter q >= 1
  Type N_crit = exp(log_N_crit);              // Allee threshold density (ind/m^2)
  Type h_allee = Type(1.0) + exp(log_h_allee_plus); // Allee shape h >= 1

  // (0,1) and bounded transforms (use TMB's invlogit)
  Type epsN = invlogit(epsN_raw);             // Conversion efficiency to new COTS (dimensionless, 0-1)
  Type w_fast = invlogit(pref_fast_raw);      // Preference for fast coral (0-1)
  Type K_f = Type(100.0) * invlogit(K_f_raw); // Fast coral carrying capacity (%) in (0,100)
  Type K_s = Type(100.0) * invlogit(K_s_raw); // Slow coral carrying capacity (%) in (0,100)
  Type alpha_fs = Type(2.0) * invlogit(alpha_fs_raw); // Competition coefficient slow->fast in (0,2)
  Type alpha_sf = Type(2.0) * invlogit(alpha_sf_raw); // Competition coefficient fast->slow in (0,2)
  Type eps_pred_f = invlogit(eps_pred_f_raw); // Predation impact efficiency on fast coral (0-1)
  Type eps_pred_s = invlogit(eps_pred_s_raw); // Predation impact efficiency on slow coral (0-1)
  Type beta_temp_cots = invlogit(beta_temp_cots_raw); // Amplitude of temp effect on COTS (0-1)

  // Observation SDs on log scale with floor (quadrature)
  Type sigma_cots = sqrt( exp(Type(2.0) * log_sigma_obs_cots) + min_sd * min_sd ); // Log-normal SD
  Type sigma_fast = sqrt( exp(Type(2.0) * log_sigma_obs_fast) + min_sd * min_sd );  // Log-normal SD
  Type sigma_slow = sqrt( exp(Type(2.0) * log_sigma_obs_slow) + min_sd * min_sd );  // Log-normal SD

  // -----------------------------
  // STATE VECTORS (predictions)
  // -----------------------------
  vector<Type> cots_pred(n); // Predicted COTS abundance (ind/m^2)
  vector<Type> fast_pred(n); // Predicted fast coral cover (%)
  vector<Type> slow_pred(n); // Predicted slow coral cover (%)

  // -----------------------------
  // INITIAL CONDITIONS (from data)
  // -----------------------------
  cots_pred(0) = cots_dat(0); // Initialize from observed data to avoid extra parameters
  fast_pred(0) = fast_dat(0); // Initialize from observed data
  slow_pred(0) = slow_dat(0); // Initialize from observed data

  // -----------------------------
  // NEGATIVE LOG-LIKELIHOOD
  // -----------------------------
  Type nll = Type(0.0); // Objective function value

  // Likelihood contribution for t = 0 (always include observations)
  nll -= dnorm( log(cots_dat(0) + eps), log(cots_pred(0) + eps), sigma_cots, true );
  nll -= dnorm( log(fast_dat(0) + eps), log(fast_pred(0) + eps), sigma_fast, true );
  nll -= dnorm( log(slow_dat(0) + eps), log(slow_pred(0) + eps), sigma_slow, true );

  // -----------------------------
  // PROCESS MODEL LOOP
  // -----------------------------
  for (int t = 1; t < n; ++t) {
    // States at previous step (no data leakage from current observations)
    Type N = cots_pred(t-1); // Previous COTS density (ind/m^2)
    Type F = fast_pred(t-1); // Previous fast coral cover (%)
    Type S = slow_pred(t-1); // Previous slow coral cover (%)

    // Exogenous drivers (use t-1 to represent conditions leading into this transition)
    Type T = sst_dat(t-1);      // Temperature (°C)
    Type I = cotsimm_dat(t-1);  // Immigration forcing (ind/m^2/year)

    // 1) Functional responses (Type III, smooth and bounded)
    Type Fq = pow(F + eps, q); // F^q with epsilon to prevent zero^q
    Type Sq = pow(S + eps, q); // S^q with epsilon
    Type f_F = Fq / (Fq + pow(H_f + eps, q) + eps); // Fast coral feeding index in (0,1)
    Type f_S = Sq / (Sq + pow(H_s + eps, q) + eps); // Slow coral feeding index in (0,1)
    Type feed_level = w_fast * f_F + (Type(1.0) - w_fast) * f_S; // Weighted prey availability (0,1)

    // 2) Per-predator consumption rates
    Type cons_F = a_max * w_fast * f_F;               // % cover pred^-1 yr^-1 on fast coral
    Type cons_S = a_max * (Type(1.0) - w_fast) * f_S; // % cover pred^-1 yr^-1 on slow coral

    // 3) Coral predation losses with smooth cap (<= available cover)
    Type demand_F = eps_pred_f * N * cons_F + eps; // Effective demand on fast coral
    Type demand_S = eps_pred_s * N * cons_S + eps; // Effective demand on slow coral
    Type loss_F = F * (Type(1.0) - exp( - demand_F / (F + eps) )); // Smoothly limited by F
    Type loss_S = S * (Type(1.0) - exp( - demand_S / (S + eps) )); // Smoothly limited by S

    // 4) Intrinsic coral growth with space limitation and interspecific competition
    Type comp_F = (F + alpha_fs * S) / (K_f + eps); // Crowding term for fast coral
    Type comp_S = (S + alpha_sf * F) / (K_s + eps); // Crowding term for slow coral
    Type dF_grow = r_f * F * (Type(1.0) - comp_F);  // Fast coral growth increment
    Type dS_grow = r_s * S * (Type(1.0) - comp_S);  // Slow coral growth increment

    // 5) Temperature-driven bleaching mortality (smooth logistic)
    Type ble_mult = Type(1.0) / (Type(1.0) + exp( - s_bleach * (T - T_bleach) )); // in (0,1)
    Type dF_bleach = m_bleach_f_max * ble_mult * F; // Fast coral bleaching loss
    Type dS_bleach = m_bleach_s_max * ble_mult * S; // Slow coral bleaching loss

    // 6) Coral updates with smooth positivity
    Type F_next = F + dF_grow - loss_F - dF_bleach;  // Proposed next fast coral cover
    Type S_next = S + dS_grow - loss_S - dS_bleach;  // Proposed next slow coral cover
    F_next = posfun_smooth(F_next, eps, pen);        // Smoothly enforce F_next >= eps
    S_next = posfun_smooth(S_next, eps, pen);        // Smoothly enforce S_next >= eps

    // 7) COTS reproduction modified by food, temperature, density, and Allee effect
    Type temp_mult = Type(1.0) + beta_temp_cots * tanh( (T - T_ref_cots) / (T_scale_cots + eps) ); // Smooth temperature effect
    Type dens_mod  = Type(1.0) / (Type(1.0) + N / (K_cots + eps)); // Smooth density limitation in (0,1)
    // Mate-finding Allee effect (broadcast spawning): success rises from ~0 at low N to ~1 above N_crit
    Type Npow = pow(N + eps, h_allee);
    Type mate_success = Npow / (Npow + pow(N_crit + eps, h_allee) + eps); // in (0,1)
    Type dN_birth  = epsN * b_max * temp_mult * feed_level * dens_mod * mate_success * N; // New COTS

    // 8) COTS mortality terms
    Type m_starv = m_starv_max * (Type(1.0) - feed_level);           // Starvation mortality rate
    Type dN_mort = (m0 + mD * N + m_starv) * N;                      // Total mortality

    // 9) Immigration forcing
    Type dN_imm = k_imm * I; // Added adults due to immigration (linear scaling)

    // 10) COTS update with smooth positivity
    Type N_next = N + dN_birth - dN_mort + dN_imm; // Proposed next COTS density
    N_next = posfun_smooth(N_next, eps, pen);      // Enforce N_next >= eps smoothly

    // Assign predictions
    fast_pred(t) = F_next; // Update fast coral prediction
    slow_pred(t) = S_next; // Update slow coral prediction
    cots_pred(t) = N_next; // Update COTS prediction

    // Observation likelihood at time t (include all observations)
    nll -= dnorm( log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots, true );
    nll -= dnorm( log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast, true );
    nll -= dnorm( log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow, true );
  }

  // -----------------------------
  // SOFT BOUND PENALTIES
  // -----------------------------
  // Encourage biologically reasonable parameter ranges without hard constraints
  auto bound_pen = [&](Type x, Type lo, Type hi, Type w)->Type{
    Type below = softplus(lo - x); // >0 if x < lo
    Type above = softplus(x - hi); // >0 if x > hi
    return w * (below * below + above * above);
  };

  // Temperature centers within plausible tropical bounds
  nll += bound_pen(T_ref_cots, Type(24.0), Type(31.0), Type(1.0)); // COTS thermal center
  nll += bound_pen(T_bleach,   Type(28.0), Type(32.0), Type(1.0)); // Bleaching midpoint

  // Predation half-saturation within 0.5-80% cover (on natural scale)
  nll += bound_pen(H_f, Type(0.5), Type(80.0), Type(0.1));
  nll += bound_pen(H_s, Type(0.5), Type(80.0), Type(0.1));

  // Allee threshold within plausible GBR adult densities (soft)
  nll += bound_pen(N_crit, Type(0.005), Type(0.5), Type(0.5));

  // Penalize violations from positivity smoothing (ensuring positivity of states)
  nll += lambda_pos * pen;

  // -----------------------------
  // REPORTING
  // -----------------------------
  REPORT(cots_pred); // Predicted COTS abundance (ind/m^2)
  REPORT(fast_pred); // Predicted fast coral cover (%)
  REPORT(slow_pred); // Predicted slow coral cover (%)

  return nll; // Return negative log-likelihood
}
