#include <TMB.hpp>

/*
Episodic COTS outbreak model with coral feedbacks (annual time step)

Equations (all variables refer to previous year t-1 unless otherwise noted):
1) Forcing identities (reported to include observations in likelihood)
   sst_pred(t)      = sst_dat(t)
   cotsimm_pred(t)  = cotsimm_dat(t)

2) SST-driven bleaching mortality (smooth threshold; per-year fraction):
   m_bleach(ST) = mB_max * logistic( k_bleach * ( ST - T_bleach ) )

3) Multi-prey Holling type II consumption per area (coral cover units per year):
   Denom = 1 + h * ( a_fast * F + a_slow * S )
   cons_F = A * a_fast * F / Denom
   cons_S = A * a_slow * S / Denom

4) Coral dynamics (logistic with shared capacity and competition, minus grazing and bleaching):
   F_t = F + rF * F * ( 1 - ( F + comp_FS * S ) / K_tot ) - cons_F - m_bleach(ST)*F
   S_t = S + rS * S * (  1 - ( S + comp_SF * F ) / K_tot ) - cons_S - m_bleach(ST)*S

5) Allee effect on COTS reproduction (smooth):
   f_Allee(A) = 1 / ( 1 + exp( -k_allee * ( A - A50 ) ) )

6) Temperature performance for COTS reproduction (Gaussian):
   f_T(ST) = exp( -0.5 * ((ST - T_opt)/sigma_T)^2 )

7) Food/condition effect on post-settlement survival (saturating with coral availability):
   CoralAvail = F + w_food_slow * S
   f_food(CoralAvail) = sat_food0 + sat_food1 * CoralAvail / ( CoralAvail + H_food )

8) COTS recruitment (to adults by next year):
   Recruit = fec_max * A * f_Allee(A) * f_T(ST) * f_food(CoralAvail) + Immigration

9) Starvation/crash mortality factor (smoothly increases when coral is low):
   f_starv(F+S) = m_starv * A * logistic( k_starv * ( c_starv - (F+S) ) )

10) COTS dynamics (adults):
   A_t = A + Recruit - ( m0 * A + m_dd * A^2 ) - f_starv(F+S)

11) Observation models:
   - COTS (cots_dat): Lognormal on densities (strictly positive)
   - Corals (fast_dat, slow_dat): Beta on proportions in (0,1)
   - Forcings (sst_dat, cotsimm_dat): Normal around identity with small SDs

Notes:
- All updates use t-1 state/forcing to compute t prediction (no data leakage).
- posfun is used to ensure non-negative states with smooth penalties.
- Small constants (eps) are used to avoid division by zero.
*/

// Use TMB's built-in invlogit (convenience.hpp). Do NOT redefine to avoid ambiguity.

template<class Type>
Type clamp01(const Type& x, const Type& eps){
  // Clamp to (eps, 1-eps) smoothly for numerical stability
  Type y = CppAD::CondExpLt(x, eps, eps, x);
  y = CppAD::CondExpGt(y, Type(1.0)-eps, Type(1.0)-eps, y);
  return y;
}

template<class Type>
Type sqr(const Type& x){
  // Simple square helper to avoid reliance on Eigen::square in scalar context
  return x * x;
}

template<class Type>
Type posfun_smooth(const Type& x, const Type& eps, Type& pen){
  // Smooth positive transform with small penalty if x < eps (AD-safe)
  // Follows an ADMB-style approach:
  // If x >= eps: return x
  // If x < eps : return eps / (2 - x/eps) and add a small quadratic penalty
  Type tmp = eps / (Type(2.0) - x/eps);
  Type res = CppAD::CondExpGe(x, eps, x, tmp);
  Type diff = eps - x;
  Type addpen = CppAD::CondExpGe(x, eps, Type(0.0), diff * diff);
  pen += addpen * Type(1e-6);
  return res;
}

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ----------------------------
  // Data
  // ----------------------------
  DATA_VECTOR(Year);            // Year (calendar year; used for alignment/reporting)
  DATA_VECTOR(cots_dat);        // Adult COTS density (ind m^-2), strictly positive
  DATA_VECTOR(fast_dat);        // Fast-growing coral cover (%) - Acropora spp.
  DATA_VECTOR(slow_dat);        // Slow-growing coral cover (%) - Faviidae/Porites spp.
  DATA_VECTOR(sst_dat);         // Sea Surface Temperature (°C)
  DATA_VECTOR(cotsimm_dat);     // Larval immigration (ind m^-2 yr^-1)

  int n = Year.size();
  Type nll = 0.0;               // Negative log-likelihood
  Type eps = Type(1e-8);        // Small constant for numerical stability
  Type pen = 0.0;               // Penalty accumulator (e.g., from posfun)

  // ----------------------------
  // Parameters (transformed to biologically meaningful ranges)
  // ----------------------------

  // Coral growth & competition
  PARAMETER(log_rF);            // log of rF (yr^-1): intrinsic growth rate of fast coral; estimated from recovery slopes
  PARAMETER(log_rS);            // log of rS (yr^-1): intrinsic growth rate of slow coral; estimated from recovery slopes
  PARAMETER(log_K_tot);         // log of K_tot (%): shared carrying capacity (total live coral substrate)
  PARAMETER(log_comp_FS);       // log of comp_FS (unitless): how much S counts against F capacity
  PARAMETER(log_comp_SF);       // log of comp_SF (unitless): how much F counts against S capacity

  // COTS feeding (multi-prey Holling II)
  PARAMETER(log_a_fast);        // log of a_fast ((% cover)^-1 yr^-1 per starfish): attack/clearance for fast coral
  PARAMETER(log_a_slow);        // log of a_slow ((% cover)^-1 yr^-1 per starfish): attack/clearance for slow coral
  PARAMETER(log_h);             // log of handling time h ((% cover)^-1): increases saturation with total prey

  // SST-driven bleaching on corals
  PARAMETER(log_mB_max);        // log of mB_max (yr^-1): maximum fractional bleaching-induced mortality
  PARAMETER(T_bleach);          // T_bleach (°C): SST at which bleaching risk is 50%
  PARAMETER(log_k_bleach);      // log of k_bleach (°C^-1): steepness of bleaching response

  // COTS reproduction & modifiers
  PARAMETER(log_fec_max);       // log of fec_max (yr^-1): max adult-to-adult recruitment rate (per adult)
  PARAMETER(log_A50);           // log of A50 (ind m^-2): Allee half-saturation for reproduction
  PARAMETER(log_k_allee);       // log of k_allee ((ind m^-2)^-1): steepness of Allee effect
  PARAMETER(log_H_food);        // log of H_food (%): half-saturation coral cover in food/condition effect
  PARAMETER(logit_w_food_slow); // logit of w_food_slow (unitless in [0,1]): weight of slow coral in food proxy
  PARAMETER(logit_sat_food0);   // logit of sat_food0 (unitless in [0,1]): baseline fraction of food effect
  PARAMETER(logit_sat_food1);   // logit of sat_food1 (unitless in [0,1]): additional fraction up to 1

  // COTS mortality terms
  PARAMETER(log_m0);            // log of m0 (yr^-1): baseline adult mortality
  PARAMETER(log_m_dd);          // log of m_dd ((ind m^-2)^-1 yr^-1): crowding mortality coefficient
  PARAMETER(log_m_starv);       // log of m_starv (yr^-1): max starvation/crash mortality multiplier
  PARAMETER(log_k_starv);       // log of k_starv (%^-1): steepness of starvation response to low coral
  PARAMETER(log_c_starv);       // log of c_starv (%): coral cover threshold where starvation risk rises

  // Temperature performance for COTS reproduction
  PARAMETER(T_opt);             // T_opt (°C): optimal SST for COTS reproductive performance
  PARAMETER(log_sigma_T);       // log of sigma_T (°C): width of temperature performance curve

  // Observation model parameters
  PARAMETER(log_sigma_cots);    // log of sigma_cots: lognormal SD for cots_dat
  PARAMETER(log_phi_fast);      // log of phi_fast: beta precision for fast_dat
  PARAMETER(log_phi_slow);      // log of phi_slow: beta precision for slow_dat
  PARAMETER(log_sst_sd);        // log of sst_sd: SD for SST forcing identity likelihood
  PARAMETER(log_imm_sd);        // log of imm_sd: SD for immigration forcing identity likelihood

  // Transforms to working scale
  Type rF        = exp(log_rF);
  Type rS        = exp(log_rS);
  Type K_tot     = exp(log_K_tot);
  Type comp_FS   = exp(log_comp_FS);
  Type comp_SF   = exp(log_comp_SF);

  Type a_fast    = exp(log_a_fast);
  Type a_slow    = exp(log_a_slow);
  Type h         = exp(log_h);

  Type mB_max    = exp(log_mB_max);
  Type k_bleach  = exp(log_k_bleach);

  Type fec_max   = exp(log_fec_max);
  Type A50       = exp(log_A50);
  Type k_allee   = exp(log_k_allee);
  Type H_food    = exp(log_H_food);
  Type w_food_slow = invlogit(logit_w_food_slow); // weight of slow coral in food proxy (0..1)
  Type sat_food0 = invlogit(logit_sat_food0);     // baseline fraction (0..1)
  Type sat_food1 = invlogit(logit_sat_food1);     // incremental fraction (0..1)

  Type m0        = exp(log_m0);
  Type m_dd      = exp(log_m_dd);
  Type m_starv   = exp(log_m_starv);
  Type k_starv   = exp(log_k_starv);
  Type c_starv   = exp(log_c_starv);

  Type sigma_T   = exp(log_sigma_T);

  Type sigma_cots = exp(log_sigma_cots);
  Type phi_fast   = exp(log_phi_fast); // ensure >0
  Type phi_slow   = exp(log_phi_slow); // ensure >0
  Type sst_sd     = exp(log_sst_sd);
  Type imm_sd     = exp(log_imm_sd);

  // Minimum SDs to avoid numerical issues
  Type min_sd_obs = Type(1e-3);
  sigma_cots = sqrt( sqr(sigma_cots) + sqr(min_sd_obs) );
  Type sst_sd_eff = sqrt( sqr(sst_sd) + sqr(Type(1e-6)) );
  Type imm_sd_eff = sqrt( sqr(imm_sd) + sqr(Type(1e-6)) );

  // ----------------------------
  // Predictions
  // ----------------------------
  vector<Type> cots_pred(n);     // Adult COTS prediction (ind m^-2)
  vector<Type> fast_pred(n);     // Fast coral prediction (% cover)
  vector<Type> slow_pred(n);     // Slow coral prediction (% cover)
  vector<Type> sst_pred(n);      // SST prediction (°C) - identity to data
  vector<Type> cotsimm_pred(n);  // Immigration prediction (ind m^-2 yr^-1) - identity to data

  // Initial conditions from data (no optimization on initial states)
  cots_pred(0)    = cots_dat(0);
  fast_pred(0)    = fast_dat(0);
  slow_pred(0)    = slow_dat(0);
  sst_pred(0)     = sst_dat(0);
  cotsimm_pred(0) = cotsimm_dat(0);

  // Likelihood contributions for t = 0 (observations at initial state)
  // COTS lognormal (strictly positive)
  nll -= dnorm( log(cots_dat(0) + eps), log(cots_pred(0) + eps), sigma_cots, true );
  // Coral beta (use proportions in (0,1))
  Type fast0_prop = clamp01(fast_dat(0) / Type(100.0), Type(1e-6));
  Type slow0_prop = clamp01(slow_dat(0) / Type(100.0), Type(1e-6));
  Type muF0 = clamp01(fast_pred(0) / Type(100.0), Type(1e-6));
  Type muS0 = clamp01(slow_pred(0) / Type(100.0), Type(1e-6));
  Type aF0 = muF0 * phi_fast;
  Type bF0 = (Type(1.0) - muF0) * phi_fast;
  Type aS0 = muS0 * phi_slow;
  Type bS0 = (Type(1.0) - muS0) * phi_slow;
  nll -= dbeta(fast0_prop, aF0 + eps, bF0 + eps, true);
  nll -= dbeta(slow0_prop, aS0 + eps, bS0 + eps, true);
  // Forcings likelihood (identity with small SDs)
  nll -= dnorm(sst_dat(0), sst_pred(0), sst_sd_eff, true);
  nll -= dnorm(cotsimm_dat(0), cotsimm_pred(0), imm_sd_eff, true);

  // Forward simulation
  for (int t = 1; t < n; t++) {
    // Carry forward forcing predictions as identities
    sst_pred(t)     = sst_dat(t);           // identity (observed forcing)
    cotsimm_pred(t) = cotsimm_dat(t);       // identity (observed forcing)

    // Previous states (t-1)
    Type A = cots_pred(t-1) + eps;          // adult COTS (ind m^-2)
    Type F = fast_pred(t-1) + eps;          // fast coral (%)
    Type S = slow_pred(t-1) + eps;          // slow coral (%)
    Type ST = sst_dat(t-1);                 // SST driver at t-1
    Type IM = cotsimm_dat(t-1);             // immigration at t-1

    // 2) Bleaching mortality fraction (0..mB_max)
    Type m_bleach = mB_max * invlogit( k_bleach * ( ST - T_bleach ) );

    // 3) Multi-prey Holling II consumption (cover units per year)
    Type Denom = Type(1.0) + h * ( a_fast * F + a_slow * S );
    Type cons_F = A * a_fast * F / (Denom + eps);
    Type cons_S = A * a_slow * S / (Denom + eps);

    // 4) Coral dynamics
    Type F_raw = F + rF * F * ( Type(1.0) - ( F + comp_FS * S ) / (K_tot + eps) ) - cons_F - m_bleach * F;
    Type S_raw = S + rS * S * ( Type(1.0) - ( S + comp_SF * F ) / (K_tot + eps) ) - cons_S - m_bleach * S;

    // Keep corals non-negative (smooth) and softly penalize over-capacity
    Type F_pos = posfun_smooth(F_raw, Type(1e-8), pen);
    Type S_pos = posfun_smooth(S_raw, Type(1e-8), pen);

    // Soft penalty if total coral exceeds K_tot (keeps within biological range without a hard cap)
    Type over = F_pos + S_pos - K_tot;
    Type over_pos = CppAD::CondExpGt(over, Type(0.0), over, Type(0.0));
    pen += sqr(over_pos) * Type(1e-4);

    fast_pred(t) = F_pos;
    slow_pred(t) = S_pos;

    // 5) Allee effect
    Type f_allee = Type(1.0) / ( Type(1.0) + exp( -k_allee * ( A - A50 ) ) );

    // 6) Temperature performance
    Type zT = (ST - T_opt) / (sigma_T + eps);
    Type f_T = exp( Type(-0.5) * zT * zT ); // in [0,1]

    // 7) Food/condition factor
    Type CoralAvail = F + w_food_slow * S;
    Type f_food = sat_food0 + sat_food1 * CoralAvail / ( CoralAvail + H_food + eps );
    // Ensure f_food in (0,1]
    f_food = clamp01(f_food, Type(1e-6));

    // 8) Recruitment (to adults)
    Type Recruit = fec_max * A * f_allee * f_T * f_food + IM;

    // 9) Starvation/crash mortality
    Type starv_factor = invlogit( k_starv * ( c_starv - (F + S) ) ); // rises as coral declines
    Type M_starv = m_starv * A * starv_factor;

    // 10) COTS dynamics
    Type A_raw = A + Recruit - ( m0 * A + m_dd * A * A ) - M_starv;
    Type A_pos = posfun_smooth(A_raw, Type(1e-10), pen);
    cots_pred(t) = A_pos;

    // 11) Likelihood contributions at time t
    // COTS lognormal
    nll -= dnorm( log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots, true );

    // Coral beta likelihoods (proportions)
    Type yF = clamp01(fast_dat(t) / Type(100.0), Type(1e-6));
    Type yS = clamp01(slow_dat(t) / Type(100.0), Type(1e-6));
    Type muF = clamp01(fast_pred(t) / Type(100.0), Type(1e-6));
    Type muS = clamp01(slow_pred(t) / Type(100.0), Type(1e-6));
    Type aF = muF * phi_fast + eps;
    Type bF = (Type(1.0) - muF) * phi_fast + eps;
    Type aS = muS * phi_slow + eps;
    Type bS = (Type(1.0) - muS) * phi_slow + eps;
    nll -= dbeta(yF, aF, bF, true);
    nll -= dbeta(yS, aS, bS, true);

    // Forcing identity likelihoods
    nll -= dnorm(sst_dat(t), sst_pred(t), sst_sd_eff, true);
    nll -= dnorm(cotsimm_dat(t), cotsimm_pred(t), imm_sd_eff, true);
  }

  // Add accumulated smooth penalties
  nll += pen;

  // Report predictions (for diagnostics and plotting)
  REPORT(Year);
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(sst_pred);
  REPORT(cotsimm_pred);

  // Also report some derived process rates for interpretability
  REPORT(rF);
  REPORT(rS);
  REPORT(K_tot);
  REPORT(comp_FS);
  REPORT(comp_SF);
  REPORT(a_fast);
  REPORT(a_slow);
  REPORT(h);
  REPORT(mB_max);
  REPORT(T_bleach);
  REPORT(k_bleach);
  REPORT(fec_max);
  REPORT(A50);
  REPORT(k_allee);
  REPORT(H_food);
  REPORT(w_food_slow);
  REPORT(sat_food0);
  REPORT(sat_food1);
  REPORT(m0);
  REPORT(m_dd);
  REPORT(m_starv);
  REPORT(k_starv);
  REPORT(c_starv);
  REPORT(T_opt);
  REPORT(sigma_T);
  REPORT(sigma_cots);
  REPORT(phi_fast);
  REPORT(phi_slow);
  REPORT(sst_sd_eff);
  REPORT(imm_sd_eff);

  return nll;
}
