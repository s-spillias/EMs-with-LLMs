#include <TMB.hpp>

// Helper functions for numerical stability
// Use TMB-provided invlogit; do not redefine to avoid conflicts.
// Stable softplus without using std::log1p (works with AD types)
template<class Type>
Type softplus(Type x){
  Type zero = Type(0);
  // softplus(x) = max(0,x) + log(1 + exp(-|x|)) implemented with AD-safe conditionals
  return CppAD::CondExpGt(
    x, zero,
    x + log(Type(1) + exp(-x)),
    log(Type(1) + exp(x))
  );
}

/* 
Model overview and equations (all time indices t refer to rows in the data; Year is the time key):
State variables:
- A_t: Adult COTS density (individuals m^-2)
- F_t: Fast coral (Acropora) cover fraction (0-1); fast_dat are percentages, so fast_pred = 100*F_t
- S_t: Slow coral (Faviidae/Porites) cover fraction (0-1); slow_pred = 100*S_t
- H_t = 1 - F_t - S_t: Free space fraction (can be <0 transiently; growth terms use saturating H_t/(H_t + K_space) to ensure smooth behavior)

Forcings (observed):
- sst_dat[t] (°C): Sea-surface temperature
- cotsimm_dat[t] (individuals m^-2 yr^-1): Larval immigration rate

Functional forms (only previous-step states used to predict next-step states):
1) Food availability (saturating with coral cover and preference; phi_food >= 1):
   food_t = wF * F_t^phi_food + wS * S_t^phi_food
   g_food_t = food_t / (K_food + food_t + eps)

2) SST effect on COTS reproduction (smooth logistic around T_sst_half):
   g_sst_t = invlogit( beta_sst * (sst_t - T_sst_half) )

3) Allee (mate-finding) effect for COTS recruitment:
   g_allee_t = A_t / (A_t + A50 + eps)

4) COTS recruitment and survival (recruitment includes overcompensation via exp(-gamma * A_t)):
   recruit_{t->t+1} = eta_rec * A_t * rA * exp(-gamma * A_t) * g_food_t * g_sst_t * g_allee_t
                      + eta_imm * cotsimm_t
   mortal_starv_t = m_starv_max * (1 - g_food_t)
   surv_factor_t  = exp( - mA - mortal_starv_t )
   A_{t+1} = A_t * surv_factor_t + recruit_{t->t+1}

5) COTS predation (Holling type II with preference-weighted availability):
   avail_t = prefF * F_t + prefS * S_t
   satFood_t = avail_t / (h_g + avail_t + eps)
   total_eaten_t = gmax * A_t * satFood_t                  [fraction of substrate yr^-1]
   shareF_t = (prefF * F_t + eps) / (avail_t + 2*eps)
   pF_t = shareF_t * total_eaten_t
   pS_t = (1 - shareF_t) * total_eaten_t

6) SST bleaching mortality (smooth logistic around T_bleach):
   bleach_t = invlogit( k_bleach * (sst_t - T_bleach) )
   mF_bleach_t = bF_max * bleach_t
   mS_bleach_t = bS_max * bleach_t

7) Coral growth and mortality (logistic-like growth limited by free space; smooth, saturating):
   H_t = 1 - F_t - S_t
   growthF_t = rF * F_t * H_t / (H_t + K_space + eps)
   growthS_t = rS * S_t * H_t / (H_t + K_space + eps)
   F_{t+1} = F_t + growthF_t - pF_t - (mF + mF_bleach_t) * F_t
   S_{t+1} = S_t + growthS_t - pS_t - (mS + mS_bleach_t) * S_t

Observation models (applied to every row; predictions indexed consistently with Year):
- cots_dat > 0: lognormal error: log(cots_dat) ~ Normal( log(cots_pred), sigma_cots )
- fast_dat, slow_dat are in % cover: use logit-normal on fractions y/100: logit(y_frac) ~ Normal( logit(pred_frac), sigma_fast/slow )
- sst_dat (°C) and cotsimm_dat (ind m^-2 yr^-1): included with Gaussian error around identity predictions (sst_pred=sst_dat, cotsimm_pred=cotsimm_dat), using minimum standard deviations to avoid degeneracy.

Notes:
- Small constants (eps) added to all denominators and log/logit transforms.
- Smooth transitions used (invlogit, softplus); we avoid using current observed values of response variables in process equations to prevent data leakage.
*/

template<class Type>
Type objective_function<Type>::operator() ()
{
  // -----------------------------
  // Data (names match data files)
  // -----------------------------
  DATA_VECTOR(Year);        // Observation years (used for alignment and reporting)
  DATA_VECTOR(sst_dat);     // Sea-surface temperature (°C), forcing time series
  DATA_VECTOR(cotsimm_dat); // Larval immigration (ind m^-2 yr^-1), forcing time series
  DATA_VECTOR(cots_dat);    // Adult COTS density (ind m^-2)
  DATA_VECTOR(fast_dat);    // Fast coral cover (%)
  DATA_VECTOR(slow_dat);    // Slow coral cover (%)

  int n = Year.size(); // number of time steps

  // ---------------------------------
  // Parameters (unconstrained scales)
  // ---------------------------------
  PARAMETER(log_rA);          // log intrinsic reproductive rate of COTS (year^-1), literature/expert-informed magnitude
  PARAMETER(log_gamma);       // log overcompensation strength in COTS recruitment (m^2 ind^-1), higher -> earlier saturation
  PARAMETER(log_A50);         // log Allee half-saturation density for COTS mating (ind m^-2)
  PARAMETER(log_eta_rec);     // log efficiency converting reproductive output to recruits (unitless)
  PARAMETER(log_eta_imm);     // log efficiency converting larval immigration to adult recruits (unitless, survival to adulthood)

  PARAMETER(log_wF);          // log weight for fast coral in food limitation (preference/quality, unitless)
  PARAMETER(log_wS);          // log weight for slow coral in food limitation (unitless)
  PARAMETER(log_Kfood);       // log half-saturation for food limitation (fraction of substrate, unitless)
  PARAMETER(log_phi_food_excess); // log of (phi_food - 1), ensures phi_food >= 1

  PARAMETER(log_gmax);        // log max coral consumption converted to cover per COTS (fraction yr^-1 per ind m^-2)
  PARAMETER(log_hg);          // log half-saturation for feeding functional response (fraction of substrate)
  PARAMETER(log_prefF);       // log preference coefficient for fast coral in feeding (unitless)
  PARAMETER(log_prefS);       // log preference coefficient for slow coral in feeding (unitless)

  PARAMETER(log_mA);          // log baseline adult COTS mortality (year^-1)
  PARAMETER(log_mstarv);      // log max starvation mortality add-on (year^-1)

  PARAMETER(T_sst_half);      // SST (°C) at which SST effect on reproduction is 50%
  PARAMETER(log_beta_sst);    // log slope of SST effect on reproduction (°C^-1), positive

  PARAMETER(T_bleach);        // SST (°C) at which bleaching mortality reaches 50%
  PARAMETER(log_k_bleach);    // log slope of bleaching response (°C^-1)
  PARAMETER(log_bF_max);      // log max bleaching-induced fractional mortality for fast coral (year^-1)
  PARAMETER(log_bS_max);      // log max bleaching-induced fractional mortality for slow coral (year^-1)

  PARAMETER(log_rF);          // log intrinsic growth rate of fast coral (year^-1)
  PARAMETER(log_rS);          // log intrinsic growth rate of slow coral (year^-1)
  PARAMETER(log_Kspace);      // log half-saturation for space-limited growth (fraction of substrate)
  PARAMETER(log_mF);          // log background mortality rate of fast coral (year^-1)
  PARAMETER(log_mS);          // log background mortality rate of slow coral (year^-1)

  PARAMETER(log_A0);          // log initial adult COTS density A_0 (ind m^-2)
  PARAMETER(logit_F0);        // logit initial fast coral fraction F_0 (unitless)
  PARAMETER(logit_S0);        // logit initial slow coral fraction S_0 (unitless)

  // Observation error parameters (kept positive and bounded away from zero by sigma_min)
  PARAMETER(log_sigma_cots);    // log SD for log(cots) observation error
  PARAMETER(log_sigma_fast);    // log SD for logit(fast_frac) observation error
  PARAMETER(log_sigma_slow);    // log SD for logit(slow_frac) observation error
  PARAMETER(log_sigma_sst);     // log SD for SST observation error
  PARAMETER(log_sigma_cotsimm); // log SD for cotsimm observation error

  // ------------------------
  // Transformed parameters
  // ------------------------
  Type eps = Type(1e-8); // small constant to avoid division by zero
  // COTS reproduction and density-dependence
  Type rA      = exp(log_rA);
  Type gamma   = exp(log_gamma);
  Type A50     = exp(log_A50);
  Type eta_rec = exp(log_eta_rec);
  Type eta_imm = exp(log_eta_imm);
  // Food limitation
  Type wF = exp(log_wF);
  Type wS = exp(log_wS);
  Type K_food = exp(log_Kfood);
  Type phi_food = Type(1.0) + exp(log_phi_food_excess);
  // Feeding and preferences
  Type gmax  = exp(log_gmax);
  Type h_g   = exp(log_hg);
  Type prefF = exp(log_prefF);
  Type prefS = exp(log_prefS);
  // Mortality
  Type mA         = exp(log_mA);
  Type m_starv_max= exp(log_mstarv);
  // SST effects
  Type beta_sst = exp(log_beta_sst);
  Type k_bleach = exp(log_k_bleach);
  Type bF_max   = exp(log_bF_max);
  Type bS_max   = exp(log_bS_max);
  // Coral demography
  Type rF = exp(log_rF);
  Type rS = exp(log_rS);
  Type K_space = exp(log_Kspace);
  Type mF = exp(log_mF);
  Type mS = exp(log_mS);
  // Initial conditions
  Type A = exp(log_A0);                          // A_0 >= 0 (ind m^-2)
  Type F = invlogit(logit_F0);                   // F_0 in (0,1)
  Type S = invlogit(logit_S0);                   // S_0 in (0,1)

  // Observation SDs with a minimum floor
  Type sigma_min_ln  = Type(0.05);  // minimum SD for log-scale errors
  Type sigma_min_logit = Type(0.05);// minimum SD for logit-scale errors
  Type sigma_min_gauss = Type(0.01);// minimum SD for Gaussian errors
  Type sigma_cots     = sqrt( exp(2.0*log_sigma_cots)     + sigma_min_ln*sigma_min_ln );
  Type sigma_fast     = sqrt( exp(2.0*log_sigma_fast)     + sigma_min_logit*sigma_min_logit );
  Type sigma_slow     = sqrt( exp(2.0*log_sigma_slow)     + sigma_min_logit*sigma_min_logit );
  Type sigma_sst      = sqrt( exp(2.0*log_sigma_sst)      + sigma_min_gauss*sigma_min_gauss );
  Type sigma_cotsimm  = sqrt( exp(2.0*log_sigma_cotsimm)  + sigma_min_gauss*sigma_min_gauss );

  // --------------------------------
  // Containers for predictions
  // --------------------------------
  vector<Type> cots_pred(n);   // predicted adult COTS density (ind m^-2)
  vector<Type> fast_pred(n);   // predicted fast coral cover (%)
  vector<Type> slow_pred(n);   // predicted slow coral cover (%)
  vector<Type> sst_pred(n);    // predicted SST (°C) - identity here
  vector<Type> cotsimm_pred(n);// predicted cots immigration (ind m^-2 yr^-1) - identity here

  // Initialize predictions at first time step using initial conditions
  cots_pred(0) = A;           // A_0 predicted at first Year
  fast_pred(0) = Type(100.0) * F;
  slow_pred(0) = Type(100.0) * S;
  sst_pred(0) = sst_dat(0);           // include in likelihood with Gaussian error
  cotsimm_pred(0) = cotsimm_dat(0);   // include in likelihood with Gaussian error

  // --------------------------------
  // Negative log-likelihood
  // --------------------------------
  Type nll = 0.0;

  // Soft priors/penalties (smooth bounds within biological ranges)
  // Center SST half-points near climatology; wide priors to act as soft bounds
  nll -= dnorm(T_sst_half, Type(28.0), Type(3.0), true);  // SST effect midpoint ~ 28°C
  nll -= dnorm(T_bleach,   Type(30.5), Type(2.0), true);  // Bleaching midpoint ~ 30.5°C
  // Growth rate priors (weakly informative)
  nll -= dnorm(log_rF, Type(log(Type(0.5))), Type(1.0), true);
  nll -= dnorm(log_rS, Type(log(Type(0.2))), Type(1.0), true);

  // --------------------------------
  // Time loop: process model
  // --------------------------------
  for(int t = 1; t < n; t++){
    // Previous-step forcings (only lagged inputs used)
    Type sst_prev = sst_dat(t-1);       // °C
    Type imm_prev = cotsimm_dat(t-1);   // ind m^-2 yr^-1

    // 1) Food availability and limitation (saturating, preference-weighted)
    Type food_prev = wF * pow(F + eps, phi_food) + wS * pow(S + eps, phi_food);
    Type g_food = food_prev / (K_food + food_prev + eps);

    // 2) SST effect on reproduction (smooth logistic)
    Type g_sst = invlogit( beta_sst * (sst_prev - T_sst_half) );

    // 3) Allee effect (mate-finding)
    Type g_allee = A / (A + A50 + eps);

    // 4) COTS recruitment and survival to next step
    Type recruit = eta_rec * A * rA * exp( -gamma * A ) * g_food * g_sst * g_allee
                   + eta_imm * imm_prev;
    Type mortal_starv = m_starv_max * (Type(1.0) - g_food);
    Type surv_factor  = exp( - mA - mortal_starv );
    Type A_next = A * surv_factor + recruit;

    // 5) Feeding (Holling type II with preference-weighted availability)
    Type avail = prefF * F + prefS * S;
    Type satFood = avail / (h_g + avail + eps);
    Type total_eaten = gmax * A * satFood; // fraction yr^-1
    Type shareF = (prefF * F + eps) / (avail + 2.0*eps);
    Type pF = shareF * total_eaten;
    Type pS = (Type(1.0) - shareF) * total_eaten;

    // 6) Bleaching mortality (temperature-driven, smooth logistic)
    Type bleach = invlogit( k_bleach * (sst_prev - T_bleach) );
    Type mF_bleach = bF_max * bleach;
    Type mS_bleach = bS_max * bleach;

    // 7) Coral growth and mortality
    Type H = Type(1.0) - F - S;
    Type growthF = rF * F * H / (H + K_space + eps);
    Type growthS = rS * S * H / (H + K_space + eps);

    Type F_temp = F + growthF - pF - (mF + mF_bleach) * F;
    Type S_temp = S + growthS - pS - (mS + mS_bleach) * S;

    // Smooth non-negativity via softplus offset (keeps >0 without hard cutoff)
    Type F_next = softplus(F_temp) + eps;
    Type S_next = softplus(S_temp) + eps;

    // Advance states
    A = A_next;
    F = F_next;
    S = S_next;

    // Store predictions in observation units
    cots_pred(t) = A;                       // ind m^-2
    fast_pred(t) = Type(100.0) * F;         // %
    slow_pred(t) = Type(100.0) * S;         // %
    sst_pred(t) = sst_dat(t);               // identity prediction included in likelihood
    cotsimm_pred(t) = cotsimm_dat(t);       // identity prediction included in likelihood
  }

  // --------------------------------
  // Likelihood: include all observations
  // --------------------------------
  for(int t = 0; t < n; t++){
    // COTS: lognormal likelihood on strictly positive values
    Type y_cots = log(cots_dat(t) + eps);
    Type mu_cots = log(cots_pred(t) + eps);
    nll -= dnorm(y_cots, mu_cots, sigma_cots, true);

    // Fast coral: logit-normal on fractions
    Type y_fast_frac = (fast_dat(t) / Type(100.0));
    Type y_fast_clamp = CppAD::CondExpLt(y_fast_frac, eps, eps, CppAD::CondExpGt(y_fast_frac, Type(1.0)-eps, Type(1.0)-eps, y_fast_frac));
    Type p_fast_frac = (fast_pred(t) / Type(100.0));
    Type p_fast_clamp = CppAD::CondExpLt(p_fast_frac, eps, eps, CppAD::CondExpGt(p_fast_frac, Type(1.0)-eps, Type(1.0)-eps, p_fast_frac));
    Type z_fast = log( y_fast_clamp + eps ) - log( Type(1.0) - y_fast_clamp + eps );
    Type mu_fast = log( p_fast_clamp + eps ) - log( Type(1.0) - p_fast_clamp + eps );
    nll -= dnorm(z_fast, mu_fast, sigma_fast, true);

    // Slow coral: logit-normal on fractions
    Type y_slow_frac = (slow_dat(t) / Type(100.0));
    Type y_slow_clamp = CppAD::CondExpLt(y_slow_frac, eps, eps, CppAD::CondExpGt(y_slow_frac, Type(1.0)-eps, Type(1.0)-eps, y_slow_frac));
    Type p_slow_frac = (slow_pred(t) / Type(100.0));
    Type p_slow_clamp = CppAD::CondExpLt(p_slow_frac, eps, eps, CppAD::CondExpGt(p_slow_frac, Type(1.0)-eps, Type(1.0)-eps, p_slow_frac));
    Type z_slow = log( y_slow_clamp + eps ) - log( Type(1.0) - y_slow_clamp + eps );
    Type mu_slow = log( p_slow_clamp + eps ) - log( Type(1.0) - p_slow_clamp + eps );
    nll -= dnorm(z_slow, mu_slow, sigma_slow, true);

    // SST: Gaussian likelihood (identity prediction)
    nll -= dnorm(sst_dat(t), sst_pred(t), sigma_sst, true);

    // Larval immigration: Gaussian likelihood (identity prediction)
    nll -= dnorm(cotsimm_dat(t), cotsimm_pred(t), sigma_cotsimm, true);
  }

  // -----------------------------
  // Reporting (all *_pred vars)
  // -----------------------------
  REPORT(Year);            // time index
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(sst_pred);
  REPORT(cotsimm_pred);

  // Report some key transformed parameters for interpretability
  ADREPORT(rA);
  ADREPORT(gamma);
  ADREPORT(A50);
  ADREPORT(eta_rec);
  ADREPORT(eta_imm);
  ADREPORT(rF);
  ADREPORT(rS);
  ADREPORT(mA);
  ADREPORT(bF_max);
  ADREPORT(bS_max);

  return nll;
}
