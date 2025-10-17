#include <TMB.hpp>

// Utility functions with numerical safeguards
template<class Type>
Type inv_logit_safe(Type x) { // Smooth 0-1 transition
  return Type(1) / (Type(1) + exp(-x));
}

template<class Type>
Type softplus(Type x) { // Smooth positivity
  return log1p(exp(x));
}

template<class Type>
Type sq(Type x) { return x * x; }

template<class Type>
Type penalty_smooth_bounds(Type x, Type lo, Type hi, Type scale) {
  // Smooth penalty if x drifts below lo or above hi (no hard constraints)
  // scale is a tuning factor to control penalty strength
  Type pen_low = sq(log1p(exp(lo - x)));   // near-zero when x >= lo
  Type pen_high = sq(log1p(exp(x - hi)));  // near-zero when x <= hi
  return scale * (pen_low + pen_high);
}

/*
Numbered model equations (discrete annual time steps; index t=1..T-1 uses lagged states t-1):

Let:
 A_t   = COTS adult density (individuals m^-2)
 F_t   = Fast-growing coral cover (Acropora, %)
 S_t   = Slow-growing coral cover (Faviidae+Porites, %)
 E_t   = Sea-surface temperature (°C)
 I_t   = COTS larval immigration (individuals m^-2 year^-1)
 Food_t= wF*F_t + wS*S_t (% weighted food availability)

1) Temperature performance (COTS): TPF_t = exp(-0.5 * ((E_t - T_opt)/T_sd)^2)
2) Food saturation: phi_food_t = Food_t / (K_food + Food_t)
3) Allee effect (smooth): phi_A_t = logistic(k_allee * (A_t - A_allee)) in (0,1)
4) COTS carrying capacity: K_A_t = kA_base + kA_food * Food_t
5) Effective COTS growth rate: r_eff_t = (r_base + r_food * phi_food_t) * TPF_t * phi_A_t
6) COTS population update (Ricker with immigration):
   A_{t+1} = A_t * exp( r_eff_t * (1 - A_t / (K_A_t + eps)) ) + imm_eff * I_t
7) Coral temperature performance:
   TPF_F_t = exp(-0.5 * ((E_t - ToptF)/TsigF)^2)
   TPF_S_t = exp(-0.5 * ((E_t - ToptS)/TsigS)^2)
8) COTS functional response on corals (Type-III) with interference:
   fr_F_t = F_t^nu / (H_F^nu + F_t^nu)
   fr_S_t = S_t^nu / (H_S^nu + S_t^nu)
   consF_t = (A_t * c_max * pref_F * fr_F_t) / (1 + interference * A_t)
   consS_t = (A_t * c_max * pref_S * fr_S_t) / (1 + interference * A_t)
9) Coral logistic growth with shared space and SST modification:
   F_{t+1} = F_t + (rF * TPF_F_t) * F_t * (1 - (F_t + S_t) / K_space) - consF_t
   S_{t+1} = S_t + (rS * TPF_S_t) * S_t * (1 - (F_t + S_t) / K_space) - consS_t

Observation model (all t):
10) cots_dat(t) ~ Lognormal(log(cots_pred(t)), sigma_cots)
11) fast_dat(t) ~ Normal(fast_pred(t),  sigma_fast)
12) slow_dat(t) ~ Normal(slow_pred(t),  sigma_slow)

Initial conditions:
A_0 = cots_dat(0); F_0 = fast_dat(0); S_0 = slow_dat(0)
Forcing pass-through:
sst_pred(t)     = sst_dat(t); cotsimm_pred(t) = cotsimm_dat(t)
*/

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;

  // Constants
  const Type eps = Type(1e-8);                 // Small constant for numerical stability
  const Type pen_scale = Type(1.0);            // Global scale for smooth bound penalties
  const Type min_sigma_coral = Type(0.1);      // Minimum SD for coral observations (cover, %)
  const Type min_sigma_cots = Type(0.05);      // Minimum SD on log-scale for COTS observations

  // DATA (must use exact names; includes the time variable 'Year')
  DATA_VECTOR(Year);         // Calendar year (integer), used only for indexing/alignment
  DATA_VECTOR(sst_dat);      // SST (°C) time series (forcing)
  DATA_VECTOR(cotsimm_dat);  // COTS larval immigration (individuals m^-2 year^-1) (forcing)
  DATA_VECTOR(cots_dat);     // Adult COTS density (individuals m^-2) (response)
  DATA_VECTOR(fast_dat);     // Fast coral cover (%) (response)
  DATA_VECTOR(slow_dat);     // Slow coral cover (%) (response)

  int n = Year.size(); // length of time series
  // Optional safety: assume all vectors have same length

  // PARAMETERS (each line includes units and role)
  PARAMETER(r_base);        // year^-1; Baseline COTS per-capita growth rate; set by literature/initial estimate, optimized
  PARAMETER(r_food);        // year^-1; Increment to COTS growth due to food saturation; literature/initial estimate, optimized
  PARAMETER(K_food);        // %; Half-saturation of food availability; initial estimate
  PARAMETER(kA_base);       // ind m^-2; Baseline carrying capacity at zero food; initial estimate
  PARAMETER(kA_food);       // (ind m^-2)/%; Carrying capacity increase per % food; initial estimate
  PARAMETER(wF);            // dimensionless; Food weight for fast coral; literature preference
  PARAMETER(wS);            // dimensionless; Food weight for slow coral; literature preference
  PARAMETER(T_opt);         // °C; COTS temperature optimum; literature-guided
  PARAMETER(T_sd);          // °C; COTS temperature breadth (std dev); literature-guided
  PARAMETER(imm_eff);       // (ind m^-2) / (ind m^-2 year^-1); Efficiency of immigration to adults next year; initial estimate
  PARAMETER(H_F);           // %; Type-III half-saturation for fast coral predation; literature/initial estimate
  PARAMETER(H_S);           // %; Type-III half-saturation for slow coral predation; literature/initial estimate
  PARAMETER(nu);            // dimensionless; Type-III exponent (>=1); literature/initial estimate
  PARAMETER(c_max);         // % per (ind m^-2) per year; Max coral consumption per starfish per year; literature/initial estimate
  PARAMETER(interference);  // (m^2 ind^-1); Predator interference scaling; initial estimate
  PARAMETER(pref_F);        // dimensionless; COTS preference weight for fast coral; literature/initial estimate
  PARAMETER(pref_S);        // dimensionless; COTS preference weight for slow coral; literature/initial estimate
  PARAMETER(rF);            // year^-1; Intrinsic growth rate of fast coral; literature/initial estimate
  PARAMETER(rS);            // year^-1; Intrinsic growth rate of slow coral; literature/initial estimate
  PARAMETER(K_space);       // %; Shared space carrying capacity for total coral cover; literature/initial estimate
  PARAMETER(ToptF);         // °C; Fast coral temperature optimum; literature-guided
  PARAMETER(TsigF);         // °C; Fast coral temperature breadth; literature-guided
  PARAMETER(ToptS);         // °C; Slow coral temperature optimum; literature-guided
  PARAMETER(TsigS);         // °C; Slow coral temperature breadth; literature-guided
  PARAMETER(A_allee);       // ind m^-2; COTS Allee threshold location; initial estimate
  PARAMETER(k_allee);       // (m^2 ind^-1); Steepness of Allee logistic; initial estimate

  // Observation error parameters (log-scale for stable positivity)
  PARAMETER(log_sigma_cots); // log SD for lognormal COTS obs; optimized
  PARAMETER(log_sigma_fast); // log SD for fast coral normal errors; optimized
  PARAMETER(log_sigma_slow); // log SD for slow coral normal errors; optimized

  // Smooth bound penalties to keep parameters in biologically plausible ranges (no hard constraints)
  Type nll = 0.0;
  nll += penalty_smooth_bounds(r_base,      Type(0.0),  Type(2.0),  pen_scale);
  nll += penalty_smooth_bounds(r_food,      Type(0.0),  Type(3.0),  pen_scale);
  nll += penalty_smooth_bounds(K_food,      Type(1.0),  Type(200.), pen_scale);
  nll += penalty_smooth_bounds(kA_base,     Type(0.0),  Type(5.0),  pen_scale);
  nll += penalty_smooth_bounds(kA_food,     Type(0.0),  Type(0.5),  pen_scale);
  nll += penalty_smooth_bounds(wF,          Type(0.0),  Type(2.0),  pen_scale);
  nll += penalty_smooth_bounds(wS,          Type(0.0),  Type(1.0),  pen_scale);
  nll += penalty_smooth_bounds(T_opt,       Type(24.0), Type(32.0), pen_scale);
  nll += penalty_smooth_bounds(T_sd,        Type(0.5),  Type(5.0),  pen_scale);
  nll += penalty_smooth_bounds(imm_eff,     Type(0.0),  Type(1.0),  pen_scale);
  nll += penalty_smooth_bounds(H_F,         Type(1.0),  Type(60.0), pen_scale);
  nll += penalty_smooth_bounds(H_S,         Type(1.0),  Type(80.0), pen_scale);
  nll += penalty_smooth_bounds(nu,          Type(1.0),  Type(3.0),  pen_scale);
  nll += penalty_smooth_bounds(c_max,       Type(0.0),  Type(50.0), pen_scale);
  nll += penalty_smooth_bounds(interference,Type(0.0),  Type(2.0),  pen_scale);
  nll += penalty_smooth_bounds(pref_F,      Type(0.0),  Type(1.0),  pen_scale);
  nll += penalty_smooth_bounds(pref_S,      Type(0.0),  Type(1.0),  pen_scale);
  nll += penalty_smooth_bounds(rF,          Type(0.0),  Type(2.0),  pen_scale);
  nll += penalty_smooth_bounds(rS,          Type(0.0),  Type(1.0),  pen_scale);
  nll += penalty_smooth_bounds(K_space,     Type(50.0), Type(100.), pen_scale);
  nll += penalty_smooth_bounds(ToptF,       Type(24.0), Type(30.0), pen_scale);
  nll += penalty_smooth_bounds(TsigF,       Type(0.5),  Type(5.0),  pen_scale);
  nll += penalty_smooth_bounds(ToptS,       Type(23.0), Type(29.0), pen_scale);
  nll += penalty_smooth_bounds(TsigS,       Type(0.5),  Type(5.0),  pen_scale);
  nll += penalty_smooth_bounds(A_allee,     Type(0.0),  Type(2.0),  pen_scale);
  nll += penalty_smooth_bounds(k_allee,     Type(0.0),  Type(10.0), pen_scale);

  // Derived observation SDs (with floors)
  Type sigma_cots = exp(log_sigma_cots) + min_sigma_cots; // log-scale SD for lognormal COTS
  Type sigma_fast = exp(log_sigma_fast) + min_sigma_coral; // SD for fast coral (%)
  Type sigma_slow = exp(log_sigma_slow) + min_sigma_coral; // SD for slow coral (%)

  // Prediction containers
  vector<Type> cots_pred(n);      // Adult COTS (ind m^-2)
  vector<Type> fast_pred(n);      // Fast coral (%)
  vector<Type> slow_pred(n);      // Slow coral (%)
  vector<Type> sst_pred(n);       // Pass-through SST (°C)
  vector<Type> cotsimm_pred(n);   // Pass-through larval immigration (ind m^-2 year^-1)

  // Initialize with data at t=0 (no data leakage thereafter)
  cots_pred(0)    = cots_dat(0);
  fast_pred(0)    = fast_dat(0);
  slow_pred(0)    = slow_dat(0);
  sst_pred(0)     = sst_dat(0);
  cotsimm_pred(0) = cotsimm_dat(0);

  // Time loop: predict using only lagged states and forcing
  for (int t = 1; t < n; t++) {
    // Previous states
    Type A_prev = cots_pred(t-1);
    Type F_prev = fast_pred(t-1);
    Type S_prev = slow_pred(t-1);
    Type E_prev = sst_dat(t-1);
    Type I_prev = cotsimm_dat(t-1);

    // Food availability (weighted % cover)
    Type Food_prev = wF * F_prev + wS * S_prev;

    // Performance curves (avoid zero division with eps in denominators)
    Type TPF = exp( -Type(0.5) * sq( (E_prev - T_opt) / (T_sd + eps) ) );          // COTS temperature performance
    Type TPF_F = exp( -Type(0.5) * sq( (E_prev - ToptF) / (TsigF + eps) ) );       // Fast coral temperature performance
    Type TPF_S = exp( -Type(0.5) * sq( (E_prev - ToptS) / (TsigS + eps) ) );       // Slow coral temperature performance

    // Food saturation and Allee effect
    Type phi_food = Food_prev / (K_food + Food_prev + eps);                        // Saturating effect of food on COTS growth
    Type phi_allee = inv_logit_safe( k_allee * (A_prev - A_allee) );               // Smooth trigger for outbreaks

    // Food-dependent carrying capacity
    Type K_A = kA_base + kA_food * Food_prev;                                      // COTS carrying capacity driven by coral

    // Effective growth rate and Ricker update for COTS
    Type r_eff = (r_base + r_food * phi_food) * TPF * phi_allee;                   // Combined modifiers
    Type A_mean = A_prev * exp( r_eff * (Type(1) - A_prev / (K_A + eps)) );        // Ricker dynamics (boom-bust capable)
    Type A_next = A_mean + imm_eff * I_prev;                                       // Additive immigration (episodic forcing)

    // Functional responses (Type III) with predator interference
    Type F_pow = pow(F_prev + eps, nu);
    Type S_pow = pow(S_prev + eps, nu);
    Type HF_pow = pow(H_F + eps, nu);
    Type HS_pow = pow(H_S + eps, nu);
    Type fr_F = F_pow / (HF_pow + F_pow + eps);
    Type fr_S = S_pow / (HS_pow + S_pow + eps);
    Type denom_int = Type(1) + interference * A_prev;                               // Smooth interference
    Type consF = (A_prev * c_max * pref_F * fr_F) / (denom_int + eps);              // % cover removed from fast coral
    Type consS = (A_prev * c_max * pref_S * fr_S) / (denom_int + eps);              // % cover removed from slow coral

    // Coral logistic growth with shared space and SST effects
    Type Tot_prev = F_prev + S_prev;
    Type F_growth = (rF * TPF_F) * F_prev * (Type(1) - Tot_prev / (K_space + eps));
    Type S_growth = (rS * TPF_S) * S_prev * (Type(1) - Tot_prev / (K_space + eps));

    // Next-step coral states (apply soft floors and soft caps)
    Type F_next = F_prev + F_growth - consF;
    Type S_next = S_prev + S_growth - consS;

    // Numerical guards: prevent negative states and keep within space capacity smoothly
    F_next = fmax(F_next, eps);
    S_next = fmax(S_next, eps);
    Type total_next = F_next + S_next;
    if (total_next > K_space) {
      // Soft renormalization to not exceed K_space (preserve composition)
      Type scale_down = (K_space - eps) / (total_next + eps);
      F_next *= scale_down;
      S_next *= scale_down;
    }

    // Assign predictions
    cots_pred(t)    = fmax(A_next, eps);
    fast_pred(t)    = F_next;
    slow_pred(t)    = S_next;
    sst_pred(t)     = sst_dat(t);          // pass-through forcing
    cotsimm_pred(t) = cotsimm_dat(t);      // pass-through forcing
  }

  // Likelihood: include every observation, every time step
  for (int t = 0; t < n; t++) {
    // COTS: lognormal on positive support
    Type y_cots = cots_dat(t);
    Type mu_log = log(cots_pred(t) + eps);
    nll -= dnorm( log(y_cots + eps), mu_log, sigma_cots, true );

    // Coral covers: normal with SD floors on raw scale
    nll -= dnorm( fast_dat(t), fast_pred(t), sigma_fast, true );
    nll -= dnorm( slow_dat(t), slow_pred(t), sigma_slow, true );
  }

  // REPORT all predictions (required)
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(sst_pred);
  REPORT(cotsimm_pred);

  // Also report some useful derived time series for diagnostics
  // (Optional, not required by naming convention)
  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);

  return nll;
}
