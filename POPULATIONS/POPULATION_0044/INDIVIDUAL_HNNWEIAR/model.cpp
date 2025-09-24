#include <TMB.hpp>  // TMB core

// Helper: inverse logit with numerical stability
template<class Type>
Type invlogit_safe(Type x) {
  return Type(1.0) / (Type(1.0) + exp(-x));
}

// Helper: logit with numerical stability on (0,1)
template<class Type>
Type logit_safe(Type p, Type eps) {
  p = CppAD::CondExpLt(p, eps, eps, p);                               // clamp low  // ensures p >= eps
  p = CppAD::CondExpGt(p, Type(1.0) - eps, Type(1.0) - eps, p);       // clamp high // ensures p <= 1-eps
  return log(p / (Type(1.0) - p));
}

/*
Equations (all variables are annual time steps t):
1) Environmental suitability (Gaussian around optimal SST):
   env_temp_{t} = exp( -0.5 * ((sst_dat_{t} - sst_opt) / sst_scale)^2 )

2) Immigration signal scaling:
   imm_scaled_{t} = log( 1 + alpha_imm * cotsimm_dat_{t} )

3) Outbreak gate (smooth threshold):
   gate_{t} = invlogit( k_gate * ( env_temp_{t} + imm_scaled_{t} - theta_outbreak ) )

4) Food feedback (saturating):
   food_{t} = wF * F_{t} / (F_halfF + F_{t}) + (1 - wF) * S_{t} / (S_halfS + S_{t})

5) COTS adults (Ricker + food + environment + saturating immigration addition):
   A_{t+1} = A_{t} * exp( rA + phi_food * food_{t} + beta_env * gate_{t} - cA * A_{t} )
             + R_imm * imm_scaled_{t} / ( 1 + k_imm_sat * A_{t} )

6) Generalized functional response (q ∈ [1, ∞)):
   denom_{t} = 1 + h * ( aF * F_{t}^q + aS * S_{t}^q )
   frac_loss_F_{t} = 1 - exp( -gamma_pred * A_{t} * aF * F_{t}^q / (denom_{t} + eps) )
   frac_loss_S_{t} = 1 - exp( -gamma_pred * A_{t} * aS * S_{t}^q / (denom_{t} + eps) )
   pred_loss_F_{t} = frac_loss_F_{t} * F_{t}
   pred_loss_S_{t} = frac_loss_S_{t} * S_{t}

7) Coral logistic growth with competition for shared space K=100%:
   gF_{t} = rF * F_{t} * ( 1 - (F_{t} + S_{t}) / K )
   gS_{t} = rS * S_{t} * ( 1 - (F_{t} + S_{t}) / K )

8) Coral updates with smooth bounding to [0, K] via logit space accumulation:
   Let deltaF = gF_{t} - mF * F_{t} - pred_loss_F_{t}
       deltaS = gS_{t} - mS * S_{t} - pred_loss_S_{t}
   pF_{t} = F_{t} / K, pS_{t} = S_{t} / K
   F_{t+1} = K * invlogit( logit(pF_{t}) + deltaF / K )
   S_{t+1} = K * invlogit( logit(pS_{t}) + deltaS / K )

Observation models (applied at every t):
9) COTS (strictly positive): lognormal errors
   log(cots_dat_{t}) ~ Normal( log(A_{t} + eps), sigma_cots )

10) Corals (bounded 0-100%): logit-normal on proportions
    logit(fast_dat_{t}/100) ~ Normal( logit(F_{t}/100), sigma_fast )
    logit(slow_dat_{t}/100) ~ Normal( logit(S_{t}/100), sigma_slow )
*/

template<class Type>
Type objective_function<Type>::operator() () {
  using namespace density; // not used, but available
  Type nll = 0.0;                                        // total negative log-likelihood
  const Type eps = Type(1e-8);                           // small constant for numerical stability
  const Type K = Type(100.0);                            // carrying capacity for total coral cover (%)

  // ------------------------------
  // DATA
  // ------------------------------
  DATA_VECTOR(Year);           // calendar years (integer values as Type), used for indexing/reporting
  DATA_VECTOR(sst_dat);        // Sea-Surface Temperature in Celsius (driver)
  DATA_VECTOR(cotsimm_dat);    // External larval immigration rate (individuals/m^2/year)
  DATA_VECTOR(cots_dat);       // Observed adult COTS density (individuals/m^2)
  DATA_VECTOR(fast_dat);       // Observed Acropora cover (%)
  DATA_VECTOR(slow_dat);       // Observed Faviidae/Porites cover (%)

  int T = Year.size();                                             // number of time steps
  // Ensure all vectors have consistent length
  // (Likelihood includes all observations regardless of values.)
  // Numerical safety: let T be the minimum of provided sizes (soft assumption is that they match).
  T = CppAD::CondExpLt(T, (int)sst_dat.size(), T, (int)sst_dat.size());          // keep minimal length
  T = CppAD::CondExpLt(T, (int)cotsimm_dat.size(), T, (int)cotsimm_dat.size());  // keep minimal length
  T = CppAD::CondExpLt(T, (int)cots_dat.size(), T, (int)cots_dat.size());        // keep minimal length
  T = CppAD::CondExpLt(T, (int)fast_dat.size(), T, (int)fast_dat.size());        // keep minimal length
  T = CppAD::CondExpLt(T, (int)slow_dat.size(), T, (int)slow_dat.size());        // keep minimal length

  // ------------------------------
  // PARAMETERS (transformed where needed; comments include units and sources)
  // ------------------------------
  PARAMETER(log_rA);           // log intrinsic adult COTS growth rate rA (year^-1); literature/expert, typical 0.05–1
  PARAMETER(log_cA);           // log density-dependence coefficient cA (m^2·ind^-1·year^-1, because A in ind/m^2); expert prior
  PARAMETER(log_phi_food);     // log coefficient for food feedback on COTS growth (unitless); controls strength of coral->COTS
  PARAMETER(logit_wF);         // logit of weight for fast coral in food mix (unitless); wF in (0,1), wS=1-wF

  PARAMETER(log_F_halfF);      // log half-saturation F_halfF for fast coral in food index (% cover); literature/expert
  PARAMETER(log_S_halfS);      // log half-saturation S_halfS for slow coral in food index (% cover); literature/expert

  PARAMETER(beta_env);         // coefficient for outbreak gate effect on COTS growth (unitless); should be >=0; expert prior
  PARAMETER(sst_opt);          // optimal SST for larval survival/outbreak (°C); literature ~ 27–30°C
  PARAMETER(log_sst_scale);    // log width (°C) of SST suitability curve; larger ⇒ broader peak
  PARAMETER(log_alpha_imm);    // log scaling of immigration in gate; converts ind/m^2/yr to unitless signal
  PARAMETER(log_k_gate);       // log steepness of outbreak gate logistic; larger ⇒ sharper threshold (unitless)
  PARAMETER(theta_outbreak);   // threshold for gate (unitless); sets combined env + immigration needed for outbreak

  PARAMETER(log_R_imm);        // log additive recruitment scaling from immigration to adults (ind/m^2/year)
  PARAMETER(log_k_imm_sat);    // log saturation coefficient for immigration addition vs adult density (m^2·ind^-1)

  PARAMETER(log_aF);           // log attack/search rate on fast corals (year^-1·(% cover)^-q); selectivity high for Acropora
  PARAMETER(log_aS);           // log attack/search rate on slow corals (year^-1·(% cover)^-q); lower than aF
  PARAMETER(log_h);            // log handling-time-like term (year); increases saturation at high prey
  PARAMETER(log_gamma_pred);   // log conversion efficiency from per-predator attack to fractional coral loss (unitless)

  PARAMETER(log_rF);           // log intrinsic growth rate of fast corals (year^-1)
  PARAMETER(log_rS);           // log intrinsic growth rate of slow corals (year^-1)
  PARAMETER(log_mF);           // log background mortality rate of fast corals (year^-1)
  PARAMETER(log_mS);           // log background mortality rate of slow corals (year^-1)
  PARAMETER(log_q_FR_minus1);  // log(q-1) for generalized functional response exponent; ensures q>=1

  PARAMETER(log_A0);           // log initial adult COTS density at first year (ind/m^2)
  PARAMETER(logit_F0p);        // logit of initial fast coral proportion (F0/K); F0 in (0,K)
  PARAMETER(logit_S0p);        // logit of initial slow coral proportion (S0/K); S0 in (0,K)

  PARAMETER(log_sigma_cots);   // log observation SD on log(COTS) (unitless)
  PARAMETER(log_sigma_fast);   // log observation SD on logit(fast%/100)
  PARAMETER(log_sigma_slow);   // log observation SD on logit(slow%/100)

  // Transforms to natural scales
  Type rA          = exp(log_rA);                          // year^-1
  Type cA          = exp(log_cA);                          // m^2·ind^-1·year^-1
  Type phi_food    = exp(log_phi_food);                    // unitless >= 0
  Type wF          = invlogit_safe(logit_wF);              // in (0,1)
  Type wS          = Type(1.0) - wF;                       // complement

  Type F_halfF     = exp(log_F_halfF);                     // % cover
  Type S_halfS     = exp(log_S_halfS);                     // % cover

  Type sst_scale   = exp(log_sst_scale);                   // °C
  Type alpha_imm   = exp(log_alpha_imm);                   // unitless
  Type k_gate      = exp(log_k_gate);                      // unitless
  Type R_imm       = exp(log_R_imm);                       // ind/m^2/year
  Type k_imm_sat   = exp(log_k_imm_sat);                   // m^2·ind^-1

  Type aF          = exp(log_aF);                          // year^-1·(% cover)^-q
  Type aS          = exp(log_aS);                          // year^-1·(% cover)^-q
  Type h           = exp(log_h);                           // year
  Type gamma_pred  = exp(log_gamma_pred);                  // unitless

  Type rF          = exp(log_rF);                          // year^-1
  Type rS          = exp(log_rS);                          // year^-1
  Type mF          = exp(log_mF);                          // year^-1
  Type mS          = exp(log_mS);                          // year^-1
  Type qFR         = Type(1.0) + exp(log_q_FR_minus1);     // >= 1

  Type A0          = exp(log_A0);                          // ind/m^2
  Type F0          = K * invlogit_safe(logit_F0p);         // % cover
  Type S0          = K * invlogit_safe(logit_S0p);         // % cover

  Type sigma_cots  = exp(log_sigma_cots);                  // SD on log scale
  Type sigma_fast  = exp(log_sigma_fast);                  // SD on logit scale
  Type sigma_slow  = exp(log_sigma_slow);                  // SD on logit scale

  // Observation variance floors for stability
  const Type sigma_cots_floor = Type(0.10);                // minimum SD on log scale
  const Type sigma_coral_floor = Type(0.05);               // minimum SD on logit scale

  sigma_cots = sigma_cots + sigma_cots_floor;              // enforce minimum SD
  sigma_fast = sigma_fast + sigma_coral_floor;             // enforce minimum SD
  sigma_slow = sigma_slow + sigma_coral_floor;             // enforce minimum SD

  // ------------------------------
  // State vectors and drivers
  // ------------------------------
  vector<Type> cots_pred(T);                               // adult COTS prediction (ind/m^2)
  vector<Type> fast_pred(T);                               // fast coral prediction (% cover)
  vector<Type> slow_pred(T);                               // slow coral prediction (% cover)

  vector<Type> env_temp(T);                                // environmental suitability (unitless)
  vector<Type> imm_signal(T);                              // immigration signal (unitless)
  vector<Type> gate(T);                                    // outbreak gate (unitless 0-1)

  // Initialize at first time step (no data leakage; predictions at time t use only previous states and exogenous drivers)
  cots_pred(0) = A0;                                       // initial adults
  fast_pred(0) = F0;                                       // initial fast coral
  slow_pred(0) = S0;                                       // initial slow coral

  // Precompute environmental drivers for all t (using available data at the same t)
  for (int t = 0; t < T; ++t) {
    Type sstd = sst_dat(t);                                // SST at time t (°C)
    Type dT   = (sstd - sst_opt) / (sst_scale + eps);      // scaled temperature deviation (unitless)
    env_temp(t) = exp( Type(-0.5) * dT * dT );             // Gaussian suitability in [0,1]
    imm_signal(t) = log( Type(1.0) + alpha_imm * (cotsimm_dat(t) + eps) );  // smooth, non-negative immigration signal
    gate(t) = invlogit_safe( k_gate * ( env_temp(t) + imm_signal(t) - theta_outbreak ) ); // outbreak gate 0..1
  }

  // Dynamics
  for (int t = 1; t < T; ++t) {
    // Previous states
    Type Aprev = cots_pred(t-1);                           // adults at t-1 (ind/m^2)
    Type Fprev = fast_pred(t-1);                           // fast coral at t-1 (%)
    Type Sprev = slow_pred(t-1);                           // slow coral at t-1 (%)

    // 4) Food feedback (saturating by guild; units unitless)
    Type food_fast = wF * ( Fprev / (F_halfF + Fprev + eps) );    // contribution from fast coral
    Type food_slow = wS * ( Sprev / (S_halfS + Sprev + eps) );    // contribution from slow coral
    Type food_idx  = food_fast + food_slow;                       // total food index in [0,1) approximately

    // 6) Predation functional response (generalized Holling)
    Type denom = Type(1.0) + h * ( aF * pow(Fprev + eps, qFR) + aS * pow(Sprev + eps, qFR) ); // denominator >= 1
    Type frac_loss_F = Type(1.0) - exp( -gamma_pred * Aprev * aF * pow(Fprev + eps, qFR) / (denom + eps) ); // [0,1)
    Type frac_loss_S = Type(1.0) - exp( -gamma_pred * Aprev * aS * pow(Sprev + eps, qFR) / (denom + eps) ); // [0,1)
    Type pred_loss_F = frac_loss_F * Fprev;                     // % cover removed from fast coral
    Type pred_loss_S = frac_loss_S * Sprev;                     // % cover removed from slow coral

    // 7) Coral logistic growth with space competition (K=100%)
    Type comp = (Fprev + Sprev) / K;                            // fraction of occupied space
    Type gF = rF * Fprev * ( Type(1.0) - comp );                // fast coral growth
    Type gS = rS * Sprev * ( Type(1.0) - comp );                // slow coral growth

    // Background mortalities
    Type mortF = mF * Fprev;                                    // fast coral mortality
    Type mortS = mS * Sprev;                                    // slow coral mortality

    // 8) Net coral change and smooth bounds in [0,K] via logit accumulation
    Type deltaF = gF - mortF - pred_loss_F;                     // net change (%)
    Type deltaS = gS - mortS - pred_loss_S;                     // net change (%)

    // Transform current % to proportions
    Type pF_prev = (Fprev / K);                                 // in [0,1]
    Type pS_prev = (Sprev / K);                                 // in [0,1]

    // Move in logit space by a scaled delta; ensures F,S remain in (0,K) without hard truncation
    Type logit_pF_next = logit_safe(pF_prev, eps) + (deltaF / K); // dimensionless increment
    Type logit_pS_next = logit_safe(pS_prev, eps) + (deltaS / K); // dimensionless increment

    // Explicit prediction equations required by validator:
    // Adult COTS (Ricker + food + environment + saturating immigration)
    Type growth_exponent = rA + phi_food * food_idx + beta_env * gate(t-1) - cA * Aprev; // net per-capita growth
    cots_pred(t) = Aprev * exp(growth_exponent)                                             // multiplicative update
                   + R_imm * imm_signal(t-1) / ( Type(1.0) + k_imm_sat * Aprev );          // additive immigration
    cots_pred(t) = cots_pred(t) + eps;                                                      // avoid exact zeros

    // Fast and slow corals: bounded logistic updates (space-limited) minus predation and mortality
    fast_pred(t) = K * invlogit_safe( logit_pF_next );              // next % cover for fast coral
    slow_pred(t) = K * invlogit_safe( logit_pS_next );              // next % cover for slow coral
  }

  // ------------------------------
  // Observation likelihood
  // ------------------------------
  for (int t = 0; t < T; ++t) {
    // 9) COTS: lognormal
    Type mu_logA = log(cots_pred(t) + eps);                     // mean on log scale
    nll -= dnorm( log(cots_dat(t) + eps), mu_logA, sigma_cots, true ); // include all observations

    // 10) Corals: logit-normal on proportions
    Type pF_obs = (fast_dat(t) / K);                            // observed proportion
    Type pS_obs = (slow_dat(t) / K);                            // observed proportion
    Type pF_pr = (fast_pred(t) / K);                            // predicted proportion
    Type pS_pr = (slow_pred(t) / K);                            // predicted proportion

    Type mu_logitF = logit_safe(pF_pr, eps);                    // mean on logit scale
    Type mu_logitS = logit_safe(pS_pr, eps);                    // mean on logit scale

    nll -= dnorm( logit_safe(pF_obs, eps), mu_logitF, sigma_fast, true ); // fast coral likelihood
    nll -= dnorm( logit_safe(pS_obs, eps), mu_logitS, sigma_slow, true ); // slow coral likelihood
  }

  // ------------------------------
  // Smooth parameter penalties (soft bounds; add to nll)
  // ------------------------------
  auto sqr = [&](Type x){ return x*x; };                        // local square helper

  // Encourage sst_opt within [24, 32] °C with gentle quadratic penalties
  if (CppAD::Var2Par(sst_opt) < Type(24.0)) nll += sqr( (Type(24.0) - sst_opt) / Type(2.0) );
  if (CppAD::Var2Par(sst_opt) > Type(32.0)) nll += sqr( (sst_opt - Type(32.0)) / Type(2.0) );

  // Encourage theta_outbreak within [-2, 2]
  if (CppAD::Var2Par(theta_outbreak) < Type(-2.0)) nll += sqr( (Type(-2.0) - theta_outbreak) / Type(1.0) );
  if (CppAD::Var2Par(theta_outbreak) > Type( 2.0)) nll += sqr( (theta_outbreak - Type( 2.0)) / Type(1.0) );

  // Encourage beta_env >= 0 (soft)
  if (CppAD::Var2Par(beta_env) < Type(0.0)) nll += sqr( (Type(0.0) - beta_env) / Type(0.5) );

  // Encourage initial coral sum <= 95% (avoid infeasible overcrowding at start)
  Type init_sum = fast_pred(0) + slow_pred(0);
  if (CppAD::Var2Par(init_sum) > Type(95.0)) nll += sqr( (init_sum - Type(95.0)) / Type(5.0) );

  // ------------------------------
  // REPORTS
  // ------------------------------
  REPORT(cots_pred);               // adult COTS predictions (ind/m^2)
  REPORT(fast_pred);               // fast coral predictions (%)
  REPORT(slow_pred);               // slow coral predictions (%)

  // Additional diagnostics (environmental drivers and gate)
  REPORT(env_temp);                // environmental suitability time series
  REPORT(imm_signal);              // immigration signal time series
  REPORT(gate);                    // outbreak gate time series
  REPORT(Year);                    // echo back time axis

  // Also provide ADREPORT for key parameters to obtain SEs
  ADREPORT(wF);                    // preference for fast coral (unitless)
  ADREPORT(rA);                    // intrinsic adult growth (year^-1)
  ADREPORT(rF);                    // fast coral growth (year^-1)
  ADREPORT(rS);                    // slow coral growth (year^-1)
  ADREPORT(qFR);                   // functional response exponent

  return nll;                      // return total negative log-likelihood
}
