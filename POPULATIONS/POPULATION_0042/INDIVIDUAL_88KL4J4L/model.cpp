#include <TMB.hpp>

// Helper functions with numerical safeguards
template<class Type>
Type invlogit_safe(Type x) { // numerically stable inverse-logit
  return Type(1.0) / (Type(1.0) + exp(-x));
}

template<class Type>
Type logit_safe(Type p, Type eps) { // safe logit for proportions in (0,1)
  p = CppAD::CondExpLt(p, eps, eps, p);
  p = CppAD::CondExpGt(p, Type(1.0) - eps, Type(1.0) - eps, p);
  return log(p / (Type(1.0) - p));
}

template<class Type>
Type softplus(Type x, Type k) { // smooth ReLU approximation; k sets smoothness
  return k * log(Type(1.0) + exp(x / k));
}

template<class Type>
Type logistic(Type x) { // standard logistic
  return Type(1.0) / (Type(1.0) + exp(-x));
}

template<class Type>
Type square(Type x) { return x * x; }

// Smooth penalty to softly bound parameters within [lo,hi]
template<class Type>
Type bound_penalty(Type x, Type lo, Type hi, Type scale) {
  Type pen = Type(0.0);
  if (CppAD::isfinite(Value(lo))) {
    pen += square( softplus(lo - x, scale) );
  }
  if (CppAD::isfinite(Value(hi))) {
    pen += square( softplus(x - hi, scale) );
  }
  return pen;
}

/*
Numbered model equations (all transitions are smooth; t indexes Year):
1) Food saturation (composite prey availability):
   Fidx_{t-1} = wF * Fast_{t-1} + wS * Slow_{t-1}
   satF_{t-1} = Fidx_{t-1} / (K_food + Fidx_{t-1})

2) Temperature performance (Gaussian thermal niche):
   EnvCOTS_{t-1} = exp( -0.5 * ((SST_{t-1} - Topt_COTS)/sigma_T_COTS)^2 )

3) Smooth Allee effect on COTS growth:
   Allee_{t-1} = A_max * ( logistic( (N_{t-1} - N_Allee) / sA ) - logistic( -N_Allee / sA ) )

4) COTS log-density update (Ricker/Gompertz hybrid with immigration):
   g_{t-1} = r0_COTS + b_food*satF_{t-1} + log(EnvCOTS_{t-1} + eps) - beta_dd * N_{t-1}
             + Allee_{t-1} + log( 1 + imm_eff * COTSImm_{t-1} + eps )
   logN_t ~ Normal( logN_{t-1} + g_{t-1}, sigma_proc_N )

5) Multi-prey Holling type II functional response (per-predator, per year):
   denom_{t-1} = 1 + aF * hF * Fast_{t-1} + aS * hS * Slow_{t-1}
   consF_{t-1} = e_pred_F * N_{t-1} * (aF * Fast_{t-1} / denom_{t-1})
   consS_{t-1} = e_pred_S * N_{t-1} * (aS * Slow_{t-1} / denom_{t-1})

6) Coral growth with space competition and thermal modulation:
   gF_{t-1} = r_fast * exp( -0.5 * ((SST_{t-1} - Topt_fast)/sigmaT_fast)^2 )
   gS_{t-1} = r_slow * exp( -0.5 * ((SST_{t-1} - Topt_slow)/sigmaT_slow)^2 )
   dF_growth = gF_{t-1} * Fast_{t-1} * ( 1 - (Fast_{t-1} + phi_FS * Slow_{t-1}) / (K_fast + eps) )
   dS_growth = gS_{t-1} * Slow_{t-1} * ( 1 - (Slow_{t-1} + phi_SF * Fast_{t-1}) / (K_slow + eps) )

7) Smooth bleaching mortality (soft threshold):
   mBleach_F = mB_fast * logistic( (SST_{t-1} - T_bleach_fast) / sTb ) * Fast_{t-1}
   mBleach_S = mB_slow * logistic( (SST_{t-1} - T_bleach_slow) / sTb ) * Slow_{t-1}

8) Coral logit-state update (Euler increment on logit scale):
   xiF_t ~ Normal( xiF_{t-1} + ( dF_growth - consF_{t-1} - mBleach_F )
                              / ( Fast_{t-1} * (1 - Fast_{t-1}) + eps ),
                   sigma_proc_F )
   xiS_t ~ Normal( xiS_{t-1} + ( dS_growth - consS_{t-1} - mBleach_S )
                              / ( Slow_{t-1} * (1 - Slow_{t-1}) + eps ),
                   sigma_proc_S )

Observation model (all times t contribute):
9) COTS (strictly positive): lognormal
   log(cots_dat_t) ~ Normal( log(cots_pred_t), sigma_obs_cots )

10) Coral cover (bounded): logit-normal on proportions
    logit(fast_dat_t / 100) ~ Normal( xiF_t, sigma_obs_fast )
    logit(slow_dat_t / 100) ~ Normal( xiS_t, sigma_obs_slow )
*/

template<class Type>
Type objective_function<Type>::operator() () {
  using namespace density;
  Type eps = Type(1e-8);          // small constant for numerical stability
  Type pen_scale = Type(0.05);    // smoothness for softplus in penalties
  Type pen_weight = Type(1.0);    // weight for parameter bound penalties
  Type min_sd = Type(0.05);       // minimum SD to avoid overconfident fits

  // DATA INPUTS (names match data files)
  DATA_VECTOR(Year);           // Year (calendar year), used for alignment/reporting
  DATA_VECTOR(cots_dat);       // Adult COTS density (ind m^-2)
  DATA_VECTOR(fast_dat);       // Fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);       // Slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);        // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);    // COTS larval immigration (ind m^-2 yr^-1)

  int T = Year.size();

  // STATE VECTORS (random effects)
  PARAMETER_VECTOR(logN_state);  // log adult COTS density state over time (log(ind m^-2))
  PARAMETER_VECTOR(xiF_state);   // logit of fast coral cover proportion over time
  PARAMETER_VECTOR(xiS_state);   // logit of slow coral cover proportion over time

  // INITIAL STATES
  PARAMETER(logN0);  // initial log COTS density at t=0 (log(ind m^-2)); from early-year data or expert opinion
  PARAMETER(xiF0);   // initial logit fast coral proportion at t=0; from early percent data /100 via logit
  PARAMETER(xiS0);   // initial logit slow coral proportion at t=0; from early percent data /100 via logit

  // COTS DYNAMICS PARAMETERS
  PARAMETER(r0_COTS);        // baseline per-capita log growth of COTS (yr^-1); literature/expert
  PARAMETER(b_food);         // strength of food saturation effect on COTS growth (dimensionless); to be estimated
  PARAMETER(K_food);         // half-saturation for food index (proportion of reef area); from feeding studies
  PARAMETER(Topt_COTS);      // temperature optimum for COTS performance (°C); literature
  PARAMETER(sigma_T_COTS);   // thermal niche width for COTS (°C); literature
  PARAMETER(beta_dd);        // density-dependence strength (m^2 ind^-1 yr^-1) in Ricker term; estimated
  PARAMETER(A_max);          // maximum Allee compensation (dimensionless log-growth units); estimated
  PARAMETER(N_Allee);        // Allee threshold density (ind m^-2); expert/literature
  PARAMETER(sA);             // smoothness (ind m^-2) of Allee logistic transition; estimated
  PARAMETER(imm_eff);        // scaling of immigration contribution (m^2 ind^-1) inside log(1+imm_eff*imm); estimated

  // FOOD COMPOSITION WEIGHTS (transformed to sum to 1 via softmax)
  PARAMETER(wf_raw);         // raw weight for fast coral in food index (unbounded); estimated
  PARAMETER(ws_raw);         // raw weight for slow coral in food index (unbounded); estimated

  // CORAL GROWTH AND COMPETITION
  PARAMETER(r_fast);         // intrinsic growth rate of fast coral (yr^-1); literature/estimated
  PARAMETER(r_slow);         // intrinsic growth rate of slow coral (yr^-1); literature/estimated
  PARAMETER(K_fast);         // carrying capacity (proportion) for fast coral; <= 1; expert
  PARAMETER(K_slow);         // carrying capacity (proportion) for slow coral; <= 1; expert
  PARAMETER(phi_FS);         // competition coefficient: effect of slow on fast (dimensionless); estimated
  PARAMETER(phi_SF);         // competition coefficient: effect of fast on slow (dimensionless); estimated

  // PREDATION FUNCTIONAL RESPONSE (multi-prey Holling II)
  PARAMETER(aF);             // attack rate on fast coral (yr^-1); higher than aS typically; estimated
  PARAMETER(aS);             // attack rate on slow coral (yr^-1); estimated
  PARAMETER(hF);             // handling time on fast coral (yr); estimated
  PARAMETER(hS);             // handling time on slow coral (yr); estimated
  PARAMETER(e_pred_F);       // conversion from per-predator attack to fast coral cover loss (proportion per predator-year); estimated
  PARAMETER(e_pred_S);       // conversion from per-predator attack to slow coral cover loss (proportion per predator-year); estimated

  // TEMPERATURE EFFECTS ON CORALS AND BLEACHING
  PARAMETER(Topt_fast);      // temperature optimum for fast coral growth (°C); literature
  PARAMETER(Topt_slow);      // temperature optimum for slow coral growth (°C); literature
  PARAMETER(sigmaT_fast);    // thermal niche width for fast coral (°C); literature
  PARAMETER(sigmaT_slow);    // thermal niche width for slow coral (°C); literature
  PARAMETER(T_bleach_fast);  // bleaching onset temperature for fast coral (°C); literature
  PARAMETER(T_bleach_slow);  // bleaching onset temperature for slow coral (°C); literature
  PARAMETER(mB_fast);        // max annual bleaching mortality rate fraction for fast coral (yr^-1); estimated
  PARAMETER(mB_slow);        // max annual bleaching mortality rate fraction for slow coral (yr^-1); estimated
  PARAMETER(sTb);            // smoothness (°C) of bleaching logistic; estimated

  // PROCESS AND OBSERVATION VARIANCES (on log or linear scale as noted)
  PARAMETER(log_sigma_proc_N); // log SD of COTS process noise; estimated
  PARAMETER(log_sigma_proc_F); // log SD of fast coral process noise (on xi scale); estimated
  PARAMETER(log_sigma_proc_S); // log SD of slow coral process noise (on xi scale); estimated

  PARAMETER(log_sigma_obs_cots); // log SD for lognormal observation of COTS; estimated
  PARAMETER(sigma_obs_fast);     // SD for logit-normal observation of fast coral; estimated
  PARAMETER(sigma_obs_slow);     // SD for logit-normal observation of slow coral; estimated

  // Derived positive SDs with minimum floors
  Type sigma_proc_N = exp(log_sigma_proc_N);
  Type sigma_proc_F = exp(log_sigma_proc_F);
  Type sigma_proc_S = exp(log_sigma_proc_S);
  Type sigma_obs_cots = exp(log_sigma_obs_cots);

  // Enforce minimum SDs smoothly
  sigma_proc_N = sqrt(sigma_proc_N * sigma_proc_N + min_sd * min_sd);
  sigma_proc_F = sqrt(sigma_proc_F * sigma_proc_F + min_sd * min_sd);
  sigma_proc_S = sqrt(sigma_proc_S * sigma_proc_S + min_sd * min_sd);
  sigma_obs_cots = sqrt(sigma_obs_cots * sigma_obs_cots + min_sd * min_sd);
  Type sigma_obs_fast_eff = sqrt(sigma_obs_fast * sigma_obs_fast + min_sd * min_sd);
  Type sigma_obs_slow_eff = sqrt(sigma_obs_slow * sigma_obs_slow + min_sd * min_sd);

  // Transform weights via softmax to ensure positivity and sum-to-one
  Type wF = exp(wf_raw);
  Type wS = exp(ws_raw);
  Type wsum = wF + wS + eps;
  wF = wF / wsum;
  wS = wS / wsum;

  // REPORT transformed weights for interpretation
  ADREPORT(wF);
  ADREPORT(wS);

  // Soft parameter bounds penalties (ecologically meaningful ranges)
  Type nll = Type(0.0);
  nll += pen_weight * bound_penalty(r0_COTS, Type(-1.0), Type(3.0), pen_scale);
  nll += pen_weight * bound_penalty(b_food, Type(0.0), Type(3.0), pen_scale);
  nll += pen_weight * bound_penalty(K_food, Type(0.01), Type(1.0), pen_scale);
  nll += pen_weight * bound_penalty(Topt_COTS, Type(20.0), Type(33.0), pen_scale);
  nll += pen_weight * bound_penalty(sigma_T_COTS, Type(0.3), Type(5.0), pen_scale);
  nll += pen_weight * bound_penalty(beta_dd, Type(0.0), Type(10.0), pen_scale);
  nll += pen_weight * bound_penalty(A_max, Type(0.0), Type(3.0), pen_scale);
  nll += pen_weight * bound_penalty(N_Allee, Type(0.0), Type(2.0), pen_scale);
  nll += pen_weight * bound_penalty(sA, Type(0.01), Type(2.0), pen_scale);
  nll += pen_weight * bound_penalty(imm_eff, Type(0.0), Type(50.0), pen_scale);

  nll += pen_weight * bound_penalty(r_fast, Type(0.0), Type(3.0), pen_scale);
  nll += pen_weight * bound_penalty(r_slow, Type(0.0), Type(1.0), pen_scale);
  nll += pen_weight * bound_penalty(K_fast, Type(0.05), Type(0.95), pen_scale);
  nll += pen_weight * bound_penalty(K_slow, Type(0.05), Type(0.95), pen_scale);
  nll += pen_weight * bound_penalty(phi_FS, Type(0.0), Type(2.0), pen_scale);
  nll += pen_weight * bound_penalty(phi_SF, Type(0.0), Type(2.0), pen_scale);

  nll += pen_weight * bound_penalty(aF, Type(0.0), Type(20.0), pen_scale);
  nll += pen_weight * bound_penalty(aS, Type(0.0), Type(20.0), pen_scale);
  nll += pen_weight * bound_penalty(hF, Type(0.0), Type(10.0), pen_scale);
  nll += pen_weight * bound_penalty(hS, Type(0.0), Type(10.0), pen_scale);
  nll += pen_weight * bound_penalty(e_pred_F, Type(0.0), Type(5.0), pen_scale);
  nll += pen_weight * bound_penalty(e_pred_S, Type(0.0), Type(5.0), pen_scale);

  nll += pen_weight * bound_penalty(Topt_fast, Type(20.0), Type(33.0), pen_scale);
  nll += pen_weight * bound_penalty(Topt_slow, Type(20.0), Type(33.0), pen_scale);
  nll += pen_weight * bound_penalty(sigmaT_fast, Type(0.3), Type(5.0), pen_scale);
  nll += pen_weight * bound_penalty(sigmaT_slow, Type(0.3), Type(5.0), pen_scale);
  nll += pen_weight * bound_penalty(T_bleach_fast, Type(26.0), Type(34.0), pen_scale);
  nll += pen_weight * bound_penalty(T_bleach_slow, Type(26.0), Type(36.0), pen_scale);
  nll += pen_weight * bound_penalty(mB_fast, Type(0.0), Type(2.0), pen_scale);
  nll += pen_weight * bound_penalty(mB_slow, Type(0.0), Type(2.0), pen_scale);
  nll += pen_weight * bound_penalty(sTb, Type(0.1), Type(3.0), pen_scale);

  nll += pen_weight * bound_penalty(log_sigma_proc_N, Type(-7.0), Type(2.0), pen_scale);
  nll += pen_weight * bound_penalty(log_sigma_proc_F, Type(-7.0), Type(2.0), pen_scale);
  nll += pen_weight * bound_penalty(log_sigma_proc_S, Type(-7.0), Type(2.0), pen_scale);
  nll += pen_weight * bound_penalty(log_sigma_obs_cots, Type(-7.0), Type(2.0), pen_scale);
  nll += pen_weight * bound_penalty(sigma_obs_fast, Type(0.02), Type(2.0), pen_scale);
  nll += pen_weight * bound_penalty(sigma_obs_slow, Type(0.02), Type(2.0), pen_scale);

  nll += pen_weight * bound_penalty(logN0, Type(-5.0), Type(3.0), pen_scale);
  nll += pen_weight * bound_penalty(xiF0, Type(-4.0), Type(4.0), pen_scale);
  nll += pen_weight * bound_penalty(xiS0, Type(-4.0), Type(4.0), pen_scale);

  // Predictions to REPORT (aligned with data)
  vector<Type> cots_dat_pred(T); // predicted COTS density (ind m^-2)
  vector<Type> fast_dat_pred(T); // predicted fast coral cover (%)
  vector<Type> slow_dat_pred(T); // predicted slow coral cover (%)

  // Process model likelihood
  for (int t = 0; t < T; t++) {
    // previous states (t-1), using initial states for t==0
    Type logN_prev = (t == 0) ? logN0 : logN_state(t - 1);
    Type xiF_prev  = (t == 0) ? xiF0  : xiF_state(t - 1);
    Type xiS_prev  = (t == 0) ? xiS0  : xiS_state(t - 1);

    // transform to natural scales
    Type N_prev = exp(logN_prev);               // ind m^-2
    Type F_prev = invlogit_safe(xiF_prev);      // proportion
    Type S_prev = invlogit_safe(xiS_prev);      // proportion

    // covariates (do NOT use current observations of response variables)
    Type sst = sst_dat(t);
    Type cimm = cotsimm_dat(t);

    // (1) Food saturation
    Type Fidx = wF * F_prev + wS * S_prev;
    Type satF = Fidx / (K_food + Fidx + eps);

    // (2) Temperature performance for COTS
    Type envC = exp( - Type(0.5) * square( (sst - Topt_COTS) / (sigma_T_COTS + eps) ) );

    // (3) Smooth Allee effect
    Type allee_center = logistic( - N_Allee / (sA + eps) ); // constant center
    Type allee_term = A_max * ( logistic( (N_prev - N_Allee) / (sA + eps) ) - allee_center );

    // (4) COTS log growth increment with density dependence and immigration
    Type g = r0_COTS
           + b_food * satF
           + log(envC + eps)
           - beta_dd * N_prev
           + allee_term
           + log( Type(1.0) + imm_eff * cimm + eps );

    Type logN_pred = logN_prev + g;

    // (5) Multi-prey Holling type II predation (per predator)
    Type denom = Type(1.0) + aF * hF * F_prev + aS * hS * S_prev;
    Type consF = e_pred_F * N_prev * (aF * F_prev / (denom + eps)); // proportion per year
    Type consS = e_pred_S * N_prev * (aS * S_prev / (denom + eps)); // proportion per year

    // (6) Coral growth with competition and thermal modulation
    Type gF = r_fast * exp( - Type(0.5) * square( (sst - Topt_fast) / (sigmaT_fast + eps) ) );
    Type gS = r_slow * exp( - Type(0.5) * square( (sst - Topt_slow) / (sigmaT_slow + eps) ) );
    Type dF_growth = gF * F_prev * ( Type(1.0) - (F_prev + phi_FS * S_prev) / (K_fast + eps) );
    Type dS_growth = gS * S_prev * ( Type(1.0) - (S_prev + phi_SF * F_prev) / (K_slow + eps) );

    // (7) Smooth bleaching mortality
    Type mBleach_F = mB_fast * logistic( (sst - T_bleach_fast) / (sTb + eps) ) * F_prev;
    Type mBleach_S = mB_slow * logistic( (sst - T_bleach_slow) / (sTb + eps) ) * S_prev;

    // (8) Logit-scale coral updates (Euler approximation)
    Type xiF_pred = xiF_prev + ( dF_growth - consF - mBleach_F ) / ( F_prev * (Type(1.0) - F_prev) + eps );
    Type xiS_pred = xiS_prev + ( dS_growth - consS - mBleach_S ) / ( S_prev * (Type(1.0) - S_prev) + eps );

    // Process likelihood contributions
    nll -= dnorm(logN_state(t), logN_pred, sigma_proc_N, true);
    nll -= dnorm(xiF_state(t),  xiF_pred,  sigma_proc_F, true);
    nll -= dnorm(xiS_state(t),  xiS_pred,  sigma_proc_S, true);

    // Predictions on data scales
    Type N_t = exp(logN_state(t));
    Type F_t = invlogit_safe(xiF_state(t));
    Type S_t = invlogit_safe(xiS_state(t));

    cots_dat_pred(t) = N_t;            // ind m^-2
    fast_dat_pred(t) = F_t * Type(100.0); // %
    slow_dat_pred(t) = S_t * Type(100.0); // %
  }

  // Observation likelihoods (all t)
  for (int t = 0; t < T; t++) {
    // COTS: lognormal likelihood
    Type y_cots = cots_dat(t);
    y_cots = CppAD::CondExpLt(y_cots, eps, eps, y_cots);
    Type mu_cots = log(cots_dat_pred(t) + eps);
    nll -= dnorm(log(y_cots), mu_cots, sigma_obs_cots, true);

    // Coral: logit-normal likelihoods on proportions
    Type y_fast_prop = (fast_dat(t) / Type(100.0));
    Type y_slow_prop = (slow_dat(t) / Type(100.0));
    Type z_fast_obs = logit_safe(y_fast_prop, eps);
    Type z_slow_obs = logit_safe(y_slow_prop, eps);

    nll -= dnorm(z_fast_obs, xiF_state(t), sigma_obs_fast_eff, true);
    nll -= dnorm(z_slow_obs, xiS_state(t), sigma_obs_slow_eff, true);
  }

  // REPORT predictions and helpful diagnostics
  REPORT(Year);
  REPORT(cots_dat_pred);
  REPORT(fast_dat_pred);
  REPORT(slow_dat_pred);

  // Additional reports for interpretation (not required by spec but useful)
  ADREPORT(r0_COTS);
  ADREPORT(b_food);
  ADREPORT(beta_dd);
  ADREPORT(A_max);
  ADREPORT(N_Allee);
  ADREPORT(imm_eff);
  ADREPORT(r_fast);
  ADREPORT(r_slow);
  ADREPORT(aF);
  ADREPORT(aS);

  return nll;
}
