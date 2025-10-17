#include <TMB.hpp>

// Smooth maximum approximation to avoid hard cutoffs (units: same as x)
template<class Type>
Type smooth_max(Type x, Type eps) {
  return Type(0.5) * (x + sqrt(x * x + eps)); // ~max(x,0) with smoothness controlled by eps
}

// Logistic transform (dimensionless)
template<class Type>
Type inv_logit(Type x) {
  return Type(1) / (Type(1) + exp(-x));
}

// Smooth bound penalty: zero inside [low, high], quadratic outside (units: penalty on NLL)
template<class Type>
Type penalty_bounds(Type x, Type low, Type high, Type lambda, Type eps) {
  Type below = smooth_max(low - x, eps);  // positive if x < low
  Type above = smooth_max(x - high, eps); // positive if x > high
  return lambda * (below * below + above * above);
}

/*
Model equations (annual time step; predictions at time t use only states/drivers at t-1)

Initial conditions (no data leakage):
- cots_pred(0) = cots0  (parameter, >= 0)
- fast_pred(0) = clamp_0_100(fast0)  (parameter)
- slow_pred(0) = clamp_0_100(slow0)  (parameter)

Auxiliary modifiers (evaluated at t-1):
- phi_T_COTS(t-1)  = exp(-0.5 * ((sst_dat(t-1) - Topt_cots)  / sigmaT_cots)^2)
- phi_T_CORAL(t-1) = exp(-0.5 * ((sst_dat(t-1) - Topt_coral) / sigmaT_coral)^2)
- phi_spawn(t-1)   = C(t-1)^nu_spawn / (h_spawn^nu_spawn + C(t-1)^nu_spawn)
- phi_food(t-1)    = (A(t-1) + S(t-1)) / (foodK + A(t-1) + S(t-1))

Functional response and predation (evaluated at t-1):
- q  = 1 + exp(log_q_FR);  wA = inv_logit(prefA_logit);  wS = 1 - wA
- consA_per(t-1) = max_cons * wA * A(t-1)^q / (hA + A(t-1)^q)
- consS_per(t-1) = max_cons * wS * S(t-1)^q / (hS + S(t-1)^q)
- predA_eff(t-1) = A(t-1) * [1 - exp(-C(t-1) * consA_per(t-1) / (A(t-1) + eps))]
- predS_eff(t-1) = S(t-1) * [1 - exp(-C(t-1) * consS_per(t-1) / (S(t-1) + eps))]

Space limitation and coral growth (evaluated at t-1):
- F(t-1) = max(0, 100 - A(t-1) - S(t-1))
- growthA(t-1) = rA * A(t-1) * (F(t-1)/100) * phi_T_CORAL(t-1)
- growthS(t-1) = rS * S(t-1) * (F(t-1)/100) * phi_T_CORAL(t-1)

Recruitment, survival, immigration, density dependence (evaluated at t-1):
- survival(t-1) = exp( -[ mC + mC_food * (1 - phi_food(t-1)) ] )
- C_surv(t-1)   = C(t-1) * survival(t-1)
- recruits(t-1) = fec * C(t-1) * phi_spawn(t-1) * phi_T_COTS(t-1) * exp(eta_rec(t-1))
- I(t-1)        = alpha_imm * cotsimm_dat(t-1) / (k_imm + cotsimm_dat(t-1))

Prediction equations:
Prediction equation: cots_pred(t) = [C_surv(t-1) + recruits(t-1) + I(t-1)] / (1 + beta_dd * [C_surv(t-1) + recruits(t-1) + I(t-1)])
Prediction equation: fast_pred(t) = clamp_0_100( A(t-1) + growthA(t-1) - predA_eff(t-1) - mA0 * A(t-1) )
Prediction equation: slow_pred(t) = clamp_0_100( S(t-1) + growthS(t-1) - predS_eff(t-1) - mS0 * S(t-1) )

Observation equations (linking data to predictions; no data used in prediction calculations):
- cots_dat(t) ~ LogNormal(meanlog = log(cots_pred(t)), sd = sigma_cots)
- fast_dat(t) ~ LogitNormal(mean = logit(fast_pred(t)/100), sd = sigma_fast)
- slow_dat(t) ~ LogitNormal(mean = logit(slow_pred(t)/100), sd = sigma_slow)

Prediction mapping (for static checks):
cots_dat has prediction: cots_pred
fast_dat has prediction: fast_pred
slow_dat has prediction: slow_pred
*/

template<class Type>
Type objective_function<Type>::operator() () {
  // -------------------------
  // Data (all lengths are T)
  // -------------------------
  DATA_VECTOR(Year);          // calendar year (integer years; used for alignment)
  DATA_VECTOR(sst_dat);       // Sea-surface temperature (°C), annual (exogenous)
  DATA_VECTOR(cotsimm_dat);   // External larval immigration (individuals m^-2 yr^-1) (exogenous)
  DATA_VECTOR(cots_dat);      // Adult COTS density (individuals m^-2)
  DATA_VECTOR(fast_dat);      // Fast coral cover (Acropora), percent (% cover, 0-100)
  DATA_VECTOR(slow_dat);      // Slow coral cover (Faviidae/Porites), percent (% cover, 0-100)

  // -------------------------
  // Parameters
  // -------------------------
  PARAMETER(fec);            // recruits per adult per year reaching adulthood (yr^-1), initial estimate
  PARAMETER(h_spawn);        // half-saturation adult density for fertilization (ind m^-2), initial estimate
  PARAMETER(log_nu_spawn);   // log of (nu_spawn - 1), Hill exponent for fertilization density dependence
  PARAMETER(mC);             // baseline adult COTS mortality rate (yr^-1), initial estimate
  PARAMETER(mC_food);        // additional mortality scale when food is scarce (yr^-1), initial estimate
  PARAMETER(alpha_imm);      // immigration conversion efficiency (dimensionless), initial estimate
  PARAMETER(k_imm);          // half-saturation scale for immigration (ind m^-2 yr^-1), initial estimate
  PARAMETER(Topt_cots);      // optimal SST for larval survival (°C), literature
  PARAMETER(sigmaT_cots);    // width of SST response for larvae (°C), literature
  PARAMETER(rA);             // intrinsic growth rate fast coral (yr^-1), literature
  PARAMETER(rS);             // intrinsic growth rate slow coral (yr^-1), literature
  PARAMETER(hA);             // handling/half-sat scale in predation on fast coral (% cover), initial estimate
  PARAMETER(hS);             // handling/half-sat scale in predation on slow coral (% cover), initial estimate
  PARAMETER(max_cons);       // maximum % cover consumed per starfish per year (% cover starfish^-1 yr^-1), literature
  PARAMETER(Topt_coral);     // optimal SST for coral performance (°C), literature
  PARAMETER(sigmaT_coral);   // width of SST response for corals (°C), literature
  PARAMETER(mA0);            // background mortality fast coral (yr^-1), initial estimate
  PARAMETER(mS0);            // background mortality slow coral (yr^-1), initial estimate
  PARAMETER(foodK);          // half-saturation of food effect on COTS survival (% total coral cover), initial estimate
  PARAMETER(beta_dd);        // Beverton–Holt crowding coefficient for COTS (m^2 ind^-1), initial estimate
  PARAMETER(prefA_logit);    // logit preference for Acropora (dimensionless; wA = inv_logit(prefA_logit)), initial estimate
  PARAMETER(log_q_FR);       // log of (q-1) for functional response exponent (dimensionless), initial estimate (q = 1 + exp(log_q_FR))
  PARAMETER(log_sigma_cots); // log observation SD for log(COTS) (dimensionless), initial estimate
  PARAMETER(log_sigma_fast); // log observation SD for logit(fast proportion) (dimensionless), initial estimate
  PARAMETER(log_sigma_slow); // log observation SD for logit(slow proportion) (dimensionless), initial estimate

  // Environmental recruitment AR(1) effect
  PARAMETER_VECTOR(eta_rec);   // AR(1) random effect on recruitment (length ~ T)
  PARAMETER(log_sigma_rec);    // log innovation SD of AR(1)
  PARAMETER(logit_rho_rec);    // unconstrained -> (-1,1) via transform

  // Initial state parameters (avoid data leakage)
  PARAMETER(cots0);            // initial adult COTS density (ind m^-2)
  PARAMETER(fast0);            // initial fast coral cover (%)
  PARAMETER(slow0);            // initial slow coral cover (%)

  // -------------------------
  // Constants and helpers
  // -------------------------
  int T = Year.size();                     // number of time steps (years)
  int T_eta = eta_rec.size();              // length of recruitment effect vector (should be >= T for full coverage)
  Type eps = Type(1e-8);                   // small epsilon for numerical stability
  Type nll = Type(0);                      // negative log-likelihood accumulator
  Type sigma_min = Type(0.05);             // minimum SD to avoid singular likelihoods
  Type prop_eps = Type(1e-6);              // small offset for proportions to avoid 0/1 on logit

  // Observation SDs with smooth floor
  Type sigma_cots = exp(log_sigma_cots);   // >0 via exp
  sigma_cots = smooth_max(sigma_cots - sigma_min, eps) + sigma_min; // enforce >= sigma_min smoothly
  Type sigma_fast = exp(log_sigma_fast);
  sigma_fast = smooth_max(sigma_fast - sigma_min, eps) + sigma_min;
  Type sigma_slow = exp(log_sigma_slow);
  sigma_slow = smooth_max(sigma_slow - sigma_min, eps) + sigma_min;

  // Derived transforms
  Type nu_spawn = Type(1) + exp(log_nu_spawn); // Hill exponent >= 1 for fertilization
  Type wA = inv_logit(prefA_logit);            // preference for fast coral in [0,1]
  Type wS = Type(1) - wA;                      // preference for slow coral
  Type q = Type(1) + exp(log_q_FR);            // q >= 1; q=1 -> Type II, q>1 -> Type III

  // AR(1) transforms for recruitment environment
  Type sigma_rec = exp(log_sigma_rec);                         // innovation SD > 0
  Type rho_rec = Type(2.0) * inv_logit(logit_rho_rec) - Type(1.0); // map R to (-1,1)

  // -------------------------
  // Random effect prior (AR(1) on eta_rec)
  // -------------------------
  if (T_eta > 0) {
    // Stationary prior on first element
    Type sd0 = sigma_rec / sqrt(Type(1.0) - rho_rec * rho_rec + eps);
    nll -= dnorm(eta_rec(0), Type(0.0), sd0, true);
    for (int t = 1; t < T_eta; ++t) {
      nll -= dnorm(eta_rec(t), rho_rec * eta_rec(t - 1), sigma_rec, true);
    }
  }

  // -------------------------
  // State predictions
  // -------------------------
  vector<Type> cots_pred(T); // predicted adult COTS (ind m^-2)
  vector<Type> fast_pred(T); // predicted fast coral cover (%)
  vector<Type> slow_pred(T); // predicted slow coral cover (%)
  vector<Type> rec_mult(T);  // report of recruitment multipliers exp(eta_rec) used for transitions to time t

  // Initialize rec_mult to 1 (will be set for t>=1 using eta_rec at t-1)
  for (int i = 0; i < T; ++i) rec_mult(i) = Type(1.0);

  // Time loop for process model (compute predictions at t using information from t-1)
  for (int t = 0; t < T; ++t) {
    if (t == 0) {
      // Initial conditions from parameters (no data leakage)
      cots_pred(0) = smooth_max(cots0, eps);                                               // >= 0
      fast_pred(0) = Type(100.0) - smooth_max(Type(100.0) - smooth_max(fast0, eps), eps);  // clamp to [0,100]
      slow_pred(0) = Type(100.0) - smooth_max(Type(100.0) - smooth_max(slow0, eps), eps);  // clamp to [0,100]
      rec_mult(0) = Type(1.0);
      continue;
    }

    // Previous state values (t-1)
    Type C = cots_pred(t - 1); // adults at time t-1
    Type A = fast_pred(t - 1); // fast coral at time t-1
    Type S = slow_pred(t - 1); // slow coral at time t-1

    // Exogenous environmental drivers at time t-1
    Type sst = sst_dat(t - 1);     // SST forcing (exogenous)
    Type imm = cotsimm_dat(t - 1); // immigration forcing (exogenous)

    // (2) Temperature modifiers (Gaussian) at t-1
    Type phi_T_COTS = exp(-Type(0.5) * pow((sst - Topt_cots) / (sigmaT_cots + eps), 2));    // larval performance 0-1
    Type phi_T_CORAL = exp(-Type(0.5) * pow((sst - Topt_coral) / (sigmaT_coral + eps), 2)); // coral performance 0-1

    // (3) Fertilization success (generalized Allee-type with Hill exponent) at t-1
    Type C_pow = pow(C, nu_spawn);
    Type h_pow = pow(h_spawn + eps, nu_spawn); // ensure positivity even if h_spawn ~ 0
    Type phi_spawn = C_pow / (h_pow + C_pow + eps); // in [0,1)

    // (4) Food limitation for COTS survival (saturating) at t-1
    Type total_coral = A + S;                     // total % cover (0-100)
    Type phi_food = total_coral / (foodK + total_coral + eps); // in (0,1), more food -> higher survival

    // (5) Selective predation per starfish (cap by availability) at t-1
    Type consA_per = max_cons * wA * pow(A, q) / (hA + pow(A, q) + eps); // % cover starfish^-1 yr^-1
    Type consS_per = max_cons * wS * pow(S, q) / (hS + pow(S, q) + eps); // % cover starfish^-1 yr^-1
    Type predA_raw = C * consA_per; // % cover yr^-1
    Type predS_raw = C * consS_per; // % cover yr^-1
    Type predA_eff = A * (Type(1) - exp(-predA_raw / (A + eps))); // smooth cap to <= A
    Type predS_eff = S * (Type(1) - exp(-predS_raw / (S + eps))); // smooth cap to <= S

    // (6) Coral growth and update (space-limited, temperature-modified, background mortality)
    Type free_space = smooth_max(Type(100.0) - A - S, eps);           // smooth >= 0
    Type growthA = rA * A * (free_space / Type(100.0)) * phi_T_CORAL; // % cover yr^-1
    Type growthS = rS * S * (free_space / Type(100.0)) * phi_T_CORAL; // % cover yr^-1
    Type A_next = A + growthA - predA_eff - mA0 * A;                  // provisional fast coral at time t
    Type S_next = S + growthS - predS_eff - mS0 * S;                  // provisional slow coral at time t
    A_next = Type(100.0) - smooth_max(Type(100.0) - smooth_max(A_next, eps), eps); // clamp to [0,100] smoothly
    S_next = Type(100.0) - smooth_max(Type(100.0) - smooth_max(S_next, eps), eps); // clamp to [0,100] smoothly

    // Environmental recruitment multiplier (AR1) used for transition t-1 -> t
    int e_idx = t - 1;
    Type eta_t1 = (e_idx < T_eta ? eta_rec(e_idx) : Type(0.0));
    Type env_rec_mult = exp(eta_t1);
    rec_mult(t) = env_rec_mult; // store multiplier associated with reaching time t

    // (7) COTS survival, recruitment, immigration, and crowding
    Type survival = exp(-(mC + mC_food * (Type(1) - phi_food)));      // fraction surviving 0-1
    Type C_surv = C * survival;                                        // adults after survival
    Type recruits = fec * C * phi_spawn * phi_T_COTS * env_rec_mult;   // new adults from local production with env pulses
    Type I = alpha_imm * (imm / (k_imm + imm + eps));                  // saturating immigration contribution
    Type C_raw_t = C_surv + recruits + I;                              // adults before crowding
    Type C_next = C_raw_t / (Type(1) + beta_dd * C_raw_t);             // Beverton–Holt self-limitation
    C_next = smooth_max(C_next, eps);                                  // ensure nonnegative

    // Assign predictions at time t
    cots_pred(t) = C_next;
    fast_pred(t) = A_next;
    slow_pred(t) = S_next;
  }

  // -------------------------
  // Likelihood (all observations included)
  // -------------------------
  // Lognormal for strictly positive COTS, with small offset to avoid log(0)
  for (int t = 0; t < T; ++t) {
    Type y = cots_dat(t);                     // observed COTS (ind m^-2)
    Type mu = log(cots_pred(t) + eps);        // mean on log scale
    Type ly = log(y + eps);                   // observed on log scale
    nll -= dnorm(ly, mu, sigma_cots, true);   // add log-density
  }

  // Logit-normal for coral proportions (fast/slow), with stabilized proportions
  for (int t = 0; t < T; ++t) {
    // Fast coral
    Type y_fast_prop = (fast_dat(t) / Type(100.0));                          // proportion
    y_fast_prop = y_fast_prop * (Type(1) - Type(2) * prop_eps) + prop_eps;   // keep in (eps,1-eps)
    Type p_fast_pred = (fast_pred(t) / Type(100.0));
    p_fast_pred = p_fast_pred * (Type(1) - Type(2) * prop_eps) + prop_eps;   // keep in (eps,1-eps)
    Type zf = log(p_fast_pred / (Type(1) - p_fast_pred));                    // logit(pred)
    Type yf = log(y_fast_prop / (Type(1) - y_fast_prop));                    // logit(obs)
    nll -= dnorm(yf, zf, sigma_fast, true);                                   // add log-density

    // Slow coral
    Type y_slow_prop = (slow_dat(t) / Type(100.0));                          // proportion
    y_slow_prop = y_slow_prop * (Type(1) - Type(2) * prop_eps) + prop_eps;   // keep in (eps,1-eps)
    Type p_slow_pred = (slow_pred(t) / Type(100.0));
    p_slow_pred = p_slow_pred * (Type(1) - Type(2) * prop_eps) + prop_eps;   // keep in (eps,1-eps)
    Type zs = log(p_slow_pred / (Type(1) - p_slow_pred));                    // logit(pred)
    Type ys = log(y_slow_prop / (Type(1) - y_slow_prop));                    // logit(obs)
    nll -= dnorm(ys, zs, sigma_slow, true);                                   // add log-density
  }

  // -------------------------
  // Smooth parameter bound penalties (biologically plausible ranges)
  // -------------------------
  Type lambda = Type(1.0); // penalty weight (dimensionless)
  nll += penalty_bounds(fec,        Type(0.0),   Type(5.0),   lambda, eps);
  nll += penalty_bounds(h_spawn,    Type(0.01),  Type(5.0),   lambda, eps);
  // Penalize Hill exponent on a plausible range [1,5]
  nll += penalty_bounds(nu_spawn,   Type(1.0),   Type(5.0),   lambda, eps);
  nll += penalty_bounds(mC,         Type(0.0),   Type(2.0),   lambda, eps);
  nll += penalty_bounds(mC_food,    Type(0.0),   Type(3.0),   lambda, eps);
  nll += penalty_bounds(alpha_imm,  Type(0.0),   Type(5.0),   lambda, eps);
  nll += penalty_bounds(k_imm,      Type(0.01),  Type(5.0),   lambda, eps);
  nll += penalty_bounds(Topt_cots,  Type(20.0),  Type(33.0),  lambda, eps);
  nll += penalty_bounds(sigmaT_cots,Type(0.2),   Type(6.0),   lambda, eps);
  nll += penalty_bounds(rA,         Type(0.0),   Type(2.0),   lambda, eps);
  nll += penalty_bounds(rS,         Type(0.0),   Type(1.0),   lambda, eps);
  nll += penalty_bounds(hA,         Type(0.1),   Type(50.0),  lambda, eps);
  nll += penalty_bounds(hS,         Type(0.1),   Type(50.0),  lambda, eps);
  nll += penalty_bounds(max_cons,   Type(0.0),   Type(100.0), lambda, eps);
  nll += penalty_bounds(Topt_coral, Type(20.0),  Type(33.0),  lambda, eps);
  nll += penalty_bounds(sigmaT_coral,Type(0.2),  Type(6.0),   lambda, eps);
  nll += penalty_bounds(mA0,        Type(0.0),   Type(0.5),   lambda, eps);
  nll += penalty_bounds(mS0,        Type(0.0),   Type(0.5),   lambda, eps);
  nll += penalty_bounds(foodK,      Type(1.0),   Type(80.0),  lambda, eps);
  nll += penalty_bounds(beta_dd,    Type(0.0),   Type(2.0),   lambda, eps);
  // bounds on AR(1) parameters
  nll += penalty_bounds(sigma_rec,  Type(0.01),  Type(3.0),   lambda, eps);    // innovation SD
  nll += penalty_bounds(rho_rec,    Type(-0.99), Type(0.99),  lambda, eps);    // autocorrelation
  // bounds on initial states and soft space constraint
  nll += penalty_bounds(cots0,      Type(0.0),   Type(10.0),  lambda, eps);
  nll += penalty_bounds(fast0,      Type(0.0),   Type(100.0), lambda, eps);
  nll += penalty_bounds(slow0,      Type(0.0),   Type(100.0), lambda, eps);
  {
    Type over = smooth_max(fast0 + slow0 - Type(100.0), eps);
    nll += lambda * over * over;
  }

  // -------------------------
  // Reporting
  // -------------------------
  REPORT(cots_pred); // predicted adult COTS (ind m^-2)
  REPORT(fast_pred); // predicted fast coral cover (%)
  REPORT(slow_pred); // predicted slow coral cover (%)
  REPORT(eta_rec);   // AR(1) recruitment effect (log-scale)
  REPORT(rec_mult);  // exp(eta_rec): multiplicative recruitment factor used for transitions to time t

  return nll;
}
