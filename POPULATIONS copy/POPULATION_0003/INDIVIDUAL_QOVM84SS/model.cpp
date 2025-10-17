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
Numbered equation summary (annual time step, index t = 0..T-2):

1) Initial conditions (no data leakage):
   cots_pred(0) = cots_dat(0)
   fast_pred(0) = fast_dat(0)
   slow_pred(0) = slow_dat(0)

2) Temperature modifiers (Gaussian/bell-shaped performance):
   phi_T_COTS(t)  = exp(-0.5 * ((sst_dat(t) - Topt_cots)  / sigmaT_cots)^2)
   phi_T_CORAL(t) = exp(-0.5 * ((sst_dat(t) - Topt_coral) / sigmaT_coral)^2)

3) Fertilization success (saturating Allee-type effect):
   phi_spawn(t) = C_t / (h_spawn + C_t)

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
   A_{t+1} = clamp_0_100( A_t + growthA(t) - predA_eff(t) - mA0 * A_t )
   S_{t+1} = clamp_0_100( S_t + growthS(t) - predS_eff(t) - mS0 * S_t )

7) COTS survival, recruitment, immigration, and crowding:
   survival(t) = exp( -[ mC + mC_food * (1 - phi_food(t)) ] )
   C_surv(t)   = C_t * survival(t)
   fec_eff(t)  = fec * [ 1 - alpha_fec_food * (1 - phi_food(t)) ]    // new: food-conditioned fecundity (0..fec)
   recruits(t) = fec_eff(t) * C_t * phi_spawn(t) * phi_T_COTS(t)
   I(t)        = alpha_imm * cotsimm_dat(t) / (k_imm + cotsimm_dat(t))
   C_raw_{t+1} = C_surv(t) + recruits(t) + I(t)
   C_{t+1}     = C_raw_{t+1} / (1 + beta_dd * C_raw_{t+1})  // Beverton–Holt crowding
*/

template<class Type>
Type objective_function<Type>::operator() () {
  // -------------------------
  // Data (all lengths are T)
  // -------------------------
  DATA_VECTOR(Year);          // calendar year (integer years; used for alignment)
  DATA_VECTOR(sst_dat);       // Sea-surface temperature (°C), annual
  DATA_VECTOR(cotsimm_dat);   // External larval immigration (individuals m^-2 yr^-1)
  DATA_VECTOR(cots_dat);      // Adult COTS density (individuals m^-2)
  DATA_VECTOR(fast_dat);      // Fast coral cover (Acropora), percent (% cover, 0-100)
  DATA_VECTOR(slow_dat);      // Slow coral cover (Faviidae/Porites), percent (% cover, 0-100)

  // -------------------------
  // Parameters
  // -------------------------
  PARAMETER(fec);            // recruits per adult per year reaching adulthood (yr^-1), initial estimate
  PARAMETER(h_spawn);        // half-saturation adult density for fertilization (ind m^-2), initial estimate
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
  PARAMETER(alpha_fec_food); // new: strength of food control on fecundity (dimensionless, 0..1), initial estimate

  // -------------------------
  // Constants and helpers
  // -------------------------
  int T = Year.size();                     // number of time steps (years)
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

  // -------------------------
  // State predictions
  // -------------------------
  vector<Type> cots_pred(T); // predicted adult COTS (ind m^-2)
  vector<Type> fast_pred(T); // predicted fast coral cover (%)
  vector<Type> slow_pred(T); // predicted slow coral cover (%)

  // Initial conditions from data (no data leakage beyond t=0)
  cots_pred(0) = cots_dat(0); // initialize with observed COTS density
  fast_pred(0) = fast_dat(0); // initialize with observed fast coral cover
  slow_pred(0) = slow_dat(0); // initialize with observed slow coral cover

  // Time loop for process model
  for (int t = 0; t < T - 1; ++t) {
    // Previous state values (t)
    Type C = cots_pred(t); // adults at time t
    Type A = fast_pred(t); // fast coral at time t
    Type S = slow_pred(t); // slow coral at time t

    // Environmental drivers at time t
    Type sst = sst_dat(t);       // SST forcing
    Type imm = cotsimm_dat(t);   // immigration forcing

    // (2) Temperature modifiers (Gaussian)
    Type phi_T_COTS = exp(-Type(0.5) * pow((sst - Topt_cots) / (sigmaT_cots + eps), 2));   // larval performance 0-1
    Type phi_T_CORAL = exp(-Type(0.5) * pow((sst - Topt_coral) / (sigmaT_coral + eps), 2)); // coral performance 0-1

    // (3) Fertilization success (saturating with adult density)
    Type phi_spawn = C / (h_spawn + C + eps); // in [0,1), avoids Allee failure at very low C

    // (4) Food limitation for COTS survival (saturating)
    Type total_coral = A + S;                     // total % cover (0-100)
    Type phi_food = total_coral / (foodK + total_coral + eps); // in (0,1), more food -> higher survival

    // Preference weights and functional response exponent
    Type wA = inv_logit(prefA_logit);  // preference for fast coral in [0,1]
    Type wS = Type(1) - wA;            // preference for slow coral
    Type q = Type(1) + exp(log_q_FR);  // q >= 1; q=1 -> Type II, q>1 -> Type III

    // (5) Selective predation per starfish (cap by availability)
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
    Type A_next = A + growthA - predA_eff - mA0 * A;                  // provisional fast coral next year
    Type S_next = S + growthS - predS_eff - mS0 * S;                  // provisional slow coral next year
    A_next = Type(100.0) - smooth_max(Type(100.0) - smooth_max(A_next, eps), eps); // clamp to [0,100] smoothly
    S_next = Type(100.0) - smooth_max(Type(100.0) - smooth_max(S_next, eps), eps); // clamp to [0,100] smoothly

    // (7) COTS survival, recruitment, immigration, and crowding
    Type survival = exp(-(mC + mC_food * (Type(1) - phi_food)));      // fraction surviving 0-1
    Type C_surv = C * survival;                                        // adults after survival
    Type fec_eff = fec * (Type(1) - alpha_fec_food * (Type(1) - phi_food)); // new: food-conditioned fecundity (0..fec)
    Type recruits = fec_eff * C * phi_spawn * phi_T_COTS;              // new adults from local production
    Type I = alpha_imm * (imm / (k_imm + imm + eps));                  // saturating immigration contribution
    Type C_raw_next = C_surv + recruits + I;                           // adults before crowding
    Type C_next = C_raw_next / (Type(1) + beta_dd * C_raw_next);       // Beverton–Holt self-limitation
    C_next = smooth_max(C_next, eps);                                  // ensure nonnegative

    // Assign to predictions (t+1)
    cots_pred(t + 1) = C_next;
    fast_pred(t + 1) = A_next;
    slow_pred(t + 1) = S_next;
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
  // prefA_logit unconstrained in R, but implied wA in (0,1); no penalty needed unless extreme values cause issues
  // log_q_FR unconstrained; q >= 1 via exp transform, so no penalty needed
  nll += penalty_bounds(alpha_fec_food, Type(0.0), Type(1.0), lambda, eps); // new: bounds for food effect on fecundity

  // -------------------------
  // Reporting
  // -------------------------
  REPORT(cots_pred); // predicted adult COTS (ind m^-2)
  REPORT(fast_pred); // predicted fast coral cover (%)
  REPORT(slow_pred); // predicted slow coral cover (%)

  return nll;
}
