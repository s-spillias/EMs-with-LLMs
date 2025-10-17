#include <TMB.hpp>

// Helper functions
template<class Type>
Type square(const Type& x) { return x * x; } // utility square

// Stable softplus compatible with AD types: log(1 + exp(x)) with overflow guard
template<class Type>
Type softplus(const Type& x) {
  Type zero = Type(0);
  // if x > 0: x + log(1 + exp(-x)); else: log(1 + exp(x))
  return CppAD::CondExpGt(x, zero,
                          x + log(Type(1) + exp(-x)),
                          log(Type(1) + exp(x)));
}

// Soft ReLU approximation of max(0, z) with smoothness controlled by k (>0)
template<class Type>
Type soft_relu(const Type& z, const Type& k) { return softplus(k * z) / (k + Type(1e-12)); } // smooth max(0,z)

// Model description (equations):
// 1) Multi-prey Holling Type II with low-cover refuge (soft threshold):
//    avail_i = P_i * invlogit(k_thr * (P_i - tau_i))
//    denom = 1 + a_F*h_F*avail_F + a_S*h_S*avail_S
//    cons_i_per_pred = a_i * avail_i / denom
//    pred_i = F_star * A * cons_i_per_pred, converted to a smooth fractional loss so pred_i <= P_i:
//    lambda_i = (F_star * A * cons_i_per_pred) / (P_i + eps)
//    pred_i = P_i * (1 - exp(-lambda_i))
// 2) Coral growth (logistic with competition) and temperature modifier (Gaussian):
//    growth_i = r_i * P_i * (1 - (F + S)/K) * exp(-((SST - Topt_i)/Tw_i)^2)
//    Background + bleaching mortality (smooth logistic with SST):
//    mort_i = [m0_i + mTmax_i * invlogit(k_bleach * (SST - T_bleach))] * P_i
//    Update: P_i(t) = P_i(t-1) + growth_i - pred_i - mort_i
// 3) COTS survival and recruitment:
//    Food availability: food = wF*F + wS*S; sat_food = food/(half_sat_food + food)
//    Temperature modifiers (Gaussian):
//    Tsurv = exp(-((SST - Topt_surv)/Tw_surv)^2), Trecr = exp(-((SST - Topt_recruit)/Tw_recruit)^2)
//    Survival rate (smooth, density-dependent):
//    dens = 1/(1 + A/K_A)
//    survival = min_surv + (surv_base - min_surv) * Tsurv * (0.5 + 0.5*sat_food) * dens
//    Recruitment (lag-1):
//    R_int = rho * A * sat_food * Trecr
//    R_ext = sigma_imm * cotsimm
//    Update: A(t) = A(t-1) * survival + R_int + R_ext
// 4) Temperature-modified feeding intensity (optional multiplicative modifier on attack rates):
//    a_i_eff = a_i * exp(beta_feed * (SST - Topt_feed))
//    (used inside the Type II consumption)
// 5) Observation model (lognormal):
//    log(y_dat + eps) ~ Normal(log(y_pred + eps), sigma), with sigma = exp(log_sigma) + sigma_min
// Notes:
// - Initial conditions: state_pred(0) = state_dat(0) for cots, fast, slow.
// - All transitions use t-1 states and forcing to avoid data leakage.
// - Small eps = 1e-8 used to prevent division by zero.
// - Smooth penalties keep parameters in plausible biological ranges without hard constraints.

template<class Type>
Type objective_function<Type>::operator() ()
{
  Type nll = 0.0;                                      // negative log-likelihood accumulator

  // -------------------------
  // Constants and safeguards
  // -------------------------
  const Type eps = Type(1e-8);                         // small constant to avoid division by zero
  const Type sigma_min = Type(0.05);                   // minimum SD in log-space for stability
  const Type k_pen = Type(10.0);                       // smooth ReLU steepness for bounds penalties
  Type pen = 0.0;                                      // parameter bounds penalty accumulator

  // -------------------------
  // Data
  // -------------------------
  DATA_VECTOR(Year);                                    // Year (calendar year), used as index and for reporting
  DATA_VECTOR(sst_dat);                                 // SST (°C), forcing
  DATA_VECTOR(cotsimm_dat);                             // COTS larval immigration (ind/m2/yr), forcing
  DATA_VECTOR(cots_dat);                                // Adult COTS abundance (ind/m2)
  DATA_VECTOR(fast_dat);                                // Fast coral cover (%, Acropora)
  DATA_VECTOR(slow_dat);                                // Slow coral cover (%, Faviidae/Porites)

  int n = cots_dat.size();                              // number of time steps (years)

  // -------------------------
  // Parameters
  // -------------------------
  // Coral growth and carrying capacity
  PARAMETER(rF);                                        // Fast coral intrinsic growth rate (yr^-1); estimate from time trends and literature ranges
  PARAMETER(rS);                                        // Slow coral intrinsic growth rate (yr^-1); estimate from time trends and literature ranges
  PARAMETER(K);                                         // Total coral carrying capacity (% cover); typically <= 100%

  // Predation functional response parameters
  PARAMETER(alphaF);                                    // Attack rate on fast coral (per predator per % per yr); higher implies stronger selectivity on fast coral
  PARAMETER(alphaS);                                    // Attack rate on slow coral (per predator per % per yr); lower than alphaF typically
  PARAMETER(hF);                                        // Handling time for fast coral (yr per %); saturates consumption at high cover
  PARAMETER(hS);                                        // Handling time for slow coral (yr per %); saturates consumption at high cover
  PARAMETER(cons_scale);                                // Consumption-to-cover loss scaling (%-cover lost per (predator-year * encounter unit))

  // Low-cover refuge (soft threshold)
  PARAMETER(tauF);                                      // Fast coral refuge threshold (% cover) where accessibility increases past this level
  PARAMETER(tauS);                                      // Slow coral refuge threshold (% cover)
  PARAMETER(kappa_thr);                                 // Slope of soft threshold (dimensionless); higher = sharper transition

  // Temperature modifiers for feeding
  PARAMETER(Topt_feed);                                 // Temperature at which feeding is most intense (°C)
  PARAMETER(beta_feed);                                 // Temperature sensitivity of feeding (per °C); modifies effective attack rates

  // Coral temperature-growth responses
  PARAMETER(ToptF);                                     // Fast coral thermal optimum for growth (°C)
  PARAMETER(TwF);                                       // Fast coral thermal width for growth (°C, >0) controlling breadth of response
  PARAMETER(ToptS);                                     // Slow coral thermal optimum for growth (°C)
  PARAMETER(TwS);                                       // Slow coral thermal width for growth (°C, >0)

  // Coral background and bleaching mortalities
  PARAMETER(m0F);                                       // Fast coral baseline mortality rate (yr^-1) at benign temps
  PARAMETER(m0S);                                       // Slow coral baseline mortality rate (yr^-1)
  PARAMETER(mTmaxF);                                    // Max additional fast coral mortality from heat stress (yr^-1)
  PARAMETER(mTmaxS);                                    // Max additional slow coral mortality from heat stress (yr^-1)
  PARAMETER(T_bleach);                                  // SST (°C) where bleaching risk inflects
  PARAMETER(k_bleach);                                  // Slope of logistic bleaching risk (per °C)

  // COTS recruitment and survival
  PARAMETER(rho);                                       // Internal recruitment rate to adult stage (yr^-1) per adult, after early-stage survival
  PARAMETER(half_sat_food);                             // Half-saturation for food effect on recruitment/survival (% cover)
  PARAMETER(sigma_imm);                                 // External immigration-to-adult efficiency (dimensionless, 0-1)

  PARAMETER(Topt_recruit);                              // Thermal optimum for recruitment (°C)
  PARAMETER(Tw_recruit);                                // Thermal width for recruitment (°C, >0)

  PARAMETER(surv_base);                                 // Baseline annual survival upper bound (dimensionless, <1)
  PARAMETER(K_A);                                       // Density scale for survival crowding (ind/m2)
  PARAMETER(Topt_surv);                                 // Thermal optimum for survival (°C)
  PARAMETER(Tw_surv);                                   // Thermal width for survival (°C, >0)
  // Fixed minimum survival (not estimated) to prevent collapse in poor conditions
  Type min_surv = Type(0.05);                           // Minimum survival floor (dimensionless)

  // Food availability weights for fast/slow coral on COTS demographic rates
  PARAMETER(w_food_F);                                  // Weight of fast coral in food availability (0-1)
  PARAMETER(w_food_S);                                  // Weight of slow coral in food availability (0-1)

  // Observation error (lognormal SDs on log-scale)
  PARAMETER(log_sigma_cots);                            // log SD for COTS observations (log-scale)
  PARAMETER(log_sigma_fast);                            // log SD for fast coral (%), log-scale errors
  PARAMETER(log_sigma_slow);                            // log SD for slow coral (%), log-scale errors

  // -------------------------
  // Parameter soft bounds (smooth penalties; no hard constraints)
  // -------------------------
  auto add_bound_pen = [&](Type x, Type lo, Type hi, Type w){
    pen += square(soft_relu(lo - x, k_pen)) * w + square(soft_relu(x - hi, k_pen)) * w;
  };

  // Suggested biological ranges (weights ~1; adjust if needed)
  add_bound_pen(rF, Type(0.0), Type(2.0), Type(1.0));
  add_bound_pen(rS, Type(0.0), Type(1.0), Type(1.0));
  add_bound_pen(K,  Type(20.0), Type(100.0), Type(1.0));

  add_bound_pen(alphaF, Type(0.0), Type(1.0), Type(1.0));
  add_bound_pen(alphaS, Type(0.0), Type(1.0), Type(1.0));
  add_bound_pen(hF, Type(0.0), Type(2.0), Type(1.0));
  add_bound_pen(hS, Type(0.0), Type(2.0), Type(1.0));
  add_bound_pen(cons_scale, Type(0.0), Type(100.0), Type(1.0));

  add_bound_pen(tauF, Type(0.0), Type(60.0), Type(1.0));
  add_bound_pen(tauS, Type(0.0), Type(60.0), Type(1.0));
  add_bound_pen(kappa_thr, Type(0.01), Type(5.0), Type(1.0));

  add_bound_pen(Topt_feed, Type(24.0), Type(32.0), Type(1.0));
  add_bound_pen(beta_feed, Type(-0.5), Type(0.5), Type(1.0));

  add_bound_pen(ToptF, Type(24.0), Type(32.0), Type(1.0));
  add_bound_pen(TwF, Type(0.1), Type(6.0), Type(1.0));
  add_bound_pen(ToptS, Type(24.0), Type(32.0), Type(1.0));
  add_bound_pen(TwS, Type(0.1), Type(6.0), Type(1.0));

  add_bound_pen(m0F, Type(0.0), Type(0.5), Type(1.0));
  add_bound_pen(m0S, Type(0.0), Type(0.5), Type(1.0));
  add_bound_pen(mTmaxF, Type(0.0), Type(0.8), Type(1.0));
  add_bound_pen(mTmaxS, Type(0.0), Type(0.8), Type(1.0));
  add_bound_pen(T_bleach, Type(27.0), Type(32.0), Type(1.0));
  add_bound_pen(k_bleach, Type(0.1), Type(5.0), Type(1.0));

  add_bound_pen(rho, Type(0.0), Type(2.0), Type(1.0));
  add_bound_pen(half_sat_food, Type(1.0), Type(80.0), Type(1.0));
  add_bound_pen(sigma_imm, Type(0.0), Type(1.0), Type(1.0));

  add_bound_pen(Topt_recruit, Type(24.0), Type(32.0), Type(1.0));
  add_bound_pen(Tw_recruit, Type(0.1), Type(6.0), Type(1.0));

  add_bound_pen(surv_base, Type(0.2), Type(0.95), Type(1.0));
  add_bound_pen(K_A, Type(0.1), Type(5.0), Type(1.0));
  add_bound_pen(Topt_surv, Type(24.0), Type(32.0), Type(1.0));
  add_bound_pen(Tw_surv, Type(0.1), Type(6.0), Type(1.0));

  add_bound_pen(w_food_F, Type(0.0), Type(1.0), Type(1.0));
  add_bound_pen(w_food_S, Type(0.0), Type(1.0), Type(1.0));
  pen += square((w_food_F + w_food_S) - Type(1.0));     // prefer weights to sum to 1 (softly)

  // -------------------------
  // Containers for predictions
  // -------------------------
  vector<Type> cots_pred(n);                            // predicted adult COTS (ind/m2)
  vector<Type> fast_pred(n);                            // predicted fast coral (%)
  vector<Type> slow_pred(n);                            // predicted slow coral (%)

  // Initialize with observed initial conditions (no optimization parameters as initials)
  cots_pred(0) = cots_dat(0);                           // initial adult COTS from data
  fast_pred(0) = fast_dat(0);                           // initial fast coral from data
  slow_pred(0) = slow_dat(0);                           // initial slow coral from data

  // Optional: store components for reporting/diagnostics
  vector<Type> R_int(n); R_int.setZero();               // internal recruitment (ind/m2/yr)
  vector<Type> R_ext(n); R_ext.setZero();               // external recruitment (ind/m2/yr)
  vector<Type> survivalA(n); survivalA.setZero();       // annual survival rate (0-1)
  vector<Type> predF(n); predF.setZero();               // predation loss fast coral (%/yr)
  vector<Type> predS(n); predS.setZero();               // predation loss slow coral (%/yr)

  // -------------------------
  // State transition loop (uses only t-1 states and forcings)
  // -------------------------
  for (int t = 1; t < n; ++t) {
    // Previous states
    Type Aprev = cots_pred(t - 1);                      // adult COTS at t-1 (ind/m2)
    Type Fprev = fast_pred(t - 1);                      // fast coral % at t-1
    Type Sprev = slow_pred(t - 1);                      // slow coral % at t-1
    Type Cprev = Fprev + Sprev;                         // total coral % at t-1

    // Forcings at t-1
    Type SST = sst_dat(t - 1);                          // SST at t-1 (°C)
    Type IMM = cotsimm_dat(t - 1);                      // external larval immigration at t-1 (ind/m2/yr)

    // Feeding temperature modifier on attack rates
    Type feed_T = exp(beta_feed * (SST - Topt_feed));   // multiplicative temp effect on attack rates

    // Low-cover availability (soft threshold) using TMB's invlogit
    Type vF = invlogit(kappa_thr * (Fprev - tauF));     // accessibility of fast coral (0-1)
    Type vS = invlogit(kappa_thr * (Sprev - tauS));     // accessibility of slow coral (0-1)
    Type availF = Fprev * vF;                           // accessible fast coral (%)
    Type availS = Sprev * vS;                           // accessible slow coral (%)

    // Multi-prey Holling type II denominators
    Type aF_eff = alphaF * feed_T;                      // temp-modified attack on fast
    Type aS_eff = alphaS * feed_T;                      // temp-modified attack on slow
    Type denom = Type(1.0) + aF_eff * hF * availF + aS_eff * hS * availS + eps; // ensure denom>0

    // Per-predator consumption rates
    Type consF_perPred = aF_eff * availF / denom;       // fast coral consumption per predator (%-units per yr)
    Type consS_perPred = aS_eff * availS / denom;       // slow coral consumption per predator (%-units per yr)

    // Total predation losses (smoothly bounded to <= current cover)
    Type rawLossF = cons_scale * Aprev * consF_perPred; // raw loss in %/yr
    Type rawLossS = cons_scale * Aprev * consS_perPred; // raw loss in %/yr
    Type lambdaF = rawLossF / (Fprev + eps);            // hazard-like fraction for fast coral
    Type lambdaS = rawLossS / (Sprev + eps);            // hazard-like fraction for slow coral
    Type lossF = Fprev * (Type(1.0) - exp(-lambdaF));   // smooth cap to <= Fprev
    Type lossS = Sprev * (Type(1.0) - exp(-lambdaS));   // smooth cap to <= Sprev

    // Temperature modifiers for coral growth (Gaussian around optima)
    Type gTF = exp(-square((SST - ToptF) / (TwF + eps))); // fast coral growth temp modifier (0-1)
    Type gTS = exp(-square((SST - ToptS) / (TwS + eps))); // slow coral growth temp modifier (0-1)

    // Logistic growth with total coral competition
    Type growthF = rF * Fprev * (Type(1.0) - (Cprev / (K + eps))) * gTF; // fast coral growth (%/yr)
    Type growthS = rS * Sprev * (Type(1.0) - (Cprev / (K + eps))) * gTS; // slow coral growth (%/yr)

    // Background + bleaching mortality (smooth logistic with SST)
    Type bleach = invlogit(k_bleach * (SST - T_bleach)); // 0-1 heat stress indicator
    Type mortF = (m0F + mTmaxF * bleach) * Fprev;       // fast coral mortality (%/yr)
    Type mortS = (m0S + mTmaxS * bleach) * Sprev;       // slow coral mortality (%/yr)

    // Coral updates
    Type Fnext = Fprev + growthF - lossF - mortF;       // updated fast coral (%)
    Type Snext = Sprev + growthS - lossS - mortS;       // updated slow coral (%)

    // Food availability for COTS demography
    Type food = w_food_F * Fprev + w_food_S * Sprev;    // weighted coral cover (%)
    Type sat_food = food / (half_sat_food + food + eps);// saturating food effect (0-1)

    // Temperature modifiers for COTS survival and recruitment
    Type Tsurv = exp(-square((SST - Topt_surv) / (Tw_surv + eps)));     // (0-1)
    Type Trecr = exp(-square((SST - Topt_recruit) / (Tw_recruit + eps)));// (0-1)

    // Density dependence in survival
    Type dens = Type(1.0) / (Type(1.0) + Aprev / (K_A + eps)); // 1/(1 + A/K_A)

    // Annual survival rate (bounded between min_surv and surv_base)
    Type surv = min_surv + (surv_base - min_surv) * Tsurv * (Type(0.5) + Type(0.5) * sat_food) * dens;
    surv = CppAD::CondExpGt(surv, Type(0.999), Type(0.999), surv); // prevent exact 1 (stability)
    survivalA(t) = surv;

    // Recruitment with 1-year lag (uses t-1 drivers)
    R_int(t) = rho * Aprev * sat_food * Trecr;          // internal recruitment (ind/m2/yr)
    R_ext(t) = sigma_imm * IMM;                         // external recruitment (ind/m2/yr)

    // Adult COTS update
    Type Anext = Aprev * surv + R_int(t) + R_ext(t);    // updated adult abundance (ind/m2)

    // Store predictions
    cots_pred(t) = Anext;                               // predicted COTS at t
    fast_pred(t) = Fnext;                               // predicted fast coral at t
    slow_pred(t) = Snext;                               // predicted slow coral at t

    // Save predation losses for reporting
    predF(t) = lossF;                                   // fast coral predation loss (%)
    predS(t) = lossS;                                   // slow coral predation loss (%)
  }

  // -------------------------
  // Observation model (lognormal on strictly positive quantities)
  // -------------------------
  Type sigma_cots = exp(log_sigma_cots) + sigma_min;    // SD for COTS on log scale
  Type sigma_fast = exp(log_sigma_fast) + sigma_min;    // SD for fast coral on log scale
  Type sigma_slow = exp(log_sigma_slow) + sigma_min;    // SD for slow coral on log scale

  for (int t = 0; t < n; ++t) {
    // All observations used; add small eps to avoid log(0)
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots, true); // COTS
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast, true); // fast coral
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow, true); // slow coral
  }

  // -------------------------
  // Add parameter penalties
  // -------------------------
  nll += pen;

  // -------------------------
  // Reporting
  // -------------------------
  REPORT(Year);                                         // echo time axis
  REPORT(sst_dat);                                      // SST forcing (for diagnostics)
  REPORT(cotsimm_dat);                                  // immigration forcing (for diagnostics)
  REPORT(cots_pred);                                    // predicted adult COTS (ind/m2)
  REPORT(fast_pred);                                    // predicted fast coral (%)
  REPORT(slow_pred);                                    // predicted slow coral (%)
  REPORT(R_int);                                        // internal recruitment time series
  REPORT(R_ext);                                        // external recruitment time series
  REPORT(survivalA);                                    // survival rate time series
  REPORT(predF);                                        // predation loss fast coral
  REPORT(predS);                                        // predation loss slow coral

  ADREPORT(cots_pred);                                  // enable standard errors for predictions
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);

  return nll;
}
