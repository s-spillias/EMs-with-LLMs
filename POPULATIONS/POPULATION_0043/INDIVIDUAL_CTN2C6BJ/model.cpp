#include <TMB.hpp>

// Helper: inverse logit for parameters constrained to (0,1)
template<class Type>
Type invlogit(Type x) {
  return Type(1) / (Type(1) + exp(-x));
}

// Helper: softplus for smooth non-negativity
template<class Type>
Type softplus(Type x) {
  return log1p(exp(x));
}

// Helper: Gaussian thermal performance curve (unitless multiplier in (0,1])
template<class Type>
Type temp_gauss(Type sst, Type Topt, Type Tsd) {
  Type eps = Type(1e-8);
  Type z = (sst - Topt) / (Tsd + eps);
  return exp(-Type(0.5) * z * z);
}

// Helper: smooth logistic switch for bleaching severity (0..1)
template<class Type>
Type smooth_switch(Type x, Type k) {
  // x = sst - T_bleach; k = steepness (>0). Returns 1/(1+exp(-k*x))
  return Type(1) / (Type(1) + exp(-k * x));
}

// Helper: positive part for smooth bound penalties
template<class Type>
Type pospart(Type x) {
  return CppAD::CondExpGt(x, Type(0), x, Type(0));
}

template<class Type>
Type objective_function<Type>::operator() ()
{
  // -----------------------------
  // DATA (time series; do not modify)
  // -----------------------------
  DATA_VECTOR(Year);        // Year (calendar year)
  DATA_VECTOR(cots_dat);    // Adult COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);    // Fast coral (Acropora) cover (%)
  DATA_VECTOR(slow_dat);    // Slow coral (Faviidae + Porites) cover (%)
  DATA_VECTOR(sst_dat);     // Sea-Surface Temperature (Celsius)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (individuals/m2/year)

  int T = Year.size(); // Number of years

  // -----------------------------
  // PARAMETERS (transformed where appropriate)
  // -----------------------------
  // Initial states (positive, estimated on log-scale)
  PARAMETER(log_C0);   // log initial COTS density in individuals/m2; initialize near first observation (log-scale)
  PARAMETER(log_F0);   // log initial fast coral cover in %; initialize near first observation (log-scale)
  PARAMETER(log_S0);   // log initial slow coral cover in %; initialize near first observation (log-scale)

  // COTS population dynamics
  PARAMETER(log_rC_max);     // log max per-capita COTS growth rate (yr^-1), resource- and temperature-limited
  PARAMETER(log_mC_base);    // log baseline COTS mortality rate (yr^-1)
  PARAMETER(log_dC);         // log density-dependent COTS mortality coefficient ((m^2/ind)/yr)
  PARAMETER(log_A50);        // log Allee scale for depensation (individuals/m2) at which growth is half of max low-density
  PARAMETER(log_Kres);       // log half-saturation for food/resource effect (proportion coral; 0-1)
  PARAMETER(Topt_C);         // Optimal SST for COTS growth (Celsius); from literature or estimated
  PARAMETER(log_Tsd_C);      // log thermal breadth (Celsius) for COTS growth Gaussian
  PARAMETER(pref_f_logit);   // logit preference for fast coral in resource proxy (unitless, maps to 0..1)
  PARAMETER(food_exp_raw);   // raw parameter shaping food-response exponent p >= 1 via p = 1 + softplus(raw)
  PARAMETER(logit_e_imm);    // logit efficiency converting larval immigration to new adults (unitless 0..1)

  // Coral intrinsic growth and space limitation
  PARAMETER(log_gF_max);   // log intrinsic growth rate of fast coral (yr^-1)
  PARAMETER(log_gS_max);   // log intrinsic growth rate of slow coral (yr^-1)
  PARAMETER(logit_Kc);     // logit of total coral carrying capacity as proportion of substrate (0..1)
  PARAMETER(alpha_fs_raw); // raw >=0 competition weight of slow on fast via softplus (unitless)
  PARAMETER(alpha_sf_raw); // raw >=0 competition weight of fast on slow via softplus (unitless)

  // Coral predation mortality (selective, saturating with prey availability)
  PARAMETER(log_betaF);   // log predation coefficient on fast coral ((m2/ind)/yr)
  PARAMETER(log_betaS);   // log predation coefficient on slow coral ((m2/ind)/yr)
  PARAMETER(log_Kpred);   // log half-saturation of predation with prey proportion (0..1, on log scale)

  // Coral temperature responses
  PARAMETER(Topt_F);      // Optimal SST for fast coral growth (Celsius)
  PARAMETER(log_Tsd_F);   // log thermal breadth (Celsius) for fast coral growth
  PARAMETER(Topt_S);      // Optimal SST for slow coral growth (Celsius)
  PARAMETER(log_Tsd_S);   // log thermal breadth (Celsius) for slow coral growth

  // Bleaching mortality (smooth logistic with SST)
  PARAMETER(T_bleach);     // SST midpoint of bleaching response (Celsius)
  PARAMETER(log_k_bleach); // log steepness of bleaching logistic (1/Celsius)
  PARAMETER(log_mbleachF); // log maximum annual bleaching mortality rate for fast coral (yr^-1)
  PARAMETER(log_mbleachS); // log maximum annual bleaching mortality rate for slow coral (yr^-1)

  // Observation error (lognormal, strictly positive)
  PARAMETER(log_sigma_cots); // log SD of observation error for COTS on log scale
  PARAMETER(log_sigma_fast); // log SD of observation error for fast coral on log scale
  PARAMETER(log_sigma_slow); // log SD of observation error for slow coral on log scale

  // -----------------------------
  // Transform parameters to working scales
  // -----------------------------
  Type eps = Type(1e-8); // small constant to avoid division by zero and logs of zero

  // Initial states
  Type C0 = exp(log_C0); // individuals/m2
  Type F0 = exp(log_F0); // %
  Type S0 = exp(log_S0); // %

  // COTS
  Type rC_max   = exp(log_rC_max);         // yr^-1
  Type mC_base  = exp(log_mC_base);        // yr^-1
  Type dC       = exp(log_dC);             // (m^2/ind)/yr
  Type A50      = exp(log_A50);            // individuals/m2
  Type Kres     = exp(log_Kres);           // proportion (0..1)
  Type Tsd_C    = exp(log_Tsd_C);          // Celsius
  Type pref_f   = invlogit(pref_f_logit);  // 0..1
  Type pref_s   = Type(1) - pref_f;        // 0..1
  Type p_food   = Type(1) + softplus(food_exp_raw); // exponent >= 1 for resource nonlinearity
  Type e_imm    = invlogit(logit_e_imm);   // 0..1

  // Coral growth and space
  Type gF_max = exp(log_gF_max);       // yr^-1
  Type gS_max = exp(log_gS_max);       // yr^-1
  Type Kc     = invlogit(logit_Kc);    // proportion 0..1
  Type alpha_fs = softplus(alpha_fs_raw); // >=0
  Type alpha_sf = softplus(alpha_sf_raw); // >=0

  // Predation
  Type betaF  = exp(log_betaF);        // (m2/ind)/yr scaling to mortality rate on fast coral
  Type betaS  = exp(log_betaS);        // (m2/ind)/yr scaling to mortality rate on slow coral
  Type Kpred  = exp(log_Kpred);        // proportion (0..1)

  // Coral temperature
  Type Tsd_F  = exp(log_Tsd_F);        // Celsius
  Type Tsd_S  = exp(log_Tsd_S);        // Celsius

  // Bleaching
  Type k_bleach   = exp(log_k_bleach); // 1/Celsius
  Type mbleachF   = exp(log_mbleachF); // yr^-1
  Type mbleachS   = exp(log_mbleachS); // yr^-1

  // Observation error (with minimum SDs)
  Type sigma_min = Type(0.05); // minimum SD on log scale for numerical stability
  Type sigma_cots = exp(log_sigma_cots) + sigma_min; // log-scale SD
  Type sigma_fast = exp(log_sigma_fast) + sigma_min; // log-scale SD
  Type sigma_slow = exp(log_sigma_slow) + sigma_min; // log-scale SD

  // -----------------------------
  // STATE VECTORS FOR PREDICTIONS
  // -----------------------------
  vector<Type> cots_pred(T); // predicted COTS (individuals/m2)
  vector<Type> fast_pred(T); // predicted fast coral (%)
  vector<Type> slow_pred(T); // predicted slow coral (%)

  // Initialize at first time step using parameters (no data leakage)
  cots_pred(0) = C0; // individuals/m2
  fast_pred(0) = F0; // %
  slow_pred(0) = S0; // %

  // -----------------------------
  // PROCESS MODEL (annual update)
  // Equations (all smooth, using t-1 states to predict t):
  //
  // 1) Resource proxy (proportion): P_t-1 = pref_f * (F%/100) + pref_s * (S%/100)
  // 2) Food limitation: R_food = P / (Kres + P)
  // 3) COTS depensation: Dep = C_prev / (C_prev + A50)
  // 4) Temperature multipliers: Gx = exp(-0.5 * ((SST - Topt_x)/Tsd_x)^2)
  // 5) COTS net rate: rC = rC_max * R_food * Gc * Dep - (mC_base + dC * C_prev)
  // 6) COTS update: C_t = C_prev * exp(rC) + e_imm * cotsimm_dat_{t-1}
  // 7) Coral competition terms (proportions): compF = (pF + alpha_fs*pS) / Kc; compS analogous
  // 8) Coral growth rates: rF = gF_max * Gf * (1 - compF); rS analogous
  // 9) Bleaching mortality: mBleach_x = mbleachX * logistic(sst - T_bleach; k_bleach)
  // 10) Predation mortality on corals (selective, saturating): 
  //     mPred_F = betaF * C_prev * ( (pref_f*pF)^p / (Kpred + (pref_f*pF)^p) ), mPred_S analogous
  // 11) Coral updates (multiplicative to ensure positivity):
  //     F_t = F_prev * exp(rF - mBleach_F - mPred_F)
  //     S_t = S_prev * exp(rS - mBleach_S - mPred_S)
  // -----------------------------
  for (int t = 1; t < T; t++) {
    int tp = t - 1; // previous time index

    // Previous states
    Type C_prev = cots_pred(tp);      // individuals/m2
    Type F_prev = fast_pred(tp);      // %
    Type S_prev = slow_pred(tp);      // %

    // Convert coral cover to proportions (0..1) for process calculations
    Type pF_prev = F_prev / Type(100.0);
    Type pS_prev = S_prev / Type(100.0);

    // Environmental drivers for the interval (t-1) -> t
    Type SST_prev  = sst_dat(tp);           // Celsius
    Type IMM_prev  = cotsimm_dat(tp);       // individuals/m2/year

    // Resource proxy (proportion of prey accessible/selected)
    Type P_prev = pref_f * pF_prev + pref_s * pS_prev; // unitless (0..1)

    // Food limitation (saturating)
    Type R_food = P_prev / (Kres + P_prev + eps); // unitless

    // Depensation (Allee effect)
    Type Dep = C_prev / (C_prev + A50 + eps); // unitless (0..1)

    // Thermal multipliers (0..1]
    Type Gc = temp_gauss(SST_prev, Topt_C, Tsd_C); // COTS
    Type Gf = temp_gauss(SST_prev, Topt_F, Tsd_F); // fast coral
    Type Gs = temp_gauss(SST_prev, Topt_S, Tsd_S); // slow coral

    // COTS net per-capita rate (yr^-1)
    Type rC = rC_max * R_food * Gc * Dep - (mC_base + dC * C_prev);

    // Immigration efficiency (adults)
    Type I_eff = e_imm * IMM_prev; // individuals/m2/year

    // Update COTS (ensure positivity by multiplicative form + immigration)
    Type C_next = C_prev * exp(rC) + I_eff; // individuals/m2

    // Predation mortality rates on corals (yr^-1), selective and saturating with prey availability
    Type numF = pow(pref_f * pF_prev + eps, p_food); // weighted prey availability for fast coral
    Type numS = pow(pref_s * pS_prev + eps, p_food); // weighted prey availability for slow coral
    Type mPred_F = betaF * C_prev * ( numF / (Kpred + numF) ); // yr^-1
    Type mPred_S = betaS * C_prev * ( numS / (Kpred + numS) ); // yr^-1

    // Bleaching mortality (yr^-1), smooth logistic of SST
    Type Hbleach = smooth_switch(SST_prev - T_bleach, k_bleach); // 0..1
    Type mBleach_F = mbleachF * Hbleach; // yr^-1
    Type mBleach_S = mbleachS * Hbleach; // yr^-1

    // Space competition (dimensionless, saturating near Kc)
    Type compF = (pF_prev + alpha_fs * pS_prev) / (Kc + eps);
    Type compS = (pS_prev + alpha_sf * pF_prev) / (Kc + eps);

    // Coral net growth rates (yr^-1)
    Type rF = gF_max * Gf * (Type(1) - compF);
    Type rS = gS_max * Gs * (Type(1) - compS);

    // Update corals multiplicatively (remain positive)
    Type F_next = F_prev * exp(rF - mBleach_F - mPred_F); // %
    Type S_next = S_prev * exp(rS - mBleach_S - mPred_S); // %

    // Assign predictions
    cots_pred(t) = C_next;
    fast_pred(t) = F_next;
    slow_pred(t) = S_next;
  }

  // -----------------------------
  // LIKELIHOOD (lognormal errors, include all observations)
  // -----------------------------
  Type nll = Type(0.0);
  for (int t = 0; t < T; t++) {
    // Log-transformed residuals with floor to avoid log(0)
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots, true); // COTS
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast, true); // fast coral
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow, true); // slow coral
  }

  // -----------------------------
  // SOFT BOUND PENALTIES (biological ranges; smooth quadratic outside bounds)
  // -----------------------------
  Type pen_w = Type(10.0); // penalty weight
  Type pen = Type(0.0);

  // Example biological ranges (on natural scales):
  // Kc in [0.1, 0.9]
  pen += pen_w * pow(pospart(Type(0.1) - Kc), 2) + pen_w * pow(pospart(Kc - Type(0.9)), 2);
  // Preference moderately biased but not extreme to avoid identifiability: pref_f in [0.4, 0.95]
  pen += pen_w * pow(pospart(Type(0.4) - pref_f), 2) + pen_w * pow(pospart(pref_f - Type(0.95)), 2);
  // Thermal breadths in [0.5, 4] C
  pen += pen_w * pow(pospart(Type(0.5) - Tsd_C), 2) + pen_w * pow(pospart(Tsd_C - Type(4.0)), 2);
  pen += pen_w * pow(pospart(Type(0.5) - Tsd_F), 2) + pen_w * pow(pospart(Tsd_F - Type(4.0)), 2);
  pen += pen_w * pow(pospart(Type(0.5) - Tsd_S), 2) + pen_w * pow(pospart(Tsd_S - Type(4.0)), 2);
  // Bleaching midpoint in [26, 32] C
  pen += pen_w * pow(pospart(Type(26.0) - T_bleach), 2) + pen_w * pow(pospart(T_bleach - Type(32.0)), 2);
  // Immigration efficiency in [0.01, 0.8]
  pen += pen_w * pow(pospart(Type(0.01) - e_imm), 2) + pen_w * pow(pospart(e_imm - Type(0.8)), 2);
  // Coral carrying capacity should exceed typical observed means: low bound helps identifiability
  // Food half-saturation Kres in [0.005, 0.5]
  pen += pen_w * pow(pospart(Type(0.005) - Kres), 2) + pen_w * pow(pospart(Kres - Type(0.5)), 2);

  nll += pen; // add penalties to objective

  // -----------------------------
  // REPORTS
  // -----------------------------
  REPORT(cots_pred); // predicted adult COTS (individuals/m2)
  REPORT(fast_pred); // predicted fast coral (%)
  REPORT(slow_pred); // predicted slow coral (%)

  // Also report some transformed parameters for diagnostics
  REPORT(pref_f);
  REPORT(pref_s);
  REPORT(Kc);
  REPORT(Kres);
  REPORT(Tsd_C);
  REPORT(Tsd_F);
  REPORT(Tsd_S);
  REPORT(e_imm);
  REPORT(p_food);

  return nll;
}
