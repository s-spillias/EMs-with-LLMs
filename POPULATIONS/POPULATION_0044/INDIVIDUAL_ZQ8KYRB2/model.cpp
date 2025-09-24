#include <TMB.hpp>

// Template Model Builder: Episodic COTS outbreaks with coral feedbacks and SST/immigration forcing
// Notes:
// - Predictions use only previous-step response states (no data leakage).
// - Coral covers are kept strictly in (0,1) and sum < 1 via smooth normalization.
// - Smooth penalties used in place of hard parameter bounds.

// Helper smooth functions
template<class Type>
Type invlogit(const Type& x){ return Type(1) / (Type(1) + exp(-x)); }

template<class Type>
Type logit01(const Type& p_raw, const Type& eps){
  // Stable logit with clamping to (eps, 1-eps)
  Type p = CppAD::CondExpLt(p_raw, eps, eps, p_raw);
  p      = CppAD::CondExpGt(p, Type(1)-eps, Type(1)-eps, p);
  return log(p / (Type(1) - p));
}

template<class Type>
Type softplus_k(const Type& x, const Type& k){
  // Smooth approx to max(0, x); larger k -> sharper transition
  return log1p(exp(k * x)) / k;
}

template<class Type>
Type box_penalty(const Type& x, const Type& lo, const Type& hi, const Type& k, const Type& w){
  // Smooth penalty outside [lo, hi] using softplus on distances
  Type pen_lo = softplus_k(lo - x, k);
  Type pen_hi = softplus_k(x - hi, k);
  return w * (pen_lo * pen_lo + pen_hi * pen_hi);
}

template<class Type>
Type objective_function<Type>::operator()(){
  Type nll = 0.0;

  // -----------------------------
  // DATA (names must match *_dat; Year variable must match "Year")
  // -----------------------------
  DATA_VECTOR(Year);          // Year time index (calendar years; used only for alignment/reporting)
  DATA_VECTOR(sst_dat);       // Sea-surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);   // COTS larval immigration (ind m^-2 yr^-1)
  DATA_VECTOR(cots_dat);      // Observed adult COTS density (ind m^-2)
  DATA_VECTOR(fast_dat);      // Observed fast coral cover (%; Acropora)
  DATA_VECTOR(slow_dat);      // Observed slow coral cover (%; Faviidae/Porites)

  int n = Year.size();

  // -----------------------------
  // PARAMETERS
  // -----------------------------
  // Coral growth and competition
  PARAMETER(log_rF);            // ln intrinsic growth rate of fast coral (yr^-1); literature/expert; expected 0.1–1.5
  PARAMETER(log_rS);            // ln intrinsic growth rate of slow coral (yr^-1); slower than fast
  PARAMETER(alpha_FS_raw);      // raw competition coefficient S->F, transformed to [0,2]; dimensionless; expert/lit
  PARAMETER(alpha_SF_raw);      // raw competition coefficient F->S, transformed to [0,2]; dimensionless; expert/lit
  PARAMETER(log_gamma_F);       // ln space-recruitment rate into open substrate for fast coral (yr^-1)
  PARAMETER(log_gamma_S);       // ln space-recruitment rate into open substrate for slow coral (yr^-1)
  PARAMETER(log_muF);           // ln background mortality rate of fast coral (yr^-1)
  PARAMETER(log_muS);           // ln background mortality rate of slow coral (yr^-1)

  // COTS functional response on corals (Beddington–DeAngelis)
  PARAMETER(log_a_f);           // ln attack rate on fast coral (yr^-1); preference captured here
  PARAMETER(log_a_s);           // ln attack rate on slow coral (yr^-1)
  PARAMETER(log_h_f);           // ln handling time for fast coral (yr); contributes to saturation
  PARAMETER(log_h_s);           // ln handling time for slow coral (yr)
  PARAMETER(log_q_int);         // ln predator interference coefficient (m^2 ind^-1); increases denominator with COTS density

  // COTS demography and feedbacks
  PARAMETER(log_mC0);           // ln baseline adult COTS mortality (yr^-1)
  PARAMETER(log_m_foodmax);     // ln max extra mortality from food limitation (yr^-1)
  PARAMETER(log_rC0);           // ln local recruitment productivity scaling to adults (yr^-1)
  PARAMETER(log_KC);            // ln density scale for Beverton–Holt-type crowding (ind m^-2)
  PARAMETER(log_A_allee);       // ln Allee density (ind m^-2) for mate-finding; sets outbreak trigger threshold
  PARAMETER(log_k_allee);       // ln Allee smoothness scale (ind m^-2); larger -> smoother transition
  PARAMETER(log_KF);            // ln half-saturation of fast-coral effect on COTS recruitment (proportion cover)
  PARAMETER(phiF_raw);          // raw shape controlling steepness of fast-coral effect; maps to [0.5, 4.0]

  // Environmental effects
  PARAMETER(Topt_C);            // Optimal SST for COTS larval survival (deg C); literature 26–29 C
  PARAMETER(log_sigmaT_C);      // ln SD of Gaussian SST effect on recruitment (deg C)
  PARAMETER(T_bleach);          // SST where bleaching stress starts (deg C); literature ~29–30 C in GBR
  PARAMETER(log_beta_bleach);   // ln strength of bleaching effect on coral net growth (unitless multiplier exponent)
  PARAMETER(bleach_sens_fast_raw); // raw [0,1] sensitivity of fast coral to bleaching; transformed via invlogit
  PARAMETER(bleach_sens_slow_raw); // raw [0,1] sensitivity of slow coral to bleaching; transformed via invlogit

  // Immigration forcing scale
  PARAMETER(log_imm_scale);     // ln scale mapping cotsimm_dat (ind m^-2 yr^-1) into adult-equivalent recruits (yr^-1)

  // Initial conditions
  PARAMETER(log_C0);            // ln initial adult COTS density (ind m^-2)
  PARAMETER(logit_F0);          // logit initial fast coral proportion (0–1); from first observation or expert knowledge
  PARAMETER(logit_S0);          // logit initial slow coral proportion (0–1); from first observation or expert knowledge

  // Observation error (minimum SD enforced in computation)
  PARAMETER(log_sigma_cots);    // ln observation SD for log(COTS) (dimensionless)
  PARAMETER(log_sigma_fast);    // ln observation SD for logit(fast cover proportion) (dimensionless)
  PARAMETER(log_sigma_slow);    // ln observation SD for logit(slow cover proportion) (dimensionless)

  // -----------------------------
  // Transforms and constants
  // -----------------------------
  Type eps = Type(1e-8);             // Small constant to stabilize divisions and logs
  Type k_soft = Type(10.0);          // Softness parameter for softplus transitions
  Type min_sd = Type(0.05);          // Minimum SD for observation models (log/logit scales)

  // Transform parameters to natural scales
  Type rF = exp(log_rF);             // Fast coral growth (yr^-1)
  Type rS = exp(log_rS);             // Slow coral growth (yr^-1)
  Type alpha_FS = Type(2.0) * invlogit(alpha_FS_raw); // Competition S->F in [0,2]
  Type alpha_SF = Type(2.0) * invlogit(alpha_SF_raw); // Competition F->S in [0,2]
  Type gamma_F = exp(log_gamma_F);   // Space recruitment (yr^-1)
  Type gamma_S = exp(log_gamma_S);   // Space recruitment (yr^-1)
  Type muF = exp(log_muF);           // Background coral loss (yr^-1)
  Type muS = exp(log_muS);           // Background coral loss (yr^-1)

  Type a_f = exp(log_a_f);           // Attack on fast (yr^-1)
  Type a_s = exp(log_a_s);           // Attack on slow (yr^-1)
  Type h_f = exp(log_h_f);           // Handling time fast (yr)
  Type h_s = exp(log_h_s);           // Handling time slow (yr)
  Type q_int = exp(log_q_int);       // Predator interference (m^2 ind^-1)

  Type mC0 = exp(log_mC0);           // Baseline COTS mortality (yr^-1)
  Type m_foodmax = exp(log_m_foodmax); // Extra mortality when food scarce (yr^-1)
  Type rC0 = exp(log_rC0);           // Recruitment productivity scale (yr^-1)
  Type KC = exp(log_KC);             // Crowding scale (ind m^-2)
  Type A_allee = exp(log_A_allee);   // Allee threshold (ind m^-2)
  Type k_allee = exp(log_k_allee);   // Allee smoothness (ind m^-2)
  Type KF = exp(log_KF);             // Half-saturation for fast coral effect (proportion)
  Type phiF = Type(0.5) + Type(3.5) * invlogit(phiF_raw); // Shape in [0.5,4.0]

  Type sigmaT_C = exp(log_sigmaT_C); // SD of SST effect (deg C)
  Type beta_bleach = exp(log_beta_bleach); // Strength of bleaching effect (unitless in exponent)
  Type sensF = invlogit(bleach_sens_fast_raw); // [0,1] sensitivity fast
  Type sensS = invlogit(bleach_sens_slow_raw); // [0,1] sensitivity slow

  Type imm_scale = exp(log_imm_scale); // Scale on immigration (yr^-1)

  Type C0 = exp(log_C0);             // Initial COTS density (ind m^-2)
  Type F0 = invlogit(logit_F0);      // Initial fast coral proportion
  Type S0 = invlogit(logit_S0);      // Initial slow coral proportion

  Type sigma_cots = exp(log_sigma_cots); // Obs SD log scale
  Type sigma_fast = exp(log_sigma_fast); // Obs SD logit scale
  Type sigma_slow = exp(log_sigma_slow); // Obs SD logit scale

  // Enforce minimum SDs smoothly (sqrt(min^2 + sd^2) ~ max-like)
  sigma_cots = sqrt(min_sd * min_sd + sigma_cots * sigma_cots);
  sigma_fast = sqrt(min_sd * min_sd + sigma_fast * sigma_fast);
  sigma_slow = sqrt(min_sd * min_sd + sigma_slow * sigma_slow);

  // Smooth penalties to keep key environmental parameters in plausible ranges
  nll += box_penalty(Topt_C, Type(20.0), Type(32.0), k_soft, Type(1.0));     // Optimal SST 20–32 C
  nll += box_penalty(T_bleach, Type(27.0), Type(33.0), k_soft, Type(1.0));   // Bleaching onset 27–33 C

  // ------------------------------------------
  // Allocate state prediction vectors
  // ------------------------------------------
  vector<Type> cots_pred(n); // COTS prediction (ind m^-2) for cots_dat
  vector<Type> fast_pred(n); // Fast coral prediction (%) for fast_dat
  vector<Type> slow_pred(n); // Slow coral prediction (%) for slow_dat

  // Helper lambda for normalization to keep coral covers in (0,1) and sum < 1:
  auto normalize_coral = [&](Type F_raw, Type S_raw){
    // Ensure non-negativity smoothly
    Type F_pos = softplus_k(F_raw, k_soft);
    Type S_pos = softplus_k(S_raw, k_soft);
    // Smoothly allocate space so F,S in (0,1) and F+S < 1
    Type denom = Type(1.0) + F_pos + S_pos; // leaves 1/(1+F+S) as free space
    Type F_out = F_pos / denom;
    Type S_out = S_pos / denom;
    return std::pair<Type,Type>(F_out, S_out);
  };

  // ------------------------------------------
  // Equations (discrete-time; t = 0...(n-1))
  //
  // (1) Coral logistic growth with competition and space-limited recruitment:
  //     dF = rF * F * (1 - (F + alpha_FS * S)) + gamma_F * L - muF * F
  //     dS = rS * S * (1 - (S + alpha_SF * F)) + gamma_S * L - muS * S
  //     where L = 1 - F - S (open space).
  //
  // (2) Bleaching modifies coral net growth via SST (smooth threshold):
  //     warm = softplus(SST - T_bleach); g_bleach = exp(-beta_bleach * warm)
  //     Apply: growth_terms *= g_bleach; extra_loss = sens * (1 - g_bleach) * coral
  //
  // (3) COTS consumption on corals: Beddington–DeAngelis functional response:
  //     Den = 1 + a_f*h_f*F + a_s*h_s*S + q_int * C
  //     ConsF = a_f * C * F / Den
  //     ConsS = a_s * C * S / Den
  //
  // (4) COTS adult dynamics:
  //     Food effect for recruitment: gF = F^phiF / (F^phiF + KF^phiF)
  //     SST recruitment effect: envT = exp(-0.5 * ((SST - Topt_C)/sigmaT_C)^2)
  //     Allee: A(C) = invlogit((C - A_allee)/k_allee)
  //     Crowding: BH = 1 / (1 + C / KC)
  //     Survival: surv = exp( - (mC0 + m_foodmax * (1 - gF)) )
  //     Recruits_local = rC0 * C * A(C) * gF * envT * BH
  //     Immigration = imm_scale * cotsimm_dat
  //     C_next = surv * C + Recruits_local + Immigration
  //
  // (5) Observation models:
  //     COTS: log(cots_dat) ~ Normal( log(cots_pred), sigma_cots )
  //     Coral: logit(fast_dat/100) ~ Normal( logit(fast_pred/100), sigma_fast ), similarly for slow
  // ------------------------------------------

  // Initialize states at t = 0
  // Normalize initial coral proportions smoothly to ensure feasibility
  std::pair<Type,Type> init_norm = normalize_coral(F0, S0);
  Type F_prev = init_norm.first;   // Fast coral proportion at t=0
  Type S_prev = init_norm.second;  // Slow coral proportion at t=0
  Type C_prev = C0;                // COTS at t=0 (ind m^-2)

  // Store t=0 predictions (report in same units as data)
  cots_pred(0) = C_prev;
  fast_pred(0) = F_prev * Type(100.0); // percent
  slow_pred(0) = S_prev * Type(100.0); // percent

  // Likelihood t=0
  // COTS (lognormal)
  nll -= dnorm(log(cots_dat(0) + eps), log(cots_pred(0) + eps), sigma_cots, true);
  // Fast coral (logit-normal)
  {
    Type y = (fast_dat(0) / Type(100.0));
    Type eta_y = logit01(y, eps);
    Type eta_hat = logit01(fast_pred(0) / Type(100.0), eps);
    nll -= dnorm(eta_y, eta_hat, sigma_fast, true);
  }
  // Slow coral (logit-normal)
  {
    Type y = (slow_dat(0) / Type(100.0));
    Type eta_y = logit01(y, eps);
    Type eta_hat = logit01(slow_pred(0) / Type(100.0), eps);
    nll -= dnorm(eta_y, eta_hat, sigma_slow, true);
  }

  // Time stepping for t = 1..n-1
  for(int t = 1; t < n; t++){
    // Forcing at previous step to avoid leakage of response variables (environment can be current without issue,
    // but we consistently use previous-year forcing for clarity)
    Type sst_prev = sst_dat(t-1);
    Type imm_prev = cotsimm_dat(t-1);

    // Open space based on previous state
    Type L_prev = Type(1.0) - F_prev - S_prev;
    L_prev = CppAD::CondExpLt(L_prev, eps, eps, L_prev); // keep non-negative

    // Bleaching modifier (smooth threshold)
    Type warm = softplus_k(sst_prev - T_bleach, k_soft);      // >= 0
    Type g_bleach = exp(-beta_bleach * warm);                 // in (0,1]

    // Functional response denominator (Beddington–DeAngelis)
    Type Den = Type(1.0)
             + a_f * h_f * F_prev
             + a_s * h_s * S_prev
             + q_int * C_prev
             + eps;

    // Coral predation losses
    Type ConsF = a_f * C_prev * F_prev / Den;                 // Fast coral loss (proportion/yr)
    Type ConsS = a_s * C_prev * S_prev / Den;                 // Slow coral loss

    // Coral net growth (logistic + space recruitment - background mortality), modulated by bleaching
    Type growF = rF * F_prev * (Type(1.0) - (F_prev + alpha_FS * S_prev));
    Type growS = rS * S_prev * (Type(1.0) - (S_prev + alpha_SF * F_prev));
    Type recF  = gamma_F * L_prev;
    Type recS  = gamma_S * L_prev;
    Type netF_mod = g_bleach * (growF + recF) - muF * F_prev - sensF * (Type(1.0) - g_bleach) * F_prev;
    Type netS_mod = g_bleach * (growS + recS) - muS * S_prev - sensS * (Type(1.0) - g_bleach) * S_prev;

    // Coral raw updates before feasibility normalization
    Type F_raw = F_prev + netF_mod - ConsF;
    Type S_raw = S_prev + netS_mod - ConsS;

    // COTS reproduction modifiers
    Type F_eff = pow(F_prev + eps, phiF) / (pow(F_prev + eps, phiF) + pow(KF + eps, phiF)); // food effect in [0,1]
    Type envT  = exp( - Type(0.5) * pow( (sst_prev - Topt_C) / (sigmaT_C + eps), Type(2.0) ) ); // Gaussian SST window
    Type Aeff  = invlogit( (C_prev - A_allee) / (k_allee + eps) ); // smooth Allee 0..1
    Type BH    = Type(1.0) / (Type(1.0) + C_prev / (KC + eps));    // crowding

    // COTS survival and recruitment
    Type surv  = exp( - (mC0 + m_foodmax * (Type(1.0) - F_eff)) ); // food-limited extra mortality
    Type Recr_local = rC0 * C_prev * Aeff * F_eff * envT * BH;     // new adults from local production
    Type Imm = imm_scale * imm_prev;                                // exogenous recruits (adults/yr)

    // COTS update
    Type C_raw = surv * C_prev + Recr_local + Imm;
    Type C_next = softplus_k(C_raw, k_soft);                        // keep positive smoothly

    // Normalize coral states to (0,1) and sum<1
    std::pair<Type,Type> norm_next = normalize_coral(F_raw, S_raw);
    Type F_next = norm_next.first;
    Type S_next = norm_next.second;

    // Save predictions in data units
    cots_pred(t) = C_next;
    fast_pred(t) = F_next * Type(100.0);
    slow_pred(t) = S_next * Type(100.0);

    // Advance state
    F_prev = F_next;
    S_prev = S_next;
    C_prev = C_next;

    // Likelihood at time t
    // COTS (lognormal)
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots, true);
    // Fast coral (logit-normal)
    {
      Type y = (fast_dat(t) / Type(100.0));
      Type eta_y = logit01(y, eps);
      Type eta_hat = logit01(fast_pred(t) / Type(100.0), eps);
      nll -= dnorm(eta_y, eta_hat, sigma_fast, true);
    }
    // Slow coral (logit-normal)
    {
      Type y = (slow_dat(t) / Type(100.0));
      Type eta_y = logit01(y, eps);
      Type eta_hat = logit01(slow_pred(t) / Type(100.0), eps);
      nll -= dnorm(eta_y, eta_hat, sigma_slow, true);
    }
  }

  // REPORT predictions (must include all *_pred)
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  // Also report transformed parameters useful for interpretation
  REPORT(rF);
  REPORT(rS);
  REPORT(alpha_FS);
  REPORT(alpha_SF);
  REPORT(gamma_F);
  REPORT(gamma_S);
  REPORT(muF);
  REPORT(muS);
  REPORT(a_f);
  REPORT(a_s);
  REPORT(h_f);
  REPORT(h_s);
  REPORT(q_int);
  REPORT(mC0);
  REPORT(m_foodmax);
  REPORT(rC0);
  REPORT(KC);
  REPORT(A_allee);
  REPORT(k_allee);
  REPORT(KF);
  REPORT(phiF);
  REPORT(sigmaT_C);
  REPORT(beta_bleach);
  REPORT(sensF);
  REPORT(sensS);
  REPORT(imm_scale);
  REPORT(C0);
  REPORT(F0);
  REPORT(S0);
  REPORT(sigma_cots);
  REPORT(sigma_fast);
  REPORT(sigma_slow);

  return nll;
}
