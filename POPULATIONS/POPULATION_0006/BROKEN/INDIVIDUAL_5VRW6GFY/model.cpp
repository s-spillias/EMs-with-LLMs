#include <TMB.hpp>

// Utility functions for smooth, stable calculations
template<class Type>
Type inv_logit(Type x) {
  return Type(1) / (Type(1) + exp(-x));
}

template<class Type>
Type softplus(Type x, Type k) { // smooth positive part; k controls sharpness
  return log(Type(1) + exp(k * x)) / k;
}

// Smooth penalty for keeping a parameter within [low, high]
template<class Type>
Type soft_bound_penalty(Type x, Type low, Type high, Type w) {
  // Adds penalty when x < low or x > high; smooth via softplus with unit sharpness
  Type pen_low  = softplus(low - x, Type(1.0));
  Type pen_high = softplus(x - high, Type(1.0));
  return w * (pen_low + pen_high);
}

/*
Ecological equations (all transitions t-1 -> t use previous predictions only):

Let:
- x_t  = COTS density (individuals/m2) at time t
- a_t  = Fast coral proportion (Acropora; 0..1) at time t
- s_t  = Slow coral proportion (Faviidae/Porites; 0..1) at time t
- E_t  = Free space proportion at time t
- SST_t = sea surface temperature (deg C) at time t
- IMM_t = larval immigration rate (individuals/m2/year) at time t

1) Space limitation (smooth):
   E_{t-1} = softplus(1 - a_{t-1} - s_{t-1}, k_space)

2) Coral growth (resource-limited, SST-modified):
   gA_{t-1} = rA * a_{t-1} * E_{t-1} * exp(beta_sst_A * (SST_{t-1} - sst_ref))
   gS_{t-1} = rS * s_{t-1} * E_{t-1} * exp(beta_sst_S * (SST_{t-1} - sst_ref))

3) Bleaching mortality (smooth thermal threshold):
   heat_{t-1} = inv_logit(k_bleach * (SST_{t-1} - sst_bleach))
   bA_{t-1} = m_bleach_A * heat_{t-1} * a_{t-1}
   bS_{t-1} = m_bleach_S * heat_{t-1} * s_{t-1}

4) COTS predation on corals (multi-prey Holling type III with predator saturation):
   Preferences (softmax): wA = exp(pref_A)/(exp(pref_A)+exp(pref_S)); wS = 1 - wA
   Type III availability: phiA = a_{t-1}^q / (hA^q + a_{t-1}^q); phiS similarly
   Predator saturation: satX = x_{t-1} / (x_{t-1} + x_half)
   Total potential loss (bounded by available coral): L_tot = m_eat * (a_{t-1} + s_{t-1}) * satX
   Allocation by availability and preference:
     shareA = wA*phiA; shareS = wS*phiS; Z = shareA + shareS + eps
     consA_{t-1} = L_tot * shareA / Z
     consS_{t-1} = L_tot * shareS / Z

5) Interspecific competition for space (indirect crowding):
   cA_{t-1} = comp_sf * a_{t-1} * s_{t-1}   (slow crowding fast)
   cS_{t-1} = comp_fs * a_{t-1} * s_{t-1}   (fast crowding slow)

6) Coral state updates:
   a_t = a_{t-1} + gA_{t-1} - bA_{t-1} - consA_{t-1} - cA_{t-1}
   s_t = s_{t-1} + gS_{t-1} - bS_{t-1} - consS_{t-1} - cS_{t-1}

7) COTS recruitment (food threshold + SST modifier) and mortality:
   prey_eff_{t-1} = inv_logit(k_prey * ( (a_{t-1} + s_{t-1}) - prey_thr ))
   R_{t-1} = rC * x_{t-1} * prey_eff_{t-1} * exp(beta_sst_C * (SST_{t-1} - sst_ref))
   I_{t-1} = psi_imm * IMM_{t-1}
   F_{t-1} = gamma_food * (consA_{t-1} + consS_{t-1})          (conversion from consumed coral to adult COTS)
   M_{t-1} = mC_nat * x_{t-1} + mC_dd * x_{t-1}^2               (natural + density-dependent mortality)
   x_t = x_{t-1} + R_{t-1} + I_{t-1} + F_{t-1} - M_{t-1}

Observation models (applied at each t):
- COTS: log(cots_dat) ~ Normal(log(cots_pred), sigma_cots) with sigma_cots >= sigma_min
- Corals (%): Use proportions p in (0,1) and logit-normal:
    logit(fast_dat/100) ~ Normal(logit(fast_pred/100), sigma_fast) with sigma_fast >= sigma_min
    logit(slow_dat/100) ~ Normal(logit(slow_pred/100), sigma_slow) with sigma_slow >= sigma_min
All likelihoods computed for all observations with small epsilons to avoid numerical issues.
*/

template<class Type>
Type objective_function<Type>::operator() ()
{
  Type nll = 0.0;
  // Small constants for numerical stability
  const Type eps = Type(1e-8);
  const Type k_space = Type(20.0);   // sharpness for softplus of free space
  const Type sigma_min = Type(0.05); // minimum observation SD
  const Type pen_w = Type(10.0);     // weight for soft bound penalties on states
  const Type pen_p = Type(1.0);      // weight for soft bound penalties on parameters

  //====================
  // Data
  //====================
  DATA_VECTOR(Year);         // Year (calendar year)
  DATA_VECTOR(cots_dat);     // Adult COTS (individuals/m2)
  DATA_VECTOR(fast_dat);     // Fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);     // Slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);      // Sea Surface Temperature (deg C)
  DATA_VECTOR(cotsimm_dat);  // COTS larval immigration (individuals/m2/year)

  //====================
  // Parameters
  //====================
  // Coral intrinsic growth (year^-1), log-scale to ensure >0
  PARAMETER(rA_log);        // log intrinsic growth of fast coral (year^-1); prior range ~ log(0.05..1.5)
  PARAMETER(rS_log);        // log intrinsic growth of slow coral (year^-1); prior range ~ log(0.02..0.5)

  // SST effects on coral growth (per °C)
  PARAMETER(beta_sst_A);    // sensitivity of fast coral growth to SST deviation (1/°C)
  PARAMETER(beta_sst_S);    // sensitivity of slow coral growth to SST deviation (1/°C)

  // Bleaching parameters
  PARAMETER(m_bleach_A_logit); // logit of max yearly bleaching mortality fraction for fast (0..1)
  PARAMETER(m_bleach_S_logit); // logit of max yearly bleaching mortality fraction for slow (0..1)
  PARAMETER(k_bleach_log);     // log steepness of bleaching logistic (unitless)
  PARAMETER(sst_bleach);       // °C threshold around which bleaching accelerates
  // COTS predation parameters
  PARAMETER(pref_A);        // preference score for fast coral (softmaxed)
  PARAMETER(pref_S);        // preference score for slow coral (softmaxed)
  PARAMETER(q_fr_log);      // log exponent q for type III functional response (>=1 via 1+exp)
  PARAMETER(hA_logit);      // logit of half-saturation (proportion) for fast coral (0..1)
  PARAMETER(hS_logit);      // logit of half-saturation (proportion) for slow coral (0..1)
  PARAMETER(m_eat_logit);   // logit of max yearly fractional coral loss due to predation (0..1)
  PARAMETER(x_half_log);    // log of half-saturation for predator saturation (individuals/m2)

  // Coral competition coefficients
  PARAMETER(comp_sf_log);   // log competition coefficient: slow crowding fast (year^-1)
  PARAMETER(comp_fs_log);   // log competition coefficient: fast crowding slow (year^-1)

  // COTS demographic parameters
  PARAMETER(rC_log);        // log per-capita recruitment rate (year^-1)
  PARAMETER(beta_sst_C);    // SST effect on COTS recruitment (1/°C)
  PARAMETER(k_prey_log);    // log steepness of prey threshold (unitless)
  PARAMETER(prey_thr_logit);// logit of total coral proportion where recruitment accelerates (0..1)
  PARAMETER(gamma_food_log);// log conversion from consumed coral fraction to adult COTS (ind/m2 per unit fraction)
  PARAMETER(mC_nat_log);    // log natural mortality (year^-1)
  PARAMETER(mC_dd_log);     // log density-dependent mortality (m2/ind/year)
  PARAMETER(psi_imm_log);   // log scale on immigration (dimensionless multiplier)

  // Observation error parameters
  PARAMETER(log_sigma_cots); // log SD for lognormal COTS observation errors
  PARAMETER(log_sigma_fast); // log SD for logit-normal fast coral obs errors
  PARAMETER(log_sigma_slow); // log SD for logit-normal slow coral obs errors

  // Reference SST (°C)
  PARAMETER(sst_ref);       // reference temperature for rate modifiers (°C)

  //====================
  // Transform parameters to natural scales
  //====================
  Type rA = exp(rA_log);                 // year^-1
  Type rS = exp(rS_log);                 // year^-1
  Type m_bleach_A = inv_logit(m_bleach_A_logit); // 0..1 fraction/year
  Type m_bleach_S = inv_logit(m_bleach_S_logit); // 0..1 fraction/year
  Type k_bleach = exp(k_bleach_log);     // >0
  Type q = Type(1.0) + exp(q_fr_log);    // >=1
  Type hA = inv_logit(hA_logit);         // 0..1
  Type hS = inv_logit(hS_logit);         // 0..1
  Type m_eat = inv_logit(m_eat_logit);   // 0..1
  Type x_half = exp(x_half_log);         // individuals/m2
  Type comp_sf = exp(comp_sf_log);       // year^-1
  Type comp_fs = exp(comp_fs_log);       // year^-1
  Type rC = exp(rC_log);                 // year^-1
  Type k_prey = exp(k_prey_log);         // >0
  Type prey_thr = inv_logit(prey_thr_logit); // 0..1
  Type gamma_food = exp(gamma_food_log); // ind/m2 per unit coral fraction consumed
  Type mC_nat = exp(mC_nat_log);         // year^-1
  Type mC_dd = exp(mC_dd_log);           // m2/ind/year
  Type psi_imm = exp(psi_imm_log);       // dimensionless
  // Preferences to weights
  Type wA = exp(pref_A) / (exp(pref_A) + exp(pref_S) + eps); // unitless, in (0,1)
  Type wS = Type(1.0) - wA;

  // Observation SDs with minimum
  Type sigma_cots = exp(log_sigma_cots) + sigma_min; // > sigma_min
  Type sigma_fast = exp(log_sigma_fast) + sigma_min; // > sigma_min
  Type sigma_slow = exp(log_sigma_slow) + sigma_min; // > sigma_min

  int n = cots_dat.size();

  //====================
  // State vectors (predictions)
  //====================
  vector<Type> cots_pred(n); // individuals/m2
  vector<Type> fast_pred(n); // %
  vector<Type> slow_pred(n); // %

  // Auxiliary reporting vectors (diagnostics)
  vector<Type> consA(n);      // proportion/year
  vector<Type> consS(n);      // proportion/year
  vector<Type> heat_stress(n);// unitless 0..1
  vector<Type> prey_eff(n);   // unitless 0..1
  vector<Type> free_space(n); // proportion

  //====================
  // Initial conditions from first observations (no estimation; avoids data leakage)
  //====================
  cots_pred(0) = cots_dat(0);       // individuals/m2
  fast_pred(0) = fast_dat(0);       // %
  slow_pred(0) = slow_dat(0);       // %

  // Penalty on initial bounds if outside plausible ranges (soft)
  {
    Type a0 = fast_pred(0) / Type(100.0);
    Type s0 = slow_pred(0) / Type(100.0);
    nll += pen_w * (softplus(-a0, Type(5.0)) + softplus(a0 - Type(1.0), Type(5.0))
                 +  softplus(-s0, Type(5.0)) + softplus(s0 - Type(1.0), Type(5.0)));
    nll += pen_w * softplus(-cots_pred(0), Type(5.0));
  }

  //====================
  // Time loop
  //====================
  for (int t = 1; t < n; t++) {
    // Previous states (predictions only)
    Type x_prev = cots_pred(t-1);             // individuals/m2
    Type a_prev = fast_pred(t-1) / Type(100.0); // proportion
    Type s_prev = slow_pred(t-1) / Type(100.0); // proportion

    // Environmental drivers at t-1
    Type sst_prev = sst_dat(t-1);
    Type imm_prev = cotsimm_dat(t-1);

    // (1) Free space (smooth positive part)
    Type E_prev = softplus(Type(1.0) - a_prev - s_prev, Type(20.0)); // <= 1 approximately
    free_space(t-1) = E_prev;

    // (2) Coral intrinsic growth, SST-modified
    Type gA = rA * a_prev * E_prev * exp(beta_sst_A * (sst_prev - sst_ref)); // proportion/year
    Type gS = rS * s_prev * E_prev * exp(beta_sst_S * (sst_prev - sst_ref)); // proportion/year

    // (3) Bleaching impact (smooth threshold)
    Type heat = inv_logit(k_bleach * (sst_prev - sst_bleach)); // 0..1
    heat_stress(t-1) = heat;
    Type bA = m_bleach_A * heat * a_prev; // proportion/year
    Type bS = m_bleach_S * heat * s_prev; // proportion/year

    // (4) COTS predation: multi-prey, type III, predator saturation
    Type phiA = pow(a_prev + eps, q) / (pow(hA + eps, q) + pow(a_prev + eps, q));
    Type phiS = pow(s_prev + eps, q) / (pow(hS + eps, q) + pow(s_prev + eps, q));
    Type shareA = wA * phiA;
    Type shareS = wS * phiS;
    Type Z = shareA + shareS + eps;
    Type satX = x_prev / (x_prev + x_half + eps);
    Type L_tot = m_eat * (a_prev + s_prev) * satX; // proportion/year, bounded by m_eat*(a+s)
    Type consA_prev = L_tot * shareA / Z; // proportion/year taken from fast
    Type consS_prev = L_tot * shareS / Z; // proportion/year taken from slow
    consA(t-1) = consA_prev;
    consS(t-1) = consS_prev;

    // (5) Competition (indirect crowding)
    Type cA = comp_sf * a_prev * s_prev; // proportion/year
    Type cS = comp_fs * a_prev * s_prev; // proportion/year

    // (6) Coral updates (t-1 -> t)
    Type a_next = a_prev + gA - bA - consA_prev - cA;
    Type s_next = s_prev + gS - bS - consS_prev - cS;

    // Soft penalties to discourage leaving [0,1]
    nll += pen_w * (softplus(-a_next, Type(5.0)) + softplus(a_next - Type(1.0), Type(5.0))
                  + softplus(-s_next, Type(5.0)) + softplus(s_next - Type(1.0), Type(5.0)));

    // (7) COTS recruitment/mortality (t-1 -> t)
    Type prey_effect = inv_logit(k_prey * ((a_prev + s_prev) - prey_thr)); // 0..1
    prey_eff(t-1) = prey_effect;
    Type R = rC * x_prev * prey_effect * exp(beta_sst_C * (sst_prev - sst_ref)); // ind/m2/year
    Type I = psi_imm * imm_prev;                       // ind/m2/year
    Type F = gamma_food * (consA_prev + consS_prev);   // ind/m2/year
    Type M = mC_nat * x_prev + mC_dd * x_prev * x_prev;// ind/m2/year
    Type x_next = x_prev + R + I + F - M;

    // Soft penalty to discourage negative COTS density
    nll += pen_w * softplus(-x_next, Type(5.0));

    // Save predictions (convert coral back to %)
    cots_pred(t) = x_next;
    fast_pred(t) = a_next * Type(100.0);
    slow_pred(t) = s_next * Type(100.0);
  }

  // Fill last-step diagnostics for completeness
  free_space(n-1) = softplus(Type(1.0) - fast_pred(n-1)/Type(100.0) - slow_pred(n-1)/Type(100.0), Type(20.0));
  heat_stress(n-1) = inv_logit(exp(k_bleach_log) * (sst_dat(n-1) - sst_bleach));
  prey_eff(n-1) = inv_logit(exp(k_prey_log) * ((fast_pred(n-1)+slow_pred(n-1))/Type(100.0) - prey_thr));
  consA(n-1) = consA(n-2); // repeat previous (not used in likelihood)
  consS(n-1) = consS(n-2);

  //====================
  // Observation likelihood
  //====================
  for (int t = 0; t < n; t++) {
    // COTS: lognormal
    Type y_c = cots_dat(t);
    Type mu_c = cots_pred(t);
    // Safeguard small/negative values in logs (always include observation)
    Type ylog = log(y_c + eps);
    Type mulog = log(mu_c + eps);
    nll -= dnorm(ylog, mulog, sigma_cots, true);

    // Coral fast: logit-normal on proportions
    Type y_f = fast_dat(t) / Type(100.0);
    Type mu_f = fast_pred(t) / Type(100.0);
    Type zf_y = logit(CppAD::CondExpLt(y_f, eps, eps, CppAD::CondExpGt(y_f, Type(1.0)-eps, Type(1.0)-eps, y_f)));
    Type zf_mu = logit(CppAD::CondExpLt(mu_f, eps, eps, CppAD::CondExpGt(mu_f, Type(1.0)-eps, Type(1.0)-eps, mu_f)));
    nll -= dnorm(zf_y, zf_mu, sigma_fast, true);

    // Coral slow: logit-normal on proportions
    Type y_s = slow_dat(t) / Type(100.0);
    Type mu_s = slow_pred(t) / Type(100.0);
    Type zs_y = logit(CppAD::CondExpLt(y_s, eps, eps, CppAD::CondExpGt(y_s, Type(1.0)-eps, Type(1.0)-eps, y_s)));
    Type zs_mu = logit(CppAD::CondExpLt(mu_s, eps, eps, CppAD::CondExpGt(mu_s, Type(1.0)-eps, Type(1.0)-eps, mu_s)));
    nll -= dnorm(zs_y, zs_mu, sigma_slow, true);
  }

  //====================
  // Soft parameter bounds (biologically motivated)
  //====================
  nll += soft_bound_penalty(rA,      Type(0.02), Type(1.5), pen_p);      // fast coral growth
  nll += soft_bound_penalty(rS,      Type(0.01), Type(0.6), pen_p);      // slow coral growth
  nll += soft_bound_penalty(beta_sst_A, Type(-0.2), Type(0.2), pen_p);   // SST effect bounds
  nll += soft_bound_penalty(beta_sst_S, Type(-0.2), Type(0.2), pen_p);
  nll += soft_bound_penalty(m_bleach_A, Type(0.0), Type(1.0), pen_p);
  nll += soft_bound_penalty(m_bleach_S, Type(0.0), Type(1.0), pen_p);
  nll += soft_bound_penalty(k_bleach, Type(0.1), Type(50.0), pen_p);
  nll += soft_bound_penalty(hA,      Type(0.01), Type(0.8), pen_p);
  nll += soft_bound_penalty(hS,      Type(0.01), Type(0.8), pen_p);
  nll += soft_bound_penalty(m_eat,   Type(0.01), Type(0.9), pen_p);
  nll += soft_bound_penalty(x_half,  Type(0.05), Type(20.0), pen_p);
  nll += soft_bound_penalty(comp_sf, Type(0.0),  Type(2.0), pen_p);
  nll += soft_bound_penalty(comp_fs, Type(0.0),  Type(2.0), pen_p);
  nll += soft_bound_penalty(rC,      Type(0.0),  Type(5.0), pen_p);
  nll += soft_bound_penalty(beta_sst_C, Type(-0.2), Type(0.4), pen_p);
  nll += soft_bound_penalty(k_prey,  Type(0.1),  Type(50.0), pen_p);
  nll += soft_bound_penalty(prey_thr,Type(0.05), Type(0.9), pen_p);
  nll += soft_bound_penalty(gamma_food, Type(0.0), Type(10.0), pen_p);
  nll += soft_bound_penalty(mC_nat,  Type(0.05), Type(3.0), pen_p);
  nll += soft_bound_penalty(mC_dd,   Type(0.0),  Type(5.0), pen_p);
  nll += soft_bound_penalty(psi_imm, Type(0.0),  Type(5.0), pen_p);

  //====================
  // Reporting
  //====================
  REPORT(cots_pred);  // predicted COTS (individuals/m2)
  REPORT(fast_pred);  // predicted fast coral cover (%)
  REPORT(slow_pred);  // predicted slow coral cover (%)
  REPORT(consA);      // consumption from fast coral (proportion/year)
  REPORT(consS);      // consumption from slow coral (proportion/year)
  REPORT(heat_stress);// thermal stress index 0..1
  REPORT(prey_eff);   // prey-driven recruitment efficiency 0..1
  REPORT(free_space); // free space 0..1
  REPORT(wA);         // preference weight fast
  REPORT(wS);         // preference weight slow
  REPORT(q);          // type III exponent
  REPORT(hA);         // half-saturation fast
  REPORT(hS);         // half-saturation slow
  REPORT(m_eat);      // max coral loss fraction
  REPORT(x_half);     // predator saturation half-density
  REPORT(rA); REPORT(rS);
  REPORT(rC);
  REPORT(mC_nat); REPORT(mC_dd);
  REPORT(gamma_food); REPORT(psi_imm);
  REPORT(sigma_cots); REPORT(sigma_fast); REPORT(sigma_slow);

  return nll;
}
