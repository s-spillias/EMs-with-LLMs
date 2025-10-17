#include <TMB.hpp>  // TMB header providing automatic differentiation and likelihood utilities
using namespace density;

// Helper: numerically stable softplus function for smooth penalties
template<class Type>
Type softplus_stable(Type x) {
  // Use a numerically stable softplus:
  // For x > 0: softplus(x) = x + log(1 + exp(-x))  -> avoids overflow and cancellation
  // For x <= 0: softplus(x) = log(1 + exp(x))      -> safe as exp(x) <= 1
  Type pos_branch = x + log(Type(1.0) + exp(-x));
  Type neg_branch = log(Type(1.0) + exp(x));
  return CppAD::CondExpGt(x, Type(0.0), pos_branch, neg_branch);
}

// Helper: smooth penalty that increases when x is below lo or above hi (both optional)
// w controls strength (higher = stronger penalty); penalty is zero-ish inside bounds.
template<class Type>
Type smooth_bounds_penalty(Type x, Type lo, Type hi, Type w) {
  Type pen_lo = softplus_stable(lo - x); // penalize when x << lo
  Type pen_hi = softplus_stable(x - hi); // penalize when x >> hi
  return w * (pen_lo + pen_hi);          // smooth, differentiable penalty
}

// Helper: clamp x to be >= lo in a differentiable way
template<class Type>
Type clamp_min(Type x, Type lo) {
  // If x < lo, return lo; else x. Uses conditional expression to keep AD graph smooth.
  return CppAD::CondExpLt(x, lo, lo, x);
}

template<class Type>
Type objective_function<Type>::operator() () {
  // --------------------------
  // DATA (READ-ONLY INPUTS)
  // --------------------------
  DATA_VECTOR(Time);    // time vector (days); corresponds to CSV column "Time (days)"
  DATA_VECTOR(N_dat);   // observed nutrient (g C m^-3)
  DATA_VECTOR(P_dat);   // observed phytoplankton (g C m^-3)
  DATA_VECTOR(Z_dat);   // observed zooplankton (g C m^-3)

  // --------------------------
  // PARAMETERS (TO ESTIMATE)
  // --------------------------
  PARAMETER(mu_max);        // day^-1 | Maximum phytoplankton specific growth rate at reference conditions
  PARAMETER(K_N);           // g C m^-3 | Half-saturation constant for nutrient uptake (Monod)
  PARAMETER(L_avail);       // dimensionless (0-1) | Effective light availability in mixed layer
  PARAMETER(K_L);           // dimensionless | Half-saturation constant for light limitation
  PARAMETER(alpha_colim);   // dimensionless (negative) | Smooth-min exponent (approximate Liebig’s minimum)
  PARAMETER(Q10);           // dimensionless | Q10 temperature coefficient for growth
  PARAMETER(T);             // deg C | Mixed layer temperature
  PARAMETER(T_ref);         // deg C | Reference temperature for Q10
  PARAMETER(g_max);         // day^-1 | Maximum zooplankton specific grazing rate
  PARAMETER(K_P);           // g C m^-3 | Half-saturation constant for zooplankton grazing
  PARAMETER(h);             // dimensionless | Hill exponent: 1=Type II, >1=Type III-like
  PARAMETER(AE_Z);          // dimensionless (0-1) | Zooplankton assimilation efficiency
  PARAMETER(m_P);           // day^-1 | Phytoplankton linear mortality/exudation
  PARAMETER(m_ZL);          // day^-1 | Zooplankton linear mortality/excretion
  PARAMETER(gamma_Z);       // (g C m^-3)^-1 day^-1 | Zooplankton quadratic mortality
  PARAMETER(r_PN);          // dimensionless (0-1) | Fraction of P mortality remineralized to N
  PARAMETER(r_ZN);          // dimensionless (0-1) | Fraction of Z mortality remineralized to N
  PARAMETER(ex_Z);          // day^-1 | Zooplankton excretion to dissolved nutrients
  PARAMETER(log_sigma_N);   // log-scale | Log observation SD for N (lognormal observation model)
  PARAMETER(log_sigma_P);   // log-scale | Log observation SD for P
  PARAMETER(log_sigma_Z);   // log-scale | Log observation SD for Z
  // New parameters: mixed-layer exchange/dilution
  PARAMETER(D_mix);         // day^-1 | Mixed-layer exchange/dilution rate
  PARAMETER(N_in);          // g C m^-3 | Nutrient concentration in inflowing/source water
  // New parameter: egestion-driven instant remineralization of unassimilated ingestion
  PARAMETER(r_egN);         // dimensionless (0-1) | Fraction of (1 - AE_Z) * grazing returned directly to nutrients

  // --------------------------
  // CONSTANTS AND ALIASES
  // --------------------------
  int n = N_dat.size();                   // number of time steps (should match length of Time, P_dat, Z_dat)
  Type tiny = Type(1e-12);                // small constant to avoid division by zero and log(0)
  Type sd_floor = Type(0.02);             // minimum observation SD added after exponentiation to prevent numerical issues

  // Observation standard deviations with fixed floor (always positive)
  Type sigma_N = exp(log_sigma_N) + sd_floor;  // SD for log(N)
  Type sigma_P = exp(log_sigma_P) + sd_floor;  // SD for log(P)
  Type sigma_Z = exp(log_sigma_Z) + sd_floor;  // SD for log(Z)

  // --------------------------
  // STATE VECTORS (PREDICTIONS)
  // --------------------------
  vector<Type> N_pred(n);  // predicted nutrient (g C m^-3)
  vector<Type> P_pred(n);  // predicted phytoplankton (g C m^-3)
  vector<Type> Z_pred(n);  // predicted zooplankton (g C m^-3)

  // Optional diagnostic outputs (process rates)
  vector<Type> uptake_pred(n);   // phytoplankton gross growth (g C m^-3 d^-1)
  vector<Type> grazing_pred(n);  // zooplankton grazing on P (g C m^-3 d^-1)
  vector<Type> remin_N_pred(n);  // total remineralization to N (g C m^-3 d^-1)
  vector<Type> temp_mod_pred(n); // temperature modifier (dimensionless)
  vector<Type> fN_pred(n);       // nutrient limitation term (0-1)
  vector<Type> fL_pred(n);       // light limitation term (0-1)
  vector<Type> co_lim_pred(n);   // co-limitation term (0-1)
  // New diagnostics for exchange/dilution
  vector<Type> mix_in_N_pred(n); // D_mix*(N_in - N) (g C m^-3 d^-1)
  vector<Type> dilP_pred(n);     // D_mix*P (g C m^-3 d^-1)
  vector<Type> dilZ_pred(n);     // D_mix*Z (g C m^-3 d^-1)
  // New diagnostic: egestion-driven recycling to nutrients
  vector<Type> egest_N_pred(n);  // r_egN*(1 - AE_Z)*grazing (g C m^-3 d^-1)

  // --------------------------
  // INITIAL CONDITIONS (from data; predictions use only previous-step states)
  // --------------------------
  // Use observed initial conditions at t0 as the starting state (common practice).
  // Subsequent steps never use current observations in state updates, avoiding data leakage.
  N_pred(0) = clamp_min(N_dat(0), tiny);
  P_pred(0) = clamp_min(P_dat(0), tiny);
  Z_pred(0) = clamp_min(Z_dat(0), tiny);

  // Initialize diagnostics at t0 to zero
  uptake_pred(0)   = Type(0.0);
  grazing_pred(0)  = Type(0.0);
  remin_N_pred(0)  = Type(0.0);
  temp_mod_pred(0) = pow(Q10, (T - T_ref) / Type(10.0));
  fN_pred(0)       = N_pred(0) / (K_N + N_pred(0));
  fL_pred(0)       = L_avail / (K_L + L_avail + tiny);
  {
    Type a = pow(clamp_min(fN_pred(0), tiny), alpha_colim);
    Type b = pow(clamp_min(fL_pred(0), tiny), alpha_colim);
    co_lim_pred(0) = pow((a + b) / Type(2.0), Type(1.0) / alpha_colim);
  }
  mix_in_N_pred(0) = D_mix * (N_in - N_pred(0));
  dilP_pred(0)     = D_mix * P_pred(0);
  dilZ_pred(0)     = D_mix * Z_pred(0);
  egest_N_pred(0)  = Type(0.0);

  // --------------------------
  // PROCESS MODEL (Euler forward integration)
  // --------------------------
  for (int i = 1; i < n; i++) {
    Type dt = Time(i) - Time(i - 1);
    dt = clamp_min(dt, Type(1e-6)); // ensure positive dt

    // Previous-step states
    Type N_prev = N_pred(i - 1);
    Type P_prev = P_pred(i - 1);
    Type Z_prev = Z_pred(i - 1);

    // Limitation terms and temperature modifier (based on previous states only)
    Type fN = N_prev / (K_N + N_prev);                          // 0-1
    Type fL = L_avail / (K_L + L_avail + tiny);                  // 0-1
    Type a  = pow(clamp_min(fN, tiny), alpha_colim);
    Type b  = pow(clamp_min(fL, tiny), alpha_colim);
    Type co_lim = pow((a + b) / Type(2.0), Type(1.0) / alpha_colim);
    Type temp_mod = pow(Q10, (T - T_ref) / Type(10.0));          // temperature modifier for biological rates

    // Rates (g C m^-3 d^-1), computed from previous states
    // Temperature-modified phytoplankton growth
    Type muP     = mu_max * temp_mod * co_lim;                   // d^-1
    Type uptake  = muP * P_prev;                                 // N -> P

    // Zooplankton grazing: Holling/Hill functional response times Z biomass, temperature-modified
    Type Ph = pow(P_prev, h);
    Type Kh = pow(K_P + tiny, h);
    Type func_resp = Ph / (Kh + Ph);
    Type gZ_rate = g_max * temp_mod;                             // d^-1
    Type grazing = gZ_rate * func_resp * Z_prev;                 // P -> Z ingestion

    // Mortalities and excretion (temperature-modified linear terms)
    Type mP_rate  = m_P  * temp_mod;                             // d^-1
    Type mZL_rate = m_ZL * temp_mod;                             // d^-1
    Type exZ_rate = ex_Z * temp_mod;                             // d^-1

    Type mortP = mP_rate  * P_prev;                              // P loss
    Type mortZ = mZL_rate * Z_prev + gamma_Z * Z_prev * Z_prev;  // Z loss (quadratic term left unscaled)
    Type excrZ = exZ_rate * Z_prev;                              // Z -> N

    // Mixed-layer exchange/dilution
    Type mix_in_N = D_mix * (N_in - N_prev);                     // N source/sink
    Type dilP     = D_mix * P_prev;                              // P sink
    Type dilZ     = D_mix * Z_prev;                              // Z sink

    // Egestion-driven instant remineralization of unassimilated ingestion
    Type egest_N = r_egN * (Type(1.0) - AE_Z) * grazing;         // -> N

    // Remineralization to nutrients
    Type remin_N = r_PN * mortP + r_ZN * mortZ + excrZ + egest_N; // -> N

    // State updates (Euler)
    Type dN = -uptake + remin_N + mix_in_N;
    Type dP =  uptake - grazing - mortP - dilP;
    Type dZ =  AE_Z * grazing - mortZ - excrZ - dilZ;

    Type N_next = N_prev + dt * dN;
    Type P_next = P_prev + dt * dP;
    Type Z_next = Z_prev + dt * dZ;

    // Enforce positivity for lognormal likelihood stability
    N_pred(i) = clamp_min(N_next, tiny);
    P_pred(i) = clamp_min(P_next, tiny);
    Z_pred(i) = clamp_min(Z_next, tiny);

    // Store diagnostics
    uptake_pred(i)   = uptake;
    grazing_pred(i)  = grazing;
    remin_N_pred(i)  = remin_N;
    temp_mod_pred(i) = temp_mod;
    fN_pred(i)       = fN;
    fL_pred(i)       = fL;
    co_lim_pred(i)   = co_lim;
    mix_in_N_pred(i) = mix_in_N;
    dilP_pred(i)     = dilP;
    dilZ_pred(i)     = dilZ;
    egest_N_pred(i)  = egest_N;
  }

  // --------------------------
  // OBSERVATION MODEL (lognormal likelihood)
  // --------------------------
  Type nll = Type(0.0);

  for (int i = 0; i < n; i++) {
    // Apply a small floor to observed values for log-transform stability
    Type lnN_obs = log(clamp_min(N_dat(i), tiny));
    Type lnP_obs = log(clamp_min(P_dat(i), tiny));
    Type lnZ_obs = log(clamp_min(Z_dat(i), tiny));

    Type lnN_pred = log(clamp_min(N_pred(i), tiny));
    Type lnP_pred = log(clamp_min(P_pred(i), tiny));
    Type lnZ_pred = log(clamp_min(Z_pred(i), tiny));

    nll -= dnorm(lnN_obs, lnN_pred, sigma_N, true);
    nll -= dnorm(lnP_obs, lnP_pred, sigma_P, true);
    nll -= dnorm(lnZ_obs, lnZ_pred, sigma_Z, true);
  }

  // --------------------------
  // REPORTS (for diagnostics and posterior summaries)
  // --------------------------
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  REPORT(uptake_pred);
  REPORT(grazing_pred);
  REPORT(remin_N_pred);
  REPORT(temp_mod_pred);
  REPORT(fN_pred);
  REPORT(fL_pred);
  REPORT(co_lim_pred);
  REPORT(mix_in_N_pred);
  REPORT(dilP_pred);
  REPORT(dilZ_pred);
  REPORT(egest_N_pred);

  ADREPORT(N_pred);
  ADREPORT(P_pred);
  ADREPORT(Z_pred);

  // Optional gentle penalties to discourage parameters wandering outside plausible ranges
  // (kept weak to avoid dominating the likelihood; can be tuned as needed)
  Type pen = Type(0.0);
  pen += smooth_bounds_penalty(L_avail, Type(0.0), Type(1.0), Type(1e-3));
  pen += smooth_bounds_penalty(AE_Z,    Type(0.0), Type(1.0), Type(1e-3));
  pen += smooth_bounds_penalty(Q10,     Type(1.0), Type(4.0), Type(1e-4));
  pen += smooth_bounds_penalty(D_mix,   Type(0.0), Type(1.0), Type(1e-4));
  pen += smooth_bounds_penalty(r_egN,   Type(0.0), Type(1.0), Type(1e-3));
  nll += pen;

  return nll;
}
