#include <TMB.hpp>  // TMB header providing automatic differentiation and likelihood utilities

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
  PARAMETER(k_shade);       // m^3 g C^-1 | Phytoplankton self-shading coefficient (Beer–Lambert-like attenuation)
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

  // --------------------------
  // CONSTANTS AND ALIASES
  // --------------------------
  int n = N_dat.size();                   // number of time steps (should match length of Time, P_dat, Z_dat)
  Type tiny = Type(1e-8);                 // small constant to avoid division by zero and log(0)
  Type sd_floor = Type(0.02);             // minimum observation SD on log scale to prevent numerical issues

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
  vector<Type> L_eff_pred(n);    // effective light availability after self-shading (0-1)

  // --------------------------
  // INITIAL CONDITIONS (from data, to avoid data-leakage in dynamics)
  // --------------------------
  N_pred(0) = N_dat(0);  // Initialize predictions with first observed value (g C m^-3)
  P_pred(0) = P_dat(0);  // Initialize predictions with first observed value (g C m^-3)
  Z_pred(0) = Z_dat(0);  // Initialize predictions with first observed value (g C m^-3)

  // Initialize diagnostics for t=0 (consistent with states)
  Type temp_mod0 = pow(Q10 + tiny, (T - T_ref) / Type(10.0));                         // temperature effect at t=0
  Type fN0 = N_pred(0) / (K_N + N_pred(0) + tiny);                                    // nutrient limitation at t=0 (Monod)
  Type L_eff0 = L_avail * exp(-k_shade * (P_pred(0) + tiny));                          // effective light at t=0 with self-shading
  Type fL0 = L_eff0 / (K_L + L_eff0 + tiny);                                          // light limitation at t=0 (saturating)
  Type co0 = pow(pow(fN0 + tiny, alpha_colim) + pow(fL0 + tiny, alpha_colim), Type(1.0) / (alpha_colim + tiny)); // smooth min
  Type uptake0 = mu_max * temp_mod0 * co0 * P_pred(0);                                // gross P growth at t=0
  Type numer = pow(P_pred(0) + tiny, h);                                              // numerator of Holling response
  Type denom = pow(K_P + tiny, h) + numer;                                            // denominator of Holling response
  Type G0 = g_max * (numer / (denom + tiny)) * Z_pred(0);                             // grazing at t=0
  Type mortP0 = m_P * P_pred(0);                                                      // linear P mortality
  Type mortZ0 = m_ZL * Z_pred(0) + gamma_Z * Z_pred(0) * Z_pred(0);                   // Z mortality (linear + quadratic)
  Type remin0 = (Type(1.0) - AE_Z) * G0 + r_PN * mortP0 + r_ZN * mortZ0 + ex_Z * Z_pred(0); // recycling to N
  uptake_pred(0) = uptake0;
  grazing_pred(0) = G0;
  remin_N_pred(0) = remin0;
  temp_mod_pred(0) = temp_mod0;
  fN_pred(0) = fN0;
  fL_pred(0) = fL0;
  co_lim_pred(0) = co0;
  L_eff_pred(0) = L_eff0;

  // --------------------------
  // PROCESS MODEL (forward simulation, Euler integration)
  // Note: predictions at time i use only states from time i-1 (no data leakage).
  // --------------------------
  for (int i = 1; i < n; i++) {
    // Time step (ensure positive)
    Type dt_raw = Time(i) - Time(i - 1);                      // raw step (days)
    Type dt = CppAD::CondExpGt(dt_raw, tiny, dt_raw, tiny);   // enforce minimum positive dt

    // Previous state (ensure strictly positive via tiny)
    Type N_prev = N_pred(i - 1) + tiny;                        // previous N
    Type P_prev = P_pred(i - 1) + tiny;                        // previous P
    Type Z_prev = Z_pred(i - 1) + tiny;                        // previous Z

    // Limitations and modifiers
    Type temp_mod = pow(Q10 + tiny, (T - T_ref) / Type(10.0));                          // temperature effect (constant here, but kept per-step for extensibility)
    Type fN = N_prev / (K_N + N_prev + tiny);                                           // Monod nutrient limitation (0-1)
    Type L_eff = L_avail * exp(-k_shade * P_prev);                                      // Self-shading reduces effective light
    Type fL = L_eff / (K_L + L_eff + tiny);                                             // Light limitation (0-1) using effective light
    Type co_lim = pow(pow(fN + tiny, alpha_colim) + pow(fL + tiny, alpha_colim), Type(1.0) / (alpha_colim + tiny)); // smooth min of fN and fL

    // Process rates (per day)
    Type uptake = mu_max * temp_mod * co_lim * P_prev;                                  // 1) Phytoplankton gross production
    Type FR_num = pow(P_prev + tiny, h);                                                // Hill numerator for grazing
    Type FR_den = pow(K_P + tiny, h) + FR_num;                                          // Hill denominator for grazing
    Type grazing = g_max * (FR_num / (FR_den + tiny)) * Z_prev;                         // 2) Zooplankton grazing on P
    Type mortP = m_P * P_prev;                                                          // 3) Phytoplankton linear mortality
    Type mortZ = m_ZL * Z_prev + gamma_Z * Z_prev * Z_prev;                             // 4) Zooplankton mortality (linear + quadratic)
    Type remin_to_N = (Type(1.0) - AE_Z) * grazing                                      // 5) Unassimilated ingestion (sloppy feeding/egestion) to N
                    + r_PN * mortP                                                      // 6) Fraction of P mortality remineralized to N
                    + r_ZN * mortZ                                                      // 7) Fraction of Z mortality remineralized to N
                    + ex_Z * Z_prev;                                                    // 8) Zooplankton excretion to N

    // Euler updates
    Type dN = (-uptake + remin_to_N) * dt;                                              // Change in N
    Type dP = ( uptake - grazing - mortP) * dt;                                         // Change in P
    Type dZ = ( AE_Z * grazing - mortZ ) * dt;                                          // Change in Z

    // Update states (ensure they remain non-negative via soft floor)
    N_pred(i) = N_prev + dN;                                                            // Update N
    P_pred(i) = P_prev + dP;                                                            // Update P
    Z_pred(i) = Z_prev + dZ;                                                            // Update Z

    // Store diagnostics
    uptake_pred(i)   = uptake;
    grazing_pred(i)  = grazing;
    remin_N_pred(i)  = remin_to_N;
    temp_mod_pred(i) = temp_mod;
    fN_pred(i)       = fN;
    fL_pred(i)       = fL;
    co_lim_pred(i)   = co_lim;
    L_eff_pred(i)    = L_eff;
  }

  // --------------------------
  // LIKELIHOOD (lognormal observation model for strictly positive data)
  // Include all observations (i = 0..n-1) with SD floor to avoid numerical issues.
  // --------------------------
  Type nll = Type(0.0);                                                                // negative log-likelihood accumulator
  for (int i = 0; i < n; i++) {
    nll -= dnorm(log(N_dat(i) + tiny), log(N_pred(i) + tiny), sigma_N, true);          // N lognormal error
    nll -= dnorm(log(P_dat(i) + tiny), log(P_pred(i) + tiny), sigma_P, true);          // P lognormal error
    nll -= dnorm(log(Z_dat(i) + tiny), log(Z_pred(i) + tiny), sigma_Z, true);          // Z lognormal error
  }

  // --------------------------
  // SMOOTH PARAMETER BOUNDS (penalties instead of hard constraints)
  // --------------------------
  // Suggested biological bounds (duplicated here for smooth penalty; see parameters.json for documentation)
  nll += smooth_bounds_penalty(mu_max,  Type(0.1),  Type(2.0),  Type(1.0));            // mu_max in [0.1, 2] d^-1
  nll += smooth_bounds_penalty(K_N,     Type(0.01), Type(1.0),  Type(1.0));            // K_N in [0.01, 1] g C m^-3
  nll += smooth_bounds_penalty(L_avail, Type(0.0),  Type(1.0),  Type(1.0));            // L_avail in [0, 1]
  nll += smooth_bounds_penalty(K_L,     Type(0.05), Type(2.0),  Type(0.5));            // K_L in [0.05, 2]
  nll += smooth_bounds_penalty(k_shade, Type(0.0),  Type(5.0),  Type(0.5));            // k_shade in [0, 5] m^3 g C^-1
  nll += smooth_bounds_penalty(alpha_colim, Type(-10.0), Type(-1.0), Type(0.5));       // alpha_colim in [-10, -1]
  nll += smooth_bounds_penalty(Q10,     Type(1.0),  Type(3.0),  Type(0.5));            // Q10 in [1, 3]
  nll += smooth_bounds_penalty(T,       Type(0.0),  Type(30.0), Type(0.2));            // T in [0, 30] deg C
  nll += smooth_bounds_penalty(T_ref,   Type(0.0),  Type(30.0), Type(0.1));            // T_ref in [0, 30] deg C
  nll += smooth_bounds_penalty(g_max,   Type(0.1),  Type(3.0),  Type(1.0));            // g_max in [0.1, 3] d^-1
  nll += smooth_bounds_penalty(K_P,     Type(0.01), Type(1.0),  Type(1.0));            // K_P in [0.01, 1] g C m^-3
  nll += smooth_bounds_penalty(h,       Type(1.0),  Type(3.0),  Type(0.5));            // h in [1, 3]
  nll += smooth_bounds_penalty(AE_Z,    Type(0.3),  Type(0.9),  Type(1.0));            // AE_Z in [0.3, 0.9]
  nll += smooth_bounds_penalty(m_P,     Type(0.0),  Type(0.5),  Type(0.5));            // m_P in [0, 0.5] d^-1
  nll += smooth_bounds_penalty(m_ZL,    Type(0.0),  Type(0.5),  Type(0.5));            // m_ZL in [0, 0.5] d^-1
  nll += smooth_bounds_penalty(gamma_Z, Type(0.0),  Type(2.0),  Type(0.2));            // gamma_Z in [0, 2] (g C m^-3)^-1 d^-1
  nll += smooth_bounds_penalty(r_PN,    Type(0.0),  Type(1.0),  Type(0.5));            // r_PN in [0, 1]
  nll += smooth_bounds_penalty(r_ZN,    Type(0.0),  Type(1.0),  Type(0.5));            // r_ZN in [0, 1]
  nll += smooth_bounds_penalty(ex_Z,    Type(0.0),  Type(0.5),  Type(0.5));            // ex_Z in [0, 0.5] d^-1
  // Observation SDs are handled by exp() transform + floor; no extra bounds needed.

  // --------------------------
  // REPORTING
  // --------------------------
  REPORT(Time);          // time axis (days)
  REPORT(N_pred);        // predicted nutrients
  REPORT(P_pred);        // predicted phytoplankton
  REPORT(Z_pred);        // predicted zooplankton
  REPORT(uptake_pred);   // diagnostic: phytoplankton gross production
  REPORT(grazing_pred);  // diagnostic: zooplankton grazing
  REPORT(remin_N_pred);  // diagnostic: remineralization to nutrients
  REPORT(temp_mod_pred); // diagnostic: temperature modifier
  REPORT(fN_pred);       // diagnostic: nutrient limitation factor
  REPORT(fL_pred);       // diagnostic: light limitation factor
  REPORT(co_lim_pred);   // diagnostic: co-limitation factor
  REPORT(L_eff_pred);    // diagnostic: effective light after self-shading
  REPORT(sigma_N);       // observation SD for N (log scale)
  REPORT(sigma_P);       // observation SD for P (log scale)
  REPORT(sigma_Z);       // observation SD for Z (log scale)

  return nll;            // return total negative log-likelihood
}

/*
Equation summary (all rates in day^-1; concentrations in g C m^-3):
1) f_N = N / (K_N + N)                               [0..1] nutrient limitation (Monod)
2) L_eff = L_avail * exp(-k_shade * P)               [0..1] effective light after self-shading
3) f_L = L_eff / (K_L + L_eff)                       [0..1] light limitation (saturating)
4) temp_mod = Q10^((T - T_ref)/10)                   temperature modifier on growth
5) co_lim = (f_N^a + f_L^a)^(1/a)                    smooth minimum of f_N and f_L for a<0 (alpha_colim)
6) uptake = mu_max * temp_mod * co_lim * P
7) grazing = g_max * (P^h / (K_P^h + P^h)) * Z       Holling-II/III functional response
8) mortP = m_P * P
9) mortZ = m_ZL * Z + gamma_Z * Z^2
10) remin_to_N = (1 - AE_Z) * grazing + r_PN * mortP + r_ZN * mortZ + ex_Z * Z
11) dN/dt = -uptake + remin_to_N
12) dP/dt =  uptake - grazing - mortP
13) dZ/dt =  AE_Z * grazing - mortZ

Numerical notes:
- tiny = 1e-8 is added to denominators and logs to ensure stability.
- dt is enforced positive using a conditional expression.
- Observation model is lognormal with a fixed SD floor (sd_floor) on the log scale.
- Smooth penalties discourage parameters from leaving biologically plausible ranges without hard constraints.
*/
