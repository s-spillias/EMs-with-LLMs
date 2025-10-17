#include <TMB.hpp>

// Smooth positive part to avoid hard cutoffs and preserve differentiability
template<class Type>
inline Type pospart(const Type& x) {
  return (x + CppAD::sqrt(x * x + Type(1e-8))) / Type(2.0); // smooth ReLU, epsilon prevents NaN
}

// Smooth quadratic penalty for parameters outside [lo, hi]
template<class Type>
inline Type range_penalty(const Type& x, const Type& lo, const Type& hi, const Type& w) {
  Type below = pospart(lo - x);    // >0 when x < lo
  Type above = pospart(x - hi);    // >0 when x > hi
  return w * (below * below + above * above); // quadratic penalty outside range
}

template<class Type>
Type objective_function<Type>::operator() () {
  // ------------------------
  // DATA
  // ------------------------
  DATA_VECTOR(Year);        // calendar year (integer-valued, but numeric vector)

  // NOTE: For NPZ interpretation:
  // - cots_dat -> observed N (nutrient)
  // - fast_dat -> observed P (phytoplankton)
  // - slow_dat -> observed Z (zooplankton)
  DATA_VECTOR(cots_dat);    // Observed nutrient concentration (positive)
  DATA_VECTOR(fast_dat);    // Observed phytoplankton concentration (positive)
  DATA_VECTOR(slow_dat);    // Observed zooplankton concentration (positive)

  DATA_VECTOR(sst_dat);     // Sea Surface Temperature (°C), environmental driver
  DATA_VECTOR(cotsimm_dat); // External nutrient input driver (used via alpha_Nin scaling)

  int T = Year.size(); // number of time steps (years)

  // ------------------------
  // PARAMETERS (NPZ)
  // ------------------------
  // Initial states
  PARAMETER(N0);  // initial nutrient concentration
  PARAMETER(P0);  // initial phytoplankton concentration
  PARAMETER(Z0);  // initial zooplankton concentration

  // Phytoplankton growth (Monod) and temperature sensitivity
  PARAMETER(mu_max);  // Maximum P growth rate (yr^-1)
  PARAMETER(K_N);     // Half-saturation for nutrient uptake (same units as N)
  PARAMETER(Q10_P);   // Q10 temperature multiplier for P growth

  // Zooplankton grazing (Holling Type II/III) and temperature sensitivity
  PARAMETER(g_max);   // Maximum grazing rate (yr^-1)
  PARAMETER(K_P);     // Half-saturation for grazing (same units as P)
  PARAMETER(eta);     // Shape exponent (>=1: Type III-like)
  PARAMETER(Q10_Z);   // Q10 temperature multiplier for Z grazing

  // Efficiencies and loss terms
  PARAMETER(e);       // Assimilation efficiency (0-1)
  PARAMETER(mP);      // Phytoplankton linear loss (yr^-1)
  PARAMETER(mZ);      // Zooplankton mortality (yr^-1)
  PARAMETER(rN);      // Fraction of losses instantly remineralized to N (0-1)

  // Temperature reference
  PARAMETER(T_ref);   // Reference temperature for Q10 scaling (°C)

  // External nutrient input scaling
  PARAMETER(alpha_Nin); // Scales cotsimm_dat into nutrient input per step

  // Observation error parameters (lognormal)
  PARAMETER(sigma_N);  // Log-space sd for N
  PARAMETER(sigma_P);  // Log-space sd for P
  PARAMETER(sigma_Z);  // Log-space sd for Z

  // ------------------------
  // NEGATIVE LOG-LIKELIHOOD AND PENALTIES
  // ------------------------
  Type nll = 0.0;
  const Type eps = Type(1e-8);      // small epsilon to stabilize divisions/logs
  const Type sd_floor = Type(0.05); // minimum sd used in likelihood for stability

  // Smooth range penalties to keep parameters in plausible bounds
  const Type w_pen = Type(1e-3);
  nll += range_penalty(N0,       Type(0.0),   Type(100.0), w_pen);
  nll += range_penalty(P0,       Type(0.0),   Type(50.0),  w_pen);
  nll += range_penalty(Z0,       Type(0.0),   Type(50.0),  w_pen);

  nll += range_penalty(mu_max,   Type(0.0),   Type(20.0),  w_pen);
  nll += range_penalty(K_N,      Type(0.001), Type(10.0),  w_pen);
  nll += range_penalty(Q10_P,    Type(1.0),   Type(3.0),   w_pen);

  nll += range_penalty(g_max,    Type(0.0),   Type(10.0),  w_pen);
  nll += range_penalty(K_P,      Type(0.001), Type(10.0),  w_pen);
  nll += range_penalty(eta,      Type(1.0),   Type(3.0),   w_pen);
  nll += range_penalty(Q10_Z,    Type(1.0),   Type(3.0),   w_pen);

  nll += range_penalty(e,        Type(0.0),   Type(1.0),   w_pen);
  nll += range_penalty(mP,       Type(0.0),   Type(5.0),   w_pen);
  nll += range_penalty(mZ,       Type(0.0),   Type(5.0),   w_pen);
  nll += range_penalty(rN,       Type(0.0),   Type(1.0),   w_pen);

  nll += range_penalty(T_ref,    Type(-2.0),  Type(35.0),  w_pen);
  nll += range_penalty(alpha_Nin,Type(0.0),   Type(100.0), w_pen);

  nll += range_penalty(sigma_N,  Type(0.01),  Type(2.0),   w_pen);
  nll += range_penalty(sigma_P,  Type(0.01),  Type(2.0),   w_pen);
  nll += range_penalty(sigma_Z,  Type(0.01),  Type(2.0),   w_pen);

  // ------------------------
  // STATE VECTORS
  // ------------------------
  vector<Type> N_pred(T); // nutrient
  vector<Type> P_pred(T); // phytoplankton
  vector<Type> Z_pred(T); // zooplankton

  // Framework-expected aliases for observed series:
  // cots_dat -> cots_pred (maps to N)
  // fast_dat -> fast_pred (maps to P)
  // slow_dat -> slow_pred (maps to Z)
  vector<Type> cots_pred(T);
  vector<Type> fast_pred(T);
  vector<Type> slow_pred(T);

  // Initialize states (keep concentrations >= 0)
  N_pred(0) = pospart(N0);
  P_pred(0) = pospart(P0);
  Z_pred(0) = pospart(Z0);

  // Initialize aliases
  cots_pred(0) = N_pred(0);
  fast_pred(0) = P_pred(0);
  slow_pred(0) = Z_pred(0);

  // ------------------------
  // FORWARD SIMULATION (use t-1 states to compute t)
  // ------------------------
  for (int t = 1; t < T; ++t) {
    // Previous states
    Type N = N_pred(t - 1);
    Type P = P_pred(t - 1);
    Type Z = Z_pred(t - 1);

    // Exogenous drivers at t-1 to avoid leakage
    Type sst = sst_dat(t - 1);
    Type Nin_raw = cotsimm_dat(t - 1); // external nutrient input series (raw)
    Type N_in = alpha_Nin * Nin_raw;

    // Temperature modifiers (Q10)
    Type fT_P = pow(Q10_P, (sst - T_ref) / Type(10.0));
    Type fT_Z = pow(Q10_Z, (sst - T_ref) / Type(10.0));

    // 1) Phytoplankton growth (Monod nutrient limitation)
    Type mu = mu_max * fT_P * (N / (K_N + N + eps));

    // 2) Zooplankton grazing on P (Holling Type II/III blend)
    Type P_eta = pow(P + eps, eta);
    Type fr = P_eta / (pow(K_P + eps, eta) + P_eta); // fraction in [0,1]
    Type G = g_max * fT_Z * Z * fr;

    // 3) Uptake and losses
    Type Uptake_N = mu * P;           // unit yield assumption
    Type MortP = mP * P;
    Type MortZ = mZ * Z;

    // 4) Remineralization back to dissolved nutrient
    Type Remin = rN * (MortP + (Type(1.0) - e) * G + MortZ);

    // 5) State updates
    Type P_next = P + mu * P - G - MortP;
    Type Z_next = Z + e * G - MortZ;
    Type N_next = N - Uptake_N + Remin + N_in;

    // Enforce non-negativity smoothly
    P_next = pospart(P_next);
    Z_next = pospart(Z_next);
    N_next = pospart(N_next);

    // Store next-step predictions
    N_pred(t) = N_next;
    P_pred(t) = P_next;
    Z_pred(t) = Z_next;

    // Keep framework aliases synchronized
    cots_pred(t) = N_next;
    fast_pred(t) = P_next;
    slow_pred(t) = Z_next;
  }

  // ------------------------
  // OBSERVATION MODEL
  // ------------------------
  // Smooth max with floor using pospart to keep AD-friendly
  Type sd_N = sigma_N + pospart(sd_floor - sigma_N);
  Type sd_P = sigma_P + pospart(sd_floor - sigma_P);
  Type sd_Z = sigma_Z + pospart(sd_floor - sigma_Z);

  for (int t = 0; t < T; ++t) {
    // All three: lognormal error with Jacobian
    // Map: cots_dat -> cots_pred (N), fast_dat -> fast_pred (P), slow_dat -> slow_pred (Z)
    {
      Type y = cots_dat(t);
      Type mu = cots_pred(t);
      Type logy = log(y + eps);
      Type logmu = log(mu + eps);
      nll -= dnorm(logy, logmu, sd_N, true);
      nll += log(y + eps); // Jacobian
    }
    {
      Type y = fast_dat(t);
      Type mu = fast_pred(t);
      Type logy = log(y + eps);
      Type logmu = log(mu + eps);
      nll -= dnorm(logy, logmu, sd_P, true);
      nll += log(y + eps);
    }
    {
      Type y = slow_dat(t);
      Type mu = slow_pred(t);
      Type logy = log(y + eps);
      Type logmu = log(mu + eps);
      nll -= dnorm(logy, logmu, sd_Z, true);
      nll += log(y + eps);
    }
  }

  // ------------------------
  // REPORTING
  // ------------------------
  // Original state names
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);

  // Framework-expected aliases
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  ADREPORT(N_pred);
  ADREPORT(P_pred);
  ADREPORT(Z_pred);

  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);

  return nll;
}
