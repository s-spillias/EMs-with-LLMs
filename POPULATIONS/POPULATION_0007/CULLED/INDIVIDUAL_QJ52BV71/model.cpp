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
  // DATA (interface expects fast/slow/cots)
  // ------------------------
  DATA_VECTOR(Year);      // time index (assumed annual steps unless otherwise specified)
  DATA_VECTOR(fast_dat);  // Observed phytoplankton-like state (e.g., P), positive
  DATA_VECTOR(slow_dat);  // Observed nutrient-like state (e.g., N), positive
  DATA_VECTOR(cots_dat);  // Observed zooplankton-like state (e.g., Z), positive

  int T = Year.size();

  // ------------------------
  // PARAMETERS
  // ------------------------
  // Initial states (mapped: slow ~ N, fast ~ P, cots ~ Z)
  PARAMETER(N0);   // initial nutrient
  PARAMETER(P0);   // initial phytoplankton
  PARAMETER(Z0);   // initial zooplankton

  // Phytoplankton growth (resource limitation via saturating uptake)
  PARAMETER(mu_max); // Max phytoplankton specific growth (yr^-1)
  PARAMETER(K_N);    // Half-saturation constant for nutrient uptake (same units as N)

  // Zooplankton grazing (Holling Type-III)
  PARAMETER(g_max);  // Max grazing rate (yr^-1)
  PARAMETER(K_P);    // Half-saturation constant for grazing (same units as P)
  PARAMETER(eta_g);  // Shape exponent (>=1), Type-III if >1

  // Efficiencies and mortalities
  PARAMETER(e_g);    // Assimilation efficiency of grazing into Z (0-1)
  PARAMETER(mP);     // Phytoplankton non-grazing loss (yr^-1)
  PARAMETER(mZ);     // Zooplankton mortality (yr^-1)

  // Remineralization and mixing
  PARAMETER(r_remin); // Fraction of organic losses instantaneously remineralized to N (0-1)
  PARAMETER(k_mix);   // Vertical exchange rate with deep pool (yr^-1)
  PARAMETER(N_deep);  // Deep nutrient concentration (same units as N)

  // Observation error (log-space SDs)
  PARAMETER(sigma_N);
  PARAMETER(sigma_P);
  PARAMETER(sigma_Z);

  // ------------------------
  // Priors/penalties (weak, for stability and plausibility)
  // ------------------------
  Type nll = 0.0;
  const Type eps = Type(1e-8);
  const Type sd_floor = Type(0.05);
  const Type w_pen = Type(1e-3);

  // Smooth range penalties
  nll += range_penalty(N0,      Type(0.0),  Type(100.0), w_pen);
  nll += range_penalty(P0,      Type(0.0),  Type(50.0),  w_pen);
  nll += range_penalty(Z0,      Type(0.0),  Type(50.0),  w_pen);

  nll += range_penalty(mu_max,  Type(0.0),  Type(30.0),  w_pen);
  nll += range_penalty(K_N,     Type(0.001),Type(20.0),  w_pen);

  nll += range_penalty(g_max,   Type(0.0),  Type(30.0),  w_pen);
  nll += range_penalty(K_P,     Type(0.001),Type(20.0),  w_pen);
  nll += range_penalty(eta_g,   Type(1.0),  Type(3.0),   w_pen);

  nll += range_penalty(e_g,     Type(0.0),  Type(1.0),   w_pen);
  nll += range_penalty(mP,      Type(0.0),  Type(10.0),  w_pen);
  nll += range_penalty(mZ,      Type(0.0),  Type(10.0),  w_pen);

  nll += range_penalty(r_remin, Type(0.0),  Type(1.0),   w_pen);
  nll += range_penalty(k_mix,   Type(0.0),  Type(10.0),  w_pen);
  nll += range_penalty(N_deep,  Type(0.0),  Type(100.0), w_pen);

  nll += range_penalty(sigma_N, Type(0.01), Type(2.0),   w_pen);
  nll += range_penalty(sigma_P, Type(0.01), Type(2.0),   w_pen);
  nll += range_penalty(sigma_Z, Type(0.01), Type(2.0),   w_pen);

  // ------------------------
  // STATE VECTORS (use interface names as primary predictions)
  // ------------------------
  vector<Type> slow_pred(T); // nutrient-like (N)
  vector<Type> fast_pred(T); // phytoplankton-like (P)
  vector<Type> cots_pred(T); // zooplankton-like (Z)

  // Initialize states (ensure non-negativity)
  slow_pred(0) = pospart(N0);
  fast_pred(0) = pospart(P0);
  cots_pred(0) = pospart(Z0);

  // ------------------------
  // FORWARD SIMULATION (t uses t-1 states; no data leakage)
  // ------------------------
  for (int t = 1; t < T; ++t) {
    Type N = slow_pred(t - 1);
    Type P = fast_pred(t - 1);
    Type Z = cots_pred(t - 1);

    // Phytoplankton growth limited by nutrient
    Type fN = N / (K_N + N + eps);
    Type growth_P = mu_max * fN * P;

    // Zooplankton grazing with Holling Type-III on P
    Type Pp = pospart(P);
    Type denom_g = pow(K_P + eps, eta_g) + pow(Pp + eps, eta_g);
    Type graze_rate = g_max * pow(Pp + eps, eta_g) / denom_g;
    Type G = graze_rate * Z; // total grazing flux

    // Non-grazing losses and mortality
    Type loss_P = mP * P;
    Type loss_Z = mZ * Z;

    // Zooplankton production (assimilated fraction of grazing)
    Type prod_Z = e_g * G;

    // Remineralization flux back to nutrient
    Type remin = r_remin * (loss_P + (Type(1.0) - e_g) * G + loss_Z);

    // Vertical mixing exchange toward deep nutrient pool
    Type mix = k_mix * (N_deep - N);

    // State updates (discrete-time Euler)
    Type N_next = N - growth_P + remin + mix;
    Type P_next = P + growth_P - G - loss_P;
    Type Z_next = Z + prod_Z - loss_Z;

    // Enforce non-negativity smoothly
    slow_pred(t) = pospart(N_next);
    fast_pred(t) = pospart(P_next);
    cots_pred(t) = pospart(Z_next);
  }

  // ------------------------
  // OBSERVATION MODEL (lognormal errors; positive supports)
  // ------------------------
  Type sd_slow = sigma_N + pospart(Type(0.05) - sigma_N);
  Type sd_fast = sigma_P + pospart(Type(0.05) - sigma_P);
  Type sd_cots = sigma_Z + pospart(Type(0.05) - sigma_Z);

  for (int t = 0; t < T; ++t) {
    // slow (nutrient-like)
    Type yS = slow_dat(t);
    Type muS = slow_pred(t);
    nll -= dnorm(log(yS + eps), log(muS + eps), sd_slow, true);
    nll += log(yS + eps); // Jacobian

    // fast (phytoplankton-like)
    Type yF = fast_dat(t);
    Type muF = fast_pred(t);
    nll -= dnorm(log(yF + eps), log(muF + eps), sd_fast, true);
    nll += log(yF + eps);

    // cots (zooplankton-like)
    Type yC = cots_dat(t);
    Type muC = cots_pred(t);
    nll -= dnorm(log(yC + eps), log(muC + eps), sd_cots, true);
    nll += log(yC + eps);
  }

  // ------------------------
  // REPORTING
  // ------------------------
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(cots_pred);

  ADREPORT(slow_pred);
  ADREPORT(fast_pred);
  ADREPORT(cots_pred);

  return nll;
}
