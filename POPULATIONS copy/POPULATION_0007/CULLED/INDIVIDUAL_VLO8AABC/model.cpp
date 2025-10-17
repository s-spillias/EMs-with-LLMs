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
  DATA_VECTOR(Year);     // calendar year (numeric vector)

  // Available response variables in the current dataset
  DATA_VECTOR(cots_dat);
  DATA_VECTOR(slow_dat);
  DATA_VECTOR(fast_dat);

  int T = Year.size(); // number of time steps

  // ------------------------
  // PARAMETERS
  // Keep NPZ-related parameters to remain compatible with parameters.json and phases,
  // but do not use them in the current likelihood since NPZ data are not provided.
  // ------------------------
  PARAMETER(mu_max);   // yr^-1: maximum phytoplankton growth rate
  PARAMETER(K_N);      // mmol N m^-3: half-saturation constant for nutrient uptake
  PARAMETER(g_max);    // yr^-1: maximum zooplankton grazing rate
  PARAMETER(K_P);      // mmol N m^-3: half-saturation constant for grazing (on P)
  PARAMETER(e);        // dimensionless (0-1): assimilation efficiency
  PARAMETER(mP);       // yr^-1: phytoplankton linear mortality
  PARAMETER(mZ);       // yr^-1: zooplankton linear mortality
  PARAMETER(rN);       // yr^-1: mixing/supply relaxation rate toward N_in
  PARAMETER(N_in);     // mmol N m^-3: supply/background nutrient concentration

  // Observation error parameters (lognormal SDs) for NPZ (kept for compatibility)
  PARAMETER(sigma_N);  // log-space sd for N
  PARAMETER(sigma_P);  // log-space sd for P
  PARAMETER(sigma_Z);  // log-space sd for Z

  // SDs for available series
  PARAMETER(sigma_cots);
  PARAMETER(sigma_slow);
  PARAMETER(sigma_fast);

  // ------------------------
  // SETUP
  // ------------------------
  Type nll = 0.0;
  const Type eps = Type(1e-8);      // small epsilon to stabilize logs
  const Type sd_floor = Type(0.05); // minimum sd used in likelihood for stability

  // Weak smooth penalties to keep parameters in plausible ranges
  const Type w_pen = Type(1e-3);
  nll += range_penalty(mu_max, Type(0.0),  Type(5.0),   w_pen);
  nll += range_penalty(K_N,    Type(0.01), Type(10.0),  w_pen);
  nll += range_penalty(g_max,  Type(0.0),  Type(5.0),   w_pen);
  nll += range_penalty(K_P,    Type(0.01), Type(10.0),  w_pen);
  nll += range_penalty(e,      Type(0.0),  Type(1.0),   w_pen);
  nll += range_penalty(mP,     Type(0.0),  Type(2.0),   w_pen);
  nll += range_penalty(mZ,     Type(0.0),  Type(2.0),   w_pen);
  nll += range_penalty(rN,     Type(0.0),  Type(5.0),   w_pen);
  nll += range_penalty(N_in,   Type(0.0),  Type(50.0),  w_pen);
  nll += range_penalty(sigma_N,Type(0.01), Type(2.0),   w_pen);
  nll += range_penalty(sigma_P,Type(0.01), Type(2.0),   w_pen);
  nll += range_penalty(sigma_Z,Type(0.01), Type(2.0),   w_pen);
  nll += range_penalty(sigma_cots, Type(0.01), Type(2.0), w_pen);
  nll += range_penalty(sigma_slow, Type(0.01), Type(2.0), w_pen);
  nll += range_penalty(sigma_fast, Type(0.01), Type(2.0), w_pen);

  // Effective observation SDs (floor-added in quadrature for smoothness)
  Type sCots = CppAD::sqrt(sigma_cots * sigma_cots + sd_floor * sd_floor);
  Type sSlow = CppAD::sqrt(sigma_slow * sigma_slow + sd_floor * sd_floor);
  Type sFast = CppAD::sqrt(sigma_fast * sigma_fast + sd_floor * sd_floor);

  // ------------------------
  // STATE PREDICTIONS (compatibility placeholders with no data leakage)
  // ------------------------
  vector<Type> cots_pred(T);
  vector<Type> slow_pred(T);
  vector<Type> fast_pred(T);

  // Initialize with first observations to avoid parameterized initial states
  cots_pred(0) = pospart(cots_dat(0));
  slow_pred(0) = pospart(slow_dat(0));
  fast_pred(0) = pospart(fast_dat(0));

  for (int t = 1; t < T; t++) {
    // Persistence dynamics using only previous predictions (no data leakage)
    cots_pred(t) = pospart(cots_pred(t - 1));
    slow_pred(t) = pospart(slow_pred(t - 1));
    fast_pred(t) = pospart(fast_pred(t - 1));
  }

  // ------------------------
  // LIKELIHOOD: lognormal errors for available series
  // ------------------------
  for (int t = 0; t < T; t++) {
    // cots
    Type yC = log(cots_dat(t) + eps);
    Type muC = log(cots_pred(t) + eps);
    nll -= dnorm(yC, muC, sCots, true);

    // slow
    Type yS = log(slow_dat(t) + eps);
    Type muS = log(slow_pred(t) + eps);
    nll -= dnorm(yS, muS, sSlow, true);

    // fast
    Type yF = log(fast_dat(t) + eps);
    Type muF = log(fast_pred(t) + eps);
    nll -= dnorm(yF, muF, sFast, true);
  }

  // ------------------------
  // REPORTING
  // ------------------------
  REPORT(Year);
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);

  return nll;
}
