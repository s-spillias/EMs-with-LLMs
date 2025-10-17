#include <TMB.hpp>

// Helper: smooth positive mapping to avoid hard truncations; returns ~max(x,0) smoothly
template<class Type>
Type smooth_positive(Type x, Type delta = Type(1e-12)) {
  return (x + sqrt(x * x + delta)) / Type(2.0);
}

// Helper: numerically stable softplus compatible with AD types: log(1 + exp(x))
// Uses a branch-free, overflow-safe form: log(1 + exp(-|x|)) + (x + |x|)/2
template<class Type>
Type softplus(Type x) {
  Type ax = fabs(x);
  return log(Type(1.0) + exp(-ax)) + (x + ax) / Type(2.0);
}

// Helper: smooth hinge penalty for violations; ~max(y, 0) but smooth and AD-safe
template<class Type>
Type smooth_hinge(Type y, Type kappa = Type(50.0)) {
  // softplus(kappa*y)/kappa approximates max(y, 0) smoothly without using log1p
  return softplus(kappa * y) / kappa;
}

// Helper: smooth minimum (Liebig-like) to combine resource limitations without kinks
// min_smooth(a, b) = (a + b - sqrt((a - b)^2 + delta)) / 2
template<class Type>
Type smooth_min(Type a, Type b, Type delta = Type(1e-12)) {
  Type diff = a - b;
  return (a + b - sqrt(diff * diff + delta)) / Type(2.0);
}

template<class Type>
Type objective_function<Type>::operator()() {
  // -----------------------------
  // DATA
  // -----------------------------
  // Time vector name must match the sanitized column name provided by the data loader.
  DATA_VECTOR(Time);                       // Time in days (matches "Time" column from the CSV)

  DATA_VECTOR(N_dat);                      // Observed nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);                      // Observed phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);                      // Observed zooplankton concentration (g C m^-3)

  // -----------------------------
  // PARAMETERS (process)
  // -----------------------------
  PARAMETER(mu_max);        // Maximum phytoplankton specific growth rate (d^-1); literature ranges ~0.3–2 d^-1
  PARAMETER(K_N);           // Half-sat constant for nutrient uptake (g C m^-3); inferred from data or literature
  PARAMETER(hN);            // Hill exponent for nutrient limitation (dimensionless >=1); hN=1 reduces to classic Monod

  PARAMETER(I0);            // Effective surface light/irradiance (relative units per day); scales light limitation
  PARAMETER(K_I);           // Half-sat constant for light-limited growth (same units as I0)
  PARAMETER(k_Ishade);      // Self-shading (attenuation) coefficient by P (m^3 gC^-1); higher => stronger light attenuation
  PARAMETER(k_Ibg);         // Background optical depth (dimensionless, approximates k_d * MLD)

  PARAMETER(q10_mu);        // Q10 for phytoplankton growth (dimensionless); typical 1.5–2.5
  PARAMETER(q10_g);         // Q10 for zooplankton ingestion (dimensionless); typical 1.5–2.5

  PARAMETER(g_max);         // Maximum zooplankton grazing rate (d^-1)
  PARAMETER(K_g);           // Half-sat constant for grazing functional response (g C m^-3)
  PARAMETER(h_exp);         // Shape exponent for grazing response (dimensionless >=1); h=2 gives Holling type III-like
  PARAMETER(c_BD);          // Predator interference coefficient for Beddington-DeAngelis denominator (m^3 gC^-1)

  PARAMETER(e_Z);           // Zooplankton assimilation efficiency (dimensionless, 0–1); fraction of ingestion to Z
  PARAMETER(mP1);           // Linear P mortality/lysis rate (d^-1)
  PARAMETER(mP2);           // Quadratic P loss rate (m^3 gC^-1 d^-1), e.g., aggregation
  PARAMETER(mZ1);           // Linear Z excretion/mortality rate (d^-1)
  PARAMETER(mZ2);           // Quadratic Z mortality (m^3 gC^-1 d^-1)

  PARAMETER(rP_N);          // Fraction of P losses remineralized to N (dimensionless 0–1)
  PARAMETER(rZ_N);          // Fraction of Z losses remineralized to N (dimensionless 0–1)

  PARAMETER(y_PN);          // Yield: g C of P produced per g C of nutrient consumed (dimensionless >0); N_uptake = P_growth / y_PN

  PARAMETER(k_mix);         // Vertical mixing rate coupling to deep pool (d^-1)
  PARAMETER(N_deep);        // Deep nutrient concentration (g C m^-3)

  // -----------------------------
  // PARAMETERS (environment and objective weight; made parameters to avoid missing DATA_SCALARs)
  // -----------------------------
  PARAMETER(T_C);           // Ambient temperature (°C), used for Q10 scaling; if unknown, estimate with low priority
  PARAMETER(T_ref);         // Reference temperature (°C) for Q10 scaling; typically near seasonal mean
  PARAMETER(penalty_w);     // Weight for bound penalties (dimensionless, e.g., 1.0); constrained to be non-negative via soft bounds

  // -----------------------------
  // PARAMETERS (observation)
  // -----------------------------
  PARAMETER(sd_N);          // Log-scale observation SD for N (dimensionless)
  PARAMETER(sd_P);          // Log-scale observation SD for P (dimensionless)
  PARAMETER(sd_Z);          // Log-scale observation SD for Z (dimensionless)

  // -----------------------------
  // NUMERICAL SAFEGUARDS
  // -----------------------------
  Type eps = Type(1e-8);         // Small constant to prevent division by zero and log(0)
  Type sd_floor = Type(0.05);    // Minimum observation SD to ensure numerical stability

  // -----------------------------
  // INITIALIZE PREDICTION VECTORS
  // -----------------------------
  int n = N_dat.size();
  vector<Type> N_pred(n);
  vector<Type> P_pred(n);
  vector<Type> Z_pred(n);

  // Initial conditions from observed data (no data leakage beyond t=0)
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);

  // -----------------------------
  // PROCESS MODEL INTEGRATION
  // -----------------------------
  // Equation set (evaluated with previous-step states only):
  // 1) fN     = N^hN / (K_N^hN + N^hN)       [Generalized Monod/Hill nutrient limitation]
  // 2) I_eff  = I0 * exp(- (k_Ibg + k_Ishade * P)) [Background + self-shaded effective irradiance]
  // 3) fI     = I_eff / (K_I + I_eff)        [Saturating light limitation]
  // 4) theta_mu = q10_mu^((T_C - T_ref)/10)  [Temperature scaling for growth]
  // 5) P_growth = mu_max * theta_mu * min_smooth(fN, fI) * P
  // 6) G_fun  = P^h / (K_g^h + P^h)          [Sigmoidal prey dependence]
  // 7) theta_g = q10_g^((T_C - T_ref)/10)    [Temperature scaling for grazing]
  // 8) Z_grazing = g_max * theta_g * (G_fun / (1 + c_BD * Z)) * Z   [Beddington-DeAngelis interference]
  // 9) Z_growth = e_Z * Z_grazing
  // 10) P_losses = mP1*P + mP2*P^2
  // 11) Z_losses = mZ1*Z + mZ2*Z^2
  // 12) N_uptake = P_growth / y_PN
  // 13) N_remin  = rP_N*P_losses + rZ_N*Z_losses + (1 - e_Z)*Z_grazing
  // 14) dN/dt = k_mix*(N_deep - N) - N_uptake + N_remin
  // 15) dP/dt = P_growth - Z_grazing - P_losses
  // 16) dZ/dt = Z_growth - Z_losses

  for (int i = 1; i < n; i++) {
    Type dt = (Time(i) - Time(i - 1)); // Variable step size in days
    // Ensure non-negative time step; if data error occurs, clamp softly
    if (dt < Type(0)) dt = eps;

    // Previous-step states (no data leakage)
    Type Np = N_pred(i - 1);
    Type Pp = P_pred(i - 1);
    Type Zp = Z_pred(i - 1);

    // Nutrient limitation (Hill/generalized Monod)
    Type Nh = pow(Np + eps, hN);
    Type Kh = pow(K_N + eps, hN);
    Type fN = Nh / (Kh + Nh + eps);                               // (1)

    // Light limitation with background and self-shading
    Type I_eff = I0 * exp(-(k_Ibg + k_Ishade * Pp));               // (2)
    Type fI    = I_eff / (K_I + I_eff + eps);                      // (3)

    // Temperature modifiers
    Type theta_mu = pow(q10_mu, (T_C - T_ref) / Type(10.0));       // (4)
    Type theta_g  = pow(q10_g,  (T_C - T_ref) / Type(10.0));       // (7)

    // Phytoplankton growth with smooth Liebig co-limitation
    Type f_lim = smooth_min(fN, fI);                               // smooth min in [0,1]
    Type P_growth = mu_max * theta_mu * f_lim * Pp;                // (5)

    // Grazing functional response (Holling-type with exponent) with BD interference
    Type Ph = pow(Pp + eps, h_exp);
    Type Kgh = pow(K_g + eps, h_exp);
    Type G_fun = Ph / (Kgh + Ph + eps);                            // (6)
    Type interference = Type(1.0) + c_BD * Zp;                     // BD denominator > 0
    Type Z_grazing = g_max * theta_g * (G_fun / interference) * Zp; // (8)
    Type Z_growth  = e_Z * Z_grazing;                              // (9)

    // Losses and remineralization
    Type P_losses  = mP1 * Pp + mP2 * Pp * Pp;                     // (10)
    Type Z_losses  = mZ1 * Zp + mZ2 * Zp * Zp;                     // (11)
    Type N_uptake  = P_growth / (y_PN + eps);                      // (12)
    Type N_remin   = rP_N * P_losses + rZ_N * Z_losses + (Type(1.0) - e_Z) * Z_grazing; // (13)

    // Tendencies
    Type dN = k_mix * (N_deep - Np) - N_uptake + N_remin;          // (14)
    Type dP = P_growth - Z_grazing - P_losses;                     // (15)
    Type dZ = Z_growth - Z_losses;                                 // (16)

    // Euler forward update with smooth non-negativity to ensure log-likelihood is defined
    Type N_next = Np + dt * dN;
    Type P_next = Pp + dt * dP;
    Type Z_next = Zp + dt * dZ;

    N_pred(i) = smooth_positive(N_next);
    P_pred(i) = smooth_positive(P_next);
    Z_pred(i) = smooth_positive(Z_next);
  }

  // -----------------------------
  // LIKELIHOOD (lognormal errors)
  // -----------------------------
  Type nll = Type(0.0);

  Type sdN_eff = sd_N + sd_floor;   // Apply floors to avoid degenerate variance
  Type sdP_eff = sd_P + sd_floor;
  Type sdZ_eff = sd_Z + sd_floor;

  for (int i = 0; i < n; i++) {
    // Always include all observations
    nll -= dnorm(log(N_dat(i) + eps), log(N_pred(i) + eps), sdN_eff, true);
    nll -= dnorm(log(P_dat(i) + eps), log(P_pred(i) + eps), sdP_eff, true);
    nll -= dnorm(log(Z_dat(i) + eps), log(Z_pred(i) + eps), sdZ_eff, true);
  }

  // -----------------------------
  // SOFT BOUNDS VIA SMOOTH PENALTIES
  // -----------------------------
  // These reflect biologically plausible ranges; see parameters.json for the same bounds.
  Type pen = Type(0.0);

  // Helper lambda to add two-sided penalty
  auto add_pen = [&](Type x, Type lo, Type hi) {
    pen += smooth_hinge(lo - x);     // below lower
    pen += smooth_hinge(x - hi);     // above upper
  };

  // Process parameter bounds
  add_pen(mu_max,  Type(0.1),  Type(3.0));
  add_pen(K_N,     Type(1e-4), Type(2.0));
  add_pen(hN,      Type(1.0),  Type(4.0));

  add_pen(I0,      Type(0.1),  Type(10.0));
  add_pen(K_I,     Type(0.01), Type(5.0));
  add_pen(k_Ishade,Type(0.0),  Type(10.0));
  add_pen(k_Ibg,   Type(0.0),  Type(10.0));

  add_pen(q10_mu,  Type(1.1),  Type(3.0));
  add_pen(q10_g,   Type(1.1),  Type(3.0));

  add_pen(g_max,   Type(0.1),  Type(5.0));
  add_pen(K_g,     Type(0.01), Type(3.0));
  add_pen(h_exp,   Type(1.0),  Type(3.0));
  add_pen(c_BD,    Type(0.0),  Type(10.0));

  add_pen(e_Z,     Type(0.1),  Type(0.8));
  add_pen(mP1,     Type(0.0),  Type(0.5));
  add_pen(mP2,     Type(0.0),  Type(2.0));
  add_pen(mZ1,     Type(0.0),  Type(0.5));
  add_pen(mZ2,     Type(0.0),  Type(2.0));

  add_pen(rP_N,    Type(0.5),  Type(1.0));
  add_pen(rZ_N,    Type(0.5),  Type(1.0));

  add_pen(y_PN,    Type(0.5),  Type(3.0));

  add_pen(k_mix,   Type(0.0),  Type(1.0));
  add_pen(N_deep,  Type(0.0),  Type(2.0));

  // Newly parameterized environment and penalty weight
  add_pen(T_C,       Type(-2.0), Type(35.0));  // typical ocean mixed-layer temperatures
  add_pen(T_ref,     Type(-2.0), Type(35.0));  // keep within plausible physical range
  add_pen(penalty_w, Type(0.0),  Type(10.0));  // non-negative weight with reasonable cap

  // Observation SDs
  add_pen(sd_N,    Type(0.02), Type(1.0));
  add_pen(sd_P,    Type(0.02), Type(1.0));
  add_pen(sd_Z,    Type(0.02), Type(1.0));

  nll += penalty_w * pen;

  // -----------------------------
  // REPORTING
  // -----------------------------
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);

  // Optionally report some derived rates at final time for diagnostics
  // (These are not used in likelihood; just helpful outputs)
  // Using last predicted step (n-1)
  if (n > 0) {
    Type Np = N_pred(n - 1);
    Type Pp = P_pred(n - 1);
    Type Zp = Z_pred(n - 1);

    Type Nh_last = pow(Np + eps, hN);
    Type Kh_last = pow(K_N + eps, hN);
    Type fN_last = Nh_last / (Kh_last + Nh_last + eps);

    Type I_eff_last  = I0 * exp(-(k_Ibg + k_Ishade * Pp));
    Type fI_last     = I_eff_last / (K_I + I_eff_last + eps);

    Type theta_mu_last = pow(q10_mu, (T_C - T_ref) / Type(10.0));
    Type theta_g_last  = pow(q10_g,  (T_C - T_ref) / Type(10.0));

    Type Ph_last = pow(Pp + eps, h_exp);
    Type Kgh_last = pow(K_g + eps, h_exp);
    Type G_fun_last = Ph_last / (Kgh_last + Ph_last + eps);

    Type interference_last = Type(1.0) + c_BD * Zp;

    // Smooth min limitation at last step
    Type f_lim_last = smooth_min(fN_last, fI_last);

    REPORT(fN_last);
    REPORT(fI_last);
    REPORT(f_lim_last);
    REPORT(theta_mu_last);
    REPORT(theta_g_last);
    REPORT(G_fun_last);
    REPORT(interference_last);
  }

  return nll;
}
