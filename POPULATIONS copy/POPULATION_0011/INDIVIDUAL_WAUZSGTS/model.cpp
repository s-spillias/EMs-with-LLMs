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

template<class Type>
Type objective_function<Type>::operator()() {
  // -----------------------------
  // DATA
  // -----------------------------
  // Time vector name must match the sanitized column name provided by the data loader.
  DATA_VECTOR(Time);                       // Time in days

  DATA_VECTOR(N_dat);                      // Observed nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);                      // Observed phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);                      // Observed zooplankton concentration (g C m^-3)

  // -----------------------------
  // PARAMETERS (process)
  // -----------------------------
  PARAMETER(mu_max);        // Maximum phytoplankton specific growth rate (d^-1)
  PARAMETER(K_N);           // Half-sat constant for nutrient uptake (g C m^-3)
  PARAMETER(N_thr);         // Smooth threshold for nutrient limitation center (g C m^-3)
  PARAMETER(thr_steep);     // Steepness of the smooth threshold (dimensionless)

  PARAMETER(I0);            // Effective surface light/irradiance (relative units per day)
  PARAMETER(K_I);           // Half-sat constant for light-limited growth (same units as I0)
  PARAMETER(k_Ishade);      // Self-shading (attenuation) coefficient by P (m^3 gC^-1)

  PARAMETER(q10_mu);        // Q10 for phytoplankton growth (dimensionless)
  PARAMETER(q10_g);         // Q10 for zooplankton ingestion (dimensionless)

  PARAMETER(g_max);         // Maximum zooplankton grazing rate (d^-1)
  PARAMETER(K_g);           // Half-sat constant for grazing functional response (g C m^-3)
  PARAMETER(h_exp);         // Shape exponent for grazing response (dimensionless >=1)
  PARAMETER(c_BD);          // Predator interference coefficient for Beddington-DeAngelis denominator (m^3 gC^-1)

  PARAMETER(e_Z);           // Zooplankton assimilation efficiency (dimensionless, 0–1)
  PARAMETER(eta_e);         // Sensitivity of e_Z to nutrient limitation (dimensionless, 0–1)
  PARAMETER(mP1);           // Linear P mortality/lysis rate (d^-1)
  PARAMETER(mP2);           // Quadratic P loss rate (m^3 gC^-1 d^-1)
  PARAMETER(mZ1);           // Linear Z excretion/mortality rate (d^-1)
  PARAMETER(mZ2);           // Quadratic Z mortality (m^3 gC^-1 d^-1)

  PARAMETER(rP_N);          // Fraction of P losses remineralized to N (dimensionless 0–1)
  PARAMETER(rZ_N);          // Fraction of Z losses remineralized to N (dimensionless 0–1)

  PARAMETER(y_PN);          // Yield: g C of P produced per g C of nutrient consumed (dimensionless >0)

  PARAMETER(k_mix);         // Vertical mixing rate coupling to deep pool (d^-1)
  PARAMETER(N_deep);        // Deep nutrient concentration (g C m^-3)

  // -----------------------------
  // PARAMETERS (environment and objective weight)
  // -----------------------------
  PARAMETER(T_C);           // Ambient temperature (°C)
  PARAMETER(T_ref);         // Reference temperature (°C) for Q10 scaling
  PARAMETER(penalty_w);     // Weight for bound penalties (dimensionless)

  // -----------------------------
  // PARAMETERS (initial states as free parameters to avoid data leakage)
  // -----------------------------
  PARAMETER(N0);            // Initial nutrient (g C m^-3)
  PARAMETER(P0);            // Initial phytoplankton (g C m^-3)
  PARAMETER(Z0);            // Initial zooplankton (g C m^-3)

  // -----------------------------
  // PARAMETERS (observation)
  // -----------------------------
  PARAMETER(sd_N);          // Log-scale observation SD for N (dimensionless)
  PARAMETER(sd_P);          // Log-scale observation SD for P (dimensionless)
  PARAMETER(sd_Z);          // Log-scale observation SD for Z (dimensionless)

  // -----------------------------
  // NUMERICAL SAFEGUARDS
  // -----------------------------
  Type eps = Type(1e-12);            // Small positive for numerical safety
  Type obs_eps = Type(1e-9);         // Small positive to avoid log(0)
  Type sd_floor = Type(0.02);        // Observation SD floor (log scale)

  // Effective observation SDs with a smooth floor to retain differentiability
  Type sdN_eff = sd_floor + softplus(sd_N - sd_floor);
  Type sdP_eff = sd_floor + softplus(sd_P - sd_floor);
  Type sdZ_eff = sd_floor + softplus(sd_Z - sd_floor);

  // -----------------------------
  // SETUP
  // -----------------------------
  int n = Time.size();
  vector<Type> N_pred(n);
  vector<Type> P_pred(n);
  vector<Type> Z_pred(n);

  // Initialize states from parameters (not from observations to avoid leakage)
  Type N = smooth_positive(N0);
  Type P = smooth_positive(P0);
  Type Z = smooth_positive(Z0);

  // Precompute temperature scalings (constant over time here)
  Type mu_T = mu_max * pow(q10_mu, (T_C - T_ref) / Type(10.0));
  Type g_T  = g_max  * pow(q10_g,  (T_C - T_ref) / Type(10.0));

  // -----------------------------
  // STATE-Space dynamics (discrete-time Euler integration)
  // -----------------------------
  Type nll = Type(0.0);

  for (int i = 0; i < n; i++) {
    // Store predictions for this time step (after previous updates)
    N_pred(i) = smooth_positive(N);
    P_pred(i) = smooth_positive(P);
    Z_pred(i) = smooth_positive(Z);

    // Observation likelihood at time i (lognormal)
    // Note: Only uses predictions; no use of current observations in state predictions.
    if (N_dat(i) > Type(0.0)) {
      nll -= dnorm(log(N_dat(i) + obs_eps), log(N_pred(i) + obs_eps), sdN_eff, true);
    }
    if (P_dat(i) > Type(0.0)) {
      nll -= dnorm(log(P_dat(i) + obs_eps), log(P_pred(i) + obs_eps), sdP_eff, true);
    }
    if (Z_dat(i) > Type(0.0)) {
      nll -= dnorm(log(Z_dat(i) + obs_eps), log(Z_pred(i) + obs_eps), sdZ_eff, true);
    }

    // Skip process update on the last time if there is no forward interval
    if (i == n - 1) break;

    // Time step (days); enforce small positive step
    Type dt_raw = Time(i + 1) - Time(i);
    Type dt = smooth_positive(dt_raw) + Type(1e-6);

    // Use previous-step states for all process calculations
    Type Np = N_pred(i);
    Type Pp = P_pred(i);
    Type Zp = Z_pred(i);

    // Light limitation with self-shading on effective irradiance
    // I_eff = I0 * exp(-k_Ishade * Pp) then Monod: f_I = I_eff / (K_I + I_eff)
    Type I_eff = I0 * exp(-k_Ishade * Pp);
    Type f_I = I_eff / (K_I + I_eff + eps);

    // Nutrient limitation: Monod-like saturation
    Type fN_sat = Np / (K_N + Np + eps);

    // Smooth nutrient threshold gate: ~0 at low N, ~1 when above threshold
    Type fN_gate = invlogit(thr_steep * (Np - N_thr));

    // Phytoplankton gross growth (g C m^-3 d^-1)
    Type P_growth = mu_T * f_I * fN_sat * fN_gate * Pp;

    // Zooplankton grazing: Holling-type with exponent h_exp and BD interference
    // Functional response on P
    Type P_eff = Pp + Type(1e-12);
    Type Ph = pow(P_eff, h_exp);
    Type Kg_h = pow(K_g + eps, h_exp);
    Type f_graz = Ph / (Kg_h + Ph);
    Type denom_BD = Type(1.0) + c_BD * Zp;
    Type Grazing = g_T * Zp * f_graz / denom_BD; // g C m^-3 d^-1

    // Variable assimilation efficiency depending on nutrient status
    Type e_Z_eff = e_Z * ((Type(1.0) - eta_e) + eta_e * fN_sat);

    // Losses
    Type P_loss = mP1 * Pp + mP2 * Pp * Pp;
    Type Z_loss = mZ1 * Zp + mZ2 * Zp * Zp;

    // Nutrient uptake and recycling
    Type N_uptake = P_growth / (y_PN + eps);
    Type N_remin = rP_N * P_loss + rZ_N * Z_loss + (Type(1.0) - e_Z_eff) * Grazing;

    // Vertical mixing term (toward deep concentration)
    Type N_mix = k_mix * (N_deep - Np);

    // State derivatives
    Type dP = P_growth - Grazing - P_loss;
    Type dZ = e_Z_eff * Grazing - Z_loss;
    Type dN = -N_uptake + N_remin + N_mix;

    // Euler update with positivity safeguard
    N = smooth_positive(Np + dN * dt);
    P = smooth_positive(Pp + dP * dt);
    Z = smooth_positive(Zp + dZ * dt);
  }

  // -----------------------------
  // Soft penalties for bounds/biological plausibility
  // -----------------------------
  Type pen = Type(0.0);

  // Non-negativity penalties
  pen += smooth_hinge(-mu_max);
  pen += smooth_hinge(-K_N);
  pen += smooth_hinge(-I0);
  pen += smooth_hinge(-K_I);
  pen += smooth_hinge(-k_Ishade);
  pen += smooth_hinge(-q10_mu);
  pen += smooth_hinge(-q10_g);
  pen += smooth_hinge(-g_max);
  pen += smooth_hinge(-K_g);
  pen += smooth_hinge(Type(1.0) - h_exp); // h_exp >= 1
  pen += smooth_hinge(-c_BD);
  pen += smooth_hinge(-mP1);
  pen += smooth_hinge(-mP2);
  pen += smooth_hinge(-mZ1);
  pen += smooth_hinge(-mZ2);
  pen += smooth_hinge(-rP_N) + smooth_hinge(rP_N - Type(1.0));
  pen += smooth_hinge(-rZ_N) + smooth_hinge(rZ_N - Type(1.0));
  pen += smooth_hinge(-y_PN);
  pen += smooth_hinge(-k_mix);
  pen += smooth_hinge(-N_deep);
  pen += smooth_hinge(-e_Z) + smooth_hinge(e_Z - Type(1.0));
  pen += smooth_hinge(-eta_e) + smooth_hinge(eta_e - Type(1.0));

  pen += smooth_hinge(-N0);
  pen += smooth_hinge(-P0);
  pen += smooth_hinge(-Z0);

  // Observation SDs handled by smooth floor; still discourage too small values
  pen += smooth_hinge(sd_floor - sd_N);
  pen += smooth_hinge(sd_floor - sd_P);
  pen += smooth_hinge(sd_floor - sd_Z);

  // Penalty weight constrained to be non-negative via soft penalty
  pen += smooth_hinge(-penalty_w);

  nll += penalty_w * pen;

  // -----------------------------
  // REPORTS
  // -----------------------------
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  ADREPORT(N_pred);
  ADREPORT(P_pred);
  ADREPORT(Z_pred);

  return nll;
}
