#include <TMB.hpp>

// Small helpers for numerical stability and smooth penalties
template<class Type>
Type softplus(Type x) { // smooth ReLU to avoid hard cutoffs
  return log(Type(1) + exp(x));
}

template<class Type>
Type inv_logit(Type x) {
  return Type(1) / (Type(1) + exp(-x));
}

template<class Type>
Type sqr(Type x) { return x * x; }

template<class Type>
Type two_pi() { return Type(6.28318530717958647692); }

template<class Type>
Type bound_penalty(Type x, Type lo, Type hi, Type strength){
  // Smooth quadratic penalty outside [lo, hi]
  Type pen = Type(0);
  pen += sqr( softplus(lo - x) ) * strength; // penalize below lower bound
  pen += sqr( softplus(x - hi) ) * strength; // penalize above upper bound
  return pen;
}

// Smooth, always-positive mapping close to identity for x>0,
// and ~0 for x<=0 without hard cutoffs (prevents log of non-positive).
template<class Type>
Type posify(Type x, Type tiny){
  // 0.5 * (x + sqrt(x^2 + tiny)) is a smooth approximation to max(x, 0)
  return Type(0.5) * (x + sqrt(x * x + tiny));
}

template<class Type>
Type objective_function<Type>::operator() ()
{
  // =========================
  // DATA
  // =========================
  // Use the exact same time variable name as provided in the data file.
  DATA_VECTOR(Time);          // time in days; corresponds to column "Time"
  DATA_VECTOR(N_dat);         // observed nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);         // observed phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);         // observed zooplankton concentration (g C m^-3)

  int n = N_dat.size();       // number of observations

  // =========================
  // PARAMETERS
  // =========================
  // Growth and limitation
  PARAMETER(mu_max);          // d^-1; maximum phytoplankton specific growth rate; initialize from literature or initial estimate using doubling times
  PARAMETER(k_N);             // g C m^-3; half-saturation constant for nutrient limitation; initial estimate from literature on Monod kinetics
  PARAMETER(omega_lim);       // unitless; logit-weight for combining N vs light limitation (w_N=inv_logit(omega_lim)); initial estimate 0 (balanced)
  PARAMETER(a_light);         // unitless in [0,1]; amplitude of seasonal light/mixing modulation on phytoplankton growth; initial estimate from seasonal range
  PARAMETER(phi_days);        // days; phase shift of seasonality; initial estimate from day-of-year peak
  PARAMETER(period_days);     // days; period of seasonality; typically ~365 d; can be tuned for experimental time windows

  // Grazing
  PARAMETER(g_max);           // d^-1; maximum zooplankton clearance/grazing rate per Z biomass; literature/initial estimate
  PARAMETER(k_P);             // g C m^-3; half-saturation constant for grazing functional response; literature/initial estimate
  PARAMETER(q_graz);          // unitless (>=1); shape parameter for Holling-II/III continuum (1: type II; >1: type III-like); literature/initial
  PARAMETER(a_season_g);      // unitless in [0,1]; amplitude of seasonal modulation on grazing capacity; initial estimate

  // Efficiencies and mortality
  PARAMETER(e_Z);             // unitless in [0,1]; zooplankton assimilation efficiency (fraction of ingested P converted to Z growth)
  PARAMETER(m_P_lin);         // d^-1; linear (background) phytoplankton mortality/excretion rate
  PARAMETER(m_P_quad);        // (g C m^-3)^-1 d^-1; quadratic phytoplankton loss (e.g., aggregation/sinking)
  PARAMETER(m_Z_lin);         // d^-1; linear zooplankton mortality/excretion
  PARAMETER(m_Z_quad);        // (g C m^-3)^-1 d^-1; quadratic (density-dependent) zooplankton loss

  // Recycling, external inputs, and export
  PARAMETER(r_P);             // unitless in [0,1]; fraction of P mortality recycled to dissolved nutrients
  PARAMETER(r_Z);             // unitless in [0,1]; fraction of unassimilated grazing flux recycled to nutrients
  PARAMETER(r_ZM);            // unitless in [0,1]; fraction of Z mortality recycled to nutrients
  PARAMETER(s_N);             // d^-1; first-order nutrient export/sinking from mixed layer
  PARAMETER(R_ext);           // g C m^-3 d^-1; external nutrient supply (e.g., mixing/upwelling)

  // Observation error (lognormal)
  PARAMETER(log_sigma_N);     // log(sd) for N observations; initialize small (e.g., log(0.1))
  PARAMETER(log_sigma_P);     // log(sd) for P observations
  PARAMETER(log_sigma_Z);     // log(sd) for Z observations

  // =========================
  // SETTINGS FOR STABILITY
  // =========================
  Type eps = Type(1e-8);            // small constant to avoid divide-by-zero and log(0)
  Type min_dt = Type(1e-6);         // minimum time step to prevent zero dt
  Type min_sd = Type(0.02);         // minimum observation SD to stabilize likelihood across magnitudes

  // =========================
  // STATE VECTORS
  // =========================
  vector<Type> N_pred(n);     // predicted nutrient concentrations (g C m^-3)
  vector<Type> P_pred(n);     // predicted phytoplankton concentrations (g C m^-3)
  vector<Type> Z_pred(n);     // predicted zooplankton concentrations (g C m^-3)

  // INITIAL CONDITIONS: set to the first data point (no data leakage beyond index 0)
  N_pred(0) = N_dat(0);       // initialize from observed N at first time
  P_pred(0) = P_dat(0);       // initialize from observed P at first time
  Z_pred(0) = Z_dat(0);       // initialize from observed Z at first time

  // =========================
  // PENALTIES FOR PARAMETER BOUNDS (smooth, not hard constraints)
  // =========================
  Type nll = Type(0); // negative log-likelihood

  // Suggested biological bounds and smooth penalties
  nll += bound_penalty(mu_max,    Type(0.0),  Type(3.0),  Type(1.0));
  nll += bound_penalty(k_N,       Type(1e-6), Type(1.0),  Type(1.0));
  nll += bound_penalty(omega_lim, Type(-5.0), Type(5.0),  Type(0.1));
  nll += bound_penalty(a_light,   Type(0.0),  Type(1.0),  Type(1.0));
  nll += bound_penalty(phi_days,  Type(0.0),  Type(365.0),Type(0.1));
  nll += bound_penalty(period_days,Type(10.0),Type(400.0),Type(0.1));

  nll += bound_penalty(g_max,     Type(0.0),  Type(5.0),  Type(1.0));
  nll += bound_penalty(k_P,       Type(1e-6), Type(1.0),  Type(1.0));
  nll += bound_penalty(q_graz,    Type(1.0),  Type(3.0),  Type(0.5));
  nll += bound_penalty(a_season_g,Type(0.0),  Type(1.0),  Type(1.0));

  nll += bound_penalty(e_Z,       Type(0.0),  Type(1.0),  Type(1.0));
  nll += bound_penalty(m_P_lin,   Type(0.0),  Type(1.0),  Type(0.5));
  nll += bound_penalty(m_P_quad,  Type(0.0),  Type(10.0), Type(0.2));
  nll += bound_penalty(m_Z_lin,   Type(0.0),  Type(1.0),  Type(0.5));
  nll += bound_penalty(m_Z_quad,  Type(0.0),  Type(10.0), Type(0.2));

  nll += bound_penalty(r_P,       Type(0.0),  Type(1.0),  Type(0.5));
  nll += bound_penalty(r_Z,       Type(0.0),  Type(1.0),  Type(0.5));
  nll += bound_penalty(r_ZM,      Type(0.0),  Type(1.0),  Type(0.5));
  nll += bound_penalty(s_N,       Type(0.0),  Type(1.0),  Type(0.5));
  nll += bound_penalty(R_ext,     Type(0.0),  Type(0.1),  Type(0.5));

  nll += bound_penalty(log_sigma_N, Type(-10.0), Type(2.0), Type(0.1));
  nll += bound_penalty(log_sigma_P, Type(-10.0), Type(2.0), Type(0.1));
  nll += bound_penalty(log_sigma_Z, Type(-10.0), Type(2.0), Type(0.1));

  // =========================
  // DERIVED CONSTANTS
  // =========================
  Type w_N = inv_logit(omega_lim);     // weight for N-limitation in geometric mean combination
  Type w_L = Type(1.0) - w_N;          // weight for light/seasonal limitation
  Type sd_N = exp(log_sigma_N) + min_sd; // stabilized observation SD for log-normal likelihood
  Type sd_P = exp(log_sigma_P) + min_sd;
  Type sd_Z = exp(log_sigma_Z) + min_sd;

  // =========================
  // DYNAMICS: DISCRETE TIME INTEGRATION (EULER)
  // =========================
  // Equations (all rates per day):
  // (1) Seasonal driver: S(t) = 0.5 + 0.5*cos(2π*(t - phi)/period)
  // (2) Light limitation: L(t) = (1 - a_light) + a_light * S(t)
  // (3) Nutrient limitation: f_N = N / (k_N + N)
  // (4) Combined limitation (geometric mean): f_lim = f_N^w_N * L(t)^w_L
  // (5) Phytoplankton growth flux: PP = mu_max * f_lim * P
  // (6) Grazing functional response: f_P = P^q / (k_P^q + P^q)
  // (7) Seasonal grazing capacity: G_cap(t) = (1 - a_season_g) + a_season_g * S(t)
  // (8) Grazing flux: G = g_max * G_cap(t) * f_P * Z
  // (9) Phytoplankton loss: M_P = m_P_lin*P + m_P_quad*P^2
  // (10) Zooplankton loss: M_Z = m_Z_lin*Z + m_Z_quad*Z^2
  // (11) Z growth: dZ = e_Z*G - M_Z
  // (12) P change: dP = PP - G - M_P
  // (13) N change: dN = -PP + r_P*M_P + r_Z*(1 - e_Z)*G + r_ZM*M_Z - s_N*N + R_ext
  for(int i = 1; i < n; i++){
    // Prior state (use only predictions to avoid data leakage)
    Type N_prev = N_pred(i-1);
    Type P_prev = P_pred(i-1);
    Type Z_prev = Z_pred(i-1);

    // Positive-valued proxies for saturating functions (smooth, no hard max)
    Type N_pos = sqrt(N_prev * N_prev + eps); // ~|N_prev| with smooth derivative
    Type P_pos = sqrt(P_prev * P_prev + eps); // ~|P_prev|
    Type Z_pos = sqrt(Z_prev * Z_prev + eps); // used only if needed for smoothness

    // Time step
    Type dt = Time(i) - Time(i-1);
    dt = dt + softplus(min_dt - dt); // smooth floor at min_dt

    // Seasonal driver
    Type S = Type(0.5) + Type(0.5) * cos( two_pi<Type>() * ( (Time(i-1) - phi_days) / (period_days + eps) ) );
    // Light/seasonal limitation for growth
    Type L_t = (Type(1.0) - a_light) + a_light * S;

    // Nutrient limitation (Monod)
    Type fN = N_pos / (k_N + N_pos + eps);

    // Combined resource limitation (geometric mean with smooth weights)
    Type f_lim = pow( (fN + eps), w_N ) * pow( (L_t + eps), w_L );

    // Effective growth rate
    Type mu_eff = mu_max * f_lim;

    // Grazing functional response (Holling II/III continuity)
    Type q = q_graz; // kept within [1,3] by penalty
    Type Pq = pow(P_pos + eps, q);
    Type kPq = pow(k_P + eps, q);
    Type fP = Pq / (kPq + Pq + eps);

    // Seasonal modulation of grazing capacity
    Type G_cap = (Type(1.0) - a_season_g) + a_season_g * S;

    // Fluxes
    Type PP = mu_eff * P_prev;                                 // primary production
    Type G  = g_max * G_cap * fP * Z_prev;                     // grazing on P
    Type M_P = m_P_lin * P_prev + m_P_quad * P_prev * P_prev;  // P losses
    Type M_Z = m_Z_lin * Z_prev + m_Z_quad * Z_prev * Z_prev;  // Z losses

    // State derivatives
    Type dP = PP - G - M_P;                                    // phytoplankton change
    Type dZ = e_Z * G - M_Z;                                   // zooplankton change
    Type dN = -PP + r_P * M_P + r_Z * (Type(1.0) - e_Z) * G
                    + r_ZM * M_Z - s_N * N_prev + R_ext;       // nutrient change

    // Euler step
    N_pred(i) = N_prev + dt * dN;
    P_pred(i) = P_prev + dt * dP;
    Z_pred(i) = Z_prev + dt * dZ;
  }

  // =========================
  // LIKELIHOOD: LOGNORMAL ERRORS
  // =========================
  for(int i = 0; i < n; i++){
    // Use small constants and a smooth positive mapping for predictions
    Type yN = N_dat(i) + eps;  // observed N (ensure strictly positive)
    Type yP = P_dat(i) + eps;  // observed P
    Type yZ = Z_dat(i) + eps;  // observed Z

    // Map predictions to strictly positive domain smoothly to avoid log of non-positive
    Type mN = posify(N_pred(i), eps);
    Type mP = posify(P_pred(i), eps);
    Type mZ = posify(Z_pred(i), eps);

    nll -= dnorm(log(yN), log(mN), sd_N, true); // N lognormal error
    nll -= dnorm(log(yP), log(mP), sd_P, true); // P lognormal error
    nll -= dnorm(log(yZ), log(mZ), sd_Z, true); // Z lognormal error
  }

  // =========================
  // REPORTING
  // =========================
  REPORT(N_pred);  // model predictions for nutrient (g C m^-3)
  REPORT(P_pred);  // model predictions for phytoplankton (g C m^-3)
  REPORT(Z_pred);  // model predictions for zooplankton (g C m^-3)
  REPORT(sd_N);    // realized SDs after flooring
  REPORT(sd_P);
  REPORT(sd_Z);
  REPORT(w_N);     // weight given to nutrient limitation
  REPORT(w_L);     // weight given to light limitation

  return nll;
}
